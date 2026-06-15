
from __future__ import annotations

import gc
import json
import os
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from tqdm import tqdm

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna

from scripts.config import DATA_DIR, MODEL_DIR, ROOT_DIR
from scripts.data_management import load_data
import scripts.indicators  # noqa: F401

warnings.filterwarnings("ignore")


########################################################################################################################

class Settings:
    VERBOSE = 0
    LOGGING = False
    GPU = {"LGBM": False, "CAT": False}
    Threaded = False

@dataclass
class UniverseConfig:
    horizon: int = 30
    max_top_tickers: int = 10

    edge_q: float = 0.02
    min_abs_pred: float = 0.003
    allow_short: bool = False
    cost_bps: float = 10.0

    replace_buffer: float = 0.002
    exit_pred_threshold: float = 0.0

    min_price: float = 5.0
    max_price: float = 5000.0

    min_profile_adv: float = 5_000_000.0

    min_rolling_dollar_volume_1d: float = 10_000_000.0
    min_rolling_dollar_volume_1h: float = 750_000.0

    max_abs_bar_return: float = 0.75

    min_rows_after_filter_1d: int = 252
    min_rows_after_filter_1h: int = 500

    rolling_vol_window_1d: int = 63
    rolling_beta_window_1d: int = 252

    rolling_vol_window_1h: int = 120
    rolling_beta_window_1h: int = 240

def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

now = datetime.now().strftime('%Y-%m-%d %H:%M')
def log(string: str, prints: bool = True):
    if prints: print(string)
    with open(f"logging_{now}.txt", "a") as f:
        f.write(f"{string}\n")

########################################################################################################################

class PurgedTimeSeriesSplit:
    def __init__(self, n_splits: int = 4, gap: int = 4, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.gap = gap
        self.embargo_pct = embargo_pct

    def split(self, dates: np.ndarray):
        unique_dates = np.array(sorted(pd.to_datetime(pd.Series(dates).unique())))
        n_dates = len(unique_dates)
        embargo = int(n_dates * self.embargo_pct)
        test_size = n_dates // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end_date_idx = (i + 1) * test_size - self.gap
            test_start_date_idx = (i + 1) * test_size + embargo
            test_end_date_idx = min(test_start_date_idx + test_size, n_dates)

            if train_end_date_idx <= 0 or test_start_date_idx >= n_dates:
                continue

            train_dates = unique_dates[:train_end_date_idx]
            test_dates = unique_dates[test_start_date_idx:test_end_date_idx]

            train_mask = np.isin(pd.to_datetime(dates), train_dates)
            test_mask = np.isin(pd.to_datetime(dates), test_dates)

            train_idx = np.flatnonzero(train_mask)
            test_idx = np.flatnonzero(test_mask)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

class TrainingManager:
    def __init__(self):
        self.config = UniverseConfig()
        self.seed = 69
        self.__test_size = 0.2

        self.profile_map = None
        self.feature_cols = None
        self.split_date = None

        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

########################################################################################################################
    # Hypers
    def _make_tuning_split(self, validation_size: float = 0.2):
        unique_dates = np.array(sorted(pd.to_datetime(self.train_df["Date"]).unique()))
        split_date = unique_dates[int(len(unique_dates) * (1 - validation_size))]

        tune_train_df = self.train_df[self.train_df["Date"] <= split_date].copy()
        tune_val_df = self.train_df[self.train_df["Date"] > split_date].copy()

        X_tune_train = tune_train_df[self.feature_cols].to_numpy(dtype=np.float32)
        y_tune_train = tune_train_df["target_excess_return"].to_numpy(dtype=np.float32)

        X_tune_val = tune_val_df[self.feature_cols].to_numpy(dtype=np.float32)
        y_tune_val = tune_val_df["target_excess_return"].to_numpy(dtype=np.float32)

        return X_tune_train, y_tune_train, X_tune_val, y_tune_val, tune_val_df

    def _portfolio_tuning_score(self, portfolio_metrics: dict[str, Any]) -> float:
        if portfolio_metrics["portfolio_days"] <= 0:
            return -np.inf

        cagr = portfolio_metrics["portfolio_cagr_like"]
        sharpe = portfolio_metrics["portfolio_sharpe_like"]
        max_drawdown = abs(portfolio_metrics["portfolio_max_drawdown"])
        turnover = portfolio_metrics["portfolio_avg_turnover"]

        return cagr + 0.05 * sharpe - 0.50 * max_drawdown - 0.002 * turnover

    def _keep_top_results(self, top_results: list[dict[str, Any]], result: dict[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
        top_results.append(result)
        top_results.sort(key=lambda r: r["score"], reverse=True)
        return top_results[:top_k]

    def tune_lightgbm(self, interval, n_trials: int = 20) -> list:
        log(f"Tuning LightGBM with {n_trials} trials...")

        X_tune_train, y_tune_train, X_tune_val, y_tune_val, tune_val_df = self._make_tuning_split()

        def objective(trial):
            sampled_params = {
                "n_estimators": trial.suggest_categorical("n_estimators", [700, 900, 1100]),
                "learning_rate": trial.suggest_float("learning_rate", 0.028, 0.052, log=True),
                "max_depth": trial.suggest_categorical("max_depth", [6, 8, 10]),
                "num_leaves": trial.suggest_categorical("num_leaves", [31, 63]),
                "min_child_samples": trial.suggest_categorical("min_child_samples", [100, 150, 200]),
                "subsample": trial.suggest_categorical("subsample", [0.8, 0.85, 0.9]),
                "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.85, 1.0]),
                "reg_alpha": trial.suggest_categorical("reg_alpha", [0.05, 0.1, 0.25, 0.5]),
                "reg_lambda": trial.suggest_categorical("reg_lambda", [0.5, 1.0, 1.5, 2.0]),
            }

            params = self._get_lgbm_params({"LGBM": {"best_params": sampled_params}})
            model = LGBMRegressor(random_state=self.seed, verbose=-1, **params)

            model.fit(X_tune_train, y_tune_train)
            pred = model.predict(X_tune_val)

            base_df = tune_val_df.copy()
            base_df["pred"] = pred

            scored_df = self._apply_signals(
                base_df,
                pred=None,
                max_top_tickers=self.config.max_top_tickers,
                allow_short=self.config.allow_short,
            )

            row_metrics = self.evaluate_strategy(scored_df)
            portfolio_metrics = self.evaluate_rotating_portfolio(
                base_df,
                max_top_tickers=self.config.max_top_tickers,
                allow_short=self.config.allow_short,
            )

            score = self._portfolio_tuning_score(portfolio_metrics)

            trial.set_user_attr("mae", float(mean_absolute_error(y_tune_val, pred)))
            trial.set_user_attr("rmse", float(mean_squared_error(y_tune_val, pred) ** 0.5))
            trial.set_user_attr("rank_ic_mean", row_metrics["rank_ic_mean"])
            trial.set_user_attr("hit_rate", row_metrics["hit_rate"])
            trial.set_user_attr("median_return", row_metrics["median_return"])
            trial.set_user_attr("portfolio", portfolio_metrics)

            log(json.dumps({
                "trial": trial.number,
                "score": float(score),
                "params": sampled_params,
                "portfolio": portfolio_metrics,
            }, indent=4))

            return score

        study = optuna.create_study(
            study_name=f"lgbm_optuna_{interval}",
            direction="maximize",
            storage=f"sqlite:///lgbm_optuna_{interval}.db",
            load_if_exists=True,
        )

        study.optimize(objective, n_trials=n_trials)

        complete_trials = [t for t in study.trials if t.value is not None]
        top_trials = sorted(complete_trials, key=lambda t: t.value, reverse=True)[:3]

        top_results = []
        for t in top_trials:
            top_results.append({
                "trial": t.number,
                "score": float(t.value),
                "params": t.params,
                "mae": t.user_attrs.get("mae"),
                "rmse": t.user_attrs.get("rmse"),
                "rank_ic_mean": t.user_attrs.get("rank_ic_mean"),
                "hit_rate": t.user_attrs.get("hit_rate"),
                "median_return": t.user_attrs.get("median_return"),
                "portfolio": t.user_attrs.get("portfolio"),
            })

        log("Top 3 LightGBM Optuna results:")
        log(json.dumps(top_results, indent=4))

        return top_results

    def tune_catboost(self, interval, n_trials: int = 75) -> list[dict[str, Any]]:
        log(f"Tuning CatBoost with Optuna for {n_trials} trials...")

        X_tune_train, y_tune_train, X_tune_val, y_tune_val, tune_val_df = self._make_tuning_split()

        def objective(trial):
            sampled_params = {
                "iterations": trial.suggest_categorical("iterations", [700, 1000, 1200]),
                "learning_rate": trial.suggest_float("learning_rate", 0.026, 0.045, log=True),
                "depth": trial.suggest_categorical("depth", [5, 6, 7]),
                "l2_leaf_reg": trial.suggest_categorical("l2_leaf_reg", [3.0, 5.0, 7.0, 10.0]),
                "random_strength": trial.suggest_categorical("random_strength", [1.0, 2.0, 3.0]),
                "bagging_temperature": trial.suggest_categorical("bagging_temperature", [0.5, 1.0, 1.5]),
                "border_count": trial.suggest_categorical("border_count", [128, 254]),
            }

            params = self._get_cat_params({"CAT": {"best_params": sampled_params}})
            model = CatBoostRegressor(random_seed=self.seed, verbose=False, **params)

            try:
                log(f"Starting CatBoost Optuna trial {trial.number}: {sampled_params}")
                print(f"Starting CatBoost Optuna trial {trial.number}", flush=True)

                model.fit(X_tune_train, y_tune_train)
                pred = model.predict(X_tune_val)

                base_df = tune_val_df.copy()
                base_df["pred"] = pred

                scored_df = self._apply_signals(
                    base_df,
                    pred=None,
                    max_top_tickers=self.config.max_top_tickers,
                    allow_short=self.config.allow_short,
                )

                row_metrics = self.evaluate_strategy(scored_df)
                portfolio_metrics = self.evaluate_rotating_portfolio(
                    base_df,
                    max_top_tickers=self.config.max_top_tickers,
                    allow_short=self.config.allow_short,
                )

                score = self._portfolio_tuning_score(portfolio_metrics)

                trial.set_user_attr("mae", float(mean_absolute_error(y_tune_val, pred)))
                trial.set_user_attr("rmse", float(mean_squared_error(y_tune_val, pred) ** 0.5))
                trial.set_user_attr("rank_ic_mean", row_metrics["rank_ic_mean"])
                trial.set_user_attr("hit_rate", row_metrics["hit_rate"])
                trial.set_user_attr("median_return", row_metrics["median_return"])
                trial.set_user_attr("portfolio", portfolio_metrics)

                result = {
                    "trial": trial.number,
                    "score": float(score),
                    "mae": trial.user_attrs["mae"],
                    "rmse": trial.user_attrs["rmse"],
                    "rank_ic_mean": row_metrics["rank_ic_mean"],
                    "hit_rate": row_metrics["hit_rate"],
                    "median_return": row_metrics["median_return"],
                    "portfolio": portfolio_metrics,
                    "params": sampled_params,
                }

                log(json.dumps(result, indent=4))

                return score

            except Exception as e:
                log(f"CatBoost Optuna trial {trial.number} failed: {e}")
                return -np.inf

            finally:
                del model
                flush_memory()

        study = optuna.create_study(
            study_name=f"cat_optuna_{interval}",
            direction="maximize",
            storage=f"sqlite:///cat_optuna_{interval}.db",
            load_if_exists=True,
        )

        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

        complete_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and np.isfinite(t.value)
        ]

        top_trials = sorted(complete_trials, key=lambda t: t.value, reverse=True)[:3]

        top_results = []
        for t in top_trials:
            top_results.append({
                "trial": t.number,
                "score": float(t.value),
                "params": t.params,
                "mae": t.user_attrs.get("mae"),
                "rmse": t.user_attrs.get("rmse"),
                "rank_ic_mean": t.user_attrs.get("rank_ic_mean"),
                "hit_rate": t.user_attrs.get("hit_rate"),
                "median_return": t.user_attrs.get("median_return"),
                "portfolio": t.user_attrs.get("portfolio"),
            })

        if not top_results:
            log("CatBoost Optuna tuning failed. No successful trials.")
            return []

        log("Top 3 CatBoost Optuna results:")
        log(json.dumps(top_results, indent=4))

        return top_results

    def run_tuning_pipeline(self, interval) -> bool:
        log(f"{'='*50}\nSTARTING {interval}\n{'=' * 50}")
        if interval == "1d":
            self.config.horizon = 40
            self.config.max_top_tickers = 30
        elif interval == "1h":
            self.config.horizon = 30
            self.config.max_top_tickers = 10

        log("Building universe dataframe...")
        data = self._build_universe_frame(interval)

        log("Preparing pooled features...")
        self._prepare_data(data)
        log("DEBUG: finished _prepare_data")
        print("DEBUG: finished _prepare_data", flush=True)

        top_lgbm_results = self.tune_lightgbm(interval, n_trials=100)
        with open(f"lgbm_top_results_{interval}.json", "w") as f:
            json.dump(top_lgbm_results, f, indent=4)

        best_lgbm_params = top_lgbm_results[0]["params"] if top_lgbm_results else {}
        with open(f"lgbm_params_{interval}.json", "w") as f:
            json.dump(best_lgbm_params, f, indent=4)

        top_cat_results = self.tune_catboost(interval, n_trials=100)
        with open(f"cat_top_results_{interval}.json", "w") as f:
            json.dump(top_cat_results, f, indent=4)

        best_cat_params = top_cat_results[0]["params"] if top_cat_results else {}
        with open(f"cat_params_{interval}.json", "w") as f:
            json.dump(best_cat_params, f, indent=4)

        return True

########################################################################################################################
    # new
    def _min_rolling_dollar_volume(self, interval: str) -> float:
        if interval == "1h":
            return self.config.min_rolling_dollar_volume_1h
        return self.config.min_rolling_dollar_volume_1d

    def _min_rows_after_filter(self, interval: str) -> int:
        if interval == "1h":
            return self.config.min_rows_after_filter_1h
        return self.config.min_rows_after_filter_1d

    def _liquidity_window(self, interval: str) -> int:
        if interval == "1h":
            return 120  # roughly a few weeks of hourly bars
        return 20  # about one trading month

    def _passes_static_liquidity_filter(self, ticker: str, meta: dict[str, Any]) -> bool:
        adv = meta.get("adv", np.nan)

        if self.config.min_profile_adv > 0 and np.isfinite(adv):
            if float(adv) < self.config.min_profile_adv:
                log(f"Skipping {ticker}: profile ADV too low ({adv:,.0f})", False)
                return False

        return True

    def _add_liquidity_columns(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        df = df.copy()

        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        window = self._liquidity_window(interval)

        df["dollar_volume"] = df[price_col].astype(float) * df["Volume"].astype(float)

        # Shifted by 1 so today’s filter uses only previous bars.
        df["rolling_dollar_volume"] = (
            df["dollar_volume"]
            .rolling(window=window, min_periods=max(5, window // 4))
            .median()
            .shift(1)
        )

        df["abs_bar_return"] = df[price_col].pct_change().abs()

        # Safe feature if you want the model to know liquidity context.
        df["liquidity_dollar_volume_log"] = np.log1p(
            df["rolling_dollar_volume"].clip(lower=0)
        )

        return df

    def _apply_liquidity_filters(self, df: pd.DataFrame, ticker: str, interval: str) -> pd.DataFrame:
        df = df.copy()

        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

        min_dollar_volume = self._min_rolling_dollar_volume(interval)

        before = len(df)

        mask = (
                df[price_col].between(self.config.min_price, self.config.max_price)
                & (df["rolling_dollar_volume"] >= min_dollar_volume)
                & (df["abs_bar_return"] <= self.config.max_abs_bar_return)
        )

        df = df[mask].copy()

        after = len(df)
        min_rows = self._min_rows_after_filter(interval)

        if after < min_rows:
            log(f"Skipping {ticker}: too few rows after liquidity filter ({after}/{before})", False)
            return pd.DataFrame()

        if after < before:
            log(f"{ticker}: liquidity filter kept {after:,}/{before:,} rows", False)

        return df

    def evaluate_cost_sensitivity(self, base_df: pd.DataFrame, bps_values: tuple = (10.0, 25.0, 50.0, 100.0)) -> dict:
        old_cost = self.config.cost_bps
        results = {}

        try:
            for bps in bps_values:
                self.config.cost_bps = bps
                results[f"{bps:g}bps"] = self.evaluate_rotating_portfolio(base_df)
        finally:
            self.config.cost_bps = old_cost

        return results

    def evaluate_portfolio_by_year(self, base_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        df = base_df.copy()
        df["year"] = pd.to_datetime(df["Date"]).dt.year

        results = {}

        for year, year_df in df.groupby("year"):
            unique_dates = year_df["Date"].nunique()

            # Skip tiny partial years.
            if unique_dates < 60:
                continue

            results[str(year)] = self.evaluate_rotating_portfolio(
                year_df.drop(columns=["year"])
            )

        return results

    def analyse_drawdown(self, base_df: pd.DataFrame) -> dict[str, Any]:
        old_cost = self.config.cost_bps

        df = base_df.copy()
        df = df.sort_values(["ticker", "Date"])
        df["next_return"] = df.groupby("ticker")["Adj Close"].shift(-1) / df["Adj Close"] - 1.0
        df = df.sort_values(["Date", "ticker"])

        dates = list(sorted(pd.to_datetime(df["Date"]).unique()))
        by_date = {date: group.copy() for date, group in df.groupby("Date", sort=True)}

        holdings = {}
        daily_returns = []
        daily_dates = []

        cost = self.config.cost_bps / 10_000
        max_positions = self.config.max_top_tickers
        max_holding_days = self.config.horizon

        for date in dates[:-1]:
            day = by_date[date].copy()
            day = day.dropna(subset=["pred", "next_return", "Adj Close"])

            if day.empty:
                continue

            exit_signal_day = self.add_cross_sectional_signals(
                day,
                "pred",
                ["Date", "profile_group"],
                allow_short=True,
            ).set_index("ticker", drop=False)

            candidates = self._daily_candidate_pool(day)
            turnover = 0

            for ticker in list(holdings.keys()):
                if ticker not in exit_signal_day.index:
                    holdings.pop(ticker)
                    turnover += 1
                    continue

                row = exit_signal_day.loc[ticker]
                side = holdings[ticker]["side"]
                age = holdings[ticker]["age"]

                opposite_signal = row["pred_signal"] == -side
                weak_long = side == 1 and row["pred"] <= self.config.exit_pred_threshold
                too_old = age >= max_holding_days

                if opposite_signal or weak_long or too_old:
                    holdings.pop(ticker)
                    turnover += 1

            for _, row in candidates.iterrows():
                ticker = row["ticker"]
                side = int(row["pred_signal"])
                score = float(row["score"])

                if ticker in holdings:
                    holdings[ticker]["score"] = score
                    holdings[ticker]["side"] = side
                    continue

                if len(holdings) < max_positions:
                    holdings[ticker] = {"side": side, "score": score, "age": 0}
                    turnover += 1
                    continue

                weakest_ticker = min(holdings, key=lambda t: holdings[t]["score"])
                weakest_score = holdings[weakest_ticker]["score"]

                if score > weakest_score + self.config.replace_buffer:
                    holdings.pop(weakest_ticker)
                    holdings[ticker] = {"side": side, "score": score, "age": 0}
                    turnover += 2

            day_indexed = day.set_index("ticker", drop=False)
            position_returns = []

            for ticker, info in holdings.items():
                if ticker not in day_indexed.index:
                    continue

                next_return = day_indexed.loc[ticker, "next_return"]
                if np.isfinite(next_return):
                    position_returns.append(info["side"] * float(next_return))

            gross_return = float(np.mean(position_returns)) if position_returns else 0.0
            turnover_cost = cost * turnover / max(1, max_positions)
            portfolio_return = gross_return - turnover_cost

            daily_dates.append(date)
            daily_returns.append(portfolio_return)

            for info in holdings.values():
                info["age"] += 1

        returns = pd.Series(daily_returns, index=pd.to_datetime(daily_dates), dtype=float)
        equity = (1.0 + returns).cumprod()
        running_peak = equity.cummax()
        drawdown = equity / running_peak - 1.0

        trough_date = drawdown.idxmin()
        peak_date = equity.loc[:trough_date].idxmax()

        return {
            "max_drawdown": float(drawdown.min()),
            "peak_date": str(peak_date.date()),
            "trough_date": str(trough_date.date()),
            "peak_equity": float(equity.loc[peak_date]),
            "trough_equity": float(equity.loc[trough_date]),
        }

    def benchmark_drawdown(self, interval: str, start_date, end_date) -> dict[str, Any]:
        spy = pd.read_parquet(os.path.join(DATA_DIR, f"SPY_{interval}.parquet"))
        spy.index = pd.to_datetime(spy.index, utc=True).tz_localize(None)
        spy = spy.loc[(spy.index >= start_date) & (spy.index <= end_date)].copy()

        if spy.empty or len(spy) < 2:
            return {
                "spy_total_return": 0.0,
                "spy_max_drawdown": 0.0,
            }

        returns = spy["Adj Close"].pct_change().dropna()
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0

        return {
            "spy_total_return": float(equity.iloc[-1] - 1.0),
            "spy_max_drawdown": float(drawdown.min()),
        }

    def _rolling_vol_window(self, interval: str) -> int:
        if interval == "1h":
            return self.config.rolling_vol_window_1h
        return self.config.rolling_vol_window_1d

    def _rolling_beta_window(self, interval: str) -> int:
        if interval == "1h":
            return self.config.rolling_beta_window_1h
        return self.config.rolling_beta_window_1d

    def _rolling_annualiser(self, interval: str) -> float:
        if interval == "1h":
            return np.sqrt(252 * 6.5)
        return np.sqrt(252)

    def _add_rolling_risk_columns(self, df: pd.DataFrame, benchmark_close: pd.Series, interval: str) -> pd.DataFrame:
        df = df.copy()

        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

        vol_window = self._rolling_vol_window(interval)
        beta_window = self._rolling_beta_window(interval)
        annualiser = self._rolling_annualiser(interval)

        close = df[price_col].astype(float)
        stock_ret = close.pct_change()

        benchmark_close = benchmark_close.reindex(df.index).ffill()
        benchmark_ret = benchmark_close.pct_change()

        min_vol_periods = max(10, vol_window // 4)
        min_beta_periods = max(20, beta_window // 4)

        df["rolling_vol"] = (
                stock_ret
                .rolling(window=vol_window, min_periods=min_vol_periods)
                .std()
                .shift(1)
                * annualiser
        )

        rolling_cov = (
            stock_ret
            .rolling(window=beta_window, min_periods=min_beta_periods)
            .cov(benchmark_ret)
            .shift(1)
        )

        rolling_market_var = (
            benchmark_ret
            .rolling(window=beta_window, min_periods=min_beta_periods)
            .var()
            .shift(1)
        )

        df["rolling_beta"] = rolling_cov / (rolling_market_var + 1e-12)
        df["rolling_beta"] = df["rolling_beta"].replace([np.inf, -np.inf], np.nan)
        df["rolling_beta"] = df["rolling_beta"].clip(-3.0, 3.0)

        df["rolling_market_corr"] = (
            stock_ret
            .rolling(window=beta_window, min_periods=min_beta_periods)
            .corr(benchmark_ret)
            .shift(1)
        )

        df["rolling_market_vol"] = (
                benchmark_ret
                .rolling(window=vol_window, min_periods=min_vol_periods)
                .std()
                .shift(1)
                * annualiser
        )

        return df

########################################################################################################################

    def add_forward_excess_target(self, df: pd.DataFrame, benchmark_close: pd.Series, beta: float | pd.Series = 1.0) -> pd.DataFrame:
        df = df.copy()

        close = df["Adj Close"]
        benchmark_close = benchmark_close.reindex(df.index).ffill()

        df["future_return"] = close.shift(-self.config.horizon) / close - 1.0
        df["benchmark_future_return"] = benchmark_close.shift(-self.config.horizon) / benchmark_close - 1.0

        if isinstance(beta, pd.Series):
            beta_used = beta.reindex(df.index).ffill()
            beta_used = beta_used.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        else:
            beta_used = 1.0 if not np.isfinite(beta) else float(beta)

        df["target_beta_used"] = beta_used
        df["target_excess_return"] = df["future_return"] - beta_used * df["benchmark_future_return"]

        return df.dropna(subset=["target_excess_return"])

    def _build_universe_frame(self, interval: str) -> pd.DataFrame:
        with open(os.path.join(DATA_DIR, "ticker_attr.json"), "r") as f:
            ticker_map = json.load(f)

        ticker_list = sorted(set(ticker_map.keys()))#[:100]

        benchmark_raw = pd.read_parquet(os.path.join(DATA_DIR, f"SPY_{interval}.parquet"))
        benchmark_raw.index.name = "Date"
        benchmark_raw.index = pd.to_datetime(benchmark_raw.index, utc=True).tz_localize(None)
        benchmark_raw = benchmark_raw[~benchmark_raw.index.duplicated(keep="first")].sort_index()
        benchmark_close = benchmark_raw["Adj Close"]

        frames = []
        for ticker in tqdm(ticker_list):
            try:
                raw = load_data(ticker, interval)
                if raw is None or raw.empty:
                    log(f"Skipping {ticker}: no data")
                    continue

                meta = ticker_map.get(ticker, {})
                if not self._passes_static_liquidity_filter(ticker, meta): continue

                df = raw.ind.add_indicators(ticker, interval, add_targets=False)

                df = self._add_liquidity_columns(df, interval)
                df = self._add_rolling_risk_columns(df, benchmark_close, interval)

                df = self.add_forward_excess_target(df, benchmark_close, df["rolling_beta"])

                df["ticker"] = ticker
                df["profile"] = "All"
                df["profile_vol"] = df["rolling_vol"]
                df["profile_beta"] = df["rolling_beta"]
                df["profile_adv_log"] = df["liquidity_dollar_volume_log"]
                df["Date"] = df.index

                df = self._apply_liquidity_filters(df, ticker, interval)
                if df.empty: continue

                df = df.reset_index(drop=True)
                frames.append(df)

            except Exception as e:
                log(f"Skipping {ticker}: {e}")

        if not frames:
            raise ValueError("No usable universe data")

        data = pd.concat(frames, axis=0, ignore_index=True)
        data = data.sort_values(["Date", "ticker"])
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()

        return data

    def _prepare_data(self, data: pd.DataFrame) -> None:
        data = data.copy()
        data["profile_group"] = data["profile"]

        # Dynamic risk buckets: each date ranks stocks by their rolling volatility.
        rank_pct = data.groupby("Date")["profile_vol"].rank(method="first", pct=True)

        data["risk_bucket"] = pd.cut(
            rank_pct,
            bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["vol_1", "vol_2", "vol_3", "vol_4", "vol_5"],
            include_lowest=True,
        )

        data["risk_bucket"] = data["risk_bucket"].astype(str).fillna("unknown")
        data["profile_group"] = data["risk_bucket"]

        data = pd.get_dummies(data, columns=["profile", "risk_bucket"], prefix=["profile", "risk_bucket"], dtype=int)

        drop_cols = {
            "Open", "High", "Low", "Close", "Adj Close", "Volume",
            "Adj Open", "Adj High", "Adj Low",
            "MA_200", "return",
            "ticker", "Date", "profile", "profile_group",
            "future_return", "benchmark_future_return", "target_excess_return",
            "target_profit", "tbm_return", "barrier_strength",
            "time_to_gain", "time_to_loss", "tp_return", "sl_return",
            "dollar_volume", "rolling_dollar_volume", "abs_bar_return", "target_beta_used"
        }

        self.feature_cols = [
            c for c in data.columns
            if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])
        ]

        unique_dates = np.array(sorted(pd.to_datetime(data["Date"]).unique()))
        self.split_date = unique_dates[int(len(unique_dates) * (1 - self.__test_size))]

        self.train_df = data[data["Date"] <= self.split_date].copy()
        self.test_df = data[data["Date"] > self.split_date].copy()

        self.X_train = self.train_df[self.feature_cols].to_numpy(dtype=np.float32)
        self.X_test = self.test_df[self.feature_cols].to_numpy(dtype=np.float32)

        self.y_train = self.train_df["target_excess_return"].to_numpy(dtype=np.float32)
        self.y_test = self.test_df["target_excess_return"].to_numpy(dtype=np.float32)

        log(f"Features: {len(self.feature_cols)}")
        log(f"Train rows: {len(self.train_df):,}")
        log(f"Test rows: {len(self.test_df):,}")
        log(f"Split date: {pd.Timestamp(self.split_date).strftime('%Y-%m-%d')}")

    def _effective_edge_q(self, df: pd.DataFrame) -> float:
        n_tickers = max(1, df["ticker"].nunique())
        return min(self.config.edge_q, self.config.max_top_tickers / n_tickers)

    def add_cross_sectional_signals(self, df: pd.DataFrame, pred_col: str, group_cols: list | None = None, allow_short: bool | None = None) -> pd.DataFrame:
        df = df.copy()
        df["pred_signal"] = 0

        if group_cols is None:
            group_cols = ["Date", "profile_group"]

        allow_short = self.config.allow_short if allow_short is None else allow_short
        edge_q = self._effective_edge_q(df)

        for _, group in df.groupby(group_cols):
            if len(group) < 5:
                continue

            upper = group[pred_col].quantile(1 - edge_q)
            lower = group[pred_col].quantile(edge_q)

            long_mask = (group[pred_col] >= upper) & (group[pred_col].abs() >= self.config.min_abs_pred)
            df.loc[group.index[long_mask], "pred_signal"] = 1

            if allow_short:
                short_mask = (group[pred_col] <= lower) & (group[pred_col].abs() >= self.config.min_abs_pred)
                df.loc[group.index[short_mask], "pred_signal"] = -1

        return df

    def _apply_signals(self, test_df: pd.DataFrame, pred: np.ndarray | None = None) -> pd.DataFrame:
        out = test_df.copy()

        if pred is not None:
            out["pred"] = pred

        return self.add_cross_sectional_signals(
            out,
            "pred",
            ["Date", "profile_group"],
        )

    def evaluate_strategy(self, df: pd.DataFrame) -> dict[str, Any]:
        trades = df[df["pred_signal"] != 0].copy()

        if trades.empty:
            return {
                "trade_rate": 0.0,
                "long_rate": 0.0,
                "short_rate": 0.0,
                "hit_rate": 0.0,
                "mean_return": 0.0,
                "median_return": 0.0,
                "sharpe_like": 0.0,
                "rank_ic_mean": 0.0,
                "rank_ic_ir": 0.0,
                "n_trades": 0,
            }

        cost = self.config.cost_bps / 10_000
        trades["strategy_return"] = trades["pred_signal"] * trades["target_excess_return"] - cost

        mean_return = trades["strategy_return"].mean()
        std_return = trades["strategy_return"].std() + 1e-9

        daily_ics = []
        for _, group in df.groupby("Date"):
            if len(group) < 5:
                continue

            ic = group["pred"].corr(group["target_excess_return"], method="spearman")

            if np.isfinite(ic):
                daily_ics.append(ic)

        rank_ic_mean = float(np.mean(daily_ics)) if daily_ics else 0.0
        rank_ic_std = float(np.std(daily_ics)) + 1e-9 if daily_ics else 1.0

        long_trades = trades[trades["pred_signal"] == 1]
        short_trades = trades[trades["pred_signal"] == -1]

        def side_stats(side_df):
            if side_df.empty:
                return {"hit_rate": 0.0, "mean_return": 0.0, "median_return": 0.0, "n": 0}

            return {
                "hit_rate": float((side_df["strategy_return"] > 0).mean()),
                "mean_return": float(side_df["strategy_return"].mean()),
                "median_return": float(side_df["strategy_return"].median()),
                "n": int(len(side_df)),
            }

        return {
            "trade_rate": float((df["pred_signal"] != 0).mean()),  # noqa
            "long_rate": float((df["pred_signal"] == 1).mean()),  # noqa
            "short_rate": float((df["pred_signal"] == -1).mean()),  # noqa
            "hit_rate": float((trades["strategy_return"] > 0).mean()),  # noqa
            "mean_return": float(mean_return),
            "median_return": float(trades["strategy_return"].median()),
            "sharpe_like": float((mean_return / std_return) * np.sqrt(252 / self.config.horizon)),
            "rank_ic_mean": rank_ic_mean,
            "rank_ic_ir": float(rank_ic_mean / rank_ic_std),
            "n_trades": int(len(trades)),
            "long_stats": side_stats(long_trades),
            "short_stats": side_stats(short_trades),
        }

    def evaluate_baselines(self) -> dict[str, dict[str, Any]]:
        results = {}
        rng = np.random.default_rng(self.seed)

        random_df = self.test_df.copy()
        random_df["pred"] = rng.normal(0, 1, len(random_df))
        random_df = self.add_cross_sectional_signals(random_df, "pred", ["Date", "profile_group"])
        results["random"] = self.evaluate_strategy(random_df)

        momentum_df = self.test_df.copy()
        momentum_df["pred"] = momentum_df["mom_1m"]
        momentum_df = self.add_cross_sectional_signals(momentum_df, "pred", ["Date", "profile_group"])
        results["momentum_1m"] = self.evaluate_strategy(momentum_df)

        reversal_df = self.test_df.copy()
        reversal_df["pred"] = -reversal_df["mom_1m"]
        reversal_df = self.add_cross_sectional_signals(reversal_df, "pred", ["Date", "profile_group"])
        results["reversal_1m"] = self.evaluate_strategy(reversal_df)

        return results

    def _daily_candidate_pool(self, day: pd.DataFrame) -> pd.DataFrame:
        signalled = self.add_cross_sectional_signals(day, "pred", ["Date", "profile_group"])

        candidates = signalled[signalled["pred_signal"] != 0].copy()
        if candidates.empty:
            return candidates

        candidates["score"] = np.where(
            candidates["pred_signal"] == 1,
            candidates["pred"],
            -candidates["pred"],
        )

        return candidates.sort_values("score", ascending=False)

    def evaluate_rotating_portfolio(self, base_df: pd.DataFrame) -> dict:
        max_holding_days = self.config.horizon

        cost = self.config.cost_bps / 10_000
        max_positions = self.config.max_top_tickers

        df = base_df.copy()
        df = df.sort_values(["ticker", "Date"])
        df["next_return"] = df.groupby("ticker")["Adj Close"].shift(-1) / df["Adj Close"] - 1.0
        df = df.sort_values(["Date", "ticker"])

        dates = list(sorted(pd.to_datetime(df["Date"]).unique()))
        by_date = {date: group.copy() for date, group in df.groupby("Date", sort=True)}

        holdings: dict[str, dict[str, Any]] = {}
        daily_returns = []
        daily_turnover = []
        daily_holding_count = []

        for date in dates[:-1]:
            day = by_date[date].copy()
            day = day.dropna(subset=["pred", "next_return", "Adj Close"])

            if day.empty:
                continue

            # This one allows short signals even in long-only mode, but only for exit warnings.
            exit_signal_day = self.add_cross_sectional_signals(
                day,
                "pred",
                ["Date", "profile_group"],
                allow_short=True,
            ).set_index("ticker", drop=False)

            candidates = self._daily_candidate_pool(day)
            turnover = 0

            # Exit holdings that disappeared, became weak, hit max age, or got opposite signal.
            for ticker in list(holdings.keys()):
                if ticker not in exit_signal_day.index:
                    holdings.pop(ticker)
                    turnover += 1
                    continue

                row = exit_signal_day.loc[ticker]
                side = holdings[ticker]["side"]
                age = holdings[ticker]["age"]

                opposite_signal = row["pred_signal"] == -side
                weak_long = side == 1 and row["pred"] <= self.config.exit_pred_threshold
                weak_short = side == -1 and row["pred"] >= -self.config.exit_pred_threshold
                too_old = age >= max_holding_days

                if opposite_signal or weak_long or weak_short or too_old:
                    holdings.pop(ticker)
                    turnover += 1

            # Add/replace holdings using today's strongest candidates.
            for _, row in candidates.iterrows():
                ticker = row["ticker"]
                side = int(row["pred_signal"])
                score = float(row["score"])

                if ticker in holdings:
                    holdings[ticker]["score"] = score
                    holdings[ticker]["side"] = side
                    continue

                if len(holdings) < max_positions:
                    holdings[ticker] = {"side": side, "score": score, "age": 0}
                    turnover += 1
                    continue

                weakest_ticker = min(holdings, key=lambda t: holdings[t]["score"])
                weakest_score = holdings[weakest_ticker]["score"]

                if score > weakest_score + self.config.replace_buffer:
                    holdings.pop(weakest_ticker)
                    holdings[ticker] = {"side": side, "score": score, "age": 0}
                    turnover += 2

            if not holdings:
                daily_returns.append(0.0)
                daily_turnover.append(turnover)
                daily_holding_count.append(0)
                continue

            day_indexed = day.set_index("ticker", drop=False)
            position_returns = []

            for ticker, info in holdings.items():
                if ticker not in day_indexed.index:
                    continue

                next_return = day_indexed.loc[ticker, "next_return"]
                if not np.isfinite(next_return):
                    continue

                position_returns.append(info["side"] * float(next_return))

            gross_return = float(np.mean(position_returns)) if position_returns else 0.0
            turnover_cost = cost * turnover / max(1, max_positions)
            portfolio_return = gross_return - turnover_cost

            daily_returns.append(portfolio_return)
            daily_turnover.append(turnover)
            daily_holding_count.append(len(holdings))

            for info in holdings.values():
                info["age"] += 1

        if not daily_returns:
            return {
                "portfolio_days": 0,
                "portfolio_total_return": 0.0,
                "portfolio_cagr_like": 0.0,
                "portfolio_sharpe_like": 0.0,
                "portfolio_max_drawdown": 0.0,
                "portfolio_win_rate": 0.0,
                "portfolio_mean_daily_return": 0.0,
                "portfolio_median_daily_return": 0.0,
                "portfolio_avg_holdings": 0.0,
                "portfolio_avg_turnover": 0.0,
            }

        returns = pd.Series(daily_returns, dtype=float)
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0

        years = len(returns) / 252
        total_return = equity.iloc[-1] - 1.0
        cagr_like = equity.iloc[-1] ** (1 / years) - 1.0 if years > 0 else 0.0
        sharpe_like = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)

        return {
            "portfolio_days": int(len(returns)),
            "portfolio_total_return": float(total_return),
            "portfolio_cagr_like": float(cagr_like),
            "portfolio_sharpe_like": float(sharpe_like),
            "portfolio_max_drawdown": float(drawdown.min()),
            "portfolio_win_rate": float((returns > 0).mean()),
            "portfolio_mean_daily_return": float(returns.mean()),
            "portfolio_median_daily_return": float(returns.median()),
            "portfolio_avg_holdings": float(np.mean(daily_holding_count)),
            "portfolio_avg_turnover": float(np.mean(daily_turnover)),
        }

    def _train_lightgbm(self, interval: str) -> dict:
        with open(os.path.join(ROOT_DIR, "results", f"lgbm_params_{interval}.json"), "r") as f:
            params = json.load(f)

        if Settings.GPU["LGBM"]:
            params.update({"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
        if Settings.Threaded:
            params.update({"num_threads": -1, "n_jobs": -1})
        params.update({"objective": "regression"})

        model = LGBMRegressor(random_state=self.seed, verbose=-1, **params)
        model.fit(self.X_train, self.y_train)

        pred = model.predict(self.X_test)

        base_df = self.test_df.copy()
        base_df["pred"] = pred

        scored_df = self._apply_signals(base_df)
        strategy_metrics = self.evaluate_strategy(scored_df)
        portfolio_metrics = self.evaluate_rotating_portfolio(base_df)
        cost_sensitivity = self.evaluate_cost_sensitivity(base_df)
        yearly_portfolio = self.evaluate_portfolio_by_year(base_df)

        drawdown_info = self.analyse_drawdown(base_df)
        benchmark_dd = self.benchmark_drawdown(
            interval,
            drawdown_info["peak_date"],
            drawdown_info["trough_date"],
        )

        drawdown_info["benchmark"] = benchmark_dd
        drawdown_info["model_type"] = "LGBM"

        log(f"{drawdown_info.get('model_type', 'Model')} drawdown analysis:")
        log(json.dumps(drawdown_info, indent=4))

        return {
            "type": "LGBM",
            "model": model,
            "test_df": scored_df,
            "base_df": base_df,
            "mae": float(mean_absolute_error(self.y_test, pred)),
            "rmse": float(mean_squared_error(self.y_test, pred) ** 0.5),
            **strategy_metrics,
            "portfolio": portfolio_metrics,
            "yearly_portfolio": yearly_portfolio,
            "cost_sensitivity": cost_sensitivity,
            "drawdown_analysis": drawdown_info,
        }

    def _train_catboost(self, interval: str) -> dict:
        with open(os.path.join(ROOT_DIR, "results", f"cat_params_{interval}.json"), "r") as f:
            params = json.load(f)

        if Settings.Threaded:
            params.update({"thread_count": -1})

        params.update({
            "loss_function": "RMSE",
            "allow_writing_files": False,
            "task_type": "GPU" if Settings.GPU["CAT"] else "CPU",
        })

        model = CatBoostRegressor(random_seed=self.seed, verbose=False, **params)
        model.fit(self.X_train, self.y_train)

        pred = model.predict(self.X_test)

        base_df = self.test_df.copy()
        base_df["pred"] = pred

        scored_df = self._apply_signals(base_df)
        strategy_metrics = self.evaluate_strategy(scored_df)
        portfolio_metrics = self.evaluate_rotating_portfolio(base_df)
        yearly_portfolio = self.evaluate_portfolio_by_year(base_df)
        cost_sensitivity = self.evaluate_cost_sensitivity(base_df)

        drawdown_info = self.analyse_drawdown(base_df)
        benchmark_dd = self.benchmark_drawdown(
            interval,
            drawdown_info["peak_date"],
            drawdown_info["trough_date"],
        )

        drawdown_info["model_type"] = "CAT"
        drawdown_info["benchmark"] = benchmark_dd

        log("CAT drawdown analysis:")
        log(json.dumps(drawdown_info, indent=4))

        return {
            "type": "CAT",
            "model": model,
            "test_df": scored_df,
            "base_df": base_df,
            "mae": float(mean_absolute_error(self.y_test, pred)),
            "rmse": float(mean_squared_error(self.y_test, pred) ** 0.5),
            **strategy_metrics,
            "portfolio": portfolio_metrics,
            "yearly_portfolio": yearly_portfolio,
            "cost_sensitivity": cost_sensitivity,
            "drawdown_analysis": drawdown_info,
        }

    def _save_model_assets(self, interval: str, results: dict, baselines: dict):
        save_folder = Path(os.path.join(MODEL_DIR, f"{interval} Model"))
        save_folder.mkdir(parents=True, exist_ok=True)

        metadata = {
            "training_date": datetime.now().strftime("%Y-%m-%d"),
            "config": asdict(self.config),
            "split_date": pd.Timestamp(self.split_date).strftime("%Y-%m-%d"),
            "feature_count": len(self.feature_cols),
            "model_results": {},
            "baselines": baselines,
        }

        for model_type, model_results in results.items():
            metadata["model_results"][model_type] = {
                key: value for key, value in model_results.items()
                if key not in {"model", "test_df", "base_df"}
            }

            if model_type == "LGBM":
                model_results["model"].booster_.save_model(str(save_folder / "lgbm_model.txt"))
            elif model_type == "CAT":
                joblib.dump(model_results["model"], save_folder / "cat_model.joblib")

        joblib.dump(self.feature_cols, save_folder / "features.joblib")
        (save_folder / "metadata.json").write_text(json.dumps(metadata, indent=4), encoding="utf-8")

    def rank_latest_predictions(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        latest_date = scored_df["Date"].max()
        latest = scored_df[scored_df["Date"] == latest_date].copy()

        cols = [
            "Date", "ticker", "profile_group", "pred", "pred_signal",
            "target_excess_return", "future_return", "benchmark_future_return",
        ]

        cols = [c for c in cols if c in latest.columns]
        latest = latest[cols].sort_values("pred", ascending=False)

        top = latest.head(min(self.config.max_top_tickers, len(latest)))
        bottom = latest.tail(min(self.config.max_top_tickers, len(latest)))

        if len(latest) <= self.config.max_top_tickers * 2:
            return latest
        return pd.concat([top, bottom], axis=0)

    def run_training_pipeline(self, interval: str, force_train: bool = True) -> bool:
        def log_update(msg):
            if Settings.LOGGING:
                log(msg)

        if interval == "1d":
            self.config.horizon = 40
            self.config.max_top_tickers = 30
        elif interval == "1h":
            self.config.horizon = 30
            self.config.max_top_tickers = 10

        save_folder = os.path.join(MODEL_DIR, f"{interval} Model")
        if all_model_assets_exist(save_folder) and not force_train:
            log_update(f"Universe model already trained: {save_folder}")
            return True

        if os.path.exists(save_folder): shutil.rmtree(save_folder)

        log_update("Building universe dataframe...")
        data = self._build_universe_frame(interval)

        log_update("Preparing pooled features...")
        self._prepare_data(data)

        baselines = self.evaluate_baselines()
        log("Baselines:")
        log(json.dumps(baselines, indent=4))

        results = {}

        log_update("Training LightGBM...")
        results["LGBM"] = self._train_lightgbm(interval)
        flush_memory()
        log(json.dumps({k: v for k, v in results["LGBM"].items() if k not in {"model", "test_df", "base_df"}}, indent=4))

        log_update("Training CatBoost...")
        results["CAT"] = self._train_catboost(interval)
        flush_memory()
        log(json.dumps({k: v for k, v in results["CAT"].items() if k not in {"model", "test_df", "base_df"}}, indent=4))

        log_update("Saving assets...")
        self._save_model_assets(interval, results, baselines)

        def final_model_score(result: dict) -> float:
            p = result["portfolio"]
            return (
                    p["portfolio_cagr_like"]
                    + 0.05 * p["portfolio_sharpe_like"]
                    - 0.50 * abs(p["portfolio_max_drawdown"])
                    - 0.002 * p["portfolio_avg_turnover"]
            )
        best_model_type = max(results, key=lambda m: final_model_score(results[m]))
        log(f"Best model: {best_model_type}")

        latest = self.rank_latest_predictions(results[best_model_type]["test_df"])
        log("Latest ranked predictions:")
        log(latest.to_string(index=False))

        return True


def all_model_assets_exist(model_path: str | os.PathLike) -> bool:
    root = Path(model_path)
    required_files = ["metadata.json", "features.joblib", "lgbm_model.txt", "cat_model.joblib"]

    return root.exists() and all((root / f).exists() and (root / f).stat().st_size > 0 for f in required_files)


if __name__ in "__main__":
    start = time.perf_counter()

    print("Training...")
    manager = TrainingManager()
    manager.run_training_pipeline("1d")

    # manager = TrainingManager()
    # manager.run_training_pipeline("1h")

    print(f"Total time: {time.perf_counter() - start:.1f}s")