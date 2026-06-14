
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
from sklearn.model_selection import ParameterSampler
import optuna

from scripts.config import DATA_DIR, MODEL_DIR
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

def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def log(string: str):
    with open(f"testing.txt", "a") as f:
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

    def _keep_top_results(self, top_results: list[dict[str, Any]], result: dict[str, Any], top_k: int = 3) -> list[
        dict[str, Any]]:
        top_results.append(result)
        top_results.sort(key=lambda r: r["score"], reverse=True)
        return top_results[:top_k]

    def tune_lightgbm(self, n_trials: int = 20) -> list:
        log(f"Tuning LightGBM with {n_trials} trials...")

        X_tune_train, y_tune_train, X_tune_val, y_tune_val, tune_val_df = self._make_tuning_split()

        def objective(trial):
            sampled_params = {
                "n_estimators": trial.suggest_categorical("n_estimators", [300, 500, 700]),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.05, log=True),
                "max_depth": trial.suggest_categorical("max_depth", [4, 5, 6, 8]),
                "num_leaves": trial.suggest_categorical("num_leaves", [31, 63, 127]),
                "min_child_samples": trial.suggest_categorical("min_child_samples", [100, 150, 200, 300]),
                "subsample": trial.suggest_categorical("subsample", [0.85, 1.0]),
                "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.75, 0.85, 1.0]),
                "reg_alpha": trial.suggest_categorical("reg_alpha", [0.1, 0.5, 1.0]),
                "reg_lambda": trial.suggest_categorical("reg_lambda", [0.5, 1.0, 2.0]),
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
            study_name="lgbm_optuna",
            direction="maximize",
            storage=f"sqlite:///lgbm_optuna.db",
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

    def tune_catboost_optuna(self, n_trials: int = 75, study_name: str = "catboost_tuning") -> list[dict[str, Any]]:
        import optuna

        log(f"Tuning CatBoost with Optuna for {n_trials} trials...")

        X_tune_train, y_tune_train, X_tune_val, y_tune_val, tune_val_df = self._make_tuning_split()

        def objective(trial):
            sampled_params = {
                "iterations": trial.suggest_categorical("iterations", [500, 700, 1000]),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.06, log=True),
                "depth": trial.suggest_categorical("depth", [4, 5, 6]),
                "l2_leaf_reg": trial.suggest_categorical("l2_leaf_reg", [3.0, 5.0, 10.0]),
                "random_strength": trial.suggest_categorical("random_strength", [0.5, 1.0, 2.0]),
                "bagging_temperature": trial.suggest_categorical("bagging_temperature", [1.0, 2.0, 3.0]),
                "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
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
            study_name=study_name,
            direction="maximize",
            storage=f"sqlite:///{study_name}.db",
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

    def run_tuning_pipeline(self, interval, force_train: bool = True) -> bool:
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

        top_lgbm_results = self.tune_lightgbm(n_trials=30)
        with open(f"lgbm_top_results_{interval}.json", "w") as f:
            json.dump(top_lgbm_results, f, indent=4)

        best_lgbm_params = top_lgbm_results[0]["params"] if top_lgbm_results else {}
        with open(f"lgbm_params_{interval}.json", "w") as f:
            json.dump(best_lgbm_params, f, indent=4)

        top_cat_results = self.tune_catboost(n_trials=30)
        with open(f"cat_top_results_{interval}.json", "w") as f:
            json.dump(top_cat_results, f, indent=4)

        best_cat_params = top_cat_results[0]["params"] if top_cat_results else {}
        with open(f"cat_params_{interval}.json", "w") as f:
            json.dump(best_cat_params, f, indent=4)

        return True

    ####################################


    @staticmethod
    def add_forward_excess_target(df: pd.DataFrame, benchmark_close: pd.Series, horizon: int, beta: float = 1.0) -> pd.DataFrame:
        df = df.copy()

        close = df["Adj Close"]
        benchmark_close = benchmark_close.reindex(df.index).ffill()

        beta = 1.0 if not np.isfinite(beta) else float(beta)

        df["future_return"] = close.shift(-horizon) / close - 1.0
        df["benchmark_future_return"] = benchmark_close.shift(-horizon) / benchmark_close - 1.0
        df["target_excess_return"] = df["future_return"] - beta * df["benchmark_future_return"]

        return df.dropna(subset=["target_excess_return"])

    def _build_universe_frame(self, interval: str) -> pd.DataFrame:
        with open(os.path.join(DATA_DIR, "ticker_attr.json"), "r") as f:
            ticker_map = json.load(f)

        ticker_list = sorted(set(ticker_map.keys()))[:100]
        # self.config.edge_q = min(self.config.edge_q * len(ticker_list), self.config.max_top_tickers) / len(ticker_list)

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

                beta = ticker_map.get(ticker, {}).get("beta", np.nan)
                df = raw.ind.add_indicators(ticker, interval, add_targets=False)
                df = self.add_forward_excess_target(df, benchmark_close, self.config.horizon, beta)

                df["ticker"] = ticker
                df["profile"] = "All"
                df["profile_vol"] = ticker_map.get(ticker, {}).get("vol", np.nan)
                df["profile_beta"] = beta
                df["profile_adv_log"] = np.log1p(ticker_map.get(ticker, {}).get("adv", np.nan))
                df["Date"] = df.index

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

        ticker_risk = data[["ticker", "profile_vol"]].drop_duplicates("ticker").dropna(subset=["profile_vol"]).copy()

        ticker_risk["risk_bucket"] = pd.qcut(
            ticker_risk["profile_vol"].rank(method="first"),
            q=5,
            labels=["vol_1", "vol_2", "vol_3", "vol_4", "vol_5"]
        )

        data = data.merge(ticker_risk[["ticker", "risk_bucket"]], on="ticker", how="left")
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

    def _effective_edge_q(self, df: pd.DataFrame, max_top_tickers: int | None = None) -> float:
        max_top_tickers = max_top_tickers or self.config.max_top_tickers
        n_tickers = max(1, df["ticker"].nunique())
        return min(self.config.edge_q, max_top_tickers / n_tickers)

    def add_cross_sectional_signals(self, df: pd.DataFrame, pred_col: str, group_cols: list | None = None, max_top_tickers: int | None = None,  allow_short: bool | None = None) -> pd.DataFrame:
        df = df.copy()
        df["pred_signal"] = 0

        if group_cols is None:
            group_cols = ["Date", "profile_group"]

        allow_short = self.config.allow_short if allow_short is None else allow_short
        edge_q = self._effective_edge_q(df, max_top_tickers)

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

    def _apply_signals(self, test_df: pd.DataFrame, pred: np.ndarray | None = None, max_top_tickers: int | None = None, allow_short: bool | None = None) -> pd.DataFrame:
        out = test_df.copy()

        if pred is not None:
            out["pred"] = pred

        return self.add_cross_sectional_signals(
            out,
            "pred",
            ["Date", "profile_group"],
            max_top_tickers=max_top_tickers,
            allow_short=allow_short,
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
            "trade_rate": float((df["pred_signal"] != 0).mean()),
            "long_rate": float((df["pred_signal"] == 1).mean()),
            "short_rate": float((df["pred_signal"] == -1).mean()),
            "hit_rate": float((trades["strategy_return"] > 0).mean()),
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


    def _daily_candidate_pool(self, day: pd.DataFrame, max_top_tickers: int, allow_short: bool) -> pd.DataFrame:
        signalled = self.add_cross_sectional_signals(
            day,
            "pred",
            ["Date", "profile_group"],
            max_top_tickers=max_top_tickers,
            allow_short=allow_short,
        )

        candidates = signalled[signalled["pred_signal"] != 0].copy()
        if candidates.empty:
            return candidates

        candidates["score"] = np.where(
            candidates["pred_signal"] == 1,
            candidates["pred"],
            -candidates["pred"],
        )

        return candidates.sort_values("score", ascending=False)

    def evaluate_rotating_portfolio(self, base_df: pd.DataFrame, max_top_tickers: int | None = None, allow_short: bool | None = None) -> dict[str, Any]:
        max_top_tickers = max_top_tickers or self.config.max_top_tickers
        allow_short = self.config.allow_short if allow_short is None else allow_short
        max_holding_days = self.config.horizon

        cost = self.config.cost_bps / 10_000
        max_positions = max_top_tickers

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
                max_top_tickers=max_top_tickers,
                allow_short=True,
            ).set_index("ticker", drop=False)

            candidates = self._daily_candidate_pool(
                day,
                max_top_tickers=max_top_tickers,
                allow_short=allow_short,
            )

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

    def evaluate_top_n_variants(self, base_df: pd.DataFrame, max_top_values: list[int], allow_short: bool | None = None) -> dict[str, Any]:
        results = {}

        for max_top in max_top_values:
            scored_df = self._apply_signals(
                base_df,
                pred=None,
                max_top_tickers=max_top,
                allow_short=allow_short,
            )

            row_metrics = self.evaluate_strategy(scored_df)
            portfolio_metrics = self.evaluate_rotating_portfolio(
                base_df,
                max_top_tickers=max_top,
                allow_short=allow_short,
            )

            results[f"top_{max_top}"] = {
                **row_metrics,
                "portfolio": portfolio_metrics,
            }

        return results



    @staticmethod
    def _get_lgbm_params(hyperparams: dict) -> dict:
        lgbm_params = dict(hyperparams.get("LGBM", {}).get("best_params", {}))

        defaults = {
            "n_estimators": 700,
            "learning_rate": 0.025,
            "max_depth": 5,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "regression",
        }

        if Settings.GPU["LGBM"]:
            defaults.update({"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})

        if Settings.Threaded:
            defaults.update({"num_threads": -1, "n_jobs": -1})

        defaults.update(lgbm_params)
        return defaults

    @staticmethod
    def _get_cat_params(hyperparams: dict) -> dict:
        cat_params = dict(hyperparams.get("CAT", {}).get("best_params", {}))

        defaults = {
            "iterations": 700,
            "learning_rate": 0.025,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "loss_function": "RMSE",
            "allow_writing_files": False,
            "task_type": "GPU" if Settings.GPU["CAT"] else "CPU",
        }

        defaults.update(cat_params)
        return defaults

    def _train_lightgbm(self, hyperparams: dict, max_top_values: list[int]) -> dict[str, Any]:
        params = self._get_lgbm_params(hyperparams)
        model = LGBMRegressor(random_state=self.seed, verbose=-1, **params)

        model.fit(self.X_train, self.y_train)

        pred = model.predict(self.X_test)

        base_df = self.test_df.copy()
        base_df["pred"] = pred

        scored_df = self._apply_signals(base_df)
        strategy_metrics = self.evaluate_strategy(scored_df)
        portfolio_metrics = self.evaluate_rotating_portfolio(base_df)
        variants = self.evaluate_top_n_variants(base_df, max_top_values)

        return {
            "type": "LGBM",
            "model": model,
            "test_df": scored_df,
            "base_df": base_df,
            "mae": float(mean_absolute_error(self.y_test, pred)),
            "rmse": float(mean_squared_error(self.y_test, pred) ** 0.5),
            **strategy_metrics,
            "portfolio": portfolio_metrics,
            "top_n_variants": variants,
        }

    def _train_catboost(self, hyperparams: dict, max_top_values: list[int]) -> dict[str, Any]:
        params = self._get_cat_params(hyperparams)
        model = CatBoostRegressor(random_seed=self.seed, verbose=False, **params)

        model.fit(self.X_train, self.y_train)

        pred = model.predict(self.X_test)

        base_df = self.test_df.copy()
        base_df["pred"] = pred

        scored_df = self._apply_signals(base_df)
        strategy_metrics = self.evaluate_strategy(scored_df)
        portfolio_metrics = self.evaluate_rotating_portfolio(base_df)
        variants = self.evaluate_top_n_variants(base_df, max_top_values)

        return {
            "type": "CAT",
            "model": model,
            "test_df": scored_df,
            "base_df": base_df,
            "mae": float(mean_absolute_error(self.y_test, pred)),
            "rmse": float(mean_squared_error(self.y_test, pred) ** 0.5),
            **strategy_metrics,
            "portfolio": portfolio_metrics,
            "top_n_variants": variants,
        }

    def _save_model_assets(self, interval, results: dict, baselines: dict):
        save_folder = Path(os.path.join(MODEL_DIR, f"Profile ({interval}) ({self.config.horizon})"))
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

    def run_training_pipeline(self, interval, force_train: bool = True) -> bool:
        def log_update(msg):
            if Settings.LOGGING:
                log(msg)

        if interval == "1d":
            self.config.horizon = 40
            self.config.max_top_tickers = 30
        elif interval == "1h":
            self.config.horizon = 30
            self.config.max_top_tickers = 10

        hyperparams = {}

        save_folder = os.path.join(MODEL_DIR, f"Profile ({interval})")
        if all_model_assets_exist(save_folder) and not force_train:
            log_update(f"Universe model already trained: {save_folder}")
            return True

        if os.path.exists(save_folder): shutil.rmtree(save_folder)

        t0 = time.perf_counter()

        log_update("Building universe dataframe...")
        data = self._build_universe_frame(interval)

        log_update("Preparing pooled features...")
        self._prepare_data(data)

        baselines = self.evaluate_baselines()
        log("Baselines:")
        log(json.dumps(baselines, indent=4))

        results = {}

        log_update("Training LightGBM...")
        results["LGBM"] = self._train_lightgbm(hyperparams, [5,10,20,30])
        flush_memory()
        log(json.dumps({k: v for k, v in results["LGBM"].items() if k not in {"model", "test_df", "base_df"}}, indent=4))

        log_update("Training CatBoost...")
        results["CAT"] = self._train_catboost(hyperparams, [5,10,20,30])
        flush_memory()
        log(json.dumps({k: v for k, v in results["CAT"].items() if k not in {"model", "test_df", "base_df"}}, indent=4))

        log_update("Saving assets...")
        self._save_model_assets(interval, results, baselines)

        best_model_type = max(results, key=lambda m: results[m].get("sharpe_like", -999))
        latest = self.rank_latest_predictions(results[best_model_type]["test_df"])

        log(f"Best model by sharpe_like: {best_model_type}")
        log("Latest ranked predictions:")
        log(latest.to_string(index=False))
        log(f"Total time: {time.perf_counter() - t0:.1f}s")

        return True


def all_model_assets_exist(model_path: str | os.PathLike) -> bool:
    root = Path(model_path)
    required_files = ["metadata.json", "features.joblib", "lgbm_model.txt", "cat_model.joblib"]

    return root.exists() and all((root / f).exists() and (root / f).stat().st_size > 0 for f in required_files)


if __name__ in "__main__":
    start = time.perf_counter()

    print("Training...")
    manager = TrainingManager()
    success = manager.run_tuning_pipeline("1d", force_train=True)

    print(time.perf_counter() - start)