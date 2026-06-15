
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

from scripts.config import DATA_DIR, MODEL_DIR, ROOT_DIR, LOG_DIR
from scripts.data_management import load_data
import scripts.indicators  # noqa: F401

warnings.filterwarnings("ignore")

########################################################################################################################

class Settings:
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
    if torch.cuda.is_available(): torch.cuda.empty_cache()

now = datetime.now().strftime('%Y-%m-%d %H:%M')
def log(string: str, prints: bool = True):
    if prints: print(string)
    with open(os.path.join(LOG_DIR, f"log [{now}].txt"), "a") as f: f.write(f"{string}\n")

########################################################################################################################

class TrainingManager:
    def __init__(self):
        self.config = UniverseConfig()
        self.seed = 69
        self.__test_size = 0.2

        self.feature_cols = None
        self.split_date = None

        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    ######################     Creating universe dataframe     ######################

    def _add_liquidity_columns(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        df = df.copy()

        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        window = 120 if interval == "1h" else 20

        df["dollar_volume"] = df[price_col].astype(float) * df["Volume"].astype(float)
        df["rolling_dollar_volume"] = (
            df["dollar_volume"]
            .rolling(window=window, min_periods=max(5, window // 4))
            .median().shift(1)
        )
        df["abs_bar_return"] = df[price_col].pct_change().abs()
        df["liquidity_dollar_volume_log"] = np.log1p(
            df["rolling_dollar_volume"].clip(lower=0)
        )
        return df

    def _apply_liquidity_filters(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        min_dollar_volume = self.config.min_rolling_dollar_volume_1h if interval == "1h" \
            else self.config.min_rolling_dollar_volume_1d
        min_rows = self.config.min_rows_after_filter_1h if interval == "1h" \
            else self.config.min_rows_after_filter_1d

        df = df.copy()
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        mask = (
                df[price_col].between(self.config.min_price, self.config.max_price)
                & (df["rolling_dollar_volume"] >= min_dollar_volume)
                & (df["abs_bar_return"] <= self.config.max_abs_bar_return)
        )
        df = df[mask].copy()

        return pd.DataFrame() if len(df) < min_rows else df

    def _add_rolling_risk_columns(self, df: pd.DataFrame, benchmark_close: pd.Series, interval: str) -> pd.DataFrame:
        df = df.copy()
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

        vol_window = self.config.rolling_vol_window_1h if interval == "1h" else self.config.rolling_vol_window_1d
        beta_window = self.config.rolling_beta_window_1h if interval == "1h" else self.config.rolling_beta_window_1d
        annualiser = np.sqrt(252 * (6.5 if interval == "1h" else 1))

        stock_ret = df[price_col].astype(float).pct_change()

        benchmark_close = benchmark_close.reindex(df.index).ffill()
        benchmark_ret = benchmark_close.pct_change()

        min_vol_periods = max(10, vol_window // 4)
        min_beta_periods = max(20, beta_window // 4)

        df["rolling_vol"] = stock_ret.rolling(window=vol_window, min_periods=min_vol_periods).std().shift(1) * annualiser
        rolling_cov = stock_ret.rolling(window=beta_window, min_periods=min_beta_periods).cov(benchmark_ret).shift(1)
        rolling_market_var = benchmark_ret.rolling(window=beta_window, min_periods=min_beta_periods).var().shift(1)

        df["rolling_beta"] = rolling_cov / (rolling_market_var + 1e-12)
        df["rolling_beta"] = df["rolling_beta"].replace([np.inf, -np.inf], np.nan)
        df["rolling_beta"] = df["rolling_beta"].clip(-3.0, 3.0)

        df["rolling_market_corr"] = (stock_ret.rolling(window=beta_window, min_periods=min_beta_periods)
                                     .corr(benchmark_ret).shift(1))
        df["rolling_market_vol"] = (benchmark_ret.rolling(window=vol_window, min_periods=min_vol_periods)
                                    .std().shift(1) * annualiser)

        return df

    def add_forward_excess_target(self, df: pd.DataFrame, benchmark_close: pd.Series, beta: float | pd.Series = 1.0, drop_unlabelled: bool = True) -> pd.DataFrame:
        df = df.copy()
        close = df["Adj Close"]
        benchmark_close = benchmark_close.reindex(df.index).ffill()

        df["future_return"] = close.shift(-self.config.horizon) / close - 1.0
        df["benchmark_future_return"] = (benchmark_close.shift(-self.config.horizon) / benchmark_close - 1.0)

        if isinstance(beta, pd.Series):
            beta_used = beta.reindex(df.index).ffill()
            beta_used = beta_used.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        else:
            beta_used = 1.0 if not np.isfinite(beta) else float(beta)

        df["target_beta_used"] = beta_used
        df["target_excess_return"] = (df["future_return"] - beta_used * df["benchmark_future_return"])

        return df.dropna(subset=["target_excess_return"]) if drop_unlabelled else df

    def build_universe_frame(self, interval: str, drop_unlabelled: bool = True, cutoff_date: pd.Timestamp | None = None) -> pd.DataFrame:
        with open(os.path.join(DATA_DIR, "ticker_attr.json"), "r") as f: ticker_map = json.load(f)
        ticker_list = sorted(set(ticker_map.keys()))

        benchmark_raw = pd.read_parquet(os.path.join(DATA_DIR, f"SPY_{interval}.parquet"))
        benchmark_raw.index.name = "Date"
        benchmark_raw.index = pd.to_datetime(benchmark_raw.index, utc=True).tz_localize(None)
        benchmark_raw = benchmark_raw[~benchmark_raw.index.duplicated(keep="first")].sort_index()
        benchmark_close = benchmark_raw["Adj Close"]

        frames = []
        for ticker in tqdm(ticker_list):
            try:
                raw = load_data(ticker, interval)
                if raw is None or raw.empty: continue

                if cutoff_date is not None: raw = raw.loc[:pd.Timestamp(cutoff_date)].copy()

                adv = ticker_map.get(ticker, {}).get("adv", np.nan)
                if self.config.min_profile_adv > max(0, float(adv)) and np.isfinite(adv): continue

                df = raw.ind.add_indicators(ticker, interval, add_targets=False)

                df = self._add_liquidity_columns(df, interval)
                df = self._apply_liquidity_filters(df, interval)

                df = self._add_rolling_risk_columns(df, benchmark_close, interval)
                df = self.add_forward_excess_target(df, benchmark_close, df["rolling_beta"], drop_unlabelled)

                if df.empty: continue

                df["ticker"] = ticker
                df["Date"] = df.index
                df = df.reset_index(drop=True)
                frames.append(df)

            except Exception: pass # noqa

        if not frames: raise ValueError("No usable universe data")

        data = pd.concat(frames, axis=0, ignore_index=True)
        data = data.sort_values(["Date", "ticker"])
        data = data.replace([np.inf, -np.inf], np.nan)

        if drop_unlabelled: data = data.dropna()
        else:
            target_cols = {"future_return", "benchmark_future_return", "target_excess_return"}
            non_target_cols = [c for c in data.columns if c not in target_cols]
            data = data.dropna(subset=non_target_cols)

        return data

    def _prepare_data(self, data: pd.DataFrame) -> None:
        data = data.copy()
        rank_pct = data.groupby("Date")["rolling_vol"].rank(method="first", pct=True)

        data["profile_group"] = pd.cut(
            rank_pct,
            bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["vol_1", "vol_2", "vol_3", "vol_4", "vol_5"],
            include_lowest=True,
        ).astype(str).fillna("unknown")

        data = pd.get_dummies(data, columns=["profile_group"], prefix=["profile_group"], dtype=int)

        drop_cols = {
            "Open", "High", "Low", "Close", "Adj Close", "Volume",
            "Adj Open", "Adj High", "Adj Low",
            "MA_200", "return",
            "ticker", "Date", "profile_group",
            "future_return", "benchmark_future_return", "target_excess_return",
            "target_profit", "tbm_return", "barrier_strength",
            "time_to_gain", "time_to_loss", "tp_return", "sl_return",
            "dollar_volume", "rolling_dollar_volume", "abs_bar_return", "target_beta_used"
        }

        self.feature_cols = [c for c in data.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])]

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

    ######################     Helpers for Training Models     ######################

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
            if year_df["Date"].nunique() < 60: continue

            results[str(year)] = self.evaluate_rotating_portfolio(
                year_df.drop(columns=["year"])
            )

        return results

    def evaluate_strategy(self, df: pd.DataFrame) -> dict[str, Any]:
        trades = df[df["pred_signal"] != 0].copy()

        if trades.empty:
            return {
                "trade_rate": 0.0, "long_rate": 0.0, "short_rate": 0.0, "hit_rate": 0.0,
                "mean_return": 0.0, "median_return": 0.0, "sharpe_like": 0.0,
                "rank_ic_mean": 0.0, "rank_ic_ir": 0.0, "n_trades": 0,
            }

        trades["strategy_return"] = trades["pred_signal"] * trades["target_excess_return"] - (
                    self.config.cost_bps / 10_000)
        mean_return = trades["strategy_return"].mean()
        std_return = trades["strategy_return"].std() + 1e-9

        daily_ics = []
        for _, group in df.groupby("Date"):
            if len(group) < 5: continue
            ic = group["pred"].corr(group["target_excess_return"], method="spearman")
            if np.isfinite(ic): daily_ics.append(ic)

        rank_ic_mean = float(np.mean(daily_ics)) if daily_ics else 0.0
        rank_ic_std = float(np.std(daily_ics)) + 1e-9 if daily_ics else 1.0

        def side_stats(side_df):
            if side_df.empty: return {"hit_rate": 0.0, "mean_return": 0.0, "median_return": 0.0, "n": 0}
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
            "long_stats": side_stats(trades[trades["pred_signal"] == 1]),
            "short_stats": side_stats(trades[trades["pred_signal"] == -1]),
        }

    def evaluate_rotating_portfolio(self, base_df: pd.DataFrame) -> dict:
        df = base_df.copy()
        df = df.sort_values(["ticker", "Date"])
        df["next_return"] = df.groupby("ticker")["Adj Close"].shift(-1) / df["Adj Close"] - 1.0
        df = df.sort_values(["Date", "ticker"])

        dates = list(sorted(pd.to_datetime(df["Date"]).unique()))
        by_date = {date: group.copy() for date, group in df.groupby("Date", sort=True)}

        holdings = {}
        daily_returns = []
        daily_turnover = []
        daily_holding_count = []
        for date in dates[:-1]:
            day = by_date[date].copy()
            day = day.dropna(subset=["pred", "next_return", "Adj Close"])
            if day.empty: continue

            exit_signal_day = (self.add_cross_sectional_signals(day, "pred", allow_short=True)
                               .set_index("ticker", drop=False))

            candidates = self._daily_candidate_pool(day)
            turnover = 0
            for ticker in list(holdings.keys()):
                if ticker not in exit_signal_day.index:
                    holdings.pop(ticker)
                    turnover += 1
                    continue

                row = exit_signal_day.loc[ticker]
                side = holdings[ticker]["side"]

                opposite_signal = row["pred_signal"] == -side
                weak_long = side == 1 and row["pred"] <= self.config.exit_pred_threshold
                weak_short = side == -1 and row["pred"] >= -self.config.exit_pred_threshold
                too_old = holdings[ticker]["age"] >= self.config.horizon

                if opposite_signal or weak_long or weak_short or too_old:
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

                if len(holdings) < self.config.max_top_tickers:
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
                if ticker not in day_indexed.index: continue

                next_return = day_indexed.loc[ticker, "next_return"]
                if not np.isfinite(next_return): continue

                position_returns.append(info["side"] * float(next_return))

            gross_return = float(np.mean(position_returns)) if position_returns else 0.0
            turnover_cost = self.config.cost_bps / 10_000 * turnover / max(1, self.config.max_top_tickers)
            portfolio_return = gross_return - turnover_cost

            daily_returns.append(portfolio_return)
            daily_turnover.append(turnover)
            daily_holding_count.append(len(holdings))

            for info in holdings.values(): info["age"] += 1

        if not daily_returns:
            return {
                "portfolio_days": 0, "portfolio_total_return": 0.0,
                "portfolio_cagr_like": 0.0, "portfolio_sharpe_like": 0.0,
                "portfolio_max_drawdown": 0.0, "portfolio_win_rate": 0.0,
                "portfolio_mean_daily_return": 0.0, "portfolio_median_daily_return": 0.0,
                "portfolio_avg_holdings": 0.0, "portfolio_avg_turnover": 0.0,
            }

        returns = pd.Series(daily_returns, dtype=float)
        equity = (1.0 + returns).cumprod()

        years = len(returns) / 252
        return {
            "portfolio_days": int(len(returns)),
            "portfolio_total_return": float(equity.iloc[-1] - 1.0),
            "portfolio_cagr_like": float(equity.iloc[-1] ** (1 / years) - 1.0 if years > 0 else 0.0),
            "portfolio_sharpe_like": float(returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)),
            "portfolio_max_drawdown": float((equity / equity.cummax() - 1.0).min()),
            "portfolio_win_rate": float((returns > 0).mean()),
            "portfolio_mean_daily_return": float(returns.mean()),
            "portfolio_median_daily_return": float(returns.median()),
            "portfolio_avg_holdings": float(np.mean(daily_holding_count)),
            "portfolio_avg_turnover": float(np.mean(daily_turnover)),
        }

    def analyse_drawdown(self, base_df: pd.DataFrame) -> dict[str, Any]:
        df = base_df.copy()
        df = df.sort_values(["ticker", "Date"])
        df["next_return"] = df.groupby("ticker")["Adj Close"].shift(-1) / df["Adj Close"] - 1.0
        df = df.sort_values(["Date", "ticker"])

        dates = list(sorted(pd.to_datetime(df["Date"]).unique()))
        by_date = {date: group.copy() for date, group in df.groupby("Date", sort=True)}

        holdings = {}
        daily_returns = []
        daily_dates = []
        for date in dates[:-1]:
            day = by_date[date].copy()
            day = day.dropna(subset=["pred", "next_return", "Adj Close"])

            if day.empty: continue

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
                too_old = age >= self.config.horizon

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

                if len(holdings) < self.config.max_top_tickers:
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
            turnover_cost = (self.config.cost_bps / 10_000) * turnover / max(1, self.config.max_top_tickers)
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

    @staticmethod
    def benchmark_drawdown(interval: str, start_date, end_date) -> dict[str, Any]:
        spy = pd.read_parquet(os.path.join(DATA_DIR, f"SPY_{interval}.parquet"))
        spy.index = pd.to_datetime(spy.index, utc=True).tz_localize(None)
        spy = spy.loc[(spy.index >= start_date) & (spy.index <= end_date)].copy()

        if spy.empty or len(spy) < 2: return {"spy_total_return": 0.0, "spy_max_drawdown": 0.0}

        equity = (spy["Adj Close"].pct_change().dropna() + 1.0).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        return {"spy_total_return": float(equity.iloc[-1] - 1.0), "spy_max_drawdown": float(drawdown.min())}

    ######################           Misc functions           ######################

    def add_cross_sectional_signals(self, df: pd.DataFrame, pred_col: str, group_cols: list | None = None, allow_short: bool | None = None) -> pd.DataFrame:
        df = df.copy()
        df["pred_signal"] = 0

        if group_cols is None: group_cols = ["Date", "profile_group"]

        allow_short = self.config.allow_short if allow_short is None else allow_short
        n_tickers = max(1, df["ticker"].nunique())
        edge_q = min(self.config.edge_q, self.config.max_top_tickers / n_tickers)

        for _, group in df.groupby(group_cols):
            if len(group) < 5: continue

            upper = group[pred_col].quantile(1 - edge_q)
            lower = group[pred_col].quantile(edge_q)

            long_mask = (group[pred_col] >= upper) & (group[pred_col].abs() >= self.config.min_abs_pred)
            df.loc[group.index[long_mask], "pred_signal"] = 1

            if allow_short:
                short_mask = (group[pred_col] <= lower) & (group[pred_col].abs() >= self.config.min_abs_pred)
                df.loc[group.index[short_mask], "pred_signal"] = -1

        return df

    def _daily_candidate_pool(self, day: pd.DataFrame) -> pd.DataFrame:
        signalled = self.add_cross_sectional_signals(day, "pred", ["Date", "profile_group"])

        candidates = signalled[signalled["pred_signal"] != 0].copy()
        if candidates.empty: return candidates

        candidates["score"] = np.where(candidates["pred_signal"] == 1, candidates["pred"], -candidates["pred"])
        return candidates.sort_values("score", ascending=False)

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

    ######################           Training logic           ######################

    def _train_lightgbm(self, interval: str) -> dict:
        with open(os.path.join(ROOT_DIR, "results", f"lgbm_params_{interval}.json"), "r") as f:
            params = json.load(f)

        if Settings.GPU["LGBM"]: params.update({"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
        if Settings.Threaded: params.update({"num_threads": -1, "n_jobs": -1})
        params.update({"objective": "regression"})

        model = LGBMRegressor(random_state=self.seed, verbose=-1, **params)
        model.fit(self.X_train, self.y_train)

        pred = model.predict(self.X_test)

        base_df = self.test_df.copy()
        base_df["pred"] = pred
        scored_df = self.add_cross_sectional_signals(base_df, "pred")

        cost_sensitivity = self.evaluate_cost_sensitivity(base_df)
        yearly_portfolio = self.evaluate_portfolio_by_year(base_df)
        strategy_metrics = self.evaluate_strategy(scored_df)
        portfolio_metrics = self.evaluate_rotating_portfolio(base_df)

        drawdown_info = self.analyse_drawdown(base_df)
        benchmark_dd = self.benchmark_drawdown(interval, drawdown_info["peak_date"], drawdown_info["trough_date"])

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

        if Settings.Threaded: params.update({"thread_count": -1})
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
        scored_df = self.add_cross_sectional_signals(base_df, "pred")

        cost_sensitivity = self.evaluate_cost_sensitivity(base_df)
        yearly_portfolio = self.evaluate_portfolio_by_year(base_df)
        strategy_metrics = self.evaluate_strategy(scored_df)
        portfolio_metrics = self.evaluate_rotating_portfolio(base_df)

        drawdown_info = self.analyse_drawdown(base_df)
        benchmark_dd = self.benchmark_drawdown(interval, drawdown_info["peak_date"], drawdown_info["trough_date"])

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

    def run_training_pipeline(self, interval: str, force_train: bool = True) -> bool:
        if interval == "1d":
            self.config.horizon = 40
            self.config.max_top_tickers = 30
        elif interval == "1h":
            self.config.horizon = 30
            self.config.max_top_tickers = 10

        save_folder = os.path.join(MODEL_DIR, f"{interval} Model [{now}]")
        if all_model_assets_exist(save_folder) and not force_train:
            log(f"Universe model already trained: {save_folder}")
            return True

        if os.path.exists(save_folder): shutil.rmtree(save_folder)

        log("Building universe dataframe...")
        data = self.build_universe_frame(interval)

        log("Preparing pooled features...")
        self._prepare_data(data)

        baselines = self.evaluate_baselines()
        log("Baselines:")
        log(json.dumps(baselines, indent=4))

        results = {}

        log("Training LightGBM...")
        results["LGBM"] = self._train_lightgbm(interval)
        flush_memory()
        log(json.dumps({k: v for k, v in results["LGBM"].items() if k not in {"model", "test_df", "base_df"}}, indent=4))

        log("Training CatBoost...")
        results["CAT"] = self._train_catboost(interval)
        flush_memory()
        log(json.dumps({k: v for k, v in results["CAT"].items() if k not in {"model", "test_df", "base_df"}}, indent=4))

        best_model_type = max(results, key=lambda m: final_model_score(results[m]))
        log(f"Best model: {best_model_type}")

        log("Saving assets...")
        save_training_run(manager=self, interval=interval, results=results, baselines=baselines, best_model_type=best_model_type)

        latest = latest_prediction_export(results[best_model_type]["test_df"], top_n=self.config.max_top_tickers, include_actuals=True)

        log("Latest ranked predictions:")
        log(latest.to_string(index=False))

        return True

########################################################################################################################

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if pd.isna(value) if isinstance(value, (float, np.floating)) else False:
        return None

    return value

def final_model_score(result: dict) -> float:
    p = result["portfolio"]

    return (
            p["portfolio_cagr_like"]
            + 0.05 * p["portfolio_sharpe_like"]
            - 0.50 * abs(p["portfolio_max_drawdown"])
            - 0.002 * p["portfolio_avg_turnover"]
    )

def strip_heavy_result_items(model_results: dict) -> dict:
    banned = {"model", "test_df", "base_df"}

    return {
        k: json_safe(v)
        for k, v in model_results.items()
        if k not in banned
    }

def latest_prediction_export(scored_df: pd.DataFrame, top_n: int = 30, include_actuals: bool = False) -> pd.DataFrame:
    latest_date = scored_df["Date"].max()
    latest = scored_df[scored_df["Date"] == latest_date].copy()
    latest = latest.sort_values("pred", ascending=False)

    cols = [
        "Date",
        "ticker",
        "profile_group",
        "pred",
        "pred_signal",
    ]

    if include_actuals:
        cols += [
            "target_excess_return",
            "future_return",
            "benchmark_future_return",
        ]

    cols = [c for c in cols if c in latest.columns]

    top = latest.head(min(top_n, len(latest)))
    bottom = latest.tail(min(top_n, len(latest)))

    if len(latest) <= top_n * 2:
        return latest[cols]

    return pd.concat([top, bottom], axis=0)[cols]

def save_training_run(manager: TrainingManager, interval: str, results: dict, baselines: dict, best_model_type: str | None = None) -> Path:
    global now
    save_folder = Path(MODEL_DIR) / f"{interval} Model [{now}]"
    save_folder.mkdir(parents=True, exist_ok=True)

    models_folder = save_folder / "models"
    predictions_folder = save_folder / "predictions"
    reports_folder = save_folder / "reports"

    models_folder.mkdir(parents=True, exist_ok=True)
    predictions_folder.mkdir(parents=True, exist_ok=True)
    reports_folder.mkdir(parents=True, exist_ok=True)

    if best_model_type is None:
        best_model_type = max(results, key=lambda m: final_model_score(results[m]))

    metadata = {
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "interval": interval,
        "best_model_type": best_model_type,
        "best_model_score": final_model_score(results[best_model_type]),
        "config": asdict(manager.config),
        "split_date": pd.Timestamp(manager.split_date).strftime("%Y-%m-%d"),
        "feature_count": len(manager.feature_cols),
        "feature_cols": list(manager.feature_cols),
        "baselines": baselines,
        "model_results": {},
    }

    for model_type, model_results in results.items():
        model_type_lower = model_type.lower()

        metadata["model_results"][model_type] = strip_heavy_result_items(model_results)

        model = model_results["model"]

        if model_type == "LGBM":
            model.booster_.save_model(str(models_folder / "lgbm_model.txt"))
            joblib.dump(model, models_folder / "lgbm_model.joblib")

        elif model_type == "CAT":
            joblib.dump(model, models_folder / "cat_model.joblib")

        else:
            joblib.dump(model, models_folder / f"{model_type_lower}_model.joblib")

        if "test_df" in model_results:
            test_df = model_results["test_df"]
            test_df.to_parquet(
                predictions_folder / f"{model_type_lower}_test_predictions.parquet",
                index=False,
            )

            latest_live = latest_prediction_export(
                test_df,
                top_n=manager.config.max_top_tickers,
                include_actuals=False,
            )
            latest_debug = latest_prediction_export(
                test_df,
                top_n=manager.config.max_top_tickers,
                include_actuals=True,
            )

            latest_live.to_csv(
                predictions_folder / f"{model_type_lower}_latest_live_predictions.csv",
                index=False,
            )
            latest_debug.to_csv(
                predictions_folder / f"{model_type_lower}_latest_debug_predictions.csv",
                index=False,
            )

        if "base_df" in model_results:
            model_results["base_df"].to_parquet(
                predictions_folder / f"{model_type_lower}_base_predictions.parquet",
                index=False,
            )

        summary_path = reports_folder / f"{model_type_lower}_summary.json"
        summary_path.write_text(
            json.dumps(metadata["model_results"][model_type], indent=4),
            encoding="utf-8",
        )

    joblib.dump(manager.feature_cols, save_folder / "features.joblib")

    metadata_path = save_folder / "metadata.json"
    metadata_path.write_text(
        json.dumps(json_safe(metadata), indent=4),
        encoding="utf-8",
    )

    latest_best = latest_prediction_export(
        results[best_model_type]["test_df"],
        top_n=manager.config.max_top_tickers,
        include_actuals=False,
    )
    latest_best.to_csv(save_folder / "latest_predictions.csv", index=False)

    log(f"Saved training run to: {save_folder}")

    return save_folder

def all_model_assets_exist(model_path: str | os.PathLike) -> bool:
    root = Path(model_path)

    required_files = [
        "metadata.json",
        "features.joblib",
        "models/lgbm_model.txt",
        "models/lgbm_model.joblib",
        "models/cat_model.joblib",
        "latest_predictions.csv",
    ]

    return root.exists() and all(
        (root / f).exists() and (root / f).stat().st_size > 0
        for f in required_files
    )

########################################################################################################################

def analyse_position_attribution(position_df: pd.DataFrame, top_n: int = 25) -> dict[str, pd.DataFrame | dict]:
    if position_df.empty:
        return {
            "ticker_summary": pd.DataFrame(),
            "best_position_days": pd.DataFrame(),
            "worst_position_days": pd.DataFrame(),
            "concentration": {},
        }

    df = position_df.copy()
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["exec_entry_date"] = pd.to_datetime(df["exec_entry_date"])
    df["exec_exit_date"] = pd.to_datetime(df["exec_exit_date"])

    ticker_summary = (
        df.groupby("ticker")
        .agg(
            days_held=("ticker", "size"),
            total_pnl=("contribution_pnl", "sum"),
            total_contribution_return=("contribution_return", "sum"),
            mean_position_return=("position_return", "mean"),
            median_position_return=("position_return", "median"),
            hit_rate=("position_return", lambda s: float((s > 0).mean())),
            best_day_return=("position_return", "max"),
            worst_day_return=("position_return", "min"),
        )
        .reset_index()
        .sort_values("total_pnl", ascending=False)
    )

    best_position_days = (
        df.sort_values("contribution_pnl", ascending=False)
        .head(top_n)
        .copy()
    )

    worst_position_days = (
        df.sort_values("contribution_pnl", ascending=True)
        .head(top_n)
        .copy()
    )

    total_positive_pnl = ticker_summary.loc[
        ticker_summary["total_pnl"] > 0,
        "total_pnl",
    ].sum()

    top_1_pnl = ticker_summary["total_pnl"].head(1).sum()
    top_3_pnl = ticker_summary["total_pnl"].head(3).sum()
    top_5_pnl = ticker_summary["total_pnl"].head(5).sum()

    concentration = {
        "total_positive_pnl": float(total_positive_pnl),
        "top_1_pnl": float(top_1_pnl),
        "top_3_pnl": float(top_3_pnl),
        "top_5_pnl": float(top_5_pnl),
        "top_1_share_of_positive_pnl": float(top_1_pnl / total_positive_pnl) if total_positive_pnl > 0 else 0.0,
        "top_3_share_of_positive_pnl": float(top_3_pnl / total_positive_pnl) if total_positive_pnl > 0 else 0.0,
        "top_5_share_of_positive_pnl": float(top_5_pnl / total_positive_pnl) if total_positive_pnl > 0 else 0.0,
    }

    return {
        "ticker_summary": ticker_summary,
        "best_position_days": best_position_days,
        "worst_position_days": worst_position_days,
        "concentration": concentration,
    }

def add_next_open_execution_returns(df: pd.DataFrame, spread_bps: float = 1.0) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    if "Adj Open" in df.columns:
        price_col = "Adj Open"
    elif {"Open", "Close", "Adj Close"}.issubset(df.columns):
        # Create an adjusted open if yfinance/cache does not already have one.
        adj_factor = df["Adj Close"] / df["Close"]
        df["exec_open"] = df["Open"] * adj_factor
        price_col = "exec_open"
    elif "Open" in df.columns:
        price_col = "Open"
    else:
        raise ValueError("Need Adj Open, or Open + Close + Adj Close, to create execution prices.")

    df = df.sort_values(["ticker", "Date"])

    grouped = df.groupby("ticker", group_keys=False)

    df["exec_entry_date"] = grouped["Date"].shift(-1)
    df["exec_exit_date"] = grouped["Date"].shift(-2)

    df["exec_entry_price_mid"] = grouped[price_col].shift(-1)
    df["exec_exit_price_mid"] = grouped[price_col].shift(-2)

    half_spread = spread_bps / 20_000  # bps -> half-spread decimal

    df["long_entry_price"] = df["exec_entry_price_mid"] * (1.0 + half_spread)
    df["long_exit_price"] = df["exec_exit_price_mid"] * (1.0 - half_spread)

    df["short_entry_price"] = df["exec_entry_price_mid"] * (1.0 - half_spread)
    df["short_exit_price"] = df["exec_exit_price_mid"] * (1.0 + half_spread)

    df["long_exec_return"] = (
            df["long_exit_price"] / df["long_entry_price"] - 1.0
    )

    df["short_exec_return"] = (
            df["short_entry_price"] / df["short_exit_price"] - 1.0
    )

    df["exec_return"] = df["long_exec_return"]

    return df.sort_values(["Date", "ticker"])

def prepare_manager_with_cutoff(manager: TrainingManager, train_data: pd.DataFrame, full_data: pd.DataFrame, cutoff_date) -> pd.DataFrame:
    cutoff_date = pd.Timestamp(cutoff_date)

    train_data = train_data.copy()
    full_data = full_data.copy()

    train_data["Date"] = pd.to_datetime(train_data["Date"])
    full_data["Date"] = pd.to_datetime(full_data["Date"])

    walk_data = full_data[full_data["Date"] > cutoff_date].copy()

    if train_data.empty: raise ValueError("No training rows before cutoff date.")
    if walk_data.empty: raise ValueError("No walk-forward rows after cutoff date.")

    data = pd.concat([train_data, walk_data], axis=0, ignore_index=True)
    rank_pct = data.groupby("Date")["rolling_vol"].rank(method="first", pct=True)

    data["profile_group"] = pd.cut(
        rank_pct,
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["vol_1", "vol_2", "vol_3", "vol_4", "vol_5"],
        include_lowest=True,
    ).astype(str).fillna("unknown")

    data = pd.get_dummies(data, columns=["profile_group"], prefix=["profile_group"], dtype=int)

    drop_cols = {
        "Open", "High", "Low", "Close", "Adj Close", "Volume",
        "Adj Open", "Adj High", "Adj Low",
        "MA_200", "return",
        "ticker", "Date", "profile_group",
        "future_return", "benchmark_future_return", "target_excess_return",
        "target_profit", "tbm_return", "barrier_strength",
        "time_to_gain", "time_to_loss", "tp_return", "sl_return",
        "dollar_volume", "rolling_dollar_volume", "abs_bar_return", "target_beta_used"
    }

    manager.feature_cols = [c for c in data.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])]

    train_df = data[(data["Date"] <= cutoff_date) & data["target_excess_return"].notna()].copy()
    walk_df = data[data["Date"] > cutoff_date].copy()

    if train_df.empty: raise ValueError("No labelled training rows before cutoff date.")
    if walk_df.empty: raise ValueError("No walk-forward rows after cutoff date.")

    manager.split_date = cutoff_date
    manager.train_df = train_df
    manager.test_df = walk_df

    manager.X_train = train_df[manager.feature_cols].to_numpy(dtype=np.float32)
    manager.y_train = train_df["target_excess_return"].to_numpy(dtype=np.float32)

    log(f"Walk-forward cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
    log(f"Train rows: {len(train_df):,}")
    log(f"Walk-forward rows: {len(walk_df):,}")
    log(f"Features: {len(manager.feature_cols)}")

    return walk_df

def fit_walk_forward_model(manager, interval: str, model_type: str):
    model_type = model_type.upper()

    if model_type == "CAT":
        params_path = Path(ROOT_DIR) / "results" / f"cat_params_{interval}.json"
        params = json.loads(params_path.read_text()) if params_path.exists() else {}

        if Settings.Threaded:
            params.update({"thread_count": -1})

        params.update({
            "loss_function": "RMSE",
            "allow_writing_files": False,
            "task_type": "GPU" if Settings.GPU["CAT"] else "CPU",
        })

        model = CatBoostRegressor(
            random_seed=manager.seed,
            verbose=False,
            **params,
        )

    elif model_type == "LGBM":
        params_path = Path(ROOT_DIR) / "results" / f"lgbm_params_{interval}.json"
        params = json.loads(params_path.read_text()) if params_path.exists() else {}

        if Settings.GPU["LGBM"]:
            params.update({
                "device_type": "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id": 0,
            })

        if Settings.Threaded:
            params.update({"num_threads": -1, "n_jobs": -1})

        params.update({"objective": "regression"})

        model = LGBMRegressor(
            random_state=manager.seed,
            verbose=-1,
            **params,
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(manager.X_train, manager.y_train)
    return model

def save_walk_forward_results(interval: str, model_type: str, months_back: int, summary: dict, daily_df: pd.DataFrame, trades_df: pd.DataFrame, position_df: pd.DataFrame | None = None, attribution: dict | None = None) -> Path:
    folder = (
            Path(MODEL_DIR)
            / f"{interval} Model [{now}]"
            / "walk_forward"
            / f"{model_type.upper()}_{months_back}m"
    )

    folder.mkdir(parents=True, exist_ok=True)

    daily_df.to_csv(folder / "daily_equity.csv", index=False)
    trades_df.to_csv(folder / "trades.csv", index=False)

    if position_df is not None and not position_df.empty:
        position_df.to_csv(folder / "position_returns.csv", index=False)

    if attribution is not None:
        ticker_summary = attribution.get("ticker_summary")
        best_position_days = attribution.get("best_position_days")
        worst_position_days = attribution.get("worst_position_days")
        concentration = attribution.get("concentration", {})

        if ticker_summary is not None and not ticker_summary.empty:
            ticker_summary.to_csv(folder / "ticker_attribution.csv", index=False)

        if best_position_days is not None and not best_position_days.empty:
            best_position_days.to_csv(folder / "best_position_days.csv", index=False)

        if worst_position_days is not None and not worst_position_days.empty:
            worst_position_days.to_csv(folder / "worst_position_days.csv", index=False)

        (folder / "attribution_summary.json").write_text(
            json.dumps(json_safe(concentration), indent=4),
            encoding="utf-8",
        )

    (folder / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=4),
        encoding="utf-8",
    )

    log(f"Saved walk-forward results to: {folder}")

    return folder

def add_consensus_ensemble_predictions(df: pd.DataFrame, top_n_per_model: int = 50, cat_weight: float = 0.6, lgbm_weight: float = 0.4) -> pd.DataFrame:
    df = df.copy()

    required_cols = {"Date", "ticker", "pred_cat", "pred_lgbm"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns for ensemble: {missing}")

    df["cat_rank_pct"] = df.groupby("Date")["pred_cat"].rank(
        method="first",
        pct=True,
    )

    df["lgbm_rank_pct"] = df.groupby("Date")["pred_lgbm"].rank(
        method="first",
        pct=True,
    )

    df["pred"] = 0.0
    df["ensemble_agreement"] = 0
    df["ensemble_score"] = 0.0

    for date, group in df.groupby("Date"):
        cat_top = set(
            group.nlargest(
                min(top_n_per_model, len(group)),
                "pred_cat",
            )["ticker"]
        )

        lgbm_top = set(
            group.nlargest(
                min(top_n_per_model, len(group)),
                "pred_lgbm",
            )["ticker"]
        )

        consensus = cat_top & lgbm_top

        if not consensus:
            continue

        idx = group[group["ticker"].isin(consensus)].index

        score = (
                cat_weight * df.loc[idx, "cat_rank_pct"]
                + lgbm_weight * df.loc[idx, "lgbm_rank_pct"]
        )

        df.loc[idx, "ensemble_agreement"] = 1
        df.loc[idx, "ensemble_score"] = score
        df.loc[idx, "pred"] = score

    # Force the signal system to rank all consensus tickers against each other
    # rather than splitting them by volatility bucket.
    df["profile_group"] = "ensemble"

    return df

def run_walk_forward_backtest(interval: str, months_back: int = 3, initial_capital: float = 1000.0) -> dict[str, Any]:
    manager = TrainingManager()

    if interval == "1d":
        manager.config.horizon = 40
        manager.config.max_top_tickers = 30
    elif interval == "1h":
        manager.config.horizon = 30
        manager.config.max_top_tickers = 10

    model_type = "ENSEMBLE"

    log("=" * 50)
    log(f"WALK-FORWARD BACKTEST: {model_type} {interval}")
    log("=" * 50)

    spy_df = pd.read_parquet(os.path.join(DATA_DIR, f"SPY_{interval}.parquet"))
    spy_df.index.name = "Date"
    spy_df.index = pd.to_datetime(spy_df.index, utc=True).tz_localize(None)

    latest_date = pd.Timestamp(spy_df.index.max())
    cutoff_date = latest_date - pd.DateOffset(months=months_back)

    log(f"Latest available date: {latest_date.strftime('%Y-%m-%d')}")
    log(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")

    log("Building training universe up to cutoff...")
    train_data = manager.build_universe_frame(interval, drop_unlabelled=True, cutoff_date=cutoff_date)

    log("Building full universe for walk-forward predictions...")
    full_data = manager.build_universe_frame(interval, drop_unlabelled=False, cutoff_date=None)

    walk_df = prepare_manager_with_cutoff(manager=manager, train_data=train_data, full_data=full_data, cutoff_date=cutoff_date)

    manager.config.max_top_tickers = 20
    X_walk = walk_df[manager.feature_cols].to_numpy(dtype=np.float32)
    walk_df = walk_df.copy()

    log("Training CAT up to cutoff...")
    cat_model = fit_walk_forward_model(manager=manager, interval=interval, model_type="CAT")

    log("Training LGBM up to cutoff...")
    lgbm_model = fit_walk_forward_model(manager=manager, interval=interval, model_type="LGBM")

    walk_df["pred_cat"] = cat_model.predict(X_walk)
    walk_df["pred_lgbm"] = lgbm_model.predict(X_walk)

    walk_df = add_consensus_ensemble_predictions(
        walk_df,
        top_n_per_model=50,
        cat_weight=0.6,
        lgbm_weight=0.4,
    )

    walk_df = add_next_open_execution_returns(walk_df, spread_bps=1.0)

    dates = list(sorted(pd.to_datetime(walk_df["Date"]).unique()))
    by_date = {
        date: group.copy()
        for date, group in walk_df.groupby("Date", sort=True)
    }

    holdings: dict[str, dict[str, Any]] = {}

    equity = float(initial_capital)
    cost = manager.config.cost_bps / 10_000
    max_positions = manager.config.max_top_tickers
    max_holding_days = manager.config.horizon

    daily_rows = []
    trade_rows = []
    position_rows = []
    for date in dates[:-1]:
        day = by_date[date].copy()
        day = day.dropna(subset=["pred", "exec_return", "exec_entry_price_mid", "exec_exit_price_mid"])

        if day.empty:
            continue

        exit_signal_day = manager.add_cross_sectional_signals(
            day,
            "pred",
            ["Date", "profile_group"],
            allow_short=True,
        ).set_index("ticker", drop=False)

        candidates = manager._daily_candidate_pool(day)
        turnover = 0

        # Exits
        for ticker in list(holdings.keys()):
            if ticker not in exit_signal_day.index:
                holdings.pop(ticker)
                turnover += 1

                trade_rows.append({
                    "Date": date,
                    "ticker": ticker,
                    "action": "EXIT_MISSING",
                })
                continue

            row = exit_signal_day.loc[ticker]
            side = holdings[ticker]["side"]
            age = holdings[ticker]["age"]

            opposite_signal = row["pred_signal"] == -side
            weak_long = side == 1 and row["pred"] <= manager.config.exit_pred_threshold
            weak_short = side == -1 and row["pred"] >= -manager.config.exit_pred_threshold
            too_old = age >= max_holding_days

            if opposite_signal or weak_long or weak_short or too_old:
                holdings.pop(ticker)
                turnover += 1

                trade_rows.append({
                    "Date": date,
                    "ticker": ticker,
                    "action": "EXIT",
                    "opposite_signal": bool(opposite_signal),
                    "weak_long": bool(weak_long),
                    "weak_short": bool(weak_short),
                    "too_old": bool(too_old),
                })

        # Entries / replacements
        for _, row in candidates.iterrows():
            ticker = row["ticker"]
            side = int(row["pred_signal"])
            score = float(row["score"])

            if ticker in holdings:
                holdings[ticker]["score"] = score
                holdings[ticker]["side"] = side
                continue

            if len(holdings) < max_positions:
                holdings[ticker] = {
                    "side": side,
                    "score": score,
                    "age": 0,
                }
                turnover += 1

                trade_rows.append({
                    "Date": date,
                    "ticker": ticker,
                    "action": "BUY" if side == 1 else "SHORT",
                    "pred": float(row["pred"]),
                    "score": score,
                })
                continue

            weakest_ticker = min(
                holdings,
                key=lambda t: holdings[t]["score"],
            )
            weakest_score = holdings[weakest_ticker]["score"]

            if score > weakest_score + manager.config.replace_buffer:
                holdings.pop(weakest_ticker)
                holdings[ticker] = {
                    "side": side,
                    "score": score,
                    "age": 0,
                }
                turnover += 2

                trade_rows.append({
                    "Date": date,
                    "ticker": weakest_ticker,
                    "action": "REPLACE_EXIT",
                })
                trade_rows.append({
                    "Date": date,
                    "ticker": ticker,
                    "action": "BUY" if side == 1 else "SHORT",
                    "pred": float(row["pred"]),
                    "score": score,
                })

        day_indexed = day.set_index("ticker", drop=False)
        position_returns = []
        position_details = []

        for ticker, info in holdings.items():
            if ticker not in day_indexed.index:
                continue

            row = day_indexed.loc[ticker]
            side = int(info["side"])

            if side == 1:
                raw_return = row["long_exec_return"]
                entry_price = row["long_entry_price"]
                exit_price = row["long_exit_price"]
            else:
                raw_return = row["short_exec_return"]
                entry_price = row["short_entry_price"]
                exit_price = row["short_exit_price"]

            if not np.isfinite(raw_return):
                continue

            position_returns.append(float(raw_return))

            position_details.append({
                "signal_date": date,
                "exec_entry_date": row["exec_entry_date"],
                "exec_exit_date": row["exec_exit_date"],
                "ticker": ticker,
                "side": side,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "position_return": float(raw_return),
                "holding_age": int(info["age"]),
                "score": float(info["score"]),
            })

        gross_return = float(np.mean(position_returns)) if position_returns else 0.0
        turnover_cost = cost * turnover / max(1, max_positions)
        daily_return = gross_return - turnover_cost

        equity_before = equity
        equity *= 1.0 + daily_return

        if position_details:
            weight = 1.0 / len(position_details)

            for row in position_details:
                contribution_return = row["position_return"] * weight
                row["weight"] = weight
                row["contribution_return"] = contribution_return
                row["contribution_pnl"] = equity_before * contribution_return
                position_rows.append(row)

        daily_rows.append({
            "Date": date,
            "equity": equity,
            "daily_return": daily_return,
            "gross_return": gross_return,
            "turnover_cost": turnover_cost,
            "turnover": turnover,
            "holdings": len(holdings),
            "held_tickers": ",".join(sorted(holdings.keys())),
        })

        for info in holdings.values(): info["age"] += 1

    daily_df = pd.DataFrame(daily_rows)
    trades_df = pd.DataFrame(trade_rows)
    position_df = pd.DataFrame(position_rows)

    if daily_df.empty:
        raise ValueError("Walk-forward produced no daily rows.")

    daily_df["Date"] = pd.to_datetime(daily_df["Date"])
    daily_df = daily_df.sort_values("Date")

    returns = daily_df["daily_return"].astype(float)
    equity_curve = daily_df["equity"].astype(float)

    drawdown = equity_curve / equity_curve.cummax() - 1.0

    total_return = equity / initial_capital - 1.0

    if interval == "1d":
        years = len(daily_df) / 252
        annualiser = np.sqrt(252)
    else:
        years = len(daily_df) / (252 * 6.5)
        annualiser = np.sqrt(252 * 6.5)

    cagr_like = (
        (equity / initial_capital) ** (1 / years) - 1.0
        if years > 0
        else 0.0
    )

    sharpe_like = returns.mean() / (returns.std() + 1e-9) * annualiser
    attribution = analyse_position_attribution(position_df)

    summary = {
        "interval": interval,
        "model_type": model_type,
        "months_back": months_back,
        "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
        "start_date": daily_df["Date"].min().strftime("%Y-%m-%d"),
        "end_date": daily_df["Date"].max().strftime("%Y-%m-%d"),
        "initial_capital": float(initial_capital),
        "final_equity": float(equity),
        "profit_loss": float(equity - initial_capital),
        "total_return": float(total_return),
        "cagr_like": float(cagr_like),
        "sharpe_like": float(sharpe_like),
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((returns > 0).mean()),
        "mean_daily_return": float(returns.mean()),
        "median_daily_return": float(returns.median()),
        "avg_holdings": float(daily_df["holdings"].mean()),
        "avg_turnover": float(daily_df["turnover"].mean()),
        "trade_count": int(len(trades_df)),
        "attribution_concentration": attribution["concentration"],
    }

    save_walk_forward_results(
        interval=interval,
        model_type=model_type,
        months_back=months_back,
        summary=summary,
        daily_df=daily_df,
        trades_df=trades_df,
        position_df=position_df,
        attribution=attribution,
    )

    log("Walk-forward summary:")
    log(json.dumps(json_safe(summary), indent=4))

    return {
        "summary": summary,
        "daily": daily_df,
        "trades": trades_df,
    }

########################################################################################################################

if __name__ == "__main__":
    start = time.perf_counter()

    print("Training...")
    # mng = TrainingManager()
    # mng.run_training_pipeline("1d")

    for m in [3,6,12]:
        run_walk_forward_backtest(
            interval="1d",
            months_back=m,
            initial_capital=1000.0,
        )

    print(f"Total time: {time.perf_counter() - start:.1f}s")