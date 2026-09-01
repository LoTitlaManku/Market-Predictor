
import gc
import json
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from tqdm import tqdm

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from scripts.config import DATA_DIR, MODEL_DIR, ROOT_DIR, LOG_DIR
from scripts.data_management import load_comparative_data, load_data
import scripts.indicators  # noqa: F401

warnings.filterwarnings("ignore")

########################################################################################################################

class Settings:
    GPU = {"LGBM": False, "CAT": False}
    Threaded = False

MODEL_PIPELINE_VERSION = 4

@dataclass
class UniverseConfig:
    horizon: int = 30
    max_top_tickers: int = 30

    edge_q: float = 0.02
    min_abs_pred: float = 0.0
    allow_short: bool = False
    cost_bps: float = 10.0

    replace_buffer: float = 0.002
    exit_pred_threshold: float = 0.0

    min_price: float = 5.0
    max_price: float = 5000.0

    # The old static share-ADV snapshot leaked today's universe into historical
    # tests and biased the universe toward cheap, speculative names.  The causal
    # lagged dollar-volume filter below is the source of truth instead.
    min_profile_adv: float | None = None

    target_column: str = "target_risk_adjusted_return"
    target_clip: float = 3.0
    max_training_years: int = 10
    recency_half_life_days: float = 1095.0

    # Periodic fitting is available for robustness experiments, but remains
    # opt-in because the current held-out comparison favours the anchored fit.
    retrain_every_n_bars_1d: int = 0
    retrain_every_n_bars_1h: int = 0
    max_loaded_model_age_days_1d: int = 45
    max_loaded_model_age_days_1h: int = 7

    ensemble_cat_weight: float = 0.5
    ensemble_top_n: int = 50

    max_entry_rolling_vol: float = 0.80
    max_entry_abs_beta: float = 2.0
    max_entry_vol_percentile: float = 0.80
    min_position_risk_scale: float = 0.33

    comparative_max_staleness_days_1d: int = 5
    comparative_max_staleness_days_1h: int = 2

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

now = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
def log(string: str, prints: bool = True):
    if prints: print(string)
    with open(os.path.join(LOG_DIR, f"log [{now}].txt"), "a") as f: f.write(f"{string}\n")

def json_safe(value):
    if isinstance(value, dict):                              return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):                     return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):                        return value.tolist()
    if isinstance(value, pd.Timestamp):                      return value.isoformat()
    if isinstance(value, Path):                              return str(value)
    if isinstance(value, np.generic):                        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):  return None

    return value

########################################################################################################################

class DataManager:
    def __init__(self, config: UniverseConfig | None = None):
        self.config = config if config else UniverseConfig()

    @staticmethod
    def _add_liquidity_columns(df: pd.DataFrame, interval: str) -> pd.DataFrame:
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

        df["rolling_vol"] = stock_ret.rolling(window=vol_window, min_periods=min_vol_periods).std().shift(
            1) * annualiser
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

    def add_forward_excess_target(
            self, df: pd.DataFrame, benchmark_close: pd.Series, interval: str,
            drop_unlabelled: bool = True
    ) -> pd.DataFrame:
        df = df.copy()
        beta = df["rolling_beta"]
        close = df["Adj Close"]
        benchmark_close = benchmark_close.reindex(df.index).ffill()

        df["future_return"] = close.shift(-self.config.horizon) / close - 1.0
        df["benchmark_future_return"] = (benchmark_close.shift(-self.config.horizon) / benchmark_close - 1.0)
        # The source row's label is not observable until this date.  Rolling
        # retraining filters on it so the horizon can never cross a model's
        # as-of date (the usual purged walk-forward requirement).
        df["target_end_date"] = pd.Series(df.index, index=df.index).shift(-self.config.horizon)

        if isinstance(beta, pd.Series):
            beta_used = beta.reindex(df.index).ffill()
            beta_used = beta_used.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        else:
            beta_used = 1.0 if not np.isfinite(beta) else beta

        df["target_beta_used"] = beta_used
        df["target_excess_return"] = (df["future_return"] - beta_used * df["benchmark_future_return"])

        bars_per_year = 252.0 * (6.5 if interval == "1h" else 1.0)
        horizon_sigma = df["rolling_vol"] * np.sqrt(self.config.horizon / bars_per_year)
        horizon_sigma = horizon_sigma.clip(lower=0.05 if interval == "1d" else 0.01)

        # A long-only book earns raw returns, so use a volatility-scaled raw
        # target.  Keep the excess version available for hedged experiments.
        df["target_risk_adjusted_return"] = (
            df["future_return"] / horizon_sigma
        ).clip(-self.config.target_clip, self.config.target_clip)
        df["target_risk_adjusted_excess_return"] = (
            df["target_excess_return"] / horizon_sigma
        ).clip(-self.config.target_clip, self.config.target_clip)

        return df.dropna(subset=[self.config.target_column]) if drop_unlabelled else df

    @staticmethod
    def load_raw_data(interval: str) -> dict[str, pd.DataFrame]:
        with open(os.path.join(DATA_DIR, "ticker_attr.json"), "r") as f:
            ticker_map = json.load(f)
        ticker_list = sorted(set(ticker_map.keys()))
        data_dict: dict[str, pd.DataFrame] = {}

        log("")
        for ticker in tqdm(ticker_list, desc="Loading raw ticker data"):
            try:
                data = load_data(ticker, interval)
                if data is None or data.empty: continue

                data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
                data = data[~data.index.duplicated(keep="last")].sort_index()

                data_dict[ticker] = data

            except Exception as e:
                log(f"Skipping raw load {ticker}: {type(e).__name__}: {e}", prints=False)

        log("")
        if not data_dict: raise ValueError("No raw universe data loaded.")
        return data_dict

    def validate_comparative_freshness(
            self, interval: str, latest_stock_date: pd.Timestamp
    ):
        max_staleness = (
            self.config.comparative_max_staleness_days_1h if interval == "1h"
            else self.config.comparative_max_staleness_days_1d
        )
        for comparative in ("SPY", "VIX", "VVIX", "TYX"):
            comparative_data = load_comparative_data(comparative, interval)
            lag = pd.Timestamp(latest_stock_date) - pd.Timestamp(comparative_data.index.max())
            if lag > pd.Timedelta(days=max_staleness):
                raise ValueError(
                    f"Stale {comparative}_{interval} data: latest comparative date "
                    f"{comparative_data.index.max():%Y-%m-%d}, latest stock date "
                    f"{pd.Timestamp(latest_stock_date):%Y-%m-%d}. Update comparative data before training."
                )

    def build_universe(self, interval: str, data_dict: dict, drop_unlabelled: bool = True, cutoff_date: pd.Timestamp | None = None) -> pd.DataFrame:
        with open(os.path.join(DATA_DIR, "ticker_attr.json"), "r") as f:
            ticker_map = json.load(f)

        latest_stock_date = max(
            pd.to_datetime(frame.index, utc=True).tz_localize(None).max()
            for frame in data_dict.values() if frame is not None and not frame.empty
        )
        self.validate_comparative_freshness(interval, latest_stock_date)

        benchmark_raw = load_comparative_data("SPY", interval)
        benchmark_raw.index.name = "Date"
        benchmark_raw.index = pd.to_datetime(benchmark_raw.index, utc=True).tz_localize(None)
        benchmark_raw = benchmark_raw[~benchmark_raw.index.duplicated(keep="first")].sort_index()
        benchmark_close = benchmark_raw["Adj Close"]

        log("")
        frames: list[pd.DataFrame] = []
        for ticker, data in tqdm(data_dict.items()):
            try:
                data = data.copy()
                if cutoff_date is not None: data = data.loc[:pd.Timestamp(cutoff_date)].copy()

                adv = ticker_map.get(ticker, {}).get("adv", np.nan)
                if (
                        self.config.min_profile_adv is not None
                        and np.isfinite(adv)
                        and self.config.min_profile_adv > max(0, float(adv))
                ):
                    continue

                df = data.ind.add_indicators(ticker, interval, add_targets=False)

                df = self._add_liquidity_columns(df, interval)
                df = self._add_rolling_risk_columns(df, benchmark_close, interval)

                df = self.add_forward_excess_target(df, benchmark_close, interval, drop_unlabelled)
                df = self._apply_liquidity_filters(df, interval)

                if df.empty: continue

                df["ticker"] = ticker
                df["Date"] = df.index
                df = df.reset_index(drop=True)
                frames.append(df)

            except Exception as e:
                log(f"Skipping {ticker}: {type(e).__name__}: {e}", prints=False)
                pass

        if not frames: raise ValueError("No usable universe data")

        data: pd.DataFrame = pd.concat(frames, axis=0, ignore_index=True)
        data = data.sort_values(["Date", "ticker"])
        data = data.replace([np.inf, -np.inf], np.nan)

        # Point-in-time cross-sectional regime context.  These use only values
        # observable at the signal close and are shared by every ticker that day.
        by_date = data.groupby("Date")
        data["Market_Breadth_Above_200"] = by_date["PDMA_200"].transform(
            lambda values: float((values > 0).mean())
        )
        data["Market_Breadth_Above_50"] = by_date["PDMA_50"].transform(
            lambda values: float((values > 0).mean())
        )
        data["Market_Median_Momentum_1m"] = by_date["mom_1m"].transform("median")
        data["Market_Return_Dispersion"] = by_date["return_lag_1"].transform("std")
        data["Market_Median_Stock_Vol"] = by_date["rolling_vol"].transform("median")

        if drop_unlabelled:
            data = data.dropna()
        else:
            target_cols = {
                "future_return", "benchmark_future_return", "target_excess_return",
                "target_risk_adjusted_return", "target_risk_adjusted_excess_return",
                "target_end_date",
            }
            non_target_cols = [c for c in data.columns if c not in target_cols]
            data = data.dropna(subset=non_target_cols)

        log("")
        return data

    def add_signals(self, df: pd.DataFrame, allow_short: bool | None = None) -> pd.DataFrame:
        df = df.copy()
        df["pred_signal"] = 0

        allow_short = self.config.allow_short if allow_short is None else allow_short
        n_tickers = max(1, df["ticker"].nunique())
        edge_q = min(self.config.edge_q, self.config.max_top_tickers / n_tickers)

        for _, group in df.groupby(["Date", "profile_group"]):
            if len(group) < 5: continue

            upper = group["pred"].quantile(1 - edge_q)
            lower = group["pred"].quantile(edge_q)

            long_mask = (group["pred"] >= upper) & (group["pred"].abs() >= self.config.min_abs_pred)
            df.loc[group.index[long_mask], "pred_signal"] = 1

            if allow_short:
                short_mask = (group["pred"] <= lower) & (group["pred"].abs() >= self.config.min_abs_pred)
                df.loc[group.index[short_mask], "pred_signal"] = -1

        return df

    def add_portfolio_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign fixed-slot, risk-scaled weights without filling unused cash."""
        required = {"Date", "rolling_vol", "Regime_Exposure"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for portfolio weights: {missing}")

        df = df.copy()
        median_vol = df.groupby("Date")["rolling_vol"].transform("median")
        risk_scale = median_vol / df["rolling_vol"].replace(0, np.nan)
        df["position_risk_scale"] = risk_scale.clip(
            self.config.min_position_risk_scale, 1.0
        ).fillna(0.0)
        df["regime_exposure"] = df["Regime_Exposure"].clip(0.0, 1.0)
        df["target_weight"] = (
            df["regime_exposure"]
            * df["position_risk_scale"]
            / max(1, self.config.max_top_tickers)
        )
        return df

class Trainer:
    def __init__(self, interval: str, config: UniverseConfig | None = None):
        self.config = config if config else UniverseConfig()
        self.predictor = Predictor(interval, self.config)
        self.manager = DataManager(self.config)
        self.interval = interval
        self.prepared_data: pd.DataFrame | None = None
        self.retraining_records: list[dict[str, Any]] = []

    @staticmethod
    def calculate_gross_return(
            position_returns: list[float], position_weights: list[float]
    ) -> float:
        if len(position_returns) != len(position_weights):
            raise ValueError("Position returns and weights must have the same length")
        if not position_returns:
            return 0.0
        returns = np.asarray(position_returns, dtype=float)
        weights = np.asarray(position_weights, dtype=float)
        return float(np.dot(returns, weights))

    @staticmethod
    def analyse_pos_attr(position_df: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
        if position_df.empty:
            return {"ticker_summary": pd.DataFrame(), "best_position_days": pd.DataFrame(),
                    "worst_position_days": pd.DataFrame(), "concentration": {}}

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

        best_position_days = df.sort_values("contribution_pnl", ascending=False).head(25).copy()
        worst_position_days = df.sort_values("contribution_pnl", ascending=True).head(25).copy()

        total_positive_pnl = ticker_summary.loc[ticker_summary["total_pnl"] > 0, "total_pnl"].sum()

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

    @staticmethod
    def add_returns(df: pd.DataFrame, data_dict: dict) -> pd.DataFrame:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])

        frames: list[pd.DataFrame] = []
        log("")
        for ticker, group in tqdm(df.groupby("ticker", sort=False), desc="Adding open returns"):
            raw = data_dict.get(ticker, pd.DataFrame())
            if raw is None or raw.empty: continue

            raw = raw.copy()
            raw.index = pd.to_datetime(raw.index, utc=True).tz_localize(None)
            raw = raw[~raw.index.duplicated(keep="first")].sort_index()

            if "Adj Open" in raw.columns:
                raw["exec_open"] = raw["Adj Open"].astype(float)
            elif {"Open", "Close", "Adj Close"}.issubset(raw.columns):
                adj_factor = raw["Adj Close"].astype(float) / raw["Close"].astype(float)
                raw["exec_open"] = raw["Open"].astype(float) * adj_factor
            elif "Open" in raw.columns:
                raw["exec_open"] = raw["Open"].astype(float)
            else:
                continue

            g = group.copy()
            signal_dates = g["Date"].to_numpy(dtype="datetime64[ns]")
            raw_dates = raw.index.to_numpy(dtype="datetime64[ns]")
            raw_open = raw["exec_open"].to_numpy(dtype=float)

            entry_idx = np.searchsorted(raw_dates, signal_dates, side="right")
            exit_idx = entry_idx + 1

            valid = exit_idx < len(raw)

            g["exec_entry_date"] = pd.NaT
            g["exec_exit_date"] = pd.NaT
            g["exec_entry_price_mid"] = np.nan
            g["exec_exit_price_mid"] = np.nan

            g.loc[valid, "exec_entry_date"] = raw.index[entry_idx[valid]]
            g.loc[valid, "exec_exit_date"] = raw.index[exit_idx[valid]]

            g.loc[valid, "exec_entry_price_mid"] = raw_open[entry_idx[valid]]
            g.loc[valid, "exec_exit_price_mid"] = raw_open[exit_idx[valid]]

            # Mark multi-day holdings at consecutive mid opens.  Trading costs
            # are charged only when a position leg changes in the simulator.
            g["long_entry_price"] = g["exec_entry_price_mid"]
            g["long_exit_price"] = g["exec_exit_price_mid"]
            g["long_exec_return"] = g["long_exit_price"] / g["long_entry_price"] - 1.0

            g["short_entry_price"] = g["exec_entry_price_mid"]
            g["short_exit_price"] = g["exec_exit_price_mid"]
            g["short_exec_return"] = g["short_entry_price"] / g["short_exit_price"] - 1.0

            g["exec_return"] = g["long_exec_return"]

            frames.append(g)

        log("")
        if not frames: raise ValueError("No execution returns could be calculated.")
        return pd.concat(frames, axis=0, ignore_index=True).sort_values(["Date", "ticker"])

    @staticmethod
    def add_benchmark_performance(daily_df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Attach SPY returns using the same next-open-to-following-open clock."""
        benchmark = load_comparative_data("SPY", interval).copy()
        benchmark.index = pd.to_datetime(benchmark.index, utc=True).tz_localize(None)
        benchmark = benchmark[~benchmark.index.duplicated(keep="last")].sort_index()

        if "Adj Open" in benchmark.columns:
            benchmark_open = benchmark["Adj Open"].astype(float)
        elif {"Open", "Close", "Adj Close"}.issubset(benchmark.columns):
            adjustment = benchmark["Adj Close"].astype(float) / benchmark["Close"].astype(float)
            benchmark_open = benchmark["Open"].astype(float) * adjustment
        elif "Open" in benchmark.columns:
            benchmark_open = benchmark["Open"].astype(float)
        else:
            raise ValueError("SPY comparative data has no usable open price")

        dates = pd.to_datetime(daily_df["Date"]).to_numpy(dtype="datetime64[ns]")
        benchmark_dates = benchmark.index.to_numpy(dtype="datetime64[ns]")
        opens = benchmark_open.to_numpy(dtype=float)
        entry_idx = np.searchsorted(benchmark_dates, dates, side="right")
        exit_idx = entry_idx + 1
        valid = exit_idx < len(benchmark)

        benchmark_return = np.zeros(len(daily_df), dtype=float)
        benchmark_return[valid] = opens[exit_idx[valid]] / opens[entry_idx[valid]] - 1.0
        result = daily_df.copy()
        result["benchmark_return"] = benchmark_return
        result["benchmark_equity"] = 1000.0 * (1.0 + result["benchmark_return"]).cumprod()
        return result

    def prepare_predictor(self, full_data: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
        cutoff_date: pd.Timestamp = pd.Timestamp(cutoff_date)  # noqa

        full_data = full_data.copy()
        full_data["Date"] = pd.to_datetime(full_data["Date"])
        full_data["target_end_date"] = pd.to_datetime(full_data["target_end_date"])

        walk_data = full_data[full_data["Date"] > cutoff_date].copy()

        if walk_data.empty: raise ValueError("No walk-forward rows after cutoff date.")

        data = full_data
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
            "target_risk_adjusted_return", "target_risk_adjusted_excess_return",
            "target_end_date",
            "target_profit", "tbm_return", "barrier_strength",
            "time_to_gain", "time_to_loss", "tp_return", "sl_return",
            "dollar_volume", "rolling_dollar_volume", "abs_bar_return", "target_beta_used",
            "ATR", "MACD_Hist", "OBV", "Treasury_30Y", "hour", "day_of_week", "month",
        }

        self.predictor.feature_cols = [c for c in data.columns if
                                c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])]

        train_df = data[data["Date"] <= cutoff_date].copy()
        walk_df = data[data["Date"] > cutoff_date].copy()

        if train_df.empty: raise ValueError("No labelled training rows before cutoff date.")
        if walk_df.empty: raise ValueError("No walk-forward rows after cutoff date.")

        self.predictor.split_date = cutoff_date
        self.predictor.test_df = walk_df
        self.predictor.set_training_frame(train_df, cutoff_date)
        self.prepared_data = data

        log(f"Walk-forward cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
        log(f"Train rows: {len(self.predictor.train_df):,}")
        log(f"Walk-forward rows: {len(walk_df):,}")
        log(f"Features: {len(self.predictor.feature_cols)}")

        return walk_df

    def retrain_frequency(self) -> int:
        return (
            self.config.retrain_every_n_bars_1h
            if self.interval == "1h"
            else self.config.retrain_every_n_bars_1d
        )

    @staticmethod
    def trim_unexecutable_tail(
            walk_df: pd.DataFrame, required_future_bars: int = 2
    ) -> pd.DataFrame:
        """Drop signal dates that cannot have a complete next-open return."""
        dates = list(sorted(pd.to_datetime(walk_df["Date"]).unique()))
        if len(dates) <= required_future_bars:
            raise ValueError("Walk-forward window has no executable signal dates")
        executable_dates = set(dates[:-required_future_bars])
        return walk_df[walk_df["Date"].isin(executable_dates)].copy()

    @staticmethod
    def make_retraining_folds(
            evaluation_dates: list[pd.Timestamp], cutoff_date: pd.Timestamp,
            retrain_every_n_bars: int,
    ) -> list[dict[str, Any]]:
        """Return strictly out-of-sample prediction folds.

        A refit made after one completed signal close is first used on the next
        signal date.  Non-positive cadence preserves the anchored one-fit mode.
        """
        dates = [pd.Timestamp(date) for date in sorted(pd.to_datetime(evaluation_dates).unique())]
        if not dates:
            return []

        cadence = len(dates) if retrain_every_n_bars <= 0 else retrain_every_n_bars
        folds: list[dict[str, Any]] = []
        for start in range(0, len(dates), cadence):
            segment = dates[start:start + cadence]
            model_as_of = pd.Timestamp(cutoff_date) if start == 0 else dates[start - 1]
            folds.append({
                "fold": len(folds),
                "model_as_of_date": model_as_of,
                "prediction_dates": segment,
            })
        return folds

    def predict_with_retraining(
            self, walk_df: pd.DataFrame, cutoff_date: pd.Timestamp
    ) -> pd.DataFrame:
        if self.prepared_data is None:
            raise ValueError("Prepared feature data is not available")

        evaluation_dates = list(sorted(pd.to_datetime(walk_df["Date"]).unique()))
        cadence = self.retrain_frequency()
        folds = self.make_retraining_folds(evaluation_dates, cutoff_date, cadence)
        predicted: list[pd.DataFrame] = []
        self.retraining_records = []

        for fold in folds:
            as_of = pd.Timestamp(fold["model_as_of_date"])
            prediction_dates = fold["prediction_dates"]
            train_df = self.prepared_data[self.prepared_data["Date"] <= as_of].copy()
            self.predictor.set_training_frame(train_df, as_of)
            self.predictor.split_date = as_of

            log(
                f"Training fold {fold['fold'] + 1}/{len(folds)} as of "
                f"{as_of:%Y-%m-%d} for {len(prediction_dates)} prediction bars..."
            )
            cat_model = self.predictor.train_model("CAT")
            lgbm_model = self.predictor.train_model("LGBM")

            segment = walk_df[walk_df["Date"].isin(prediction_dates)].copy()
            X_segment = segment[self.predictor.feature_cols].to_numpy(dtype=np.float32)
            segment["pred_cat"] = cat_model.predict(X_segment)
            segment["pred_lgbm"] = lgbm_model.predict(X_segment)
            segment["model_as_of_date"] = as_of
            segment["retrain_fold"] = int(fold["fold"])
            predicted.append(segment)

            max_label_end = pd.to_datetime(self.predictor.train_df["target_end_date"]).max()
            self.retraining_records.append({
                "fold": int(fold["fold"]),
                "model_as_of_date": as_of,
                "prediction_start": min(prediction_dates),
                "prediction_end": max(prediction_dates),
                "prediction_bars": len(prediction_dates),
                "training_rows": len(self.predictor.train_df),
                "latest_training_feature_date": pd.to_datetime(self.predictor.train_df["Date"]).max(),
                "latest_label_end_date": max_label_end,
            })

            del cat_model, lgbm_model
            flush_memory()

        if not predicted:
            raise ValueError("No walk-forward predictions were generated")
        return pd.concat(predicted, ignore_index=True).sort_values(["Date", "ticker"])

    def save_results(
            self, months_back: int | float, summary: dict, daily_df: pd.DataFrame,
            trades_df: pd.DataFrame, position_df: pd.DataFrame | None = None,
            attribution: dict | None = None, signal_df: pd.DataFrame | None = None,
            retraining_df: pd.DataFrame | None = None,
            predictions_df: pd.DataFrame | None = None,
    ) -> Path:

        folder = Path(MODEL_DIR) / f"{self.interval} Model [{now}]" / "walk_forward" / str(months_back)
        folder.mkdir(parents=True, exist_ok=True)

        daily_df.to_csv(folder / "daily_equity.csv", index=False)
        trades_df.to_csv(folder / "trades.csv", index=False)
        if signal_df is not None:
            signal_df.to_csv(folder / "signal_diagnostics.csv", index=False)
        if retraining_df is not None:
            retraining_df.to_csv(folder / "retraining_folds.csv", index=False)
        if predictions_df is not None:
            prediction_cols = [
                "Date", "ticker", "model_as_of_date", "retrain_fold",
                "pred", "pred_raw", "pred_cat", "pred_lgbm", "ensemble_score",
                "ensemble_agreement", "risk_eligible", "entry_eligible",
                "rolling_vol", "rolling_beta", "vol_rank_pct", "target_weight",
                "regime_exposure", "position_risk_scale", "exec_return",
            ]
            predictions_df[[
                col for col in prediction_cols if col in predictions_df.columns
            ]].to_parquet(folder / "walk_forward_predictions.parquet", index=False)

        if position_df is not None and not position_df.empty:
            position_df.to_csv(folder / "position_returns.csv", index=False)

        if attribution is not None:
            ticker_summary = attribution.get("ticker_summary")
            best_position_days = attribution.get("best_position_days")
            worst_position_days = attribution.get("worst_position_days")
            concentration = attribution.get("concentration", {})

            if ticker_summary is not None and not ticker_summary.empty:  # noqa
                ticker_summary.to_csv(folder / "ticker_attribution.csv", index=False)  # noqa

            if best_position_days is not None and not best_position_days.empty:  # noqa
                best_position_days.to_csv(folder / "best_position_days.csv", index=False)  # noqa

            if worst_position_days is not None and not worst_position_days.empty:  # noqa
                worst_position_days.to_csv(folder / "worst_position_days.csv", index=False)  # noqa

            (folder / "attr_summary.json").write_text(json.dumps(json_safe(concentration), indent=4), encoding="utf-8")

        (folder / "summary.json").write_text(json.dumps(json_safe(summary), indent=4), encoding="utf-8")

        log(f"Saved walk-forward results to: {folder}")
        return folder

    def run_training(
            self, months_back: int | float, retrain_every_n_bars: int | None = None
    ) -> dict[str, Any]:
        if retrain_every_n_bars is not None:
            if self.interval == "1h":
                self.config.retrain_every_n_bars_1h = int(retrain_every_n_bars)
            else:
                self.config.retrain_every_n_bars_1d = int(retrain_every_n_bars)
        if self.interval == "1d":
            self.config.horizon = 40
            self.config.max_top_tickers = 30
        elif self.interval == "1h":
            self.config.horizon = 30
            self.config.max_top_tickers = 10

        model_type = "ENSEMBLE"
        retrain_every_n_bars = self.retrain_frequency()
        retraining_name = (
            f"periodic expanding fit every {retrain_every_n_bars} bars"
            if retrain_every_n_bars > 0 else "single fit at cutoff"
        )
        log(
            f"{'=' * 50}\nPURGED WALK-FORWARD BACKTEST ({self.interval}): "
            f"cutoff = {months_back} months ({retraining_name})\n{'=' * 50}"
        )

        log("Loading all tickers...")
        data_dict = self.manager.load_raw_data(self.interval)

        latest_date = max(
            pd.Timestamp(frame.index.max())
            for frame in data_dict.values() if frame is not None and not frame.empty
        )
        if isinstance(months_back, int):
            cutoff_date: pd.Timestamp = latest_date - pd.DateOffset(months=months_back)  # noqa
        else:
            cutoff_date: pd.Timestamp = latest_date - pd.DateOffset(days=round(months_back * 30))  # noqa

        log(f"Latest available date: {latest_date.strftime('%Y-%m-%d')}")
        log(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")

        # Build features once.  target_end_date controls which labels are
        # actually available at each later refit.
        log("Building full point-in-time universe...")
        full_data = self.manager.build_universe(self.interval, data_dict, False)

        walk_df = self.prepare_predictor(full_data, cutoff_date)
        walk_df = self.trim_unexecutable_tail(walk_df)
        walk_df = self.predict_with_retraining(walk_df, cutoff_date)

        # Merge model scores and calculate execution returns
        walk_df = self.predictor.add_ensemble_predictions(walk_df)
        walk_df = self.manager.add_portfolio_weights(walk_df)
        walk_df = self.add_returns(walk_df, data_dict)

        dates = list(sorted(pd.to_datetime(walk_df["Date"]).unique()))
        by_date = {date: group.copy() for date, group in walk_df.groupby("Date", sort=True)}

        holdings: dict[str, dict[str, Any]] = {}
        equity = 1000.0

        daily_rows = []
        trade_rows = []
        position_rows = []
        signal_rows = []
        for date in dates:
            # Get predictions/returns for this date and drop invalid records
            day = by_date[date].copy()
            day = day.dropna(subset=["pred", "exec_return", "exec_entry_price_mid", "exec_exit_price_mid"])
            if day.empty: continue

            # Build signal table for the day
            exit_signal_day = self.manager.add_signals(day, allow_short=True).set_index("ticker", drop=False)
            signalled = self.manager.add_signals(day)
            candidates = signalled[
                (signalled["pred_signal"] != 0)
                & (signalled["ensemble_agreement"] == 1)
                & (signalled["entry_eligible"] == 1)
            ].copy()

            # Sort entry candidates based on directional confidence strength
            if not candidates.empty:
                candidates["score"] = np.where(
                    candidates["pred_signal"] == 1,
                    candidates["ensemble_score"],
                    1.0 - candidates["ensemble_score"],
                )
                candidates = candidates.sort_values("score", ascending=False)

            turnover = 0
            entries = 0
            replacements = 0
            exits_missing = 0
            exits_opposite = 0
            exits_weak = 0
            exits_age = 0

            # Scan and manage open portfolio positions
            for ticker in list(holdings.keys()):
                # Liquidate if ticker drops out of the target universe
                if ticker not in exit_signal_day.index:
                    holdings.pop(ticker)
                    turnover += 1
                    exits_missing += 1
                    trade_rows.append({"Date": date, "ticker": ticker, "action": "EXIT_MISSING"})
                    continue

                row = exit_signal_day.loc[ticker]
                side = holdings[ticker]["side"]
                age = holdings[ticker]["age"]

                # Check if position breaks risk limits or holding constraints
                opposite_signal = row["pred_signal"] == -side
                weak_long = side == 1 and row["pred"] <= self.config.exit_pred_threshold
                weak_short = side == -1 and row["pred"] >= -self.config.exit_pred_threshold
                too_old = age >= self.config.horizon

                # Exit if signal has flipped, weakened, disappeared, or holding is too old
                if opposite_signal or weak_long or weak_short or too_old:  # noqa
                    holdings.pop(ticker)
                    turnover += 1
                    exits_opposite += int(bool(opposite_signal))
                    exits_weak += int(bool(weak_long or weak_short))
                    exits_age += int(bool(too_old))

                    trade_rows.append({
                        "Date": date, "ticker": ticker, "action": "EXIT", "opposite_signal": bool(opposite_signal),
                        "weak_long": bool(weak_long), "weak_short": bool(weak_short), "too_old": bool(too_old),
                    })

            # Process new portfolio trade entries and allocations
            for _, row in candidates.iterrows():
                ticker = row["ticker"]
                side = int(row["pred_signal"])
                score = float(row["score"])

                # If already held, just refresh its score
                if ticker in holdings:
                    holdings[ticker]["score"] = score
                    holdings[ticker]["side"] = side
                    continue

                # If portfolio has room, add the new candidate
                if len(holdings) < self.config.max_top_tickers:
                    holdings[ticker] = {"side": side, "score": score, "age": 0}
                    turnover += 1
                    entries += 1

                    trade_rows.append({
                        "Date": date, "ticker": ticker, "action": "BUY" if side == 1 else "SHORT",
                        "pred": float(row["pred"]), "score": score,
                        "rolling_vol": float(row["rolling_vol"]),
                        "rolling_beta": float(row["rolling_beta"]),
                        "target_weight": float(row["target_weight"]),
                    })
                    continue

                # If full, replace the weakest holding
                weakest_ticker = min(holdings, key=lambda t: holdings[t]["score"])
                weakest_score = holdings[weakest_ticker]["score"]
                if score > weakest_score + self.config.replace_buffer:
                    holdings.pop(weakest_ticker)
                    holdings[ticker] = {"side": side, "score": score, "age": 0}
                    turnover += 2
                    entries += 1
                    replacements += 1

                    trade_rows.append({"Date": date, "ticker": weakest_ticker, "action": "REPLACE_EXIT"})
                    trade_rows.append({
                        "Date": date, "ticker": ticker, "action": "BUY" if side == 1 else "SHORT",
                        "pred": float(row["pred"]), "score": score,
                        "rolling_vol": float(row["rolling_vol"]),
                        "rolling_beta": float(row["rolling_beta"]),
                        "target_weight": float(row["target_weight"]),
                    })

            day_indexed = day.set_index("ticker", drop=False)
            position_returns = []
            position_weights = []
            position_details = []

            # Calculate one-day execution return for every open holding
            for ticker, info in holdings.items():
                if ticker not in day_indexed.index: continue

                row = day_indexed.loc[ticker]
                side = int(info["side"])

                # Use long or short execution prices depending on position side
                if side == 1:
                    raw_return = row["long_exec_return"]
                    entry_price = row["long_entry_price"]
                    exit_price = row["long_exit_price"]
                else:
                    raw_return = row["short_exec_return"]
                    entry_price = row["short_entry_price"]
                    exit_price = row["short_exit_price"]

                if not np.isfinite(raw_return): continue

                position_weight = float(row["target_weight"])
                position_returns.append(raw_return)
                position_weights.append(position_weight)

                position_details.append({
                    "signal_date": date, "exec_entry_date": row["exec_entry_date"],
                    "exec_exit_date": row["exec_exit_date"],
                    "ticker": ticker, "side": side, "entry_price": entry_price, "exit_price": exit_price,
                    "position_return": raw_return, "holding_age": int(info["age"]), "score": float(info["score"]),
                    "weight": position_weight, "regime_exposure": float(row["regime_exposure"]),
                    "position_risk_scale": float(row["position_risk_scale"]),
                    "rolling_vol": float(row["rolling_vol"]),
                    "rolling_beta": float(row["rolling_beta"]),
                })

            # Fixed-slot weights leave unfilled or risk-scaled capacity in cash.
            gross_return = self.calculate_gross_return(position_returns, position_weights)
            turnover_cost = (self.config.cost_bps / 10_000) * turnover / max(1, self.config.max_top_tickers)
            daily_return = gross_return - turnover_cost

            # Compute and apply compounded growth changes to total capital
            equity_before = equity
            equity *= 1.0 + daily_return

            # Append performance context to position details tracking log
            if position_details:
                for row in position_details:
                    contribution_return = row["position_return"] * row["weight"]
                    row["contribution_return"] = contribution_return
                    row["contribution_pnl"] = equity_before * contribution_return
                    position_rows.append(row)

            # Store unified daily execution benchmarks
            daily_rows.append({
                "Date": date,
                "equity": equity,
                "daily_return": daily_return,
                "gross_return": gross_return,
                "turnover_cost": turnover_cost,
                "turnover": turnover,
                "holdings": len(holdings),
                "gross_exposure": float(sum(row["weight"] for row in position_details)),
                "regime_exposure": float(day["regime_exposure"].median()),
                "held_tickers": ",".join(sorted(holdings.keys())),
            })

            both_positive = (signalled["pred_cat"] > self.config.min_abs_pred) & (
                signalled["pred_lgbm"] > self.config.min_abs_pred
            )
            positive_agreement = both_positive & signalled["ensemble_agreement"].eq(1)
            signal_rows.append({
                "Date": date,
                "model_as_of_date": pd.to_datetime(signalled["model_as_of_date"]).max(),
                "retrain_fold": int(signalled["retrain_fold"].max()),
                "universe_rows": int(len(signalled)),
                "pred_cat_positive": int((signalled["pred_cat"] > self.config.min_abs_pred).sum()),
                "pred_lgbm_positive": int((signalled["pred_lgbm"] > self.config.min_abs_pred).sum()),
                "both_models_positive": int(both_positive.sum()),
                "positive_agreement": int(positive_agreement.sum()),
                "risk_eligible": int(signalled["risk_eligible"].eq(1).sum()),
                "agreement_and_risk_eligible": int(signalled["entry_eligible"].eq(1).sum()),
                "quantile_long_signals": int(signalled["pred_signal"].eq(1).sum()),
                "final_candidates": int(len(candidates)),
                "entries": entries,
                "replacements": replacements,
                "exits_missing": exits_missing,
                "exits_opposite": exits_opposite,
                "exits_weak": exits_weak,
                "exits_age": exits_age,
                "holdings": len(holdings),
                "gross_exposure": float(sum(row["weight"] for row in position_details)),
                "regime_exposure": float(day["regime_exposure"].median()),
                "median_pred_raw": float(signalled["pred_raw"].median()),
                "max_pred_raw": float(signalled["pred_raw"].max()),
            })

            # Increment active position lifetime counters
            for info in holdings.values(): info["age"] += 1

        # Consolidate matrix output logs into pandas dataframes
        daily_df = pd.DataFrame(daily_rows)
        trades_df = pd.DataFrame(trade_rows)
        position_df = pd.DataFrame(position_rows)
        signal_df = pd.DataFrame(signal_rows)

        if daily_df.empty: raise ValueError("Walk-forward produced no daily rows.")

        # Sort aggregated performance results sequentially by timestamp
        daily_df["Date"] = pd.to_datetime(daily_df["Date"])
        daily_df = daily_df.sort_values("Date")
        daily_df = self.add_benchmark_performance(daily_df, self.interval)

        # Compute max drawdown curves and overall backtest performance
        returns = daily_df["daily_return"].astype(float)
        equity_curve = daily_df["equity"].astype(float)
        drawdown = equity_curve / equity_curve.cummax() - 1.0
        total_return = equity / 1000.0 - 1.0
        benchmark_returns = daily_df["benchmark_return"].astype(float)
        benchmark_equity = daily_df["benchmark_equity"].astype(float)
        benchmark_drawdown = benchmark_equity / benchmark_equity.cummax() - 1.0
        benchmark_total_return = float(benchmark_equity.iloc[-1] / 1000.0 - 1.0)
        active_daily_return = returns - benchmark_returns

        # Configure time normalization metrics matching execution intervals
        multiplier = 6.5 if self.interval == "1h" else 1
        years = len(daily_df) / (252 * multiplier)
        annualiser = np.sqrt(252 * multiplier)

        # Generate standardized portfolio tracking analytics
        cagr_like = (equity / 1000.0) ** (1 / years) - 1.0 if years > 0 else 0.0
        sharpe_like = returns.mean() / (returns.std() + 1e-9) * annualiser  # noqa
        information_ratio_like = (
            active_daily_return.mean() / (active_daily_return.std() + 1e-9) * annualiser
        )
        attribution = self.analyse_pos_attr(position_df)
        zero_holdings = daily_df["holdings"].eq(0)
        zero_groups = zero_holdings.ne(zero_holdings.shift()).cumsum()
        longest_zero_streak = int(
            zero_holdings.groupby(zero_groups).sum().max()
        ) if zero_holdings.any() else 0
        retraining_df = pd.DataFrame(self.retraining_records)
        retraining_mode = (
            f"periodic_expanding_every_{retrain_every_n_bars}_bars"
            if retrain_every_n_bars > 0 else "single_fit_at_cutoff"
        )

        # Build output metadata dict package
        summary = {
            "interval": self.interval,
            "model_type": model_type,
            "retraining": retraining_mode,
            "retrain_every_n_bars": int(retrain_every_n_bars),
            "model_fit_count": int(len(retraining_df)),
            "feature_count": int(len(self.predictor.feature_cols)),
            "target_column": self.config.target_column,
            "months_back": months_back,
            "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
            "start_date": daily_df["Date"].min().strftime("%Y-%m-%d"),  # noqa
            "end_date": daily_df["Date"].max().strftime("%Y-%m-%d"),  # noqa
            "initial_capital": 1000.0,
            "final_equity": float(equity),
            "profit_loss": float(equity - 1000.0),
            "total_return": float(total_return),
            "benchmark_total_return": benchmark_total_return,
            "active_total_return": float(total_return - benchmark_total_return),
            "cagr_like": float(cagr_like),
            "sharpe_like": float(sharpe_like),
            "information_ratio_like": float(information_ratio_like),
            "max_drawdown": float(drawdown.min()),
            "benchmark_max_drawdown": float(benchmark_drawdown.min()),
            "win_rate": (returns > 0).mean(),
            "mean_daily_return": returns.mean(),
            "median_daily_return": returns.median(),
            "avg_holdings": float(daily_df["holdings"].mean()),
            "avg_gross_exposure": float(daily_df["gross_exposure"].mean()),
            "avg_regime_exposure": float(daily_df["regime_exposure"].mean()),
            "avg_turnover": float(daily_df["turnover"].mean()),
            "max_daily_turnover": int(daily_df["turnover"].max()),
            "zero_holding_days": int(zero_holdings.sum()),
            "longest_zero_holding_streak": longest_zero_streak,
            "no_candidate_days": int(signal_df["final_candidates"].eq(0).sum()),
            "avg_daily_candidates": float(signal_df["final_candidates"].mean()),
            "trade_count": int(len(trades_df)),
            "attribution_concentration": attribution["concentration"],
        }

        # Save serialized simulation logs directly to workspace paths
        self.save_results(
            months_back, summary, daily_df, trades_df, position_df, attribution,
            signal_df=signal_df, retraining_df=retraining_df, predictions_df=walk_df,
        )

        log("Walk-forward summary:")
        log(json.dumps(json_safe(summary), indent=4))
        return {
            "summary": summary,
            "daily": daily_df,
            "trades": trades_df,
            "signals": signal_df,
            "retraining": retraining_df,
        }

class Predictor:
    def __init__(self, interval: str, config: UniverseConfig | None = None):
        self.config = config if config else UniverseConfig()
        self.datamanager = DataManager(self.config)
        self.interval = interval
        self.seed = 69

        self.feature_cols = None
        self.split_date = None

        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.sample_weight = None
        self.as_of_date = None

    def make_sample_weights(
            self, dates: pd.Series, reference_date: pd.Timestamp
    ) -> np.ndarray:
        """Equalise each market date, then exponentially favour recent history."""
        dates = pd.to_datetime(dates)
        reference_date = pd.Timestamp(reference_date)
        if self.config.recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be positive")

        age_days = (reference_date - dates).dt.days.clip(lower=0).astype(float)
        recency = np.power(2.0, -age_days / self.config.recency_half_life_days)
        rows_per_date = dates.groupby(dates).transform("size").astype(float)
        weights = recency / rows_per_date
        mean_weight = float(weights.mean())
        if not np.isfinite(mean_weight) or mean_weight <= 0:
            raise ValueError("Could not construct finite training sample weights")
        return (weights / mean_weight).to_numpy(dtype=np.float32)

    def set_training_frame(
            self, train_df: pd.DataFrame, reference_date: pd.Timestamp
    ) -> None:
        target_col = self.config.target_column
        if target_col not in train_df.columns:
            raise ValueError(f"Configured target column is missing: {target_col}")
        if "target_end_date" not in train_df.columns:
            raise ValueError("target_end_date is required for leakage-safe training")

        reference_date = pd.Timestamp(reference_date)
        target_end_date = pd.to_datetime(train_df["target_end_date"], errors="coerce")
        eligible = (
            train_df[target_col].notna()
            & target_end_date.notna()
            & target_end_date.le(reference_date)
        )
        train_df = train_df.loc[eligible].copy()
        if self.config.max_training_years > 0:
            history_start = reference_date - pd.DateOffset(years=self.config.max_training_years)
            train_df = train_df[pd.to_datetime(train_df["Date"]) >= history_start].copy()
        if train_df.empty:
            raise ValueError("No rows remain after applying the training-history window")
        if pd.to_datetime(train_df["target_end_date"]).max() > reference_date:
            raise AssertionError("Training labels cross the model as-of date")

        self.train_df = train_df
        self.X_train = train_df[self.feature_cols].to_numpy(dtype=np.float32)
        self.y_train = train_df[target_col].to_numpy(dtype=np.float32)
        self.sample_weight = self.make_sample_weights(train_df["Date"], reference_date)
        self.as_of_date = reference_date

    def _prepare_data(self, data: pd.DataFrame, train: bool = True) -> pd.DataFrame:
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
            "target_risk_adjusted_return", "target_risk_adjusted_excess_return",
            "target_end_date",
            "target_profit", "tbm_return", "barrier_strength",
            "time_to_gain", "time_to_loss", "tp_return", "sl_return",
            "dollar_volume", "rolling_dollar_volume", "abs_bar_return", "target_beta_used",
            "ATR", "MACD_Hist", "OBV", "Treasury_30Y", "hour", "day_of_week", "month",
        }

        if train:
            reference_date = pd.Timestamp(data["Date"].max())
            data = data[data[self.config.target_column].notna()].copy()
            self.feature_cols = [c for c in data.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])]
            self.split_date = data["Date"].max()

            self.test_df = None
            self.set_training_frame(data, reference_date)

            self.X_test = None
            self.y_test = None

            log(f"Full training rows: {len(self.train_df):,}")
            log(f"Features: {len(self.feature_cols)}")
            log(f"Latest labelled date: {pd.Timestamp(self.split_date).strftime('%Y-%m-%d')}") # noqa

        else:
            if self.feature_cols is None: raise ValueError("feature_cols is not set.")
            for col in self.feature_cols:
                if col not in data.columns:
                    data[col] = 0

        return data

    def train_model(self, model_type: str):
        params_path = Path(ROOT_DIR) / "results" / f"{model_type.lower()}_params_{self.interval}.json"
        params = json.loads(params_path.read_text()) if params_path.exists() else {}

        if model_type == "CAT":
            if Settings.Threaded: params.update({"thread_count": -1})
            params.update({
                "loss_function": "RMSE",
                "allow_writing_files": False,
                "task_type": "GPU" if Settings.GPU["CAT"] else "CPU",
            })

            model = CatBoostRegressor(random_seed=self.seed, verbose=False, **params)

        elif model_type == "LGBM":
            if Settings.GPU["LGBM"]: params.update({"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
            if Settings.Threaded: params.update({"num_threads": -1, "n_jobs": -1})
            params.update({"objective": "regression"})

            model = LGBMRegressor(random_state=self.seed, verbose=-1, **params)

        else: raise ValueError(f"Unknown model type: {model_type}")

        model.fit(self.X_train, self.y_train, sample_weight=self.sample_weight)
        return model

    def save_models(self, results: dict):
        local_now = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
        model_folder = Path(MODEL_DIR) / "models" / f"{local_now}"
        predictions_folder = Path(MODEL_DIR) / "predictions"

        model_folder.mkdir(parents=True, exist_ok=True)
        predictions_folder.mkdir(parents=True, exist_ok=True)

        metadata = {
            "pipeline_version": MODEL_PIPELINE_VERSION,
            "training_date": local_now,
            "interval": self.interval,
            "config": asdict(self.config),
            "end_date": pd.Timestamp(self.split_date).strftime("%Y-%m-%d"),
            "as_of_date": pd.Timestamp(self.as_of_date).strftime("%Y-%m-%d"),
            "target_column": self.config.target_column,
            "feature_count": len(self.feature_cols),
            "feature_cols": list(self.feature_cols),
        }

        for model_type, model in results.items():
            if model_type == "LGBM":
                model.booster_.save_model(str(model_folder / "lgbm_model.txt"))
                joblib.dump(model, model_folder / "lgbm_model.joblib")
            elif model_type == "CAT":
                joblib.dump(model, model_folder / "cat_model.joblib")
            else:
                joblib.dump(model, model_folder / f"{model_type}_model.joblib")

        joblib.dump(self.feature_cols, model_folder / "features.joblib")

        metadata_path = model_folder / "metadata.json"
        metadata_path.write_text(json.dumps(json_safe(metadata), indent=4), encoding="utf-8")

    def load_models(self, load_model: str = "latest") -> dict:
        models_root = Path(MODEL_DIR) / "models"

        if load_model == "latest":
            model_folders = sorted([p for p in models_root.iterdir() if p.is_dir()])
            if not model_folders: raise FileNotFoundError(f"No saved models found in {models_root}")
            model_folder = model_folders[-1]
        else:
            model_folder = models_root / load_model
            if not model_folder.exists():
                raise FileNotFoundError(f"Model folder does not exist: {model_folder}")

        self.feature_cols = joblib.load(model_folder / "features.joblib")

        metadata_path = model_folder / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            pipeline_version = metadata.get("pipeline_version")
            if pipeline_version != MODEL_PIPELINE_VERSION:
                raise ValueError(
                    f"Model {model_folder.name} uses pipeline version {pipeline_version!r}; "
                    f"version {MODEL_PIPELINE_VERSION} is required. Retrain before paper prediction."
                )
            self.split_date = pd.Timestamp(metadata.get("end_date"))
            self.as_of_date = pd.Timestamp(metadata.get("as_of_date"))
        else:
            raise ValueError(
                f"Model {model_folder.name} has no metadata; retrain with pipeline "
                f"version {MODEL_PIPELINE_VERSION} before paper prediction."
            )

        models = {
            "LGBM": joblib.load(model_folder / "lgbm_model.joblib"),
            "CAT": joblib.load(model_folder / "cat_model.joblib"),
        }
        return models

    def add_ensemble_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        missing = {"Date", "ticker", "pred_cat", "pred_lgbm"} - set(df.columns)
        if missing: raise ValueError(f"Missing columns for ensemble: {missing}")

        df["cat_rank_pct"] = df.groupby("Date")["pred_cat"].rank(method="first", pct=True)
        df["lgbm_rank_pct"] = df.groupby("Date")["pred_lgbm"].rank(method="first", pct=True)
        df["vol_rank_pct"] = df.groupby("Date")["rolling_vol"].rank(method="average", pct=True)

        cat_weight = float(np.clip(self.config.ensemble_cat_weight, 0.0, 1.0))
        lgbm_weight = 1.0 - cat_weight
        df["pred_raw"] = cat_weight * df["pred_cat"] + lgbm_weight * df["pred_lgbm"]
        df["pred"] = df["pred_raw"]
        df["ensemble_agreement"] = 0
        df["risk_eligible"] = 0
        df["entry_eligible"] = 0
        df["ensemble_score"] = (
            cat_weight * df["cat_rank_pct"] + lgbm_weight * df["lgbm_rank_pct"]
        )
        df["model_dispersion"] = (df["pred_cat"] - df["pred_lgbm"]).abs()

        for date, group in df.groupby("Date"):
            top_n = min(self.config.ensemble_top_n, len(group))
            cat_top = set(group.nlargest(top_n, "pred_cat")["ticker"])
            lgbm_top = set(group.nlargest(top_n, "pred_lgbm")["ticker"])
            cat_bottom = set(group.nsmallest(top_n, "pred_cat")["ticker"])
            lgbm_bottom = set(group.nsmallest(top_n, "pred_lgbm")["ticker"])

            long_consensus = cat_top & lgbm_top
            short_consensus = cat_bottom & lgbm_bottom
            long_idx = group[
                group["ticker"].isin(long_consensus)
                & (group["pred_cat"] > self.config.min_abs_pred)
                & (group["pred_lgbm"] > self.config.min_abs_pred)
            ].index
            short_idx = group[
                group["ticker"].isin(short_consensus)
                & (group["pred_cat"] < -self.config.min_abs_pred)
                & (group["pred_lgbm"] < -self.config.min_abs_pred)
            ].index

            agreement_idx = long_idx.union(short_idx)
            df.loc[agreement_idx, "ensemble_agreement"] = 1

            risk_ok = (
                group["rolling_vol"].between(0.0, self.config.max_entry_rolling_vol, inclusive="right")
                & group["rolling_beta"].abs().le(self.config.max_entry_abs_beta)
                & df.loc[group.index, "vol_rank_pct"].le(self.config.max_entry_vol_percentile)
            )
            df.loc[group.index[risk_ok], "risk_eligible"] = 1
            eligible_idx = group.index[risk_ok].intersection(agreement_idx)
            df.loc[eligible_idx, "entry_eligible"] = 1

        df["profile_group"] = "ensemble"
        return df

    def predict_latest(
            self, data_dict: dict, models: dict, pred_data: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        if pred_data is None:
            log("Building latest prediction universe...")
            pred_data = self.datamanager.build_universe(self.interval, data_dict, drop_unlabelled=False)
        pred_data = self._prepare_data(pred_data, train=False)

        latest_date: pd.Timestamp = pred_data["Date"].max()
        if self.as_of_date is not None:
            max_age_days = (
                self.config.max_loaded_model_age_days_1h
                if self.interval == "1h"
                else self.config.max_loaded_model_age_days_1d
            )
            model_age = latest_date - pd.Timestamp(self.as_of_date)
            if model_age > pd.Timedelta(days=max_age_days):
                raise ValueError(
                    f"Loaded model is stale: as-of {pd.Timestamp(self.as_of_date):%Y-%m-%d}, "
                    f"prediction date {latest_date:%Y-%m-%d}, maximum age {max_age_days} days. "
                    "Retrain before producing paper orders."
                )
        latest = pred_data[pred_data["Date"] == latest_date].copy()

        X_latest = latest[self.feature_cols].to_numpy(dtype=np.float32)

        latest["pred_lgbm"] = models["LGBM"].predict(X_latest)
        latest["pred_cat"] = models["CAT"].predict(X_latest)

        latest = self.add_ensemble_predictions(latest)
        latest = self.datamanager.add_portfolio_weights(latest)
        latest = self.datamanager.add_signals(latest)

        picks = latest[
            (latest["ensemble_agreement"] == 1)
            & (latest["entry_eligible"] == 1)
            & (latest["pred_signal"] == 1)
        ].copy()
        picks = picks.sort_values("ensemble_score", ascending=False).head(self.config.max_top_tickers)

        picks["signal_date"] = latest_date
        picks["paper_action"] = "BUY_NEXT_OPEN"

        cols = [
            "signal_date", "ticker", "paper_action", "target_weight",
            "pred", "pred_raw", "pred_cat", "pred_lgbm",
            "ensemble_score", "cat_rank_pct", "lgbm_rank_pct",
            "risk_eligible", "entry_eligible", "regime_exposure",
            "position_risk_scale", "rolling_vol", "rolling_beta",
        ]
        picks = picks[[c for c in cols if c in picks.columns]]

        pred_dir = Path(MODEL_DIR) / "predictions"
        date_str = pd.Timestamp(latest_date).strftime("%Y-%m-%d")

        latest.to_parquet(pred_dir / f"all_predictions_{self.interval}_{date_str}.parquet", index=False)
        latest.to_parquet(pred_dir / "latest_all_predictions.parquet", index=False)

        both_positive = (latest["pred_cat"] > self.config.min_abs_pred) & (
            latest["pred_lgbm"] > self.config.min_abs_pred
        )
        diagnostics = {
            "Date": latest_date,
            "model_as_of_date": self.as_of_date,
            "universe_rows": len(latest),
            "pred_cat_positive": int((latest["pred_cat"] > self.config.min_abs_pred).sum()),
            "pred_lgbm_positive": int((latest["pred_lgbm"] > self.config.min_abs_pred).sum()),
            "both_models_positive": int(both_positive.sum()),
            "positive_agreement": int((both_positive & latest["ensemble_agreement"].eq(1)).sum()),
            "risk_eligible": int(latest["risk_eligible"].eq(1).sum()),
            "agreement_and_risk_eligible": int(latest["entry_eligible"].eq(1).sum()),
            "quantile_long_signals": int(latest["pred_signal"].eq(1).sum()),
            "final_candidates": len(picks),
            "regime_exposure": float(latest["regime_exposure"].median()),
            "median_pred_raw": float(latest["pred_raw"].median()),
            "max_pred_raw": float(latest["pred_raw"].max()),
        }
        (pred_dir / f"signal_diagnostics_{self.interval}_{date_str}.json").write_text(
            json.dumps(json_safe(diagnostics), indent=4), encoding="utf-8"
        )
        (pred_dir / "latest_signal_diagnostics.json").write_text(
            json.dumps(json_safe(diagnostics), indent=4), encoding="utf-8"
        )

        picks.to_csv(pred_dir / f"paper_signals_{self.interval}_{date_str}.csv", index=False)
        picks.to_csv(pred_dir / "latest_paper_signals.csv", index=False)

        self.update_paper_ledger(latest, picks)
        return picks

    def update_paper_ledger(self, latest: pd.DataFrame, picks: pd.DataFrame) -> None:
        paper_dir = Path(MODEL_DIR) / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)

        latest_date = pd.Timestamp(latest["Date"].max())
        date_str = latest_date.strftime("%Y-%m-%d")

        holdings_path = paper_dir / f"paper_holdings_{self.interval}.csv"
        orders_path = paper_dir / f"paper_orders_{self.interval}_{date_str}.csv"
        latest_orders_path = paper_dir / f"latest_paper_orders_{self.interval}.csv"

        holding_cols = [
            "ticker", "side", "entry_date", "age",
            "entry_score", "last_score", "last_pred",
            "last_pred_cat", "last_pred_lgbm", "last_seen_date",
            "target_weight", "last_regime_exposure",
        ]

        if holdings_path.exists():
            holdings = pd.read_csv(holdings_path)
        else:
            holdings = pd.DataFrame(columns=holding_cols)

        for col in holding_cols:
            if col not in holdings.columns:
                holdings[col] = np.nan

        holdings["ticker"] = holdings["ticker"].astype(str)
        holdings["side"] = pd.to_numeric(holdings["side"], errors="coerce").fillna(1).astype(int)
        holdings["age"] = pd.to_numeric(holdings["age"], errors="coerce").fillna(0).astype(int)

        # Recalculate exit signals with shorts allowed so existing longs can receive opposite signals.
        exit_latest = self.datamanager.add_signals(latest.copy(), allow_short=True)
        latest_by_ticker = exit_latest.set_index("ticker", drop=False)

        orders = []
        keep_rows = []

        for _, holding in holdings.iterrows():
            ticker = holding["ticker"]

            if ticker not in latest_by_ticker.index:
                orders.append({
                    "date": latest_date,
                    "ticker": ticker,
                    "action": "SELL_NEXT_OPEN",
                    "reason": "missing_from_latest_universe",
                })
                continue

            row = latest_by_ticker.loc[ticker]
            side = int(holding["side"])
            age = int(holding["age"]) + 1

            opposite_signal = int(row["pred_signal"]) == -side
            weak_long = side == 1 and float(row["pred"]) <= self.config.exit_pred_threshold
            weak_short = side == -1 and float(row["pred"]) >= -self.config.exit_pred_threshold
            too_old = age >= self.config.horizon

            if opposite_signal or weak_long or weak_short or too_old:
                reasons = []
                if opposite_signal: reasons.append("opposite_signal")
                if weak_long: reasons.append("weak_long")
                if weak_short: reasons.append("weak_short")
                if too_old: reasons.append("too_old")

                orders.append({
                    "date": latest_date,
                    "ticker": ticker,
                    "action": "SELL_NEXT_OPEN" if side == 1 else "COVER_NEXT_OPEN",
                    "reason": ",".join(reasons),
                    "pred": float(row["pred"]),
                    "pred_signal": int(row["pred_signal"]),
                    "age": age,
                })
                continue

            holding = holding.copy()
            holding["age"] = age
            holding["last_score"] = float(row["ensemble_score"])
            holding["last_pred"] = float(row["pred"])
            holding["last_pred_cat"] = float(row["pred_cat"])
            holding["last_pred_lgbm"] = float(row["pred_lgbm"])
            holding["last_seen_date"] = latest_date
            holding["target_weight"] = float(row["target_weight"])
            holding["last_regime_exposure"] = float(row["regime_exposure"])

            keep_rows.append(holding.to_dict())

            orders.append({
                "date": latest_date,
                "ticker": ticker,
                "action": "HOLD",
                "reason": "still_valid",
                "pred": float(row["pred"]),
                "pred_signal": int(row["pred_signal"]),
                "age": age,
                "target_weight": float(row["target_weight"]),
                "regime_exposure": float(row["regime_exposure"]),
            })

        new_holdings = pd.DataFrame(keep_rows, columns=holding_cols)
        held_tickers = set(new_holdings["ticker"]) if not new_holdings.empty else set()

        # Add only genuinely new buy candidates. Repeated buy signals become HOLD, not more buying.
        for _, row in picks.iterrows():
            ticker = str(row["ticker"])

            if ticker in held_tickers:
                continue

            if len(new_holdings) >= self.config.max_top_tickers:
                orders.append({
                    "date": latest_date,
                    "ticker": ticker,
                    "action": "SKIP_BUY_FULL",
                    "reason": "portfolio_full",
                    "pred": float(row["pred"]),
                })
                continue

            new_row = {
                "ticker": ticker,
                "side": 1,
                "entry_date": latest_date,
                "age": 0,
                "entry_score": float(row["ensemble_score"]),
                "last_score": float(row["ensemble_score"]),
                "last_pred": float(row["pred"]),
                "last_pred_cat": float(row["pred_cat"]),
                "last_pred_lgbm": float(row["pred_lgbm"]),
                "last_seen_date": latest_date,
                "target_weight": float(row["target_weight"]),
                "last_regime_exposure": float(row["regime_exposure"]),
            }

            new_holdings = pd.concat([new_holdings, pd.DataFrame([new_row])], ignore_index=True)
            held_tickers.add(ticker)

            orders.append({
                "date": latest_date,
                "ticker": ticker,
                "action": "BUY_NEXT_OPEN",
                "reason": "new_buy_signal",
                "pred": float(row["pred"]),
                "target_weight": float(row["target_weight"]),
                "regime_exposure": float(row["regime_exposure"]),
            })

        new_holdings.to_csv(holdings_path, index=False)

        orders_df = pd.DataFrame(orders)
        orders_df.to_csv(orders_path, index=False)
        orders_df.to_csv(latest_orders_path, index=False)

    def run_pipeline(self, load_model=None) -> bool:
        if self.interval == "1d":
            self.config.horizon = 40
            self.config.max_top_tickers = 30
        elif self.interval == "1h":
            self.config.horizon = 30
            self.config.max_top_tickers = 10

        log("Loading raw data...")
        data_dict = self.datamanager.load_raw_data(self.interval)

        if load_model is not None:
            log("Loading models...")
            models = self.load_models(load_model)
            prediction_data = None

        else:
            log("Building universe dataframe...")
            data = self.datamanager.build_universe(
                self.interval, data_dict, drop_unlabelled=False
            )

            log("Preparing pooled features...")
            self._prepare_data(data)
            prediction_data = data

            models = {}

            log("Training LightGBM...")
            models["LGBM"] = self.train_model("LGBM")
            flush_memory()

            log("Training CatBoost...")
            models["CAT"] = self.train_model("CAT")
            flush_memory()

            log("Saving assets...")
            self.save_models(models)

        self.predict_latest(data_dict, models, prediction_data)
        return True

########################################################################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run purged stock-model walk-forward tests")
    parser.add_argument("--months", nargs="+", type=int, default=[3,6,12])
    parser.add_argument(
        "--retrain-bars", type=int, default=None,
        help="Optional refit cadence in signal bars; omit for one anchored fit",
    )
    parser.add_argument(
        "--anchored", action="store_true",
        help="Use one model fitted at the initial cutoff (quick diagnostic)",
    )
    args = parser.parse_args()

    start = time.perf_counter()

    # print("Training...")
    # mng = Predictor("1d")
    # mng.run_pipeline()

    trainer = Trainer("1d")
    cadence = 0 if args.anchored else args.retrain_bars
    for mon in args.months:
        trainer.run_training(months_back=mon, retrain_every_n_bars=cadence)

    print(f"Total time: {time.perf_counter() - start:.1f}s")
