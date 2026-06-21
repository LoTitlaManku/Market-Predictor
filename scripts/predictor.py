
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

    def add_forward_excess_target(self, df: pd.DataFrame, benchmark_close: pd.Series, drop_unlabelled: bool = True) -> pd.DataFrame:
        df = df.copy()
        beta = df["rolling_beta"]
        close = df["Adj Close"]
        benchmark_close = benchmark_close.reindex(df.index).ffill()

        df["future_return"] = close.shift(-self.config.horizon) / close - 1.0
        df["benchmark_future_return"] = (benchmark_close.shift(-self.config.horizon) / benchmark_close - 1.0)

        if isinstance(beta, pd.Series):
            beta_used = beta.reindex(df.index).ffill()
            beta_used = beta_used.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        else:
            beta_used = 1.0 if not np.isfinite(beta) else beta

        df["target_beta_used"] = beta_used
        df["target_excess_return"] = (df["future_return"] - beta_used * df["benchmark_future_return"])

        return df.dropna(subset=["target_excess_return"]) if drop_unlabelled else df

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
                data = data[~data.index.duplicated(keep="first")].sort_index()

                data_dict[ticker] = data

            except Exception as e:
                log(f"Skipping raw load {ticker}: {type(e).__name__}: {e}", prints=False)

        log("")
        if not data_dict: raise ValueError("No raw universe data loaded.")
        return data_dict

    def build_universe(self, interval: str, data_dict: dict, drop_unlabelled: bool = True, cutoff_date: pd.Timestamp | None = None) -> pd.DataFrame:
        with open(os.path.join(DATA_DIR, "ticker_attr.json"), "r") as f:
            ticker_map = json.load(f)

        benchmark_raw = pd.read_parquet(os.path.join(DATA_DIR, f"SPY_{interval}.parquet"))
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
                if self.config.min_profile_adv > max(0, float(adv)) and np.isfinite(adv): continue

                df = data.ind.add_indicators(ticker, interval, add_targets=False)

                df = self._add_liquidity_columns(df, interval)
                df = self._add_rolling_risk_columns(df, benchmark_close, interval)

                df = self.add_forward_excess_target(df, benchmark_close, drop_unlabelled)
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

        if drop_unlabelled:
            data = data.dropna()
        else:
            target_cols = {"future_return", "benchmark_future_return", "target_excess_return"}
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

class Trainer:
    def __init__(self, interval: str, config: UniverseConfig | None = None):
        self.config = config if config else UniverseConfig()
        self.predictor = Predictor(interval, self.config)
        self.manager = DataManager(self.config)
        self.interval = interval

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
        half_spread = 1.0 / 20_000
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

            g["long_entry_price"] = g["exec_entry_price_mid"] * (1.0 + half_spread)
            g["long_exit_price"] = g["exec_exit_price_mid"] * (1.0 - half_spread)
            g["long_exec_return"] = g["long_exit_price"] / g["long_entry_price"] - 1.0

            g["short_entry_price"] = g["exec_entry_price_mid"] * (1.0 - half_spread)
            g["short_exit_price"] = g["exec_exit_price_mid"] * (1.0 + half_spread)
            g["short_exec_return"] = g["short_entry_price"] / g["short_exit_price"] - 1.0

            g["exec_return"] = g["long_exec_return"]

            frames.append(g)

        log("")
        if not frames: raise ValueError("No execution returns could be calculated.")
        return pd.concat(frames, axis=0, ignore_index=True).sort_values(["Date", "ticker"])

    def prepare_predictor(self, train_data: pd.DataFrame, full_data: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
        cutoff_date: pd.Timestamp = pd.Timestamp(cutoff_date)  # noqa

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

        self.predictor.feature_cols = [c for c in data.columns if
                                c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])]

        train_df = data[(data["Date"] <= cutoff_date) & data["target_excess_return"].notna()].copy()
        walk_df = data[data["Date"] > cutoff_date].copy()

        if train_df.empty: raise ValueError("No labelled training rows before cutoff date.")
        if walk_df.empty: raise ValueError("No walk-forward rows after cutoff date.")

        self.predictor.split_date = cutoff_date
        self.predictor.train_df = train_df
        self.predictor.test_df = walk_df

        self.predictor.X_train = train_df[self.predictor.feature_cols].to_numpy(dtype=np.float32)
        self.predictor.y_train = train_df["target_excess_return"].to_numpy(dtype=np.float32)

        log(f"Walk-forward cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
        log(f"Train rows: {len(train_df):,}")
        log(f"Walk-forward rows: {len(walk_df):,}")
        log(f"Features: {len(self.predictor.feature_cols)}")

        return walk_df

    def save_results(
            self, months_back: int | float, summary: dict, daily_df: pd.DataFrame,
            trades_df: pd.DataFrame, position_df: pd.DataFrame | None = None, attribution: dict | None = None
    ) -> Path:

        folder = Path(MODEL_DIR) / f"{self.interval} Model [{now}]" / "walk_forward" / str(months_back)
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

    def run_training(self, months_back: int | float) -> dict[str, Any]:
        if self.interval == "1d":
            self.config.horizon = 40
            self.config.max_top_tickers = 30
        elif self.interval == "1h":
            self.config.horizon = 30
            self.config.max_top_tickers = 10

        model_type = "ENSEMBLE"
        log(f"{'=' * 50}\nWALK-FORWARD BACKTEST ({self.interval}): cutoff = {months_back} months\n{'=' * 50}")

        log("Loading all tickers...")
        data_dict = self.manager.load_raw_data(self.interval)

        latest_date = pd.Timestamp(next(iter(data_dict.values())).index.max())
        if isinstance(months_back, int):
            cutoff_date: pd.Timestamp = latest_date - pd.DateOffset(months=months_back)  # noqa
        else:
            cutoff_date: pd.Timestamp = latest_date - pd.DateOffset(days=round(months_back * 30))  # noqa

        log(f"Latest available date: {latest_date.strftime('%Y-%m-%d')}")
        log(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")

        # Construct training universe up to the split date
        log("Building training universe up to cutoff...")
        train_data = self.manager.build_universe(self.interval, data_dict, True, cutoff_date)

        # Construct full evaluation dataset across entire timeline
        log("Building full universe for walk-forward predictions...")
        full_data = self.manager.build_universe(self.interval, data_dict, False)

        walk_df = self.prepare_predictor(train_data, full_data, cutoff_date)

        # Slice features for walk-forward out-of-sample predictions
        self.config.max_top_tickers = 20
        X_walk = walk_df[self.predictor.feature_cols].to_numpy(dtype=np.float32)
        walk_df = walk_df.copy()

        log("Training CAT up to cutoff...")
        cat_model = self.predictor.train_model("CAT")

        log("Training LGBM up to cutoff...")
        lgbm_model = self.predictor.train_model("LGBM")

        # Generate predictions across the walk-forward window
        walk_df["pred_cat"] = cat_model.predict(X_walk)
        walk_df["pred_lgbm"] = lgbm_model.predict(X_walk)

        # Merge model scores and calculate execution returns
        walk_df = self.predictor.add_ensemble_predictions(walk_df)
        walk_df = self.add_returns(walk_df, data_dict)

        dates = list(sorted(pd.to_datetime(walk_df["Date"]).unique()))
        by_date = {date: group.copy() for date, group in walk_df.groupby("Date", sort=True)}

        holdings: dict[str, dict[str, Any]] = {}
        equity = 1000.0

        daily_rows = []
        trade_rows = []
        position_rows = []
        for date in dates[:-1]:
            # Get predictions/returns for this date and drop invalid records
            day = by_date[date].copy()
            day = day.dropna(subset=["pred", "exec_return", "exec_entry_price_mid", "exec_exit_price_mid"])
            if day.empty: continue

            # Build signal table for the day
            exit_signal_day = self.manager.add_signals(day, allow_short=True).set_index("ticker", drop=False)
            signalled = self.manager.add_signals(day)
            candidates = signalled[signalled["pred_signal"] != 0].copy()

            # Sort entry candidates based on directional confidence strength
            if not candidates.empty:
                candidates["score"] = np.where(candidates["pred_signal"] == 1, candidates["pred"], -candidates["pred"])
                candidates = candidates.sort_values("score", ascending=False)

            turnover = 0

            # Scan and manage open portfolio positions
            for ticker in list(holdings.keys()):
                # Liquidate if ticker drops out of the target universe
                if ticker not in exit_signal_day.index:
                    holdings.pop(ticker)
                    turnover += 1
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

                    trade_rows.append({
                        "Date": date, "ticker": ticker, "action": "BUY" if side == 1 else "SHORT",
                        "pred": float(row["pred"]), "score": score,
                    })
                    continue

                # If full, replace the weakest holding
                weakest_ticker = min(holdings, key=lambda t: holdings[t]["score"])
                weakest_score = holdings[weakest_ticker]["score"]
                if score > weakest_score + self.config.replace_buffer:
                    holdings.pop(weakest_ticker)
                    holdings[ticker] = {"side": side, "score": score, "age": 0}
                    turnover += 2

                    trade_rows.append({"Date": date, "ticker": weakest_ticker, "action": "REPLACE_EXIT"})
                    trade_rows.append({
                        "Date": date, "ticker": ticker, "action": "BUY" if side == 1 else "SHORT",
                        "pred": float(row["pred"]), "score": score,
                    })

            day_indexed = day.set_index("ticker", drop=False)
            position_returns = []
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

                position_returns.append(raw_return)

                position_details.append({
                    "signal_date": date, "exec_entry_date": row["exec_entry_date"],
                    "exec_exit_date": row["exec_exit_date"],
                    "ticker": ticker, "side": side, "entry_price": entry_price, "exit_price": exit_price,
                    "position_return": raw_return, "holding_age": int(info["age"]), "score": float(info["score"]),
                })

            # Equal-weight portfolio return for the day minus cost overhead
            gross_return = float(np.mean(position_returns)) if position_returns else 0.0
            turnover_cost = (self.config.cost_bps / 10_000) * turnover / max(1, self.config.max_top_tickers)
            daily_return = gross_return - turnover_cost

            # Compute and apply compounded growth changes to total capital
            equity_before = equity
            equity *= 1.0 + daily_return

            # Append performance context to position details tracking log
            if position_details:
                weight = 1.0 / len(position_details)

                for row in position_details:
                    contribution_return = row["position_return"] * weight
                    row["weight"] = weight
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
                "held_tickers": ",".join(sorted(holdings.keys())),
            })

            # Increment active position lifetime counters
            for info in holdings.values(): info["age"] += 1

        # Consolidate matrix output logs into pandas dataframes
        daily_df = pd.DataFrame(daily_rows)
        trades_df = pd.DataFrame(trade_rows)
        position_df = pd.DataFrame(position_rows)

        if daily_df.empty: raise ValueError("Walk-forward produced no daily rows.")

        # Sort aggregated performance results sequentially by timestamp
        daily_df["Date"] = pd.to_datetime(daily_df["Date"])
        daily_df = daily_df.sort_values("Date")

        # Compute max drawdown curves and overall backtest performance
        returns = daily_df["daily_return"].astype(float)
        equity_curve = daily_df["equity"].astype(float)
        drawdown = equity_curve / equity_curve.cummax() - 1.0
        total_return = equity / 1000.0 - 1.0

        # Configure time normalization metrics matching execution intervals
        multiplier = 6.5 if self.interval == "1h" else 1
        years = len(daily_df) / (252 * multiplier)
        annualiser = np.sqrt(252 * multiplier)

        # Generate standardized portfolio tracking analytics
        cagr_like = (equity / 1000.0) ** (1 / years) - 1.0 if years > 0 else 0.0
        sharpe_like = returns.mean() / (returns.std() + 1e-9) * annualiser  # noqa
        attribution = self.analyse_pos_attr(position_df)

        # Build output metadata dict package
        summary = {
            "interval": self.interval,
            "model_type": model_type,
            "months_back": months_back,
            "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
            "start_date": daily_df["Date"].min().strftime("%Y-%m-%d"),  # noqa
            "end_date": daily_df["Date"].max().strftime("%Y-%m-%d"),  # noqa
            "initial_capital": 1000.0,
            "final_equity": float(equity),
            "profit_loss": float(equity - 1000.0),
            "total_return": float(total_return),
            "cagr_like": float(cagr_like),
            "sharpe_like": float(sharpe_like),
            "max_drawdown": float(drawdown.min()),
            "win_rate": (returns > 0).mean(),
            "mean_daily_return": returns.mean(),
            "median_daily_return": returns.median(),
            "avg_holdings": float(daily_df["holdings"].mean()),
            "avg_turnover": float(daily_df["turnover"].mean()),
            "trade_count": int(len(trades_df)),
            "attribution_concentration": attribution["concentration"],
        }

        # Save serialized simulation logs directly to workspace paths
        self.save_results(months_back, summary, daily_df, trades_df, position_df, attribution)

        log("Walk-forward summary:")
        log(json.dumps(json_safe(summary), indent=4))
        return {"summary": summary, "daily": daily_df, "trades": trades_df}

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
            "target_profit", "tbm_return", "barrier_strength",
            "time_to_gain", "time_to_loss", "tp_return", "sl_return",
            "dollar_volume", "rolling_dollar_volume", "abs_bar_return", "target_beta_used"
        }

        if train:
            data = data[data["target_excess_return"].notna()].copy()
            self.feature_cols = [c for c in data.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])]
            self.split_date = data["Date"].max()

            self.train_df = data.copy()
            self.test_df = None

            self.X_train = self.train_df[self.feature_cols].to_numpy(dtype=np.float32)
            self.y_train = self.train_df["target_excess_return"].to_numpy(dtype=np.float32) # noqa

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

        model.fit(self.X_train, self.y_train)
        return model

    def save_models(self, results: dict):
        local_now = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
        model_folder = Path(MODEL_DIR) / "models" / f"{local_now}"
        predictions_folder = Path(MODEL_DIR) / "predictions"

        model_folder.mkdir(parents=True, exist_ok=True)
        predictions_folder.mkdir(parents=True, exist_ok=True)

        metadata = {
            "training_date": local_now,
            "interval": self.interval,
            "config": asdict(self.config),
            "end_date": pd.Timestamp(self.split_date).strftime("%Y-%m-%d"),
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
            self.split_date = pd.Timestamp(metadata.get("end_date"))

        models = {
            "LGBM": joblib.load(model_folder / "lgbm_model.joblib"),
            "CAT": joblib.load(model_folder / "cat_model.joblib"),
        }
        return models

    @staticmethod
    def add_ensemble_predictions(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        missing = {"Date", "ticker", "pred_cat", "pred_lgbm"} - set(df.columns)
        if missing: raise ValueError(f"Missing columns for ensemble: {missing}")

        df["cat_rank_pct"] = df.groupby("Date")["pred_cat"].rank(method="first", pct=True)
        df["lgbm_rank_pct"] = df.groupby("Date")["pred_lgbm"].rank(method="first", pct=True)

        df["pred"] = 0.0
        df["ensemble_agreement"] = 0
        df["ensemble_score"] = 0.0
        for date, group in df.groupby("Date"):
            cat_top = set(group.nlargest(min(50, len(group)), "pred_cat")["ticker"])
            lgbm_top = set(group.nlargest(min(50, len(group)), "pred_lgbm")["ticker"])

            consensus = cat_top & lgbm_top
            if not consensus: continue

            idx = group[group["ticker"].isin(consensus)].index
            score = 0.6 * df.loc[idx, "cat_rank_pct"] + 0.4 * df.loc[idx, "lgbm_rank_pct"]

            df.loc[idx, "ensemble_agreement"] = 1
            df.loc[idx, "ensemble_score"] = score
            df.loc[idx, "pred"] = score

        df["profile_group"] = "ensemble"
        return df

    def predict_latest(self, data_dict: dict, models: dict) -> pd.DataFrame:
        log("Building latest prediction universe...")
        pred_data = self.datamanager.build_universe(self.interval, data_dict, drop_unlabelled=False)
        pred_data = self._prepare_data(pred_data, train=False)

        latest_date: pd.Timestamp = pred_data["Date"].max()
        latest = pred_data[pred_data["Date"] == latest_date].copy()

        X_latest = latest[self.feature_cols].to_numpy(dtype=np.float32)

        latest["pred_lgbm"] = models["LGBM"].predict(X_latest)
        latest["pred_cat"] = models["CAT"].predict(X_latest)

        latest = self.add_ensemble_predictions(latest)
        latest = self.datamanager.add_signals(latest)

        picks = latest[(latest["ensemble_agreement"] == 1) & (latest["pred_signal"] == 1)].copy()
        picks = picks.sort_values("pred", ascending=False).head(self.config.max_top_tickers)

        picks["signal_date"] = latest_date
        picks["paper_action"] = "BUY_NEXT_OPEN"
        picks["target_weight"] = 1.0 / len(picks) if len(picks) else 0.0

        cols = [
            "signal_date", "ticker", "paper_action", "target_weight",
            "pred", "pred_cat", "pred_lgbm",
            "ensemble_score", "cat_rank_pct", "lgbm_rank_pct",
        ]
        picks = picks[[c for c in cols if c in picks.columns]]

        pred_dir = Path(MODEL_DIR) / "predictions"
        date_str = pd.Timestamp(latest_date).strftime("%Y-%m-%d")

        latest.to_parquet(pred_dir / f"all_predictions_{self.interval}_{date_str}.parquet", index=False)
        latest.to_parquet(pred_dir / "latest_all_predictions.parquet", index=False)

        picks.to_csv(pred_dir / f"paper_signals_{self.interval}_{date_str}.csv", index=False)
        picks.to_csv(pred_dir / "latest_paper_signals.csv", index=False)

        log(f"Saved all predictions to: {pred_dir / f'all_predictions_{self.interval}_{date_str}.parquet'}")
        log(f"Saved paper signals to: {pred_dir / f'paper_signals_{self.interval}_{date_str}.csv'}")

        log("Paper trade picks:")
        log(picks.to_string(index=False))

        return picks

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
            models = self.load_models(load_model)

        else:
            log("Building universe dataframe...")
            data = self.datamanager.build_universe(self.interval, data_dict)

            log("Preparing pooled features...")
            self._prepare_data(data)

            models = {}

            log("Training LightGBM...")
            models["LGBM"] = self.train_model("LGBM")
            flush_memory()

            log("Training CatBoost...")
            models["CAT"] = self.train_model("CAT")
            flush_memory()

            log("Saving assets...")
            self.save_models(models)

        self.predict_latest(data_dict, models)
        return True

########################################################################################################################

if __name__ == "__main__":
    start = time.perf_counter()

    # print("Training...")
    # mng = Predictor("1d")
    # mng.run_pipeline()

    # trainer = Trainer("1d")
    # for mon in [3,6,12]:
    #     trainer.run_training(months_back=mon)

    print(f"Total time: {time.perf_counter() - start:.1f}s")