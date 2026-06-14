
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

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.nn import GRU

from scripts.config import DATA_DIR, GROUP_DIR, MODEL_DIR
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
    horizon: int = 20
    top_q: float = 0.90
    bottom_q: float = 0.10
    min_abs_pred: float = 0.003
    allow_short: bool = True
    cost_bps: float = 10.0

def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    @staticmethod
    def add_forward_excess_target(df: pd.DataFrame, benchmark_close: pd.Series, horizon: int) -> pd.DataFrame:
        df = df.copy()

        close = df["Adj Close"]
        benchmark_close = benchmark_close.reindex(df.index).ffill()

        df["future_return"] = close.shift(-horizon) / close - 1.0
        df["benchmark_future_return"] = benchmark_close.shift(-horizon) / benchmark_close - 1.0
        df["target_excess_return"] = df["future_return"] - df["benchmark_future_return"]

        return df.dropna(subset=["target_excess_return"])

    def _build_universe_frame(self, interval: str, group: str) -> pd.DataFrame:
        with open(os.path.join(GROUP_DIR, f"Profile {group}", "tickers.json"), "r") as f:
            self.ticker_map = json.load(f)

        benchmark_raw = pd.read_parquet(os.path.join(GROUP_DIR, f"Profile {group}", f"benchmark_{interval}.parquet"))
        benchmark_raw.index.name = "Date"
        benchmark_raw.index = pd.to_datetime(benchmark_raw.index, utc=True).tz_localize(None)
        benchmark_raw = benchmark_raw[~benchmark_raw.index.duplicated(keep="first")].sort_index()
        benchmark_close = benchmark_raw["Adj Close"]

        frames = []
        for ticker, meta in self.ticker_map.items():
            try:
                raw = load_data(ticker, interval)
                if raw is None or raw.empty:
                    print(f"Skipping {ticker}: no data")
                    continue

                df = raw.ind.add_indicators(ticker, interval, add_targets=False)
                df = self.add_forward_excess_target(df, benchmark_close, self.config.horizon)

                df["ticker"] = ticker
                df["profile"] = f"Profile {group}"
                df["profile_vol"] = meta["vol"]
                df["profile_beta"] = meta["beta"]
                df["profile_adv_log"] = np.log1p(meta["adv"]) if np.isfinite(meta["adv"]) else np.nan
                df["Date"] = df.index

                frames.append(df)

            except Exception as e:
                print(f"Skipping {ticker}: {e}")

        if not frames:
            raise ValueError("No usable universe data")

        data = pd.concat(frames, axis=0)
        data = data.sort_values(["Date", "ticker"])
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()

        return data

    def _prepare_data(self, data: pd.DataFrame) -> None:
        data = data.copy()
        data["profile_group"] = data["profile"]
        data = pd.get_dummies(data, columns=["profile"], prefix="profile", dtype=int)

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

        self.X_train = self.train_df[self.feature_cols].values
        self.X_test = self.test_df[self.feature_cols].values

        self.y_train = self.train_df["target_excess_return"].values.astype(np.float32)
        self.y_test = self.test_df["target_excess_return"].values.astype(np.float32)

        print(f"Features: {len(self.feature_cols)}")
        print(f"Train rows: {len(self.train_df):,}")
        print(f"Test rows: {len(self.test_df):,}")
        print(f"Split date: {pd.Timestamp(self.split_date).strftime('%Y-%m-%d')}")

    def add_cross_sectional_signals(self, df: pd.DataFrame, pred_col: str, group_cols: list) -> pd.DataFrame:
        df = df.copy()
        df["pred_signal"] = 0

        if group_cols is None:
            group_cols = ["Date", "profile_group"]

        for _, group in df.groupby(group_cols):
            if len(group) < 5:
                continue

            upper = group[pred_col].quantile(self.config.top_q)
            lower = group[pred_col].quantile(self.config.bottom_q)

            long_mask = (group[pred_col] >= upper) & (group[pred_col].abs() >= self.config.min_abs_pred)
            df.loc[group.index[long_mask], "pred_signal"] = 1

            if self.config.allow_short:
                short_mask = (group[pred_col] <= lower) & (group[pred_col].abs() >= self.config.min_abs_pred)
                df.loc[group.index[short_mask], "pred_signal"] = -1

        return df

    def _apply_signals(self, test_df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
        out = test_df.copy()
        out["pred"] = pred

        out = self.add_cross_sectional_signals(out, "pred", ["Date", "profile_group"])

        return out

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

    def _train_lightgbm(self, hyperparams: dict) -> dict[str, Any]:
        params = self._get_lgbm_params(hyperparams)
        model = LGBMRegressor(random_state=self.seed, verbose=-1, **params)

        model.fit(self.X_train, self.y_train)

        pred = model.predict(self.X_test)
        scored_df = self._apply_signals(self.test_df, pred)

        strategy_metrics = self.evaluate_strategy(scored_df)

        return {
            "type": "LGBM",
            "model": model,
            "test_df": scored_df,
            "mae": float(mean_absolute_error(self.y_test, pred)),
            "rmse": float(mean_squared_error(self.y_test, pred) ** 0.5),
            **strategy_metrics,
        }

    def _train_catboost(self, hyperparams: dict) -> dict[str, Any]:
        params = self._get_cat_params(hyperparams)
        model = CatBoostRegressor(random_seed=self.seed, verbose=False, **params)

        model.fit(self.X_train, self.y_train)

        pred = model.predict(self.X_test)
        scored_df = self._apply_signals(self.test_df, pred)

        strategy_metrics = self.evaluate_strategy(scored_df)

        return {
            "type": "CAT",
            "model": model,
            "test_df": scored_df,
            "mae": float(mean_absolute_error(self.y_test, pred)),
            "rmse": float(mean_squared_error(self.y_test, pred) ** 0.5),
            **strategy_metrics,
        }

    def _save_model_assets(self, group, interval, results: dict, baselines: dict):
        save_folder = Path(os.path.join(MODEL_DIR, f"Profile {group} ({interval})"))
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
                if key not in {"model", "test_df"}
            }

            if model_type == "LGBM":
                model_results["model"].booster_.save_model(str(save_folder / "lgbm_model.txt"))
            elif model_type == "CAT":
                joblib.dump(model_results["model"], save_folder / "cat_model.joblib")

            model_results["test_df"].to_parquet(save_folder / f"{model_type.lower()}_test_predictions.parquet", index=False)

        joblib.dump(self.feature_cols, save_folder / "features.joblib")
        (save_folder / "metadata.json").write_text(json.dumps(metadata, indent=4), encoding="utf-8")

    @staticmethod
    def rank_latest_predictions(scored_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        latest_date = scored_df["Date"].max()
        latest = scored_df[scored_df["Date"] == latest_date].copy()

        cols = [
            "Date", "ticker", "profile_group", "pred", "pred_signal",
            "target_excess_return", "future_return", "benchmark_future_return",
        ]

        cols = [c for c in cols if c in latest.columns]
        latest = latest[cols].sort_values("pred", ascending=False)

        return pd.concat([latest.head(top_n), latest.tail(top_n)], axis=0)

    def run_training_pipeline(self, group, interval, status_signal: tuple | None = None, force_train: bool = True) -> bool:
        def log_update(msg, force_print=False):
            if status_signal:
                u_queue, core_key = status_signal
                u_queue.put((core_key, {"Current Task": msg}))
                if force_print:
                    print(msg)
            elif Settings.LOGGING or force_print:
                print(msg)

        hyperparams = {}

        save_folder = os.path.join(MODEL_DIR, f"Profile {group} ({interval})")
        if all_model_assets_exist(save_folder) and not force_train:
            log_update(f"Universe model already trained: {save_folder}", True)
            return True

        if os.path.exists(save_folder): shutil.rmtree(save_folder)

        t0 = time.perf_counter()

        log_update("Building universe dataframe...", True)
        data = self._build_universe_frame(interval, group)

        log_update("Preparing pooled features...", True)
        self._prepare_data(data)

        baselines = self.evaluate_baselines()
        print("Baselines:")
        print(json.dumps(baselines, indent=4))

        results = {}

        log_update("Training LightGBM...", True)
        results["LGBM"] = self._train_lightgbm(hyperparams)
        flush_memory()
        print(json.dumps({k: v for k, v in results["LGBM"].items() if k not in {"model", "test_df"}}, indent=4))

        log_update("Training CatBoost...", True)
        results["CAT"] = self._train_catboost(hyperparams)
        flush_memory()
        print(json.dumps({k: v for k, v in results["CAT"].items() if k not in {"model", "test_df"}}, indent=4))

        log_update("Saving assets...", True)
        self._save_model_assets(group, interval, results, baselines)

        best_model_type = max(results, key=lambda m: results[m].get("sharpe_like", -999))
        latest = self.rank_latest_predictions(results[best_model_type]["test_df"], top_n=20)

        print(f"Best model by sharpe_like: {best_model_type}")
        print("Latest ranked predictions:")
        print(latest.to_string(index=False))
        print(f"Total time: {time.perf_counter() - t0:.1f}s")

        return True


def all_model_assets_exist(model_path: str | os.PathLike) -> bool:
    root = Path(model_path)
    required_files = ["metadata.json", "features.joblib", "lgbm_model.txt", "cat_model.joblib"]

    return root.exists() and all((root / f).exists() and (root / f).stat().st_size > 0 for f in required_files)


if __name__ in "__main__":
    start = time.perf_counter()

    print("Training...")
    success = TrainingManager().run_training_pipeline("A", "1d", force_train=True)
    # print(success)
    # print("Predicting...")
    # run_prediction_pipeline("AAPL", "1d")

    print(time.perf_counter() - start)