
# Standard library imports
import gc
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# External library imports
import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Custom imports
from scripts.data_management import load_data
from scripts.config import MODEL_DIR, DATA_DIR, HYPER_DIR
import scripts.indicators  # noqa: F401

warnings.filterwarnings("ignore")


class Settings:
    VERBOSE = 0
    LOGGING = False
    GPU = {"LGBM": False, "CAT": False, "LSTM": True}
    Threaded = False


def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


########################################################################################################################

@dataclass
class Trainer:
    def __init__(self):
        self.seed = 69
        self.__test_size = 0.2

        top_q: float = 0.90
        bottom_q: float = 0.10
        min_abs_pred: float = 0.003
        allow_short: bool = True
        cost_bps: float = 10.0

    @staticmethod
    def assign_profile(vol: float, beta: float, adv: float) -> str:
        if vol < 0.18:
            return "A"

        if 0.18 <= vol < 0.28:
            if beta < 0.9:
                return "E"
            return "B" if adv > 5_000_000 else "F"

        if 0.28 <= vol < 0.40:
            return "C"

        return "D"

    def build_profile_map(self, tickers: list, interval: str, lookback_rows: int = 504, min_rows: int = 252,) -> dict[str, dict[str, float | str]]:
        spy_df = pd.read_parquet(os.path.join(DATA_DIR, f'SPY_{interval}.parquet'))
        if spy_df is None or spy_df.empty:
            raise ValueError(f"Not enough data for SPY")

        spy_returns = spy_df["Adj Close"].pct_change().rename("benchmark_return")
        profile_map: dict[str, dict[str, float | str]] = {}

        for ticker in tickers:
            try:
                df = load_data(ticker, interval)

                if df is None or df.empty or len(df) < min_rows:
                    print(f"Skipping {ticker}: not enough data")
                    continue

                returns = df["Adj Close"].pct_change().rename("stock_return")
                aligned = pd.concat([returns, spy_returns], axis=1).dropna().tail(lookback_rows)

                if len(aligned) < min_rows:
                    print(f"Skipping {ticker}: not enough aligned benchmark rows")
                    continue

                vol = aligned["stock_return"].std() * np.sqrt(252)
                beta = aligned["stock_return"].cov(aligned["benchmark_return"])
                beta /= aligned["benchmark_return"].var() + 1e-12
                adv = df["Volume"].tail(252).mean()

                if not np.isfinite(vol) or not np.isfinite(beta) or not np.isfinite(adv):
                    print(f"Skipping {ticker}: non-finite profile values")
                    continue

                profile_map[ticker] = {
                    "profile": self.assign_profile(float(vol), float(beta), float(adv)),
                    "vol": float(vol),
                    "beta": float(beta),
                    "adv": float(adv),
                }

            except Exception as e:
                print(f"Skipping {ticker}: {e}")

        return profile_map

    @staticmethod
    def add_forward_excess_target(df: pd.DataFrame, benchmark_close: pd.Series, horizon: int = 5) -> pd.DataFrame:
        df = df.copy()

        close = df["Adj Close"]
        bench = benchmark_close.reindex(df.index).ffill()

        df["future_return"] = close.shift(-horizon) / close - 1.0
        df["benchmark_future_return"] = bench.shift(-horizon) / bench - 1.0
        df["target_excess_return"] = df["future_return"] - df["benchmark_future_return"]

        return df.dropna(subset=["target_excess_return"])

    def build_universe_frame(self, profile_map: dict, interval: str, benchmark: str = "SPY", horizon: int = 20) -> pd.DataFrame:
        benchmark_df = load_data(benchmark, interval)
        if benchmark_df is None or benchmark_df.empty:
            raise ValueError(f"No benchmark data for {benchmark}")

        benchmark_close = benchmark_df["Adj Close"]
        frames = []

        for ticker, meta in profile_map.items():
            try:
                raw = load_data(ticker, interval)

                if raw is None or raw.empty:
                    print(f"Skipping {ticker}: no raw data")
                    continue

                df = raw.ind.add_indicators(ticker, interval, add_targets=False)
                df = self.add_forward_excess_target(df, benchmark_close, horizon=horizon)

                df["ticker"] = ticker
                df["profile"] = str(meta["profile"])
                df["profile_vol"] = float(meta["vol"])
                df["profile_beta"] = float(meta["beta"])
                df["profile_adv_log"] = np.log1p(float(meta["adv"]))
                df["Date"] = df.index

                frames.append(df)
                print(f"Loaded {ticker}: {len(df)} rows, profile {meta['profile']}")

            except Exception as e:
                print(f"Skipping {ticker}: {e}")

        if not frames:
            raise ValueError("No usable ticker data")

        data = pd.concat(frames, axis=0)
        data = data.sort_values(["Date", "ticker"])
        data = data.replace([np.inf, -np.inf], np.nan)
        return data.dropna()

    def prepare_universe_data(self, data: pd.DataFrame):
        data = data.copy()
        data["profile_group"] = data["profile"]
        data = pd.get_dummies(data, columns=["profile"], prefix="profile", dtype=int)

        drop_cols = {
            "Open", "High", "Low", "Close", "Adj Close", "Volume",
            "Adj Open", "Adj High", "Adj Low",
            "MA_200", "return",
            "ticker", "Date", "profile_group",
            "future_return", "benchmark_future_return", "target_excess_return",
            "target_profit", "tbm_return", "barrier_strength",
            "time_to_gain", "time_to_loss", "tp_return", "sl_return",
        }

        feature_cols = [
            c for c in data.columns
            if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])
        ]

        dates = np.array(sorted(pd.to_datetime(data["Date"]).unique()))
        split_date = dates[int(len(dates) * (1 - self.__test_size))]

        train_df = data[data["Date"] <= split_date].copy()
        test_df = data[data["Date"] > split_date].copy()

        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values

        y_train = train_df["target_excess_return"].values.astype(np.float32)
        y_test = test_df["target_excess_return"].values.astype(np.float32)

        return train_df, test_df, X_train, X_test, y_train, y_test, feature_cols, split_date

    @staticmethod
    def add_cross_sectional_signals(df: pd.DataFrame, pred_col: str = "pred", group_cols: list[str] | None = None, top_q: float = 0.90, bottom_q: float = 0.10,
        min_abs_pred: float = 0.003,
        allow_short: bool = True,
    ) -> pd.DataFrame:
        df = df.copy()
        df["pred_signal"] = 0

        if group_cols is None:
            group_cols = ["Date", "profile_group"]

        for _, group in df.groupby(group_cols):
            if len(group) < 5:
                continue

            upper = group[pred_col].quantile(top_q)
            lower = group[pred_col].quantile(bottom_q)

            long_mask = (group[pred_col] >= upper) & (group[pred_col].abs() >= min_abs_pred)
            df.loc[group.index[long_mask], "pred_signal"] = 1

            if allow_short:
                short_mask = (group[pred_col] <= lower) & (group[pred_col].abs() >= min_abs_pred)
                df.loc[group.index[short_mask], "pred_signal"] = -1

        return df


def evaluate_strategy(df: pd.DataFrame, horizon: int = 20, cost_bps: float = 10.0) -> dict[str, Any]:
    df = df.copy()
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
            "n_trades": 0,
        }

    cost = cost_bps / 10_000
    trades["strategy_return"] = trades["pred_signal"] * trades["target_excess_return"] - cost

    mean_return = trades["strategy_return"].mean()
    std_return = trades["strategy_return"].std() + 1e-9

    return {
        "trade_rate": float((df["pred_signal"] != 0).mean()),
        "long_rate": float((df["pred_signal"] == 1).mean()),
        "short_rate": float((df["pred_signal"] == -1).mean()),
        "hit_rate": float((trades["strategy_return"] > 0).mean()),
        "mean_return": float(mean_return),
        "median_return": float(trades["strategy_return"].median()),
        "sharpe_like": float((mean_return / std_return) * np.sqrt(252 / horizon)),
        "n_trades": int(len(trades)),
    }


def evaluate_baselines(
    test_df: pd.DataFrame,
    horizon: int = 20,
    allow_short: bool = True,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, Any]]:
    results = {}
    rng = np.random.default_rng(seed)

    random_df = test_df.copy()
    random_df["pred"] = rng.normal(0, 1, len(random_df))
    random_df = add_cross_sectional_signals(random_df, allow_short=allow_short)
    results["random"] = evaluate_strategy(random_df, horizon=horizon)

    momentum_df = test_df.copy()
    momentum_df["pred"] = momentum_df["mom_1m"]
    momentum_df = add_cross_sectional_signals(momentum_df, allow_short=allow_short)
    results["momentum_1m"] = evaluate_strategy(momentum_df, horizon=horizon)

    reversal_df = test_df.copy()
    reversal_df["pred"] = -reversal_df["mom_1m"]
    reversal_df = add_cross_sectional_signals(reversal_df, allow_short=allow_short)
    results["reversal_1m"] = evaluate_strategy(reversal_df, horizon=horizon)

    return results


def make_lgbm_model(seed: int = DEFAULT_SEED) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=700,
        learning_rate=0.025,
        max_depth=5,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="regression",
        random_state=seed,
        verbose=-1,
    )


def train_global_universe_model(
    profile_map: dict[str, dict[str, Any]],
    config: UniverseConfig | None = None,
) -> dict[str, Any]:
    if config is None:
        config = UniverseConfig()

    data = build_universe_frame(
        profile_map=profile_map,
        interval=config.interval,
        benchmark=config.benchmark,
        horizon=config.horizon,
    )

    train_df, test_df, X_train, X_test, y_train, y_test, feature_cols, split_date = prepare_universe_data(
        data,
        test_size=config.test_size,
    )

    model = make_lgbm_model(seed=config.seed)
    model.fit(X_train, y_train)

    test_df = test_df.copy()
    test_df["pred"] = model.predict(X_test)

    test_df = add_cross_sectional_signals(
        test_df,
        pred_col="pred",
        group_cols=["Date", "profile_group"],
        top_q=config.top_q,
        bottom_q=config.bottom_q,
        min_abs_pred=config.min_abs_pred,
        allow_short=config.allow_short,
    )

    metrics = evaluate_strategy(test_df, horizon=config.horizon, cost_bps=config.cost_bps)
    baselines = evaluate_baselines(
        test_df,
        horizon=config.horizon,
        allow_short=config.allow_short,
        seed=config.seed,
    )

    pred_mae = mean_absolute_error(y_test, test_df["pred"].values)
    pred_rmse = mean_squared_error(y_test, test_df["pred"].values) ** 0.5

    print("Global model metrics:")
    print(json.dumps(metrics, indent=4))

    print("Baselines:")
    print(json.dumps(baselines, indent=4))

    return {
        "model": model,
        "feature_cols": feature_cols,
        "train_df": train_df,
        "test_df": test_df,
        "metrics": metrics,
        "baselines": baselines,
        "pred_mae": float(pred_mae),
        "pred_rmse": float(pred_rmse),
        "split_date": str(pd.Timestamp(split_date)),
        "config": config,
    }


def train_profile_models(
    profile_map: dict[str, dict[str, Any]],
    config: UniverseConfig | None = None,
) -> dict[str, Any]:
    if config is None:
        config = UniverseConfig(save_name="profile_universe_excess_return")

    data = build_universe_frame(
        profile_map=profile_map,
        interval=config.interval,
        benchmark=config.benchmark,
        horizon=config.horizon,
    )

    results: dict[str, Any] = {}

    for profile in sorted(data["profile"].unique()):
        profile_data = data[data["profile"] == profile].copy()

        if profile_data["ticker"].nunique() < 5:
            print(f"Skipping profile {profile}: fewer than 5 tickers")
            continue

        train_df, test_df, X_train, X_test, y_train, y_test, feature_cols, split_date = prepare_universe_data(
            profile_data,
            test_size=config.test_size,
        )

        model = make_lgbm_model(seed=config.seed)
        model.fit(X_train, y_train)

        test_df = test_df.copy()
        test_df["pred"] = model.predict(X_test)

        test_df = add_cross_sectional_signals(
            test_df,
            pred_col="pred",
            group_cols=["Date"],
            top_q=config.top_q,
            bottom_q=config.bottom_q,
            min_abs_pred=config.min_abs_pred,
            allow_short=config.allow_short,
        )

        metrics = evaluate_strategy(test_df, horizon=config.horizon, cost_bps=config.cost_bps)
        baselines = evaluate_baselines(
            test_df,
            horizon=config.horizon,
            allow_short=config.allow_short,
            seed=config.seed,
        )

        results[profile] = {
            "model": model,
            "feature_cols": feature_cols,
            "test_df": test_df,
            "metrics": metrics,
            "baselines": baselines,
            "split_date": str(pd.Timestamp(split_date)),
        }

        print(f"Profile {profile} metrics:")
        print(json.dumps(metrics, indent=4))

    return results


def save_training_result(result: dict[str, Any], config: UniverseConfig) -> str:
    save_folder = Path(MODEL_DIR) / config.save_name
    save_folder.mkdir(parents=True, exist_ok=True)

    joblib.dump(result["model"], save_folder / "model.joblib")
    joblib.dump(result["feature_cols"], save_folder / "features.joblib")

    result["test_df"].to_parquet(save_folder / "test_predictions.parquet", index=False)

    metadata = {
        "training_date": datetime.now().strftime("%Y-%m-%d"),
        "config": config.__dict__,
        "metrics": result["metrics"],
        "baselines": result["baselines"],
        "pred_mae": result["pred_mae"],
        "pred_rmse": result["pred_rmse"],
        "split_date": result["split_date"],
    }

    (save_folder / "metadata.json").write_text(json.dumps(metadata, indent=4), encoding="utf-8")
    return str(save_folder)


def rank_latest_predictions(result: dict[str, Any], top_n: int = 20) -> pd.DataFrame:
    test_df = result["test_df"].copy()
    latest_date = test_df["Date"].max()
    latest = test_df[test_df["Date"] == latest_date].copy()

    cols = ["Date", "ticker", "profile_group", "pred", "pred_signal", "target_excess_return"]
    latest = latest[cols].sort_values("pred", ascending=False)

    return pd.concat([latest.head(top_n), latest.tail(top_n)], axis=0)


def run_from_ticker_list(
    tickers: list[str],
    config: UniverseConfig | None = None,
    profile_map_path: str | os.PathLike | None = None,
) -> dict[str, Any]:
    if config is None:
        config = UniverseConfig()

    if profile_map_path and Path(profile_map_path).exists():
        profile_map = load_profile_map(profile_map_path)
    else:
        profile_map = build_profile_map(
            tickers=tickers,
            interval=config.interval,
            benchmark=config.benchmark,
        )

        if profile_map_path:
            save_profile_map(profile_map, profile_map_path)

    result = train_global_universe_model(profile_map, config=config)
    save_path = save_training_result(result, config)

    print(f"Saved model assets to: {save_path}")

    latest = rank_latest_predictions(result, top_n=20)
    print("Latest ranked predictions:")
    print(latest.to_string(index=False))

    return result


if __name__ == "__main__":
    config = UniverseConfig(
        interval="1d",
        benchmark="SPY",
        horizon=20,
        test_size=0.2,
        top_q=0.90,
        bottom_q=0.10,
        min_abs_pred=0.003,
        allow_short=True,
        cost_bps=10.0,
        save_name="global_universe_excess_return",
    )

    # Replace DEFAULT_STARTER_UNIVERSE with your top-1000 ticker list, e.g:
    # tickers = load_tickers_from_txt("top_1000_tickers.txt")
    tickers = DEFAULT_STARTER_UNIVERSE

    run_from_ticker_list(
        tickers=tickers,
        config=config,
        profile_map_path="profile_map.json",
    )
