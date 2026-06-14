
# Standard library imports
import gc
import json
import os
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

# External library imports
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from skorch import dataset

# Custom imports
from scripts.data_management import load_data
from scripts.config import DATA_DIR, MODEL_DIR
import scripts.indicators  # noqa

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

class LSTMBrain(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2, output_dim: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        return self.fc(out)


class ConstantRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.full(len(X), self.value, dtype=np.float32)


class PurgedTimeSeriesSplit:
    def __init__(self, n_splits: int = 4, gap: int = 4, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.gap = gap
        self.embargo_pct = embargo_pct

    def split(self, X: np.ndarray):
        n_samples = len(X)
        embargo = int(n_samples * self.embargo_pct)
        test_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = (i + 1) * test_size - self.gap
            test_start = (i + 1) * test_size + embargo
            test_end = min(test_start + test_size, n_samples)

            if train_end <= 0:
                continue

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            if len(test_indices) > 0:
                yield train_indices, test_indices


class TrainingManager:
    def __init__(self):
        self.seed = 69
        self.__test_size = 0.2

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.signal_train = None
        self.signal_test = None
        self.returns_train = None
        self.returns_test = None
        self.feature_cols = None
        self.scaler = None

    def _prepare_data(self, df: pd.DataFrame):
        drop_cols = {
            "Open", "High", "Low", "Close", "Adj Close", "Volume", "MA_200",
            "return", "target_profit", "tbm_return", "barrier_strength",
            "time_to_gain", "time_to_loss", "tp_return", "sl_return"
        }

        self.feature_cols = [c for c in df.columns if c not in drop_cols]

        split_idx = int(len(df) * (1 - self.__test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(train_df[self.feature_cols].values)
        self.X_test = self.scaler.transform(test_df[self.feature_cols].values)

        self.y_train = train_df[["time_to_gain", "time_to_loss"]].values.astype(np.float32)
        self.y_test = test_df[["time_to_gain", "time_to_loss"]].values.astype(np.float32)

        self.signal_train = train_df["target_profit"].values.astype(int)
        self.signal_test = test_df["target_profit"].values.astype(int)

        self.returns_train = train_df["tbm_return"].values.astype(np.float32)
        self.returns_test = test_df["tbm_return"].values.astype(np.float32)

    @staticmethod
    def times_to_signal(predicted_times: np.ndarray) -> np.ndarray:
        predicted_times = np.asarray(predicted_times, dtype=float)

        gain_time = np.clip(predicted_times[:, 0], 1.0, 21.0)
        loss_time = np.clip(predicted_times[:, 1], 1.0, 21.0)

        edge = loss_time - gain_time
        soonest = np.minimum(gain_time, loss_time)

        signal = np.zeros(len(predicted_times), dtype=int)
        trade_mask = (soonest <= 20) & (np.abs(edge) >= 0.75)

        signal[trade_mask & (edge > 0)] = 1
        signal[trade_mask & (edge < 0)] = -1

        return signal

    @staticmethod
    def times_to_signal_score(predicted_times: np.ndarray) -> float:
        predicted_times = np.asarray(predicted_times, dtype=float).reshape(1, 2)

        gain_time = float(np.clip(predicted_times[0, 0], 1.0, 21.0))
        loss_time = float(np.clip(predicted_times[0, 1], 1.0, 21.0))

        edge = loss_time - gain_time
        raw_signal = np.clip(edge / 20, -1.0, 1.0)

        return float((raw_signal + 1.0) / 2.0)

    @staticmethod
    def evaluate_performance(interval: str, actual_signal: np.ndarray, predicted_signal: np.ndarray, actual_returns: np.ndarray) -> tuple:
        actual_signal = np.asarray(actual_signal).reshape(-1)
        predicted_signal = np.asarray(predicted_signal).reshape(-1)
        actual_returns = np.asarray(actual_returns).reshape(-1)

        if not (len(actual_signal) == len(predicted_signal) == len(actual_returns)):
            raise ValueError(
                f"Evaluation arrays must align: actual={len(actual_signal)}, "
                f"predicted={len(predicted_signal)}, returns={len(actual_returns)}"
            )

        direction_accuracy = float(np.mean(actual_signal == predicted_signal))

        if interval == "1d":
            bars = 252
        elif interval == "1h":
            bars = 1638
        else:
            raise ValueError("Interval length not valid.")

        custom_scores = np.nan_to_num(actual_returns * predicted_signal)

        missed_breakout = (actual_signal != 0) & (predicted_signal == 0)
        custom_scores[missed_breakout] = -np.abs(actual_returns[missed_breakout]) * 0.5

        opposite_mask = (actual_signal != 0) & (predicted_signal == -actual_signal)
        custom_scores[opposite_mask] *= 2.0

        hold_right = (actual_signal == 0) & (predicted_signal == 0)
        if np.any(hold_right):
            custom_scores[hold_right] = np.abs(actual_returns[hold_right]).mean() * 0.1

        util_score = (np.mean(custom_scores) / (np.std(custom_scores) + 1e-9)) * np.sqrt(bars)

        cumulative_returns = np.cumsum(custom_scores)
        if len(cumulative_returns) > 1:
            x = np.arange(len(cumulative_returns))
            slope, _, r_value, _, _ = stats.linregress(x, cumulative_returns)
            stability = r_value ** 2 if slope > 0 else 0.0
        else:
            stability = 0.0

        trade_rate = float(np.mean(predicted_signal != 0))
        if trade_rate < 0.01:
            util_score -= 1.0

        obj_score = 0.55 * util_score + 0.25 * direction_accuracy + 0.20 * stability

        return obj_score, direction_accuracy, util_score, stability, trade_rate

    @staticmethod
    def create_3d_sequences(X: np.ndarray, y: np.ndarray, returns: np.ndarray, signals: np.ndarray, window: int = 30):
        x3, y3, r3, s3 = [], [], [], []

        for i in range(len(X) - window):
            x3.append(X[i:i + window])
            y3.append(y[i + window])
            r3.append(returns[i + window])
            s3.append(signals[i + window])

        return (
            np.array(x3, dtype=np.float32),
            np.array(y3, dtype=np.float32),
            np.array(r3, dtype=np.float32),
            np.array(s3, dtype=int)
        )

    @staticmethod
    def _get_lgbm_params(hyperparams: dict) -> dict:
        lgbm_params = hyperparams.get("LGBM", {}).get("best_params", {})
        if Settings.GPU["LGBM"]:
            lgbm_params.update({"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
        if Settings.Threaded:
            lgbm_params.update({"num_threads": -1, "n_jobs": -1})

        defaults = {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 5,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "objective": "regression",
        }
        defaults.update(lgbm_params)
        return defaults

    @staticmethod
    def _get_cat_params(hyperparams: dict) -> dict:
        cat_params = hyperparams.get("CAT", {}).get("best_params", {})
        cat_params["task_type"] = "GPU" if Settings.GPU["CAT"] else "CPU"

        defaults = {
            "iterations": 500,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "loss_function": "RMSE",
            "allow_writing_files": False,
        }
        defaults.update(cat_params)
        return defaults

    @staticmethod
    def _get_lstm_params(hyperparams: dict) -> tuple[dict, type]:
        lstm_params = hyperparams.get("LSTM", {}).get("best_params", {})

        optimizer_name = lstm_params.pop("optimizer_name", "AdamW")
        optimizers = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW, "RMSprop": torch.optim.RMSprop}
        optimizer = optimizers.get(optimizer_name, torch.optim.AdamW)

        defaults = {
            "module__hidden_dim": 64,
            "module__num_layers": 2,
            "module__dropout": 0.2,
            "max_epochs": 40,
            "batch_size": 32,
            "lr": 1e-3,
            "optimizer__weight_decay": 1e-5,
        }
        defaults.update(lstm_params)

        return defaults, optimizer

    def _train_lightgbm(self, interval: str, hyperparams: dict) -> dict:
        params = self._get_lgbm_params(hyperparams)
        tscv = PurgedTimeSeriesSplit(n_splits=3, gap=20, embargo_pct=0.01)
        scores = []

        for train_idx, val_idx in tscv.split(self.X_train):
            gain_model = LGBMRegressor(random_state=self.seed, verbose=-1, **params)
            loss_model = LGBMRegressor(random_state=self.seed, verbose=-1, **params)

            gain_model.fit(self.X_train[train_idx], self.y_train[train_idx, 0])
            loss_model.fit(self.X_train[train_idx], self.y_train[train_idx, 1])

            pred_times = np.column_stack([
                gain_model.predict(self.X_train[val_idx]),
                loss_model.predict(self.X_train[val_idx])
            ])

            pred_signal = self.times_to_signal(pred_times)
            obj_score, _, _, _, _ = self.evaluate_performance(
                interval, self.signal_train[val_idx], pred_signal, self.returns_train[val_idx]
            )
            scores.append(obj_score)

        gain_model = LGBMRegressor(random_state=self.seed, verbose=-1, **params)
        loss_model = LGBMRegressor(random_state=self.seed, verbose=-1, **params)

        gain_model.fit(self.X_train, self.y_train[:, 0])
        loss_model.fit(self.X_train, self.y_train[:, 1])

        test_times = np.column_stack([gain_model.predict(self.X_test), loss_model.predict(self.X_test)])
        test_signal = self.times_to_signal(test_times)
        obj_score, acc, util_score, stability, trade_rate = self.evaluate_performance(
            interval, self.signal_test, test_signal, self.returns_test
        )

        mae = mean_absolute_error(self.y_test, test_times)

        return {
            "type": "LGBM",
            "gain_model": gain_model,
            "loss_model": loss_model,
            "mae": float(mae),
            "accuracy": acc,
            "stability": stability,
            "util_score": util_score,
            "wf_util_score": float(np.mean(scores)),
            "obj_score": obj_score,
            "trade_rate": trade_rate,
        }

    def _train_catboost(self, interval: str, hyperparams: dict) -> dict:
        params = self._get_cat_params(hyperparams)
        tscv = PurgedTimeSeriesSplit(n_splits=3, gap=20, embargo_pct=0.01)
        scores = []

        for train_idx, val_idx in tscv.split(self.X_train):
            gain_model = CatBoostRegressor(random_seed=self.seed, verbose=False, **params)
            loss_model = CatBoostRegressor(random_seed=self.seed, verbose=False, **params)

            gain_model.fit(self.X_train[train_idx], self.y_train[train_idx, 0])
            loss_model.fit(self.X_train[train_idx], self.y_train[train_idx, 1])

            pred_times = np.column_stack([
                gain_model.predict(self.X_train[val_idx]),
                loss_model.predict(self.X_train[val_idx])
            ])

            pred_signal = self.times_to_signal(pred_times)
            obj_score, _, _, _, _ = self.evaluate_performance(
                interval, self.signal_train[val_idx], pred_signal, self.returns_train[val_idx]
            )
            scores.append(obj_score)

        gain_model = CatBoostRegressor(random_seed=self.seed, verbose=False, **params)
        loss_model = CatBoostRegressor(random_seed=self.seed, verbose=False, **params)

        gain_model.fit(self.X_train, self.y_train[:, 0])
        loss_model.fit(self.X_train, self.y_train[:, 1])

        test_times = np.column_stack([gain_model.predict(self.X_test), loss_model.predict(self.X_test)])
        test_signal = self.times_to_signal(test_times)
        obj_score, acc, util_score, stability, trade_rate = self.evaluate_performance(
            interval, self.signal_test, test_signal, self.returns_test
        )

        mae = mean_absolute_error(self.y_test, test_times)

        return {
            "type": "CAT",
            "gain_model": gain_model,
            "loss_model": loss_model,
            "mae": float(mae),
            "accuracy": acc,
            "stability": stability,
            "util_score": util_score,
            "wf_util_score": float(np.mean(scores)),
            "obj_score": obj_score,
            "trade_rate": trade_rate,
        }

    def _train_lstm(self, interval: str, hyperparams: dict) -> dict:
        params, optimizer = self._get_lstm_params(hyperparams)
        window = 30

        if len(self.X_train) <= window * 3:
            raise ValueError(f"Not enough training rows for LSTM: got {len(self.X_train)}, need > {window * 3}")

        x_train_3d, y_train_seq, returns_train_seq, signal_train_seq = self.create_3d_sequences(
            self.X_train, self.y_train, self.returns_train, self.signal_train, window=window
        )

        X_test_padded = np.vstack((self.X_train[-window:], self.X_test))
        y_test_padded = np.vstack((self.y_train[-window:], self.y_test))
        returns_test_padded = np.concatenate((self.returns_train[-window:], self.returns_test))
        signal_test_padded = np.concatenate((self.signal_train[-window:], self.signal_test))

        x_test_3d, y_test_seq, returns_test_seq, signal_test_seq = self.create_3d_sequences(
            X_test_padded, y_test_padded, returns_test_padded, signal_test_padded, window=window
        )

        input_dim = x_train_3d.shape[2]
        device = "cuda" if torch.cuda.is_available() and Settings.GPU["LSTM"] else "cpu"

        def build_model() -> NeuralNetRegressor:
            return NeuralNetRegressor(
                LSTMBrain,
                module__input_dim=input_dim,
                module__output_dim=2,
                criterion=nn.SmoothL1Loss,
                optimizer=optimizer,
                train_split=dataset.ValidSplit(0.2, stratified=False),
                iterator_train__shuffle=False,
                device=device,
                verbose=Settings.VERBOSE,
                callbacks=[("early_stopping", EarlyStopping(monitor="valid_loss", patience=7, lower_is_better=True))],
                **params,
            )

        tscv = PurgedTimeSeriesSplit(n_splits=2, gap=window+20, embargo_pct=0.01)
        scores = []

        for train_idx, val_idx in tscv.split(x_train_3d):
            fold_model = build_model()
            fold_model.fit(x_train_3d[train_idx], y_train_seq[train_idx])
            pred_times = fold_model.predict(x_train_3d[val_idx])

            pred_signal = self.times_to_signal(pred_times)
            obj_score, _, _, _, _ = self.evaluate_performance(
                interval, signal_train_seq[val_idx], pred_signal, returns_train_seq[val_idx]
            )
            scores.append(obj_score)

        model = build_model()
        model.fit(x_train_3d, y_train_seq)

        test_times = model.predict(x_test_3d)
        test_signal = self.times_to_signal(test_times)
        obj_score, acc, util_score, stability, trade_rate = self.evaluate_performance(
            interval, signal_test_seq, test_signal, returns_test_seq
        )

        mae = mean_absolute_error(y_test_seq, test_times)

        return {
            "type": "LSTM",
            "model": model,
            "mae": float(mae),
            "accuracy": acc,
            "stability": stability,
            "util_score": util_score,
            "wf_util_score": float(np.mean(scores)),
            "obj_score": obj_score,
            "trade_rate": trade_rate,
        }

    def _save_model_assets(self, ticker: str, interval: str, training_data_end: pd.Timestamp, results: dict) -> None:
        save_folder = os.path.join(MODEL_DIR, f"{ticker}_{interval}")
        os.makedirs(save_folder, exist_ok=True)

        metadata = {
            "training_date": datetime.now().strftime("%Y-%m-%d"),
            "training_data_end": training_data_end.strftime("%Y-%m-%d %H:%M"),
            "ticker": ticker,
            "interval": interval,
            "model_results": {},
        }

        for model_type, model_results in results.items():
            metadata["model_results"][model_type] = {
                "mae": model_results.get("mae"),
                "accuracy": model_results.get("accuracy"),
                "util_score": model_results.get("util_score"),
                "wf_util_score": model_results.get("wf_util_score"),
                "stability": model_results.get("stability"),
                "obj_score": model_results.get("obj_score"),
                "trade_rate": model_results.get("trade_rate"),
            }

            if model_type == "LGBM":
                model_results["gain_model"].booster_.save_model(os.path.join(save_folder, "lgbm_gain_model.txt"))
                model_results["loss_model"].booster_.save_model(os.path.join(save_folder, "lgbm_loss_model.txt"))
            elif model_type == "CAT":
                joblib.dump(model_results["gain_model"], os.path.join(save_folder, "cat_gain_model.joblib"))
                joblib.dump(model_results["loss_model"], os.path.join(save_folder, "cat_loss_model.joblib"))
            elif model_type == "LSTM":
                joblib.dump(model_results["model"], os.path.join(save_folder, "lstm_model.joblib"))

        with open(os.path.join(save_folder, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        joblib.dump(self.scaler, os.path.join(save_folder, "scaler.joblib"))
        joblib.dump(self.feature_cols, os.path.join(save_folder, "features.joblib"))

    def run_training_pipeline(self, ticker: str, interval: str, status_signal: tuple | None = None, force_train: bool = False) -> bool:
        def log_update(msg, force_print=False):
            if status_signal:
                u_queue, core_key = status_signal
                u_queue.put((core_key, {"Current Task": msg}))
                if force_print: print(msg)

            elif Settings.LOGGING or force_print: print(msg)

        model_path = os.path.join(MODEL_DIR, f"{ticker}_{interval}")
        if all_ticker_models_exist(model_path) and not force_train:
            log_update(f"Model {ticker} ({interval}) already trained", True)
            return True

        if os.path.exists(model_path): shutil.rmtree(model_path)

        log_update("Loading data...", True)
        raw_data = load_data(ticker, interval)
        if raw_data is None or raw_data.empty:
            log_update(f"No raw data for {ticker} ({interval})", True)
            return False

        log_update("Adding features and regression targets...", True)
        df = raw_data.ind.add_indicators(ticker, interval, add_targets=True)
        if len(df) < 300:
            log_update(f"Insufficient processed data for {ticker} ({interval}) — need 300+, got {len(df)}", True)
            return False

        for col in ["time_to_gain", "time_to_loss"]:
            vals, counts = np.unique(df[col], return_counts=True)
            print(col)
            for v, c in zip(vals, counts):
                print(v, c, round(c / len(df), 3))

        exit()

        log_update("Preparing features...", True)
        self._prepare_data(df)

        # with open(os.path.join(DATA_DIR, "hyperparameters.json"), "r") as f:
        #     hyperparams = json.load(f)[interval]
        hyperparams = {}

        results = {}

        t0 = time.perf_counter()
        log_update("Training LightGBM...", True)
        results["LGBM"] = self._train_lightgbm(interval, hyperparams)
        flush_memory()
        log_update(f"LightGBM done in {time.perf_counter() - t0:.1f}s", True)

        t0 = time.perf_counter()
        log_update("Training CatBoost...", True)
        results["CAT"] = self._train_catboost(interval, hyperparams)
        flush_memory()
        log_update(f"CatBoost done in {time.perf_counter() - t0:.1f}s", True)

        t0 = time.perf_counter()
        log_update("Training LSTM...", True)
        results["LSTM"] = self._train_lstm(interval, hyperparams)
        flush_memory()
        log_update(f"LSTM done in {time.perf_counter() - t0:.1f}s", True)

        log_update("Saving assets...", True)
        self._save_model_assets(ticker, interval, raw_data.index.max(), results)
        return True


def all_ticker_models_exist(model_path: str) -> bool:
    root = Path(model_path)
    if not root.exists():
        return False

    required_files = [
        "scaler.joblib", "features.joblib", "metadata.json", "lstm_model.joblib",
        "lgbm_gain_model.txt", "lgbm_loss_model.txt",
        "cat_gain_model.joblib", "cat_loss_model.joblib"
    ]

    return all((root / file_name).exists() and (root / file_name).stat().st_size > 0 for file_name in required_files)
