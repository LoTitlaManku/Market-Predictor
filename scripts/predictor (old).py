
# Standard library imports
import json
import os
import time
from pathlib import Path
import shutil
import warnings
from datetime import datetime, timedelta

# External library imports
import joblib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from lightgbm import LGBMClassifier
from lightgbm import Booster as LGBMBooster
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier, dataset
from skorch.callbacks import EarlyStopping, EpochScoring
from safetensors.torch import save_file, load_file
from scipy import stats # noqa
import gc
from lightgbm import LGBMRegressor
from lightgbm import Booster as LGBMBooster
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from skorch import NeuralNetRegressor, dataset
from skorch.callbacks import EarlyStopping

# Set environment variables and filters
warnings.filterwarnings("ignore")
NYSE_CAL = mcal.get_calendar('NYSE')

# Custom imports
from scripts.data_management import load_data
from scripts.config import LEDGER_DIR, MODEL_DIR, DATA_DIR
import scripts.indicators # noqa

class Settings:
    VERBOSE = 0                                        # Set whether to display model logging or not
    LOGGING = False                                    # Set whether to display prints for training stages
    GPU = {"LGBM": False, "CAT": False, "LSTM": True}  # Set which model types can use GPU
    Threaded = False                                   # Set whether a singular model is being trained on multiple threads

def flush_memory():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

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

# Custom Purged & Embargoed TimeSeriesSplit for Financial Data
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
            test_end = test_start + test_size

            if test_end > n_samples:
                test_end = n_samples

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            if len(test_indices) > 0:
                yield train_indices, test_indices

# Class to control and train models
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
    def times_to_signal(predicted_times: np.ndarray, min_edge: float = 0.75) -> np.ndarray:
        predicted_times = np.asarray(predicted_times, dtype=float)

        gain_time = np.clip(predicted_times[:, 0], 1.0, 21.0)
        loss_time = np.clip(predicted_times[:, 1], 1.0, 21.0)

        edge = loss_time - gain_time
        soonest = np.minimum(gain_time, loss_time)

        signal = np.zeros(len(predicted_times), dtype=int)
        trade_mask = (soonest <= 20) & (np.abs(edge) >= min_edge)

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

    def _train_lightgbm(self, interval: str, hyperparams: dict) -> dict:
        lgbm_params = hyperparams["LGBM"]["best_params"]
        if Settings.GPU["LGBM"]:
            lgbm_params.update({"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
        if Settings.Threaded:
            lgbm_params.update({"num_threads": -1, "n_jobs": -1})

        tscv = PurgedTimeSeriesSplit(n_splits=3, gap=20, embargo_pct=0.01)
        scores = []

        for train_idx, val_idx in tscv.split(self.X_train):
            gain_model = LGBMRegressor(random_state=self.seed, verbose=-1, **lgbm_params)
            loss_model = LGBMRegressor(random_state=self.seed, verbose=-1, **lgbm_params)

            gain_model.fit(self.X_train[train_idx], self.y_train[train_idx, 0])
            loss_model.fit(self.X_train[train_idx], self.y_train[train_idx, 1])

            pred_times = np.column_stack([
                gain_model.predict(self.X_train[val_idx]),
                loss_model.predict(self.X_train[val_idx])
            ])

            pred_signal = self.times_to_signal(pred_times)
            obj_score, _, _, _, _ = self.evaluate_performance(
                interval,
                self.signal_train[val_idx],
                pred_signal,
                self.returns_train[val_idx]
            )
            scores.append(obj_score)

        gain_model = LGBMRegressor(random_state=self.seed, verbose=-1, **lgbm_params)
        loss_model = LGBMRegressor(random_state=self.seed, verbose=-1, **lgbm_params)

        gain_model.fit(self.X_train, self.y_train[:, 0])
        loss_model.fit(self.X_train, self.y_train[:, 1])

        test_times = np.column_stack([
            gain_model.predict(self.X_test),
            loss_model.predict(self.X_test)
        ])

        test_signal = self.times_to_signal(test_times)
        obj_score, acc, util_score, stability, trade_rate = self.evaluate_performance(
            interval,
            self.signal_test,
            test_signal,
            self.returns_test
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

    # def _train_catboost(self, interval: str, horizon: int, hyperparams: dict) -> dict:
    #     cat_params = hyperparams["CAT"]["best_params"]
    #     if Settings.GPU["CAT"]: cat_params["task_type"] = "GPU"
    #
    #     ...
    #
    # def _train_lstm(self, interval: str, horizon: int, hyperparams: dict) -> dict:
    #     lstm_params = hyperparams["LSTM"]["best_params"]
    #     optimizers = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW, "RMSprop": torch.optim.RMSprop}  # noqa
    #     optimizer = optimizers.get(lstm_params.pop("optimizer_name"), torch.optim.Adam)
    #
    #     window = 30
    #
    #     X_test_padded = np.vstack((self.X_train[-window:], self.X_test))
    #     y_test_padded = np.concatenate((self.y_train.values[-window:], self.y_test.values))
    #
    #     x_train_3d, y_train_seq = self.create_3d_sequences(self.X_train, self.y_train.values, window=window)
    #     x_test_3d, y_test_seq = self.create_3d_sequences(X_test_padded, y_test_padded, window=window)
    #
    #     # Shift targets to {0,1,2} for CrossEntropyLoss; keep originals for evaluate_performance
    #     y_train_shifted = (y_train_seq + 1).astype(np.int64)
    #     returns_train_seq = self.returns_train[window:]
    #     input_dim = x_train_3d.shape[2]
    #
    #     # Calculate exact class weights to prevent the LSTM from ignoring minority classes
    #     classes = np.unique(y_train_shifted)
    #     weights = compute_class_weight('balanced', classes=classes, y=y_train_shifted)
    #     weight_tensor = torch.tensor(weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')
    #
    #     def build_skorch_model() -> NeuralNetClassifier:
    #         return NeuralNetClassifier(
    #             LSTMBrain,
    #             module__input_dim=input_dim,
    #             module__output_dim=3,
    #             criterion=nn.CrossEntropyLoss,
    #             criterion__weight=weight_tensor,
    #             optimizer=optimizer,
    #             train_split=dataset.ValidSplit(0.2, stratified=False),
    #             iterator_train__shuffle=False,
    #             device='cuda' if torch.cuda.is_available() and Settings.GPU["LSTM"] else 'cpu',
    #             verbose=Settings.VERBOSE,
    #             callbacks=[
    #                 ('early_stopping', EarlyStopping(
    #                     monitor='valid_loss',
    #                     patience=5,
    #                     lower_is_better=True
    #                 )),
    #                 ('val_acc', EpochScoring(
    #                     scoring='accuracy',
    #                     name='valid_acc',
    #                     lower_is_better=False
    #                 ))
    #             ],
    #             **lstm_params
    #         )
    #
    #     # Walk-forward validation across folds of the training sequences
    #     tscv = PurgedTimeSeriesSplit(n_splits=2, gap=window + horizon, embargo_pct=0.01)
    #     scores = []
    #     for train_idx, val_idx in tscv.split(x_train_3d):
    #         fold_model = build_skorch_model()
    #         fold_model.fit(x_train_3d[train_idx], y_train_shifted[train_idx])
    #         preds = fold_model.predict(x_train_3d[val_idx]) - 1
    #
    #         # Penalize models that take zero trades to close the safe haven loophole
    #         if np.all(preds == 0):
    #             util_score = -1.0
    #         else:
    #             # Optimize for Sharpe Ratio
    #             _, util_score, _ = self.evaluate_performance(
    #                 interval, y_train_seq[val_idx], preds, returns_train_seq[val_idx]
    #             )
    #         scores.append(util_score)
    #
    #     # Final model trained on all training sequences
    #     model = build_skorch_model()
    #     model.fit(x_train_3d, y_train_shifted)
    #     test_preds = model.predict(x_test_3d) - 1
    #
    #     returns_test_seq = np.concatenate((self.returns_train[-window:], self.returns_test))[window:]
    #     accuracy, util_score, stability = self.evaluate_performance(interval, y_test_seq, test_preds, returns_test_seq)
    #
    #     return {
    #         'type': 'LSTM', 'model': model,
    #         'accuracy': accuracy, 'stability': stability,
    #         'util_score': util_score, 'wf_util_score': float(np.mean(scores)),
    #     }

    # def _save_model_assets(self, ticker: str, interval: str, training_data_end: pd.Timestamp, full_results: dict) -> None:
    #     save_folder = os.path.join(MODEL_DIR, f"{ticker}_{interval}")
    #     if not os.path.exists(save_folder): os.makedirs(save_folder)
    #
    #     metadata = {
    #         "training_date":      datetime.now().strftime("%Y-%m-%d"),
    #         "training_data_end":  training_data_end.strftime("%Y-%m-%d"),
    #         "ticker":             ticker,
    #         "interval":           interval,
    #         "model_results":      {k: {} for k in full_results.keys()}
    #     }
    #
    #     for horizon, results in full_results.items():
    #         horizon_folder = os.path.join(save_folder, horizon)
    #         if not os.path.exists(horizon_folder): os.makedirs(horizon_folder)
    #
    #         for model_type, model_results in results.items():
    #             model = model_results['model']
    #
    #             metadata["model_results"][horizon][model_type] = {
    #                 "accuracy":       model_results['accuracy'],
    #                 "util_score":     model_results['util_score'],
    #                 "wf_util_score":  model_results['wf_util_score'],
    #                 "stability":      model_results['stability'],
    #             }
    #
    #             if model_type == 'LSTM':
    #                 save_file(model.module_.state_dict(), os.path.join(horizon_folder, "lstm_model.safetensors"))
    #             elif model_type == 'LGBM':
    #                 model.booster_.save_model(os.path.join(horizon_folder, "lgbm_model.txt"))
    #             else:
    #                 joblib.dump(model, os.path.join(horizon_folder, f"cat_model.joblib"))
    #
    #     with open(os.path.join(save_folder, 'metadata.json'), 'w') as f:
    #         json.dump(metadata, f, indent=4)
    #
    #     joblib.dump(self.scaler, os.path.join(save_folder, "scaler.joblib"))
    #     joblib.dump(self.feature_cols, os.path.join(save_folder, "features.joblib"))

    def run_training_pipeline(self, ticker: str, interval: str, status_signal: tuple | None = None, force_train: bool = False) -> bool:
        def log_update(msg, force_print=False):
            if status_signal:
                u_queue, core_key = status_signal
                u_queue.put((core_key, {"Current Task": msg}))
                if force_print: print(msg)

            elif Settings.LOGGING: print(msg)

        model_path = os.path.join(MODEL_DIR, f"{ticker}_{interval}")
        if all_ticker_models_exist(model_path) and not force_train:
            log_update(f"Model {ticker} ({interval}) already trained", True)
            return True

        if os.path.exists(model_path): shutil.rmtree(model_path)

        log_update("Loading data...")
        raw_data = load_data(ticker, interval)
        if raw_data is None or raw_data.empty:
            log_update(f"No raw data for {ticker} ({interval})", True)
            return False

        # with open(os.path.join(DATA_DIR, "hyperparameters.json"), "r") as f:
        #     hyperparams = json.load(f)
        # hypers = hyperparams[interval]

        results = {}
        log_update("Adding features...")
        df = raw_data.ind.add_indicators(ticker, interval)
        if len(df) < 300:
            log_update(f"Insufficient processed data for {ticker} ({interval}) — need 300+, got {len(df)}", True)
            return False

        log_update("Preparing features...")
        self._prepare_data(df)
        time1 = time.perf_counter()

        print("Tuning LightGBM...")
        hypers = {"LGBM": {"best_params" : {"lr": 0.005, "n_estimators": 500, "max_depth": 5}}}
        results["LGBM"] = self._train_lightgbm(interval, hypers)
        flush_memory()

        time2 = time.perf_counter()
        print(f"Time taken: {time2 - time1}s")

        # print("Tuning CatBoost...")
        # results["CAT"]  = self._train_catboost(interval, horizon_hypers)
        # flush_memory()

        time3 = time.perf_counter()
        print(f"Time taken: {time3 - time2}s")

        # print("Tuning LSTM...")
        # results["LSTM"] = self._train_lstm    (interval, horizon_hypers)
        # flush_memory()

        time4 = time.perf_counter()
        print(f"Time taken: {time4 - time3}s")

        # log_update("Saving assets...")
        # self._save_model_assets(ticker, interval, raw_data.index.max(), results)
        return True

########################################################################################################################

def all_ticker_models_exist(model_path: str, horizons: list) -> bool:
    # root = Path(model_path)
    # if not root.exists(): return False
    #
    # roots = ["scaler.joblib", "features.joblib", "metadata.json"]
    # models = ["lgbm_model.txt", "cat_model.joblib", "lstm_model.safetensors"]
    #
    # def is_valid(file_path: Path) -> bool:
    #     return file_path.exists() and file_path.stat().st_size > 0
    #
    # if not  all(is_valid(root / f) for f in roots): return False
    # return  all(is_valid(root / str(h) / m) for h in horizons for m in models)
    """Logic to be added"""
    return False

# def save_prediction(ticker: str, interval: str, forecast_results: dict) -> None:
#     if len(forecast_results) < 1: return
#
#     ledger_file = os.path.join(LEDGER_DIR, f"{ticker}_ledger.csv")
#     new_entries = []
#     for horizon, data in forecast_results.items():
#         existing_match = find_prediction_match(ticker, interval, [horizon], data["Start_Date"])
#         if existing_match is not None and not existing_match.empty: continue
#
#         new_entries.append({
#             "Interval": interval,
#             "Horizon": horizon,
#             'Start_Date': data["Start_Date"],
#             "End_Date": data["End_Date"],
#             "Current_Price": round(data['Current_Price'], 2),
#             'AVG_signal': f"{data['AVG_signal']:.3f}",
#             'CAT_signal': f"{data['CAT_signal']:.3f}",
#             'LGBM_signal': f"{data['LGBM_signal']:.3f}",
#             'LSTM_signal': f"{data['LSTM_signal']:.3f}",
#         })
#
#     # Add prediction data to the ledger
#     df_new = pd.DataFrame(new_entries)
#     if not os.path.exists(ledger_file): df_new.to_csv(ledger_file, index=False)
#     else: df_new.to_csv(ledger_file, mode='a', header=False, index=False)
#
# def load_prediction(ticker: str, interval: str, date: datetime) -> dict | None:
#     # Filter for the specific data line matching the exact runtime parameters
#     match = find_prediction_match(ticker, interval, ["2", "4", "8"], date)
#
#     if match is None or match.empty: return None
#     match_dicts = match.reset_index().to_dict(orient='records')
#
#     # Rebuild the forecast_results dict
#     try:
#         forecast_results = {}
#         for row in match_dicts:
#             forecast_results[row["Horizon"]] = {
#                 "Date_Predicted": pd.to_datetime(row['Open_Date'], format="ISO8601"),
#                 "Current_Price": float(row['Current_Price']),
#                 'AVG_signal': float(row['AVG_signal']),
#                 'CAT_signal': float(row['CAT_signal']),
#                 'LGBM_signal': float(row['LGBM_signal']),
#                 'LSTM_signal': float(row['LSTM_signal']),
#             }
#         return forecast_results
#
#     except Exception: return None # noqa
#
# def find_prediction_match(ticker: str, interval: str, horizons: list, date: datetime) -> pd.DataFrame | None:
#     ledger_file = os.path.join(LEDGER_DIR, f"{ticker}_ledger.csv")
#     try:
#         ledger = pd.read_csv(ledger_file)
#         if ledger.empty or len(ledger) < 1: return None
#     except FileNotFoundError: return None
#
#     ledger = pd.read_csv(ledger_file)
#     ledger['Start_Date'] = pd.to_datetime(ledger['Start_Date'], format='ISO8601')
#
#     # Check if any entry matches current ticker and last trade date
#     date = date.strftime("%Y-%m-%d %H:%M")
#     match = ledger[(ledger['Interval'] == interval) & (ledger['Start_Date'] == date) & (ledger['Horizon'].isin(horizons))]
#     return match
#
# ########################################################################################################################
#
# def run_prediction_pipeline(ticker: str, interval: str) -> dict:
#     horizons = {
#         "1h": {2: 2, 4: 4, 8: 25},  # bars: hours
#         "1d": {2: 2, 4: 4, 8: 10},  # bars: days
#     }.get(interval, {})
#
#     df, assets = prepare_prediction_data(ticker, interval, horizons)
#     if any(v is None for v in [df, assets]):
#         print("No data or assets.")
#         return {}
#
#     last_trade_date = df.index[-1]
#     forecast_results = load_prediction(ticker, interval, last_trade_date)
#
#     if forecast_results is None:
#         forecast_results = generate_forecasts(df, ticker, interval, horizons, assets)
#         save_prediction(ticker, interval, forecast_results)
#
#     return forecast_results
#
# def prepare_prediction_data(ticker: str, interval: str, horizons: dict) -> tuple:
#     model_path = os.path.join(MODEL_DIR, f"{ticker}_{interval}")
#     manager = TrainingManager()
#
#     # Ensure data exists
#     df = load_data(ticker, interval)
#     if df is None: print("No data"); return None, None
#
#     # If its hourly data, and market has just opened, add in the day opening price
#     now_utc = pd.Timestamp.now(tz='UTC')
#     schedule = NYSE_CAL.schedule(start_date=now_utc, end_date=now_utc)
#     if not schedule.empty:
#         market_open = schedule.iloc[0]['market_open']
#         if interval == "1h" and market_open <= now_utc <= (market_open + timedelta(hours=1)):
#
#             temp_data = yf.download(ticker, period="1d", interval="1d", progress=False)
#             today_open = float(temp_data['Open'].values[-1][0])
#
#             new_row = pd.DataFrame({
#                 'Open': [today_open], 'High': [today_open],
#                 'Low': [today_open], 'Close': [today_open], 'Volume': [0], "Adj Close": [today_open]
#             }, index=[(market_open-timedelta(hours=1)).tz_localize(None)])
#
#             df = pd.concat([df, new_row])
#
#     # Trains a model if needed using the base implementation checks
#     if not all_ticker_models_exist(model_path, list(horizons.keys())):
#         success = manager.run_training_pipeline(ticker, interval)
#         if not success: return None, None
#
#     scaler = joblib.load(f"{model_path}/scaler.joblib")
#     features = joblib.load(f"{model_path}/features.joblib")
#
#     return df, (scaler, features, model_path)
#
# def get_market_dates(latest_date: pd.Timestamp, horizons: dict, interval: str) -> dict:
#     market_targets = {}
#
#     # latest_date is the OPENING time
#     if latest_date.tz is None:
#         latest_date = latest_date.tz_localize('UTC')
#
#     end_search = latest_date + pd.Timedelta(days=35)
#     schedule = NYSE_CAL.schedule(start_date=latest_date, end_date=end_search)
#
#     if interval == "1d":
#         # For daily: List of DAYs (e.g. '2026-03-06')
#         valid_times = schedule.index.normalize() # noqa
#     else:
#         # For 1h/15m: List of CLOSING times of the HOUR (e.g. 15:30 to 21:00)
#         valid_times = mcal.date_range(schedule, frequency="1h")
#
#     for step, time_dif in horizons.items():
#         if interval == "1d":
#             target_dt = latest_date + timedelta(days=time_dif)
#
#             # Keep rolling forward if it lands on a weekend or holiday
#             while target_dt.strftime("%Y-%m-%d") not in valid_times:
#                 target_dt += timedelta(days=1)
#
#                 if (target_dt - latest_date).days > 35:
#                     target_dt = None
#                     break
#
#         elif interval == "1h":
#             target_dt = latest_date + timedelta(hours=(time_dif+1))
#
#             if target_dt not in valid_times:
#                 if time_dif in [4, 25] and (target_dt - timedelta(hours=1)) in valid_times:
#                     target_dt = target_dt - timedelta(hours=1)
#                 else: target_dt = None
#
#         elif interval == "15m":
#             target_dt = None
#             pass
#
#         else: raise NotImplementedError("Interval not implemented.")
#
#         market_targets[step] = target_dt
#
#     return market_targets
#
# def generate_forecasts(df: pd.DataFrame, ticker: str, interval: str, horizons: dict, assets: tuple) -> dict:
#     scaler, features, model_folder = assets
#     last_trade_date = df.index[-1]
#
#     with open(os.path.join(model_folder, 'metadata.json'), 'r') as f:
#         meta = json.load(f)
#
#     with open(os.path.join(DATA_DIR, "hyperparameters.json"), "r") as f:
#         hyperparams = json.load(f)[interval]
#
#     forecast_results = {}
#
#     target_dates = get_market_dates(last_trade_date, horizons, interval)
#     if len(target_dates) < 1: return {}
#
#     # Calculate forecasts
#     for step, actual_time in horizons.items():
#         if target_dates[step] is None: continue
#
#         processed_df = df.ind.add_indicators(ticker, interval, step)
#         if len(df) < 300: continue
#
#         str_step = str(step)
#         hypers = hyperparams[str_step]
#         horizon_folder = os.path.join(model_folder, str_step)
#
#         probs   = {"LSTM": 0.5, "LGBM": 0.5, "CAT": 0.5}
#         weights = {"LSTM": 0.0, "LGBM": 0.0, "CAT": 0.0}
#
#         # LSTM prediction
#         lstm_path = os.path.join(horizon_folder, "lstm_model.safetensors")
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         brain_params = {
#             "hidden_dim": hypers["LSTM"]["best_params"].get("module__hidden_dim", 64),
#             "num_layers": hypers["LSTM"]["best_params"].get("module__num_layers", 2),
#             "dropout": hypers["LSTM"]["best_params"].get("module__dropout", 0.2),
#         }
#
#         recent_data = processed_df[features].tail(30)
#         scaled_seq = scaler.transform(recent_data)
#         x_3d = np.expand_dims(scaled_seq, axis=0).astype(np.float32)
#
#         brain = LSTMBrain(
#             input_dim=len(features),
#             output_dim=3,
#             **brain_params
#         )
#         state_dict = load_file(lstm_path)
#         brain.load_state_dict(state_dict)
#         brain.to(device)
#         brain.eval()
#
#         with torch.no_grad():
#             logits = brain(torch.from_numpy(x_3d).to(device))
#             p_lstm = torch.softmax(logits, dim=-1).cpu().numpy()[0]
#             probs["LSTM"] = float((p_lstm[2] - p_lstm[0] + 1) / 2)
#
#         # LGBM prediction
#         lgbm_path = os.path.join(horizon_folder, "lgbm_model.txt")
#         scaled_row = scaler.transform(processed_df[features].iloc[-1:])
#         booster = LGBMBooster(model_file=lgbm_path)
#         p_lgbm = booster.predict(scaled_row)[0]
#         probs["LGBM"] = float((p_lgbm[2] - p_lgbm[0] + 1) / 2)
#
#         # CAT prediction
#         cat_path = os.path.join(horizon_folder, "cat_model.joblib")
#         scaled_row = scaler.transform(processed_df[features].iloc[-1:])
#         cat_model = joblib.load(cat_path)
#         p_cat = cat_model.predict_proba(scaled_row)[0]
#         probs["CAT"] = float((p_cat[2] - p_cat[0] + 1) / 2)
#
#         # Calculate weights
#         for m_type in ["LSTM", "LGBM", "CAT"]:
#             model_meta = meta.get("model_results", {}).get(str_step, {}).get(m_type, {})
#             results_weight = {"LGBM": 0.33, "Cat": 0.33, "LSTM": 0.33}.get(m_type, 0.33)
#
#             util_score     =  max(0.0, model_meta.get("util_score", 0.0))
#             wf_util_score  =  max(0.0, model_meta.get("wf_util_score", 0.0))
#             acc            =  max(0.0, model_meta.get("accuracy", 0.0) - 0.33)
#
#             weights[m_type] = (results_weight * 0.0) + (wf_util_score * 0.5) + (util_score * 0.3) + (acc * 0.2)
#
#         total_weight = sum(weights.values())
#         avg_proba = sum(probs[m] * weights[m] for m in probs) / total_weight if total_weight > 0 else 0.5
#
#         # Calculate predicted target bounds
#         forecast_results[str_step] = {
#             "Start_Date":     last_trade_date.strftime("%Y-%m-%d %H:%M"),
#             "End_Date":       target_dates[step].strftime("%Y-%m-%d %H:%M"),
#             "Current_Price":  float(df['Adj Close'].iloc[-1]),
#             'AVG_signal':     avg_proba,
#             'CAT_signal':     probs["CAT"],
#             'LGBM_signal':    probs["LGBM"],
#             'LSTM_signal':    probs["LSTM"],
#         }
#
#     return forecast_results
