
# Standard library imports
import json
import os
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
from PyQt6.QtCore import QThread, pyqtSignal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier, dataset
from skorch.callbacks import EarlyStopping, EpochScoring
from safetensors.torch import save_file, load_file

# Set environment variables and filters
warnings.filterwarnings("ignore")
NYSE_CAL = mcal.get_calendar('NYSE')

# Custom imports
from scripts.data_management import load_data
from scripts.config import LEDGER_DIR, MODEL_DIR, DATA_DIR
import scripts.indicators # noqa

class Settings:
    VERBOSE = 0 # Set whether to display model logging or not
    LOGGING = False # Set whether to display prints for training stages
    GPU = True # Set whether to use GPU if possible
    Threaded = False # Set whether a singular model is being trained on multiple threads

############################################################################

# Class to create separate thread for trainer so can run concurrently with gui
class TrainingWorker(QThread):
    # Signal to send the results back to the GUI when finished
    training_finished: pyqtSignal = pyqtSignal(dict)
    training_error: pyqtSignal = pyqtSignal(str)

    def __init__(self, ticker: str, interval: str):
        super().__init__()
        self.ticker = ticker
        self.interval = interval

    # Run the prediction for its instance
    def run(self):
        try:
            forecast_results = run_prediction_pipeline(self.ticker, self.interval)
            self.training_finished.emit(forecast_results)
        except Exception as e:
            self.training_error.emit(str(e))

class LSTMBrain(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        # Note: Dropout in nn.LSTM only works if layers > 1
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        # Apply dropout to the final hidden state (last layer)
        out = self.dropout(hn[-1])
        return self.sigmoid(self.fc(out)).squeeze(-1)

class LSTM:
    def __init__(self, seed: int):
        self.seed = seed

    def train(self, hyperparams: dict, features_train: pd.DataFrame, target_train: pd.Series, features_test: pd.DataFrame,
                    target_test: pd.Series, price_returns: np.ndarray) -> dict:

        data_normalizer = StandardScaler()

        # Scale the data first
        features_train_normalized = data_normalizer.fit_transform(features_train)
        features_test_normalized = data_normalizer.transform(features_test)

        # Walk-Forward Validation
        time_splitter = TimeSeriesSplit(n_splits=3)
        validation_scores = []

        for train_idx, val_idx in time_splitter.split(features_train_normalized):
            # Create sequences for this specific fold
            x_fold_train, y_fold_train = self.create_3d_sequences(features_train_normalized[train_idx], target_train.iloc[train_idx].values)
            x_fold_val, y_fold_val = self.create_3d_sequences(features_train_normalized[val_idx], target_train.iloc[val_idx].values)

            if len(x_fold_train) == 0 or len(x_fold_val) == 0: continue

            fold_model = self.get_lstm_competitor(input_dim=x_fold_train.shape[2], hyperparams=hyperparams)
            fold_model.fit(x_fold_train, y_fold_train)
            validation_scores.append(fold_model.score(x_fold_val, y_fold_val))

        # Turn into 3D "Movies" (14-day lookback)
        x_train_3d, y_train_3d = self.create_3d_sequences(features_train_normalized, target_train.values)
        x_test_3d, y_test_3d = self.create_3d_sequences(features_test_normalized, target_test.values)

        # Train
        model = self.get_lstm_competitor(input_dim=x_train_3d.shape[2], hyperparams=hyperparams)
        model.fit(x_train_3d, y_train_3d)

        test_predictions = model.predict(x_test_3d)

        accuracy, sharpe, abs_sharpe, needs_flip = TrainingManager.evaluate_performance(
            target_test.iloc[14:],
            test_predictions,
            price_returns[14:]
        )

        return {
            'model_type': 'LSTM',
            'accuracy': accuracy,
            'walk_forward_accuracy': np.mean(validation_scores) if validation_scores else 0,
            'sharpe_ratio': sharpe,
            'absolute_sharpe': abs_sharpe,
            'logic_flipped': needs_flip,
            'raw_predictions': test_predictions,
            'trained_model_object': model,
            'feature_scaler': data_normalizer
        }

    @staticmethod
    def get_lstm_competitor(input_dim, hyperparams):
        optimizers = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}
        if isinstance(hyperparams.get("optimizer"), str):
            hyperparams.update({"optimizer": optimizers.get(hyperparams["optimizer"], torch.optim.Adam)})

        return NeuralNetClassifier(
            LSTMBrain,
            module__input_dim=input_dim,
            train_split=dataset.ValidSplit(0.2, stratified=False),
            iterator_train__shuffle=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=Settings.VERBOSE,
            criterion=nn.BCELoss,
            callbacks=[
                ('early_stopping', EarlyStopping(
                    monitor='valid_loss',
                    patience=5,  # Stop if no improvement after 5 epochs
                    lower_is_better=True
                )),
                ('val_acc', EpochScoring(
                    scoring='accuracy',
                    name='valid_acc',
                    lower_is_better=False
                ))
            ],
            **hyperparams
        )

    @staticmethod
    def create_3d_sequences(data, targets, window_size: int = 14):
        x, y = [], []

        # Convert to numpy arrays if they are pandas objects to ensure consistent behavior
        data_arr = data.values if hasattr(data, 'values') else data
        target_arr = targets.values if hasattr(targets, 'values') else targets

        for i in range(len(data_arr) - window_size):
            # Slicing works the same on both types
            x.append(data_arr[i: i + window_size])
            # Standard integer indexing now works for both
            y.append(target_arr[i + window_size])

        return np.array(x).astype(np.float32), np.array(y).astype(np.float32)

# Class to control and train models
class TrainingManager:
    def __init__(self):
        self.seed = 69
        self.sharpe_threshold = 0.50 # Min sharpe value for model to be useful
        self.__test_size = 0.2 # How much of data used to test vs train

    # Evaluate model performance with accuracy and sharpe ratio
    @staticmethod
    def evaluate_performance(actual_direction: pd.Series, predicted_direction: np.ndarray, actual_returns: np.ndarray):
        # How often was the AI right about Up vs Down?
        hit_rate = accuracy_score(actual_direction, predicted_direction)
        # Following the AI, what would a daily wallet look like?
        daily_strategy_returns = actual_returns * predicted_direction
        # How risky were those returns
        volatility = daily_strategy_returns.std()
        # Calculate the reward-to-risk ratio (Annualized)
        sharpe_ratio = (daily_strategy_returns.mean() / volatility) * np.sqrt(252) if volatility != 0 else 0

        # Return the score
        return hit_rate, sharpe_ratio, abs(sharpe_ratio), (sharpe_ratio < 0)

    # Train and evaluate LightGBM with walk-forward validation
    def _train_lightgbm(self, hyperparams: dict, features_train: pd.DataFrame, target_train: pd.Series, features_test: pd.DataFrame, target_test: pd.Series, price_returns: np.ndarray) -> dict:
        # Put all indicators on the same scale (0 to 1 range) so it matches inference
        data_normalizer = StandardScaler()
        features_train_normalized = pd.DataFrame(data_normalizer.fit_transform(features_train), columns=features_train.columns)
        features_test_normalized = data_normalizer.transform(features_test)

        # Handle class imbalance (Make 'Up' and 'Down' days equally important)
        scale_pos_weight = (len(target_train) - target_train.sum()) / target_train.sum()

        # Initialize the LightGBM model
        if Settings.GPU:
            hyperparams.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
        if Settings.Threaded:
            hyperparams.update({"num_threads": 1, "n_jobs": 1})

        model = LGBMClassifier(random_state=self.seed, scale_pos_weight=scale_pos_weight, verbose=-1, **hyperparams)

        # Walk-Forward Validation
        time_splitter = TimeSeriesSplit(n_splits=3)
        validation_scores = []
        for train_index, val_index in time_splitter.split(features_train):
            # Train on the "past" segment, test on the "future" segment
            model.fit(features_train_normalized.iloc[train_index], target_train.iloc[train_index])
            validation_scores.append(model.score(features_train_normalized.iloc[val_index], target_train.iloc[val_index]))

        # Final test on totally unseen data
        model.fit(features_train_normalized, target_train)
        test_predictions = model.predict(features_test_normalized)

        # Get the score
        accuracy, sharpe, abs_sharpe, needs_flip = self.evaluate_performance(target_test, test_predictions,price_returns)
        # Return all the information from the model
        return {'model_type': 'LGBM', 'accuracy': accuracy, 'walk_forward_accuracy': np.mean(validation_scores),
                'sharpe_ratio': sharpe, 'absolute_sharpe': abs_sharpe, 'logic_flipped': needs_flip,
                'raw_predictions': test_predictions, 'trained_model_object': model, 'feature_scaler': data_normalizer}

    # Train and evaluate Lasso Logistic Regression with walk-forward validation
    def _train_lasso_regression(self, hyperparams: dict, features_train: pd.DataFrame, target_train: pd.Series, features_test: pd.DataFrame, target_test: pd.Series, price_returns: np.ndarray) -> dict:
        # Put all indicators on the same scale (0 to 1 range)
        data_normalizer = StandardScaler()

        # Scale the training and test data while keeping the column names
        features_train_normalized = pd.DataFrame(data_normalizer.fit_transform(features_train), columns=features_train.columns)
        features_test_normalized = data_normalizer.transform(features_test)

        # Initialize Logistic Regression with a "Lasso" (L1) penalty
        model = LogisticRegression(
            solver='saga',
            random_state=self.seed,
            class_weight='balanced',  # Automatically handles Up/Down day imbalance
            **hyperparams
        )

        # Walk-Forward Validation
        time_splitter = TimeSeriesSplit(n_splits=3)
        validation_scores = []
        for train_idx, val_idx in time_splitter.split(features_train_normalized):
            model.fit(features_train_normalized.iloc[train_idx], target_train.iloc[train_idx])
            validation_scores.append(model.score(features_train_normalized.iloc[val_idx], target_train.iloc[val_idx]))

        # Final test on totally unseen data
        model.fit(features_train_normalized, target_train)
        test_predictions = model.predict(features_test_normalized)

        # Get the score
        accuracy, sharpe, abs_sharpe, needs_flip = self.evaluate_performance(target_test, test_predictions, price_returns)
        # Return all the information from the model
        return {'model_type': 'Lasso', 'accuracy': accuracy, 'walk_forward_accuracy': np.mean(validation_scores),
            'sharpe_ratio': sharpe, 'absolute_sharpe': abs_sharpe, 'logic_flipped': needs_flip,
            'raw_predictions': test_predictions, 'trained_model_object': model, 'feature_scaler': data_normalizer}

    # Train and evaluate SVC with walk-forward validation
    def _train_support_vector(self, hyperparams: dict, features_train: pd.DataFrame, target_train: pd.Series, features_test: pd.DataFrame, target_test: pd.Series, price_returns: np.ndarray) -> dict:
        # Put all indicators on the same scale (0 to 1 range) [SVC very sensitive to scale]
        data_normalizer = StandardScaler()

        # Scale the training and test data while keeping the column names
        features_train_normalized = pd.DataFrame(data_normalizer.fit_transform(features_train), columns=features_train.columns)
        features_test_normalized = data_normalizer.transform(features_test)

        # Initialize Support Vector model
        model = SVC(
            random_state=self.seed,
            class_weight='balanced', # Automatically handles Up/Down day imbalance
            probability=True, # To get probability scores
            **hyperparams
        )

        # Walk-Forward Validation
        time_splitter = TimeSeriesSplit(n_splits=3)
        validation_scores = []
        for train_idx, val_idx in time_splitter.split(features_train_normalized):
            model.fit(features_train_normalized.iloc[train_idx], target_train.iloc[train_idx])
            validation_scores.append(model.score(features_train_normalized.iloc[val_idx], target_train.iloc[val_idx]))

        # Final test on totally unseen data
        model.fit(features_train_normalized, target_train)
        test_predictions = model.predict(features_test_normalized)

        # Get the score
        accuracy, sharpe, abs_sharpe, needs_flip = self.evaluate_performance(target_test, test_predictions, price_returns)
        # Return all the information from the model
        return {
            'model_type': 'SVC', 'accuracy': accuracy, 'walk_forward_accuracy': np.mean(validation_scores),
            'sharpe_ratio': sharpe, 'absolute_sharpe': abs_sharpe, 'logic_flipped': needs_flip,
            'raw_predictions': test_predictions, 'trained_model_object': model, 'feature_scaler': data_normalizer}

    def _train_lstm(self, hyperparams: dict, features_train: pd.DataFrame, target_train: pd.Series, features_test: pd.DataFrame, target_test: pd.Series, price_returns: np.ndarray) -> dict:

        lstm_model = LSTM(self.seed)
        return lstm_model.train(hyperparams, features_train, target_train, features_test, target_test, price_returns)

    # Save models for all horizons with the best performing model type
    def _save_model_assets(self, ticker: str, interval: str, training_data_end: pd.Timestamp, full_results: dict, features_data: pd.DataFrame, targets_dataframe: pd.DataFrame, all_hyperparameters) -> None:
        # Create the folder for this specific stock's model data to save
        save_folder = os.path.join(MODEL_DIR, f"{ticker}_{interval}")
        if not os.path.exists(save_folder): os.makedirs(save_folder)

        scaler = StandardScaler().fit(features_data)
        scaled_features = scaler.transform(features_data) # noqa

        meta_full = {
            "training date": datetime.now().strftime("%Y-%m-%d"),
            "training data end": str(training_data_end.strftime("%Y-%m-%d")),
        }
        for h, horizon_results in full_results.items():
            hyperparameters: list = all_hyperparameters[f"{h}{interval[1]}"]
            meta_full[str(h)] = {
                "best model": max(horizon_results, key=lambda x: x["absolute_sharpe"])["model_type"],
            }

            target_col = f'target_cls_{h}{interval[1]}'
            valid_idx = targets_dataframe[target_col].notna()
            x_final = scaled_features[valid_idx.values]
            y_final = targets_dataframe[target_col][valid_idx]

            for result in horizon_results:
                horizon_folder = os.path.join(save_folder, f"{h}_horizon_models")
                if not os.path.exists(horizon_folder): os.makedirs(horizon_folder)

                model_type = result["model_type"]
                model_file = f"{model_type}_model_{h}{interval[1]}"

                meta_full[str(h)][f"{model_type}_result"] = {
                    k: (v.item() if hasattr(v, 'item') else v)
                    for k, v in result.items()
                    if k not in {"raw_predictions", "trained_model_object", "feature_scaler"}
                }

                # Train Classifier
                if model_type == 'LSTM':
                    lstm_base = LSTM(self.seed)
                    x_3d, y_3d = lstm_base.create_3d_sequences(x_final, y_final)

                    model = lstm_base.get_lstm_competitor(input_dim=x_3d.shape[2],
                                                          hyperparams=next((hypers["best_params"] for hypers in hyperparameters if hypers["model_type"] == "LSTM"), {}))
                    model.fit(x_3d, y_3d)
                    save_file(model.module_.state_dict(), os.path.join(horizon_folder, f"{model_file}.safetensors"))

                elif model_type == 'LGBM':
                    up_days = y_final.sum()
                    down_days = len(y_final) - up_days
                    spw = down_days / up_days if up_days > 0 else 1.0

                    lgbm_hypers = next((hypers["best_params"] for hypers in hyperparameters if hypers["model_type"] == "LGBM"), {})
                    if Settings.GPU:
                        lgbm_hypers.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
                    if Settings.Threaded:
                        lgbm_hypers.update({"num_threads": 1, "n_jobs": 1})

                    model = LGBMClassifier(random_state=self.seed, scale_pos_weight=spw, verbose=-1, **lgbm_hypers)
                    model.fit(x_final, y_final)
                    model.booster_.save_model(os.path.join(horizon_folder, f"{model_file}.txt"))

                else:
                    if model_type == 'Lasso':
                        model = LogisticRegression(solver='saga', random_state=self.seed, class_weight='balanced',
                                                   **next((hypers["best_params"] for hypers in hyperparameters if hypers["model_type"] == "Lasso"), {}))
                    else:  # SVC
                        model = SVC(random_state=self.seed, class_weight='balanced', probability=True,
                                    **next((hypers["best_params"] for hypers in hyperparameters if hypers["model_type"] == "SVC"), {}))

                    model.fit(x_final, y_final)
                    joblib.dump(model, os.path.join(horizon_folder, f"{model_file}.joblib"))

        with open(os.path.join(save_folder, 'metadata.json'), 'w') as f:
            json.dump(meta_full, f, indent=4)
        joblib.dump(scaler, os.path.join(save_folder, "scaler.joblib"))
        joblib.dump(list(features_data.columns), os.path.join(save_folder, "features.joblib"))

    # Run all helper functions and consolidate the best model
    def run_training_pipeline(self, ticker: str, interval: str, override_data: pd.DataFrame = None, status_signal: tuple = None, force_train: bool = False) -> bool:
        def log_update(msg):
            if status_signal:
                # Expect status_signal to be (update_queue, core_key)
                u_queue, core_key = status_signal
                u_queue.put((core_key, {"Current Task": msg}))
            elif Settings.LOGGING:
                print(msg)

        model_path = os.path.join(MODEL_DIR, f"{ticker}_{interval}")
        if all_ticker_models_exist(model_path, interval) and not force_train:
            log_update(f"Model {ticker} is already fully trained")
            return True

        # Remove any corrupt model paths
        if os.path.exists(model_path): shutil.rmtree(model_path)

        # Train and build models for ticker
        # Load data and add indicators
        log_update("Loading data...")
        data = override_data if override_data is not None else load_data(ticker, interval)
        if data is None: print("No data"); return False

        log_update("Adding features...")
        df = data.ind.add_indicators(ticker, interval)
        if len(df) < 300: print(f"Insufficient data for {ticker} ({interval}) - need 300+, got {len(df)}"); return False

        # Get best hyperparameters
        with open(os.path.join(DATA_DIR, "model_hyperparameters.json"), "r") as f:
            all_hyperparameters = json.load(f)

        log_update("Preparing data...")
        # Partition data
        train_size = int(len(df) * (1 - self.__test_size))
        train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

        # Remove "Cheat" columns and raw price data AI shouldn't see directly
        # ('target' as is answer key, 'Close' as is too easy to cheat with)
        drop_columns = [c for c in df.columns if 'target' in c or c in ['Open', 'High', 'Low', 'Close', "Adj Close", 'Volume', 'MA_200', 'return']]
        train_columns = [c for c in df.columns if c not in drop_columns]

        # Create input features and answers
        period = "h" if 'h' in interval else "d"
        features_train, features_test = train_data[train_columns], test_data[train_columns]

        results = {}

        for h in ([1,2,4,8] if period == "h" else [1,2,5,21]):
            log_update(f"Training for horizon {h}...")
            hyperparameters: list = all_hyperparameters[f"{h}{interval[1]}"]
            targets_train, targets_test = train_data[f'target_cls_{h}{period}'], test_data[f'target_cls_{h}{period}']
            actual_returns_test = test_data['return'].values

            # Model competition
            log_update(f"Training LightGBM ({h}{period})")
            lgbm = self._train_lightgbm(next((hypers["best_params"] for hypers in hyperparameters if hypers["model_type"] == "LGBM"), None), features_train, targets_train, features_test, targets_test, actual_returns_test)

            log_update(f"Training Lasso ({h}{period})")
            lasso = self._train_lasso_regression(next((hypers["best_params"] for hypers in hyperparameters if hypers["model_type"] == "Lasso"), None), features_train, targets_train, features_test, targets_test, actual_returns_test)

            log_update(f"Training SVC ({h}{period})")
            svc = self._train_support_vector(next((hypers["best_params"] for hypers in hyperparameters if hypers["model_type"] == "SVC"), None), features_train, targets_train, features_test, targets_test, actual_returns_test)

            log_update(f"Training LSTM ({h}{period})")
            lstm = self._train_lstm(next((hypers["best_params"] for hypers in hyperparameters if hypers["model_type"] == "LSTM"), None), features_train, targets_train, features_test, targets_test, actual_returns_test)

            horizon_results = [lgbm, lasso, svc, lstm]
            for r in horizon_results:
                # A model is 'Stable' if the test accuracy is close to the walk-forward accuracy
                stability = abs(r['accuracy'] - r['walk_forward_accuracy'])
                r["stability"] = stability

            results[h] = horizon_results

        # Save winning model data
        log_update("Saving assets")
        self._save_model_assets(ticker, interval, df.index.max(), results, features_train, train_data, all_hyperparameters)
        return True

############################################################################

# Save prediction to ledger
def save_prediction(ticker: str, interval: str, forecast_results: dict) -> None:
    # Check if that exact prediction has already been saved (note: using same model will always return the same prediction for the same data)
    if len(forecast_results) < 1: return

    # Iterate through the 3 horizons the model predicted and extract the necessary information
    ledger_file = os.path.join(LEDGER_DIR, f"{ticker}_ledger.csv")
    new_entries = []
    for step, data in forecast_results.items():
        new_entries.append({
            "Interval": interval,
            'Horizon': f"{data['time_difference']}{interval[1]}",
            'Open_Date': data["Date_Predicted"],
            'Target_Date': data['Target_Date'].strftime('%Y-%m-%d %H:%M'),
            "Current_Price": round(data['Current_Price'], 2),
            'Predicted_Price': round(data['Predicted_Price'], 2),
            'Actual_Price': np.nan,
            'Predicted_Max_Price': round(data['up'], 2),
            'Predicted_Min_Price': round(data['lo'], 2),
            'LSTM_signal': f"{data['LSTM_signal']:.3f}",
            'LGBM_signal': f"{data['LGBM_signal']:.3f}",
            'SVC_signal': f"{data['SVC_signal']:.3f}",
            'LASSO_signal': f"{data['LASSO_signal']:.3f}",
            'AVG_signal': f"{data['AVG_signal']:.3f}",
        })

    # Add prediction data to the ledger
    df_new = pd.DataFrame(new_entries)
    if not os.path.exists(ledger_file): df_new.to_csv(ledger_file, index=False)
    else: df_new.to_csv(ledger_file, mode='a', header=False, index=False)

# Load prediction from ledger
def load_prediction(ticker: str, interval: str, date: datetime) -> dict | None:
    ledger_file = os.path.join(LEDGER_DIR, f"{ticker}_ledger.csv")

    try:
        ledger = pd.read_csv(ledger_file)
        if ledger.empty or len(ledger) < 1: return None
    except FileNotFoundError: return None

    # Filter for the specific data
    date = date.strftime("%Y-%m-%d %H:%M")
    match = ledger[(ledger['Interval'] == interval) & (ledger['Open_Date'] == date)]
    if match.empty: return None
    match_dicts = match.reset_index().to_dict(orient='records')

    # Rebuild the forecast_results dict
    try:
        forecast_results = {}
        for i, step in enumerate([1, 2, 4, 8] if "h" in interval else [1, 2, 5, 21]):
            forecast_results[step] = {
                "Date_Predicted": pd.to_datetime(match_dicts[i]['Open_Date'], format="ISO8601"),
                'Target_Date': pd.to_datetime(match_dicts[i]['Target_Date'], format='ISO8601'),
                "Current_Price": float(match_dicts[i]['Current_Price']),
                'Predicted_Price': float(match_dicts[i]['Predicted_Price']),
                'up': float(match_dicts[i]['Predicted_Max_Price']),
                'lo': float(match_dicts[i]['Predicted_Min_Price']),
                'time_difference': int(match_dicts[i]['Horizon'][:-1]),
                'LSTM_signal': float(match_dicts[i]['LSTM_signal']),
                'LGBM_signal': float(match_dicts[i]['LGBM_signal']),
                'SVC_signal': float(match_dicts[i]['SVC_signal']),
                'LASSO_signal': float(match_dicts[i]['LASSO_signal']),
                'AVG_signal': float(match_dicts[i]['AVG_signal']),
            }
    except Exception: return None

    return forecast_results

# Checks if there exists an entry in the ledger for those details
def prediction_saved(ticker: str, interval: str, date: datetime, horizons: list[str]) -> bool:
    ledger_file = os.path.join(LEDGER_DIR, f"{ticker}_ledger.csv")
    if not os.path.exists(ledger_file): return False

    ledger = pd.read_csv(ledger_file)
    ledger['Open_Date'] = pd.to_datetime(ledger['Open_Date'], format='ISO8601')

    # Check if any entry matches current ticker and last trade date
    match = ledger[(ledger['Interval'] == interval) & (ledger['Open_Date'] == date) & (ledger["Horizon"].isin(horizons))]
    return not match.empty

# Run all helper functions to display a prediction
def run_prediction_pipeline(ticker: str, interval: str) -> dict:
    # Add in technical indicators
    processed_df, assets = prepare_prediction_data(ticker, interval)
    if any(v is None for v in [processed_df, assets]):
        print("No data or assets.")
        return {}

    last_trade_date = processed_df.index[-1]

    # Load or create and save the prediction
    forecast_results = load_prediction(ticker, interval, last_trade_date)
    if forecast_results is None:
        # Create a dict with basic information about the state of the prediction and stock
        is_hour = "h" in interval
        tech_info = (
            {1:1, 2:2, 4:4, 8:25} if is_hour else {1:1, 2:2, 5:7, 21:28}, # horizons
            "h" if is_hour else "d", # period
            last_trade_date,
            float(processed_df['Adj Close'].iloc[-1]), # current_price
        )

        # Generate a prediction
        forecast_results = generate_forecasts(processed_df, assets, tech_info)
        save_prediction(ticker, interval, forecast_results)

    return forecast_results

def all_ticker_models_exist(ticker_dir: str, interval: str) -> bool:
    root = Path(ticker_dir)
    if not root.exists():
        return False

    required_root = ["features.joblib", "metadata.json", "scaler.joblib"]
    for file in required_root:
        if not (root / file).exists():
            return False

    horizons = [1, 2, 4, 8] if 'h' in interval else [1, 2, 5, 21]
    model_types = {
        "LGBM":  ".txt",
        "LSTM":  ".safetensors",
        "Lasso": ".joblib",
        "SVC":   ".joblib"
    }

    for h in horizons:
        h_folder = root / f"{h}_horizon_models"

        if not h_folder.exists():
            return False

        for model_name, ext in model_types.items():
            # Matches format: LGBM_model_1h.txt, etc.
            expected_file = f"{model_name}_model_{h}{interval[1]}{ext}"
            file_path = h_folder / expected_file

            if not file_path.exists():
                return False
            elif file_path.stat().st_size == 0:
                return False

    return True

# Adds in technical indicators, trains model if needed or loads it
def prepare_prediction_data(ticker: str, interval: str) -> tuple:
    model_path = os.path.join(MODEL_DIR, f"{ticker}_{interval}")
    manager = TrainingManager()

    # Ensure data exists
    df = load_data(ticker, interval)
    if df is None: print("No data"); return None, None

    # If its hourly data, and market has just opened, add in the day opening price
    now_utc = pd.Timestamp.now(tz='UTC')
    schedule = NYSE_CAL.schedule(start_date=now_utc, end_date=now_utc)
    if not schedule.empty:
        market_open = schedule.iloc[0]['market_open']
        if interval == "1h" and market_open <= now_utc <= (market_open + timedelta(hours=1)):

            temp_data = yf.download(ticker, period="1d", interval="1d", progress=False)
            today_open = float(temp_data['Open'].values[-1][0])

            new_row = pd.DataFrame({
                'Open': [today_open], 'High': [today_open],
                'Low': [today_open], 'Close': [today_open], 'Volume': [0], "Adj Close": [today_open]
            }, index=[(market_open-timedelta(hours=1)).tz_localize(None)])

            df = pd.concat([df, new_row])

    # Trains a model if needed
    if not all_ticker_models_exist(model_path, interval):
        # print("Empty or missing model files. Training...")
        success = manager.run_training_pipeline(ticker, interval)
        if not success:
            # print("Failed to train models.")
            return None, None

    # Load assets
    processed_df = df.ind.add_indicators(ticker, interval)
    if processed_df.empty:
        # print("Failed to calculate technical indicators.")
        return None, None

    scaler = joblib.load(f"{model_path}/scaler.joblib")
    features = joblib.load(f"{model_path}/features.joblib")

    return processed_df, (scaler, features, model_path)

def get_market_dates(latest_date, horizons, period):
    market_targets = {}

    # latest_date is the OPENING time
    if latest_date.tz is None:
        latest_date = latest_date.tz_localize('UTC')

    end_search = latest_date + pd.Timedelta(days=35)
    schedule = NYSE_CAL.schedule(start_date=latest_date, end_date=end_search)

    if period == "d":
        # For daily: List of DAYs (e.g. '2026-03-06')
        valid_times = schedule.index.normalize()
    else:
        # For hourly: List of CLOSING times of the HOUR (e.g. 15:30 to 21:00)
        valid_times = mcal.date_range(schedule, frequency="1h")

    for step, time_dif in horizons.items():
        if period == "h":
            target_dt = latest_date + timedelta(hours=(time_dif+1))

            if target_dt not in valid_times:
                if time_dif in [4, 25] and (target_dt - timedelta(hours=1)) in valid_times:
                    target_dt = target_dt - timedelta(hours=1)
                else: target_dt = None

        else:
            target_dt = latest_date + timedelta(days=time_dif)

            # Keep rolling forward if it lands on a weekend or holiday
            while target_dt.strftime("%Y-%m-%d") not in valid_times:
                target_dt += timedelta(days=1)

                if (target_dt - latest_date).days > 35:
                    target_dt = None
                    break

        market_targets[step] = target_dt

    return market_targets

# Predict the price movement
def generate_forecasts(processed_df: pd.DataFrame, assets: tuple, tech_info: tuple) -> dict:
    scaler, features, model_folder = assets
    horizons, period, last_trade_date, current_price =  tech_info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join(model_folder, 'metadata.json'), 'r') as f:
        meta = json.load(f)

    with open(os.path.join(DATA_DIR, "model_hyperparameters.json"), 'r') as f:
        hyper_meta = json.load(f)

    current_volatility_atr = float(processed_df['ATR'].iloc[-1])
    forecast_results = {}

    target_dates = get_market_dates(last_trade_date, horizons, period)
    if len(target_dates) < 1: return {}

    # Calculate forecastSs
    for step, actual_time in horizons.items():
        if target_dates[step] is None: continue

        probs = {"LSTM": 0.5, "LGBM": 0.5, "SVC": 0.5, "Lasso": 0.5}
        weights = {"LSTM": 0, "LGBM": 0, "SVC": 0, "Lasso": 0}
        global_meta: list = hyper_meta[f"{step}{period}"]

        horizon_folder = os.path.join(model_folder, f"{step}_horizon_models")
        for model_filename in os.listdir(horizon_folder):
            model_path = os.path.join(horizon_folder, model_filename)

            if ".safetensors" in model_filename:
                model_type = "LSTM"
                recent_data = processed_df[features].tail(14)
                scaled_seq = scaler.transform(recent_data)

                # Convert to 3D: (1, 14, num_features)
                x_3d = np.expand_dims(scaled_seq, axis=0).astype(np.float32)

                params = next(m["best_params"] for m in global_meta if m["model_type"] == model_type)
                brain = LSTMBrain(
                    input_dim=len(features),
                    hidden_dim=params["module__hidden_dim"],
                    layers=params["module__layers"],
                    dropout=params["module__dropout"],
                )

                state_dict = load_file(model_path)
                brain.load_state_dict(state_dict)
                brain.to(device)
                brain.eval()

                with torch.no_grad():
                    probs[model_type] = 1 - float(brain(torch.from_numpy(x_3d)).item())

            elif ".txt" in model_filename:
                model_type = "LGBM"
                scaled_row = scaler.transform(processed_df[features].iloc[-1:])
                booster = LGBMBooster(model_file=model_path)
                probs[model_type] = 1 - float(booster.predict(scaled_row)[0])

            else: # Joblib (Lasso or SVC)
                scaled_row = scaler.transform(processed_df[features].iloc[-1:])
                model = joblib.load(model_path)
                if "SVC" in model_filename:
                    model_type = "SVC"
                    probs[model_type] = 1 - float(model.predict_proba(scaled_row)[0][1])
                else:
                    model_type = "Lasso"
                    probs[model_type] = 1 - float(model.predict_proba(scaled_row)[0][1])

            mcc_val = next(m["mcc"] for m in global_meta if m["model_type"] == model_type)
            if mcc_val > 0:
                ticker_weight = meta.get(str(step), {}).get(f"{model_type}_result", {}).get("absolute_sharpe", 0)
                global_weight = next(abs(m["sharpe_ratio"]) for m in global_meta if m["model_type"] == model_type)
                results_weight = next(w for m, w in {"LGBM": 0.4, "SVC": 0.4, "Lasso": 0.1, "LSTM": 0.1}.items() if m == model_type)

                weights[model_type] = (results_weight * 0.5) + (ticker_weight * 0.3) + (global_weight * 0.2)

        total_weight = sum(weights.values())
        avg_up_proba = sum(probs[m] * weights[m] for m in probs) / total_weight if total_weight > 0 else 0.5

        # Calculate predicted price
        direction_multiplier = 1 if avg_up_proba > 0.5 else -1
        confidence_strength = 2 * (max(avg_up_proba, 1 - avg_up_proba) - 0.5)
        expected_move_magnitude = current_volatility_atr * np.sqrt(step)

        predicted_price = current_price + (direction_multiplier * expected_move_magnitude * confidence_strength)
        capped_width = min(expected_move_magnitude * (1.0 + confidence_strength), current_price * 0.15)

        forecast_results[step] = {
            "Date_Predicted": last_trade_date.strftime("%Y-%m-%d %H:%M"),
            'Target_Date': target_dates[step],
            "Current_Price": current_price,
            'Predicted_Price': predicted_price,
            'up': predicted_price + capped_width,
            'lo': predicted_price - capped_width,
            'time_difference': actual_time,
            'LSTM_signal': probs["LSTM"],
            'LGBM_signal': probs["LGBM"],
            'SVC_signal': probs["SVC"],
            'LASSO_signal': probs["Lasso"],
            'AVG_signal': avg_up_proba,
        }

    return forecast_results

############################################################################

