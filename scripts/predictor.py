
# Standard library imports
import json
import os
from pathlib import Path
import shutil
import warnings
from datetime import datetime

# External library imports
import joblib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from lightgbm import LGBMClassifier
from PyQt6.QtCore import QThread, pyqtSignal
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
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
    VERBOSE = 0      # Set whether to display model logging or not
    LOGGING = False  # Set whether to display prints for training stages
    GPU = True       # Set whether to use GPU if possible
    Threaded = False # Set whether a singular model is being trained on multiple threads

########################################################################################################################

class LSTMBrain(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2, output_dim: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        return self.fc(out)


class TrainingManager:
    def __init__(self):
        self.seed = 69
        self.__test_size = 0.2

    @staticmethod
    def evaluate_performance(actual: np.ndarray, predicted: np.ndarray, actual_returns: np.ndarray) -> tuple:
        hit_rate = accuracy_score(actual, predicted)

        # Simulate trading: multiply signal by the bar's actual return
        # -1=short mirrors the return, 0=flat earns nothing, 1=long earns the return
        strategy_returns = actual_returns * predicted
        volatility = strategy_returns.std()
        sharpe = (strategy_returns.mean() / volatility) * np.sqrt(252) if volatility != 0 else 0.0

        return hit_rate, sharpe

    def score(self, model, X_train, y_train, val_idx, actual_returns_train):
        preds = model.predict(X_train[val_idx]) - 1

        fold_acc, fold_sharpe = self.evaluate_performance(
            y_train.iloc[val_idx].values,
            preds,
            actual_returns_train[val_idx]
        )

        return fold_sharpe

    @staticmethod
    def create_3d_sequences(data: np.ndarray, targets: np.ndarray, window: int = 30) -> tuple:
        x, y = [], []
        for i in range(len(data) - window):
            x.append(data[i: i + window])
            y.append(targets[i + window])
        return np.array(x, dtype=np.float32), np.array(y)

    def _train_lgbm(self, hyperparams: dict, X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray,  y_test: pd.Series, actual_returns_test: np.ndarray) -> dict:
        lgbm_params = hyperparams.copy()
        lgbm_params.update(dict(
            objective='multiclass',
            num_class=3,
            n_estimators=200, # Will change with hyperparameter tuning
            class_weight='balanced',
            random_state=self.seed,
            verbose=-1
        ))
        if Settings.GPU:
            lgbm_params.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
        if Settings.Threaded:
            lgbm_params.update({"num_threads": 1, "n_jobs": 1})

        # LightGBM multiclass requires labels 0-indexed, so shift {-1,0,1} -> {0,1,2}
        y_train_shifted = y_train + 1

        model = LGBMClassifier(**lgbm_params)

        # Walk-forward validation on training data only (no peeking at test set)
        time_splitter = TimeSeriesSplit(n_splits=3)
        wf_scores = []
        for train_idx, val_idx in time_splitter.split(X_train):
            model.fit(X_train[train_idx], y_train_shifted.iloc[train_idx])
            wf_scores.append(self.score(model, X_train, y_train, val_idx, actual_returns_test))

        # Retrain on full training set, then evaluate on the held-out test set
        model.fit(X_train, y_train_shifted)
        test_preds = model.predict(X_test) - 1  # Shift back to {-1,0,1} for evaluation

        accuracy, sharpe = self.evaluate_performance(y_test.values, test_preds, actual_returns_test)
        wf_mean = float(np.mean(wf_scores))

        return {
            'type': 'lgbm', 'model': model,
            'accuracy': accuracy, 'walk_forward_accuracy': wf_mean,
            'sharpe': sharpe, 'stability': abs(accuracy - wf_mean),
        }

    def _train_catboost(self, hyperparams: dict, X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray,  y_test: pd.Series, actual_returns_test: np.ndarray) -> dict:
        cat_params = hyperparams.copy()
        model = CatBoostClassifier(
            iterations=500, # Will change with hyperparameter tuning
            loss_function='MultiClass',
            auto_class_weights='Balanced',
            task_type="GPU" if Settings.GPU else "CPU",
            verbose=False,
            random_state=self.seed,
            **cat_params
        )

        y_train_shifted = y_train + 1

        time_splitter = TimeSeriesSplit(n_splits=3)
        wf_scores = []
        for train_idx, val_idx in time_splitter.split(X_train):
            model.fit(X_train[train_idx], y_train_shifted.iloc[train_idx])
            wf_scores.append(self.score(model, X_train, y_train, val_idx, actual_returns_test))

        model.fit(X_train, y_train_shifted)
        test_preds = model.predict(X_test).flatten() - 1  # Shift back to {-1,0,1}

        accuracy, sharpe = self.evaluate_performance(y_test.values, test_preds, actual_returns_test)
        wf_mean = float(np.mean(wf_scores))

        return {
            'type': 'cat', 'model': model,
            'accuracy': accuracy, 'walk_forward_accuracy': wf_mean,
            'sharpe': sharpe, 'stability': abs(accuracy - wf_mean),
        }

    def _train_lstm(self, hyperparams: dict, X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray,  y_test: pd.Series, actual_returns_test: np.ndarray) -> dict:
        lstm_params = hyperparams.copy()

        # Build (samples, window, features) sequences from the flat scaled arrays
        x_train_3d, y_train_seq = self.create_3d_sequences(X_train, y_train.values)
        x_test_3d,  y_test_seq  = self.create_3d_sequences(X_test,  y_test.values)

        # Shift targets to {0,1,2} for CrossEntropyLoss; keep originals for evaluate_performance
        y_train_shifted = (y_train_seq + 1).astype(np.int64)

        input_dim = x_train_3d.shape[2]

        def build_skorch_model() -> NeuralNetClassifier:
            return NeuralNetClassifier(
                LSTMBrain,
                module__input_dim=input_dim,
                module__hidden_dim=64, # Will change with hyperparameter tuning
                module__num_layers=2,
                module__dropout=0.2, # Will change with hyperparameter tuning
                module__output_dim=3,
                criterion=nn.CrossEntropyLoss,
                optimizer=torch.optim.Adam, # Will change with hyperparameter tuning
                lr=0.001, # Will change with hyperparameter tuning
                max_epochs=50, # Will change with hyperparameter tuning
                train_split=dataset.ValidSplit(0.2, stratified=False),
                iterator_train__shuffle=False,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=Settings.VERBOSE,
                callbacks=[
                    ('early_stopping', EarlyStopping(
                        monitor='valid_loss',
                        patience=5,
                        lower_is_better=True
                    )),
                    ('val_acc', EpochScoring(
                        scoring='accuracy',
                        name='valid_acc',
                        lower_is_better=False
                    ))
                ],
            **lstm_params
            )

        # Walk-forward validation across folds of the training sequences
        time_splitter = TimeSeriesSplit(n_splits=3)
        wf_scores = []
        for train_idx, val_idx in time_splitter.split(x_train_3d):
            fold_model = build_skorch_model()
            fold_model.fit(x_train_3d[train_idx], y_train_shifted[train_idx])
            wf_scores.append(self.score(fold_model, X_train, y_train, val_idx, actual_returns_test))

        # Final model trained on all training sequences
        model = build_skorch_model()
        model.fit(x_train_3d, y_train_shifted)

        # Shift predictions back to {-1,0,1} for evaluation
        test_preds = model.predict(x_test_3d) - 1

        # Align returns with the sequence offset (first LSTM_WINDOW bars are consumed as context)
        accuracy, sharpe = self.evaluate_performance(y_test_seq, test_preds, actual_returns_test[30:])
        wf_mean = float(np.mean(wf_scores)) if wf_scores else 0.0

        return {
            'type': 'lstm', 'model': model,
            'accuracy': accuracy, 'walk_forward_accuracy': wf_mean,
            'sharpe': sharpe, 'stability': abs(accuracy - wf_mean),
        }

    def _save_model_assets(self, ticker: str, interval: str, training_data_end: pd.Timestamp, results: list, feature_columns: list, scaler: StandardScaler) -> None:

        save_folder = os.path.join(MODEL_DIR, f"{ticker}_{interval}")
        if not os.path.exists(save_folder): os.makedirs(save_folder)

        metadata = {
            "training_date":      datetime.now().strftime("%Y-%m-%d"),
            "training_data_end":  training_data_end.strftime("%Y-%m-%d"),
            "models":             {}
        }

        for result in results:
            model_type = result['type']
            model = result['model']

            metadata["models"][model_type] = {
                "accuracy":              result['accuracy'],
                "walk_forward_accuracy": result['walk_forward_accuracy'],
                "sharpe":                result['sharpe'],
                "stability":             result['stability'],
            }

            if model_type == 'lstm':
                save_file(model.module_.state_dict(), os.path.join(save_folder, "lstm_model.safetensors"))
            elif model_type == 'lgbm':
                model.booster_.save_model(os.path.join(save_folder, "lgbm_model.txt"))
            else:
                joblib.dump(model, os.path.join(save_folder, f"{model_type}_model.joblib"))

        with open(os.path.join(save_folder, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

        joblib.dump(scaler, os.path.join(save_folder, "scaler.joblib"))
        joblib.dump(feature_columns, os.path.join(save_folder, "features.joblib"))

    def run_training_pipeline(self, ticker: str, interval: str, override_data: pd.DataFrame = None, status_signal: tuple = None, force_train: bool = False) -> bool:
        def log_update(msg):
            if status_signal:
                u_queue, core_key = status_signal
                u_queue.put((core_key, {"Current Task": msg}))
            elif Settings.LOGGING:
                print(msg)

        model_path = os.path.join(MODEL_DIR, f"{ticker}_{interval}")
        if all_ticker_models_exist(model_path) and not force_train:
            log_update(f"Model {ticker} ({interval}) already trained")
            return True

        if os.path.exists(model_path): shutil.rmtree(model_path)

        log_update("Loading data...")
        data = override_data if override_data is not None else load_data(ticker, interval)
        if data is None: print("No data"); return False

        hyperparams = {}

        log_update("Adding features...")
        df = data.ind.add_indicators(ticker, interval)
        if len(df) < 300:
            print(f"Insufficient data for {ticker} ({interval}) — need 300+, got {len(df)}")
            return False

        # print(df['target_profit'].value_counts(normalize=True))
        # exit()

        train_size = int(len(df) * (1 - self.__test_size))
        train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

        drop_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'MA_200', 'return', 'target_profit']
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X_train_raw, X_test_raw = train_df[feature_cols], test_df[feature_cols]
        y_train, y_test = train_df['target_profit'], test_df['target_profit']
        actual_returns_test = test_df['return'].values

        log_update("Scaling features...")
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test  = scaler.transform(X_test_raw)

        log_update("Training LGBM...")
        lgbm_result = self._train_lgbm(hyperparams, X_train, y_train, X_test, y_test, actual_returns_test)

        log_update("Training CatBoost...")
        cat_result = self._train_catboost(hyperparams, X_train, y_train, X_test, y_test, actual_returns_test)

        log_update("Training LSTM...")
        lstm_result = self._train_lstm(hyperparams, X_train, y_train, X_test, y_test, actual_returns_test)

        results = [lgbm_result, cat_result, lstm_result]

        log_update("Saving assets...")
        self._save_model_assets(ticker, interval, df.index.max(), results, feature_cols, scaler)
        return True

########################################################################################################################

def all_ticker_models_exist(model_path: str) -> bool:
    root = Path(model_path)
    if not root.exists(): return False

    required = [
        "lgbm_model.txt",
        "cat_model.joblib",
        "lstm_model.safetensors",
        "scaler.joblib",
        "features.joblib",
        "metadata.json",
    ]
    return all((root / f).exists() and (root / f).stat().st_size > 0 for f in required)

########################################################################################################################

if __name__ in "__main__":
    Settings.LOGGING = True
    trainer = TrainingManager()
    trainer.run_training_pipeline("AAPL", "1d", force_train=True)