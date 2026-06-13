# Standard library imports
import json
import time
import warnings
import os

# External library imports
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier
import optuna
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback
from scipy import stats # noqa
import gc

# Set environment variables and filters
warnings.filterwarnings("ignore")
NYSE_CAL = mcal.get_calendar('NYSE')

# Custom imports
from scripts.config import DATA_DIR
import scripts.indicators_tuning # noqa

class Settings:
    VERBOSE = 1 # Set whether to display logging or not
    GPU     = True

def flush_memory():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

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

# Custom Skorch callback to report intermediate validation metrics to Optuna for pruning
class OptunaPruningCallback(Callback):
    def __init__(self, trial, monitor: str = 'train_loss'):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, net, **kwargs):
        epoch = len(net.history) - 1
        current_val = net.history[-1, self.monitor]
        self.trial.report(current_val, epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()

# Class to control and train models
class TrainingManager:
    def __init__(self):
        self.seed = 69
        self.__test_size = 0.2

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.returns_train = None
        self.returns_test = None

    def _prepare_data(self, df: pd.DataFrame):
        drop_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'MA_200', 'return', 'target_profit', 'tbm_return']
        feature_cols = [c for c in df.columns if c not in drop_cols]

        # Partition the fully processed data
        split_idx = int(len(df) * (1 - self.__test_size))
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

        X_train_raw = train_df[feature_cols].values
        X_test_raw = test_df[feature_cols].values

        self.y_train,       self.y_test       = train_df['target_profit'],     test_df['target_profit']
        self.returns_train, self.returns_test = train_df['tbm_return'].values, test_df['tbm_return'].values

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train_raw)
        self.X_test = scaler.transform(X_test_raw)

    @staticmethod
    def evaluate_performance(interval: str, actual: np.ndarray, predicted: np.ndarray, actual_returns: np.ndarray) -> tuple:
        actual = np.asarray(actual)
        predicted = np.asarray(predicted).reshape(-1)
        actual_returns = np.asarray(actual_returns).reshape(-1)

        acc = accuracy_score(actual, predicted)

        # Strategy return aligns with the direction predicted:
        # Long (1) gets the path return, Short (-1) gets inverted path return, Hold (0) gets 0
        strategy_returns = np.nan_to_num(actual_returns * predicted)

        if len(strategy_returns) < 2 or np.std(strategy_returns) == 0:
            sharpe = 0.0
        else:
            if interval == "1d": bars = 252
            elif interval == "1h": bars = 1638
            else: raise NotImplementedError("Interval length not implemented yet.")

            sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * np.sqrt(bars)

        # Calculate stability
        cumulative_returns = np.cumsum(strategy_returns)
        if len(cumulative_returns) > 1:
            x = np.arange(len(cumulative_returns))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, cumulative_returns)
            stability = r_value ** 2 if slope > 0 else 0.0
        else:
            stability = 0.0

        return acc, sharpe, stability

    @staticmethod
    def create_3d_sequences(data: np.ndarray, targets: np.ndarray, window: int = 30) -> tuple:
        x, y = [], []
        for i in range(len(data) - window):
            x.append(data[i: i + window])
            y.append(targets[i + window])
        return np.array(x, dtype=np.float32), np.array(y)

    # Train and tune LightGBM with walk-forward validation
    def _train_lightgbm(self, interval: str, horizon: int) -> dict:
        y_train_shifted = self.y_train + 1
        tscv = PurgedTimeSeriesSplit(n_splits=3, gap=horizon, embargo_pct=0.01)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 15, 255),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self.seed,
                'objective': 'multiclass',
                'num_class': 3,
                'class_weight': 'balanced',
                'verbose': -1,
                'n_jobs': -1,  # Maximize local CPU efficiency
                'device_type': 'gpu' if Settings.GPU else "cpu"
            }

            model = LGBMClassifier(**params)
            scores = []

            for train_idx, val_idx in tscv.split(self.X_train):
                model.fit(self.X_train[train_idx], y_train_shifted.iloc[train_idx])
                preds = model.predict(self.X_train[val_idx]) - 1

                # Penalize models that take zero trades to close the safe haven loophole
                if np.all(preds == 0):
                    local_sharpe = -1.0
                else:
                    # Optimize for Sharpe Ratio
                    _, local_sharpe, _ = self.evaluate_performance(
                        interval, self.y_train.iloc[val_idx].values, preds, self.returns_train[val_idx]
                    )
                scores.append(local_sharpe)
            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30) # noqa

        best_params = study.best_params.copy()
        best_params.update({
            'objective': 'multiclass', 'num_class': 3,
            'class_weight': 'balanced', 'random_state': self.seed,
            'verbose': -1, 'device_type': 'gpu' if Settings.GPU else "cpu"
        })

        best_model = LGBMClassifier(**best_params)
        best_model.fit(self.X_train, y_train_shifted)

        test_preds = best_model.predict(self.X_test) - 1

        # print("Long:", np.mean(test_preds == 1))    # noqa
        # print("Short:", np.mean(test_preds == -1))  # noqa
        # print("Hold:", np.mean(test_preds == 0))    # noqa
        # print(confusion_matrix(self.y_test, test_preds))

        accuracy, sharpe, stability = self.evaluate_performance(interval, self.y_test.values, test_preds, self.returns_test)
        return {
            'model_type': 'LGBM',
            'accuracy': accuracy,
            'wf_sharpe': study.best_value,
            'sharpe': sharpe,
            'stability': stability,
            'best_params': best_params
        }

    # Train and tune CatBoost with walk-forward validation
    def _train_catboost(self, interval: str, horizon: int) -> dict:
        y_train_shifted = self.y_train + 1
        tscv = PurgedTimeSeriesSplit(n_splits=3, gap=horizon, embargo_pct=0.01)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 150, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 20.0, log=True),
                'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'loss_function': 'MultiClass',
                'auto_class_weights': 'Balanced',
                'task_type': "CPU",
                'verbose': False,
                'random_state': self.seed
            }

            model = CatBoostClassifier(**params)
            scores = []

            for train_idx, val_idx in tscv.split(self.X_train):
                if y_train_shifted.iloc[train_idx].nunique() <= 1:
                    local_sharpe = -1.0
                else:
                    model.fit(self.X_train[train_idx], y_train_shifted.iloc[train_idx])
                    preds = model.predict(self.X_train[val_idx]).flatten() - 1

                    # Penalize models that take zero trades to close the safe haven loophole
                    if np.all(preds == 0):
                        local_sharpe = -1.0
                    else:
                        # Optimize for Sharpe Ratio
                        _, local_sharpe, _ = self.evaluate_performance(
                            interval, self.y_train.iloc[val_idx].values, preds, self.returns_train[val_idx]
                        )
                scores.append(local_sharpe)
            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30) # noqa

        best_params = study.best_params.copy()
        best_params.update({
            'loss_function': 'MultiClass', 'auto_class_weights': 'Balanced',
            'task_type': "CPU", 'verbose': False, 'random_state': self.seed
        })

        # Safeguard final training if the entire dataset contains only one class
        if y_train_shifted.nunique() <= 1:
            unique_val = y_train_shifted.iloc[0]
            test_preds = np.full(len(self.X_test), unique_val - 1)
        else:
            best_model = CatBoostClassifier(**best_params)
            best_model.fit(self.X_train, y_train_shifted)
            test_preds = best_model.predict(self.X_test).flatten() - 1

        # print("Long:", np.mean(test_preds == 1))    # noqa
        # print("Short:", np.mean(test_preds == -1))  # noqa
        # print("Hold:", np.mean(test_preds == 0))    # noqa
        # print(confusion_matrix(self.y_test, test_preds))

        accuracy, sharpe, stability = self.evaluate_performance(interval, self.y_test.values, test_preds, self.returns_test)
        return {
            'model_type': 'CAT',
            'accuracy': accuracy,
            'wf_sharpe': study.best_value,
            'sharpe': sharpe,
            'stability': stability,
            'best_params': best_params
        }

    # Train and tune LSTM with walk-forward validation
    def _train_lstm(self, interval: str, horizon: int) -> dict:
        window = 30
        X_test_padded = np.vstack((self.X_train[-window:], self.X_test))
        y_test_padded = np.concatenate((self.y_train.values[-window:], self.y_test.values))

        x_train_3d, y_train_seq = self.create_3d_sequences(self.X_train, self.y_train.values, window=window)
        x_test_3d, y_test_seq = self.create_3d_sequences(X_test_padded, y_test_padded, window=window)

        y_train_shifted = (y_train_seq + 1).astype(np.int64)
        returns_train_seq = self.returns_train[window:]
        input_dim = x_train_3d.shape[2]

        # Calculate exact class weights to prevent the LSTM from ignoring minority classes
        classes = np.unique(y_train_shifted)
        weights = compute_class_weight('balanced', classes=classes, y=y_train_shifted)
        weight_tensor = torch.tensor(weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')

        # Use our custom Purged & Embargoed split for accurate financial evaluation
        tscv = PurgedTimeSeriesSplit(n_splits=2, gap=window+horizon, embargo_pct=0.01)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            hidden_dim = trial.suggest_categorical('module__hidden_dim', [32, 64, 128])
            num_layers = trial.suggest_int('module__num_layers', 1, 3)
            dropout = trial.suggest_float('module__dropout', 0.1, 0.5)
            weight_decay = trial.suggest_float('optimizer__weight_decay', 1e-6, 1e-2, log=True)
            max_epochs = trial.suggest_int('max_epochs', 20, 60)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            optimizer_name = trial.suggest_categorical('optimizer_name', ["Adam", "AdamW", "RMSprop"])

            optimizers = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW, "RMSprop": torch.optim.RMSprop} # noqa

            def build_model():
                return NeuralNetClassifier(
                    LSTMBrain,
                    module__input_dim=input_dim,
                    module__hidden_dim=hidden_dim,
                    module__num_layers=num_layers,
                    module__dropout=dropout,
                    module__output_dim=3,
                    criterion=nn.CrossEntropyLoss,
                    criterion__weight=weight_tensor,
                    optimizer=optimizers[optimizer_name],
                    optimizer__weight_decay=weight_decay,
                    lr=lr,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    train_split=None,
                    iterator_train__shuffle=False,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    verbose=0,
                    callbacks=[('pruning', OptunaPruningCallback(trial, 'train_loss'))]
                )

            scores = []
            for train_idx, val_idx in tscv.split(x_train_3d):
                model = build_model()
                model.fit(x_train_3d[train_idx], y_train_shifted[train_idx])
                preds = model.predict(x_train_3d[val_idx]) - 1

                # Penalize models that take zero trades to close the safe haven loophole
                if np.all(preds == 0):
                    local_sharpe = -1.0
                else:
                    # Optimize for Sharpe Ratio
                    _, local_sharpe, _ = self.evaluate_performance(
                        interval, y_train_seq[val_idx], preds, returns_train_seq[val_idx]
                    )
                scores.append(local_sharpe)
            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10) # noqa

        best_params = study.best_params.copy()
        opt_name = best_params.pop('optimizer_name')
        optimizers = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW, "RMSprop": torch.optim.RMSprop}

        final_model = NeuralNetClassifier(
            LSTMBrain,
            module__input_dim=input_dim,
            module__output_dim=3,
            criterion=nn.CrossEntropyLoss,
            criterion__weight=weight_tensor,
            optimizer=optimizers[opt_name],
            train_split=None,
            iterator_train__shuffle=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=Settings.VERBOSE,
            **best_params
        )

        final_model.fit(x_train_3d, y_train_shifted)
        test_preds = final_model.predict(x_test_3d) - 1

        # print("Long:", np.mean(test_preds == 1))    # noqa
        # print("Short:", np.mean(test_preds == -1))  # noqa
        # print("Hold:", np.mean(test_preds == 0))    # noqa
        # print(confusion_matrix(self.y_test, test_preds))

        returns_test_seq = np.concatenate((self.returns_train[-window:], self.returns_test))[window:]
        accuracy, sharpe, stability = self.evaluate_performance(interval, y_test_seq, test_preds, returns_test_seq)

        best_params['optimizer_name'] = opt_name
        return {
            'model_type': 'LSTM',
            'accuracy': accuracy,
            'wf_sharpe': study.best_value,
            'sharpe': sharpe,
            'stability': stability,
            'best_params': best_params
        }

    # Run all helper functions and consolidate the best model
    def run_training_pipeline(self, interval) -> bool:
        data = pd.read_parquet(os.path.join(DATA_DIR, f"SPY_{interval}.parquet"))
        if data is None: print("No data"); return False

        all_horizons = {
            "15m": {2: 0.5,  4: 1,  8: 2,    13: 3.25},  # bars: hours
            "1h":  {1: 1,    2: 2,  4: 4,    8: 25},     # bars: hours
            "1d":  {1: 1,    4: 4,  10: 14,  20: 28}     # bars: days
        }
        horizons = all_horizons[interval]

        def save():
            with open(f"results/tuned_results_{interval}.json", "w") as f:
                json.dump(results, f, indent=4)

        results = {}
        for horizon in horizons.keys():
            print(f"Adding features ({horizon})...")
            df = data.ind.add_indicators(interval, horizon)
            if len(df) < 300:
                print(f"Insufficient data (need 300+, got {len(df)})")
                return False

            print("Scaling features...")
            self._prepare_data(df)
            results[horizon] = {}

            print("Tuning LightGBM...")
            results[horizon]["LGBM"] = self._train_lightgbm(interval, horizon)
            flush_memory()
            save()

            print("Tuning CatBoost...")
            results[horizon]["CAT"]  = self._train_catboost(interval, horizon)
            flush_memory()
            save()

            print("Tuning LSTM...")
            results[horizon]["LSTM"] = self._train_lstm(interval, horizon)
            flush_memory()
            save()

        return True


if __name__ == "__main__":
    start = time.perf_counter()
    m = TrainingManager()
    for inter in ["1h", "1d"]:
        m.run_training_pipeline(inter)

    end = time.perf_counter()
    print(f"Time: {end-start}s")
