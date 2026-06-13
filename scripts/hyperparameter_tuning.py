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
from scripts.config import DATA_DIR, HYPER_DIR
import scripts.indicators_tuning # noqa

class Settings:
    VERBOSE = 0
    GPU = {"LGBM": False, "CAT": True, "LSTM": True}

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
    def __init__(self, n_splits: int, gap: int, embargo_pct: float = 0.01):
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

        self.n_tickers = 1

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.returns_train = None
        self.returns_test = None
        self.barrier_train = None
        self.lstm_bar_train = None

        self.lstm_x_train = None
        self.lstm_y_train = None
        self.lstm_ret_train = None
        self.lstm_x_test = None
        self.lstm_y_test = None
        self.lstm_ret_test = None
        self.barrier_test = None
        self.lstm_bar_test = None

        self.feature_cols = None
        self.scaler = None

    def _prepare_data(self, dfs_dict: dict):
        first_df = list(dfs_dict.values())[0]
        drop_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'MA_200', 'return', 'target_profit', 'tbm_return', 'barrier_strength']
        self.feature_cols = [c for c in first_df.columns if c not in drop_cols]

        train_dfs_dict = {}
        test_dfs_dict = {}
        train_raw_list = []

        for ticker, df in dfs_dict.items():
            split_idx = int(len(df) * (1 - self.__test_size))
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

            train_dfs_dict[ticker] = train_df
            test_dfs_dict[ticker] = test_df
            train_raw_list.append(train_df[self.feature_cols].values)

        self.scaler = StandardScaler()
        self.scaler.fit(np.vstack(train_raw_list))

        train_df_concat = pd.concat(train_dfs_dict.values()).sort_index()
        test_df_concat = pd.concat(test_dfs_dict.values()).sort_index()

        self.X_train = self.scaler.transform(train_df_concat[self.feature_cols].values)
        self.X_test = self.scaler.transform(test_df_concat[self.feature_cols].values)
        self.y_train = train_df_concat['target_profit']
        self.y_test = test_df_concat['target_profit']
        self.returns_train = train_df_concat['tbm_return'].values
        self.returns_test = test_df_concat['tbm_return'].values
        self.barrier_train = train_df_concat['barrier_strength'].values
        self.barrier_test = test_df_concat['barrier_strength'].values

        x_tr_list, y_tr_list, ret_tr_list, bar_tr_list, dt_tr_list = [], [], [], [], []
        x_te_list, y_te_list, ret_te_list, bar_te_list, dt_te_list = [], [], [], [], []

        for ticker in dfs_dict.keys():
            train_df = train_dfs_dict[ticker]
            test_df = test_dfs_dict[ticker]

            # Sequence training set
            x3_tr, y_tr, r_tr, b_tr, d_tr = self.create_3d_sequences(train_df, self.feature_cols, self.scaler, window=30)
            x_tr_list.append(x3_tr); y_tr_list.append(y_tr); ret_tr_list.append(r_tr); bar_tr_list.append(b_tr); dt_tr_list.append(d_tr)

            # Pad test set with last 30 days of train set so we don't lose the first month of test predictions
            padded_test_df = pd.concat([train_df.iloc[-30:], test_df])
            x3_te, y_te, r_te, b_te, d_te = self.create_3d_sequences(padded_test_df, self.feature_cols, self.scaler, window=30)
            x_te_list.append(x3_te); y_te_list.append(y_te); ret_te_list.append(r_te); bar_te_list.append(b_te); dt_te_list.append(d_te)

        # Combine and Time-Sort Train Sequences
        all_dts_tr = np.concatenate(dt_tr_list)
        sort_idx_tr = np.argsort(all_dts_tr)
        self.lstm_x_train = np.concatenate(x_tr_list)[sort_idx_tr]
        self.lstm_y_train = np.concatenate(y_tr_list)[sort_idx_tr]
        self.lstm_ret_train = np.concatenate(ret_tr_list)[sort_idx_tr]
        self.lstm_bar_train = np.concatenate(bar_tr_list)[sort_idx_tr]

        # Combine and Time-Sort Test Sequences
        all_dts_te = np.concatenate(dt_te_list)
        sort_idx_te = np.argsort(all_dts_te)
        self.lstm_x_test = np.concatenate(x_te_list)[sort_idx_te]
        self.lstm_y_test = np.concatenate(y_te_list)[sort_idx_te]
        self.lstm_ret_test = np.concatenate(ret_te_list)[sort_idx_te]
        self.lstm_bar_test = np.concatenate(bar_te_list)[sort_idx_te]

        self.n_tickers = len(dfs_dict)

    @staticmethod
    def evaluate_performance(interval: str, actual: np.ndarray, predicted: np.ndarray, actual_returns: np.ndarray, barrier_strength: np.ndarray) -> tuple:
        actual = np.asarray(actual)
        predicted = np.asarray(predicted).reshape(-1)
        actual_returns = np.asarray(actual_returns).reshape(-1)
        barrier_strength = np.asarray(barrier_strength).reshape(-1)

        acc = accuracy_score(actual, predicted)

        if interval == "1d": bars = 252
        elif interval == "1h": bars = 1638
        else: raise ValueError("Interval length not valid.")

        custom_scores = np.nan_to_num(actual_returns * predicted)

        opposite_mask = (actual != 0) & (predicted == -actual)
        custom_scores[opposite_mask] = 2.0 * custom_scores[opposite_mask]

        missed_breakout = (actual != 0) & (predicted == 0)
        custom_scores[missed_breakout] = -np.abs(actual_returns[missed_breakout]) * 0.5

        near_miss_penalty = np.clip(barrier_strength, 0, 1)
        wrong_hold_trade = (actual == 0) & (predicted != 0)
        bad_wrong_hold_trade = wrong_hold_trade & (custom_scores < 0)
        custom_scores[bad_wrong_hold_trade] *= (1.0 - near_miss_penalty[bad_wrong_hold_trade])

        hold_right_mask = (actual == 0) & (predicted == 0)

        if np.any(hold_right_mask):
            custom_scores[hold_right_mask] = np.abs(actual_returns[hold_right_mask]).mean() * 0.1

        # Annualize the performance score card
        util_score = (np.mean(custom_scores) / (np.std(custom_scores) + 1e-9)) * np.sqrt(bars)

        # Calculate stability
        cumulative_returns = np.cumsum(custom_scores)
        if len(cumulative_returns) > 1:
            x = np.arange(len(cumulative_returns))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, cumulative_returns)
            stability = r_value ** 2 if slope > 0 else 0.0
        else:
            stability = 0.0

        obj_score = 0.5 * util_score + 0.3 * acc + 0.2 * stability

        return obj_score, acc, util_score, stability

    @staticmethod
    def create_3d_sequences(df: pd.DataFrame, feature_cols: list, scaler: StandardScaler, window: int = 30) -> tuple:
        data = scaler.transform(df[feature_cols].values)
        targets = df['target_profit'].values
        returns = df['tbm_return'].values
        barriers = df['barrier_strength'].values
        dates = df.index.values

        x, y, rets, bars, dts = [], [], [], [], []
        for i in range(len(data) - window):
            x.append(data[i: i + window])
            y.append(targets[i + window])
            rets.append(returns[i + window])
            bars.append(barriers[i + window])
            dts.append(dates[i + window])

        return np.array(x, dtype=np.float32), np.array(y), np.array(rets), np.array(bars), np.array(dts)

    @staticmethod
    def create_weights(y_shifted: np.ndarray, cap: float = 3.0) -> np.ndarray:
        y_shifted = np.asarray(y_shifted, dtype=int)
        counts = np.bincount(y_shifted, minlength=3).astype(float)
        total = counts.sum()

        raw_weights = total / (3.0 * np.maximum(counts, 1.0))
        capped_weights = np.clip(raw_weights, 0.25, cap)

        return capped_weights / capped_weights.mean()

    def _train_lightgbm(self, interval: str, horizon: int) -> dict:
        y_train_shifted = self.y_train + 1
        tscv = PurgedTimeSeriesSplit(n_splits=3, gap=horizon * self.n_tickers, embargo_pct=0.01)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        weight_cap = 3.0 if interval == "1d" else 5.0
        class_weights = self.create_weights(y_train_shifted.values, cap=weight_cap)
        class_weight = {i: float(class_weights[i]) for i in range(3)}

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
                'class_weight': class_weight,
                'verbose': -1,
                'n_jobs': -1,
                'device_type': 'cuda' if Settings.GPU["LGBM"] else "cpu"
            }

            model = LGBMClassifier(**params)
            scores = []

            for train_idx, val_idx in tscv.split(self.X_train):
                model.fit(self.X_train[train_idx], y_train_shifted.iloc[train_idx])
                preds = model.predict(self.X_train[val_idx]) - 1

                if np.all(preds == 0):
                    obj_score = -1.0
                else:
                    obj_score, _, _, _ = self.evaluate_performance(
                        interval, self.y_train.iloc[val_idx].values, preds, self.returns_train[val_idx],
                        self.barrier_train[val_idx]
                    )
                scores.append(obj_score)
            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30) # noqa

        best_params = study.best_params.copy()
        best_params.update({
            'objective': 'multiclass', 'num_class': 3,
            'class_weight': class_weight, 'random_state': self.seed,
            'verbose': -1, 'device_type': 'gpu' if Settings.GPU["LGBM"] else "cpu"
        })

        best_model = LGBMClassifier(**best_params)
        best_model.fit(self.X_train, y_train_shifted)

        test_preds = best_model.predict(self.X_test) - 1
        obj_score, accuracy, util_score, stability = self.evaluate_performance(
            interval, self.y_test.values, test_preds, self.returns_test, self.barrier_test
        )

        return {
            'model_type': 'LGBM',
            'accuracy': accuracy,
            'wf_util_score': study.best_value,
            'util_score': util_score,
            'stability': stability,
            'obj_score': obj_score,
            'best_params': best_params
        }

    def _train_catboost(self, interval: str, horizon: int) -> dict:
        y_train_shifted = self.y_train + 1
        tscv = PurgedTimeSeriesSplit(n_splits=3, gap=horizon * self.n_tickers, embargo_pct=0.01)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        weight_cap = 3.0 if interval == "1d" else 5.0
        class_weights = [float(w) for w in self.create_weights(y_train_shifted.values, cap=weight_cap)]

        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 150, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 20.0, log=True),
                'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'loss_function': 'MultiClass',
                'class_weights': class_weights,
                'task_type': "GPU" if Settings.GPU["CAT"] else "CPU",
                'verbose': False,
                'random_state': self.seed
            }

            model = CatBoostClassifier(**params)
            scores = []

            for train_idx, val_idx in tscv.split(self.X_train):
                if y_train_shifted.iloc[train_idx].nunique() <= 1:
                    obj_score = -1.0
                else:
                    model.fit(self.X_train[train_idx], y_train_shifted.iloc[train_idx])
                    preds = model.predict(self.X_train[val_idx]).flatten() - 1

                    if np.all(preds == 0):
                        obj_score = -1.0
                    else:
                        obj_score, _, _, _ = self.evaluate_performance(
                            interval, self.y_train.iloc[val_idx].values, preds, self.returns_train[val_idx],
                            self.barrier_train[val_idx]
                        )
                scores.append(obj_score)
            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30) # noqa

        best_params = study.best_params.copy()
        best_params.update({
            'loss_function': 'MultiClass', 'auto_class_weights': 'Balanced',
            'task_type': "GPU" if Settings.GPU["CAT"] else "CPU", 'verbose': False, 'random_state': self.seed
        })

        if y_train_shifted.nunique() <= 1:
            unique_val = y_train_shifted.iloc[0]
            test_preds = np.full(len(self.X_test), unique_val - 1)
        else:
            best_model = CatBoostClassifier(**best_params)
            best_model.fit(self.X_train, y_train_shifted)
            test_preds = best_model.predict(self.X_test).flatten() - 1

        obj_score, accuracy, util_score, stability = self.evaluate_performance(
            interval, self.y_test.values, test_preds, self.returns_test, self.barrier_test
        )

        return {
            'model_type': 'CAT',
            'accuracy': accuracy,
            'wf_util_score': study.best_value,
            'util_score': util_score,
            'stability': stability,
            'obj_score': obj_score,
            'best_params': best_params
        }

    def _train_lstm(self, interval: str, horizon: int) -> dict:
        y_train_shifted = (self.lstm_y_train + 1).astype(np.int64)
        input_dim = self.lstm_x_train.shape[2]

        weight_cap = 3.0 if interval == "1d" else 5.0
        class_weights = self.create_weights(y_train_shifted, cap=weight_cap)

        device = 'cuda' if torch.cuda.is_available() and Settings.GPU["LSTM"] else 'cpu'
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        tscv = PurgedTimeSeriesSplit(n_splits=2, gap=(30+horizon)*self.n_tickers, embargo_pct=0.01)
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

            optimizers = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW, "RMSprop": torch.optim.RMSprop}

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
                    device=device,
                    verbose=0,
                    callbacks=[('pruning', OptunaPruningCallback(trial, 'train_loss'))]
                )

            scores = []
            for train_idx, val_idx in tscv.split(self.lstm_x_train):
                model = build_model()
                model.fit(self.lstm_x_train[train_idx], y_train_shifted[train_idx])
                preds = model.predict(self.lstm_x_train[val_idx]) - 1

                if np.all(preds == 0):
                    obj_score = -1.0
                else:
                    obj_score, _, _, _ = self.evaluate_performance(
                        interval, self.lstm_y_train[val_idx], preds, self.lstm_ret_train[val_idx],
                        self.lstm_bar_train[val_idx]
                    )
                scores.append(obj_score)
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
            device=device,
            verbose=Settings.VERBOSE,
            **best_params
        )

        final_model.fit(self.lstm_x_train, y_train_shifted)
        test_preds = final_model.predict(self.lstm_x_test) - 1

        obj_score, accuracy, util_score, stability = self.evaluate_performance(
            interval, self.lstm_y_test, test_preds, self.lstm_ret_test, self.lstm_bar_test
        )

        best_params['optimizer_name'] = opt_name
        return {
            'model_type': 'LSTM',
            'accuracy': accuracy,
            'wf_util_score': study.best_value,
            'util_score': util_score,
            'stability': stability,
            'obj_score': obj_score,
            'best_params': best_params
        }

    # Run all helper functions and consolidate the best model
    def run_training_pipeline(self, profile_name: str, interval: str) -> bool:
        profile_path = os.path.join(HYPER_DIR, profile_name)
        if not os.path.exists(profile_path):
            print(f"Profile directory not found: {profile_path}")
            return False

        # Load all tickers within that profile folder
        tickers = [f.split('_')[0] for f in os.listdir(profile_path) if f.endswith(f"_{interval}.parquet")]
        if not tickers:
            print(f"No valid .parquet files found for interval {interval} in {profile_path}")
            return False

        def save():
            with open(f"results/tuned_{profile_name}_{interval}.json", "w") as f:
                json.dump(results, f, indent=4)

        results = {}
        for horizon in {"1h": [2, 4, 8], "1d": [5, 10, 20]}[interval]:
            dfs_dict = {}
            for ticker in tickers:
                data = pd.read_parquet(os.path.join(profile_path, f"{ticker}_{interval}.parquet"))

                df = data.ind.add_indicators(ticker, interval, horizon)
                if len(df) >= 300:
                    dfs_dict[ticker] = df
                else:
                    print(f"Insufficient data for {ticker} (need 300+, got {len(df)})")

                # dist = df['target_profit'].value_counts(normalize=True).sort_index()
                # print(f"{ticker} {interval} h={horizon} labels: {dist.to_dict()}")

            # pooled_labels = pd.concat([df['target_profit'] for df in dfs_dict.values()])
            # pooled_dist = pooled_labels.value_counts(normalize=True).sort_index()
            # pooled_counts = pooled_labels.value_counts().sort_index()
            # print(f"{profile_name} {interval} h={horizon} pooled labels:")
            # print("Distribution:", pooled_dist.to_dict())
            # print("Counts:", pooled_counts.to_dict())

            if not dfs_dict:
                print(f"Insufficient pooled data across all tickers for horizon {horizon}.")
                continue

            print(f"Scaling and pooling features ({horizon})...")
            self._prepare_data(dfs_dict)
            str_hor = str(horizon)
            results[str_hor] = {}

            print("Tuning LightGBM...")
            results[str_hor]["LGBM"] = self._train_lightgbm(interval, horizon)
            flush_memory()
            save()

            print("Tuning CatBoost...")
            results[str_hor]["CAT"]  = self._train_catboost(interval, horizon)
            flush_memory()
            save()

            print("Tuning LSTM...")
            results[str_hor]["LSTM"] = self._train_lstm(interval, horizon)
            flush_memory()
            save()

        return True


if __name__ == "__main__":
    start = time.perf_counter()
    os.makedirs("results", exist_ok=True)
    m = TrainingManager()
    for prof in ["Profile A", "Profile B", "Profile C", "Profile D", "Profile E", "Profile F"]:
        for inter in ["1h", "1d"]:
            m.run_training_pipeline(prof, inter)

    end = time.perf_counter()
    print(f"Time: {end-start}s")
