
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
        actual = np.asarray(actual)
        predicted = np.asarray(predicted).reshape(-1)
        actual_returns = np.asarray(actual_returns).reshape(-1)

        if not (len(actual) == len(predicted) == len(actual_returns)):
            raise ValueError(
                f"Evaluation arrays must align: actual={len(actual)}, "
                f"predicted={len(predicted)}, returns={len(actual_returns)}"
            )

        acc = accuracy_score(actual, predicted)

        # Strategy return aligns with the direction predicted:
        # Long (1) gets the path return, Short (-1) gets inverted path return, Hold (0) gets 0
        strategy_returns = np.nan_to_num(actual_returns * predicted)

        if len(strategy_returns) < 2 or np.std(strategy_returns) == 0:
            sharpe = 0.0
        else:
            sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * np.sqrt(252)

        # Calculate stability
        cumulative_returns = np.cumsum(strategy_returns)
        if len(cumulative_returns) > 1:
            from scipy import stats
            x = np.arange(len(cumulative_returns))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, cumulative_returns)
            stability = r_value ** 2 if slope > 0 else 0.0
        else:
            stability = 0.0

        return acc, sharpe, stability

    def _score_fold(self, model, X_val: np.ndarray, y_val: np.ndarray, returns_val: np.ndarray) -> float:
        preds = np.asarray(model.predict(X_val)).reshape(-1) - 1
        accuracy, _, _ = self.evaluate_performance(y_val, preds, returns_val)
        return accuracy

    @staticmethod
    def create_3d_sequences(data: np.ndarray, targets: np.ndarray, window: int = 30) -> tuple:
        x, y = [], []
        for i in range(len(data) - window):
            x.append(data[i: i + window])
            y.append(targets[i + window])
        return np.array(x, dtype=np.float32), np.array(y)

    def _train_lgbm(self, horizon: int, hyperparams: dict, X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray,  y_test: pd.Series, actual_returns_train: np.ndarray, actual_returns_test: np.ndarray) -> dict:
        lgbm_params = hyperparams.copy()
        lgbm_params.update(dict(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            objective='multiclass',
            num_class=3,
            class_weight='balanced',
            random_state=self.seed,
            importance_type='gain',
            verbose=-1
        ))
        if Settings.GPU:
            lgbm_params.update({"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
        if Settings.Threaded:
            lgbm_params.update({"num_threads": 1, "n_jobs": 1})

        # LightGBM multiclass requires labels 0-indexed, so shift {-1,0,1} -> {0,1,2}
        y_train_shifted = y_train + 1

        model = LGBMClassifier(**lgbm_params)

        # Walk-forward validation on training data only (no peeking at test set)
        time_splitter = TimeSeriesSplit(n_splits=3, gap=horizon)
        wf_scores = []
        for train_idx, val_idx in time_splitter.split(X_train):
            model.fit(X_train[train_idx], y_train_shifted.iloc[train_idx])
            wf_scores.append(self._score_fold(model, X_train[val_idx], y_train.iloc[val_idx].values, actual_returns_train[val_idx]))

        # Retrain on full training set, then evaluate on the held-out test set
        model.fit(X_train, y_train_shifted)
        test_preds = model.predict(X_test) - 1  # Shift back to {-1,0,1} for evaluation

        accuracy, sharpe, stability = self.evaluate_performance(y_test.values, test_preds, actual_returns_test)
        wf_mean = float(np.mean(wf_scores))

        return {
            'type': 'LGBM', 'model': model,
            'accuracy': accuracy, 'walk_forward_accuracy': wf_mean,
            'sharpe': sharpe, 'stability': stability,
        }

    def _train_catboost(self, horizon: int, hyperparams: dict, X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray,  y_test: pd.Series, actual_returns_train: np.ndarray, actual_returns_test: np.ndarray) -> dict:
        cat_params = hyperparams.copy()
        model = CatBoostClassifier(
            iterations=500, # Will change with hyperparameter tuning
            learning_rate=0.05,
            depth=6,
            loss_function='MultiClass',
            auto_class_weights='Balanced',
            task_type="GPU" if Settings.GPU else "CPU",
            verbose=False,
            random_state=self.seed,
            **cat_params
        )

        y_train_shifted = y_train + 1

        time_splitter = TimeSeriesSplit(n_splits=3, gap=horizon)
        wf_scores = []
        for train_idx, val_idx in time_splitter.split(X_train):
            model.fit(X_train[train_idx], y_train_shifted.iloc[train_idx])
            wf_scores.append(self._score_fold(model, X_train[val_idx], y_train.iloc[val_idx].values, actual_returns_train[val_idx]))

        model.fit(X_train, y_train_shifted)
        test_preds = model.predict(X_test).flatten() - 1  # Shift back to {-1,0,1}

        accuracy, sharpe, stability = self.evaluate_performance(y_test.values, test_preds, actual_returns_test)
        wf_mean = float(np.mean(wf_scores))

        return {
            'type': 'CAT', 'model': model,
            'accuracy': accuracy, 'walk_forward_accuracy': wf_mean,
            'sharpe': sharpe, 'stability': stability,
        }

    def _train_lstm(self, horizon: int, hyperparams: dict, X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray,  y_test: pd.Series, actual_returns_train: np.ndarray, actual_returns_test: np.ndarray) -> dict:
        lstm_params = hyperparams.copy()

        window = 30
        X_test_padded = np.vstack((X_train[-window:], X_test))
        y_test_padded = np.concatenate((y_train.values[-window:], y_test.values))

        # Build (samples, window, features) sequences from the flat scaled arrays
        x_train_3d, y_train_seq = self.create_3d_sequences(X_train, y_train.values)
        x_test_3d,  y_test_seq  = self.create_3d_sequences(X_test_padded, y_test_padded, window=window)

        # Shift targets to {0,1,2} for CrossEntropyLoss; keep originals for evaluate_performance
        y_train_shifted = (y_train_seq + 1).astype(np.int64)
        returns_train_seq = actual_returns_train[window:]

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
        time_splitter = TimeSeriesSplit(n_splits=3, gap=horizon)
        wf_scores = []
        for train_idx, val_idx in time_splitter.split(x_train_3d):
            fold_model = build_skorch_model()
            fold_model.fit(x_train_3d[train_idx], y_train_shifted[train_idx])
            wf_scores.append(self._score_fold(fold_model, x_train_3d[val_idx], y_train_seq[val_idx], returns_train_seq[val_idx]))

        # Final model trained on all training sequences
        model = build_skorch_model()
        model.fit(x_train_3d, y_train_shifted)

        # Shift predictions back to {-1,0,1} for evaluation
        test_preds = model.predict(x_test_3d) - 1

        # Align returns with the sequence offset (first LSTM_WINDOW bars are consumed as context)
        accuracy, sharpe, stability = self.evaluate_performance(y_test_seq, test_preds, actual_returns_test)
        wf_mean = float(np.mean(wf_scores)) if wf_scores else 0.0

        return {
            'type': 'LSTM', 'model': model,
            'accuracy': accuracy, 'walk_forward_accuracy': wf_mean,
            'sharpe': sharpe, 'stability': stability,
        }

    def _save_model_assets(self, ticker: str, interval: str, horizon: int, training_data_end: pd.Timestamp, results: list, feature_columns: list, scaler: StandardScaler) -> None:

        save_folder = os.path.join(MODEL_DIR, f"{ticker}_{interval}")
        if not os.path.exists(save_folder): os.makedirs(save_folder)

        metadata = {
            "training_date":      datetime.now().strftime("%Y-%m-%d"),
            "training_data_end":  training_data_end.strftime("%Y-%m-%d"),
            "interval":           interval,
            "horizon":            horizon,
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

            if model_type == 'LSTM':
                save_file(model.module_.state_dict(), os.path.join(save_folder, "lstm_model.safetensors"))
            elif model_type == 'LGBM':
                model.booster_.save_model(os.path.join(save_folder, "lgbm_model.txt"))
            else:
                joblib.dump(model, os.path.join(save_folder, f"cat_model.joblib"))

        with open(os.path.join(save_folder, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

        joblib.dump(scaler, os.path.join(save_folder, "scaler.joblib"))
        joblib.dump(feature_columns, os.path.join(save_folder, "features.joblib"))

    def run_training_pipeline(self, ticker: str, interval: str, horizon: int = 4, override_data: pd.DataFrame = None, status_signal: tuple = None, force_train: bool = False) -> bool:
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
        raw_data = override_data if override_data is not None else load_data(ticker, interval)
        if raw_data is None or raw_data.empty:
            log_update(f"No raw data for {ticker} ({interval})", True)
            return False

        log_update("Adding features...")
        df = raw_data.ind.add_indicators(ticker, interval, horizon)
        if len(df) < 300:
            log_update(f"Insufficient processed data for {ticker} ({interval}) — need 300+, got {len(df)}", True)
            return False

        log_update("Scaling features...")
        drop_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'MA_200', 'return', 'target_profit', 'tbm_return']
        feature_cols = [c for c in df.columns if c not in drop_cols]

        X, y = df[feature_cols].values, df['target_profit']

        split_idx = int(len(df) * (1 - self.__test_size))
        X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        actual_returns_train = df['tbm_return'].loc[y_train.index].values
        actual_returns_test = df['tbm_return'].loc[y_test.index].values
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test  = scaler.transform(X_test_raw)

        hyperparams = {}

        log_update("Training LGBM...")
        lgbm_result = self._train_lgbm(horizon, hyperparams, X_train, y_train, X_test, y_test, actual_returns_train, actual_returns_test)

        log_update("Training CatBoost...")
        cat_result = self._train_catboost(horizon, hyperparams, X_train, y_train, X_test, y_test, actual_returns_train, actual_returns_test)

        log_update("Training LSTM...")
        lstm_result = self._train_lstm(horizon, hyperparams, X_train, y_train, X_test, y_test, actual_returns_train, actual_returns_test)

        results = [lgbm_result, cat_result, lstm_result]

        log_update("Saving assets...")
        self._save_model_assets(ticker, interval, horizon, df.index.max(), results, feature_cols, scaler)
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

def save_prediction(ticker: str, interval: str, forecast_results: dict) -> None:
    if len(forecast_results) < 1: return

    ledger_file = os.path.join(LEDGER_DIR, f"{ticker}_ledger.csv")
    new_entries = []
    for _, data in forecast_results.items():
        existing_match = find_prediction_match(ticker, interval, data["Date_Predicted"])
        if existing_match is not None and not existing_match.empty: continue

        new_entries.append({
            "Interval": interval,
            'Open_Date': data["Date_Predicted"],
            "Current_Price": round(data['Current_Price'], 2),
            'AVG_signal': f"{data['AVG_signal']:.3f}",
            'CAT_signal': f"{data['CAT_signal']:.3f}",
            'LGBM_signal': f"{data['LGBM_signal']:.3f}",
            'LSTM_signal': f"{data['LSTM_signal']:.3f}",
        })

    # Add prediction data to the ledger
    df_new = pd.DataFrame(new_entries)
    if not os.path.exists(ledger_file): df_new.to_csv(ledger_file, index=False)
    else: df_new.to_csv(ledger_file, mode='a', header=False, index=False)

def load_prediction(ticker: str, interval: str, date: datetime) -> dict | None:
    # Filter for the specific data line matching the exact runtime parameters
    match = find_prediction_match(ticker, interval, date)
    if match is None or match.empty: return None
    match_dicts = match.reset_index().to_dict(orient='records')

    # Rebuild the forecast_results dict
    try:
        forecast_results = {}
        row = match_dicts[0]
        forecast_results["Temp"] = {
            "Date_Predicted": pd.to_datetime(row['Open_Date'], format="ISO8601"),
            "Current_Price": float(row['Current_Price']),
            'AVG_signal': float(row['AVG_signal']),
            'CAT_signal': float(row['CAT_signal']),
            'LGBM_signal': float(row['LGBM_signal']),
            'LSTM_signal': float(row['LSTM_signal']),
        }
        return forecast_results

    except Exception: return None # noqa

def find_prediction_match(ticker: str, interval: str, date: datetime) -> pd.DataFrame | None:
    ledger_file = os.path.join(LEDGER_DIR, f"{ticker}_ledger.csv")
    try:
        ledger = pd.read_csv(ledger_file)
        if ledger.empty or len(ledger) < 1: return None
    except FileNotFoundError: return None

    ledger = pd.read_csv(ledger_file)
    ledger['Open_Date'] = pd.to_datetime(ledger['Open_Date'], format='ISO8601')

    # Check if any entry matches current ticker and last trade date
    date = date.strftime("%Y-%m-%d %H:%M")
    match = ledger[(ledger['Interval'] == interval) & (ledger['Open_Date'] == date)]
    return match

########################################################################################################################

def run_prediction_pipeline(ticker: str, interval: str, horizon: int = 4) -> dict:
    processed_df, assets = prepare_prediction_data(ticker, interval, horizon)
    if any(v is None for v in [processed_df, assets]):
        print("No data or assets.")
        return {}

    last_trade_date = processed_df.index[-1]

    # Load or generate from new assets ledger
    forecast_results = load_prediction(ticker, interval, last_trade_date)
    if forecast_results is None:
        is_hour = "h" in interval
        tech_info = (
            {horizon: horizon},
            "h" if is_hour else "d",
            last_trade_date,
            float(processed_df['Adj Close'].iloc[-1]),
        )

        forecast_results = generate_forecasts(processed_df, assets, tech_info)
        save_prediction(ticker, interval, forecast_results)

    return forecast_results

def prepare_prediction_data(ticker: str, interval: str, horizon: int = 4) -> tuple:
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

    # Trains a model if needed using the base implementation checks
    if not all_ticker_models_exist(model_path):
        success = manager.run_training_pipeline(ticker, interval, horizon=horizon)
        if not success: return None, None

    # Load assets
    processed_df = df.ind.add_indicators(ticker, interval, horizon)
    if processed_df.empty: return None, None

    scaler = joblib.load(f"{model_path}/scaler.joblib")
    features = joblib.load(f"{model_path}/features.joblib")

    return processed_df, (scaler, features, model_path)

def get_market_dates(latest_date, horizons: dict, period: str) -> dict:
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

def generate_forecasts(processed_df: pd.DataFrame, assets: tuple, tech_info: tuple) -> dict:
    scaler, features, model_folder = assets
    horizons, period, last_trade_date, current_price = tech_info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join(model_folder, 'metadata.json'), 'r') as f:
        meta = json.load(f)

    forecast_results = {}

    target_dates = get_market_dates(last_trade_date, horizons, period)
    if len(target_dates) < 1: return {}

    # Calculate forecasts
    for step, actual_time in horizons.items():
        if target_dates[step] is None: continue

        probs = {"LSTM": 0.5, "LGBM": 0.5, "CAT": 0.5}
        weights = {"LSTM": 0, "LGBM": 0, "CAT": 0}

        # LSTM prediction
        lstm_path = os.path.join(model_folder, "lstm_model.safetensors")
        recent_data = processed_df[features].tail(30)
        scaled_seq = scaler.transform(recent_data)
        x_3d = np.expand_dims(scaled_seq, axis=0).astype(np.float32)

        brain = LSTMBrain(
            input_dim=len(features),
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            output_dim=3
        )
        state_dict = load_file(lstm_path)
        brain.load_state_dict(state_dict)
        brain.to(device)
        brain.eval()

        with torch.no_grad():
            logits = brain(torch.from_numpy(x_3d).to(device))
            p_lstm = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            probs["LSTM"] = float((p_lstm[2] - p_lstm[0] + 1) / 2)

        # LGBM prediction
        lgbm_path = os.path.join(model_folder, "lgbm_model.txt")
        scaled_row = scaler.transform(processed_df[features].iloc[-1:])
        booster = LGBMBooster(model_file=lgbm_path)
        p_lgbm = booster.predict(scaled_row)[0]
        probs["LGBM"] = float((p_lgbm[2] - p_lgbm[0] + 1) / 2)

        # CAT prediction
        cat_path = os.path.join(model_folder, "cat_model.joblib")
        scaled_row = scaler.transform(processed_df[features].iloc[-1:])
        cat_model = joblib.load(cat_path)
        p_cat = cat_model.predict_proba(scaled_row)[0]
        probs["CAT"] = float((p_cat[2] - p_cat[0] + 1) / 2)

        # Calculate tracking layer weight priorities based on new multiclass metadata
        for m_type in ["LSTM", "LGBM", "CAT"]:
            model_meta = meta.get("models", {}).get(m_type, {})
            ticker_sharpe = abs(model_meta.get("sharpe", 0.0))
            ticker_accuracy = model_meta.get("accuracy", 0.0)
            results_weight = {"LGBM": 0.33, "Cat": 0.33, "LSTM": 0.34}.get(m_type, 0.33)

            weights[m_type] = (results_weight * 0.5) + (ticker_sharpe * 0.3) + (ticker_accuracy * 0.2)

        total_weight = sum(weights.values())
        avg_proba = sum(probs[m] * weights[m] for m in probs) / total_weight if total_weight > 0 else 0.5

        # Calculate predicted target bounds
        forecast_results[step] = {
            "Date_Predicted": last_trade_date.strftime("%Y-%m-%d %H:%M"),
            "Current_Price": current_price,
            'AVG_signal': avg_proba,
            'CAT_signal': probs["CAT"],
            'LGBM_signal': probs["LGBM"],
            'LSTM_signal': probs["LSTM"],
        }

    return forecast_results
