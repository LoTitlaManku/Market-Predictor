
# Standard library imports
import json
import os
import warnings

# External library imports
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import talib
from pykalman import KalmanFilter
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, make_scorer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit

# Set environment variables and filters
warnings.filterwarnings("ignore")
NYSE_CAL = mcal.get_calendar('NYSE')

# Custom imports
from scripts.config import DATA_DIR

class Settings:
    VERBOSE = 1 # Set whether to display logging or not

############################################################################

class LSTMBrain(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, layers: int = 2, dropout: float = 0.3):
        super().__init__()
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
        # We take the hidden state of the last layer
        _, (hn, _) = self.lstm(x)
        # Apply dropout to the final 'thought' before the linear layer
        out = self.dropout(hn[-1])
        return self.sigmoid(self.fc(out)).squeeze(-1)

class LSTM:
    def __init__(self, seed: int):
        self.seed = seed

    def train(self, features_train: pd.DataFrame, target_train: pd.Series, features_test: pd.DataFrame,
              target_test: pd.Series, price_returns: np.ndarray) -> dict:

        data_normalizer = StandardScaler()
        features_train_normalized = data_normalizer.fit_transform(features_train)
        features_test_normalized = data_normalizer.transform(features_test)

        x_train_3d, y_train_3d = self.create_3d_sequences(features_train_normalized, target_train.values)
        x_test_3d, y_test_3d = self.create_3d_sequences(features_test_normalized, target_test.values)

        tscv = TimeSeriesSplit(n_splits=2)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        mcc_scorer = make_scorer(matthews_corrcoef)

        def objective(trial):
            # Lowering max LR slightly to prevent divergence
            lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
            hidden_dim = trial.suggest_categorical('module__hidden_dim', [32, 64])
            # Adding weight decay to fight the 'tanking' loss
            weight_decay = trial.suggest_float('optimizer__weight_decay', 1e-5, 1e-2, log=True)
            dropout = trial.suggest_float('module__dropout', 0.2, 0.5)

            max_epochs = trial.suggest_int('max_epochs', 20, 50)
            batch_size = trial.suggest_categorical('batch_size', [32, 64])
            layers = trial.suggest_int('module__layers', 1, 2)
            optimizer_name = trial.suggest_categorical('optimizer_name', ["Adam", "AdamW"])

            optimizers = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}

            params = {
                'lr': lr,
                'optimizer__weight_decay': weight_decay,
                'module__hidden_dim': hidden_dim,
                'module__dropout': dropout,
                'max_epochs': max_epochs,
                'batch_size': batch_size,
                'module__layers': layers,
                'optimizer': optimizers[optimizer_name],
                'verbose': Settings.VERBOSE,
            }

            model = self.get_lstm_competitor(input_dim=x_train_3d.shape[2], tuned_params=params)

            # Use MCC to score the trials instead of accuracy
            scores = cross_val_score(model, x_train_3d, y_train_3d, cv=tscv, scoring=mcc_scorer, n_jobs=1)
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)  # Increased trials slightly to find the decay balance

        best_params = study.best_params.copy()
        optimizer_name = best_params.pop('optimizer_name')
        optimizers = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}

        final_params = best_params.copy()
        final_params['optimizer'] = optimizers[optimizer_name]
        final_params['max_epochs'] = 200
        final_params['verbose'] = 1
        final_params['train_split'] = ValidSplit(cv=0.2, stratified=False)

        best_model = self.get_lstm_competitor(input_dim=x_train_3d.shape[2], tuned_params=final_params, use_valid=True)
        best_model.fit(x_train_3d, y_train_3d)

        test_predictions = best_model.predict(x_test_3d)

        # Ensure we align target test for evaluation after the 14-day window slice
        aligned_target_test = target_test.iloc[14:]
        accuracy, sharpe, abs_sharpe, needs_flip = TrainingManager.evaluate_performance(
            aligned_target_test, test_predictions, price_returns[14:]
        )
        mcc_score = matthews_corrcoef(aligned_target_test, test_predictions)

        best_params['optimizer'] = optimizer_name

        return {
            'model_type': 'LSTM',
            'mcc': mcc_score,
            'accuracy': accuracy,
            'walk_forward_mcc': study.best_value,
            'sharpe_ratio': sharpe,
            'best_params': best_params
        }

    @staticmethod
    def get_lstm_competitor(input_dim: int, tuned_params: dict = None, use_valid: bool = False):
        params = {
            'module__input_dim': input_dim,
            'max_epochs': 100,
            'lr': 0.001,
            'batch_size': 64,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'criterion': nn.BCELoss,
            'optimizer': torch.optim.Adam,
            'verbose': 0,
            'train_split': None
        }

        if tuned_params:
            params.update(tuned_params)

        # Monitor validation loss if a split is provided
        monitor_stat = 'valid_loss' if (use_valid and params.get('train_split') is not None) else 'train_loss'

        return NeuralNetClassifier(
            LSTMBrain,
            callbacks=[
                ('early_stopping', EarlyStopping(monitor=monitor_stat, patience=15, lower_is_better=True))
            ],
            **params
        )

    @staticmethod
    def create_3d_sequences(data, targets, window_size=14):
        x, y = [], []
        for i in range(len(data) - window_size):
            x.append(data[i: i + window_size])
            y.append(targets[i + window_size])
        return np.array(x).astype(np.float32), np.array(y).astype(np.float32)

# Class to control and train models
class TrainingManager:
    def __init__(self):
        self.seed = 69
        self.sharpe_threshold = 0.50 # Min sharpe value for model to be useful
        self.__test_size = 0.2 # How much of data used to test vs train

    @staticmethod
    def _integrate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
        target_file = os.path.join(DATA_DIR, f"master_sentiment.parquet")

        global_sent = pd.read_parquet(target_file)
        global_sent['event_date'] = pd.to_datetime(global_sent['event_date'])

        sent_df = global_sent.groupby('event_date').agg({
            'avg_tone': 'mean',  # Average mood across all companies
            'article_count': 'sum'  # Total news volume for the whole market
        }).reset_index()
        sent_df = sent_df.sort_values('event_date')

        # Sentiment Impact: tone * log(count + 1)
        sent_df['sentiment_impact'] = sent_df['avg_tone'] * np.log1p(sent_df['article_count'])

        # Sentiment simple moving averages
        sent_df['sentiment_sma_7d'] = sent_df['avg_tone'].rolling(7, min_periods=1).mean()
        sent_df['sentiment_sma_30d'] = sent_df['avg_tone'].rolling(30, min_periods=1).mean()

        # Sentiment Volatility (Rolling Standard Deviation)
        sent_df['sentiment_volatility_7d'] = sent_df['avg_tone'].rolling(7, min_periods=1).std().fillna(0)

        # Sentiment Momentum (The gap between short and long term vibes)
        # Positive = Sentiment is improving; Negative = Sentiment is cooling off
        sent_df['sentiment_momentum'] = sent_df['sentiment_sma_7d'] - sent_df['sentiment_sma_30d']

        # Sentiment volume Z-Score (The "Shock" factor)
        # This identifies days when news volume is significantly higher than usual for THAT specific ticker
        sent_df['sentiment_volume_zscore'] = (
                sent_df['article_count'] / (sent_df['article_count'].rolling(30).mean() + 1e-9)
        ).fillna(0)

        # Other sentiment features
        sent_df['sentiment_shock'] = sent_df['avg_tone'].diff().fillna(0)
        sent_df['negative_pressure'] = (
            np.abs(np.minimum(sent_df['avg_tone'], 0)) * np.log1p(sent_df['article_count'])
        )
        sent_df['days_since_news'] = (sent_df['article_count'] > 0).astype(int).groupby((sent_df['article_count'] > 0).cumsum()).cumcount()

        sent_df['merge_date'] = sent_df['event_date'] + pd.Timedelta(days=1)
        sent_df = sent_df[sent_df.columns.difference(['ticker', 'event_date'])]

        index_name = df.index.name if df.index.name else 'index'
        df['merge_date'] = pd.to_datetime(df.index).normalize()
        df = df.reset_index()

        df = pd.merge(df, sent_df, on='merge_date', how='left')
        df = df.set_index(index_name)
        df = df.drop(columns=['merge_date']).fillna(0)

        return df

    @staticmethod
    def calculate_hurst(series, window=100):
        if len(series) < window: return 0.5
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) + 1e-9 for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    # Calculate technical indicators
    def calculate_technical_indicators(self, df: pd.DataFrame, interval: str, training: bool = True) -> pd.DataFrame:
        df = self._integrate_sentiment(df)

        # Horizon (h) targets for selected period (p)
        p = "h" if "h" in interval else "d"
        for h in ([1,2,4,8] if "h" in interval else [1,2,5,21]):
            df[f'target_cls_{h}{p}'] = df['Adj Close'].shift(-h).gt(df['Adj Close']).astype(int)

        df['return'] = df['Adj Close'].pct_change()
        for i in range(1, 4): df[f'return_lag_{i}'] = df['return'].shift(i)

        # Technical indicators
        df['RSI'] = talib.RSI(df['Adj Close'], timeperiod=14)
        macd, _, macdhist = talib.MACD(df['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD_Hist'] = macdhist
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Adj Close'], timeperiod=14)
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Adj Close'], timeperiod=14)
        df['MA_200'] = talib.SMA(df['Adj Close'], timeperiod=200)
        df['PDMA_200'] = (df['Adj Close'] / df['MA_200']) - 1

        df['OBV'] = talib.OBV(df['Adj Close'], df['Volume'])
        upper, mid, lower = talib.BBANDS(df['Adj Close'], timeperiod=20)
        df['BBP'] = (df['Adj Close'] - lower) / (upper - lower)
        df['ROC'] = talib.ROC(df['Close'], timeperiod=10)

        # Deviations using all of OHLC
        df['range_pct'] = (df['High'] - df['Low']) / df['Close']
        df['body_pct'] = (df['Close'] - df['Open']) / df['Close']
        df['upper_shadow_pct'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
        df['lower_shadow_pct'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']

        # Hurst Exponent
        df['Hurst_Exponent'] = df['Close'].rolling(window=100, min_periods=100).apply(self.calculate_hurst, raw=True)
        df['Hurst_Exponent'] = df['Hurst_Exponent'].fillna(0.5)

        # Kalman Filter
        def get_kalman_filter(series):
            kf = KalmanFilter(transition_matrices=[1],
                              observation_matrices=[1],
                              initial_state_mean=series.iloc[0],
                              initial_state_covariance=1,
                              observation_covariance=1,
                              transition_covariance=0.01)
            state_means, _ = kf.filter(series.values)
            return state_means.flatten()

        df['Kalman_Price'] = get_kalman_filter(df['Close'])
        df['Kalman_Dev'] = (df['Close'] - df['Kalman_Price']) / df['Kalman_Price']  # Deviation from "True" price

        # Efficiency ratio
        price_diff = df['Close'].diff(20).abs()
        volatility = df['Close'].diff().abs().rolling(20).sum()
        df['Efficiency_Ratio'] = price_diff / volatility  # 1.0 = Strong Trend, 0.0 = Choppy/Noisy

        # Other indicators
        df['vol_ratio'] = df["return"].rolling(5).std() / df["return"].rolling(50).std()
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        if training:
            df = df.dropna()

        return df

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
    def _train_lightgbm(self, features_train: pd.DataFrame, target_train: pd.Series, features_test: pd.DataFrame,
                        target_test: pd.Series, price_returns: np.ndarray) -> dict:
        # Put all indicators on the same scale (0 to 1 range) so it matches inference
        data_normalizer = StandardScaler()
        features_train_normalized = pd.DataFrame(data_normalizer.fit_transform(features_train),
                                                 columns=features_train.columns)
        features_test_normalized = data_normalizer.transform(features_test)

        # Handle class imbalance (Make 'Up' and 'Down' days equally important)
        scale_pos_weight = (len(target_train) - target_train.sum()) / target_train.sum()

        tscv = TimeSeriesSplit(n_splits=3)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        mcc_scorer = make_scorer(matthews_corrcoef)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_depth': trial.suggest_categorical('max_depth', [3, 7, -1]),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'random_state': self.seed,
                'scale_pos_weight': scale_pos_weight,
                'verbose': -1,
                'device': "gpu",
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
            }

            # For 0.0 values in regularization, we check explicitly
            if trial.suggest_categorical('use_reg_alpha_zero', [True, False]):
                params['reg_alpha'] = 0.0
            if trial.suggest_categorical('use_reg_lambda_zero', [True, False]):
                params['reg_lambda'] = 0.0

            model = LGBMClassifier(**params)

            # Use MCC to score the trials instead of accuracy
            scores = cross_val_score(model, features_train_normalized, target_train, cv=tscv, scoring=mcc_scorer, n_jobs=1)
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)  # 30 trials is fast for LGBM

        best_params = study.best_params.copy()

        # Clean up utility boolean parameters we used for 0 values
        use_alpha = best_params.pop('use_reg_alpha_zero', False)
        use_lambda = best_params.pop('use_reg_lambda_zero', False)
        if use_alpha: best_params['reg_alpha'] = 0.0
        if use_lambda: best_params['reg_lambda'] = 0.0

        best_model = LGBMClassifier(
            random_state=self.seed,
            scale_pos_weight=scale_pos_weight,
            verbose=-1,
            device="gpu",
            gpu_platform_id=0,
            gpu_device_id=0,
            **best_params
        )

        best_model.fit(features_train_normalized, target_train)
        test_predictions = best_model.predict(features_test_normalized)

        accuracy, sharpe, abs_sharpe, needs_flip = self.evaluate_performance(target_test, test_predictions,
                                                                             price_returns)
        mcc_score = matthews_corrcoef(target_test, test_predictions)

        return {
            'model_type': 'LGBM',
            'mcc': mcc_score,
            'accuracy': accuracy,
            'walk_forward_mcc': study.best_value,
            'sharpe_ratio': sharpe,
            'best_params': best_params
        }

    # Train and evaluate Lasso Logistic Regression with walk-forward validation
    def _train_lasso_regression(self, features_train: pd.DataFrame, target_train: pd.Series,
                                features_test: pd.DataFrame, target_test: pd.Series,
                                price_returns: np.ndarray) -> dict:
        # Put all indicators on the same scale (0 to 1 range)
        data_normalizer = StandardScaler()

        # Scale the training and test data while keeping the column names
        features_train_normalized = pd.DataFrame(data_normalizer.fit_transform(features_train),
                                                 columns=features_train.columns)
        features_test_normalized = data_normalizer.transform(features_test)

        tscv = TimeSeriesSplit(n_splits=3)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        mcc_scorer = make_scorer(matthews_corrcoef)

        def objective(trial):
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
            params = {
                'C': trial.suggest_float('C', 1e-3, 100.0, log=True),
                'penalty': penalty,
                'solver': 'saga',  # Necessary for l1/elasticnet and large-ish data
                'max_iter': 5000,  # Ensure convergence
                'class_weight': 'balanced',
                'random_state': self.seed
            }

            if penalty == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.1, 0.9)

            model = LogisticRegression(**params)

            # Use MCC to score the trials instead of accuracy
            scores = cross_val_score(model, features_train_normalized, target_train, cv=tscv, scoring=mcc_scorer, n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        best_params = study.best_params.copy()

        final_params = {
            'solver': 'saga',
            'max_iter': 5000,
            'class_weight': 'balanced',
            'random_state': self.seed,
            **best_params
        }

        best_model = LogisticRegression(**final_params)
        best_model.fit(features_train_normalized, target_train)

        test_predictions = best_model.predict(features_test_normalized)

        accuracy, sharpe, abs_sharpe, needs_flip = self.evaluate_performance(target_test, test_predictions,
                                                                             price_returns)
        mcc_score = matthews_corrcoef(target_test, test_predictions)

        return {
            'model_type': 'Lasso',
            'mcc': mcc_score,
            'accuracy': accuracy,
            'walk_forward_mcc': study.best_value,
            'sharpe_ratio': sharpe,
            'best_params': best_params
        }

    # Train and evaluate SVC with walk-forward validation
    def _train_support_vector(self, features_train: pd.DataFrame, target_train: pd.Series,
                              features_test: pd.DataFrame,
                              target_test: pd.Series, price_returns: np.ndarray) -> dict:
        # Put all indicators on the same scale (0 to 1 range) [SVC very sensitive to scale]
        data_normalizer = StandardScaler()

        # Scale the training and test data while keeping the column names
        features_train_normalized = pd.DataFrame(data_normalizer.fit_transform(features_train),
                                                 columns=features_train.columns)
        features_test_normalized = data_normalizer.transform(features_test)

        tscv = TimeSeriesSplit(n_splits=3)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        mcc_scorer = make_scorer(matthews_corrcoef)

        def objective(trial):
            kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])
            params = {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'kernel': kernel,
                'probability': True,
                'class_weight': 'balanced',
                'random_state': self.seed
            }

            gamma_type = trial.suggest_categorical('gamma_type', ['scale', 'auto', 'float'])
            if gamma_type == 'float':
                params['gamma'] = trial.suggest_float('gamma_float', 0.01, 0.1)
            else:
                params['gamma'] = gamma_type

            model = SVC(**params)

            # Use MCC to score the trials instead of accuracy
            scores = cross_val_score(model, features_train_normalized, target_train, cv=tscv, scoring=mcc_scorer, n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=15)

        best_params = study.best_params.copy()
        gamma_val = best_params.get('gamma_float', best_params.get('gamma_type'))

        final_params = {
            'C': best_params['C'],
            'kernel': best_params['kernel'],
            'gamma': gamma_val,
            'probability': True,
            'class_weight': 'balanced',
            'random_state': self.seed
        }

        best_model = SVC(**final_params)
        best_model.fit(features_train_normalized, target_train)

        test_predictions = best_model.predict(features_test_normalized)

        accuracy, sharpe, abs_sharpe, needs_flip = self.evaluate_performance(target_test, test_predictions, price_returns)
        mcc_score = matthews_corrcoef(target_test, test_predictions)

        clean_best_params = {'C': best_params['C'], 'kernel': best_params['kernel'], 'gamma': gamma_val}

        return {
            'model_type': 'SVC',
            'mcc': mcc_score,
            'accuracy': accuracy,
            'walk_forward_mcc': study.best_value,
            'sharpe_ratio': sharpe,
            'best_params': clean_best_params
        }

    def _train_lstm(self, features_train: pd.DataFrame, target_train: pd.Series, features_test: pd.DataFrame,
                    target_test: pd.Series, price_returns: np.ndarray) -> dict:

        lstm_model = LSTM(self.seed)
        return lstm_model.train(features_train, target_train, features_test, target_test, price_returns)

    # Run all helper functions and consolidate the best model
    def run_training_pipeline(self) -> bool:
        interval = "1h"

        # Remove any corrupt model paths (i.e. not all necessary files exist validated before call)
        data = pd.read_parquet(os.path.join(DATA_DIR, f"SPY_{interval}.parquet"))
        if data is None: print("No data"); return False

        df = self.calculate_technical_indicators(data, interval)
        if len(df) < 300: print(f"Insufficient data (need 300+, got {len(df)})"); return False

        # Partition data
        train_size = int(len(df) * (1 - self.__test_size))
        train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

        # Remove "Cheat" columns and raw price data AI shouldn't see directly
        # ('target' as is answer key, 'Close' as is too easy to cheat with)
        drop_columns = [c for c in df.columns if 'target' in c or c in ['Open', 'High', 'Low', 'Close', "Adj Close", 'Volume', 'MA_200', 'return']]
        train_columns = [c for c in df.columns if c not in drop_columns]

        # Create input features and answers
        period = interval[1]
        features_train, features_test = train_data[train_columns], test_data[train_columns]

        results = {}

        for h in ([1,2,4,8] if period == "h" else [1,2,5,21]):
            print(f"Training for period {h}...")

            # Alpha Target = (Future Return > Recent Median Return)
            future_ret_train = (train_data['Adj Close'].shift(-h) / train_data['Adj Close']) - 1
            future_ret_test = (test_data['Adj Close'].shift(-h) / test_data['Adj Close']) - 1

            # Median return of the last month (21 days) - calculated per-set to match indices
            median_train = train_data['return'].rolling(window=21).median().fillna(0)
            median_test = test_data['return'].rolling(window=21).median().fillna(0)

            # Create the Alpha target (1 if outperforming its own trend, 0 otherwise)
            targets_train = (future_ret_train > median_train).astype(int).iloc[:-h]
            targets_test = (future_ret_test > median_test).astype(int)

            # Adjust features and actual returns
            features_train_adj = features_train.iloc[:-h]
            actual_returns_test = test_data['return'].values

            # Model competition
            print("Training LSTM...")
            d = self._train_lstm(features_train_adj, targets_train, features_test, targets_test, actual_returns_test)
            print("Training LGBM...")
            a = self._train_lightgbm(features_train_adj, targets_train, features_test, targets_test, actual_returns_test)
            print("Training lasso...")
            b = self._train_lasso_regression(features_train_adj, targets_train, features_test, targets_test, actual_returns_test)
            print("Training SVC...")
            c = self._train_support_vector(features_train_adj, targets_train, features_test, targets_test, actual_returns_test)

            results[h] = [a,b,c,d]

        # Save winning model data
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
        return True



m = TrainingManager()
m.run_training_pipeline()
