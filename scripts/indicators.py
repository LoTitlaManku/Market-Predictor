import os

import pandas as pd
import numpy as np
import talib
from pykalman import KalmanFilter

from scripts.config import DATA_DIR

@pd.api.extensions.register_dataframe_accessor("ind")
class TechnicalAnalysisAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def add_indicators(self, ticker: str, interval: str, horizon: int = 4) -> pd.DataFrame:
        df = self._obj
        df.index = df.index.astype('datetime64[ms]')

        # Add all indicators
        df = self._add_sentiment(df, ticker)
        df = self._add_technical_indicators(df)
        df = self._add_vix(df, interval)
        df = self._add_spy(df, interval)
        df = self._add_vix_plus(df, interval)
        df = self._add_macro_context(df, interval)
        df = self._add_targets(df, interval, horizon)

        # TEMP FILE
        # df.to_csv("temp_df.csv", index=True)

        return df.dropna()

    @staticmethod
    def _add_sentiment(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        target_file = os.path.join(DATA_DIR, f"master_sentiment.parquet")

        sent_df = pd.read_parquet(target_file, filters=[('ticker', '==', ticker)])
        if len(sent_df) < 300: return df

        sent_df['event_date'] = pd.to_datetime(sent_df['event_date'])
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
    def _add_technical_indicators(df: pd.DataFrame):
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
        df['BBP'] = df['BBP'].replace([np.inf, -np.inf], 0.5)
        df['ROC'] = talib.ROC(df['Close'], timeperiod=10)

        # Deviations using all of OHLC
        df['range_pct'] = (df['High'] - df['Low']) / df['Close']
        df['body_pct'] = (df['Close'] - df['Open']) / df['Close']
        df['upper_shadow_pct'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
        df['lower_shadow_pct'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']

        # Hurst Exponent
        def calculate_hurst(series, window=100):
            if len(series) < window: return 0.5
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) + 1e-9 for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0

        df['Hurst_Exponent'] = df['Close'].rolling(window=100, min_periods=100).apply(calculate_hurst, raw=True)
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
        df['vol_ratio'] = df['return'].rolling(5).std() / df['return'].rolling(50).std()
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        return df

    @staticmethod
    def _add_targets(df: pd.DataFrame, interval: str, horizon: int) -> pd.DataFrame:
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer")

        # Calculate future returns over the given horizon window
        tbm_returns = (df['Adj Close'].shift(-horizon) / df['Adj Close']) - 1

        # Use a rolling window of PAST realized returns to define the barriers (No lookahead bias)
        if interval == "1d": bars = 252
        elif interval == "1h": bars = 1638
        elif interval == "15m": bars = 6552
        else: raise ValueError("Invalid interval")

        historical_returns = df['Adj Close'].pct_change(horizon)

        upper_barrier = historical_returns.rolling(window=bars, min_periods=30).quantile(0.80)
        lower_barrier = historical_returns.rolling(window=bars, min_periods=30).quantile(0.20)

        labels = np.zeros(len(df))

        # Assign labels based on quantiles
        labels[tbm_returns >= upper_barrier] = 1
        labels[tbm_returns <= lower_barrier] = -1

        df['target_profit'] = labels.astype(int)
        df['tbm_return'] = tbm_returns.fillna(0)

        # Filter out rows where rolling barriers are not yet calculated or future horizon is missing
        df = df.iloc[:-horizon]
        valid_idx = upper_barrier.dropna().index.intersection(df.index)

        return df.loc[valid_idx]

    @staticmethod
    def _add_vix(df: pd.DataFrame, interval: str):
        # Market fear context
        vix_data = pd.read_parquet(os.path.join(DATA_DIR, f'VIX_{interval}.parquet'))
        vix_data.index.name = "Date"
        vix_data.index = pd.to_datetime(vix_data.index, utc=True).tz_localize(None)
        vix_data.index = vix_data.index.astype('datetime64[ms]')
        vix_data = vix_data[~vix_data.index.duplicated(keep='first')]

        df = pd.merge_asof(
            df,
            vix_data[['Close']].rename(columns={'Close': 'VIX_Level'}),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        # Fear Momentum
        df['VIX_Change'] = df['VIX_Level'].pct_change().fillna(0)
        # Relative Volatility
        vix_ma = df['VIX_Level'].rolling(window=60).mean()
        df['VIX_Relative'] = (df['VIX_Level'] / vix_ma).fillna(1.0)

        return df

    @staticmethod
    def _add_spy(df: pd.DataFrame, interval: str):
        # Market context
        spy_data = pd.read_parquet(os.path.join(DATA_DIR, f'SPY_{interval}.parquet'))
        spy_data.index.name = "Date"
        spy_data.index = pd.to_datetime(spy_data.index, utc=True).tz_localize(None)
        spy_data = spy_data[~spy_data.index.duplicated(keep='first')]

        aligned_spy = spy_data.reindex(df.index).ffill()
        spy_returns = aligned_spy['Close'].pct_change()
        stock_returns = df['return']

        # Rolling Beta (60-period): Covariance(stock, market) / Variance(market)
        rolling_cov = stock_returns.rolling(window=60).cov(spy_returns)
        rolling_var = spy_returns.rolling(window=60).var()
        df['Market_Beta'] = (rolling_cov / rolling_var + 1e-9).fillna(1.0)  # Assume 1.0 if no data
        df['Fear_Correlation'] = stock_returns.rolling(window=60).corr(df['VIX_Change']).fillna(0)
        df['Relative_Strength'] = (df['Adj Close'] / aligned_spy['Close']).pct_change().fillna(0)

        return df

    @staticmethod
    def _add_vix_plus(df: pd.DataFrame, interval: str):
        # Market fear volume context
        vvix_data = pd.read_parquet(os.path.join(DATA_DIR, f'VVIX_{interval}.parquet'))
        vvix_data.index.name = "Date"
        vvix_data.index = pd.to_datetime(vvix_data.index, utc=True).tz_localize(None)
        vvix_data = vvix_data[~vvix_data.index.duplicated(keep='first')]

        aligned_vvix = vvix_data.reindex(df.index).ffill().bfill()

        df['VVIX_Level'] = aligned_vvix['Close']
        df['VIX_Quality_Ratio'] = df['VVIX_Level'] / df['VIX_Level']

        return df

    @staticmethod
    def _add_macro_context(df: pd.DataFrame, interval: str):
        # Interest Rates (^TYX - 30 Year Yield)
        tyx_data = pd.read_parquet(os.path.join(DATA_DIR, f'TYX_{interval}.parquet'))
        tyx_data.index.name = "Date"
        tyx_data.index = pd.to_datetime(tyx_data.index, utc=True).tz_localize(None)
        tyx_data = tyx_data[~tyx_data.index.duplicated(keep='first')]

        df = pd.merge_asof(
            df,
            tyx_data[['Close']].rename(columns={'Close': 'Treasury_30Y'}),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        return df