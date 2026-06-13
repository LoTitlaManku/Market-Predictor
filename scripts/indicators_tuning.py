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
        df = self._add_technical_indicators(df, interval)
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
        if len(sent_df) < 300:
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
    def _add_technical_indicators(df: pd.DataFrame, interval: str):
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

        # Garman-Klass Volatility
        df['GK_vol'] = np.sqrt(
            0.5 * np.log(df['High'] / df['Low']) ** 2
            - (2 * np.log(2) - 1) * np.log(df['Close'] / df['Open']) ** 2
        ).rolling(20).mean()

        # Overnight Gap
        df['overnight_gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

        # Distance from N-period high/low
        lookback_52w = 252 if interval == "1d" else 1638
        df['dist_from_52w_high'] = (df['Close'] / df['Close'].rolling(lookback_52w).max()) - 1
        df['dist_from_52w_low'] = (df['Close'] / df['Close'].rolling(lookback_52w).min()) - 1

        # Return skewness and kurtosis
        df['return_skew_20'] = df['return'].rolling(20).skew()
        df['return_kurt_20'] = df['return'].rolling(20).kurt()
        df['return_skew_60'] = df['return'].rolling(60).skew()

        # Multiple-period PDMA
        df['PDMA_50'] = (df['Close'] / talib.SMA(df['Adj Close'], 50)) - 1
        df['PDMA_20'] = (df['Close'] / talib.SMA(df['Adj Close'], 20)) - 1

        # Stochastic %%
        df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])

        # Volume Rate of Change
        df['VROC_10'] = talib.ROC(df['Volume'], timeperiod=10)

        # Multi-period momentum returns
        df['mom_1m'] = df['Close'].pct_change(21).shift(1)
        df['mom_3m'] = df['Close'].pct_change(63).shift(1)
        df['mom_6m'] = df['Close'].pct_change(126).shift(1)

        # Efficiency ratio
        price_diff = df['Close'].diff(20).abs()
        volatility = df['Close'].diff().abs().rolling(20).sum()
        df['Efficiency_Ratio'] = price_diff / volatility

        # Other indicators
        df['vol_ratio'] = df['return'].rolling(5).std() / df['return'].rolling(50).std()
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        return df

    @staticmethod
    def _add_targets(df: pd.DataFrame, interval: str, horizon: int) -> pd.DataFrame:
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        future_returns = np.zeros(len(df))
        labels = np.zeros(len(df))
        barrier_strength = np.zeros(len(df))

        close = df['Adj Close'].values
        high = df['High'].values
        low = df['Low'].values
        atr = df['ATR'].values

        up_strength_hist = np.full(len(df), np.nan)
        down_strength_hist = np.full(len(df), np.nan)

        for i in range(len(df) - horizon):
            entry = close[i]

            if np.isnan(entry) or np.isnan(atr[i]) or atr[i] <= 0:
                continue

            atr_return = atr[i] / entry
            future_high = np.nanmax(high[i + 1:i + horizon + 1])
            future_low = np.nanmin(low[i + 1:i + horizon + 1])

            up_strength_hist[i] = ((future_high / entry) - 1) / (atr_return + 1e-9)
            down_strength_hist[i] = ((entry / future_low) - 1) / (atr_return + 1e-9)

        if interval == "1d":
            lookback = 252
            min_periods = 60
        elif interval == "1h":
            lookback = 1638
            min_periods = 200
        else:
            raise ValueError("Invalid interval")

        up_strength_series = pd.Series(up_strength_hist, index=df.index)
        down_strength_series = pd.Series(down_strength_hist, index=df.index)

        up_barrier_mult = up_strength_series.shift(horizon).rolling(lookback, min_periods).quantile(0.80)
        down_barrier_mult = down_strength_series.shift(horizon).rolling(lookback, min_periods).quantile(0.80)

        fallback_mult = {
            "1h": {2: 0.8, 4: 1.2, 8: 1.6},
            "1d": {2: 0.5, 4: 0.8, 8: 1.1},
        }

        fallback = fallback_mult.get(interval, {}).get(horizon, 1.0)
        up_barrier_mult = up_barrier_mult.fillna(fallback).clip(0.25, 3.0)
        down_barrier_mult = down_barrier_mult.fillna(fallback).clip(0.25, 3.0)

        for i in range(len(df) - horizon):
            entry = close[i]

            if np.isnan(entry) or np.isnan(atr[i]) or atr[i] <= 0:
                labels[i] = 0
                future_returns[i] = 0
                barrier_strength[i] = 0
                continue

            tp_return = (atr[i] * up_barrier_mult.iloc[i]) / entry
            sl_return = (atr[i] * down_barrier_mult.iloc[i]) / entry

            take_profit = entry * (1.0 + tp_return)
            stop_loss = entry * (1.0 - sl_return)

            hit_label = 0
            hit_return = (close[i + horizon] / entry) - 1

            max_up_strength = 0.0
            max_down_strength = 0.0

            for j in range(1, horizon + 1):
                high_j = high[i + j]
                low_j = low[i + j]

                up_strength = ((high_j / entry) - 1) / (tp_return + 1e-9)
                down_strength = ((entry / low_j) - 1) / (sl_return + 1e-9)

                max_up_strength = max(max_up_strength, up_strength)
                max_down_strength = max(max_down_strength, down_strength)

                if low_j <= stop_loss:
                    hit_label = -1
                    hit_return = -sl_return
                    break

                if high_j >= take_profit:
                    hit_label = 1
                    hit_return = tp_return
                    break

            labels[i] = hit_label
            future_returns[i] = hit_return
            barrier_strength[i] = max(max_up_strength, max_down_strength)

        df['target_profit'] = labels.astype(int)
        df['tbm_return'] = future_returns
        df['barrier_strength'] = np.clip(barrier_strength, 0.0, 1.0)

        return df.iloc[:-horizon]

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