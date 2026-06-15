import os
import time

import pandas as pd
import numpy as np
import talib
from pykalman import KalmanFilter

from scripts.config import DATA_DIR

@pd.api.extensions.register_dataframe_accessor("ind")
class TechnicalAnalysisAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def add_indicators(self, ticker: str, interval: str, add_targets: bool = True) -> pd.DataFrame:
        df = self._obj
        df.index = df.index.astype('datetime64[ms]')

        # Add all indicators
        df = self._add_sentiment(df, ticker)
        df = self._add_technical_indicators(df, interval)
        df = self._add_vix(df, interval)
        df = self._add_spy(df, interval)
        df = self._add_vix_plus(df, interval)
        df = self._add_macro_context(df, interval)

        if add_targets:
            df = self._add_targets(df, interval)

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
        df["return"] = df["Adj Close"].pct_change()
        for i in range(1, 4):
            df[f"return_lag_{i}"] = df["return"].shift(i)

        # Adjustments
        adj_close = df["Adj Close"].to_numpy(dtype=float)
        raw_close = df["Close"].to_numpy(dtype=float)
        adj_factor = np.divide(
            adj_close,
            raw_close,
            out=np.ones_like(adj_close, dtype=float),
            where=np.isfinite(raw_close) & (raw_close != 0)
        )

        df["Adj High"] = df["High"].to_numpy(dtype=float) * adj_factor
        df["Adj Low"] = df["Low"].to_numpy(dtype=float) * adj_factor
        df["Adj Open"] = df["Open"] * adj_factor

        # Technical indicators
        df['RSI'] = talib.RSI(df['Adj Close'], timeperiod=14)
        macd, _, macdhist = talib.MACD(df['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD_Hist'] = macdhist
        df['ADX'] = talib.ADX(df['Adj High'], df['Adj Low'], df['Adj Close'], timeperiod=14)
        df['ATR'] = talib.ATR(df["Adj High"], df['Adj Low'], df['Adj Close'], timeperiod=14)
        df['MA_200'] = talib.SMA(df['Adj Close'], timeperiod=200)
        df['PDMA_200'] = (df['Adj Close'] / df['MA_200']) - 1

        df['OBV'] = talib.OBV(df['Adj Close'], df['Volume'])
        upper, mid, lower = talib.BBANDS(df['Adj Close'], timeperiod=20)
        df['BBP'] = (df['Adj Close'] - lower) / (upper - lower)
        df['BBP'] = df['BBP'].replace([np.inf, -np.inf], 0.5)
        df['ROC'] = talib.ROC(df['Adj Close'], timeperiod=10)

        # Deviations using all of OHLC
        df['range_pct'] = (df['Adj High'] - df['Adj Low']) / df['Adj Close']
        df['body_pct'] = (df['Adj Close'] - df['Adj Open']) / df['Adj Close']
        df['upper_shadow_pct'] = (df['Adj High'] - df[['Adj Open', 'Adj Close']].max(axis=1)) / df['Adj Close']
        df['lower_shadow_pct'] = (df[['Adj Open', 'Adj Close']].min(axis=1) - df['Adj Low']) / df['Adj Close']

        # Garman-Klass Volatility
        df['GK_vol'] = np.sqrt(
            0.5 * np.log(df['Adj High'] / df['Adj Low']) ** 2
            - (2 * np.log(2) - 1) * np.log(df['Adj Close'] / df['Adj Open']) ** 2
        ).rolling(20).mean()

        # Overnight Gap
        df['overnight_gap'] = (df['Adj Open'] - df['Adj Close'].shift(1)) / df['Adj Close'].shift(1)

        # Distance from N-period high/low
        lookback_52w = 252 if interval == "1d" else 1638
        df['dist_from_52w_high'] = (df['Adj Close'] / df['Adj Close'].rolling(lookback_52w).max()) - 1
        df['dist_from_52w_low'] = (df['Adj Close'] / df['Adj Close'].rolling(lookback_52w).min()) - 1

        # Return skewness and kurtosis
        df['return_skew_20'] = df['return'].rolling(20).skew()
        df['return_kurt_20'] = df['return'].rolling(20).kurt()
        df['return_skew_60'] = df['return'].rolling(60).skew()

        # Multiple-period PDMA
        df['PDMA_50'] = (df['Adj Close'] / talib.SMA(df['Adj Close'], 50)) - 1
        df['PDMA_20'] = (df['Adj Close'] / talib.SMA(df['Adj Close'], 20)) - 1

        # Stochastic %%
        df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['Adj High'], df['Adj Low'], df['Adj Close'])

        # Volume Rate of Change
        df['VROC_10'] = talib.ROC(df['Volume'], timeperiod=10)

        # Multi-period momentum returns
        df['mom_1m'] = df['Adj Close'].pct_change(21).shift(1)
        df['mom_3m'] = df['Adj Close'].pct_change(63).shift(1)
        df['mom_6m'] = df['Adj Close'].pct_change(126).shift(1)

        # Efficiency ratio
        price_diff = df['Adj Close'].diff(20).abs()
        volatility = df['Adj Close'].diff().abs().rolling(20).sum()
        df['Efficiency_Ratio'] = price_diff / volatility

        # Other indicators
        df['vol_ratio'] = df['return'].rolling(5).std() / df['return'].rolling(50).std()
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        return df

    @staticmethod
    def _add_targets(df: pd.DataFrame, interval: str) -> pd.DataFrame:
        window = 20
        df = df.copy()
        n = len(df)

        time_to_gain = np.full(n, window + 1, dtype=np.float32)
        time_to_loss = np.full(n, window + 1, dtype=np.float32)
        tbm_returns = np.zeros(n, dtype=np.float32)
        target_profit = np.zeros(n, dtype=np.int8)
        barrier_strength = np.zeros(n, dtype=np.float32)
        tp_returns = np.zeros(n, dtype=np.float32)
        sl_returns = np.zeros(n, dtype=np.float32)

        close = df["Adj Close"].to_numpy(dtype=float)
        high = df["Adj High"].to_numpy(dtype=float)
        low = df["Adj Low"].to_numpy(dtype=float)
        atr = df["ATR"].to_numpy(dtype=float)

        rolling_window = 252 if interval == "1d" else 1000
        up_quantile = 0.45
        down_quantile = 0.45
        atr_return = pd.Series(atr / close, index=df.index).replace([np.inf, -np.inf], np.nan)

        future_max_return = np.full(n, np.nan, dtype=float)
        future_min_return = np.full(n, np.nan, dtype=float)

        fallback_up_mult = {"1h": 1.2, "1d": 0.8}.get(interval, 1.0)
        fallback_down_mult = {"1h": 0.7, "1d": 0.3}.get(interval, 0.6)

        fallback_up = atr_return * fallback_up_mult
        fallback_down = atr_return * fallback_down_mult

        for i in range(n - window):
            entry = close[i]

            if not np.isfinite(entry) or entry <= 0:
                continue

            future_high = np.nanmax(high[i + 1:i + window + 1])
            future_low = np.nanmin(low[i + 1:i + window + 1])

            future_max_return[i] = (future_high / entry) - 1.0
            future_min_return[i] = (entry / future_low) - 1.0

        hist_up = pd.Series(future_max_return, index=df.index)
        hist_down = pd.Series(future_min_return, index=df.index)

        adaptive_up = hist_up.shift(window).rolling(
            rolling_window,
            min_periods=max(50, rolling_window // 5)
        ).quantile(up_quantile)

        adaptive_down = hist_down.shift(window).rolling(
            rolling_window,
            min_periods=max(50, rolling_window // 5)
        ).quantile(down_quantile)

        adaptive_up = adaptive_up.fillna(fallback_up).clip(
            lower=atr_return * 0.5,
            upper=atr_return * 3.0
        )

        adaptive_down = adaptive_down.fillna(fallback_down).clip(
            lower=atr_return * 1.1,
            upper=atr_return * 2.5
        )

        adaptive_up = adaptive_up.to_numpy(dtype=float)
        adaptive_down = adaptive_down.to_numpy(dtype=float)

        # print("ATR return median:", np.nanmedian(atr_return))
        # print("Adaptive up median:", np.nanmedian(adaptive_up))
        # print("Adaptive down median:", np.nanmedian(adaptive_down))
        # print("Future max median:", np.nanmedian(future_max_return))
        # print("Future min median:", np.nanmedian(future_min_return))

        for i in range(n - window):
            entry = close[i]
            tp_return = adaptive_up[i]
            sl_return = adaptive_down[i]

            if (
                    not np.isfinite(entry)
                    or not np.isfinite(tp_return)
                    or not np.isfinite(sl_return)
                    or entry <= 0
                    or tp_return <= 0
                    or sl_return <= 0
            ):
                continue

            take_profit = entry * (1.0 + tp_return)
            stop_loss = entry * (1.0 - sl_return)

            tp_returns[i] = tp_return
            sl_returns[i] = sl_return

            gain_hit_time = window + 1
            loss_hit_time = window + 1
            max_up_strength = 0.0
            max_down_strength = 0.0

            for j in range(1, window + 1):
                high_j = high[i + j]
                low_j = low[i + j]

                if not np.isfinite(high_j) or not np.isfinite(low_j):
                    continue

                up_strength = ((high_j / entry) - 1.0) / (tp_return + 1e-9)
                down_strength = ((entry / low_j) - 1.0) / (sl_return + 1e-9)

                max_up_strength = max(max_up_strength, up_strength)
                max_down_strength = max(max_down_strength, down_strength)

                if gain_hit_time == window + 1 and high_j >= take_profit:
                    gain_hit_time = float(j)

                if loss_hit_time == window + 1 and low_j <= stop_loss:
                    loss_hit_time = float(j)

                if gain_hit_time != window + 1 and loss_hit_time != window + 1:
                    break

            time_to_gain[i] = gain_hit_time
            time_to_loss[i] = loss_hit_time
            barrier_strength[i] = max(max_up_strength, max_down_strength)

            if gain_hit_time < loss_hit_time:
                target_profit[i] = 1
                tbm_returns[i] = tp_return
            elif loss_hit_time < gain_hit_time:
                target_profit[i] = -1
                tbm_returns[i] = -sl_return
            else:
                target_profit[i] = 0
                if i + window < n and np.isfinite(close[i + window]):
                    tbm_returns[i] = (close[i + window] / entry) - 1.0

        df["time_to_gain"] = time_to_gain
        df["time_to_loss"] = time_to_loss
        df["target_profit"] = target_profit
        df["tbm_return"] = tbm_returns
        df["barrier_strength"] = np.clip(barrier_strength, 0.0, 1.0)
        df["tp_return"] = tp_returns
        df["sl_return"] = sl_returns

        return df.iloc[:-window]

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
            vix_data[['Adj Close']].rename(columns={'Adj Close': 'VIX_Level'}),
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
        spy_returns = aligned_spy['Adj Close'].pct_change()
        stock_returns = df['return']

        # Rolling Beta (60-period): Covariance(stock, market) / Variance(market)
        rolling_cov = stock_returns.rolling(window=60).cov(spy_returns)
        rolling_var = spy_returns.rolling(window=60).var()
        df['Market_Beta'] = (rolling_cov / rolling_var + 1e-9).fillna(1.0)  # Assume 1.0 if no data
        df['Fear_Correlation'] = stock_returns.rolling(window=60).corr(df['VIX_Change']).fillna(0)
        df['Relative_Strength'] = (df['Adj Close'] / aligned_spy['Adj Close']).pct_change().fillna(0)

        return df

    @staticmethod
    def _add_vix_plus(df: pd.DataFrame, interval: str):
        # Market fear volume context
        vvix_data = pd.read_parquet(os.path.join(DATA_DIR, f'VVIX_{interval}.parquet'))
        vvix_data.index.name = "Date"
        vvix_data.index = pd.to_datetime(vvix_data.index, utc=True).tz_localize(None)
        vvix_data = vvix_data[~vvix_data.index.duplicated(keep='first')]

        aligned_vvix = vvix_data.reindex(df.index).ffill().bfill()

        df['VVIX_Level'] = aligned_vvix['Adj Close']
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
            tyx_data[['Adj Close']].rename(columns={'Adj Close': 'Treasury_30Y'}),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        return df