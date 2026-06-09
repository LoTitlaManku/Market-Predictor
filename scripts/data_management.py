
# Standard library imports
import os
from datetime import datetime, timezone, timedelta
import json
import re

# External library imports
import pandas as pd
import pandas_market_calendars as mcal
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import yfinance as yf
from yfinance import shared
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QLabel, QProgressBar
from tqdm import tqdm

# Custom imports
from scripts.config import CACHE_DIR, LEDGER_DIR, DATA_DIR, IMG_DIR

NYSE_CAL = mcal.get_calendar('NYSE')

############################################################################

# Helper function to find the absolute path of image files
def abs_file(file: str) -> str:
    return os.path.join(IMG_DIR, file).replace("\\", "/")

# Helper function to load data for a stock
def load_data(ticker: str, interval: str = "1d") -> pd.DataFrame | None:
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{interval}.csv")

    # Return cache file if it exists
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df.index.name = "Date"
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        return df

    # Download the appropriate data from yahoo finance elsewise
    period = "730d" if interval in ["1h", "4h"] else "max"
    try: data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    except: return None
    if data.empty: return None

    # Flattens columns if MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        cols: pd.MultiIndex = data.columns
        data.columns = cols.get_level_values(0)

    data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
    data.index.name = "Date"

    now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
    schedule = NYSE_CAL.schedule(start_date=now_utc_naive, end_date=now_utc_naive)

    if not schedule.empty:
        mkt_open = schedule.iloc[0]['market_open'].replace(tzinfo=None)
        mkt_close = schedule.iloc[0]['market_close'].replace(tzinfo=None)

        # If we are currently between open and close, the last downloaded row is "Live"
        if mkt_open <= now_utc_naive <= mkt_close:
            data = data.iloc[:-1]

    # Save it and return data
    data.to_csv(cache_file, index=True)
    return data

# Helper function to load the latest n days for a stock
def peek_data(ticker: str, days: int, interval: str = "15m") -> pd.DataFrame | None:
    # Find data or download if doesn't exist
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{interval}.csv")
    if not os.path.exists(cache_file): load_data(ticker, interval)

    # Ensure data in correct format
    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).tz_localize(None)

    if df.empty: return None

    # Return the appropriate range of data
    cutoff_date = df.index.max() - pd.Timedelta(days=days)
    return df[df.index >= cutoff_date]

# Helper function to check whether a ticker is valid
def validate_ticker(ticker: str) -> bool:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        return not data.empty
    except: return False

def is_market_open(date_to_check: pd.Timestamp = None, daily: bool = False) -> bool:
    if date_to_check is None:
        date_to_check = datetime.now()

    # Ensure date is UTC Timestamp
    if date_to_check.tz is None:
        date_to_check = date_to_check.tz_localize('UTC')
    else:
        date_to_check = date_to_check.tz_convert('UTC')

    start_search = date_to_check - pd.Timedelta(days=1)
    end_search = date_to_check + pd.Timedelta(days=1)
    schedule = NYSE_CAL.schedule(start_date=start_search, end_date=end_search)

    if daily:
        # For daily: List of DAYs (e.g. '2026-03-06')
        valid_times = schedule.index.normalize()
        date_to_check = date_to_check.strftime("%Y-%m-%d")
    else:
        # For hourly: List of CLOSING times of the HOUR (e.g. 15:30 to 21:00)
        valid_times = mcal.date_range(schedule, frequency="1h")

    return date_to_check in valid_times

# Helper class to update data for downloaded stocks every 15 minutes
class UpdateWorker(QThread):
    # Thread signals
    progress_msg: pyqtSignal = pyqtSignal(str)
    progress_val: pyqtSignal = pyqtSignal(int)
    updates_finished: pyqtSignal = pyqtSignal()

    def __init__(self):
        super().__init__()
        # Priority tickers to ensure any added to graph are completely up to date
        self.priority_tickers = []
        self._is_running = True

    # Helper function to run checks and emit a finished signal
    def run(self):
        # self.data_updater()
        self.sentiment_update()
        self.update_spy()
        # self.check_accuracy()
        self.updates_finished.emit()

    # Helper function to iterate through cache data to update
    @staticmethod
    def data_updater():
        with os.scandir(CACHE_DIR) as entries:
            files = [e for e in entries if e.is_file()]
            ticker_list = sorted(list({f.name.split("_")[0] for f in files}))

            # Find the entry with the oldest modification time
            oldest_file = min(files, key=lambda e: e.stat().st_mtime)
            start_date = datetime.fromtimestamp(os.path.getmtime(oldest_file.path), tz=timezone.utc) - pd.Timedelta(days=1)

        for interval in ["1h", "1d"]:
            # Download data
            shared._ERRORS = {}
            print(f"Downloading {interval} data...")
            batch_data = yf.download(ticker_list, start=start_date, interval=interval,
                                     group_by='ticker', auto_adjust=False, progress=True)

            # If any failed, retry the download for just those
            if shared._ERRORS:
                failed_tickers = list(shared._ERRORS.keys())
                print(f"\nRetrying failed tickers: {failed_tickers}")
                shared._ERRORS = {}
                extra_data = yf.download(failed_tickers, start=start_date, interval=interval,
                                         group_by='ticker', auto_adjust=False, progress=True)

                if not extra_data.empty:
                    batch_data = pd.concat([batch_data, extra_data], axis=1)

            # Find whether the market is open
            now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
            schedule = NYSE_CAL.schedule(start_date=now_utc_naive, end_date=now_utc_naive)
            is_market_currently_open = False

            if not schedule.empty:
                mkt_open = schedule.iloc[0]['market_open'].replace(tzinfo=None)
                mkt_close = schedule.iloc[0]['market_close'].replace(tzinfo=None)
                is_market_currently_open = mkt_open <= now_utc_naive <= mkt_close

            for ticker in tqdm(ticker_list, desc=f"Processing {interval}"):
                try:
                    new_rows = batch_data[ticker].dropna(how='all')
                    if new_rows.empty: continue

                    new_rows.index = pd.to_datetime(new_rows.index, utc=True).tz_localize(None)
                    if is_market_currently_open:
                        new_rows = new_rows.iloc[:-1]

                    new_rows.index.name = "Date"
                    new_rows.index = pd.to_datetime(new_rows.index, utc=True).tz_localize(None).strftime('%Y-%m-%d %H:%M:%S')

                    cache_path = os.path.join(CACHE_DIR, f"{ticker}_{interval}.csv")
                    existing_df = load_data(ticker, interval)

                    updated_df = pd.concat([existing_df, new_rows])
                    updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
                    updated_df = updated_df.loc[:, ~updated_df.columns.duplicated()]
                    updated_df.to_csv(cache_path)

                except Exception:
                    continue

    @staticmethod
    def update_comparatives():
        for comparative in ["^VIX", "^VVIX", "^TYX", "SPY"]:
            for interval in ["1h", "1d"]:
                # Get the needed interval format for yfinance from filename
                seconds_map = {"m": 60, "h": 3600, "d": 86400}
                unit, value = ''.join(filter(str.isalpha, interval)), int(''.join(filter(str.isdigit, interval)))
                interval_seconds = seconds_map[unit] * value

                # Load existing cached stock data from file
                cache_file = os.path.join(DATA_DIR, f"{comparative.replace("^", "")}_{interval}.parquet")

                df = pd.read_parquet(cache_file)
                df.index.name = "Date"
                df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
                df = df.loc[:, ~df.columns.duplicated()]

                # Find time period for which data needs to be downloaded
                time_diff = datetime.now(timezone.utc).replace(tzinfo=None) - df.index[-1]
                period = f"{int(min((time_diff.total_seconds() // 86400) + 5, 700))}d"

                needs_update = (time_diff.total_seconds() >= interval_seconds)
                if needs_update:
                    # Fetch for the period that has passed
                    new_data = yf.download(comparative, period=period, interval=interval, progress=False, auto_adjust=False)
                    if new_data.empty: return

                    if isinstance(new_data.columns, pd.MultiIndex):
                        new_data.columns = new_data.columns.get_level_values(0)

                    new_data.index = pd.to_datetime(new_data.index, utc=True).tz_localize(None)
                    new_data.index.name = "Date"

                    now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
                    schedule = NYSE_CAL.schedule(start_date=now_utc_naive, end_date=now_utc_naive)

                    if not schedule.empty:
                        mkt_open = schedule.iloc[0]['market_open'].replace(tzinfo=None)
                        mkt_close = schedule.iloc[0]['market_close'].replace(tzinfo=None)

                        # If we are currently between open and close, the last downloaded row is "Live"
                        if mkt_open <= now_utc_naive <= mkt_close:
                            new_data = new_data.iloc[:-1]

                    # Append and save
                    updated_df = pd.concat([df, new_data])
                    updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
                    updated_df = updated_df.loc[:, ~updated_df.columns.duplicated()]
                    updated_df.to_parquet(cache_file)

    @staticmethod
    def sentiment_update():
        from google.cloud import bigquery
        sent_client = bigquery.Client(
            project="market-predictor-throwaway",
            client_options={"quota_project_id": "market-predictor-throwaway"}
        )

        with open(os.path.join(DATA_DIR, "ticker_map.json"), "r") as f:
            ticker_map = json.load(f)

        with open(os.path.join(DATA_DIR, "valid_tickers_with_history.json"), "r") as f:
            company_tickers = json.load(f)

        sent_dir = os.path.join(DATA_DIR, "master_sentiment.parquet")
        sent_df = pd.read_parquet(sent_dir)

        start_date = sent_df["event_date"].max() - timedelta(days=1)
        end_date = pd.Timestamp.now().normalize()

        if start_date == end_date - timedelta(days=2):
            print("Do not need to update")
            return

        company_names = [name.lower() for name, ticker in ticker_map.items() if ticker in set(company_tickers)]
        half = len(company_names) // 2
        regex_parts = [
            "|".join([rf"\b{re.escape(name)}\b" for name in company_names[:half]]),
            "|".join([rf"\b{re.escape(name)}\b" for name in company_names[half:]])
        ]

        all_results = []
        for i, reg_part in enumerate(regex_parts):
            query = f"""
                SELECT
                    DATE(_PARTITIONTIME) AS event_date,
                    LOWER(V2Organizations) AS organizations,
                    AVG(CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) AS avg_tone,
                    COUNT(*) AS article_count
                FROM
                    -- Using the strictly partitioned table to save quota
                    `gdelt-bq.gdeltv2.gkg_partitioned`
                WHERE
                    _PARTITIONTIME BETWEEN TIMESTAMP('{start_date}') AND TIMESTAMP('{end_date}')
                    AND REGEXP_CONTAINS(LOWER(V2Organizations), r'''({reg_part})''')
                GROUP BY
                    event_date, organizations
                HAVING 
                    article_count > 2
                ORDER BY
                    event_date ASC
            """
            df = sent_client.query(query).to_dataframe()

            if not df.empty:
                df['matched'] = df['organizations'].str.extract(f'({reg_part})', flags=re.IGNORECASE, expand=False)
                df['ticker'] = df['matched'].str.lower().map(ticker_map)

                df = df.dropna(subset=['ticker'])
                all_results.append(df[['ticker', 'event_date', 'avg_tone', 'article_count']])

        if not all_results:
            print("No data retrieved.")
            return

        new_df = pd.concat(all_results, ignore_index=True)
        new_df['event_date'] = pd.to_datetime(new_df['event_date'])

        new_df = new_df[new_df['event_date'] < end_date]

        new_df['weighted_tone'] = new_df['avg_tone'] * new_df['article_count']
        new_df = new_df.groupby(['ticker', 'event_date']).agg({
            'weighted_tone': 'sum',
            'article_count': 'sum'
        }).reset_index()
        new_df['avg_tone'] = new_df['weighted_tone'] / new_df['article_count']
        new_df = new_df.drop(columns=['weighted_tone'])

        full_df = pd.concat([sent_df, new_df]).drop_duplicates(subset=['ticker', 'event_date'], keep='last')

        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        market_days = pd.date_range(start=full_df['event_date'].min(), end=end_date - timedelta(days=1), freq=us_bd)

        mux = pd.MultiIndex.from_product([company_tickers, market_days], names=['ticker', 'event_date'])
        full_df = full_df.set_index(['ticker', 'event_date']).reindex(mux).reset_index()

        full_df['has_news'] = full_df['article_count'].notna().astype(int)
        full_df['article_count'] = full_df['article_count'].fillna(0)
        full_df['avg_tone'] = full_df['avg_tone'].fillna(0)

        full_df.to_parquet(sent_dir, index=False)

    # Helper function to iterate through ledgers to validate
    def check_accuracy(self):
        # TQDM also used for console feedback
        ledgers = os.listdir(LEDGER_DIR)
        for i, filename in enumerate(tqdm(ledgers, desc="Checking accuracy", unit="ledger")):
            filename: str = filename

            # Emit progress to show in widgets in main menu
            self.progress_msg.emit(f"Checking Ledger: {filename}")
            self.progress_val.emit(int((i / len(ledgers)) * 100))

            # Validate ledger
            ticker = filename.split("_")[0]
            self.validate_ledger(ticker, os.path.join(LEDGER_DIR, filename))

        self.progress_msg.emit("Completed ledger check")
        self.progress_val.emit(100)

    # Helper function to validate ledger for a stock
    @staticmethod
    def validate_ledger(ticker: str, ledger_path: str):
        print(f"Validating Ledger: {ticker}")
        # Load ledger and find all entries that are unvalidated
        ledger = pd.read_csv(ledger_path)
        NaNs = ledger['Actual_Price'].isna() # noqa
        if not NaNs.any(): return

        # Load existing hourly and daily data for the stock
        ledger['Target_Date'] = pd.to_datetime(ledger['Target_Date'], format='ISO8601')
        df_h = load_data(ticker, "1h")
        df_d = load_data(ticker, "1d")

        # Iterate through all rows in ledger and validate
        updated = False
        for idx, row in ledger[NaNs].iterrows():
            target_date = row['Target_Date'] - (pd.Timedelta(hours=1) if row['Interval'] == "1h" else pd.Timedelta(0))
            df = df_h if row['Interval'] == "1h" else df_d

            if row["Interval"] == "1d":
                target_date = target_date.normalize()

            # If predicted date has passed, determine correctness of prediction
            if target_date in df.index:
                start_price = row["Current_Price"]
                actual_price = float(df.asof(target_date)['Close'])
                pred_price = float(row['Predicted_Price'])
                direction = row['Direction']

                # Check if the prediction is correct
                direction_correct = (("UP" in direction and actual_price > start_price)
                                     or ("DOWN" in direction and actual_price < start_price))
                price_accurate = abs(actual_price - pred_price) / actual_price <= 0.02

                # Update validation fields in the records as integers for pandas datatype consistency
                ledger.at[idx, 'Actual_Price'] = round(actual_price, 2)
                ledger.at[idx, 'Is_Correct'] = int(direction_correct and price_accurate)
                updated = True

            # If predicted date outside market hours set invalid
            elif not is_market_open(target_date, daily=True):
                ledger.at[idx, "Actual_Price"] = -1
                ledger.at[idx, "Is_Correct"] = -1
                updated = True

        if updated:
            # Change dates back into strings for pandas datatype consistency
            ledger['Target_Date'] = ledger['Target_Date'].dt.strftime('%Y-%m-%d %H:%M')
            ledger.to_csv(ledger_path, index=False)

# Manager class to control when data updates happen
class UpdateManager(QObject):
    timer: QTimer
    utc_open: pd.Timestamp
    utc_close: pd.Timestamp

    def __init__(self, progress_label: QLabel = None, progress_bar: QProgressBar = None):
        super().__init__()
        self.worker = UpdateWorker()

        self.labels = all(label is not None for label in [progress_label, progress_bar])
        if self.labels:
            # Save parent widgets that display progress
            self.plabel = progress_label
            self.pbar = progress_bar

            # Connect signals
            self.worker.progress_msg.connect(self.plabel.setText)
            self.worker.progress_val.connect(self.pbar.setValue)

        # Start timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.start_updating)

        # Run once on startup
        self.start_updating()

        # Calculate how long to wait till next update then run every hour
        now = datetime.now(timezone.utc)
        target = now.replace(minute=31, second=0, microsecond=0)
        if now >= target:
            target += timedelta(hours=1)

        initial_delay_ms = int((target - now).total_seconds() * 1000)
        QTimer.singleShot(initial_delay_ms, self.update_loop)

    # Helper function loop to run updating script
    def update_loop(self):
        # Check if market is open
        if not is_market_open(): return

        # Run loop
        self.start_updating()
        if self.labels: self.plabel.setText("Up to date")

        self.timer.start(3600000) # 1 hour

    # Helper function to start the update worker
    def start_updating(self):
        if self.worker.isRunning(): return
        self.worker.start()

    # Helper function to add a stock to prioritise to ensure fully updated data being used
    def prioritize(self, ticker: str):
        if ticker in self.worker.priority_tickers: return
        self.worker.priority_tickers.append(ticker)

