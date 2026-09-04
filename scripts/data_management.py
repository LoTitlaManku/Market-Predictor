
# Standard library imports
import os
import json
import re
from functools import lru_cache

# External library imports
import pandas as pd
import pandas_market_calendars as mcal
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import yfinance as yf
from yfinance import shared
from tqdm import tqdm

# Custom imports
from scripts.config import CACHE_DIR, DATA_DIR

NYSE_CAL = mcal.get_calendar('NYSE')

############################################################################

def utc_now_naive() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def _normalise_comparative_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, naive-UTC market-comparison frame.

    Some legacy CSV files contain an old integer index which pandas interprets as
    nanoseconds after 1970.  Those rows are invalid; the parquet history remains
    authoritative and valid newer CSV rows are overlaid on top of it.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    index = pd.to_datetime(df.index, utc=True, errors="coerce", format="mixed").tz_localize(None)
    valid = index.notna() & (index.year > 1970)
    df = df.loc[valid].copy()
    df.index = index[valid]
    df.index.name = "Date"
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    if "Adj Close" in df.columns:
        df = df.dropna(subset=["Adj Close"])
    return df


@lru_cache(maxsize=16)
def _load_comparative_data_cached(key: str, interval: str) -> pd.DataFrame:
    """Combine the clean parquet history with any newer CSV observations."""
    name = key.replace("^", "")
    frames: list[pd.DataFrame] = []

    parquet_path = os.path.join(DATA_DIR, f"{name}_{interval}.parquet")
    if os.path.exists(parquet_path):
        frames.append(_normalise_comparative_frame(pd.read_parquet(parquet_path)))

    csv_path = os.path.join(DATA_DIR, f"{name}_{interval}.csv")
    if os.path.exists(csv_path):
        csv_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        frames.append(_normalise_comparative_frame(csv_df))

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise FileNotFoundError(f"No comparative data found for {name}_{interval}")

    data = pd.concat(frames, axis=0)
    data = data[~data.index.duplicated(keep="last")].sort_index()
    if "Adj Close" not in data.columns:
        raise ValueError(f"Comparative data for {name}_{interval} has no Adj Close column")
    return data


def load_comparative_data(key: str, interval: str) -> pd.DataFrame:
    """Load a defensive copy of a cached comparative market series."""
    return _load_comparative_data_cached(key, interval).copy()


def clear_comparative_data_cache() -> None:
    _load_comparative_data_cached.cache_clear()

# Helper function to load data for a stock
def load_data(ticker: str, interval: str = "1d") -> pd.DataFrame | None:
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{interval}.csv")

    # Return cache file if it exists
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df.index.name = "Date"
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    # Download the appropriate data from yahoo finance elsewise
    period = "730d" if interval in ["1h", "4h"] else "max"
    try: data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    except: return None # noqa
    if data is None or data.empty: return None

    # Flattens columns if MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        cols: pd.MultiIndex = data.columns # noqa
        data.columns = cols.get_level_values(0)

    data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
    data.index.name = "Date"

    now_utc_naive = utc_now_naive()
    schedule = NYSE_CAL.schedule(start_date=now_utc_naive, end_date=now_utc_naive)

    if not schedule.empty:
        mkt_open = pd.Timestamp(schedule.iloc[0]["market_open"]).tz_convert("UTC").tz_localize(None)
        mkt_close = pd.Timestamp(schedule.iloc[0]["market_close"]).tz_convert("UTC").tz_localize(None)

        # If we are currently between open and close, the last downloaded row is "Live"
        if mkt_open <= now_utc_naive <= mkt_close:
            data = data.iloc[:-1]

    # Save it and return data
    data.to_csv(cache_file, index=True)
    return data

def get_day_close(date_to_check: pd.Timestamp | None = None) -> pd.Timestamp:
    if date_to_check is None: date_to_check = pd.Timestamp.now(tz="UTC")

    # Ensure date is UTC Timestamp
    if date_to_check.tz is None: date_to_check = date_to_check.tz_localize('UTC')
    else: date_to_check = date_to_check.tz_convert('UTC')

    start_search = date_to_check - pd.Timedelta(days=3)
    end_search = date_to_check + pd.Timedelta(days=3)
    schedule = NYSE_CAL.schedule(start_date=start_search, end_date=end_search)

    # List of CLOSING times of the HOUR (e.g. 15:30 to 21:00)
    valid_times = mcal.date_range(schedule, frequency="1h")
    day_times = [t for t in valid_times if t.date() == date_to_check.date()]

    # Return as Timestamp or None
    return day_times[-1].tz_localize(None) if day_times else None # noqa

# Helper class to update data for downloaded stocks every 15 minutes
class UpdateWorker:
    def __init__(self):
        super().__init__()
        self.priority_tickers = []
        self._is_running = True

    # Helper function to iterate through cache data to update
    @staticmethod
    def data_updater():
        with (os.scandir(CACHE_DIR) as entries):
            files = [e for e in entries if e.is_file()]
            ticker_list = sorted(list({f.name.split("_")[0] for f in files}))

            # Find the entry with the oldest modification time
            oldest_file = min(files, key=lambda e: e.stat().st_mtime)
            start_date = pd.Timestamp.fromtimestamp(os.path.getmtime(oldest_file.path), tz="UTC").tz_localize(None)
            start_date -= pd.Timedelta(days=5)

        # Download data
        shared._ERRORS = {}
        print(f"Downloading 1d data...")
        batch_data = yf.download(
            ticker_list, start=start_date, interval="1d", group_by='ticker', auto_adjust=False, progress=False
        )
        if batch_data is None or batch_data.empty: return

        # If any failed, retry the download for just those
        if shared._ERRORS: # noqa
            failed_tickers = list(shared._ERRORS.keys()) # noqa
            print(f"\nRetrying failed tickers: {failed_tickers}")
            shared._ERRORS = {}
            extra_data = yf.download(
                failed_tickers, start=start_date, interval="1d", group_by='ticker', auto_adjust=False, progress=False
            )

            if extra_data is not None and not extra_data.empty:
                batch_data = pd.concat([batch_data, extra_data], axis=1)

        for ticker in tqdm(ticker_list, desc="Processing 1d"):
            try:
                new_rows = batch_data[ticker].dropna(how='all')
                if new_rows.empty: continue

                new_rows.index = pd.to_datetime(new_rows.index, utc=True).tz_localize(None)
                close = get_day_close(utc_now_naive())
                if (close is not None and close.normalize() < utc_now_naive() < close
                                      and new_rows.index[-1].date() == close.date()):
                    new_rows = new_rows.iloc[:-1]

                new_rows.index.name = "Date"
                new_rows.index = pd.to_datetime(new_rows.index, utc=True).tz_localize(None).strftime('%Y-%m-%d %H:%M:%S')

                cache_path = os.path.join(CACHE_DIR, f"{ticker}_1d.csv")
                existing_df = load_data(ticker, "1d")

                updated_df = pd.concat([existing_df, new_rows])
                updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
                updated_df = updated_df.loc[:, ~updated_df.columns.duplicated()]
                updated_df.to_csv(cache_path)

            except Exception: continue # noqa

    @staticmethod
    def update_comparatives():
        for comparative in ["^VIX", "^VVIX", "^TYX", "SPY"]:
            # Start from the clean parquet history and overlay valid newer CSV rows.
            # The predictor consumes this same merged representation.
            name = comparative.replace("^", "")
            cache_file = os.path.join(DATA_DIR, f"{name}_1d.csv")
            parquet_file = os.path.join(DATA_DIR, f"{name}_1d.parquet")
            df = load_comparative_data(comparative, "1d")

            # Fetch for the period that has passed
            start_date = df.index.max() - pd.Timedelta(days=5)
            new_data = yf.download(comparative, start=start_date, interval="1d", progress=False, auto_adjust=False)
            if new_data is None or new_data.empty: continue

            if isinstance(new_data.columns, pd.MultiIndex):
                new_data.columns = new_data.columns.get_level_values(0)

            new_data.index = pd.to_datetime(new_data.index, utc=True).tz_localize(None)
            new_data = new_data[new_data.index.notna()]
            new_data.index.name = "Date"

            close = get_day_close(utc_now_naive())
            if (close is not None and close.normalize() < utc_now_naive() < close
                    and new_data.index[-1].date() == close.date()):
                new_data = new_data.iloc[:-1]

            # Append and save
            updated_df = pd.concat([df, new_data])
            updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
            updated_df = updated_df.loc[:, ~updated_df.columns.duplicated()]
            updated_df = updated_df.sort_index()
            updated_df.to_csv(cache_file)
            updated_df.to_parquet(parquet_file)

        clear_comparative_data_cache()

    @staticmethod
    def sentiment_update():
        import google.auth
        from google.cloud import bigquery # noqa

        scopes = ["https://www.googleapis.com/auth/bigquery"]
        credentials, project = google.auth.default(scopes=scopes)

        sent_client = bigquery.Client(
            project="lotitlamanku-market-predictor",
            credentials=credentials,
            client_options={"quota_project_id": "lotitlamanku-market-predictor"}
        )

        with open(os.path.join(DATA_DIR, "ticker_map.json"), "r") as f: ticker_map = json.load(f)
        with open(os.path.join(DATA_DIR, "valid_tickers_with_history.json"), "r") as f: company_tickers = json.load(f)

        ticker_map = {
            str(name).lower(): str(ticker).upper()
            for name, ticker in ticker_map.items()
        }

        sent_dir = os.path.join(DATA_DIR, "master_sentiment.parquet")
        sent_df = pd.read_parquet(sent_dir)

        sent_df["event_date"] = pd.to_datetime(sent_df["event_date"]).dt.normalize()
        start_date = sent_df["event_date"].max() - pd.Timedelta(days=1) # noqa
        end_date = utc_now_naive().normalize() + pd.Timedelta(days=1)

        company_names = [name for name, ticker in ticker_map.items() if ticker in company_tickers]
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
                    `gdelt-bq.gdeltv2.gkg_partitioned`
                WHERE
                    _PARTITIONTIME BETWEEN TIMESTAMP('{start_date}') AND TIMESTAMP('{end_date}')
                    AND REGEXP_CONTAINS(LOWER(V2Organizations), r'''({reg_part})''')
                GROUP BY
                    event_date, organizations
                HAVING
                    article_count > 0
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

        new_df: pd.DataFrame = pd.concat(all_results, ignore_index=True)
        new_df['event_date'] = pd.to_datetime(new_df['event_date'])

        new_df = new_df[new_df["event_date"] < utc_now_naive().normalize()] # noqa

        new_df['weighted_tone'] = new_df['avg_tone'] * new_df['article_count']
        new_df = new_df.groupby(['ticker', 'event_date']).agg({
            'weighted_tone': 'sum',
            'article_count': 'sum'
        }).reset_index()
        new_df['avg_tone'] = new_df['weighted_tone'] / new_df['article_count']
        new_df = new_df.drop(columns=['weighted_tone'])

        full_df = pd.concat([sent_df, new_df]).drop_duplicates(subset=['ticker', 'event_date'], keep='last')

        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        market_days = pd.date_range(start=full_df['event_date'].min(), end=end_date, freq=us_bd) # noqa

        mux = pd.MultiIndex.from_product([company_tickers, market_days], names=['ticker', 'event_date'])
        full_df = full_df.set_index(['ticker', 'event_date']).reindex(mux).reset_index()

        full_df["has_news"] = full_df["article_count"].fillna(0).gt(0).astype(int)
        full_df['article_count'] = full_df['article_count'].fillna(0)
        full_df['avg_tone'] = full_df['avg_tone'].fillna(0)

        full_df = full_df[full_df["event_date"] < utc_now_naive().normalize()]  # noqa
        full_df.to_parquet(sent_dir, index=False)
