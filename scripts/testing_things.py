
""" downloading data things """
def data_update():
    import os
    import pandas as pd
    import yfinance as yf
    from yfinance import shared
    from datetime import datetime, timezone
    from tqdm import tqdm

    from config import CACHE_DIR
    from data_management import NYSE_CAL, load_data

    with os.scandir(CACHE_DIR) as entries:
        files = [e for e in entries if e.is_file()]
        ticker_list = sorted(list({f.name.split("_")[0] for f in files}))

        # Find the entry with the oldest modification time
        oldest_file = min(files, key=lambda e: e.stat().st_mtime)
        start_date = os.path.getmtime(oldest_file.path) - pd.Timedelta(days=1)

    for interval in ["1h", "1d"]:
        # Download data
        shared._ERRORS = {}
        batch_data = yf.download(ticker_list, start=start_date, interval=interval,
                                 group_by='ticker', auto_adjust=False, progress=True)

        # If any failed, retry the download for just those
        if shared._ERRORS:
            print("Retrying failed tickers...")
            failed_tickers = list(shared._ERRORS.keys())
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

            except Exception: continue

def initial_download():
    import os
    import pandas as pd
    import yfinance as yf
    from datetime import datetime, timezone
    from tqdm import tqdm
    import json

    from config import CACHE_DIR, DATA_DIR
    from data_management import NYSE_CAL

    # with open(os.path.join(DATA_DIR, "ticker_map.json"), "r") as f:
    #     ticker_map = json.load(f)
    #     ticker_list = sorted([f for f in ticker_map.values()])

    ticker_list = ["CYBR", "JEF", "MMC", "ROST", "RYAAY", "AA", "CYBR", "KIM", "MMC"]

    for interval in ["1h", "1d"]:
        for ticker in tqdm(ticker_list, desc=f"Downloading for {interval}"):
            try:
                ticker_df = yf.download(ticker, interval=interval, period="max", auto_adjust=False, progress=False)
                if ticker_df.empty: continue

                if isinstance(ticker_df.columns, pd.MultiIndex):
                    ticker_df.columns = ticker_df.columns.get_level_values(0)

                now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
                schedule = NYSE_CAL.schedule(start_date=now_utc_naive, end_date=now_utc_naive)
                is_market_currently_open = False

                if not schedule.empty:
                    mkt_open = schedule.iloc[0]['market_open'].replace(tzinfo=None)
                    mkt_close = schedule.iloc[0]['market_close'].replace(tzinfo=None)
                    is_market_currently_open = mkt_open <= now_utc_naive <= mkt_close

                if is_market_currently_open:
                    ticker_df = ticker_df.iloc[:-1]

                ticker_df.index = pd.to_datetime(ticker_df.index, utc=True).tz_localize(None)
                ticker_df.index.name = "Date"
                ticker_df.index = ticker_df.index.strftime('%Y-%m-%d %H:%M:%S')

                cache_path = os.path.join(CACHE_DIR, f"{ticker}_{interval}.csv")
                ticker_df.to_csv(cache_path)

            except Exception as e:
                tqdm.write(f"Error updating {ticker}: {e}")
                continue

def get_spy():
    import pandas as pd
    import os
    from config import DATA_DIR
    import yfinance as yf
    from datetime import datetime, timezone
    from data_management import NYSE_CAL

    for interval in ["1h", "1d"]:
        # df = pd.read_csv(os.path.join(DATA_DIR, f"SPY_{interval}.csv"))

        data = yf.download("SPY", interval=interval, period="max", auto_adjust=False, progress=False)

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

        data.to_parquet(os.path.join(DATA_DIR, f"SPY_{interval}.parquet"))

def get_special(key):
    import pandas as pd
    import os
    from config import DATA_DIR
    import yfinance as yf
    from datetime import datetime, timezone
    from data_management import NYSE_CAL

    data = yf.download(key, interval="1d", period="max", auto_adjust=False, progress=True)
    # data = pd.read_parquet(os.path.join(DATA_DIR, f"{key}_1d.parquet"))

    if data.empty or data is None:
        print(f"Empty data: {key} - 1d")
        return

    # Flattens columns if MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        cols: pd.MultiIndex = data.columns
        data.columns = cols.get_level_values(0)

    data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
    data.index.name = "Date"

    now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
    schedule = NYSE_CAL.schedule(start_date=now_utc_naive, end_date=now_utc_naive)
    if not schedule.empty:
        mkt_close = schedule.iloc[0]['market_close'].replace(tzinfo=None)

        # If we are currently before close, the last downloaded row is "Live"
        if now_utc_naive <= mkt_close:
            data = data.iloc[:-1]

    data.to_csv(os.path.join(DATA_DIR, f"{key}_1d.csv"))

########################################################################################################################

def assign_profile(vol: float, beta: float, adv: float) -> str:
    if vol < 0.18:
        return "A"

    if 0.18 <= vol < 0.28:
        if beta < 0.9:
            return "E"
        return "B" if adv > 5_000_000 else "F"

    if 0.28 <= vol < 0.40:
        return "C"

    return "D"

def build_profile_map():
    import os
    import json
    import pandas as pd
    import numpy as np

    from scripts.config import DATA_DIR
    from scripts.data_management import load_data

    with open(os.path.join(DATA_DIR, "ticker_map.json"), "r") as f:
        ticker_map = json.load(f)
        tickers = sorted(list(ticker_map.values()))

    spy_df = pd.read_parquet(os.path.join(DATA_DIR, 'SPY_1d.parquet'))
    spy_returns = spy_df["Adj Close"].pct_change().rename("benchmark_return")

    profile_map = {"A": {}, "B": {}, "C": {}, "D": {}, "E": {}, "F": {}}
    for ticker in tickers:
        # try:
        df = load_data(ticker, "1d")

        if df is None or df.empty or len(df) < 300:
            print(f"Skipping {ticker}: not enough data")
            continue

        returns = df["Adj Close"].pct_change().rename("stock_return")
        aligned = pd.concat([returns, spy_returns], axis=1, sort=True).dropna().tail(500)

        if len(aligned) < 300:
            print(f"Skipping {ticker}: not enough aligned benchmark rows")
            continue

        vol = aligned["stock_return"].std() * np.sqrt(252)
        beta = aligned["stock_return"].cov(aligned["benchmark_return"])
        beta /= aligned["benchmark_return"].var() + 1e-12
        adv = df["Volume"].tail(252).mean()

        if not np.isfinite(vol) or not np.isfinite(beta) or not np.isfinite(adv):
            print(f"Skipping {ticker}: non-finite profile values")
            continue


        profile_map[assign_profile(float(vol), float(beta), float(adv))][ticker] = {
            "vol": float(vol),
            "beta": float(beta),
            "adv": float(adv),
        }

        # except Exception as e:
        #     print(f"Skipping {ticker}: {e}")

    # for profile, tickers in profile_map.items():
    #     with open(os.path.join(GROUP_DIR, f"Profile {profile}", "tickers.json"), "w") as f:
    #         json.dump(tickers, f)

########################################################################################################################

def find_latest():
    import pandas as pd
    import os
    from scripts.config import DATA_DIR

    sent_df = pd.read_parquet(os.path.join(DATA_DIR, f"master_sentiment.parquet"), filters=[('ticker', '==', "AAPL")])
    sent_df = sent_df.sort_values('event_date')

    print(sent_df['event_date'].max())

def updates(sent: bool = False, spy: bool = False, cache: bool = False):
    from data_management import UpdateWorker
    updater = UpdateWorker()
    if sent:
        print("--- Updating Global Sentiment ---")
        updater.sentiment_update()
    if spy:
        print("--- Updating Global Comparison data ---")
        updater.update_comparatives()
    if cache:
        print(f"--- Updating Prices for tickers ---")
        updater.data_updater()

########################################################################################################################

if __name__ in "__main__":
    import time
    start = time.perf_counter()

    # import time_machine
    # from datetime import datetime, timezone
    # target_time = datetime(2026, 6, 10, 15, 0, 0, tzinfo=timezone.utc)
    # with time_machine.travel(target_time):
    #     f()

    updates(  # Whether to update:
        sent=True,  # News sentiment
        spy=True,  # Market sentiment indicators
        cache=True,  # Stock cache
    )
    # find_latest()

    from predictor import Predictor
    print("Training...")
    mng = Predictor("1d")
    mng.run_pipeline()

    # from folder_trees import generate_tree
    # generate_tree("/home/god/Projects/market_predictor", ignore_paths=[".bin", ".venv", "cache_files", "imgs"])


    print(time.perf_counter() - start)
    pass


    # import folder_trees
    # folder_trees.generate_tree("C:/Users/adlan_3zfnjq7/Desktop/Alex - Main/Projects/LoTi-Log", ignore_paths=[".briefcase"])
