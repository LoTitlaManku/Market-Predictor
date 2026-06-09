""""""
# from edgar import Company, set_identity
# set_identity("Name email@gmail.com")

##############################################################################################################
""" dir things """
def list_dir():
    import os
    import json
    import pandas as pd
    from tqdm import tqdm
    from scripts.config import MODEL_DIR, DATA_DIR, LEDGER_DIR

    with open(os.path.join(DATA_DIR, "ticker_map.json"), "r") as f:
        ticker_map = json.load(f)
        ticker_list = set(sorted(list(ticker_map.values()))[::-1])

    predicted = set()
    for ledger in tqdm(os.listdir(LEDGER_DIR)):
        data = pd.read_csv(os.path.join(LEDGER_DIR, ledger))
        if data.empty or len(data) < 300: continue
        predicted.add(ledger.split("_")[0])

    missing = ticker_list - predicted
    print(f"Missing: {missing}")
    print(f"-> {len(missing)}")

def remove_ticker_info():
    import os
    import shutil
    from tqdm import tqdm

    from config import MODEL_DIR, CACHE_DIR, LEDGER_DIR

    tickers = {'HCA', 'GPC', 'EXC', 'CYBR', 'HBM', 'BEN', 'GLW', 'EMN', 'KVUE', 'ESLT', 'SN', 'DE', 'LRN', 'LSCC', 'MMC', 'EPAM', 'OVV', 'EXE', 'JAZZ', 'GILD', 'JD', 'PCAR', 'ENB', 'FRPT', 'UAL', 'WEC', 'WHR', 'LPX', 'KEYS', 'ELF', 'EXEL', 'SOFI', 'WDC', 'GIS', 'ITT', 'J', 'JBHT', 'TAP', 'CRL', 'LASR', 'GME', 'HMY', 'NLY', 'EXPD', 'ICLR', 'KDP', 'KEY', 'JCI', 'HON', 'ELAN', 'PPG', 'T', 'HUM', 'HL', 'ELV', 'GIL', 'RAPT', 'NVO', 'IVZ', 'GLD'}
    for ticker in tqdm(tickers):
        try: os.remove(os.path.join(LEDGER_DIR, f"{ticker}_ledger.csv"))
        except: pass

        # try: shutil.rmtree(os.path.join(MODEL_DIR, f"{ticker}_1h"))
        # except: pass
        # try: shutil.rmtree(os.path.join(MODEL_DIR, f"{ticker}_1d"))
        # except: pass

        # try: os.remove(os.path.join(CACHE_DIR, f"{ticker}_1h.csv"))
        # except: pass
        # try: os.remove(os.path.join(CACHE_DIR, f"{ticker}_1d.csv"))
        # except: pass

def check_model_corruption():
    import os
    import json

    from tqdm import tqdm

    from scripts.predictor import all_ticker_models_exist
    from scripts.config import DATA_DIR, MODEL_DIR

    with open(os.path.join(DATA_DIR, "ticker_map.json"), "r") as f:
        ticker_map = json.load(f)
        ticker_list = sorted(list(ticker_map.values()))[::-1]

    corrupt = []
    for ticker in tqdm(ticker_list):
        if not all_ticker_models_exist(os.path.join(MODEL_DIR, f"{ticker}_1h"), "1h"):
            corrupt.append((ticker, "1h"))

        if not all_ticker_models_exist(os.path.join(MODEL_DIR, f"{ticker}_1d"), "1d"):
            corrupt.append((ticker, "1d"))

    print(corrupt)

##############################################################################################################
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

    for interval in ["1h", "1d"]:
        data = yf.download(key, interval=interval, period="max", auto_adjust=False, progress=True)

        if data.empty:
            print("Empty data")
            continue

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

        data.to_parquet(os.path.join(DATA_DIR, f"{key}_{interval}.parquet"))

##############################################################################################################
""" testing pipeline """
class DetailedException(Exception): pass

def test_train():
    import pandas as pd
    import os

    from scripts.config import MODEL_DIR
    from scripts.predictor import TrainingManager, all_ticker_models_exist, Settings
    from scripts.data_management import load_data

    # ticker = "ASML"
    # interval = "1h"

    ticker_intervals = {'FNB_1d', 'TECH_1d', 'LNG_1d', 'SW_1d', 'ARM_1h', 'FERG_1d', 'MNST_1d', 'BRO_1d', 'FLUT_1d', 'RGLD_1d', 'KGC_1d', 'VSEC_1d', 'RCAT_1d', 'SU_1d'}

    Settings.LOGGING = True

    for ticker, interval in [(t.split("_")) for t in ticker_intervals]:
        # print("Testing", ticker, interval)
        # try:
            full_data = load_data(ticker, interval)
            cutoff_date = full_data.index.max() - pd.Timedelta(days=(60 if interval == "1d" else 20))
            training_data = full_data[full_data.index < cutoff_date]

            if not all_ticker_models_exist(os.path.join(MODEL_DIR, f"{ticker}_{interval}"), interval):
                trainer = TrainingManager()
                trainer.run_training_pipeline(ticker, interval, override_data=training_data)

        # except Exception as e:
        #     print(f"Error training {ticker} on {interval}:\n--> {type(e).__name__} - {e}")

def test_predict():
    import os
    import json
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from lightgbm import Booster as LGBMBooster
    from safetensors.torch import load_file

    from scripts.config import MODEL_DIR, DATA_DIR, LEDGER_DIR
    from scripts.data_management import load_data
    from scripts.predictor import LSTMBrain, save_prediction, get_market_dates, prediction_saved, all_ticker_models_exist
    import scripts.indicators # noqa

    ticker = "AAOI"
    interval = "1d"

    try:
        ## -----  SETUP  ----- ##
        full_data = load_data(ticker, interval)
        model_folder = os.path.join(MODEL_DIR, f"{ticker}_{interval}")

        if not all_ticker_models_exist(model_folder, interval):
            raise Exception("Not all ticker models exist.")

        with open(os.path.join(model_folder, 'metadata.json'), 'r') as f:
            meta = json.load(f)
        with open(os.path.join(DATA_DIR, "model_hyperparameters.json"), 'r') as f:
            hyper_meta = json.load(f)

        ## -----  AI BRAINS  ----- ##
        scaler = joblib.load(f"{model_folder}/scaler.joblib")
        features = joblib.load(f"{model_folder}/features.joblib")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_registry = {}
        horizons = {1: 1, 2: 2, 4: 4, 8: 25} if "h" in interval else {1: 1, 2: 2, 5: 7, 21: 28}
        period = "h" if interval == "1h" else "d"

        for step in horizons:
            horizon_folder = os.path.join(model_folder, f"{step}_horizon_models")
            if not os.path.exists(horizon_folder): continue

            model_registry[step] = {"models": {}, "weights": {}}
            global_meta = hyper_meta.get(f"{step}{period}", [])

            for model_filename in os.listdir(horizon_folder):
                model_path = os.path.join(horizon_folder, model_filename)

                # Identify model type
                try:
                    if ".safetensors" in model_filename:
                        model_type = "LSTM"
                        params = next(m["best_params"] for m in global_meta if m["model_type"] == model_type)
                        # Initialize brain once
                        brain = LSTMBrain(
                            input_dim=len(features),
                            hidden_dim=params["module__hidden_dim"],
                            layers=params["module__layers"],
                            dropout=params["module__dropout"],
                        )
                        brain.load_state_dict(load_file(model_path))
                        brain.to(device).eval()
                        model_registry[step]["models"][model_type] = brain

                    elif ".txt" in model_filename:
                        model_type = "LGBM"
                        model_registry[step]["models"][model_type] = LGBMBooster(model_file=model_path)

                    else:
                        model_type = "SVC" if "SVC" in model_filename else "LASSO"
                        model_registry[step]["models"][model_type] = joblib.load(model_path)

                except Exception as e:
                    raise DetailedException(
                        f"\n[CORRUPT MODEL] {ticker} ({interval}) | Horizon: {step}{period} | File: {model_filename}\n"
                        f"-->{type(e).__name__} - {e}"
                    )

        ## -----  Walk forward predictions  ----- ##
        last_train_date = pd.to_datetime(meta["training data end"])
        processed_df = full_data.ind.add_indicators(ticker, interval)

        trained_data = processed_df[processed_df.index <= last_train_date].tail(400).copy()
        testing_data = processed_df[processed_df.index > last_train_date]

        ledger_file = os.path.join(LEDGER_DIR, f"{ticker}_ledger.csv")
        ledger = None
        if os.path.exists(ledger_file):
            ledger: pd.DataFrame = pd.read_csv(ledger_file)
            ledger['Open_Date'] = pd.to_datetime(ledger['Open_Date'], format='ISO8601')

        weights = {
            "1h": {"LSTM": 0.35, "LGBM": 0.25, "LASSO": 0.25, "SVC": -0.15},
            "25h": {"SVC": 0.4, "LASSO": 0.2, "LGBM": 0.2, "LSTM": 0.2},
            "1d": {"SVC": -0.5, "LSTM": 0.2, "LGBM": 0.2, "LASSO": 0.1},
            "2d": {"SVC": -0.7, "LGBM": 0.1, "LASSO": 0.1, "LSTM": -0.1},
            "28d": {"SVC": 0.5, "LGBM": 0.3, "LASSO": -0.1, "LSTM": -0.1},
            "default": {"LSTM": 0.3, "LGBM": 0.3, "SVC": 0.2, "LASSO": 0.2}
        }

        history = trained_data.copy()
        for current_time in testing_data.index:
            current_row = testing_data.loc[[current_time]]
            history = pd.concat([history, current_row])
            recent_history = history.tail(400).copy()

            if ledger is not None:
                # Check if any entry matches current ticker and last trade date
                match: pd.DataFrame = ledger[(ledger['Interval'] == interval) &
                                             (ledger['Open_Date'] == current_time)]
                if not match.empty: continue

            current_price = recent_history['Adj Close'].iloc[-1]
            current_volatility_atr = float(recent_history['ATR'].iloc[-1])

            target_dates = get_market_dates(current_time, horizons, period)
            if len(target_dates) < 1: continue

            step_forecasts = {}
            for step, bundle in model_registry.items():
                if target_dates[step] is None: continue

                signals = {}
                for model_type, model_obj in bundle["models"].items():
                    if model_type == "LSTM":
                        recent_data = recent_history[features].tail(14)
                        scaled_seq = scaler.transform(recent_data)
                        x_3d = np.expand_dims(scaled_seq, axis=0).astype(np.float32)

                        with torch.no_grad():
                            signals[model_type] = (float(model_obj(torch.from_numpy(x_3d).to(device)).item()) - 0.5) * -2

                    elif model_type == "LGBM":
                        scaled_row = scaler.transform(recent_history[features].iloc[-1:])
                        signals[model_type] = (float(model_obj.predict(scaled_row)[0]) - 0.5) * -2

                    else:  # Lasso / SVC
                        scaled_row = scaler.transform(recent_history[features].iloc[-1:])
                        signals[model_type] = (float(model_obj.predict_proba(scaled_row)[0][1]) - 0.5) * -2

                # Average prediction
                horizon_weights = weights.get(f"{horizons[step]}{period}",
                                              {"LSTM": 0.3, "LGBM": 0.3, "SVC": 0.2, "LASSO": 0.2})
                avg_signal = sum(horizon_weights[key] * signals[key] for key in horizon_weights.keys())

                # Calculate predicted price
                expected_move_magnitude = current_volatility_atr * np.sqrt(step)
                predicted_price = current_price + (2 * avg_signal * expected_move_magnitude)
                capped_width = min(expected_move_magnitude * abs(avg_signal / 2), current_price * 0.1)

                step_forecasts[step] = {
                    "Date_Predicted": current_time.strftime("%Y-%m-%d %H:%M"),
                    'Target_Date': target_dates[step],
                    "Current_Price": current_price,
                    'Predicted_Price': predicted_price,
                    'up': predicted_price + capped_width,
                    'lo': predicted_price - capped_width,
                    'time_difference': horizons[step],
                    'LSTM_signal': signals["LSTM"],
                    'LGBM_signal': signals["LGBM"],
                    'SVC_signal': signals["SVC"],
                    'LASSO_signal': signals["LASSO"],
                    'AVG_signal': avg_signal,

                }

            save_prediction(ticker, interval, step_forecasts)

    except Exception as e:
        if isinstance(e, DetailedException):
            print(e)
        else:
            print(f"\nERROR on {ticker} ({interval})\n--->{type(e).__name__}: {e}")

##############################################################################################################

def validate_ledgers():
    import os

    import pandas as pd
    from tqdm import tqdm

    from data_management import load_data, is_market_open
    from scripts.config import LEDGER_DIR

    for ledger in tqdm(os.listdir(LEDGER_DIR), desc="Validating ledgers", unit="ledger"):
        ledger_path = os.path.join(LEDGER_DIR, ledger)
        ticker = ledger.split("_")[0]

        # Load ledger and find all entries that are unvalidated
        ledger = pd.read_csv(ledger_path)
        NaNs = ledger['Actual_Price'].isna()  # noqa
        if not NaNs.any(): continue

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
                ledger.at[idx, 'Actual_Price'] = round(float(df.asof(target_date)['Close']), 2)
                updated = True

            # If predicted date outside market hours set invalid
            elif not is_market_open(target_date, daily=True):
                ledger.at[idx, "Actual_Price"] = -1
                updated = True

        if updated:
            # Change dates back into strings for pandas datatype consistency
            ledger['Target_Date'] = ledger['Target_Date'].dt.strftime('%Y-%m-%d %H:%M')
            ledger.to_csv(ledger_path, index=False)

def find_dupes():
    from tqdm import tqdm
    import pandas as pd
    import os

    from scripts.config import LEDGER_DIR

    # LEDGER_DIR = LEDGER_DIR.replace("ledgers", "ledgers_1")

    all_data = []
    for filename in tqdm(os.listdir(LEDGER_DIR.replace("ledgers", "ledgers")), desc="Analysing ledgers", unit="ledger"):
        try:
            filepath = os.path.join(LEDGER_DIR.replace("ledgers", "ledgers"), filename)
            ledger = pd.read_csv(filepath)

            completed = ledger.dropna(subset=['Actual_Price']).copy()
            if not completed.empty:
                all_data.append(completed)

        except Exception as e:
            print(f"\nError processing {filename}: {e}")
    if not all_data:
        print("\nNo valid completed predictions found to analyze.")
        exit()
    data = pd.concat(all_data, ignore_index=True)

    # Find all rows that share the same Horizon and Open_Date
    duplicate_mask = data.duplicated(subset=['Horizon', 'Open_Date'], keep=False)

    # Filter the dataframe to see the duplicates
    duplicates = data[duplicate_mask].sort_values(by=['Open_Date', 'Horizon'])

    if not duplicates.empty:
        print(f"\nFound {len(duplicates)} duplicate entries based on Horizon and Open_Date:")
    else:
        print("\nNo duplicates found.")

##############################################################################################################

if __name__ in "__main__":
    import time
    start = time.perf_counter()

    from predictor import TrainingManager, Settings
    Settings.LOGGING = True
    print("Starting...")
    success = TrainingManager().run_training_pipeline("AAPL", "1d", horizon=4)
    print(success)

    # initial_download()

    # import folder_trees
    # folder_trees.generate_tree("C:/Users/adlan_3zfnjq7/Desktop/Alex - Main/Projects/LoTi-Log", ignore_paths=[".briefcase"])

    # find_missing_files()
    # remove_ticker_info()

    # get_special("^VIX")
    # get_special("^VVIX")
    # get_special("^TYX")

    # test_train()
    # test_predict()
    # validate_ledgers()

    # find_dupes()
    # list_dir()
    # check_model_corruption()

    print(time.perf_counter() - start)
    pass


# tickers with no models:
# {'CRL_1h', 'AMGN_1d', 'HSIC_1d', 'SUI_1d', 'ICLR_1h', 'ISRG_1d', 'FAST_1d', 'AXTI_1d', 'OVV_1h', 'CTAS_1d', 'BAH_1d', 'KMX_1d', 'PEGA_1d', 'MRK_1d', 'MCHP_1d', 'RBC_1d', 'PCAR_1h', 'ESLT_1h', 'BEN_1h', 'DUOL_1d', 'TEVA_1d', 'SAN_1d', 'TRU_1d', 'PPG_1h', 'SCCO_1d', 'DE_1h', 'UHS_1d', 'CHTR_1d', 'FFIV_1d', 'MARA_1d', 'EFX_1d', 'AMD_1d', 'USFD_1d', 'INTC_1d', 'GWRE_1d', 'EXK_1d', 'WHR_1h', 'TCOM_1d', 'SWK_1d', 'KTOS_1d', 'WIX_1d', 'MTCH_1d', 'ALLE_1d', 'EOG_1d', 'PR_1d', 'GLW_1h', 'LFUS_1d', 'EPAM_1h', 'WDC_1h', 'EMR_1d'}

# tickers with fucked models:
# {'TAP_1d', 'FRPT_1d', 'SN_1h', 'WEC_1h', 'HUM_1d', 'NLY_1h', 'NLY_1d', 'HUM_1h', 'FRPT_1h', 'TAP_1h', 'LASR_1h', 'WEC_1d', 'CYBR_1d', 'CYBR_1h', 'T_1h', 'MMC_1h', 'SN_1d', 'LASR_1d', 'EXPD_1d', 'GPC_1d', 'T_1d', 'GPC_1h', 'MMC_1d', 'EXPD_1h'}