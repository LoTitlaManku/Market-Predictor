
import time
import os # noqa (it is used idk why pycharm says it isn't)
import sys
import json
from multiprocessing import Manager, Queue, Pool
from multiprocessing.queues import Queue as MPQueue
import logging
import warnings
from typing import Optional

import pandas as pd
import numpy as np
import joblib
import torch # noqa # must be imported here first to initialise DLLS correctly
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QPushButton
from PyQt6.QtCore import QTimer
from sympy.codegen.ast import continue_ # noqa (says unused ngl don't remmeber if i use it)
from tqdm import tqdm

from scripts.config import DATA_DIR, MODEL_DIR, LEDGER_DIR
from scripts.data_management import load_data, UpdateWorker
import scripts.indicators  # noqa

logger = logging.getLogger('yfinance')
logger.setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", message="Your application has authenticated using end user credentials")

class DetailedException(Exception): pass

# For training threads
update_queue: Optional[MPQueue] = None
core_queue: Optional[MPQueue] = None
failed_tickers_list: list = None
kill_set: list = None

# For prediction threads
predict_tasks: dict = None
predict_fails: list = None

############################################################################

class CoreDashboard(QMainWindow):
    timer: QTimer
    def __init__(self, status_dict, update_queue, core_queue, result_obj, kill_set): # noqa
        super().__init__()
        self.status_dict = status_dict
        self.update_queue = update_queue
        self.core_queue = core_queue
        self.result_obj = result_obj
        self.kill_set = kill_set

        self.setWindowTitle("Model Training Monitor")
        self.resize(800, 400)

        # Setup Table
        self.table = QTableWidget(len(status_dict) + 1, 6)
        self.table.setHorizontalHeaderLabels(["Core #", "Ticker/Interval", "Task", "Completed", "Kill buttons", "Resume buttons"])

        for i, core_key in enumerate(self.status_dict.keys()):
            # Extract the ID number from "Core #X"
            core_id = int(core_key.split("#")[-1])

            # Initialise cells
            self.table.setItem(i, 0, QTableWidgetItem(core_key))
            self.table.setItem(i, 1, QTableWidgetItem("None"))
            self.table.setItem(i, 2, QTableWidgetItem("None"))
            self.table.setItem(i, 3, QTableWidgetItem("0"))

            # Create stop buttons
            btn: QPushButton = QPushButton("X")
            btn.clicked.connect(lambda checked, c_id=core_id: self.stop_specific_core(c_id))
            self.table.setCellWidget(i, 4, btn)

            # Create resume buttons
            btn: QPushButton = QPushButton("✔")
            btn.clicked.connect(lambda checked, c_id=core_id: self.start_specific_core(c_id))
            btn.setEnabled(False)
            self.table.setCellWidget(i, 5, btn)

        # Total row
        self.table.setItem(len(status_dict), 0, QTableWidgetItem("TOTAL"))
        self.table.setItem(len(status_dict), 1, QTableWidgetItem("-"))
        self.table.setItem(len(status_dict), 2, QTableWidgetItem("-"))
        self.table.setItem(len(status_dict), 3, QTableWidgetItem("0"))

        stop_all_btn: QPushButton = QPushButton("X (all)")
        stop_all_btn.setStyleSheet("font-weight: bold;")
        stop_all_btn.clicked.connect(self.stop_all_cores)
        self.table.setCellWidget(len(status_dict), 4, stop_all_btn)

        resume_all_btn: QPushButton = QPushButton("✔ (all)")
        resume_all_btn.setStyleSheet("font-weight: bold;")
        resume_all_btn.clicked.connect(self.start_all_cores)
        self.table.setCellWidget(len(status_dict), 5, resume_all_btn)

        self.setCentralWidget(self.table)

        # Timer to refresh UI
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(200)  # 5fps

    def start_all_cores(self):
        # Remove every core ID from the kill set
        for core_key in self.status_dict.keys():
            core_id = int(core_key.split("#")[-1])
            if core_id in self.kill_set: self.kill_set.remove(core_id)
            if "TERMINATED" in self.status_dict[core_key]["Current Task"]: self.core_queue.put(core_id)

        # Disable/enable all buttons
        for row in range(0, len(self.status_dict) + 1):
            self.table.cellWidget(row, 4).setEnabled(True)
            self.table.cellWidget(row, 5).setEnabled(False)

    def start_specific_core(self, core_id):
        print(f"Starting core {core_id}")
        self.kill_set.remove(core_id)
        self.core_queue.put(core_id)

        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == f"Core #{core_id}":
                self.table.cellWidget(row, 4).setEnabled(True)
                self.table.cellWidget(row, 5).setEnabled(False)

        print("Finished resuming")

    def stop_all_cores(self):
        print("Stopping all cores")
        # Add every core ID to the kill set
        for core_key in self.status_dict.keys():
            core_id = int(core_key.split("#")[-1])
            if core_id not in self.kill_set: self.kill_set.append(core_id)

        # Disable all buttons
        for row in range(self.table.rowCount()):
            self.table.cellWidget(row, 4).setEnabled(False)
            self.table.cellWidget(row, 5).setEnabled(True)

        print("Stopped all cores")

    def stop_specific_core(self, core_id):
        print(f"Stopping core {core_id}")
        self.kill_set.append(core_id)
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == f"Core #{core_id}":
                self.table.cellWidget(row, 4).setEnabled(False)
                self.table.cellWidget(row, 5).setEnabled(True)

        print("Finished killing")

    def refresh_data(self):
        while not self.update_queue.empty():
            try:
                core_key, new_data = self.update_queue.get_nowait()

                if "Amount Completed" in new_data:
                    self.status_dict[core_key]["Amount Completed"] += new_data["Amount Completed"]

                self.status_dict[core_key].update({
                    k: v for k, v in new_data.items() if k != "Amount Completed"
                })
            except:
                break

        total_completed = 0
        for i, (core_key, info) in enumerate(self.status_dict.items()):
            self.table.item(i, 1).setText(str(info["Current ticker/interval"]))
            self.table.item(i, 2).setText(str(info["Current Task"]))
            self.table.item(i, 3).setText(str(info["Amount Completed"]))

            total_completed += info["Amount Completed"]

        self.table.item(len(self.status_dict), 3).setText(str(total_completed))

        if self.result_obj.ready():
            self.timer.stop()
            print("Training Complete. Closing Dashboard.")
            self.close()

############################################################################

def get_max_cores():
    import os
    try:
        # Linux/Modal specific
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # Fallback for Windows/Local testing
        return os.cpu_count() or 1

def initialise_worker(q, core_q, f_list, k_set):
    global update_queue, core_queue, failed_tickers_list, kill_set
    update_queue = q
    core_queue = core_q
    failed_tickers_list = f_list
    kill_set = k_set

    import torch
    if torch.cuda.is_available():
        torch.cuda.init()

def train_model(ticker):
    global update_queue, core_queue, failed_tickers_list, kill_set

    core_num = core_queue.get()
    core_key: str = f"Core #{core_num}"

    try:
        if core_num in kill_set:
            update_queue.put((core_key, {"Current ticker/interval": "None", "Current Task": "TERMINATED"}))
            return  # Exit the function, worker won't take more tasks

        from scripts.predictor import TrainingManager, all_ticker_models_exist, Settings
        Settings.GPU = False
        Settings.Threaded = True
        Settings.LOGGING = False

        for interval in ["1h", "1d"]:
            update_queue.put((core_key, {
                "Current ticker/interval": f"{ticker} ({interval})",
                "Current Task": "Initialising"
            }))

            full_data = load_data(ticker, interval)
            # cutoff_date = full_data.index.max() - pd.Timedelta(days=(60 if interval == "1d" else 20))
            cutoff_date = pd.to_datetime("2026-02-01")
            training_data = full_data[full_data.index < cutoff_date]

            if not all_ticker_models_exist(os.path.join(MODEL_DIR, f"{ticker}_{interval}"), interval):
                trainer = TrainingManager()
                success = trainer.run_training_pipeline(ticker, interval, override_data=training_data, status_signal=(update_queue, core_key))
                if not success:
                    failed_tickers_list.append((ticker, interval))

                update_queue.put((core_key, {
                    "Amount Completed": 1
                }))

            else:
                update_queue.put((core_key, {
                    "Current ticker/interval": f"{ticker} ({interval})",
                    "Current Task": "Skipped (Already Trained)"
                }))
                time.sleep(0.1)

            update_queue.put((core_key, {
                "Current ticker/interval": "None",
                "Current Task": "Waiting..."
            }))

    except Exception as e:
        update_queue.put((core_key, {
            "Current ticker/interval": "None",
            "Current Task": "CRASHED"
        }))
        print(f"Core {core_num} crashed on {ticker}:\n{type(e).__name__} - {e}")

    finally:
        # Ensure the core is returned or handled
        if core_num in kill_set:
            update_queue.put((core_key, {"Current Task": "STOPPED"}))
        else:
            core_queue.put(core_num)

def run_training(free_cores: int = 0):
    global update_queue, core_queue

    # with open(os.path.join(DATA_DIR, "ticker_map.json"), "r") as f:
    #     ticker_map = json.load(f)
    #     ticker_list = sorted(list(ticker_map.values()))[::-1]

    ticker_intervals = {'CRL_1h', 'AMGN_1d', 'HSIC_1d', 'SUI_1d', 'ICLR_1h', 'ISRG_1d', 'FAST_1d', 'AXTI_1d', 'OVV_1h', 'CTAS_1d', 'BAH_1d', 'KMX_1d', 'PEGA_1d', 'MRK_1d', 'MCHP_1d', 'RBC_1d', 'PCAR_1h', 'ESLT_1h', 'BEN_1h', 'DUOL_1d', 'TEVA_1d', 'SAN_1d', 'TRU_1d', 'PPG_1h', 'SCCO_1d', 'DE_1h', 'UHS_1d', 'CHTR_1d', 'FFIV_1d', 'MARA_1d', 'EFX_1d', 'AMD_1d', 'USFD_1d', 'INTC_1d', 'GWRE_1d', 'EXK_1d', 'WHR_1h', 'TCOM_1d', 'SWK_1d', 'KTOS_1d', 'WIX_1d', 'MTCH_1d', 'ALLE_1d', 'EOG_1d', 'PR_1d', 'GLW_1h', 'LFUS_1d', 'EPAM_1h', 'WDC_1h', 'EMR_1d'}
    ticker_list = sorted([t.split("_")[0] for t in ticker_intervals])

    num_cores = get_max_cores() - free_cores

    with Manager() as manager:
        shared_failed_list = manager.list()
        shared_kill_set = manager.list()
        update_queue = Queue()
        core_queue = Queue()

        status_dict = {}
        for core in range(1, num_cores + 1):
            core_queue.put(core)
            status_dict[f"Core #{core}"] = {
                "Current ticker/interval": "None",
                "Current Task": "None",
                "Amount Completed": 0,
            }

        app = QApplication(sys.argv)
        try:
            with Pool(processes=num_cores, initializer=initialise_worker, initargs=(update_queue, core_queue, shared_failed_list, shared_kill_set)) as pool:
                result = pool.starmap_async(train_model, [(t,) for t in ticker_list])

                dashboard = CoreDashboard(status_dict, update_queue, core_queue, result, shared_kill_set)
                dashboard.show()

                app.exec()
                result.get()

        except KeyboardInterrupt:
            print("\n[!] User interrupted training. Cleaning up...")

        finally:
            final_failed = list(shared_failed_list)
            print(f"\n--- Training Summary ---")
            if final_failed:
                print(f"Failed Tickers ({len(final_failed)}): {final_failed}")
            else:
                print("No failures recorded.")

############################################################################

def predict_model(ticker, interval):
    global predict_tasks, predict_fails
    task_key = f"{ticker}_{interval}"
    predict_tasks[task_key] = time.perf_counter()
    try:
        import torch
        from safetensors.torch import load_file
        from scripts.predictor import LSTMBrain, save_prediction, get_market_dates, all_ticker_models_exist, prediction_saved
        from lightgbm import Booster as LGBMBooster

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
        horizons = {1:1, 2:2, 4:4, 8:25} if "h" in interval else {1:1, 2:2, 5:7, 21:28}
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
        if isinstance(e, DetailedException): print(e)
        else: print(f"\nERROR on {ticker} ({interval})\n--->{type(e).__name__}: {e}")

        if predict_fails is not None:
            predict_fails.append(task_key)
    finally:
        predict_tasks[task_key] = -1

def initialise_predict_worker(u_q, f_l):
    global predict_tasks, predict_fails
    predict_tasks = u_q
    predict_fails = f_l

    import torch
    if torch.cuda.is_available():
        torch.cuda.init()

def run_predictions(free_cores: int = 0):
    global predict_tasks, predict_fails

    print("\n--- Predicting Stocks ---")
    num_cores = max(1, get_max_cores() - free_cores)
    print(f"-> Using {num_cores} CPU cores...")

    # with open(os.path.join(DATA_DIR, "ticker_map.json"), "r") as f:
    #     ticker_map = json.load(f)
    #     ticker_list = sorted(list(ticker_map.values()))

    ticker_list = sorted(list({'BAX', 'TECH', 'SN', 'TTD', 'TDG', 'TSCO', 'WDC', 'CYBR', 'TXT', 'TWLO', 'WEC', 'PCAR', 'TTEK', 'TRV', 'TXRH', 'ICLR', 'TXN', 'ESLT', 'TTMI', 'MMC', 'EGO', 'FRPT', 'TSLA', 'TDY', 'TSEM', 'USO', 'BBIO', 'TSN', 'TTE', 'TW', 'LASR', 'GLW', 'BAM', 'BAC', 'TD', 'TAP', 'UTHR', 'BALL', 'OVV', 'BABA', 'EXPD', 'EG', 'AZO', 'BAH', 'WHR', 'HUM', 'TEAM', 'DE', 'BAP', 'TT', 'TYL', 'CRL', 'T', 'B', 'AYI', 'EPAM', 'BA', 'AZN', 'GPC', 'TSM', 'TTWO', 'NLY', 'PPG', 'BEN'}))

    tasks = []
    for ticker in ticker_list:
        for interval in ["1h", "1d"]:
            tasks.append((ticker, interval))

    with Manager() as manager:
        predict_tasks = manager.dict()
        predict_fails = manager.list()

        try:
            with Pool(processes=num_cores, initializer=initialise_predict_worker, initargs=(predict_tasks, predict_fails)) as pool:

                result = pool.starmap_async(predict_model, tasks)

                pbar = tqdm(total=len(tasks), desc="Filling ledgers", unit="task", colour="cyan")

                while not result.ready():
                    # Check for updates from workers
                    current_statuses = dict(predict_tasks)
                    for task_name, status in current_statuses.items():
                        if status == -1:
                            # Worker finished
                            pbar.update(1)
                            del predict_tasks[task_name]
                        elif (time.perf_counter() - status) > 120:
                            # Timeout
                            print(f"\n[TIMEOUT] {task_name} exceeded 120s limit.")
                            predict_fails.append(task_name)
                            del predict_tasks[task_name]

                    time.sleep(1)  # Don't hammer the CPU

        except KeyboardInterrupt:
            print("\n[!] User interrupted predictions. Cleaning up processes...")

        finally:
            if 'pbar' in locals():
                pbar.close()

            print("\n--- Prediction Summary ---")
            if predict_fails:
                print(f"Failed ({len(predict_fails)}): {list(predict_fails)}")
            else:
                print("All predictions completed successfully.")

############################################################################

def updates(sent, spy, cache):
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

############################################################################

if __name__ == '__main__':

    import psutil
    import os

    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

    # import time_machine
    # target_time = datetime(2026, 3, 13, 12, 0, 0, tzinfo=timezone.utc)
    # with time_machine.travel(target_time):
        # function()

    # updates(  # Whether to update:
    #     sent=True,  # News sentiment
    #     spy=True,  # Market sentiment indicators
    #     cache=True,  # Stock cache
    # )

    run_training(
        free_cores=4 # How many CPU cores do you want left free
                     # Not necessary as there are stop buttons
    )

    # run_predictions(
    #     free_cores=1 # How many CPU cores do you want left free
    # )




