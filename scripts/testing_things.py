
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
    from scripts.data_management import UpdateWorker
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

def repair_all_cache_csvs():
    import os
    import pandas as pd
    from tqdm import tqdm
    from scripts.config import CACHE_DIR

    files = [f for f in os.listdir(CACHE_DIR) if f.endswith("_1d.csv")]

    for filename in tqdm(files, desc="Cleaning cache"):
        file_path = os.path.join(CACHE_DIR, filename)
        df = pd.read_csv(file_path, index_col=0)

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df['not_null_count'] = df.notnull().sum(axis=1)
        df = df.sort_values(by=['not_null_count'], ascending=False)

        df = df[~df.index.duplicated(keep='first')]
        df = df.drop(columns=['not_null_count'])
        df.to_csv(file_path)


########################################################################################################################

def find_latest_walk_forward(
        interval: str = "1d", months: int | float = 12, model_dir=None,
        include_experiments: bool = False,
):
    """Locate the newest completed walk-forward folder without hardcoded dates."""
    import json
    from pathlib import Path
    from scripts.config import MODEL_DIR

    root = Path(MODEL_DIR if model_dir is None else model_dir)
    candidates = []
    for path in root.glob(f"{interval} Model */walk_forward/{months}/summary.json"):
        if not (path.parent / "daily_equity.csv").exists():
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("experiment_note") and not include_experiments:
            continue
        candidates.append(path.parent)
    if not candidates:
        raise FileNotFoundError(
            f"No completed {interval} walk-forward result for {months} months in {root}"
        )
    return max(candidates, key=lambda path: (path / "summary.json").stat().st_mtime)


def _zero_holding_runs(df):
    import pandas as pd

    zero = df["holdings"].eq(0)
    groups = zero.ne(zero.shift()).cumsum()
    runs = []
    for _, group in df[zero].groupby(groups[zero]):
        runs.append({
            "start": pd.Timestamp(group["Date"].min()),
            "end": pd.Timestamp(group["Date"].max()),
            "bars": int(len(group)),
        })
    return pd.DataFrame(runs, columns=["start", "end", "bars"])


def summarise_walk_forward(folder):
    """Build a read-only monthly and signal-starvation report for one run."""
    import json
    from pathlib import Path
    import pandas as pd

    folder = Path(folder)
    daily_path = folder / "daily_equity.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"Missing walk-forward equity file: {daily_path}")

    daily = pd.read_csv(daily_path)
    required = {"Date", "equity", "daily_return", "holdings", "turnover"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily_equity.csv is missing required columns: {sorted(missing)}")
    daily["Date"] = pd.to_datetime(daily["Date"], errors="raise")
    daily = daily.sort_values("Date").reset_index(drop=True)

    for optional, default in (
        ("gross_exposure", 0.0), ("regime_exposure", 1.0), ("gross_return", 0.0),
    ):
        if optional not in daily.columns:
            daily[optional] = default

    monthly_aggregations = {
        "end_equity": ("equity", "last"),
        "return_pct": ("daily_return", lambda values: ((1.0 + values).prod() - 1.0) * 100.0),
        "avg_holdings": ("holdings", "mean"),
        "avg_gross_exposure": ("gross_exposure", "mean"),
        "avg_regime_exposure": ("regime_exposure", "mean"),
        "total_turnover": ("turnover", "sum"),
        "max_turnover": ("turnover", "max"),
        "zero_holding_days": ("holdings", lambda values: int(values.eq(0).sum())),
        "bars": ("holdings", "size"),
    }
    if "benchmark_return" in daily.columns:
        monthly_aggregations["benchmark_return_pct"] = (
            "benchmark_return", lambda values: ((1.0 + values).prod() - 1.0) * 100.0
        )

    monthly = (
        daily.assign(month=daily["Date"].dt.to_period("M"))
        .groupby("month", observed=True)
        .agg(**monthly_aggregations)
        .reset_index()
    )

    signal_path = folder / "signal_diagnostics.csv"
    signals = pd.read_csv(signal_path, parse_dates=["Date", "model_as_of_date"]) \
        if signal_path.exists() else pd.DataFrame()

    trades_path = folder / "trades.csv"
    trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
    exit_reasons = {}
    if not trades.empty and "action" in trades.columns:
        exits = trades[trades["action"].astype(str).str.startswith("EXIT")]
        for reason in ("weak_long", "weak_short", "opposite_signal", "too_old"):
            if reason in exits.columns:
                values = exits[reason].astype(str).str.lower().eq("true")
                exit_reasons[reason] = int(values.sum())
        exit_reasons["missing"] = int(exits["action"].eq("EXIT_MISSING").sum())

    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {
        "folder": folder,
        "summary": summary,
        "daily": daily,
        "monthly": monthly,
        "zero_holding_runs": _zero_holding_runs(daily),
        "signals": signals,
        "trades": trades,
        "exit_reasons": exit_reasons,
    }


def plot_performance_metrics(csv_path: str, show: bool = True, save_path=None):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)

    # 1. Equity Curve
    axes[0].plot(df['Date'], df['equity'], label='Equity', color='tab:blue', linewidth=1.5)
    if 'benchmark_equity' in df.columns:
        axes[0].plot(df['Date'], df['benchmark_equity'], label='SPY Benchmark',
                     color='tab:grey', linestyle='--', linewidth=1.0)
    axes[0].set_title('Portfolio Equity Curve')
    axes[0].set_ylabel('Equity (£)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')

    drawdown = df['equity'] / df['equity'].cummax() - 1.0
    drawdown_axis = axes[0].twinx()
    drawdown_axis.fill_between(df['Date'], drawdown * 100.0, 0, color='tab:red', alpha=0.12)
    drawdown_axis.set_ylabel('Drawdown (%)', color='tab:red')
    drawdown_axis.set_ylim(min(-1.0, float(drawdown.min() * 120.0)), 0.0)

    # 2. Returns
    if 'daily_return' in df.columns:
        axes[1].plot(df['Date'], df['daily_return'], label='Daily Return', color='tab:green', alpha=0.7, linewidth=1)
    if 'gross_return' in df.columns:
        axes[1].plot(df['Date'], df['gross_return'], label='Gross Return', color='tab:orange', alpha=0.5, linewidth=1)
    if 'benchmark_return' in df.columns:
        axes[1].plot(df['Date'], df['benchmark_return'], label='SPY Return',
                     color='tab:grey', alpha=0.5, linewidth=1)
    axes[1].set_title('Returns')
    axes[1].set_ylabel('Return')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper left')

    # 3. Turnover & Holdings
    ax3 = axes[2]
    if 'turnover' in df.columns:
        ax3.plot(df['Date'], df['turnover'], label='Turnover', color='tab:purple', alpha=0.8, linewidth=1)
        ax3.set_ylabel('Turnover', color='tab:purple')

    if 'holdings' in df.columns:
        ax3_twin = ax3.twinx()
        ax3_twin.plot(df['Date'], df['holdings'], label='Holdings Count', color='tab:red', alpha=0.6, linewidth=1)
        ax3_twin.set_ylabel('Holdings Count', color='tab:red')
        ax3_twin.grid(False)

    axes[2].set_title('Turnover and Holdings')
    axes[2].grid(True, alpha=0.3)

    # 4. Exposure explains whether low activity was a regime decision or a
    # forecast/filter bottleneck.
    if 'gross_exposure' in df.columns:
        axes[3].plot(df['Date'], df['gross_exposure'], label='Gross Exposure',
                     color='tab:blue', linewidth=1.2)
    if 'regime_exposure' in df.columns:
        axes[3].plot(df['Date'], df['regime_exposure'], label='Regime Limit',
                     color='tab:orange', linestyle='--', linewidth=1.0)
    axes[3].set_title('Portfolio and Regime Exposure')
    axes[3].set_ylabel('Fraction')
    axes[3].set_xlabel('Date')
    axes[3].set_ylim(bottom=0)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper left')

    if 'holdings' in df.columns:
        for _, run in _zero_holding_runs(df).iterrows():
            for axis in axes:
                axis.axvspan(run['start'], run['end'] + pd.Timedelta(days=1),
                             color='grey', alpha=0.08)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    return fig

########################################################################################################################

if __name__ == "__main__":
    import time
    start = time.perf_counter()

    # import argparse
    # parser = argparse.ArgumentParser(description="Inspect a completed walk-forward run")
    # parser.add_argument("--interval", default="1d")
    # parser.add_argument("--months", type=int, default=12)
    # parser.add_argument("--folder", default=None)
    # parser.add_argument("--save", default=None, help="Optional image output path")
    # parser.add_argument("--no-show", action="store_true")
    # parser.add_argument("--include-experiments", action="store_true")
    # args = parser.parse_args()

    # import time_machine
    # from datetime import datetime, timezone
    # target_time = datetime(2026, 6, 10, 15, 0, 0, tzinfo=timezone.utc)
    # with time_machine.travel(target_time):
    #     f()

    # updates(  # Whether to update:
    #     sent=True,  # News sentiment
    #     spy=True,  # Market sentiment indicators
    #     cache=True,  # Stock cache
    # )
    # find_latest()

    # from predictor import Predictor
    # print("Training...")
    # mng = Predictor("1d")
    # mng.run_pipeline("latest")

    # from folder_trees import generate_tree
    # generate_tree("/home/god/Projects/market_predictor", ignore_paths=[".bin", ".venv", "cache_files", "imgs"])

    # result_folder = args.folder or find_latest_walk_forward(
    #     args.interval, args.months, include_experiments=args.include_experiments
    # )
    # report = summarise_walk_forward(result_folder)
    # print(f"Result: {report['folder']}")
    # print(report["monthly"].to_string(index=False))
    # if not report["zero_holding_runs"].empty:
    #     print("\nZero-holding runs:")
    #     print(report["zero_holding_runs"].to_string(index=False))
    # if report["exit_reasons"]:
    #     print(f"\nExit reasons: {report['exit_reasons']}")
    # if not report["signals"].empty:
    #     print("\nLatest signal funnel:")
    #     print(report["signals"].tail(10).to_string(index=False))

    # plot_performance_metrics(
    #     str(report["folder"] / "daily_equity.csv"),
    #     show=not args.no_show,
    #     save_path=args.save,
    # )

    plot_performance_metrics(
        "/home/god/Projects/market_predictor/model/1d Model [2026-09-01 20:11]/walk_forward/12/daily_equity.csv",
        show=True,
    )

    print(time.perf_counter() - start)
    pass


    # import folder_trees
    # folder_trees.generate_tree("C:/Users/adlan_3zfnjq7/Desktop/Alex - Main/Projects/LoTi-Log", ignore_paths=[".briefcase"])
