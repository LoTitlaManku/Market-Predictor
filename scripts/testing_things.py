
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

    for optional, default in (("gross_exposure", 0.0), ("regime_exposure", 1.0), ("gross_return", 0.0)):
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

    # Equity Curve
    axes[0].plot(df['Date'], df['equity'], label='Equity', color='tab:blue', linewidth=1.5)
    if 'benchmark_equity' in df.columns:
        axes[0].plot(df['Date'], df['benchmark_equity'], label='SPY Benchmark', color='tab:grey', linestyle='--', linewidth=1.0)
    axes[0].set_title('Portfolio Equity Curve')
    axes[0].set_ylabel('Equity (£)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')

    drawdown = df['equity'] / df['equity'].cummax() - 1.0
    drawdown_axis = axes[0].twinx()
    drawdown_axis.fill_between(df['Date'], drawdown * 100.0, 0, color='tab:red', alpha=0.12)
    drawdown_axis.set_ylabel('Drawdown (%)', color='tab:red')
    drawdown_axis.set_ylim(min(-1.0, float(drawdown.min() * 120.0)), 0.0)

    # Returns
    if 'daily_return' in df.columns:
        axes[1].plot(df['Date'], df['daily_return'], label='Daily Return', color='tab:green', alpha=0.7, linewidth=1)
    if 'gross_return' in df.columns:
        axes[1].plot(df['Date'], df['gross_return'], label='Gross Return', color='tab:orange', alpha=0.5, linewidth=1)
    if 'benchmark_return' in df.columns:
        axes[1].plot(df['Date'], df['benchmark_return'], label='SPY Return', color='tab:grey', alpha=0.5, linewidth=1)
    axes[1].set_title('Returns')
    axes[1].set_ylabel('Return')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper left')

    # Turnover & Holdings
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

    # Exposure explains whether low activity was a regime decision or a forecast/filter bottleneck.
    if 'gross_exposure' in df.columns:
        axes[3].plot(df['Date'], df['gross_exposure'], label='Gross Exposure', color='tab:blue', linewidth=1.2)
    if 'regime_exposure' in df.columns:
        axes[3].plot(df['Date'], df['regime_exposure'], label='Regime Limit', color='tab:orange', linestyle='--', linewidth=1.0)
    axes[3].set_title('Portfolio and Regime Exposure')
    axes[3].set_ylabel('Fraction')
    axes[3].set_xlabel('Date')
    axes[3].set_ylim(bottom=0)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper left')

    if 'holdings' in df.columns:
        for _, run in _zero_holding_runs(df).iterrows():
            for axis in axes:
                axis.axvspan(run['start'], run['end'] + pd.Timedelta(days=1), color='grey', alpha=0.08)

    plt.tight_layout()
    if save_path is not None: fig.savefig(save_path, dpi=160, bbox_inches='tight')
    if show: plt.show()
    return fig

########################################################################################################################

from dataclasses import dataclass

@dataclass
class RunSettings:
    task: str

    session_name: str
    initial_capital: float
    update_data: bool
    update_sentiment: bool
    model_mode: str
    retrain_every_n_bars: int | None
    vendor_grace_minutes: int
    strict_data: bool
    as_of: str | None

    report_months: int
    report_folder: str | None
    report_save_path: str | None
    report_show_plot: bool
    report_include_experiments: bool

if __name__ == "__main__":

    settings = RunSettings(
        task="paper",  # "paper" for daily signals, or "report" to inspect a backtest.

        # Paper trading
        session_name="paper_1d",
        initial_capital=1000.0,
        update_data=True,
        update_sentiment=True,
        model_mode="auto",  # "auto", "always", or "never"
        retrain_every_n_bars=None,  # None remembers the session setting; 0 is anchored.
        vendor_grace_minutes=15,
        strict_data=True,
        as_of=None,  # ISO timestamp used only to backfill a missed day.

        # Backtest report (only used when task="report")
        report_months=12,
        report_folder=None,
        report_save_path=None,
        report_show_plot=True,
        report_include_experiments=False,
    )

    if settings.task == "paper":
        from scripts.paper_trading import run_daily_paper_session

        run_daily_paper_session(
            session_name=settings.session_name,
            initial_capital=settings.initial_capital,
            update_data=settings.update_data,
            update_sentiment=settings.update_sentiment,
            model_mode=settings.model_mode,
            retrain_every_n_bars=settings.retrain_every_n_bars,
            vendor_grace_minutes=settings.vendor_grace_minutes,
            strict_data=settings.strict_data,
            now=settings.as_of,
            verbose=True,
        )

    elif settings.task == "report":
        def find_latest_walk_forward(months: int | float = 12, model_dir=None, include_experiments: bool = False):
            import json
            from pathlib import Path
            from scripts.config import MODEL_DIR

            root = Path(MODEL_DIR if model_dir is None else model_dir)
            candidates = []
            for path in root.glob(f"1d Model */walk_forward/{months}/summary.json"):
                if not (path.parent / "daily_equity.csv").exists():
                    continue
                metadata = json.loads(path.read_text(encoding="utf-8"))
                if metadata.get("experiment_note") and not include_experiments:
                    continue
                candidates.append(path.parent)
            if not candidates:
                raise FileNotFoundError(
                    f"No completed 1d walk-forward result for {months} months in {root}"
                )
            return max(candidates, key=lambda path: (path / "summary.json").stat().st_mtime)

        result_folder = settings.report_folder or find_latest_walk_forward(settings.report_months, settings.report_include_experiments)
        report = summarise_walk_forward(result_folder)

        print(f"Result: {report['folder']}")
        print(report["monthly"].to_string(index=False))
        plot_performance_metrics(
            str(report["folder"] / "daily_equity.csv"),
            show=settings.report_show_plot,
            save_path=settings.report_save_path,
        )
