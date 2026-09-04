"""Stateful daily paper trading on the walk-forward execution clock.

The public entry point lives in :mod:`scripts.testing_things`.  This module
contains the state machine and accounting helpers so they can be tested
without downloading data or fitting the production models.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.config import MODEL_DIR
from scripts.data_management import NYSE_CAL, UpdateWorker, load_comparative_data
from scripts.predictor import (
    MODEL_PIPELINE_VERSION,
    Predictor,
    Trainer,
    UniverseConfig,
    flush_memory,
    json_safe,
)

ORDER_COLUMNS = [
    "Date", "expected_execution_open", "ticker", "instruction", "action",
    "reason", "side", "age", "pred", "pred_cat", "pred_lgbm", "score",
    "previous_target_weight", "target_weight", "target_percent",
    "target_value_gbp", "weight_change", "costed_turnover_leg",
]

HOLDING_COLUMNS = [
    "ticker", "side", "entry_date", "age", "entry_score", "score",
    "pred", "pred_cat", "pred_lgbm", "target_weight", "regime_exposure",
    "position_risk_scale", "rolling_vol", "rolling_beta",
]

POSITION_COLUMNS = [
    "signal_date", "exec_entry_date", "exec_exit_date", "ticker", "side",
    "entry_price", "exit_price", "position_return", "holding_age", "score",
    "weight", "regime_exposure", "position_risk_scale", "rolling_vol",
    "rolling_beta", "contribution_return", "contribution_pnl",
]

DAILY_COLUMNS = [
    "Date", "equity", "daily_return", "gross_return", "turnover_cost",
    "turnover", "holdings", "gross_exposure", "regime_exposure",
    "held_tickers", "benchmark_return", "benchmark_equity",
    "exec_entry_date", "exec_exit_date", "missing_position_returns",
]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def _as_utc(now: pd.Timestamp | str | None) -> pd.Timestamp:
    if now is None:
        return pd.Timestamp.now(tz="UTC")
    value = pd.Timestamp(now)
    if value.tzinfo is None:
        value = value.tz_localize("Europe/London")
    return value.tz_convert("UTC")


def latest_completed_nyse_session(
        now: pd.Timestamp | str | None = None, vendor_grace_minutes: int = 15,
) -> pd.Timestamp:
    """Return the last daily session whose close plus vendor grace has passed."""
    now_utc = _as_utc(now)
    schedule = NYSE_CAL.schedule(
        start_date=(now_utc - pd.Timedelta(days=14)).date(),
        end_date=now_utc.date(),
    )
    if schedule.empty:
        raise RuntimeError("Could not resolve an NYSE trading session")
    completed = schedule[
        schedule["market_close"] + pd.Timedelta(minutes=vendor_grace_minutes) <= now_utc
    ]
    if completed.empty:
        raise RuntimeError("No completed NYSE session is available yet")
    return pd.Timestamp(completed.index[-1]).tz_localize(None).normalize()


def _next_market_open(signal_date: pd.Timestamp) -> pd.Timestamp:
    schedule = NYSE_CAL.schedule(
        start_date=(pd.Timestamp(signal_date) + pd.Timedelta(days=1)).date(),
        end_date=(pd.Timestamp(signal_date) + pd.Timedelta(days=12)).date(),
    )
    if schedule.empty:
        raise RuntimeError(f"Could not resolve the next market open after {signal_date:%Y-%m-%d}")
    return pd.Timestamp(schedule.iloc[0]["market_open"]).tz_convert("Europe/London")


def _session_days(session_dir: Path) -> list[Path]:
    days_root = session_dir / "days"
    if not days_root.exists():
        return []
    return sorted(
        path for path in days_root.iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    )


def _latest_day_manifest(session_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    days = _session_days(session_dir)
    if not days:
        return None, None
    path = days[-1]
    return path, json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _session_config(
        session_dir: Path, *, session_name: str, interval: str,
        initial_capital: float, retrain_every_n_bars: int,
) -> dict[str, Any]:
    config_path = session_dir / "config.json"
    strategy_config = UniverseConfig()
    strategy_config.horizon = 40 if interval == "1d" else 30
    strategy_config.max_top_tickers = 30 if interval == "1d" else 10
    requested = {
        "session_name": session_name,
        "interval": interval,
        "initial_capital": float(initial_capital),
        "pipeline_version": MODEL_PIPELINE_VERSION,
        "retrain_every_n_bars": int(retrain_every_n_bars),
        "execution_clock": "signal close -> next adjusted open -> following adjusted open",
        "position_limit": 30 if interval == "1d" else 10,
        "strategy_config": asdict(strategy_config),
    }
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        for key in (
            "session_name", "interval", "initial_capital", "pipeline_version",
            "retrain_every_n_bars", "strategy_config",
        ):
            if existing.get(key) != requested.get(key):
                raise ValueError(
                    f"Paper session {session_name!r} was created with {key}="
                    f"{existing.get(key)!r}, not {requested.get(key)!r}. Use a new session name."
                )
        return existing
    requested["created_at"] = pd.Timestamp.now(tz="UTC")
    _atomic_json(config_path, requested)
    return requested


def _truncate_and_validate_data(
        data_dict: dict[str, pd.DataFrame], cutoff: pd.Timestamp,
        *, strict: bool, minimum_coverage: float = 0.80,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    truncated: dict[str, pd.DataFrame] = {}
    latest_by_ticker: dict[str, pd.Timestamp] = {}
    cutoff = pd.Timestamp(cutoff).normalize()

    for ticker, raw in data_dict.items():
        if raw is None or raw.empty:
            continue
        frame = raw.copy()
        frame.index = pd.to_datetime(frame.index, utc=True).tz_localize(None)
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        frame = frame[frame.index.normalize() <= cutoff]
        if frame.empty:
            continue
        truncated[ticker] = frame
        latest_by_ticker[ticker] = pd.Timestamp(frame.index.max()).normalize()

    if not truncated:
        raise ValueError("No stock data remained at the completed-session cutoff")

    latest_stock_date = max(latest_by_ticker.values())
    active_threshold = cutoff - pd.Timedelta(days=14)
    active = [ticker for ticker, date in latest_by_ticker.items() if date >= active_threshold]
    current = [ticker for ticker in active if latest_by_ticker[ticker] == cutoff]
    coverage = len(current) / max(1, len(active))
    report = {
        "cutoff_date": cutoff,
        "latest_stock_date": latest_stock_date,
        "loaded_tickers": len(truncated),
        "recently_active_tickers": len(active),
        "tickers_at_cutoff": len(current),
        "cutoff_coverage": coverage,
        "minimum_required_coverage": minimum_coverage,
    }

    if strict and latest_stock_date != cutoff:
        raise ValueError(
            f"Price update is incomplete: expected {cutoff:%Y-%m-%d}, newest cached bar is "
            f"{latest_stock_date:%Y-%m-%d}. No orders were generated."
        )
    if strict and coverage < minimum_coverage:
        raise ValueError(
            f"Only {coverage:.1%} of recently active tickers have a {cutoff:%Y-%m-%d} bar; "
            f"at least {minimum_coverage:.0%} is required. No orders were generated."
        )
    return truncated, report


def _read_prior_holdings(session_dir: Path) -> pd.DataFrame:
    day_path, _ = _latest_day_manifest(session_dir)
    if day_path is None:
        return pd.DataFrame(columns=HOLDING_COLUMNS)
    path = day_path / "holdings_state.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=HOLDING_COLUMNS)


def transition_portfolio(
        latest: pd.DataFrame, prior_holdings: pd.DataFrame,
        config: UniverseConfig, *, signal_date: pd.Timestamp,
        expected_execution_open: pd.Timestamp, equity_reference: float,
) -> dict[str, Any]:
    """Apply the same exit/entry/replacement state transition as Trainer."""
    latest = latest.copy()
    signal_date = pd.Timestamp(signal_date)
    manager = Predictor("1d", config).datamanager
    exit_rows = manager.add_signals(latest, allow_short=True).set_index("ticker", drop=False)
    latest_rows = latest.set_index("ticker", drop=False)
    candidates = latest[
        latest["pred_signal"].ne(0)
        & latest["ensemble_agreement"].eq(1)
        & latest["entry_eligible"].eq(1)
    ].copy()
    candidates["score"] = np.where(
        candidates["pred_signal"].eq(1),
        candidates["ensemble_score"],
        1.0 - candidates["ensemble_score"],
    )
    candidates = candidates.sort_values("score", ascending=False)

    active: dict[str, dict[str, Any]] = {}
    prior_weights: dict[str, float] = {}
    for row in prior_holdings.to_dict("records"):
        ticker = str(row.get("ticker"))
        if not ticker or ticker == "nan":
            continue
        active[ticker] = {
            "ticker": ticker,
            "side": int(row.get("side", 1)),
            "entry_date": row.get("entry_date"),
            "age": int(row.get("age", 0)),
            "entry_score": float(row.get("entry_score", row.get("score", 0.0))),
            "score": float(row.get("score", 0.0)),
        }
        prior_weights[ticker] = float(row.get("target_weight", 0.0))

    events: list[dict[str, Any]] = []
    counters = {
        "entries": 0, "replacements": 0, "exits_missing": 0,
        "exits_opposite": 0, "exits_weak": 0, "exits_age": 0,
    }

    def exit_event(ticker: str, reason: str, *, replacement: bool = False) -> None:
        info = active.pop(ticker)
        row = exit_rows.loc[ticker] if ticker in exit_rows.index else None
        events.append({
            "ticker": ticker,
            "instruction": "SELL" if info["side"] == 1 else "BUY",
            "action": "SELL_NEXT_OPEN" if info["side"] == 1 else "COVER_NEXT_OPEN",
            "reason": reason,
            "side": info["side"],
            "age": info["age"],
            "pred": float(row["pred"]) if row is not None else np.nan,
            "pred_cat": float(row["pred_cat"]) if row is not None else np.nan,
            "pred_lgbm": float(row["pred_lgbm"]) if row is not None else np.nan,
            "score": float(info["score"]),
            "previous_target_weight": prior_weights.get(ticker, 0.0),
            "target_weight": 0.0,
            "costed_turnover_leg": True,
            "replacement": replacement,
        })

    for ticker in list(active):
        if ticker not in exit_rows.index:
            counters["exits_missing"] += 1
            exit_event(ticker, "missing_from_latest_universe")
            continue
        row = exit_rows.loc[ticker]
        info = active[ticker]
        side = int(info["side"])
        opposite = int(row["pred_signal"]) == -side
        weak_long = side == 1 and float(row["pred"]) <= config.exit_pred_threshold
        weak_short = side == -1 and float(row["pred"]) >= -config.exit_pred_threshold
        too_old = int(info["age"]) >= config.horizon
        if opposite or weak_long or weak_short or too_old:
            reasons = []
            if opposite:
                reasons.append("opposite_signal")
                counters["exits_opposite"] += 1
            if weak_long or weak_short:
                reasons.append("weak_long" if weak_long else "weak_short")
                counters["exits_weak"] += 1
            if too_old:
                reasons.append("too_old")
                counters["exits_age"] += 1
            exit_event(ticker, ",".join(reasons))

    entered: set[str] = set()
    for _, row in candidates.iterrows():
        ticker = str(row["ticker"])
        side = int(row["pred_signal"])
        score = float(row["score"])
        if ticker in active:
            active[ticker]["score"] = score
            active[ticker]["side"] = side
            continue

        if len(active) >= config.max_top_tickers:
            weakest = min(active, key=lambda key: active[key]["score"])
            if score <= float(active[weakest]["score"]) + config.replace_buffer:
                continue
            exit_event(weakest, "replaced_by_stronger_candidate", replacement=True)
            counters["replacements"] += 1

        active[ticker] = {
            "ticker": ticker,
            "side": side,
            "entry_date": signal_date,
            "age": 0,
            "entry_score": score,
            "score": score,
        }
        entered.add(ticker)
        counters["entries"] += 1
        events.append({
            "ticker": ticker,
            "instruction": "BUY" if side == 1 else "SELL",
            "action": "BUY_NEXT_OPEN" if side == 1 else "SHORT_NEXT_OPEN",
            "reason": "new_signal" if counters["replacements"] == 0 else "new_or_replacement_signal",
            "side": side,
            "age": 0,
            "pred": float(row["pred"]),
            "pred_cat": float(row["pred_cat"]),
            "pred_lgbm": float(row["pred_lgbm"]),
            "score": score,
            "previous_target_weight": 0.0,
            "target_weight": float(row["target_weight"]),
            "costed_turnover_leg": True,
        })

    snapshots: list[dict[str, Any]] = []
    next_state: list[dict[str, Any]] = []
    for ticker in sorted(active):
        if ticker not in latest_rows.index:
            continue
        info = active[ticker]
        row = latest_rows.loc[ticker]
        target_weight = float(row["target_weight"])
        snapshot = {
            "ticker": ticker,
            "side": int(info["side"]),
            "entry_date": info["entry_date"],
            "age": int(info["age"]),
            "entry_score": float(info["entry_score"]),
            "score": float(info["score"]),
            "pred": float(row["pred"]),
            "pred_cat": float(row["pred_cat"]),
            "pred_lgbm": float(row["pred_lgbm"]),
            "target_weight": target_weight,
            "regime_exposure": float(row["regime_exposure"]),
            "position_risk_scale": float(row["position_risk_scale"]),
            "rolling_vol": float(row["rolling_vol"]),
            "rolling_beta": float(row["rolling_beta"]),
        }
        snapshots.append(snapshot)
        next_row = dict(snapshot)
        next_row["age"] = int(info["age"]) + 1
        next_state.append(next_row)

        if ticker not in entered:
            previous_weight = prior_weights.get(ticker, 0.0)
            change = target_weight - previous_weight
            events.append({
                "ticker": ticker,
                "instruction": "HOLD",
                "action": "REBALANCE_NEXT_OPEN" if abs(change) > 1e-8 else "HOLD",
                "reason": "target_weight_changed" if abs(change) > 1e-8 else "still_valid",
                "side": int(info["side"]),
                "age": int(info["age"]),
                "pred": float(row["pred"]),
                "pred_cat": float(row["pred_cat"]),
                "pred_lgbm": float(row["pred_lgbm"]),
                "score": float(info["score"]),
                "previous_target_weight": previous_weight,
                "target_weight": target_weight,
                # This deliberately matches Trainer: only entry/exit legs are
                # charged, even though target weights are recomputed daily.
                "costed_turnover_leg": False,
            })

    orders = pd.DataFrame(events)
    if orders.empty:
        orders = pd.DataFrame(columns=ORDER_COLUMNS)
    else:
        orders.insert(0, "Date", signal_date)
        orders.insert(1, "expected_execution_open", expected_execution_open)
        orders["target_percent"] = orders["target_weight"].astype(float) * 100.0
        orders["target_value_gbp"] = orders["target_weight"].astype(float) * equity_reference
        orders["weight_change"] = (
            orders["target_weight"].astype(float)
            - orders["previous_target_weight"].astype(float)
        )
        for column in ORDER_COLUMNS:
            if column not in orders.columns:
                orders[column] = np.nan
        orders = orders[ORDER_COLUMNS]
        priority = {"SELL": 0, "BUY": 1, "HOLD": 2}
        orders["_priority"] = orders["instruction"].map(priority).fillna(9)
        orders = orders.sort_values(["_priority", "score"], ascending=[True, False]).drop(columns="_priority")

    snapshot_df = pd.DataFrame(snapshots, columns=HOLDING_COLUMNS)
    state_df = pd.DataFrame(next_state, columns=HOLDING_COLUMNS)
    turnover = int(orders.get("costed_turnover_leg", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    counters["turnover"] = turnover
    counters["holdings"] = len(snapshot_df)
    counters["gross_exposure"] = float(snapshot_df["target_weight"].sum()) if not snapshot_df.empty else 0.0
    return {
        "orders": orders,
        "portfolio": snapshot_df,
        "holdings_state": state_df,
        "counters": counters,
        "candidate_count": len(candidates),
    }


def _adjusted_open(frame: pd.DataFrame) -> pd.Series | None:
    if "Adj Open" in frame.columns:
        return frame["Adj Open"].astype(float)
    if {"Open", "Close", "Adj Close"}.issubset(frame.columns):
        factor = frame["Adj Close"].astype(float) / frame["Close"].astype(float)
        return frame["Open"].astype(float) * factor
    if "Open" in frame.columns:
        return frame["Open"].astype(float)
    return None


def _execution_window(opens: pd.Series, signal_date: pd.Timestamp) -> tuple[Any, ...] | None:
    series = opens.dropna().copy()
    series.index = pd.to_datetime(series.index, utc=True).tz_localize(None)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    dates = series.index.to_numpy(dtype="datetime64[ns]")
    entry_index = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(signal_date)), side="right"))
    exit_index = entry_index + 1
    if exit_index >= len(series):
        return None
    return (
        series.index[entry_index], series.index[exit_index],
        float(series.iloc[entry_index]), float(series.iloc[exit_index]),
    )


def settle_ready_days(
        session_dir: Path, data_dict: dict[str, pd.DataFrame], *,
        interval: str, cost_bps: float, max_positions: int,
        as_of_date: pd.Timestamp | None = None,
) -> int:
    """Settle each immutable signal snapshot once its second future open exists."""
    benchmark = load_comparative_data("SPY", interval)
    if as_of_date is not None:
        benchmark_index = pd.to_datetime(benchmark.index, utc=True).tz_localize(None)
        benchmark = benchmark.loc[benchmark_index.normalize() <= pd.Timestamp(as_of_date).normalize()].copy()
    benchmark_open = _adjusted_open(benchmark)
    if benchmark_open is None:
        raise ValueError("SPY has no usable open price for paper accounting")

    settled = 0
    for day_dir in _session_days(session_dir):
        metrics_path = session_dir / "settlements" / day_dir.name / "metrics.json"
        if metrics_path.exists():
            continue
        signal_date = pd.Timestamp(day_dir.name)
        benchmark_window = _execution_window(benchmark_open, signal_date)
        if benchmark_window is None:
            break
        benchmark_entry_date, benchmark_exit_date, benchmark_entry, benchmark_exit = benchmark_window

        portfolio_path = day_dir / "portfolio.csv"
        portfolio = pd.read_csv(portfolio_path) if portfolio_path.exists() else pd.DataFrame(columns=HOLDING_COLUMNS)
        position_rows: list[dict[str, Any]] = []
        missing = 0
        for row in portfolio.to_dict("records"):
            ticker = str(row["ticker"])
            raw = data_dict.get(ticker)
            opens = _adjusted_open(raw) if raw is not None and not raw.empty else None
            window = _execution_window(opens, signal_date) if opens is not None else None
            if window is None:
                missing += 1
                continue
            entry_date, exit_date, entry_price, exit_price = window
            side = int(row.get("side", 1))
            position_return = (
                exit_price / entry_price - 1.0
                if side == 1 else entry_price / exit_price - 1.0
            )
            weight = float(row.get("target_weight", 0.0))
            position_rows.append({
                "signal_date": signal_date,
                "exec_entry_date": entry_date,
                "exec_exit_date": exit_date,
                "ticker": ticker,
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "position_return": position_return,
                "holding_age": int(row.get("age", 0)),
                "score": float(row.get("score", 0.0)),
                "weight": weight,
                "regime_exposure": float(row.get("regime_exposure", 0.0)),
                "position_risk_scale": float(row.get("position_risk_scale", 0.0)),
                "rolling_vol": float(row.get("rolling_vol", np.nan)),
                "rolling_beta": float(row.get("rolling_beta", np.nan)),
                "contribution_return": position_return * weight,
                "contribution_pnl": np.nan,
            })

        positions = pd.DataFrame(position_rows, columns=POSITION_COLUMNS)
        gross_return = float(positions["contribution_return"].sum()) if not positions.empty else 0.0
        manifest = json.loads((day_dir / "manifest.json").read_text(encoding="utf-8"))
        turnover = int(manifest["portfolio"]["turnover"])
        turnover_cost = float(cost_bps / 10_000.0 * turnover / max(1, max_positions))
        metrics = {
            "Date": signal_date,
            "gross_return": gross_return,
            "turnover_cost": turnover_cost,
            "daily_return": gross_return - turnover_cost,
            "turnover": turnover,
            "holdings": int(manifest["portfolio"]["holdings"]),
            "gross_exposure": float(positions["weight"].sum()) if not positions.empty else 0.0,
            "regime_exposure": float(manifest["portfolio"]["regime_exposure"]),
            "held_tickers": manifest["portfolio"]["held_tickers"],
            "benchmark_return": benchmark_exit / benchmark_entry - 1.0,
            "exec_entry_date": benchmark_entry_date,
            "exec_exit_date": benchmark_exit_date,
            "missing_position_returns": missing,
            "settled_at": pd.Timestamp.now(tz="UTC"),
        }
        settlement_dir = metrics_path.parent
        _atomic_csv(settlement_dir / "position_returns.csv", positions)
        _atomic_json(metrics_path, metrics)
        settled += 1
    return settled


def _read_json_rows(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def rebuild_session_outputs(session_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Rebuild all cumulative reports from completed immutable day records."""
    day_dirs = _session_days(session_dir)
    manifests = _read_json_rows([path / "manifest.json" for path in day_dirs])

    order_frames = [
        _read_csv(path / "orders.csv", ORDER_COLUMNS)
        for path in day_dirs if (path / "orders.csv").exists()
    ]
    orders = pd.concat(order_frames, ignore_index=True) if order_frames else pd.DataFrame(columns=ORDER_COLUMNS)
    _atomic_csv(session_dir / "orders_history.csv", orders)
    if day_dirs:
        _atomic_csv(
            session_dir / "latest_orders.csv",
            _read_csv(day_dirs[-1] / "orders.csv", ORDER_COLUMNS),
        )

    trades = orders[orders.get("costed_turnover_leg", False).astype(str).str.lower().eq("true")].copy() \
        if not orders.empty else pd.DataFrame(columns=ORDER_COLUMNS)
    _atomic_csv(session_dir / "trades.csv", trades)

    portfolio_frames = []
    diagnostic_rows = []
    retraining_rows = []
    snapshot_rows = []
    prediction_frames = []
    for day_dir, manifest in zip(day_dirs, manifests):
        portfolio_path = day_dir / "portfolio.csv"
        if portfolio_path.exists():
            portfolio = pd.read_csv(portfolio_path)
            portfolio.insert(0, "Date", pd.Timestamp(day_dir.name))
            portfolio_frames.append(portfolio)
        diagnostic_rows.append(manifest["diagnostics"])
        snapshot_rows.append({"Date": manifest["signal_date"], **manifest["portfolio"]})
        if manifest.get("training"):
            retraining_rows.append(manifest["training"])
        prediction_path = session_dir / "predictions" / f"all_predictions_{config['interval']}_{day_dir.name}.parquet"
        if prediction_path.exists():
            prediction_frames.append(pd.read_parquet(prediction_path))

    holdings_history = pd.concat(portfolio_frames, ignore_index=True) \
        if portfolio_frames else pd.DataFrame(columns=["Date", *HOLDING_COLUMNS])
    _atomic_csv(session_dir / "holdings_history.csv", holdings_history)
    diagnostics = pd.DataFrame(diagnostic_rows)
    _atomic_csv(session_dir / "signal_diagnostics.csv", diagnostics)
    snapshots = pd.DataFrame(snapshot_rows)
    _atomic_csv(session_dir / "snapshot_days.csv", snapshots)
    retraining = pd.DataFrame(retraining_rows)
    _atomic_csv(session_dir / "retraining_folds.csv", retraining)

    if day_dirs:
        current_holdings = pd.read_csv(day_dirs[-1] / "holdings_state.csv")
    else:
        current_holdings = pd.DataFrame(columns=HOLDING_COLUMNS)
    _atomic_csv(session_dir / "current_holdings.csv", current_holdings)

    if prediction_frames:
        prediction_columns = [
            "Date", "ticker", "model_as_of_date", "model_name",
            "pred", "pred_raw", "pred_cat", "pred_lgbm",
            "ensemble_score", "ensemble_agreement", "risk_eligible",
            "entry_eligible", "rolling_vol", "rolling_beta", "vol_rank_pct",
            "target_weight", "regime_exposure", "position_risk_scale",
        ]
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions = predictions[[column for column in prediction_columns if column in predictions.columns]]
        predictions.to_parquet(session_dir / "paper_predictions.parquet", index=False)

    settlement_paths = sorted((session_dir / "settlements").glob("*/metrics.json")) \
        if (session_dir / "settlements").exists() else []
    settlement_rows = _read_json_rows(settlement_paths)
    equity = float(config["initial_capital"])
    benchmark_equity = float(config["initial_capital"])
    daily_rows = []
    equity_before: dict[str, float] = {}
    for row in settlement_rows:
        date = pd.Timestamp(row["Date"])
        equity_before[date.strftime("%Y-%m-%d")] = equity
        equity *= 1.0 + float(row["daily_return"])
        benchmark_equity *= 1.0 + float(row["benchmark_return"])
        daily_rows.append({
            **row,
            "Date": date,
            "equity": equity,
            "benchmark_equity": benchmark_equity,
        })
    daily = pd.DataFrame(daily_rows)
    for column in DAILY_COLUMNS:
        if column not in daily.columns:
            daily[column] = pd.Series(dtype=float if column not in {"Date", "held_tickers", "exec_entry_date", "exec_exit_date"} else object)
    daily = daily[DAILY_COLUMNS]
    _atomic_csv(session_dir / "daily_equity.csv", daily)

    position_frames = []
    for path in settlement_paths:
        position_path = path.parent / "position_returns.csv"
        if not position_path.exists():
            continue
        positions = pd.read_csv(position_path)
        if positions.empty:
            continue
        positions["signal_date"] = pd.to_datetime(positions["signal_date"])
        positions["contribution_pnl"] = positions.apply(
            lambda row: equity_before.get(pd.Timestamp(row["signal_date"]).strftime("%Y-%m-%d"), 0.0)
            * float(row["contribution_return"]), axis=1,
        )
        position_frames.append(positions)
    positions = pd.concat(position_frames, ignore_index=True) \
        if position_frames else pd.DataFrame(columns=POSITION_COLUMNS)
    _atomic_csv(session_dir / "position_returns.csv", positions)

    initial = float(config["initial_capital"])
    if daily.empty:
        returns = pd.Series(dtype=float)
        benchmark_returns = pd.Series(dtype=float)
        max_drawdown = benchmark_max_drawdown = 0.0
        sharpe = information_ratio = 0.0
        end_date = None
        start_date = None
        cagr_like = 0.0
    else:
        returns = daily["daily_return"].astype(float)
        benchmark_returns = daily["benchmark_return"].astype(float)
        max_drawdown = float((daily["equity"] / daily["equity"].cummax() - 1.0).min())
        benchmark_max_drawdown = float(
            (daily["benchmark_equity"] / daily["benchmark_equity"].cummax() - 1.0).min()
        )
        sharpe = float(returns.mean() / (returns.std() + 1e-9) * np.sqrt(252))
        active = returns - benchmark_returns
        information_ratio = float(active.mean() / (active.std() + 1e-9) * np.sqrt(252))
        end_date = pd.Timestamp(daily["Date"].max())
        start_date = pd.Timestamp(daily["Date"].min())
        years = len(daily) / 252.0
        cagr_like = float((equity / initial) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    zero_holdings = daily["holdings"].eq(0) if not daily.empty else pd.Series(dtype=bool)
    zero_groups = zero_holdings.ne(zero_holdings.shift()).cumsum() if not daily.empty else pd.Series(dtype=int)
    longest_zero = int(zero_holdings.groupby(zero_groups).sum().max()) if zero_holdings.any() else 0
    current_model = manifests[-1].get("model_name") if manifests else None
    summary = {
        "session_name": config["session_name"],
        "interval": config["interval"],
        "pipeline_version": config["pipeline_version"],
        "execution_clock": config["execution_clock"],
        "model_type": "ENSEMBLE",
        "target_column": config.get("strategy_config", {}).get("target_column"),
        "feature_count": manifests[-1].get("feature_count") if manifests else None,
        "retrain_every_n_bars": config["retrain_every_n_bars"],
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial,
        "final_equity": float(equity),
        "profit_loss": float(equity - initial),
        "total_return": float(equity / initial - 1.0),
        "benchmark_total_return": float(benchmark_equity / initial - 1.0),
        "active_total_return": float((equity - benchmark_equity) / initial),
        "cagr_like": cagr_like,
        "sharpe_like": sharpe,
        "information_ratio_like": information_ratio,
        "max_drawdown": max_drawdown,
        "benchmark_max_drawdown": benchmark_max_drawdown,
        "win_rate": float((returns > 0).mean()) if not returns.empty else 0.0,
        "mean_daily_return": float(returns.mean()) if not returns.empty else 0.0,
        "median_daily_return": float(returns.median()) if not returns.empty else 0.0,
        "signal_days": len(day_dirs),
        "settled_days": len(daily),
        "pending_days": len(day_dirs) - len(daily),
        "performance_through_signal_date": end_date,
        "latest_signal_date": pd.Timestamp(day_dirs[-1].name) if day_dirs else None,
        "current_model": current_model,
        "model_fit_count": len(retraining),
        "current_holdings": len(current_holdings),
        "avg_holdings": float(daily["holdings"].mean()) if not daily.empty else 0.0,
        "avg_gross_exposure": float(daily["gross_exposure"].mean()) if not daily.empty else 0.0,
        "avg_regime_exposure": float(daily["regime_exposure"].mean()) if not daily.empty else 0.0,
        "avg_turnover": float(daily["turnover"].mean()) if not daily.empty else 0.0,
        "max_daily_turnover": int(daily["turnover"].max()) if not daily.empty else 0,
        "zero_holding_days": int(zero_holdings.sum()) if not daily.empty else 0,
        "longest_zero_holding_streak": longest_zero,
        "no_candidate_days": int(diagnostics["final_candidates"].eq(0).sum())
        if not diagnostics.empty and "final_candidates" in diagnostics else 0,
        "avg_daily_candidates": float(diagnostics["final_candidates"].mean())
        if not diagnostics.empty and "final_candidates" in diagnostics else 0.0,
        "trade_count": len(trades),
    }

    attribution = Trainer.analyse_pos_attr(positions)
    for name, key in (
        ("ticker_attribution.csv", "ticker_summary"),
        ("best_position_days.csv", "best_position_days"),
        ("worst_position_days.csv", "worst_position_days"),
    ):
        frame = attribution[key]
        _atomic_csv(session_dir / name, frame)
    _atomic_json(session_dir / "attr_summary.json", attribution["concentration"])
    summary["attribution_concentration"] = attribution["concentration"]
    _atomic_json(session_dir / "summary.json", summary)

    state = {
        "session_name": config["session_name"],
        "latest_signal_date": summary["latest_signal_date"],
        "performance_through_signal_date": end_date,
        "current_model": current_model,
        "current_holdings": len(current_holdings),
        "settled_equity": equity,
        "pending_days": summary["pending_days"],
        "updated_at": pd.Timestamp.now(tz="UTC"),
    }
    _atomic_json(session_dir / "state.json", state)
    return summary


def _compatible_model(
        predictor: Predictor, cutoff: pd.Timestamp, preferred: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    root = Path(MODEL_DIR) / "models"
    if not root.exists():
        return None, None
    # Once a session has selected a model, never jump to a model trained by a
    # different paper session merely because it is newer in the global store.
    names = [preferred] if preferred else [
        path.name for path in sorted(root.iterdir(), reverse=True) if path.is_dir()
    ]
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        folder = root / name
        metadata_path = folder / "metadata.json"
        required = [folder / "features.joblib", folder / "cat_model.joblib", folder / "lgbm_model.joblib"]
        if not metadata_path.exists() or not all(path.exists() for path in required):
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("pipeline_version") != MODEL_PIPELINE_VERSION:
                continue
            if metadata.get("interval") != predictor.interval:
                continue
            if metadata.get("target_column") != predictor.config.target_column:
                continue
            stored_strategy = metadata.get("config", {})
            training_keys = (
                "horizon", "target_column", "target_clip", "max_training_years",
                "recency_half_life_days", "max_abs_bar_return",
                "min_rolling_dollar_volume_1d", "rolling_vol_window_1d",
                "rolling_beta_window_1d",
            )
            if any(
                stored_strategy.get(key) != getattr(predictor.config, key)
                for key in training_keys
            ):
                continue
            as_of = pd.Timestamp(metadata["as_of_date"])
            age_limit = predictor.config.max_loaded_model_age_days_1d \
                if predictor.interval == "1d" else predictor.config.max_loaded_model_age_days_1h
            if as_of > cutoff or cutoff - as_of > pd.Timedelta(days=age_limit):
                continue
            return predictor.load_models(name), name
        except (ValueError, OSError, KeyError, json.JSONDecodeError):
            continue
    return None, None


def _missing_signal_sessions(last_signal: pd.Timestamp, cutoff: pd.Timestamp) -> list[pd.Timestamp]:
    schedule = NYSE_CAL.schedule(
        start_date=(pd.Timestamp(last_signal) + pd.Timedelta(days=1)).date(),
        end_date=pd.Timestamp(cutoff).date(),
    )
    return [pd.Timestamp(value).tz_localize(None).normalize() for value in schedule.index]


def run_daily_paper_session(
        session_name: str = "paper_1d", initial_capital: float = 1000.0, *,
        interval: str = "1d", update_data: bool = True,
        update_sentiment: bool = True, model_mode: str = "auto",
        retrain_every_n_bars: int | None = None, vendor_grace_minutes: int = 15,
        strict_data: bool = True, now: pd.Timestamp | str | None = None,
        session_root: str | Path | None = None, verbose: bool = True,
) -> dict[str, Any]:
    """Update data, generate next-open instructions, and journal a paper day."""
    if interval != "1d":
        raise ValueError("The daily paper runner currently supports interval='1d' only")
    if model_mode not in {"auto", "always", "never"}:
        raise ValueError("model_mode must be 'auto', 'always', or 'never'")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", session_name):
        raise ValueError("session_name may contain only letters, numbers, '.', '_' and '-'")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    root = Path(session_root) if session_root is not None else Path(MODEL_DIR) / "paper_runs"
    session_dir = root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    existing_config_path = session_dir / "config.json"
    if retrain_every_n_bars is None:
        if existing_config_path.exists():
            existing_config = json.loads(existing_config_path.read_text(encoding="utf-8"))
            retrain_every_n_bars = int(existing_config.get("retrain_every_n_bars", 0))
        else:
            retrain_every_n_bars = 0
    if retrain_every_n_bars < 0:
        raise ValueError("retrain_every_n_bars cannot be negative")
    stored_config = _session_config(
        session_dir, session_name=session_name, interval=interval,
        initial_capital=initial_capital,
        retrain_every_n_bars=retrain_every_n_bars,
    )

    now_utc = _as_utc(now)
    cutoff = latest_completed_nyse_session(now_utc, vendor_grace_minutes)
    next_open_for_cutoff = _next_market_open(cutoff).tz_convert("UTC")
    if now_utc >= next_open_for_cutoff:
        raise RuntimeError(
            "The next open for the latest completed signal has already passed. Run this command "
            "before the NYSE opens, or after today's close plus the vendor grace period."
        )
    update_errors = []
    if update_data:
        updater = UpdateWorker()
        for label, operation in (
            ("comparatives", updater.update_comparatives),
            ("prices", updater.data_updater),
            ("sentiment", updater.sentiment_update if update_sentiment else None),
        ):
            if operation is None:
                continue
            try:
                if verbose:
                    print(f"Updating {label}...")
                operation()
            except Exception as exc:
                update_errors.append({"component": label, "error": f"{type(exc).__name__}: {exc}"})
                if strict_data or label != "sentiment":
                    _atomic_json(session_dir / "latest_data_status.json", {
                        "cutoff_date": cutoff, "update_errors": update_errors,
                    })
                    raise

    model_config = UniverseConfig()
    model_config.horizon = 40
    model_config.max_top_tickers = 30
    predictor = Predictor(interval, model_config)
    raw_data = predictor.datamanager.load_raw_data(interval)
    raw_data, data_report = _truncate_and_validate_data(raw_data, cutoff, strict=strict_data)
    data_report["update_requested"] = update_data
    data_report["update_sentiment_requested"] = update_sentiment
    data_report["update_errors"] = update_errors
    data_report["completed_session"] = cutoff
    _atomic_json(session_dir / "latest_data_status.json", data_report)

    settled_now = settle_ready_days(
        session_dir, raw_data, interval=interval, cost_bps=model_config.cost_bps,
        max_positions=model_config.max_top_tickers, as_of_date=cutoff,
    )
    summary_before = rebuild_session_outputs(session_dir, stored_config)

    latest_day, latest_manifest = _latest_day_manifest(session_dir)
    if latest_day is not None:
        last_signal = pd.Timestamp(latest_day.name)
        if last_signal > cutoff:
            raise ValueError(
                f"Session already contains {last_signal:%Y-%m-%d}, later than the completed-data cutoff "
                f"{cutoff:%Y-%m-%d}."
            )
        if last_signal == cutoff:
            orders = pd.read_csv(latest_day / "orders.csv")
            result = {
                "status": "already_generated",
                "signal_date": cutoff,
                "expected_execution_open": latest_manifest["expected_execution_open"],
                "model_action": "unchanged_same_date",
                "model_name": latest_manifest["model_name"],
                "orders": orders,
                "holdings": pd.read_csv(latest_day / "holdings_state.csv"),
                "summary": summary_before,
                "settled_now": settled_now,
                "session_dir": session_dir,
            }
            if verbose:
                _print_paper_result(result)
            return result
        missing = _missing_signal_sessions(last_signal, cutoff)
        if len(missing) > 1:
            dates = ", ".join(date.strftime("%Y-%m-%d") for date in missing[:-1])
            raise RuntimeError(
                f"Paper session missed completed trading day(s): {dates}. The runner stopped instead of "
                "silently changing holding ages and decisions. Backfill the earliest date first with the "
                "function's now= argument or the CLI --as-of ISO timestamp, then run normally again."
            )

    preferred_model = latest_manifest.get("model_name") if latest_manifest else None
    signals_since_fit = 0
    for day in reversed(_session_days(session_dir)):
        manifest = json.loads((day / "manifest.json").read_text(encoding="utf-8"))
        signals_since_fit += 1
        if str(manifest.get("model_action", "")).startswith("trained"):
            break
    cadence_due = retrain_every_n_bars > 0 and signals_since_fit >= retrain_every_n_bars

    models = None
    model_name = None
    prediction_data = None
    model_action = "loaded"
    training_record = None
    if model_mode != "always" and not cadence_due:
        models, model_name = _compatible_model(predictor, cutoff, preferred_model)
    if models is None:
        if model_mode == "never":
            raise FileNotFoundError(
                "No compatible, fresh model is available for this session. Run with model_mode='auto' "
                "to create one."
            )
        universe = predictor.datamanager.build_universe(interval, raw_data, drop_unlabelled=False)
        if pd.Timestamp(universe["Date"].max()).normalize() != cutoff:
            raise ValueError(
                f"Prepared universe ends {pd.Timestamp(universe['Date'].max()):%Y-%m-%d}; "
                f"expected completed session {cutoff:%Y-%m-%d}."
            )
        predictor._prepare_data(universe, train=True)
        prediction_data = universe
        models = {}
        models["LGBM"] = predictor.train_model("LGBM")
        flush_memory()
        models["CAT"] = predictor.train_model("CAT")
        flush_memory()
        model_folder = predictor.save_models(models)
        model_name = model_folder.name
        if model_mode == "always":
            model_action = "trained_always"
        elif cadence_due:
            model_action = "trained_cadence"
        elif preferred_model:
            model_action = "trained_stale_or_incompatible"
        else:
            model_action = "trained_initial"
        training_record = {
            "fold": int(sum(
                str(json.loads((day / 'manifest.json').read_text()).get('model_action', '')).startswith('trained')
                for day in _session_days(session_dir)
            )),
            "model_as_of_date": predictor.as_of_date,
            "prediction_start": cutoff,
            "prediction_end": cutoff,
            "prediction_bars": 1,
            "training_rows": len(predictor.train_df),
            "latest_training_feature_date": pd.to_datetime(predictor.train_df["Date"]).max(),
            "latest_label_end_date": pd.to_datetime(predictor.train_df["target_end_date"]).max(),
            "model_name": model_name,
            "reason": model_action,
        }

    details = predictor.predict_latest(
        raw_data, models, prediction_data,
        prediction_dir=session_dir / "predictions",
        update_legacy_paper_ledger=False,
        return_details=True,
    )
    latest = details["latest"]
    signal_date = pd.Timestamp(latest["Date"].max()).normalize()
    if signal_date != cutoff:
        raise ValueError(
            f"Prediction universe ends {signal_date:%Y-%m-%d}; expected {cutoff:%Y-%m-%d}. "
            "No portfolio state was changed."
        )
    expected_open = _next_market_open(signal_date)
    transition = transition_portfolio(
        latest, _read_prior_holdings(session_dir), model_config,
        signal_date=signal_date, expected_execution_open=expected_open,
        equity_reference=float(summary_before["final_equity"]),
    )

    diagnostics = dict(details["diagnostics"])
    diagnostics.update(transition["counters"])
    diagnostics.update({
        "Date": signal_date,
        "model_as_of_date": predictor.as_of_date,
        "model_name": model_name,
        "model_action": model_action,
        "final_candidates": transition["candidate_count"],
    })
    portfolio = transition["portfolio"]
    regime_exposure = float(latest["regime_exposure"].median())
    manifest = {
        "signal_date": signal_date,
        "expected_execution_open": expected_open,
        "model_name": model_name,
        "model_as_of_date": predictor.as_of_date,
        "model_action": model_action,
        "pipeline_version": MODEL_PIPELINE_VERSION,
        "feature_count": len(predictor.feature_cols),
        "diagnostics": diagnostics,
        "portfolio": {
            "turnover": transition["counters"]["turnover"],
            "holdings": len(portfolio),
            "gross_exposure": float(portfolio["target_weight"].sum()) if not portfolio.empty else 0.0,
            "regime_exposure": regime_exposure,
            "held_tickers": ",".join(sorted(portfolio["ticker"].astype(str))) if not portfolio.empty else "",
        },
        "training": training_record,
        "data_status": data_report,
        "created_at": pd.Timestamp.now(tz="UTC"),
    }
    day_dir = session_dir / "days" / signal_date.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(day_dir / "orders.csv", transition["orders"])
    _atomic_csv(day_dir / "portfolio.csv", portfolio)
    _atomic_csv(day_dir / "holdings_state.csv", transition["holdings_state"])
    _atomic_json(day_dir / "diagnostics.json", diagnostics)
    # The manifest is the commit marker and is intentionally written last.
    _atomic_json(day_dir / "manifest.json", manifest)

    summary = rebuild_session_outputs(session_dir, stored_config)
    result = {
        "status": "generated",
        "signal_date": signal_date,
        "expected_execution_open": expected_open,
        "model_action": model_action,
        "model_name": model_name,
        "orders": transition["orders"],
        "holdings": transition["holdings_state"],
        "diagnostics": diagnostics,
        "summary": summary,
        "settled_now": settled_now,
        "session_dir": session_dir,
    }
    if verbose:
        _print_paper_result(result)
    return result


def _print_paper_result(result: dict[str, Any]) -> None:
    summary = result["summary"]
    print(
        f"\nPaper session: {Path(result['session_dir']).name} | signal "
        f"{pd.Timestamp(result['signal_date']):%Y-%m-%d} | {result['status']}"
    )
    print(
        f"Model: {result['model_name']} ({result['model_action']}); next execution: "
        f"{pd.Timestamp(result['expected_execution_open'])}"
    )
    print(
        f"Settled equity: £{summary['final_equity']:,.2f} through "
        f"{summary['performance_through_signal_date'] or 'no settled signal yet'}; "
        f"{summary['pending_days']} signal day(s) awaiting future opens."
    )
    orders = result["orders"]
    if orders.empty:
        print("No positions or next-open orders.")
        return
    display_columns = [
        "instruction", "ticker", "action", "reason", "target_percent",
        "target_value_gbp", "pred",
    ]
    display = orders[display_columns].copy()
    display["target_percent"] = display["target_percent"].map(lambda value: f"{float(value):.2f}%")
    display["target_value_gbp"] = display["target_value_gbp"].map(lambda value: f"£{float(value):.2f}")
    display["pred"] = pd.to_numeric(display["pred"], errors="coerce").map(
        lambda value: "" if pd.isna(value) else f"{value:.4f}"
    )
    print("\nNext-open instructions:")
    print(display.to_string(index=False))
