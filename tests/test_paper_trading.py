import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.paper_trading import (
    HOLDING_COLUMNS,
    _session_config,
    latest_completed_nyse_session,
    rebuild_session_outputs,
    run_daily_paper_session,
    settle_ready_days,
    transition_portfolio,
)
from scripts.predictor import UniverseConfig


def prediction_rows(tickers):
    count = len(tickers)
    return pd.DataFrame({
        "Date": pd.Timestamp("2026-09-01"),
        "ticker": tickers,
        "profile_group": "ensemble",
        "pred": [0.10 + i * 0.01 for i in range(count)],
        "pred_raw": [0.10 + i * 0.01 for i in range(count)],
        "pred_cat": [0.10 + i * 0.01 for i in range(count)],
        "pred_lgbm": [0.10 + i * 0.01 for i in range(count)],
        "pred_signal": [0] * count,
        "ensemble_score": [0.50 + i * 0.01 for i in range(count)],
        "ensemble_agreement": [0] * count,
        "entry_eligible": [0] * count,
        "target_weight": [0.10] * count,
        "regime_exposure": [1.0] * count,
        "position_risk_scale": [1.0] * count,
        "rolling_vol": [0.30] * count,
        "rolling_beta": [1.0] * count,
    })


class PaperClockTests(unittest.TestCase):
    def test_completed_session_uses_close_grace_and_preopen_previous_day(self):
        before_open = latest_completed_nyse_session("2026-09-02 12:00:00+00:00")
        after_close = latest_completed_nyse_session("2026-09-02 20:20:00+00:00")

        self.assertEqual(before_open, pd.Timestamp("2026-09-01"))
        self.assertEqual(after_close, pd.Timestamp("2026-09-02"))


class PortfolioTransitionTests(unittest.TestCase):
    def test_full_portfolio_replaces_weakest_and_counts_two_legs(self):
        config = UniverseConfig(max_top_tickers=2, replace_buffer=0.002, horizon=40)
        latest = prediction_rows(["A", "B", "C", "D", "E"])
        latest[["pred", "pred_raw", "pred_cat", "pred_lgbm"]] = [
            [0.20, 0.20, 0.20, 0.20],
            [0.30, 0.30, 0.30, 0.30],
            [0.40, 0.40, 0.40, 0.40],
            [0.05, 0.05, 0.05, 0.05],
            [-0.10, -0.10, -0.10, -0.10],
        ]
        latest.loc[latest["ticker"].eq("C"), ["pred_signal", "ensemble_agreement", "entry_eligible"]] = 1
        latest.loc[latest["ticker"].eq("C"), "ensemble_score"] = 0.99
        prior = pd.DataFrame([
            {
                "ticker": "A", "side": 1, "entry_date": "2026-08-20", "age": 3,
                "entry_score": 0.10, "score": 0.10, "target_weight": 0.10,
            },
            {
                "ticker": "B", "side": 1, "entry_date": "2026-08-20", "age": 3,
                "entry_score": 0.80, "score": 0.80, "target_weight": 0.10,
            },
        ])

        result = transition_portfolio(
            latest, prior, config,
            signal_date=pd.Timestamp("2026-09-01"),
            expected_execution_open=pd.Timestamp("2026-09-02 14:30", tz="Europe/London"),
            equity_reference=1000.0,
        )

        self.assertEqual(set(result["holdings_state"]["ticker"]), {"B", "C"})
        self.assertEqual(result["counters"]["turnover"], 2)
        self.assertEqual(result["counters"]["replacements"], 1)
        actions = result["orders"].set_index("ticker")["action"].to_dict()
        self.assertEqual(actions["A"], "SELL_NEXT_OPEN")
        self.assertEqual(actions["C"], "BUY_NEXT_OPEN")
        self.assertEqual(actions["B"], "HOLD")

    def test_hold_weight_change_is_visible_but_not_a_backtest_turnover_leg(self):
        config = UniverseConfig(max_top_tickers=4)
        latest = prediction_rows(["A", "B", "C", "D", "E"])
        latest[["pred", "pred_raw", "pred_cat", "pred_lgbm"]] = [
            [0.20, 0.20, 0.20, 0.20],
            [0.30, 0.30, 0.30, 0.30],
            [0.40, 0.40, 0.40, 0.40],
            [0.05, 0.05, 0.05, 0.05],
            [-0.10, -0.10, -0.10, -0.10],
        ]
        latest.loc[latest["ticker"].eq("A"), "target_weight"] = 0.08
        prior = pd.DataFrame([{
            "ticker": "A", "side": 1, "entry_date": "2026-08-20", "age": 2,
            "entry_score": 0.5, "score": 0.5, "target_weight": 0.10,
        }])

        result = transition_portfolio(
            latest, prior, config,
            signal_date=pd.Timestamp("2026-09-01"),
            expected_execution_open=pd.Timestamp("2026-09-02 14:30", tz="Europe/London"),
            equity_reference=1000.0,
        )

        order = result["orders"].iloc[0]
        self.assertEqual(order["instruction"], "HOLD")
        self.assertEqual(order["action"], "REBALANCE_NEXT_OPEN")
        self.assertFalse(bool(order["costed_turnover_leg"]))
        self.assertAlmostEqual(float(order["target_value_gbp"]), 80.0)
        self.assertEqual(result["counters"]["turnover"], 0)


class PaperAccountingTests(unittest.TestCase):
    @staticmethod
    def frame(opens):
        dates = pd.to_datetime([date for date, _ in opens])
        values = [value for _, value in opens]
        return pd.DataFrame({
            "Open": values, "Close": values, "Adj Close": values,
        }, index=dates)

    def test_settlement_waits_for_second_open_and_matches_fixed_slot_math(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            day = session / "days" / "2026-01-02"
            day.mkdir(parents=True)
            portfolio = pd.DataFrame([{
                "ticker": "AAA", "side": 1, "entry_date": "2026-01-02", "age": 0,
                "entry_score": 0.9, "score": 0.9, "pred": 0.2,
                "pred_cat": 0.2, "pred_lgbm": 0.2, "target_weight": 0.25,
                "regime_exposure": 1.0, "position_risk_scale": 1.0,
                "rolling_vol": 0.3, "rolling_beta": 1.0,
            }], columns=HOLDING_COLUMNS)
            portfolio.to_csv(day / "portfolio.csv", index=False)
            portfolio.assign(age=1).to_csv(day / "holdings_state.csv", index=False)
            pd.DataFrame().to_csv(day / "orders.csv", index=False)
            (day / "manifest.json").write_text(json.dumps({
                "signal_date": "2026-01-02",
                "expected_execution_open": "2026-01-05T14:30:00+00:00",
                "model_name": "test",
                "model_action": "loaded",
                "diagnostics": {"Date": "2026-01-02", "final_candidates": 1},
                "training": None,
                "portfolio": {
                    "turnover": 1, "holdings": 1, "gross_exposure": 0.25,
                    "regime_exposure": 1.0, "held_tickers": "AAA",
                },
            }), encoding="utf-8")

            stock = self.frame([
                ("2026-01-02", 99.0), ("2026-01-05", 100.0), ("2026-01-06", 110.0),
            ])
            spy = self.frame([
                ("2026-01-02", 200.0), ("2026-01-05", 201.0), ("2026-01-06", 203.0),
            ])
            with patch("scripts.paper_trading.load_comparative_data", return_value=spy):
                not_ready = settle_ready_days(
                    session, {"AAA": stock.loc[:"2026-01-05"]}, interval="1d",
                    cost_bps=10.0, max_positions=4, as_of_date=pd.Timestamp("2026-01-05"),
                )
                settled = settle_ready_days(
                    session, {"AAA": stock}, interval="1d", cost_bps=10.0,
                    max_positions=4, as_of_date=pd.Timestamp("2026-01-06"),
                )

            self.assertEqual(not_ready, 0)
            self.assertEqual(settled, 1)
            metrics = json.loads(
                (session / "settlements" / "2026-01-02" / "metrics.json").read_text()
            )
            self.assertAlmostEqual(metrics["gross_return"], 0.025)
            self.assertAlmostEqual(metrics["turnover_cost"], 0.00025)
            self.assertAlmostEqual(metrics["daily_return"], 0.02475)

            config = {
                "session_name": "test", "interval": "1d", "pipeline_version": 4,
                "execution_clock": "test", "retrain_every_n_bars": 0,
                "initial_capital": 1000.0,
            }
            summary = rebuild_session_outputs(session, config)
            self.assertAlmostEqual(summary["final_equity"], 1024.75)

    def test_same_date_runner_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "same_day"
            session.mkdir()
            _session_config(
                session, session_name="same_day", interval="1d",
                initial_capital=1000.0, retrain_every_n_bars=0,
            )
            day = session / "days" / "2026-09-01"
            day.mkdir(parents=True)
            holdings = pd.DataFrame([{
                "ticker": "AAA", "side": 1, "entry_date": "2026-08-20", "age": 7,
                "entry_score": 0.8, "score": 0.8, "pred": 0.1,
                "pred_cat": 0.1, "pred_lgbm": 0.1, "target_weight": 0.03,
                "regime_exposure": 1.0, "position_risk_scale": 1.0,
                "rolling_vol": 0.3, "rolling_beta": 1.0,
            }], columns=HOLDING_COLUMNS)
            holdings.to_csv(day / "portfolio.csv", index=False)
            holdings.to_csv(day / "holdings_state.csv", index=False)
            orders = pd.DataFrame([{
                "Date": "2026-09-01", "expected_execution_open": "2026-09-02 14:30:00+01:00",
                "ticker": "AAA", "instruction": "HOLD", "action": "HOLD",
                "reason": "still_valid", "side": 1, "age": 6, "pred": 0.1,
                "pred_cat": 0.1, "pred_lgbm": 0.1, "score": 0.8,
                "previous_target_weight": 0.03, "target_weight": 0.03,
                "target_percent": 3.0, "target_value_gbp": 30.0,
                "weight_change": 0.0, "costed_turnover_leg": False,
            }])
            orders.to_csv(day / "orders.csv", index=False)
            (day / "manifest.json").write_text(json.dumps({
                "signal_date": "2026-09-01",
                "expected_execution_open": "2026-09-02T14:30:00+01:00",
                "model_name": "model-x", "model_action": "loaded",
                "diagnostics": {"Date": "2026-09-01", "final_candidates": 1},
                "training": None,
                "portfolio": {
                    "turnover": 0, "holdings": 1, "gross_exposure": 0.03,
                    "regime_exposure": 1.0, "held_tickers": "AAA",
                },
            }), encoding="utf-8")
            raw = self.frame([("2026-09-01", 100.0)])

            with (
                patch("scripts.predictor.DataManager.load_raw_data", return_value={"AAA": raw}),
                patch("scripts.paper_trading.settle_ready_days", return_value=0),
            ):
                result = run_daily_paper_session(
                    session_name="same_day", update_data=False, update_sentiment=False,
                    model_mode="never", now="2026-09-02 12:00:00+00:00",
                    session_root=root, strict_data=False, verbose=False,
                )

            self.assertEqual(result["status"], "already_generated")
            self.assertEqual(int(result["holdings"].iloc[0]["age"]), 7)
            self.assertEqual(len(list((session / "days").iterdir())), 1)

    def test_new_daily_cycle_commits_orders_diagnostics_and_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = self.frame([("2026-09-01", 100.0)])
            latest = prediction_rows(["A", "B", "C", "D", "E"])
            latest.loc[
                latest["ticker"].eq("C"),
                ["pred_signal", "ensemble_agreement", "entry_eligible"],
            ] = 1
            latest.loc[latest["ticker"].eq("C"), "ensemble_score"] = 0.99

            def fake_prediction(predictor, *_args, **_kwargs):
                predictor.as_of_date = pd.Timestamp("2026-09-01")
                predictor.feature_cols = ["feature"]
                return {
                    "latest": latest,
                    "picks": latest[latest["ticker"].eq("C")],
                    "diagnostics": {
                        "Date": pd.Timestamp("2026-09-01"),
                        "model_as_of_date": pd.Timestamp("2026-09-01"),
                        "universe_rows": 5,
                        "pred_cat_positive": 5,
                        "pred_lgbm_positive": 5,
                        "both_models_positive": 5,
                        "positive_agreement": 1,
                        "risk_eligible": 5,
                        "agreement_and_risk_eligible": 1,
                        "quantile_long_signals": 1,
                        "final_candidates": 1,
                        "regime_exposure": 1.0,
                        "median_pred_raw": 0.12,
                        "max_pred_raw": 0.14,
                    },
                }

            with (
                patch("scripts.predictor.DataManager.load_raw_data", return_value={"A": raw}),
                patch("scripts.paper_trading.settle_ready_days", return_value=0),
                patch("scripts.paper_trading._compatible_model", return_value=({"CAT": object(), "LGBM": object()}, "model-x")),
                patch("scripts.paper_trading.Predictor.predict_latest", new=fake_prediction),
            ):
                result = run_daily_paper_session(
                    session_name="new_cycle", update_data=False, update_sentiment=False,
                    now="2026-09-02 12:00:00+00:00", session_root=root,
                    strict_data=False, verbose=False,
                )

            session = root / "new_cycle"
            self.assertEqual(result["status"], "generated")
            self.assertEqual(result["orders"].iloc[0]["instruction"], "BUY")
            self.assertTrue((session / "days" / "2026-09-01" / "manifest.json").exists())
            self.assertTrue((session / "signal_diagnostics.csv").exists())
            self.assertTrue((session / "orders_history.csv").exists())
            self.assertTrue((session / "current_holdings.csv").exists())
            summary = json.loads((session / "summary.json").read_text())
            self.assertEqual(summary["signal_days"], 1)
            self.assertEqual(summary["pending_days"], 1)


if __name__ == "__main__":
    unittest.main()
