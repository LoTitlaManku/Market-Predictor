import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import data_management
from scripts.predictor import DataManager, Predictor, Trainer, UniverseConfig
from scripts.testing_things import find_latest_walk_forward, summarise_walk_forward


class ComparativeDataTests(unittest.TestCase):
    def tearDown(self) -> None:
        data_management.clear_comparative_data_cache()

    def test_loader_merges_clean_history_and_ignores_1970_index_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = pd.DataFrame(
                {"Adj Close": [100.0, 101.0]},
                index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
            )
            parquet.to_parquet(root / "SPY_1d.parquet")

            csv = pd.DataFrame(
                {"Adj Close": [1.0, 102.0]},
                index=["1970-01-01 00:00:00.000000001", "2020-01-06"],
            )
            csv.to_csv(root / "SPY_1d.csv")

            with patch.object(data_management, "DATA_DIR", str(root)):
                data_management.clear_comparative_data_cache()
                loaded = data_management.load_comparative_data("SPY", "1d")

            self.assertEqual(loaded.index.min(), pd.Timestamp("2020-01-02"))
            self.assertEqual(loaded.index.max(), pd.Timestamp("2020-01-06"))
            self.assertEqual(float(loaded.loc["2020-01-06", "Adj Close"]), 102.0)
            self.assertTrue(loaded.index.is_unique)
            self.assertTrue(loaded.index.is_monotonic_increasing)

    def test_stale_comparative_data_fails_fast(self):
        fresh = pd.DataFrame(
            {"Adj Close": [100.0]}, index=pd.to_datetime(["2026-08-31"])
        )
        stale = pd.DataFrame(
            {"Adj Close": [20.0]}, index=pd.to_datetime(["2026-06-17"])
        )

        def comparative(key: str, _interval: str) -> pd.DataFrame:
            return stale if key == "VIX" else fresh

        manager = DataManager(UniverseConfig(comparative_max_staleness_days_1d=5))
        with patch("scripts.predictor.load_comparative_data", side_effect=comparative):
            with self.assertRaisesRegex(ValueError, "Stale VIX_1d data"):
                manager.validate_comparative_freshness("1d", pd.Timestamp("2026-08-31"))


class PredictionLogicTests(unittest.TestCase):
    @staticmethod
    def prediction_frame(pred_cat: np.ndarray, pred_lgbm: np.ndarray) -> pd.DataFrame:
        count = len(pred_cat)
        return pd.DataFrame({
            "Date": pd.Timestamp("2026-01-05"),
            "ticker": [f"T{i}" for i in range(count)],
            "pred_cat": pred_cat,
            "pred_lgbm": pred_lgbm,
            "rolling_vol": np.linspace(0.2, 0.5, count),
            "rolling_beta": np.linspace(0.5, 1.0, count),
            "Regime_Exposure": 1.0,
        })

    def test_ensemble_preserves_raw_forecasts_and_separates_rank_score(self):
        config = UniverseConfig(
            ensemble_top_n=2,
            ensemble_cat_weight=0.25,
            max_entry_rolling_vol=2.0,
            max_entry_abs_beta=5.0,
            max_entry_vol_percentile=1.0,
        )
        predictor = Predictor("1d", config)
        source = self.prediction_frame(
            np.array([-0.4, -0.2, 0.1, 0.2, 0.4, 0.6]),
            np.array([-0.5, -0.1, 0.05, 0.3, 0.5, 0.7]),
        )
        original = source.copy(deep=True)

        result = predictor.add_ensemble_predictions(source)

        pd.testing.assert_frame_equal(source, original)
        np.testing.assert_allclose(result["pred_cat"], original["pred_cat"])
        np.testing.assert_allclose(result["pred_lgbm"], original["pred_lgbm"])
        expected = 0.25 * original["pred_cat"] + 0.75 * original["pred_lgbm"]
        np.testing.assert_allclose(result["pred_raw"], expected)
        np.testing.assert_allclose(result["pred"], expected)
        self.assertFalse(np.allclose(result["ensemble_score"], result["pred_raw"]))
        self.assertNotEqual(float(result.loc[2, "pred_raw"]), 0.0)

    def test_negative_forecasts_do_not_create_long_entries(self):
        config = UniverseConfig(
            ensemble_top_n=3,
            max_entry_rolling_vol=2.0,
            max_entry_abs_beta=5.0,
            max_entry_vol_percentile=1.0,
        )
        predictor = Predictor("1d", config)
        values = -np.linspace(0.5, 0.01, 10)
        result = predictor.add_ensemble_predictions(
            self.prediction_frame(values, values - 0.01)
        )
        result = predictor.datamanager.add_signals(result)
        long_entries = result[
            (result["pred_signal"] == 1)
            & (result["ensemble_agreement"] == 1)
            & (result["entry_eligible"] == 1)
        ]
        self.assertTrue(long_entries.empty)

    def test_entry_risk_caps_reject_high_volatility_and_beta(self):
        config = UniverseConfig(
            ensemble_top_n=4,
            max_entry_rolling_vol=0.8,
            max_entry_abs_beta=2.0,
            max_entry_vol_percentile=1.0,
        )
        predictor = Predictor("1d", config)
        frame = self.prediction_frame(
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.1, 0.2, 0.3, 0.4]),
        )
        frame["rolling_vol"] = [0.4, 0.8, 0.81, 0.4]
        frame["rolling_beta"] = [1.0, 2.0, 1.0, 2.01]

        result = predictor.add_ensemble_predictions(frame)

        self.assertEqual(result["entry_eligible"].tolist(), [1, 1, 0, 0])

    def test_date_balanced_recency_weights_have_three_year_half_life(self):
        predictor = Predictor("1d", UniverseConfig(recency_half_life_days=1095.0))
        reference = pd.Timestamp("2026-01-01")
        dates = pd.Series([
            reference,
            reference,
            reference - pd.Timedelta(days=1095),
        ])
        weights = predictor.make_sample_weights(dates, reference)
        recent_total = float(weights[:2].sum())
        old_total = float(weights[2])
        self.assertAlmostEqual(old_total / recent_total, 0.5, places=6)

    def test_fixed_slots_and_regime_scaling_leave_cash(self):
        config = UniverseConfig(max_top_tickers=4)
        manager = DataManager(config)
        frame = pd.DataFrame({
            "Date": pd.Timestamp("2026-01-05"),
            "rolling_vol": [0.4] * 4,
            "Regime_Exposure": [0.5] * 4,
        })
        weighted = manager.add_portfolio_weights(frame)
        self.assertTrue(np.allclose(weighted["target_weight"], 0.125))

        gross = Trainer.calculate_gross_return(
            [0.10, 0.10], weighted["target_weight"].iloc[:2].tolist()
        )
        self.assertAlmostEqual(gross, 0.025)

    def test_target_end_date_uses_actual_future_bar(self):
        index = pd.to_datetime([
            "2026-01-02", "2026-01-05", "2026-01-08", "2026-01-09"
        ])
        frame = pd.DataFrame({
            "Adj Close": [100.0, 101.0, 102.0, 103.0],
            "rolling_beta": 1.0,
            "rolling_vol": 0.2,
        }, index=index)
        benchmark = pd.Series([100.0, 100.5, 101.0, 101.5], index=index)
        manager = DataManager(UniverseConfig(horizon=2))

        result = manager.add_forward_excess_target(
            frame, benchmark, "1d", drop_unlabelled=False
        )

        self.assertEqual(result.loc[index[0], "target_end_date"], index[2])
        self.assertEqual(result.loc[index[1], "target_end_date"], index[3])
        self.assertTrue(pd.isna(result.loc[index[2], "target_end_date"]))

    def test_training_frame_purges_labels_unavailable_as_of_date(self):
        predictor = Predictor("1d", UniverseConfig(max_training_years=0))
        predictor.feature_cols = ["feature"]
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "target_end_date": pd.to_datetime(["2026-01-04", "2026-01-08", "2026-01-05"]),
            "target_risk_adjusted_return": [0.1, 0.2, 0.3],
            "feature": [1.0, 2.0, 3.0],
        })

        predictor.set_training_frame(frame, pd.Timestamp("2026-01-05"))

        self.assertEqual(predictor.train_df["feature"].tolist(), [1.0, 3.0])
        self.assertLessEqual(
            pd.to_datetime(predictor.train_df["target_end_date"]).max(),
            pd.Timestamp("2026-01-05"),
        )

    def test_retraining_folds_are_strict_and_cover_each_date_once(self):
        dates = list(pd.bdate_range("2026-01-05", periods=8))
        folds = Trainer.make_retraining_folds(
            dates, pd.Timestamp("2026-01-02"), retrain_every_n_bars=3
        )

        flattened = [date for fold in folds for date in fold["prediction_dates"]]
        self.assertEqual(flattened, dates)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertLess(fold["model_as_of_date"], min(fold["prediction_dates"]))

    def test_unexecutable_tail_is_removed_before_model_fits(self):
        dates = pd.bdate_range("2026-01-05", periods=6)
        frame = pd.DataFrame({"Date": dates, "ticker": "T"})

        result = Trainer.trim_unexecutable_tail(frame, required_future_bars=2)

        self.assertEqual(result["Date"].tolist(), list(dates[:4]))

    def test_periodic_prediction_refits_and_never_uses_future_labels(self):
        class DummyModel:
            def __init__(self, value):
                self.value = value

            def predict(self, features):
                return np.full(len(features), self.value, dtype=float)

        config = UniverseConfig(
            max_training_years=0,
            retrain_every_n_bars_1d=2,
        )
        trainer = Trainer("1d", config)
        trainer.predictor.feature_cols = ["feature"]
        dates = pd.bdate_range("2026-01-01", periods=9)
        trainer.prepared_data = pd.DataFrame({
            "Date": dates,
            "target_end_date": dates + pd.to_timedelta([1, 1, 1, 5, 5, 1, 1, 1, 1], unit="D"),
            "target_risk_adjusted_return": np.linspace(0.1, 0.9, len(dates)),
            "feature": np.arange(len(dates), dtype=float),
            "ticker": "T",
        })
        cutoff = dates[3]
        walk = trainer.prepared_data[trainer.prepared_data["Date"] > cutoff].copy()
        dummy_models = [DummyModel(value) for value in range(6)]

        with (
            patch.object(trainer.predictor, "train_model", side_effect=dummy_models) as train,
            patch("scripts.predictor.log"),
        ):
            result = trainer.predict_with_retraining(walk, cutoff)

        self.assertEqual(train.call_count, 6)
        self.assertEqual(result["Date"].tolist(), list(dates[4:]))
        self.assertEqual(result["retrain_fold"].nunique(), 3)
        for record in trainer.retraining_records:
            self.assertLessEqual(
                pd.Timestamp(record["latest_label_end_date"]),
                pd.Timestamp(record["model_as_of_date"]),
            )

    def test_benchmark_uses_same_next_open_execution_clock(self):
        benchmark = pd.DataFrame({
            "Adj Open": [100.0, 101.0, 103.0, 104.0],
            "Adj Close": [100.0, 101.0, 103.0, 104.0],
        }, index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"]))
        daily = pd.DataFrame({
            "Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        })

        with patch("scripts.predictor.load_comparative_data", return_value=benchmark):
            result = Trainer.add_benchmark_performance(daily, "1d")

        self.assertAlmostEqual(result.loc[0, "benchmark_return"], 103.0 / 101.0 - 1.0)
        self.assertAlmostEqual(result.loc[1, "benchmark_return"], 104.0 / 103.0 - 1.0)


class WalkForwardReportingTests(unittest.TestCase):
    def test_latest_report_skips_labelled_experiments_by_default(self):
        import os

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stable = root / "1d Model [stable]" / "walk_forward" / "3"
            experiment = root / "1d Model [experiment]" / "walk_forward" / "3"
            stable.mkdir(parents=True)
            experiment.mkdir(parents=True)
            (stable / "summary.json").write_text("{}", encoding="utf-8")
            (experiment / "summary.json").write_text(
                '{"experiment_note": "diagnostic"}', encoding="utf-8"
            )
            (stable / "daily_equity.csv").write_text("Date,equity\n", encoding="utf-8")
            (experiment / "daily_equity.csv").write_text("Date,equity\n", encoding="utf-8")
            os.utime(stable / "summary.json", (1, 1))
            os.utime(experiment / "summary.json", (2, 2))

            self.assertEqual(find_latest_walk_forward("1d", 3, root), stable)
            self.assertEqual(
                find_latest_walk_forward("1d", 3, root, include_experiments=True),
                experiment,
            )

    def test_monthly_report_and_contiguous_zero_holding_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            daily = pd.DataFrame({
                "Date": pd.to_datetime([
                    "2026-07-30", "2026-07-31", "2026-08-03", "2026-08-04", "2026-08-05"
                ]),
                "equity": [1000.0, 1010.0, 1010.0, 1010.0, 1020.0],
                "daily_return": [0.0, 0.01, 0.0, 0.0, 0.00990099],
                "gross_return": [0.0, 0.01, 0.0, 0.0, 0.00990099],
                "turnover": [1, 2, 0, 0, 1],
                "holdings": [1, 2, 0, 0, 1],
                "gross_exposure": [0.1, 0.2, 0.0, 0.0, 0.1],
                "regime_exposure": [1.0] * 5,
            })
            daily.to_csv(folder / "daily_equity.csv", index=False)
            (folder / "summary.json").write_text("{}", encoding="utf-8")

            report = summarise_walk_forward(folder)

            august = report["monthly"].loc[
                report["monthly"]["month"].astype(str).eq("2026-08")
            ].iloc[0]
            self.assertEqual(int(august["zero_holding_days"]), 2)
            self.assertEqual(len(report["zero_holding_runs"]), 1)
            self.assertEqual(int(report["zero_holding_runs"].iloc[0]["bars"]), 2)


if __name__ == "__main__":
    unittest.main()
