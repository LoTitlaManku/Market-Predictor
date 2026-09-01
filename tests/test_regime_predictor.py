import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import data_management
from scripts.predictor import DataManager, Predictor, Trainer, UniverseConfig


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


if __name__ == "__main__":
    unittest.main()
