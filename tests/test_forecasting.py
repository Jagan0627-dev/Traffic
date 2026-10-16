import unittest

import numpy as np
import pandas as pd

from forecasting import (
    celsius_to_kelvin,
    make_prediction_features,
    train_and_evaluate,
)


def sample_dataset(rows=120):
    times = pd.date_range("2018-09-25", periods=rows, freq="h")
    frame = pd.DataFrame(
        {
            "date_time": times,
            "traffic_volume": 1000 + np.arange(rows) % 24 * 10,
            "temp": 288.0 + np.arange(rows) % 10,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 40,
            "weather_main": np.where(np.arange(rows) % 2, "Rain", "Clear"),
            "holiday": "None",
        }
    )
    return frame


class TrainAndEvaluateTests(unittest.TestCase):
    def test_celsius_input_converts_to_dataset_kelvin(self):
        self.assertAlmostEqual(celsius_to_kelvin(14.85), 288.0)

    def test_trains_and_evaluates_with_chronological_holdout(self):
        result = train_and_evaluate(sample_dataset())

        self.assertEqual(set(result.metrics), {"MAE", "MSE", "RMSE", "R²"})
        self.assertTrue(all(np.isfinite(value) for value in result.metrics.values()))
        self.assertEqual(len(result.evaluation), 24)
        self.assertGreaterEqual(
            result.evaluation.index.min(),
            pd.Timestamp("2018-09-29 00:00:00"),
        )
        self.assertEqual(result.rows_used, 120)
        self.assertEqual(result.duplicate_rows, 0)

    def test_drops_duplicates_and_invalid_targets(self):
        data = sample_dataset()
        duplicate = data.iloc[[0]]
        invalid = data.iloc[[1]].copy()
        invalid["date_time"] = pd.Timestamp("2018-10-01 00:00")
        data = pd.concat([data, duplicate, invalid], ignore_index=True)
        data.loc[len(data) - 1, "traffic_volume"] = np.nan

        result = train_and_evaluate(data)

        self.assertEqual(result.duplicate_rows, 1)
        self.assertEqual(result.discarded_rows, 1)
        self.assertEqual(result.rows_used, 120)

    def test_rejects_csv_without_required_dataset_columns(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            train_and_evaluate(pd.DataFrame({"date_time": ["2018-09-30"]}))

    def test_prediction_features_use_training_schema_and_validate_inputs(self):
        features = make_prediction_features(
            pd.Timestamp("2018-09-30"),
            hour=17,
            temperature=288,
            rainfall=0,
            snowfall=0,
            cloud_cover=40,
            weather="Rain",
            holiday="None",
        )

        self.assertEqual(list(features.columns), [
            "temp", "rain_1h", "snow_1h", "clouds_all", "hour", "day",
            "month", "weekday", "year", "day_of_year", "hour_sin", "hour_cos",
            "weekday_sin", "weekday_cos", "month_sin", "month_cos",
            "weather_main", "holiday",
        ])
        self.assertEqual(features.iloc[0]["hour"], 17)
        with self.assertRaisesRegex(ValueError, "Hour"):
            make_prediction_features(
                pd.Timestamp("2018-09-30"), 24, 288, 0, 0, 40, "Rain", "None"
            )


if __name__ == "__main__":
    unittest.main()
