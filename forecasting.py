"""Training and feature preparation for Metro Interstate traffic-volume estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REQUIRED_COLUMNS = {
    "date_time",
    "traffic_volume",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "holiday",
}
NUMERIC_FEATURES = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "hour",
    "day",
    "month",
    "weekday",
    "year",
    "day_of_year",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
]
CATEGORICAL_FEATURES = ["weather_main", "holiday"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def celsius_to_kelvin(temperature_celsius: float) -> float:
    """Convert the GUI's Celsius input to the Kelvin unit used by the dataset."""
    return float(temperature_celsius) + 273.15


@dataclass
class TrainingResult:
    """Trained full-data model and chronological holdout evaluation."""

    model: Pipeline
    metrics: dict[str, float]
    evaluation: pd.DataFrame
    rows_used: int
    discarded_rows: int
    duplicate_rows: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp


def _features_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build consistent numeric calendar and categorical input features."""
    date_time = pd.to_datetime(frame["date_time"], errors="coerce")
    features = frame[
        ["temp", "rain_1h", "snow_1h", "clouds_all", "weather_main", "holiday"]
    ].copy()

    features["hour"] = date_time.dt.hour
    features["day"] = date_time.dt.day
    features["month"] = date_time.dt.month
    features["weekday"] = date_time.dt.dayofweek
    features["year"] = date_time.dt.year
    features["day_of_year"] = date_time.dt.dayofyear
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
    features["weekday_sin"] = np.sin(2 * np.pi * features["weekday"] / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * features["weekday"] / 7)
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype(object)
        features.loc[features[column].isna(), column] = np.nan
    return features[FEATURE_COLUMNS]


def _build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=150,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_evaluate(data: pd.DataFrame) -> TrainingResult:
    """Clean the Metro CSV, evaluate chronologically, then fit a prediction model."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise ValueError(
            "This does not look like the Metro Interstate dataset. Missing required "
            f"columns: {', '.join(missing_columns)}."
        )

    original_rows = len(data)
    clean = data.drop_duplicates().copy()
    duplicate_rows = original_rows - len(clean)
    clean["date_time"] = pd.to_datetime(clean["date_time"], errors="coerce")
    clean["traffic_volume"] = pd.to_numeric(clean["traffic_volume"], errors="coerce")
    for column in ["temp", "rain_1h", "snow_1h", "clouds_all"]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    valid_target = (
        clean["date_time"].notna()
        & clean["traffic_volume"].notna()
        & np.isfinite(clean["traffic_volume"])
    )
    discarded_rows = int((~valid_target).sum())
    clean = clean.loc[valid_target].sort_values("date_time").reset_index(drop=True)
    if len(clean) < 50:
        raise ValueError(
            "At least 50 valid records are needed for a reliable chronological "
            "train/test evaluation. Please upload the complete dataset."
        )

    features = _features_from_frame(clean)
    target = clean["traffic_volume"].astype(float)
    split_index = int(len(clean) * 0.8)
    if split_index < 2 or len(clean) - split_index < 2:
        raise ValueError("The dataset is too small to create a train/test split.")

    evaluation_model = _build_pipeline()
    evaluation_model.fit(features.iloc[:split_index], target.iloc[:split_index])
    actual = target.iloc[split_index:].to_numpy()
    predicted = evaluation_model.predict(features.iloc[split_index:])
    mse = float(mean_squared_error(actual, predicted))

    final_model = _build_pipeline()
    final_model.fit(features, target)

    evaluation = pd.DataFrame(
        {
            "Actual": actual,
            "Predicted": predicted,
            "Date/time": clean["date_time"].iloc[split_index:].to_numpy(),
        }
    ).set_index("Date/time")
    return TrainingResult(
        model=final_model,
        metrics={
            "MAE": float(mean_absolute_error(actual, predicted)),
            "MSE": mse,
            "RMSE": float(np.sqrt(mse)),
            "R²": float(r2_score(actual, predicted)),
        },
        evaluation=evaluation,
        rows_used=int(len(clean)),
        discarded_rows=discarded_rows,
        duplicate_rows=duplicate_rows,
        start_time=clean["date_time"].min(),
        end_time=clean["date_time"].max(),
    )


def make_prediction_features(
    prediction_date: pd.Timestamp,
    hour: int,
    temperature: float,
    rainfall: float,
    snowfall: float,
    cloud_cover: float,
    weather: str,
    holiday: str,
) -> pd.DataFrame:
    """Convert validated GUI values into the exact training feature schema."""
    if not 0 <= hour <= 23:
        raise ValueError("Hour must be between 0 and 23.")
    if temperature < 0:
        raise ValueError("Temperature must be zero or greater Kelvin.")
    if rainfall < 0 or snowfall < 0:
        raise ValueError("Rainfall and snowfall must be zero or greater.")
    if not 0 <= cloud_cover <= 100:
        raise ValueError("Cloud cover must be between 0 and 100 percent.")
    if not weather.strip() or not holiday.strip():
        raise ValueError("Choose both a weather condition and a holiday status.")

    date_time = pd.Timestamp(prediction_date).normalize() + pd.Timedelta(hours=hour)
    row = pd.DataFrame(
        {
            "date_time": [date_time],
            "temp": [temperature],
            "rain_1h": [rainfall],
            "snow_1h": [snowfall],
            "clouds_all": [cloud_cover],
            "weather_main": [weather],
            "holiday": [holiday],
        }
    )
    return _features_from_frame(row)
