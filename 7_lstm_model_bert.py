"""
FinBERT-LSTM modelling with reliability-weighted sentiment features.

This script evaluates two variants:
 1. Baseline FinBERT-LSTM (uses daily FinBERT sentiment averages).
 2. Weighted FinBERT-LSTM (uses reliability & time-decay weighted sentiment).

Both runs report RMSE, MAE, MAPE, and R², save prediction CSVs, and emit
comparison plots for downstream analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

import compat_warnings  # noqa: F401


SEQUENCE_LENGTH = 10
TRAIN_SPLIT = 0.8
EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 0.002
RANDOM_SEED = 42

FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

RESULTS_DIR = Path("results")
PLOT_DIR = Path("plots")

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler


def _prepare_stock_data() -> pd.DataFrame:
    stock_df = pd.read_csv("stock_price.csv")
    if "Date" not in stock_df.columns:
        raise ValueError(
            "stock_price.csv is missing the 'Date' column. "
            "Run 2_stock_data_collection.py to regenerate it."
        )
    stock_df["Date"] = _parse_date_column(stock_df["Date"]).dt.strftime("%Y-%m-%d")
    return stock_df[["Date"] + FEATURE_COLUMNS]


def _prepare_sentiment_data() -> pd.DataFrame:
    sentiment_df = pd.read_csv("sentiment.csv")
    sentiment_df["Date"] = _parse_date_column(sentiment_df["Date"]).dt.strftime("%Y-%m-%d")
    expected_columns = {"finbert_sentiment", "weighted_finbert_sentiment"}
    if not expected_columns.issubset(sentiment_df.columns):
        raise ValueError(
            "sentiment.csv does not contain the weighted sentiment columns. "
            "Run 4_news_sentiment_analysis.py first."
        )
    return sentiment_df[["Date", "finbert_sentiment", "weighted_finbert_sentiment"]]


def _merge_features(stock_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    merged = stock_df.merge(sentiment_df, on="Date", how="left")
    merged = merged.fillna({"finbert_sentiment": 0.0, "weighted_finbert_sentiment": 0.0})
    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged


def _build_sequences(
    df: pd.DataFrame, feature_cols: List[str], target_col: str
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    features = df[feature_cols].values.astype(np.float32)
    targets = df[[target_col]].values.astype(np.float32)

    sequence_features, sequence_targets = [], []
    for idx in range(len(df) - SEQUENCE_LENGTH):
        sequence_features.append(features[idx : idx + SEQUENCE_LENGTH])
        sequence_targets.append(targets[idx + SEQUENCE_LENGTH])

    X = np.array(sequence_features, dtype=np.float32)
    y = np.array(sequence_targets, dtype=np.float32)

    if len(X) == 0:
        raise ValueError("Not enough rows to build sequences. Increase data length.")

    split_index = int(len(X) * TRAIN_SPLIT)
    X_train_raw, X_test_raw = X[:split_index], X[split_index:]
    y_train_raw, y_test_raw = y[:split_index], y[split_index:]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_train_scaled = feature_scaler.fit_transform(
        X_train_raw.reshape(-1, X_train_raw.shape[2])
    ).reshape(X_train_raw.shape)
    X_test_scaled = feature_scaler.transform(
        X_test_raw.reshape(-1, X_test_raw.shape[2])
    ).reshape(X_test_raw.shape)

    y_train_scaled = target_scaler.fit_transform(y_train_raw)
    y_test_scaled = target_scaler.transform(y_test_raw)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, feature_scaler, target_scaler


def _parse_date_column(series: pd.Series) -> pd.Series:
    trimmed = series.astype(str).str[:10]
    parsed = pd.to_datetime(trimmed, format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any():
        raise ValueError("Encountered invalid date entries in stock or sentiment data.")
    return parsed.dt.normalize()


def _prepare_dataset(
    merged: pd.DataFrame, sentiment_column: str
) -> DatasetBundle:
    feature_cols = FEATURE_COLUMNS + [sentiment_column]
    X_train, y_train, X_test, y_test, feature_scaler, target_scaler = _build_sequences(
        merged, feature_cols, "Close"
    )
    return DatasetBundle(X_train, y_train, X_test, y_test, feature_scaler, target_scaler)


def _build_model(input_shape: Tuple[int, int]) -> Sequential:
    model = Sequential(
        [
            LSTM(64, activation="tanh", return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, activation="tanh", return_sequences=False),
            Dense(16, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


def _train_and_evaluate(bundle: DatasetBundle) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model = _build_model((bundle.X_train.shape[1], bundle.X_train.shape[2]))
    model.fit(
        bundle.X_train,
        bundle.y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    predictions = model.predict(bundle.X_test, verbose=0)
    y_true = bundle.target_scaler.inverse_transform(bundle.y_test).squeeze(-1)
    y_pred = bundle.target_scaler.inverse_transform(predictions).squeeze(-1)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

    return metrics, y_true, y_pred


def _save_outputs(name: str, metrics: Dict[str, float], actual: np.ndarray, predicted: np.ndarray) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)

    pd.DataFrame(
        {"actual_close": actual, "predicted_close": predicted}
    ).to_csv(RESULTS_DIR / f"{name}_predictions.csv", index=False)

    pd.Series(metrics).to_json(RESULTS_DIR / f"{name}_metrics.json", indent=2)

    plt.figure(figsize=(10, 4))
    plt.plot(actual, label="Actual", color="black", linewidth=2)
    plt.plot(predicted, label="Predicted", color="tab:blue", linewidth=1.5)
    plt.title(f"{name.replace('_', ' ').title()} — Actual vs Predicted Close")
    plt.xlabel("Test timestep")
    plt.ylabel("Close price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{name}_prediction.png", dpi=200)
    plt.close()


def main() -> None:
    stock_df = _prepare_stock_data()
    sentiment_df = _prepare_sentiment_data()
    merged = _merge_features(stock_df, sentiment_df)

    experiments = {
        "baseline_finbert": "finbert_sentiment",
        "weighted_finbert": "weighted_finbert_sentiment",
    }

    for name, sentiment_field in experiments.items():
        bundle = _prepare_dataset(merged, sentiment_field)
        metrics, actual, predicted = _train_and_evaluate(bundle)
        _save_outputs(name, metrics, actual, predicted)

        print(f"\nResults for {name}:")
        print(f"  RMSE : {metrics['rmse']:.3f}")
        print(f"  MAE  : {metrics['mae']:.3f}")
        print(f"  MAPE : {metrics['mape']:.3%}")
        print(f"  R^2  : {metrics['r2']:.3f}")
        print(f"  Saved predictions to results/{name}_predictions.csv")
        print(f"  Saved plot to plots/{name}_prediction.png")


if __name__ == "__main__":
    main()
