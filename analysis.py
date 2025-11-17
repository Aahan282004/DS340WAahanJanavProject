"""
Post-training analysis for the Weighted FinBERT-LSTM project.

Reads the prediction CSVs and metrics JSON files emitted by
`weighted_finbert_lstm.py`, prints a comparison table, and saves a combined
plot showing actual vs predicted close prices for the baseline and weighted
models.
"""

from pathlib import Path
import json
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

import compat_warnings  # noqa: F401


RESULTS_DIR = Path("results")
PLOT_DIR = Path("plots")
COMPARISON_FIG = PLOT_DIR / "comparison_prediction.png"
RESIDUAL_FIG = PLOT_DIR / "residual_distribution.png"

MODELS = ("baseline_finbert", "weighted_finbert")


def _load_metrics(name: str) -> Dict[str, float]:
    metrics_path = RESULTS_DIR / f"{name}_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing metrics file for {name}. "
            "Run `python weighted_finbert_lstm.py` first."
        )
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_predictions(name: str) -> pd.DataFrame:
    predictions_path = RESULTS_DIR / f"{name}_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing prediction file for {name}. "
            "Run `python weighted_finbert_lstm.py` first."
        )
    df = pd.read_csv(predictions_path)
    expected_cols = {"actual_close", "predicted_close"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"{predictions_path} does not contain {expected_cols}")
    return df


def _print_metrics_table() -> None:
    records = []
    for name in MODELS:
        metrics = _load_metrics(name)
        records.append(
            {
                "model": name,
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
                "MAPE": metrics["mape"],
                "R2": metrics["r2"],
            }
        )
    frame = pd.DataFrame(records)
    frame = frame.set_index("model")
    print("\nEvaluation summary:")
    print(frame.to_string(float_format=lambda v: f"{v:0.4f}"))


def _plot_predictions() -> None:
    baseline = _load_predictions("baseline_finbert")
    weighted = _load_predictions("weighted_finbert")

    actual = baseline["actual_close"].to_numpy()
    if not (weighted["actual_close"].to_numpy() == actual).all():
        raise ValueError("Actual series mismatch between baseline and weighted outputs.")

    plt.figure(figsize=(10, 4))
    plt.plot(actual, label="Actual", color="black", linewidth=2)
    plt.plot(
        baseline["predicted_close"],
        label="Baseline FinBERT-LSTM",
        color="tab:orange",
        linewidth=1.5,
    )
    plt.plot(
        weighted["predicted_close"],
        label="Weighted FinBERT-LSTM",
        color="tab:blue",
        linewidth=1.5,
    )
    plt.xlabel("Test timestep")
    plt.ylabel("Close price (USD)")
    plt.title("Actual vs Predicted Close — Baseline vs Weighted")
    plt.legend()
    plt.tight_layout()

    PLOT_DIR.mkdir(exist_ok=True)
    plt.savefig(COMPARISON_FIG, dpi=200)
    plt.close()
    print(f"Combined plot saved to: {COMPARISON_FIG}")


def _plot_residuals() -> None:
    baseline = _load_predictions("baseline_finbert")
    weighted = _load_predictions("weighted_finbert")
    baseline_residuals = baseline["actual_close"] - baseline["predicted_close"]
    weighted_residuals = weighted["actual_close"] - weighted["predicted_close"]

    plt.figure(figsize=(8, 4))
    plt.hist(
        baseline_residuals,
        bins=30,
        alpha=0.6,
        density=True,
        label="Baseline residuals",
        color="tab:orange",
    )
    plt.hist(
        weighted_residuals,
        bins=30,
        alpha=0.6,
        density=True,
        label="Weighted residuals",
        color="tab:blue",
    )
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Residual (Actual - Predicted Close)")
    plt.ylabel("Density")
    plt.title("Residual distribution — Baseline vs Weighted")
    plt.legend()
    plt.tight_layout()

    PLOT_DIR.mkdir(exist_ok=True)
    plt.savefig(RESIDUAL_FIG, dpi=200)
    plt.close()
    print(f"Residual plot saved to: {RESIDUAL_FIG}")


def main() -> None:
    _print_metrics_table()
    _plot_predictions()
    _plot_residuals()


if __name__ == "__main__":
    main()
