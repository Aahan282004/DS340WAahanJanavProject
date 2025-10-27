"""
Generate FinBERT sentiment features with reliability and recency weighting.

This script upgrades the original sentiment extraction step by:
 1. Melting daily news headlines into a long format.
 2. Scoring each headline with FinBERT on the [-1, 1] scale.
 3. Applying source reliability weights (heavier weight for vetted outlets).
 4. Applying an exponential time decay so recent news dominates the signal.

Results are merged with the trading calendar from `stock_price.csv`
and exported to `sentiment.csv` with both the raw and weighted scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


MODEL_NAME = "ProsusAI/finbert"
LOOKBACK_DAYS = 3
HALF_LIFE_DAYS = 1.5

# Adjust or extend this list to plug in more sentiment sources.
SOURCE_CONFIG: List[dict] = [
    {
        "name": "nyt",
        "path": Path("news_data.csv"),
        "date_column": "Date",
        "text_columns": None,  # use all news columns
        "reliability": 0.9,
    },
]


def _load_source(cfg: dict) -> Optional[pd.DataFrame]:
    csv_path: Path = cfg["path"]
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    date_column: str = cfg["date_column"]
    if date_column not in df.columns:
        raise ValueError(f"{csv_path} is missing required column '{date_column}'")

    text_columns: Optional[Iterable[str]] = cfg.get("text_columns")
    if text_columns is None:
        text_columns = [col for col in df.columns if col != date_column]

    melted = df.melt(
        id_vars=[date_column],
        value_vars=list(text_columns),
        var_name="field",
        value_name="text",
    )
    melted = melted.dropna(subset=["text"])
    melted["text"] = melted["text"].astype(str).str.strip()
    melted = melted[melted["text"] != ""]
    melted = melted.rename(columns={date_column: "date"})
    melted["source"] = cfg["name"]
    return melted[["date", "source", "text"]]


def _load_all_sources() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for cfg in SOURCE_CONFIG:
        df = _load_source(cfg)
        if df is not None:
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No sentiment sources found. Make sure the paths in SOURCE_CONFIG exist."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = _parse_date_column(combined["date"])
    return combined


def _parse_date_column(series: pd.Series) -> pd.Series:
    """
    Normalise date strings that may contain timezone offsets by truncating to
    YYYY-MM-DD and parsing with pandas.
    """
    trimmed = series.astype(str).str[:10]
    parsed = pd.to_datetime(trimmed, format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any():
        raise ValueError("Encountered invalid date entries while parsing sentiment data.")
    return parsed.dt.normalize()


def _build_analyzer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def _score_sentiment(analyzer, text: str) -> float:
    result = analyzer(text[:512])[0]
    label = result["label"].lower()
    score = float(result["score"])
    if label == "positive":
        return score
    if label == "negative":
        return -score
    return 0.0


def _apply_reliability(df: pd.DataFrame) -> pd.DataFrame:
    weight_map = {
        cfg["name"]: cfg.get("reliability", 0.5)
        for cfg in SOURCE_CONFIG
        if cfg["path"].exists()
    }
    df["reliability"] = df["source"].map(weight_map).fillna(0.5)
    return df


def _aggregate_with_decay(
    entries: pd.DataFrame,
    reference_dates: pd.Series,
) -> pd.DataFrame:
    entries = entries.sort_values("date")
    decay_lambda = np.log(2) / HALF_LIFE_DAYS
    results = []

    for current_date in reference_dates:
        lower = current_date - pd.Timedelta(days=LOOKBACK_DAYS)
        window = entries.loc[
            (entries["date"] >= lower) & (entries["date"] <= current_date)
        ].copy()

        if window.empty:
            results.append(
                {
                    "Date": current_date,
                    "finbert_sentiment": 0.0,
                    "weighted_finbert_sentiment": 0.0,
                    "message_count": 0,
                }
            )
            continue

        window["days_back"] = (current_date - window["date"]).dt.days
        window["time_weight"] = np.exp(-decay_lambda * window["days_back"])
        window["combined_weight"] = window["reliability"] * window["time_weight"]

        unweighted = window["sentiment"].mean()
        total_weight = window["combined_weight"].sum()
        if total_weight == 0:
            weighted = 0.0
        else:
            weighted = float(
                (window["combined_weight"] * window["sentiment"]).sum() / total_weight
            )

        results.append(
            {
                "Date": current_date,
                "finbert_sentiment": float(unweighted),
                "weighted_finbert_sentiment": weighted,
                "message_count": int(len(window)),
            }
        )

    aggregated = pd.DataFrame(results)
    aggregated["Date"] = aggregated["Date"].dt.strftime("%Y-%m-%d")
    return aggregated


def main() -> None:
    entries = _load_all_sources()
    analyzer = _build_analyzer()
    entries["sentiment"] = entries["text"].apply(
        lambda text: _score_sentiment(analyzer, text)
    )
    entries = _apply_reliability(entries)

    stock_df = pd.read_csv("stock_price.csv")
    stock_df["Date"] = _parse_date_column(stock_df["Date"])
    timeline = stock_df["Date"].drop_duplicates().sort_values()

    sentiment = _aggregate_with_decay(entries, timeline)
    sentiment.to_csv("sentiment.csv", index=False)


if __name__ == "__main__":
    main()
