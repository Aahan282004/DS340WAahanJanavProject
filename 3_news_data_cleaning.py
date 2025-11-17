from pathlib import Path

import pandas as pd

import compat_warnings  # noqa: F401


BASE_DIR = Path(__file__).resolve().parent
NEWS_PATH = BASE_DIR / "news.csv"
STOCK_PATH = BASE_DIR / "stock_price.csv"
OUTPUT_PATH = BASE_DIR / "news_data.csv"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    return pd.read_csv(path)


def _normalize_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str).str[:10], format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Invalid dates detected in column '{series.name}'.")
    return parsed.dt.strftime("%Y-%m-%d")


def main() -> None:
    news_df = _load_csv(NEWS_PATH)
    stock_df = _load_csv(STOCK_PATH)

    if "Date" not in news_df.columns or "Date" not in stock_df.columns:
        raise ValueError("Both news.csv and stock_price.csv must contain a 'Date' column.")

    stock_df["Date"] = _normalize_dates(stock_df["Date"])
    news_df["Date"] = _normalize_dates(news_df["Date"])

    valid_dates = set(stock_df["Date"])
    filtered = news_df[news_df["Date"].isin(valid_dates)].reset_index(drop=True)
    filtered.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
