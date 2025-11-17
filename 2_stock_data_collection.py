from pathlib import Path

import pandas as pd
import yfinance as yf

import compat_warnings  # noqa: F401


OUTPUT_PATH = Path("stock_price.csv")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the yfinance MultiIndex columns to a simple Index."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def download_stock_data(ticker: str, start: str, end: str) -> Path:
    """
    Download stock price data from Yahoo Finance and export a clean CSV.
    """
    stock_data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    if stock_data.empty:
        raise RuntimeError(f"No data returned from Yahoo Finance for {ticker}.")

    df = _flatten_columns(pd.DataFrame(stock_data)).reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    desired_order = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    # Only keep the columns that are present in the response.
    keep_columns = [col for col in desired_order if col in df.columns]
    if keep_columns:
        df = df[keep_columns]

    df.to_csv(OUTPUT_PATH, index=False)
    return OUTPUT_PATH


if __name__ == "__main__":
    path = download_stock_data("NDX", "2020-10-01", "2022-09-30")
    print(f"Stock data saved to {path}")
