import datetime
import os
from typing import List

import numpy as np
import pandas as pd
from pynytimes import NYTAPI

import compat_warnings  # noqa: F401  # keeps urllib3 LibreSSL warning quiet


NYT_API_KEY = os.environ.get("NYT_API_KEY", "5UI21WrJdSgZtHZpljOncwS0qMuJuOcs")


def get_news(year: int, month: int, day: int) -> List[str]:
    """
    Fetch the top 10 finance-related headlines for a given day from NYT.

    The upstream API sporadically returns `None` (instead of a dict/list) when
    the request is rate-limited or the response payload is empty, so we guard
    against those cases and return an empty list instead of crashing the run.
    """
    nyt = NYTAPI(NYT_API_KEY, parse_dates=True)
    try:
        articles = nyt.article_search(
            results=10,
            dates={
                "begin": datetime.datetime(year, month, day),
                "end": datetime.datetime(year, month, day),
            },
            options={
                "sort": "relevance",
                "news_desk": [
                    "Business",
                    "Business Day",
                    "Entrepreneurs",
                    "Financial",
                    "Technology",
                ],
                "section_name": ["Business", "Business Day", "Technology"],
            },
        )
    except (TypeError, ValueError) as exc:
        print(
            f"[WARN] Failed to fetch NYT articles for {year}-{month:02}-{day:02}: {exc}"
        )
        return []

    if not articles:
        print(f"[WARN] No NYT articles returned for {year}-{month:02}-{day:02}")
        return []

    headlines: List[str] = []
    for article in articles:
        abstract = article.get("abstract") or ""
        abstract = abstract.replace(",", "").strip()
        if abstract:
            headlines.append(abstract)
    return headlines

df = pd.DataFrame()



def generate_news_file():
    """
    store news headings everyday of Q3 2022 in csv
    """
    start = '2020-10-01'
    end = '2022-09-30'
    mydates = pd.date_range(start, end)
    dates = []
    for i in range(len(mydates)):
        dates.append(mydates[i].strftime("%Y-%m-%d"))
    matrix = np.zeros((len(dates) + 1, 11), dtype=object)  
    matrix[0, 0] = "Date"

    for i in range(10):
        matrix[0, i + 1] = f"News {i + 1}"
    for i in range(len(dates)):
        matrix[i + 1, 0] = dates[i]
        y, m, d = dates[i].split("-")
        news_list = get_news(int(y), int(m), int(d))
        for j in range(len(news_list)):
            matrix[i + 1, j + 1] = news_list[j]
    df = pd.DataFrame(matrix)
    df.to_csv("news.csv", index=False)


generate_news_file()
