import pandas as pd 
import numpy as np

news_df = pd.read_csv("news.csv")
stock_df = pd.read_csv("stock_price.csv")

stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.strftime("%Y-%m-%d")

news_df = news_df[news_df['Date'].isin(stock_df['Date'].tolist())]

news_df.to_csv("news_data.csv", index=False)
