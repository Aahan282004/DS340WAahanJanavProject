# Weighted FinBERT-LSTM
Stock price forecasting with sentiment-weighted FinBERT embeddings and LSTM sequences.

This project extends the baseline FinBERT-LSTM framework from *Predicting Stock Prices with FinBERT-LSTM: Integrating News Sentiment Analysis* by adding a sentiment weighting mechanism. Each textual signal receives:

- **Source reliability weights** – authoritative publishers (e.g., financial news outlets) receive higher trust scores than crowd-sourced feeds such as Reddit.
- **Recency weights** – recent articles have a stronger influence via an exponential time-decay schedule.

The weighted sentiment signal is merged with traditional market features (`Open`, `High`, `Low`, `Close`, `Volume`) and fed into an LSTM forecaster. The pipeline also trains the unweighted FinBERT-LSTM baseline for side-by-side evaluation using RMSE, MAE, MAPE, and R².

## Repository layout

| Script | Purpose |
| ------ | ------- |
| `1_news_collection.py` – `3_news_data_cleaning.py` | Collect and align daily market news with trading days. |
| `4_news_sentiment_analysis.py` | Generates FinBERT sentiment with reliability/time-decay weighting. |
| `5_MLP_model.py`, `6_LSTM_model.py` | Original baselines from the reference paper. |
| `7_lstm_model_bert.py` | Compares baseline vs. weighted FinBERT-LSTM forecasts and saves outputs. |

## Weighted sentiment workflow

1. **Build the sentiment table**  
   Update `SOURCE_CONFIG` in `4_news_sentiment_analysis.py` with the CSV files you want to use:
   ```python
   SOURCE_CONFIG = {
       "nyt": {
           "path": Path("news_data.csv"),
           "reliability": 0.9,
           "date_column": "Date",
       },
       "reddit": {
           "path": Path("reddit_posts.csv"),
           "reliability": 0.45,
           "date_column": "date",
           "text_columns": ["title"],
       },
       ...
   }
   ```
   Run:
   ```bash
   python 4_news_sentiment_analysis.py
   ```
   The script saves `weighted_sentiment.csv` with both unweighted and weighted FinBERT scores for each trading day.

2. **Train and compare models**  
   ```bash
   python 7_lstm_model_bert.py
   ```
   The script outputs metrics for:
   - `baseline_finbert`: unweighted sentiment features.
   - `weighted_finbert`: reliability & time-decayed sentiment features.

   Metrics reported: RMSE, MAE, MAPE, R². Prediction CSVs and metric JSON files are written to `results/`, and run-specific plots land in `plots/`.

3. **Generate comparison plot & table**  
   ```bash
   python analysis.py
   ```
   Creates `plots/comparison_prediction.png` and prints a side-by-side metrics table using the saved results.

## Customisation tips

- Adjust `LOOKBACK_DAYS` and `HALF_LIFE_DAYS` in `4_news_sentiment_analysis.py` to control how far sentiment propagates and how quickly it decays.
- Add or remove data sources by editing the `SOURCE_CONFIG` dictionary.
- Modify the LSTM architecture or hyperparameters (`SEQUENCE_LENGTH`, `EPOCHS`, `LEARNING_RATE`) in `7_lstm_model_bert.py` to explore different model capacities.

## Requirements

- Python 3.9+
- `transformers`, `torch`, `tensorflow`, `scikit-learn`, `pandas`, `numpy`

Install dependencies:
```bash
pip install -r requirements.txt
```

> **Note**: Loading FinBERT requires downloading the pretrained model from Hugging Face during the first run.
