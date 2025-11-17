import pandas as pd 
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
finbert = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
finbert_pipeline = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)


def FinBERT_sentiment_score(headings):
    """
    compute sentiment score using pretrained FinBERT on -1 to 1 scale. -1 being negative and 1 being positive
    """
    if isinstance(headings, str):
        text = headings.strip()
        inputs = [text] if text and text != '0' else []
    else:
        inputs = []
        for h in headings:
            if isinstance(h, str):
                text = h.strip()
                if text and text != '0':
                    inputs.append(text)
            elif h is not None and not (isinstance(h, float) and np.isnan(h)):
                inputs.append(str(h))
    if not inputs:
        return 0
    result = finbert_pipeline(inputs)
    if isinstance(result, dict):
        result = [result]
    scores = []
    for res in result:
        label = res['label'].lower()
        score = res['score']
        if label == "positive":
            scores.append(score)
        elif label == "negative":
            scores.append(-score)
        else:
            scores.append(0)
    return float(np.mean(scores))


def VADER_sentiment_score(heading):
    """
    compute sentiment score using pretrained VADER on -1 to 1 scale. -1 being negative and 1 being positive
    """
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    result = analyzer.polarity_scores(heading)
    if result['pos'] == max(result['neg'], result['neu'], result['pos']):
        return result['pos']
    if result['neg'] == max(result['neg'], result['neu'], result['pos']):
        return (0 - result['neg'])
    else:
        return 0

news_df = pd.read_csv("news_data.csv")



BERT_sentiment = []


for i in range(len(news_df)):
    news_list = news_df.iloc[i, 1:].tolist()
    news_list = [item for item in news_list if pd.notna(item)]
    score_BERT = FinBERT_sentiment_score(news_list)
    BERT_sentiment.append(score_BERT)


# print(news_df.iloc[129])

news_df['FinBERT score'] = BERT_sentiment

news_df.to_csv("sentiment.csv")
