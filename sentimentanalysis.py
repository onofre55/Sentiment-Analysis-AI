#DataCamp NLTK Sentiment Analysis Tutorial for Beginners

# Importing libraries

import pandas as pd
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report

# Downloading nltk corpus
import nltk

nltk.download('all')

# Loading the amazon review dataset

df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')

# Initializing NLTK sentiment analyzer

analyzer = SentimentIntensityAnalyzer()

# Creating get_sentiment function

def get_sentiment(text):

    scores = analyzer.polarity_scores(text)

    sentiment = 1 if scores['pos'] > 0 else 0

    return sentiment

# Applying get_sentiment function

print(' ')
print(' ')
print('Sentiment Prediction')
print(' ')


df['sentiment'] = df['reviewText'].apply(get_sentiment)

print(df.head())

print(' ')
print(' ')

print('Classification Report')
print(' ')

# Accuracy of the model is 80%

print(classification_report(df['Positive'], df['sentiment']))



