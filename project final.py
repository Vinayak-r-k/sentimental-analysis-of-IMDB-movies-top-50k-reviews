import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from matplotlib import style

style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
stemmer = PorterStemmer()
import pandas as pd


pd.set_option('display.max_colwidth', 150)

def data_processing(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\bbr\b', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)
sia = SentimentIntensityAnalyzer()
def polarity(text):
    return sia.polarity_scores(text)['compound']
def sentiment(label):
    if label >= 0.05:
        return "positive"
    elif label <= -0.05:
        return "negative"
    else:
        return "neutral"

choice = input("\nDo you want to write your own review? (y/n): ").lower()

if choice == 'y':
    loop=int(input("\nhow many reviews do you want to write:"))
    while(loop>0):
        review = input("\nEnter your movie review: ")

        cleaned_review = data_processing(review)
        pol = polarity(cleaned_review)
        pred = sentiment(pol)
        print("\nPredicted Sentiment:", pred)

        loop-=1
else:
    df = pd.read_csv(r"C:\Users\vinay\Downloads\project internship\IMDB Dataset.csv")
    n = int(input("\nEnter the number of reviews to be accessed: "))
    df = df.head(n)
    text_df = df.drop(['sentiment'], axis=1)
    actual_sentiment = df['sentiment']
    print("\nBefore cleaning:\n\n", text_df.head(n))
    text_df['review'] = text_df['review'].apply(data_processing)
    text_df = text_df.drop_duplicates('review')
    print("\nAfter cleaning:\n\n", text_df.head(n))
    text_df['polarity'] = text_df['review'].apply(polarity)
    text_df['predicted_sentiment'] = text_df['polarity'].apply(sentiment)
    text_df['actual_sentiment'] = actual_sentiment[:len(text_df)]  
    text_df['is_correct'] = text_df['predicted_sentiment'] == text_df['actual_sentiment']
    correct = text_df['is_correct'].sum()
    incorrect = len(text_df) - correct
    print("\n\t\t\tResults:")
    print(text_df[['predicted_sentiment', 'actual_sentiment', 'is_correct']])
    print(f"\nCorrect predictions: {correct}")
    print(f"Incorrect predictions: {incorrect}")
