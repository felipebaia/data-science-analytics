import nltk
nltk.download('stopwords')
nltk.download('rslp')
from collections import Counter
from typing import Dict, Optional
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import RSLPStemmer
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")
import plotly.express as px
import string
import joblib

column_name_mapping = {
    "column_review_title": "Summary",
    "column_review_text": "Text",
    "column_review_rating": "Score",
    "column_review_date": "Time",
    "column_product_identifier": "ProductId"

}