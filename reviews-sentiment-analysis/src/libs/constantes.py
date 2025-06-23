import nltk
import os
nltk.download('stopwords')
nltk.download('rslp')
from collections import Counter
from autocorrect import Speller
from typing import Dict, Optional
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import ToktokTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import RSLPStemmer, WordNetLemmatizer
import numpy as np
import spacy
import plotly.express as px
import string
import joblib
import openpyxl

nlp = spacy.load("en_core_web_sm")
nltk.download('wordnet')

tokenizer = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()

column_name_mapping = {
    "column_review_title": "Summary",
    "column_review_text": "Text",
    "column_review_rating": "Score",
    "column_review_date": "Time",
    "column_product_identifier": "ProductId"

}