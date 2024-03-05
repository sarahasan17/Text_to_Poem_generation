import joblib
import numpy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from model import vectorizer

def predict(data):
    trans = vectorizer.transform(pd.DataFrame(data)[0])
    clf = joblib.load('model.sav')

    return clf.predict(trans)
