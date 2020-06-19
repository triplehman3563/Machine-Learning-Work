import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


docs = np.array([
        'This is why I hate the Da Vinci Code, it is so boring',
        'The code is written in Python',
        'This is fucking horrible'])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

## Vectorized
vectorizer = CountVectorizer()
BOW = vectorizer.fit_transform(docs)
print('Bag of words in an alphabet order :')
print(BOW.toarray())
print('\n')
print('List of words in an order of occurrence time')
print(vectorizer.vocabulary_)
print('\n')

## TF-IDF
np.set_printoptions(precision=1)
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
X = tfidf.fit_transform(BOW)
print('TF-IDF of words in an alphabet order')
print(X.toarray())
print('\n')