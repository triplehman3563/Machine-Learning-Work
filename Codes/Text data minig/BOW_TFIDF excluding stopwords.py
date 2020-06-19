import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn as nn
import torch as t


docs = np.array([
        'Topic modeling describes the broad task of assigning topics to unlabelled text documents.',
'A typical application would be the categorization of documents in a large text corpus of newspaper articles',
 'Consider topic modeling as a clustering task, an unsupervised learning'
        ])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

## Vectorized excluding stop word
vectorizer_nonStop = CountVectorizer(stop_words="english")
BOW_nonStop = vectorizer_nonStop.fit_transform(docs)
print('Bag of non-stop words in an alphabet order :')
print(BOW_nonStop.toarray())
print('\n')
print('List of non-stop words in an order of occurence time')

## TF-IDF of non-stop words
np.set_printoptions(precision=2)
tfidf_nonStop = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
X_nonStop = tfidf_nonStop.fit_transform(BOW_nonStop)
print(vectorizer_nonStop.vocabulary_)
print('TF-IDF of non-stop words in an alphabet order')
print(X_nonStop.toarray())
print('\n')
#print(X_nonStop)
vectorizer = CountVectorizer()
BOW = vectorizer.fit_transform(docs)
print(vectorizer.vocabulary_)
word_to_ix = vectorizer.vocabulary_
embeds = nn.Embedding(29, 3)  
lookup_tensor = t.tensor([word_to_ix["topic"]], dtype=t.long)
written_embed = embeds(lookup_tensor)
print(written_embed)
lookup_tensor = t.tensor([word_to_ix["modeling"]], dtype=t.long)
written_embed = embeds(lookup_tensor)
print(written_embed)
lookup_tensor = t.tensor([word_to_ix["describes"]], dtype=t.long)
written_embed = embeds(lookup_tensor)
print(written_embed)
lookup_tensor = t.tensor([word_to_ix["broad"]], dtype=t.long)
written_embed = embeds(lookup_tensor)
print(written_embed)
