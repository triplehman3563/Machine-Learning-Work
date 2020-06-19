from sklearn.feature_extraction.text import CountVectorizer
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

t.manual_seed(2)

docs = np.array([
       'Topic modeling describes the broad task of assigning topics to unlabelled text documents.',
'A typical application would be the categorization of documents in a large text corpus of newspaper articles',
 'Consider topic modeling as a clustering task, an unsupervised learning'
        ])

vectorizer = CountVectorizer()
BOW = vectorizer.fit_transform(docs)
print(vectorizer.vocabulary_)
word_to_ix = vectorizer.vocabulary_

# 16 words in vocab, 2 dimensional embeddings
embeds = nn.Embedding(29, 3)  
lookup_tensor = t.tensor([word_to_ix["broad"]], dtype=t.long)
written_embed = embeds(lookup_tensor)
print(written_embed)