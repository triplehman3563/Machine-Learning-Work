# import plot_decision_regions
from PlotClassification import plot_decision_regions

import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

## SK kernel pca
from sklearn.decomposition import KernelPCA

## split data to train and tset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
#kpca = KernelPCA(n_components=2, kernel='poly', gamma=15)

kpca.fit(X_train)
X_kpca_train = kpca.transform(X_train)
X_kpca_test = kpca.transform(X_test)

plt.scatter(X_kpca_train[y_train == 0, 0], X_kpca_train[y_train == 0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_kpca_train[y_train == 1, 0], X_kpca_train[y_train == 1, 1],
            color='blue', marker='o', alpha=0.5)
plt.xlabel('KPC 1')
plt.ylabel('KPC 2')
plt.tight_layout()
plt.show()

## Using the result of kernel PCA to classifier
lr = LogisticRegression()
lr = lr.fit(X_kpca_train, y_train)
y_hat = lr.predict(X_kpca_test)
f1 = f1_score(y_test, y_hat, average='micro') 
print('f1 score =', "%.2f" % f1)

plot_decision_regions(X_kpca_test, y_test, classifier=lr)
plt.xlabel('KPCA 1')
plt.ylabel('KPCA 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


