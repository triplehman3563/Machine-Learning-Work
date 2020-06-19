# import plot_decision_regions
from PlotClassification import plot_decision_regions

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

## plot non-linear picture
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
plt.tight_layout()
plt.show()

## split data to train and tset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

## PCA
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

plt.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1],
              color='red', marker='^', alpha=0.5)
plt.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1],
              color='blue', marker='o', alpha=0.5)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.tight_layout()
plt.show()

## Using the result of PCA to classifier
lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)

# F_1 score
y_hat = lr.predict(X_test_pca)
f1 = f1_score(y_test, y_hat, average='micro') 
print('f1 score (PCA) =', "%.2f" % f1)

# Plotting
plot_decision_regions(X_test_pca, y_test,classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


