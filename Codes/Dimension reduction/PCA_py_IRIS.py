import numpy as np
import matplotlib.pyplot as plt

# import plot_decision_regions
from PlotClassification import plot_decision_regions

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.metrics import f1_score

## load data
iris = datasets.load_iris()
X = iris.data[:, [0, 1, 2, 3]]
y = iris.target

## split data to train and tset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

## Steps to perform PCA
# standardize the dataset.
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# construct the covariance matrix
cov_mat = np.cov(X_train_std.T)

# decompose the covariance matrix 
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# construct a projection matrix W
w = eigen_vecs[:, 0:2]

# transform X' = X．W
X_train_pca = X_train_std.dot(w)
X_test_pca = X_test_std.dot(w)

## Using the result of PCA to classifier
lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)

## plot X_train_pca
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# Plotting
X_combined_std = np.vstack((X_train_pca, X_test_pca))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# F_1 score
y_hat = lr.predict(X_test_pca)
f1 = f1_score(y_test, y_hat, average='micro') 
print('f1 score (PCA) =', "%.2f" % f1)