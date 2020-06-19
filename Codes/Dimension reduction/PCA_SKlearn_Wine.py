import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.metrics import f1_score

## load data
wine = datasets.load_wine()
X = wine.data[:, 0:]
y = wine.target
print(np.unique(y))

## split data to 75% train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1, stratify=y)

# standardize the dataset.
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

## Using SKlearn to perform PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=1, svd_solver='auto')

# transform X' = X．W
pca.fit(X_train_std)
X_train_pca = pca.transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

print('X'+"'"+' Dimension=', X_train_pca.shape)

## Using the result of PCA to classifier
lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)

# F_1 score
y_hat = lr.predict(X_test_pca)
f1 = f1_score(y_test, y_hat, average='micro') 
print('f1 score (PCA) =', "%.2f" % f1)

## Using all feature and compare
lr_all_fea = LogisticRegression()
lr_all_fea = lr_all_fea.fit(X_train_std, y_train)

y_hat_all_fea = lr_all_fea.predict(X_test_std)
f1_all_fea = f1_score(y_test, y_hat_all_fea, average='micro') 
print('f1 score (All features)=', "%.2f" % f1_all_fea)