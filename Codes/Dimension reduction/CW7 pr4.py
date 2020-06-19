
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import neighbors, datasets
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
## load data
wine = datasets.load_wine()
X = wine.data[:, 1:9]
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
f1=[None]*12
accuracy=[None]*12
#for loop 
for i in range(1, 5):

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    lda = LDA(n_components=i)
    lda.fit(X_train_std, y_train)
    X_train_lda = lda.transform(X_train_std)
    X_test_lda = lda.transform(X_test_std)

    lr = LogisticRegression()
    lr = lr.fit(X_train_lda, y_train)
    y_hat = lr.predict(X_test_lda)
    f1[i] = f1_score(y_test, y_hat, average='micro') 
    print('f1 score (LDA) =', "%.2f" % f1[i])

