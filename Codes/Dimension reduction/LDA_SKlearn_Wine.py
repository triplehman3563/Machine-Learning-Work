import numpy as np

# import plot_decision_regions
from PlotClassification import plot_decision_regions

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.metrics import f1_score

## load data
wine = datasets.load_wine()
X = wine.data[:, 0:]
y = wine.target
print(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)
lda.fit(X_train_std, y_train)
X_train_lda = lda.transform(X_train_std)
X_test_lda = lda.transform(X_test_std)

lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
y_hat = lr.predict(X_test_lda)
f1 = f1_score(y_test, y_hat, average='micro') 
print('f1 score (LDA) =', "%.2f" % f1)

plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()