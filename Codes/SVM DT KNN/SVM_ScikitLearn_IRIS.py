import numpy as np
import matplotlib.pyplot as plt

# import plot_decision_regions
from PlotClassification import plot_decision_regions

# sci-kit learn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Loading the Iris dataset from scikit-learn
wine = datasets.load_wine() 
X = wine.data[:, 11:12]
y = wine.target
#iris = datasets.load_iris()
#X = iris.data[:, [2, 3]]
#y = iris.target
print('Class labels:', np.unique(y))

# Splitting data into 70% training and 30% test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)

# Standardizing the features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1000, random_state=1)
#svm = SVC(kernel='poly', random_state=1, gamma=10000, C=1000)
svm.fit(X_train_std,y_train)

y_hat=svm.predict(X_test_std)
print(y_hat)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, 
                      y_combined,
                      classifier=svm, 
                      test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()








