import numpy as np
import matplotlib.pyplot as plt

# import plot_decision_regions
from PlotClassification import plot_decision_regions
# sci-kit learn
from sklearn import datasets
# sci-kit learn
from sklearn.svm import SVC
np.random.seed(1)
wine = datasets.load_wine() 
X = wine.data[:, 10:12]
y = wine.target
#X_xor = np.random.randn(200, 2)
#y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y = np.where(y, 1, -1)
svm = SVC(kernel='rbf', random_state=1, gamma=1, C=10.0)
#  svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X, y)

plot_decision_regions(X, y,
                      classifier=svm)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)







