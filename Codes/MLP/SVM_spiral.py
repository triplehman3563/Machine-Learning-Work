import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# import plot_decision_regions
from PlotClassification import plot_decision_regions


def spiral(seed=1984):

    np.random.seed(seed)
    n = 200  
    dim = 2  
    class_num = 3  

    x = np.zeros((n*class_num, dim))
    t = np.zeros(n*class_num, dtype=np.int)

    for j in range(class_num):
        for i in range(n):
            rate = i / n
            radius = 1.0*rate
            theta = j*4.0 + 4.0*rate + np.random.randn()*0.2

            ix = n*j + i
            x[ix] = np.array([radius*np.sin(theta),
                              radius*np.cos(theta)]).flatten()
            t[ix] = j

    return x, t


## load spiral  
X, y = spiral()


## plot spiral
point_n = 200
class_n = 3
markers = ['o', 'x', '^']
for i in range(class_n):
    plt.scatter(X[i*point_n:(i+1)*point_n, 0], X[i*point_n:(i+1)*point_n, 1], s=40, marker=markers[i])
plt.show()


## Standardized
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)


## classifier SVM
svm = SVC(kernel='linear', C=1.0, max_iter=-1, 
                     tol=1e-4, verbose=False, random_state=1)
# svm = SVC(kernel='rbf', random_state=1, gamma=0.25, C=1.0)
svm.fit(X_std, y)
print('SVM')
print("Training set score: %f" % svm.score(X_std, y))
plot_decision_regions(X=X_std, y=y,
                      classifier=svm,)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


