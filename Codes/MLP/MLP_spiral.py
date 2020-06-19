import numpy as np
import matplotlib.pyplot as plt
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


## classifier MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=False, tol=1e-4, random_state=1,
                    learning_rate_init=0.1)

mlp.fit(X_std, y)
print('MLP')
print("Training set score: %f" % mlp.score(X_std, y))
plot_decision_regions(X=X_std, y=y, classifier=mlp,)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


