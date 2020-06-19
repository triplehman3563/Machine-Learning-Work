import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import plot_decision_regions
from PlotClassification import plot_decision_regions

# Implementing an adaptive linear neuron in Python
class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,  
            # in the case of logistic regression (as we will see later), 
            # we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


# Reading-in the Iris data
df = pd.read_csv('iris.csv', header=None)

# select setosa and versicolor
y = df.iloc[1:101, 4].values
y = np.where(y == 'setosa', -1, 1)
# extract sepal length and petal length
X = df.iloc[1:101, [0, 2]].values
# change str to float
X = X.astype(np.float)

# compare learning rate
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(n_iter=4, eta=0.001).fit(X, y)
print('ada1.cost=', ada1.cost_)
ax[0].plot(range(1, len(ada1.cost_) + 1), (ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Sum-squared-error')
ax[0].set_title('GD Adaline - Learning rate 0.001')

ada2 = AdalineGD(n_iter=4, eta=0.0001).fit(X, y)
print('ada2.cost=', ada2.cost_)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('GD Adaline - Learning rate 0.0001')
plt.show()

# Choose 0.0001 learning rate training until convergence
ada3 = AdalineGD(n_iter=500, eta=0.0001).fit(X, y)
plt.plot(range(1, len(ada3.cost_) + 1), ada3.cost_)
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.title('GD Adaline - Learning rate 0.0001')
plt.show()

# standardize features
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# Training a adaptive linear model on the Iris dataset
ada = AdalineGD(n_iter=10, eta=0.01)
ada.fit(X_std, y)

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.title('GD Adaline - Learning rate 0.01 with standardization')
plt.tight_layout()
plt.show()

# Plotting decision regions
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

