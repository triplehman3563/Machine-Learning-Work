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

## LDA 
# Calculate the mean vectors for each class
np.set_printoptions(precision=4)
mean_vecs = []
for label in np.unique(y):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    # print('MV %s: %s\n' % (label, mean_vecs[label]))

# Compute the within-class scatter matrix SW
d = X.shape[1]  # number of features
S_W = np.zeros((d, d))
for label, mv in zip(np.unique(y), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter

# Compute the between-class scatter matrix SB
mean_overall = np.mean(X_train_std, axis=0)
d = X.shape[1]  # number of features
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train == i, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)  # make column vector
    mean_overall = mean_overall.reshape(d, 1)  # make column vector
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

# Solve the generalized eigenvalue 
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))

X_train_lda = X_train_std.dot(w)
X_test_lda = X_test_std.dot(w)

lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
y_hat = lr.predict(X_test_lda)
f1 = f1_score(y_test, y_hat, average='micro') 
print('f1 score (LDA) =', "%.2f" % f1)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0],
                X_train_lda[y_train == l, 1],
                c=c, label=l, marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Plotting
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()