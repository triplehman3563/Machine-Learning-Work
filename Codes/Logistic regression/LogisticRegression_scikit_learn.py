import numpy as np
import matplotlib.pyplot as plt

# import plot_decision_regions
from PlotClassification import plot_decision_regions

# sci-kit learn
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import multilabel_confusion_matrix

# Loading the Iris dataset from scikit-learn
iris = datasets.load_iris()
X = iris.data[:, :4]
y = iris.target

print('Class labels:', np.unique(y))

# Splitting data into 75% training and 25% test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1, stratify=y)

# Standardizing the features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1.0, random_state=1)
lr.fit(X_train_std, y_train)

y_pred = lr.predict(X_test_std)


#Compute performance
mcm = multilabel_confusion_matrix(y_test, y_pred)
tn = mcm[:, 0, 0]
tp = mcm[:, 1, 1]
fn = mcm[:, 1, 0]
fp = mcm[:, 0, 1]
target_names = ['Setosa', 'Versicolor', 'Virginica']
print('Confusion matrix')
print(metrics.confusion_matrix(y_test, y_pred))
print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))





confmat = confusion_matrix(y_test, y_pred)



# Plot confusion matrix
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

# Plotting
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
weights, params = [], [] 
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c, random_state=1) 
    lr.fit(X_train_std, y_train) 
    weights.append(lr.coef_[2]) 
    params.append(10.**c)
weights = np.array(weights) 
plt.plot(params, weights[:, 0],label='petal length') 
plt.plot(params, weights[:, 1],linestyle='‐‐',label='petal width') 
plt.ylabel('weight coefficient') 
plt.xlabel('C')
plt.legend(loc='upper left') 
plt.xscale('log')
plt.show()

