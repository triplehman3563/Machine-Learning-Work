import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


## load dataset
df = pd.read_csv('mnist_784.csv', header=0)
y = df.iloc[:, -1].values
print(y.shape)
X = df.iloc[:, 0:-1].values
print(X.shape)


## plot dataset 
images_and_labels = list(zip(X, y))
for index, (image, label) in enumerate(images_and_labels[:10]):
    plt.axis('off')
    plt.imshow(image.reshape(28, 28), cmap='gray', interpolation='nearest')
    plt.title(label)
    plt.tight_layout()
    plt.show()


## preprocessing 
X = X / 255.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)

## classifier MLP one hidden layer
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose= False, tol=1e-4, random_state=1,
                    learning_rate_init=0.1)
mlp.fit(X_train, y_train)
print('One hidden layer')
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))


## classifier MLP two hidden layers
from sklearn.neural_network import MLPClassifier
mlp2 = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose= False, tol=1e-4, random_state=1,
                    learning_rate_init=0.1)

mlp2.fit(X_train, y_train)
print('Two hidden layer')
print("Training set score: %f" % mlp2.score(X_train, y_train))
print("Test set score: %f" % mlp2.score(X_test, y_test))