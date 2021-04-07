import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from DataGenerator import load_data_from_image
import pandas as pd


def display(dataset, mesh, axis):
    X, y = dataset
    axis.scatter(X[:, 0], X[:, 1], marker='o', c=y,
                edgecolor='k')
    axis.set_xlim(mesh[0].min(), mesh[0].max())
    axis.set_ylim(mesh[1].min(), mesh[1].max())
    axis.set_aspect(1)


def get_mesh(X, h=0.2):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


dataset = load_data_from_image('set2.png', 1000)
X, y = dataset
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
classifiers = [
    KNeighborsClassifier(1, metric='euclidean'),
    KNeighborsClassifier(13, metric='euclidean'),
    KNeighborsClassifier(1, algorithm='brute', metric='mahalanobis', metric_params={'V': np.cov(X_train)}),
    KNeighborsClassifier(9, weights='distance', metric='euclidean'),
]
fig, ax = plt.subplots(len(classifiers)+1, 1, figsize=(20, 20))
mesh = get_mesh(X)
display(dataset, mesh, ax[0])
for idx, cls in enumerate(classifiers):
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    df = pd.DataFrame({'pred': y_pred, 'pos_x': X_test[:, 0], 'pos_y': X_test[:, 1], 'y_test': y_test})
    df_wrong = df[df.pred!=df.y_test]
    X_wrong, y_wrong = df_wrong[['pos_x', 'pos_y']].values, df_wrong['y_test']
    display([X_wrong, y_wrong], mesh, ax[idx+1])
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# fig.tight_layout()
plt.show()