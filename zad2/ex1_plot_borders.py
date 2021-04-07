import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from DataGenerator import load_data_from_image
import pandas as pd

cmap = ListedColormap(['#FF00BD', '#F2CA19', '#0000FF'])

def main():
    dataset = load_data_from_image('set2.png', 800)
    X, y = dataset
    split_data = train_test_split(X, y, test_size=.4, random_state=42)
    classifiers = [
        KNeighborsClassifier(1, metric='euclidean'),
        KNeighborsClassifier(13, metric='euclidean'),
        KNeighborsClassifier(1, algorithm='brute', metric='mahalanobis', metric_params={'V': np.cov(split_data[1])}),
        KNeighborsClassifier(9, weights='distance', metric='euclidean'),
    ]
    fig, ax, mesh = initialize_display(len(classifiers)+1, dataset)

    plot_classifiers(classifiers, split_data, mesh, ax)

    plt.show()

def initialize_display(rows, dataset):
    """ prepare axes, mesh and display original dataset on the first axis. """
    fig, ax = plt.subplots(rows, 1, figsize=(20, 20))
    mesh = get_mesh(dataset)
    display(dataset, mesh, ax[0])
    return fig, ax, mesh

def get_mesh(dataset, h=1.):
    X = dataset[0]
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def display(dataset, mesh, axis):
    X, y = dataset
    axis.scatter(X[:, 0], X[:, 1], marker='o', c=y, cmap=cmap)
    axis.set_xlim(mesh[0].min(), mesh[0].max())
    axis.set_ylim(mesh[1].min(), mesh[1].max())
    axis.set_aspect(1)

def plot_classifiers(classifiers, split_data, mesh, axes):
    X_train, X_test, y_train, y_test = split_data
    for idx, cls in enumerate(classifiers):
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        z = cls.predict(np.c_[mesh[0].ravel(), mesh[1].ravel()])
        z = z.reshape(mesh[0].shape)
        axes[idx+1].contourf(mesh[0], mesh[1], z, cmap=cmap, alpha=.5)
        display((X_train, y_train), mesh, axes[idx+1])




main()