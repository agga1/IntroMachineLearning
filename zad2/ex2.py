import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from DataGenerator import load_data_from_image
import pandas as pd

def main():
    dataset = load_data_from_image('set2.png', 800)

    clf1_kwargs = {'metric': 'euclidean'}
    clf2_kwargs = {'metric': 'euclidean', 'weights': 'distance'}
    clfs_kwargs = [clf1_kwargs, clf2_kwargs]
    for clf_kwargs in clfs_kwargs:
        mean, std = evaluate_clf_params(dataset, **clf_kwargs)
        print(mean, std)

def evaluate_clf_params(dataset, repeat=16, **clf_kwargs):
    accuracies = []
    for i in range(repeat):
        X, y = dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        k = find_best_k(X_train, y_train, **clf_kwargs)
        acc = get_accuracy(X_train, X_test, y_train, y_test, k, **clf_kwargs)
        accuracies.append(acc)
        print(k)
    series = pd.Series(data=accuracies)
    print(series)
    return series.mean(), series.std()

def find_best_k(X, y, **clf_kwargs):
    max_acc = 0
    max_ind = -1
    for k in range(1, 21):
        acc = get_avg_accuracy(k, X, y, **clf_kwargs)
        if acc > max_acc:
            max_ind = k
            max_acc = acc
    return max_ind

def get_avg_accuracy(k, X, y, repeat=16, **clf_kwargs):
    results = []
    for i in range(repeat):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
        accuracy = get_accuracy(X_train, X_test, y_train, y_test, k, **clf_kwargs)
        results.append(accuracy)
    return sum(results)/len(results)

def get_accuracy(X_train, X_test, y_train, y_test, k, **clf_kwargs):
    classifier = KNeighborsClassifier(k, **clf_kwargs)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

main()