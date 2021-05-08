import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import random
mrange = (0, 10)

def main():
    data = pd.read_csv("menu.csv")
    X, y = clean_data(data)
    print(X.head())
    kmeans_experiments(X, y)


def my_random(X, n_clusters, random_state):
    return [[random.uniform(*mrange) for _ in range(X.shape[1])] for _i in range(n_clusters)]

def clean_data(data, y_col='Category'):
    serving = data['Serving Size'].apply(lambda x: float(x.split(" ", 1)[0]))
    data.drop(columns=['Total Fat (% Daily Value)', 'Saturated Fat (% Daily Value)', 'Cholesterol (% Daily Value)',
                       'Sodium (% Daily Value)', 'Carbohydrates (% Daily Value)', 'Dietary Fiber (% Daily Value)',
                       'Vitamin A (% Daily Value)', 'Vitamin C (% Daily Value)', 'Calcium (% Daily Value)',
                       'Iron (% Daily Value)', 'Serving Size', 'Item'], inplace=True)
    y = data[y_col]
    X = data.drop(columns=[y_col])
    for col in X.columns:
        X[col] = X[col].astype(float)
        X[col] = X[col]/serving
    scaler = MinMaxScaler(feature_range=mrange) # todo range? center data?
    X[X.columns] = scaler.fit_transform(X)
    return X, y



def kmeans_experiments(X, y):
    initializations = [  my_random] #, 'random'] # todo w pełni losowa z zakresu każdej cechy (losujemy każdy środek z rozkładem jednostajnym)
    for initial in initializations:
        monitor_clustering(X, initial)


def monitor_clustering(X, init_f, T=1):
    print("monitoring", init_f)
    max_iter = 12
    db = []
    for T in range(100):
        kmeans = KMeans(n_clusters=5, max_iter=max_iter, n_init=1, init=init_f, verbose=True)
        kmeans, idb_scores = kmeans.fit(X)
        for idx, val in enumerate(idb_scores):
            db.append([idx, val])
    db = pd.DataFrame(db, columns=["nr_iter", "db_score"])
    db = db.groupby('nr_iter').agg([np.mean, double_std])
    db['db_score'].plot(y='mean', yerr='double_std', capsize=5, fmt='.--', grid=True,
                        title=f'average db score progression with init function={init_f}')
    plt.ylabel("mean db score")
    plt.xlim(-1, 12)
    plt.ylim(0, 2.3)
    plt.show()

def double_std(x):
    return np.std(x) * 2

if __name__ == '__main__':
    main()