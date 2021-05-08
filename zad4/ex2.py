import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

from ex1 import clean_data, my_random


def main():
    data = pd.read_csv("menu.csv")
    X, y = clean_data(data)
    k_values = [i for i in range(3, 24)]
    plot_by_k(k_values, X)
def double_std(x):
    return np.std(x) * 2

def plot_by_k(k_values, X):
    vals = []
    for k in k_values:
        print("k=",k)
        for T in range(10):
            kmeans = KMeans(n_clusters=k, max_iter=6, n_init=10, init=my_random, verbose=False)
            kmeans, idb_scores = kmeans.fit(X)
            vals.append([k, idb_scores[-1]])
    df = pd.DataFrame(vals, columns=['k', 'db_score'])
    print(df.head())
    db = df.groupby('k').agg([np.mean, double_std])
    db['db_score'].plot(y='mean', yerr='double_std', capsize=5, fmt='.--', grid=True,
                        title=f'average db score')
    plt.ylabel("mean db score")
    plt.xlim(0, 25)
    # plt.ylim(0, 1.8)
    plt.show()
    pass

if __name__ == '__main__':
    main()