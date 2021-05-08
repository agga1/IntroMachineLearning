import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

from ex1 import clean_data, my_random

k = 9
pd.set_option("display.max_rows", 200)
def main():
    data = pd.read_csv("menu.csv")
    X, y = clean_data(data)
    analize(X, y)



def double_std(x):
    return np.std(x) * 2

def analize(X, y):
    kmeans = KMeans(n_clusters=k, max_iter=50, n_init=20, init=my_random, verbose=False)
    kmeans, idb_scores = kmeans.fit(X)
    labels = kmeans.labels_
    df = pd.DataFrame({"label": labels, "name": y})
    for i in range(k):
        tmp = df[df['label']==i]
        print("class size:", len(tmp))
        print("representants (showing only if at least 2 per type)")
        for name in y.unique():
            if len(tmp[tmp.name==name]) < 1:
                continue
            print('\t',name, len(tmp[tmp.name==name]))

if __name__ == '__main__':
    main()