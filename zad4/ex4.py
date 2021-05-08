import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from ex1 import clean_data, my_random


def visualize_2d(dataset, labels):
    pca = apply_pca(dataset, n_components=2)
    new_images = pca.transform(dataset)
    df = pd.DataFrame(new_images, columns=['x', 'y'])
    df['labels'] = labels
    print(df)
    plt.scatter(df.x, df.y, c=df.labels, edgecolors='black')
    plt.show()
    print("reduced dataset shape:", new_images.shape)

def apply_pca(dataset, *args, **kwargs):
    print("dataset shape:", dataset.shape)
    pca = PCA(*args, **kwargs)
    return pca.fit(dataset)

def main():
    data = pd.read_csv("menu.csv")
    X, y = clean_data(data)
    kmeans = KMeans(n_clusters=9, max_iter=50, n_init=20, init='random', verbose=False)
    kmeans, idb_scores = kmeans.fit(X)
    labels = kmeans.labels_
    visualize_2d(X, labels)
    visualize_2d(kmeans.cluster_centers_ , range(9))
    labels = {"Breakfast":0, "Beef & Pork": 1, "Chicken & Fish":2, "Salads":3,
              "Snacks & Sides": 4, "Desserts":5, "Beverages":6, "Coffee & Tea":7, "Smoothies & Shakes":8
              }
    visualize_2d(X, [labels[name] for name in y])

if __name__ == '__main__':
    main()

