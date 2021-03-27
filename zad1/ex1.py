import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import double_std


def sample_points(count, d):
    """ returns number of points that fall inside and outside of n-dimensional ball
        written in n-dimensional square"""
    points = np.random.uniform(low=-1.0, high=1.0, size=(count, d))
    radii = 1
    inner = points[np.sum(points ** 2, axis=1) <= radii]
    outer = points[np.sum(points ** 2, axis=1) > radii]
    return len(inner), len(outer)


def show_results(results):
    df = pd.DataFrame(data=results, columns=["dimensions", "% in"])
    grouped = df.groupby("dimensions").agg([np.mean, double_std])
    grouped["% in"].plot(y='mean', yerr='double_std', capsize=5, fmt='.--', grid=True,
                         title='Hypersphere volume in relation to dimensions')
    plt.ylabel("% of points within hypersphere")
    plt.show()


def main():
    results = []
    for i in range(10):
        for dim in range(2, 20, 1):
            inn, outt = sample_points(10000, dim)
            results.append([dim, inn / (inn + outt)])
    show_results(results)

main()
