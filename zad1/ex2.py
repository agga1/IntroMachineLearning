import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import double_std


def sample_distance(count, d):
    points = np.random.uniform(low=-1.0, high=1.0, size=(count, d))
    return distance_mean_std(points)

def distance_mean_std(points):
    all_dist = []
    for ind1 in range(len(points-1)):
        for ind2 in range(ind1+1, len(points)):
            dist = np.linalg.norm(points[ind1]-points[ind2])
            all_dist.append(dist)
    s = pd.Series(data=all_dist)
    return s.mean(), s.std()

def show_results(results):
    df = pd.DataFrame(results, columns=['dimensions', 'mean_dist', 'std_dist'])
    df['std_mean_ratio'] = df['std_dist'] / df['mean_dist']
    gp = df.groupby("dimensions").agg([np.mean, double_std])
    gp["mean_dist"].plot(y='mean', yerr='double_std', capsize=5, fmt='.--', grid=True,
                         title='mean distance between points', ylabel='mean distance')
    gp["std_dist"].plot(y='mean', yerr='double_std', capsize=5, fmt='.--', grid=True,
                        title='std of distances between points', ylabel='distances std')
    gp["std_mean_ratio"].plot(y='mean', yerr='double_std', capsize=5, fmt='.--', grid=True, title='std to mean ratio',
                              ylabel='std to mean ratio')
    plt.show()


results = []
for i in range(10):
    for d in range(100, 1000, 100):
        results.append([d, *sample_distance(100, d)])
show_results(results)


