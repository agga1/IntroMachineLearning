import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sample_vectors_angle(count_points,count_vectors, d):
    angles = []
    points = np.random.uniform(low=-1.0, high=1.0, size=(count_points, d))
    for _ in range(count_vectors):
        points4 = points[np.random.choice(points.shape[0], 4, replace=False), :]
        v1 = to_vector(*points4[:2])
        v2 = to_vector(*points4[2:])
        angles.append(get_angle_deg(v1, v2))
    return angles

def to_bins(angles, n_bins,  bin_span=10):
    bins = [0]*n_bins
    for a in angles:
        bin_nr = int(a/bin_span)
        bins[bin_nr] += 1
    return bins


def to_vector(point1, point2):
    return point1 - point2

def get_angle_deg(vector1, vector2):
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    return np.arccos(unit_vector1@unit_vector2)*180/np.pi

def display_results(dim, means, stds, bin_span, n_bins):
    ticks = [bin_span * i for i in range(n_bins)]
    print(len(means), len(stds), len(ticks))
    df = pd.DataFrame({'ticks': ticks, 'mean': means,'std': stds})
    df.plot(kind='bar', x='ticks', y='mean', yerr='std', capsize=3,
            title=f'distribution of angle between vectors in {dim} dimensions', xlabel='angle [degrees]', ylabel='nr of occurrences')
    plt.show()
    print(df.head())

def get_distribution(dim, tries, n_bins=18, bin_span=10):
    all_results = np.zeros((tries, n_bins))
    for i in range(tries):
        angles = sample_vectors_angle(1000, 5000, dim)
        bins = to_bins(angles, 18, 10)
        all_results[i] = bins
    print(all_results)
    means = all_results.mean(axis=0)
    stds = all_results.std(axis=0)
    stds = 2*stds
    display_results(dim, means, stds, bin_span, n_bins)

get_distribution(3, 10)
