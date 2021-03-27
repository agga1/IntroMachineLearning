import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import double_std

def sample_vectors_angle(count_points,count_vectors, d):
    angles = []
    points = np.random.uniform(low=-1.0, high=1.0, size=(count_points, d))
    for _ in range(count_vectors):
        points4 = points[np.random.choice(points.shape[0], 4, replace=False), :]
        v1 = to_vector(*points4[:2])
        v2 = to_vector(*points4[2:])
        angles.append(get_angle(v1, v2))
    s = pd.Series(data=angles)
    return s.mean(), s.std()

def to_vector(point1, point2):
    return point1 - point2

def get_angle(vector1, vector2):
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    return np.arccos(unit_vector1@unit_vector2)*180/np.pi

def show_results(all_angles):
    df = pd.DataFrame(all_angles, columns=['dimensions', 'mean_angle', 'std_angle'])
    gp = df.groupby("dimensions").agg([np.mean, double_std])
    gp["mean_angle"].plot(y='mean', yerr='double_std', capsize=5, fmt='.--', grid=True,
                          title='mean angle between vectors', ylabel='mean angle [degrees]')
    plt.ylim((0, 180))
    plt.show()
    gp["std_angle"].plot(y='mean', yerr='double_std', capsize=5, fmt='.--', grid=True,
                         title='std of angles between vectors', ylabel='angle std [degrees]')
    plt.show()

all_angles = []
for i in range(10):
    for dim in range(2, 100, 5):
        all_angles.append([dim, *sample_vectors_angle(100,1000, dim)])
show_results(all_angles)

