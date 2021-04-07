from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

RED = (237,  28,  36)
GREEN = ( 34, 177,  76)
BLUE = ( 63,  72, 204)
color_to_class = {RED: 0, GREEN: 1, BLUE: 2}

def load_data_from_image(filename, size=None):
    image = Image.open(filename).convert('RGB')
    data = np.asarray(image)
    print("shape:", data.shape)
    all_points = []
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            color = tuple(data[x,y])
            if color in color_to_class.keys():
                new_point = [add_noise((x,y)), color_to_class[color]]
                all_points.append(new_point)
    if size is not None:
        all_points = random.sample(all_points, size)
    return to_standardized_notation(all_points)

def add_noise(point2d, noise_radius=1):
    x_noise = random.random() * noise_radius
    y_noise = random.random() * noise_radius
    return [point2d[1] + x_noise, point2d[0] + y_noise]


def to_standardized_notation(dataset):
    X = np.array([elem[0] for elem in dataset])
    Y = np.array([elem[1] for elem in dataset])
    return X,Y
