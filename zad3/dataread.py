from PIL import Image, ImageOps
import numpy as np
from os import listdir, path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
remembered_shape = None

def load_image_as_vector(filename):
    image = Image.open(filename)
    image_grey = ImageOps.grayscale(image)
    image_grey.thumbnail((100, 100))  # resize
    image_grey.save("grey.png")
    data = np.asarray(image_grey)
    global remembered_shape
    remembered_shape = data.shape
    return data.flatten()

def vector_to_image(vector: np.array, name="re.png", title="title"):
    in_shape = vector.reshape(remembered_shape)
    image = Image.fromarray(in_shape).convert('L')
    plt.imshow(image, cmap='Greys')
    plt.title(f"var={title}")
    plt.show()
    image.save(name)

def visualize_components(components, variances):
    print("components shape:", components.shape)
    scaler = MinMaxScaler(feature_range=(0, 255))
    scaled = scaler.fit_transform(components)
    asint = np.rint(scaled)
    for row in range(asint.shape[0]):
        vector_to_image(asint[row], "comps.png", variances[row])
    pass

def load_dataset(directory, labels_file=None):
    images = []
    for f in listdir(directory):
        image = load_image_as_vector(path.join(directory, f))
        images.append(image)
    images = np.array(images)
    print("images shape:", remembered_shape)
    if labels_file is not None:
        labels = pd.read_csv(labels_file, header=None)
        return images, np.array(labels)
    return images

def apply_pca(dataset, *args, **kwargs):
    print("dataset shape:", dataset.shape)
    pca = PCA(*args, **kwargs)
    ans = pca.fit(dataset)
    vector_to_image(ans.mean_, "m2.png")
    print(ans.explained_variance_ratio_)
    print(ans.explained_variance_ratio_.shape)
    visualize_components(ans.components_, ans.explained_variance_ratio_)



im, labels = load_dataset("cutlery", "labels.csv")
apply_pca(im)
# print(labels)
