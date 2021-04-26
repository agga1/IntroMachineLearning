from PIL import Image, ImageOps
import numpy as np
from os import listdir, path
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
remembered_shape = None

def load_dataset(directory, labels_file=None):
    images = []
    for f in listdir(directory):
        image = load_image_as_vector(path.join(directory, f))
        images.append(image)
    images = np.array(images)
    print("images shape:", remembered_shape)
    if labels_file is not None:
        labels = pd.read_csv(labels_file, header=None)
        return images, np.array(labels).flatten()
    return images

def load_image_as_vector(filename):
    image = Image.open(filename)
    image_grey = ImageOps.grayscale(image)
    image_grey.thumbnail((100, 100))  # resize
    image_grey.save("grey.png")
    data = np.asarray(image_grey)
    global remembered_shape
    remembered_shape = data.shape
    return data.flatten()

def vector_to_image(vector: np.array, name="re.png", title="title", show=True):
    in_shape = vector.reshape(remembered_shape)
    image = Image.fromarray(in_shape).convert('L')
    if show:
        plt.imshow(image, cmap='Greys')
        plt.title(title)
        plt.show()
    image.save(name)

def visualize_components(components, variances):
    print("components shape:", components.shape)
    scaler = MinMaxScaler(feature_range=(0, 255))
    scaled = scaler.fit_transform(components)
    scaled_int = np.rint(scaled)
    for row in range(scaled_int.shape[0]):
        vector_to_image(scaled_int[row], "comps.png", title=f"var={variances[row]}")
    pass

def apply_pca(dataset, *args, **kwargs):
    print("dataset shape:", dataset.shape)
    pca = PCA(*args, **kwargs)
    return pca.fit(dataset)

def visualize_pca(pca):
    vector_to_image(pca.mean_, "mean.png", "mean image")
    visualize_components(pca.components_, pca.explained_variance_ratio_)

def visualize_reduced(dataset, n_components, labels):
    pca = apply_pca(dataset, n_components=n_components)
    print("reduced components shape:", pca.components_.shape)
    new_images = pca.transform(dataset)
    print("reduced dataset shape:", new_images.shape)
    reconstructed = pca.inverse_transform(new_images)
    for row in range(reconstructed.shape[0]):
        vector_to_image(reconstructed[row], title=labels[row])
    print("reconstructed dataset shape:",reconstructed.shape)

label_to_ordinal = {"knife": 0, "fork": 1, "spoon": 2}
def visualize_2d(dataset, labels):
    pca = apply_pca(dataset, n_components=2)
    new_images = pca.transform(dataset)
    df = pd.DataFrame(new_images, columns=['x', 'y'])
    df['labels'] = [label_to_ordinal[label] for label in labels]
    print(df)
    plt.scatter(df.x, df.y, c=df.labels, edgecolors='black', cmap=ListedColormap(['#FF00BD', '#F2CA19', '#0000FF']))
    plt.show()
    print("reduced dataset shape:", new_images.shape)


def main():
    dataset, labels = load_dataset("cutlery", "labels.csv")
    pca = apply_pca(dataset)
    # visualize_pca(pca)
    # visualize_reduced(dataset, 4, labels)
    # visualize_reduced(dataset, 16, labels)
    visualize_2d(dataset, labels)



main()
