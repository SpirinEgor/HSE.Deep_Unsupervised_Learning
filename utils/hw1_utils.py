from itertools import product
from os.path import join
from typing import Tuple, Callable

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from utils.pr1_utils import save_training
from utils.utils import get_data_dir, load_pickled_data, show_samples, save_training_plot


def q1_a_sample_data(image_file: str, n: int, d: int) -> Tuple[np.ndarray, np.ndarray]:
    im = Image.open(image_file).resize((d, d)).convert("L")
    im = np.array(im).astype("float32")
    dist = im / im.sum()

    pairs = list(product(range(d), range(d)))
    indices = np.random.choice(len(pairs), size=n, replace=True, p=dist.reshape(-1))
    samples = [pairs[i] for i in indices]

    return dist, np.array(samples)


def prepare_q1_a_data(dataset_type: int) -> Tuple[np.darray, np.ndarray, int, int]:
    data_dir = get_data_dir(1)
    if dataset_type == 1:
        n, d = 10000, 25
        true_dist, data = q1_a_sample_data(join(data_dir, "smiley.jpg"), n, d)
    elif dataset_type == 2:
        n, d = 100000, 200
        true_dist, data = q1_a_sample_data(join(data_dir, "geoffrey-hinton.jpg"), n, d)
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
    return true_dist, data, n, d


def get_data_q1_a(dataset_type: int) -> np.ndarray:
    _, data, _, _ = prepare_q1_a_data(dataset_type)
    return data


def visualize_q1a_data(dataset_type: int):
    _, data, _, d = get_data_q1_a(dataset_type)
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]

    train_dist, test_dist = np.zeros((d, d)), np.zeros((d, d))
    for i in range(len(train_data)):
        train_dist[train_data[i][0], train_data[i][1]] += 1
    train_dist /= train_dist.sum()

    for i in range(len(test_data)):
        test_dist[test_data[i][0], test_data[i][1]] += 1
    test_dist /= test_dist.sum()

    print(f"Dataset {dataset_type}")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("Train Data")
    ax1.imshow(train_dist)
    ax1.axis("off")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x0")

    ax2.set_title("Test Data")
    ax2.imshow(test_dist)
    ax2.axis("off")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x0")

    plt.show()


def prepare_q1_b_data(dataset_type: int) -> Tuple[str, Tuple, np.ndarray, np.ndarray]:
    data_dir = get_data_dir(1)
    if dataset_type == 1:
        train_data, test_data = load_pickled_data(join(data_dir, "shapes.pkl"))
        img_shape = (20, 20)
        name = "Shape"
    elif dataset_type == 2:
        train_data, test_data = load_pickled_data(join(data_dir, "mnist.pkl"))
        img_shape = (28, 28)
        name = "MNIST"
    else:
        raise Exception(f"Invalid dataset type: {dataset_type}")
    return name, img_shape, train_data, test_data


def get_data_q1_b(dataset_type: int) -> np.ndarray:
    _, _, train_data, _ = prepare_q1_b_data(dataset_type)
    return train_data


def visualize_q1b_data(dataset_type: int):
    name, _, train_data, _ = prepare_q1_b_data(dataset_type)
    indices = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[indices] * 255
    show_samples(images, title=f"{name} Samples")


def q1_save_results(dataset_type: int, part: str, fn: Callable):
    if part == "a":
        true_dist, data, n, d = prepare_q1_a_data(dataset_type)
        split = int(0.8 * len(data))
        train_data, test_data = data[:split], data[split:]

        save_training(
            fn,
            train_data,
            test_data,
            d,
            dataset_type,
            f"Q2({part}) Dataset {dataset_type}",
            f"results/q2_{part}_dset{dataset_type}",
        )

    elif part == "b":
        name, img_shape, train_data, test_data = prepare_q1_b_data(dataset_type)
        train_losses, test_losses, samples = fn(train_data, test_data, img_shape, dataset_type)
        samples = samples.astype("float32") * 255
        print(f"Final Test Loss: {test_losses[-1]:.4f}")
        save_training_plot(
            train_losses,
            test_losses,
            f"Q2({part}) Dataset {dataset_type} Train Plot",
            f"results/q2_{part}_dset{dataset_type}_train_plot.png",
        )
        show_samples(samples, f"results/q2_{part}_dset{dataset_type}_samples.png")
    else:
        raise ValueError(f"Invalid part: {part}")
