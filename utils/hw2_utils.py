from os.path import join
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from utils.hw1_utils import prepare_q1_b_data
from utils.utils import get_data_dir, load_pickled_data, save_training_plot, show_samples


def prepare_q1ab_data(dataset_type: int) -> Tuple[str, Tuple, np.ndarray, np.ndarray]:
    data_dir = get_data_dir(1)
    if dataset_type == 1:
        train_data, test_data = load_pickled_data(join(data_dir, "shapes_colored.pkl"))
        img_shape = (20, 20, 3)
        name = "Colored Shape"
    elif dataset_type == 2:
        train_data, test_data = load_pickled_data(join(data_dir, "mnist_colored.pkl"))
        img_shape = (28, 28, 3)
        name = "Colored MNIST"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    return name, img_shape, train_data, test_data


def q1ab_save_results(dataset_type, part, fn):
    _, img_shape, train_data, test_data = prepare_q1ab_data(dataset_type)

    train_losses, test_losses, samples = fn(train_data, test_data, img_shape, dataset_type)
    samples = samples.astype("float32") / 3 * 255

    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    save_training_plot(
        train_losses,
        test_losses,
        f"Q1({part}) Dataset {dataset_type} Train Plot",
        f"results/q1_{part}_dset{dataset_type}_train_plot.png",
    )
    show_samples(samples, f"results/q1_{part}_dset{dataset_type}_samples.png")


def visualize_q1a_data(dataset_type):
    name, _, train_data, test_data = prepare_q1ab_data(dataset_type)
    indices = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[indices].astype("float32") / 3 * 255
    show_samples(images, title=f"{name} Samples")


def q1ab_get_data(dataset_type):
    _, _, train_data, _ = prepare_q1ab_data(dataset_type)
    return train_data


def prepare_q1c_data(dataset_type: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple, int]:
    data_dir = get_data_dir(1)
    if dataset_type == 1:
        train_data, test_data, train_labels, test_labels = load_pickled_data(
            join(data_dir, "shapes.pkl"), include_labels=True
        )
        img_shape, n_classes = (20, 20), 4
    elif dataset_type == 2:
        train_data, test_data, train_labels, test_labels = load_pickled_data(
            join(data_dir, "mnist.pkl"), include_labels=True
        )
        img_shape, n_classes = (28, 28), 10
    else:
        raise Exception(f"Invalid dataset type: {dataset_type}")
    return train_data, train_labels, test_data, test_labels, img_shape, n_classes


def q1c_save_results(dataset_type, q1_c):
    train_data, train_labels, test_data, test_labels, img_shape, n_classes = prepare_q1c_data(dataset_type)

    train_losses, test_losses, samples = q1_c(
        train_data, train_labels, test_data, test_labels, img_shape, n_classes, dataset_type
    )
    samples = samples.astype("float32") * 255

    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    save_training_plot(
        train_losses,
        test_losses,
        f"Q1(c) Dataset {dataset_type} Train Plot",
        f"results/q1_c_dset{dataset_type}_train_plot.png",
    )
    show_samples(samples, f"results/q1_c_dset{dataset_type}_samples.png")


def q1c_get_data(dataset_type):
    train_data, train_labels, _, _, img_shape, n_classes = prepare_q1c_data(dataset_type)
    return train_data, train_labels, img_shape, n_classes


# Bonuses
def b1a_save_results(b1_a):
    _, img_shape, train_data, test_data = prepare_q1ab_data(2)
    train_losses, test_losses, samples = b1_a(train_data, test_data, img_shape)
    samples = samples.astype("float32") / 3 * 255
    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    save_training_plot(train_losses, test_losses, f"B1(a) Train Plot", f"results/b1_a_train_plot.png")
    show_samples(samples, f"results/b1_a_samples.png")


def b1b_save_results(b1_b):
    _, img_shape, train_data, test_data = prepare_q1ab_data(2)
    train_losses, test_losses, gray_samples, color_samples = b1_b(train_data, test_data, img_shape)
    gray_samples, color_samples = gray_samples.astype("float32"), color_samples.astype("float32")
    gray_samples *= 255
    gray_samples = gray_samples.repeat(3, axis=-1)
    color_samples = color_samples / 3 * 255
    samples = np.stack((gray_samples, color_samples), axis=1).reshape((-1,) + img_shape)

    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    save_training_plot(train_losses, test_losses, f"B1(b) Train Plot", f"results/b1_b_train_plot.png")
    show_samples(samples, f"results/b1_b_samples.png")


def b1ab_get_data():
    _, img_shape, train_data, _ = prepare_q1ab_data(2)
    return train_data, img_shape


def preprocess_b1c_data(data: np.ndarray) -> np.ndarray:
    data = torch.tensor(data, dtype=torch.float).permute(0, 3, 1, 2)
    data = F.interpolate(data, scale_factor=2, mode="bilinear")
    data = data.permute(0, 2, 3, 1) > 0.5
    return data.numpy().astype("uint8")


def b1c_save_results(b1_c):
    _, _, train_data, test_data = prepare_q1_b_data(2)
    train_data = preprocess_b1c_data(train_data)
    test_data = preprocess_b1c_data(test_data)

    train_losses, test_losses, samples = b1_c(train_data, test_data)
    samples = samples.astype("float32") * 255
    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    save_training_plot(train_losses, test_losses, f"B1(c) Train Plot", f"results/b1_c_train_plot.png")
    show_samples(samples, f"results/b1_c_samples.png")


def b1c_get_data():
    (
        _,
        _,
        train_data,
        _,
    ) = prepare_q1_b_data(2)
    return preprocess_b1c_data(train_data)
