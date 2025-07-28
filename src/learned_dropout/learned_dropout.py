import torch

from src import dataset_creator
from src.learned_dropout.models import MLP, ResNet
from src.learned_dropout.models_standard import MLPStandard, ResNetStandard
from src.learned_dropout.trainer import train, train_standard
from src.learned_dropout.domain import Dataset

import matplotlib.pyplot as plt
import torch

def plot_2d_tensor(x: torch.Tensor, y: torch.Tensor):
    """
    Plots a 2D tensor, coloring points based on their labels.

    Args:
        x: torch.Tensor of shape (n, 2), the points to plot.
        y: torch.Tensor of shape (n,), the labels (0 or 1).
    """
    colors = ['blue' if label == 0 else 'red' for label in y]

    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 0].numpy(), x[:, 1].numpy(), c=colors, alpha=0.6)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('2D Data Points by Class')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(5000)
    n = 1000
    n_test = 1000  # Validation set size
    true_d = 2
    noisy_d = 0
    batch_size = 200
    h_list = [50, 50]
    epochs = 5000
    k = 0.2
    lr_weights = 0.001
    lr_dropout = 0.003
    weight_decay = 0.001
    relus = True

    problem = dataset_creator.HyperXorNormal(true_d, 0.8, noisy_d, random_basis=True)
    # Generate the training dataset
    x, y = problem.generate_dataset(n)  # x: [n, d], y: [n,]
    y = y.float()  # BCEWithLogitsLoss expects float labels
    # Generate the validation dataset
    x_val, y_val = problem.generate_dataset(n_test)  # x_val: [n_test, d], y_val: [n_test,]
    y_val = y_val.float()
    # Create an instance of Dataset with these tensors
    dataset = Dataset(x, y, x_val, y_val)
    d = true_d + noisy_d

    # plot_2d_tensor(x, y)

    # model = MLP(d, n, h_list, relus=relus)
    # model = ResNet(d, n, h_list, relus=relus, layer_norm=True)
    # train(dataset, model, batch_size, epochs, k, lr_weights, lr_dropout, weight_decay, do_track=True, track_weights=False)

    # model = MLPStandard(d, h_list, relus=relus)
    # model = ResNetStandard(d, h_list, relus=relus, layer_norm=True)
    # train_standard(dataset, model, batch_size, epochs, lr_weights, weight_decay, do_track=True, track_weights=False)

