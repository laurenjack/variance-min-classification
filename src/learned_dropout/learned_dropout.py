import torch

from src import dataset_creator
from src.learned_dropout.models import MLP, ResNet
from src.learned_dropout.models_standard import MLPStandard, ResNetStandard
from src.learned_dropout.trainer import train, train_standard

class Dataset:
    def __init__(self, x, y, x_val, y_val):
        self.x = x
        self.y = y
        self.x_val = x_val
        self.y_val = y_val

    def __iter__(self):
        # This allows unpacking like: x, y, x_val, y_val = dataset
        return iter((self.x, self.y, self.x_val, self.y_val))


if __name__ == '__main__':
    torch.manual_seed(3997)
    n = 100
    n_test = 100  # Validation set size
    true_d = 2
    noisy_d = 20
    batch_size = 25
    h_list = [16, 16]
    epochs = 10000
    k = 0.5
    lr_weights = 0.003
    lr_dropout = 0.003
    weight_decay = 0.001
    relus = True

    problem = dataset_creator.HyperXorNormal(true_d, 0.8, noisy_d)
    # Generate the training dataset
    x, y = problem.generate_dataset(n)  # x: [n, d], y: [n,]
    y = y.float()  # BCEWithLogitsLoss expects float labels
    # Generate the validation dataset
    x_val, y_val = problem.generate_dataset(n_test)  # x_val: [n_test, d], y_val: [n_test,]
    y_val = y_val.float()
    # Create an instance of Dataset with these tensors
    dataset = Dataset(x, y, x_val, y_val)
    d = true_d + noisy_d

    # model = MLP(d, n, h_list, relus=relus)
    model = ResNet(d, n, h_list, relus=relus, layer_norm=True)
    train(dataset, model, batch_size, epochs, k, lr_weights, lr_dropout, weight_decay, do_track=True, track_weights=False)

    # model = MLPStandard(d, h_list, relus=relus)
    # model = ResNetStandard(d, h_list, relus=relus, layer_norm=True)
    # train_standard(dataset, model, batch_size, epochs, lr_weights, weight_decay, track_weights=False)

