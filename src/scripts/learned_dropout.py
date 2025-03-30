import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from src import dataset_creator


class ModelTracker:
    def __init__(self):
        # Each element in these lists is a list (one per layer) of numpy arrays recorded at that epoch.
        self.keep_history = []   # List of lists; each inner list holds the keep probabilities for all layers (input and hidden).
        self.weight_history = [] # List of lists; each inner list holds the weight matrices for all linear layers.
        self.val_acc_history = []  # Validation accuracy for network2 over epochs.

    def update(self, model):
        # Record current keep probabilities for each layer.
        self.keep_history.append([torch.sigmoid(c.detach()).cpu().numpy() for c in model.c_list])
        # Record current weight matrices for each linear layer.
        self.weight_history.append([layer.weight.detach().cpu().numpy().copy() for layer in model.layers])

    def plot(self):
        epochs = range(len(self.keep_history))

        # Plot keep probabilities for each dropout parameter (layer).
        for i in range(len(self.keep_history[0])):
            plt.figure()
            # For each epoch, extract the keep probabilities for layer i.
            keep_array = np.array([epoch_keep[i] for epoch_keep in self.keep_history])
            for j in range(keep_array.shape[1]):
                plt.plot(epochs, keep_array[:, j], label=f"keep[{i}][{j}]")
            plt.xlabel("Epoch")
            plt.ylabel("Keep Probability")
            if i == 0:
                plt.title("Keep Probabilities for Input Layer")
            else:
                plt.title(f"Keep Probabilities for Hidden Layer {i}")
            plt.legend()
            plt.show()

        # # Plot weights for each linear layer.
        # for i in range(len(self.weight_history[0])):
        #     plt.figure()
        #     # For each epoch, extract the weight matrix for layer i.
        #     weight_array = np.array([epoch_weights[i] for epoch_weights in self.weight_history])
        #     out_dim, in_dim = weight_array.shape[1], weight_array.shape[2]
        #     for row in range(out_dim):
        #         for col in range(in_dim):
        #             plt.plot(epochs, weight_array[:, row, col], label=f"w[{i}][{row},{col}]")
        #     plt.xlabel("Epoch")
        #     plt.ylabel("Weight Value")
        #     if i == 0:
        #         plt.title("Weights for Linear Layer 0 (Input → Hidden 1)")
        #     elif i == len(self.weight_history[0]) - 1:
        #         plt.title(f"Weights for Linear Layer {i} (Hidden {i} → Output)")
        #     else:
        #         plt.title(f"Weights for Linear Layer {i} (Hidden {i} → Hidden {i+1})")
        #     plt.legend()
        #     plt.show()

        # Plot validation accuracy for Network 2.
        plt.figure()
        # Note: We start plotting at epoch 1 since the first tracker update was before any training.
        plt.plot(epochs[1:], np.array(self.val_acc_history), label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy for Network 2")
        plt.legend()
        plt.show()


def train_model(n, n_test, true_d, noisy_d, batch_size, h_list, epochs, k, lr_weights, lr_dropout, weight_decay, relus):
    # d is the total number of features.
    d = true_d + noisy_d
    r = 0.5

    class MLP(nn.Module):
        def __init__(self, d, h_list):
            super(MLP, self).__init__()
            L = len(h_list)
            # Build L+1 linear layers.
            self.layers = nn.ModuleList()
            # First linear layer: from input (d) to first hidden layer.
            self.layers.append(nn.Linear(d, h_list[0], bias=False))
            # Hidden layers (if L > 1).
            for i in range(1, L):
                self.layers.append(nn.Linear(h_list[i-1], h_list[i], bias=False))
            # Final linear layer: from last hidden layer to output (1).
            self.layers.append(nn.Linear(h_list[-1], 1, bias=False))
            # Create dropout parameters (c_list) for input and each hidden layer (L+1 in total).
            self.c_list = nn.ParameterList()
            self.c_list.append(nn.Parameter(torch.zeros(d)))  # For input.
            for i in range(L):
                self.c_list.append(nn.Parameter(torch.zeros(h_list[i])))

        def forward_network1(self, x, r):
            """
            Forward pass for Network 1.
            Applies dropout by sampling binary masks using keep probabilities computed from c_list (using detached versions).
            """
            n_samples = x.size(0)
            current = x
            # Iterate over each linear layer.
            for i, layer in enumerate(self.layers):
                # Compute keep probability p for layer i.
                p = (1 - r) * torch.sigmoid(self.c_list[i].detach()) + r * 0.5
                mask = torch.bernoulli(p.expand(n_samples, -1))
                current = current * mask
                # Apply linear transformation.
                current = layer(current)
                # Apply activation (if relus is True) for all but the final layer.
                if i < len(self.layers) - 1:
                    if relus:
                        current = F.relu(current)
            logits = current.squeeze(1)
            return logits

        def forward_network2(self, x):
            """
            Forward pass for Network 2.
            Scales the inputs and hidden activations by differentiable keep probabilities (from c_list) and uses detached weights.
            """
            current = x
            for i, layer in enumerate(self.layers):
                p = torch.sigmoid(self.c_list[i])
                current = current * p
                current = F.linear(current, layer.weight.detach())
                if i < len(self.layers) - 1:
                    if relus:
                        current = F.relu(current)
            logits = current.squeeze(1)
            return logits

        def var_network2(self, k):
            """
            Defines a variance term as k multiplied by the product of the sums of the differentiable keep probabilities.
            """
            var = k
            for c in self.c_list:
                var = var * torch.sum(torch.sigmoid(c))
            return var

    # Instantiate the model.
    model = MLP(d, h_list)

    # Define the loss function.
    criterion = nn.BCEWithLogitsLoss()

    # Separate parameters: weights and dropout parameters.
    weights_params = []
    for layer in model.layers:
        weights_params += list(layer.parameters())
    dropout_params = list(model.c_list)

    # Create two Adam optimizers.
    optimizer_weights = optim.AdamW(model.parameters(), lr=lr_weights, betas=(0.9, 0.999), eps=1e-8,
                                          weight_decay=weight_decay)
    optimizer_dropout = optim.Adam(dropout_params, lr=lr_dropout)

    # Instantiate the model tracker.
    tracker = ModelTracker()

    # Create the problem generator.
    problem = dataset_creator.HyperXorNormal(true_d, 0.8, noisy_d)
    x, y = problem.generate_dataset(n)  # x: [n, d], y: [n,]
    y = y.float()  # BCEWithLogitsLoss expects float labels
    x_val, y_val = problem.generate_dataset(n_test)  # x_val: [n_test, d], y_val: [n_test,]
    y_val = y_val.float()

    for epoch in range(epochs):
        # Update tracker at the beginning of each epoch.
        tracker.update(model)

        # Shuffle the dataset indices.
        permutation = torch.randperm(n)
        for i in range(0, n, batch_size):
            batch_indices = permutation[i:i+batch_size]
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]

            ## Network 1: update weights only.
            logits1 = model.forward_network1(x_batch, r=r)
            loss1 = criterion(logits1, y_batch)
            optimizer_weights.zero_grad()
            loss1.backward()
            optimizer_weights.step()

            ## Network 2: update dropout parameters (c_list) only.
            logits2 = model.forward_network2(x_batch)
            loss2 = criterion(logits2, y_batch)
            var_term = model.var_network2(k)
            loss2 = loss2 + var_term
            optimizer_dropout.zero_grad()
            loss2.backward()
            optimizer_dropout.step()

        # Compute validation accuracy on the full validation set for network 2.
        with torch.no_grad():
            logits_val = model.forward_network2(x_val)
            preds = (torch.sigmoid(logits_val) >= 0.5).float()
            accuracy = (preds == y_val).float().mean().item()
        tracker.val_acc_history.append(accuracy)

        if epoch % 10 == 0:
            # Compute training accuracy on the full training set using network 2.
            with torch.no_grad():
                logits_train = model.forward_network2(x)
                preds_train = (torch.sigmoid(logits_train) >= 0.5).float()
                train_acc = (preds_train == y).float().mean().item()
            print(f"Epoch {epoch}: loss1 = {loss1.item():.4f}, loss2 = {loss2.item():.4f}, train_acc = {train_acc:.4f}, val_acc = {accuracy:.4f}")

    # Final tracker update at end of training.
    tracker.update(model)
    # Plot the evolution of the keep probabilities, weights, and validation accuracy.
    tracker.plot()

    return model


if __name__ == '__main__':
    torch.manual_seed(3991)
    n = 100
    n_test = 100  # Validation set size
    true_d = 2
    noisy_d = 80
    weight_decay = 0.001
    batch_size = 25
    # Instead of a single integer h, we now set h_list. For a single hidden layer
    h_list = [24]
    epochs = 1000
    k = 0.005
    lr_weights = 0.01
    lr_dropout = 0.01
    weight_decay = 0.001
    train_model(n, n_test, true_d, noisy_d, batch_size, h_list, epochs, k, lr_weights, lr_dropout, weight_decay, True)
