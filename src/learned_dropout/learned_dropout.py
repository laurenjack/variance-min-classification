import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src import dataset_creator
from src.learned_dropout.model_tracker import ModelTracker


def train_model(n, n_test, true_d, noisy_d, batch_size, h_list, epochs, k, lr_weights, lr_dropout, weight_decay, relus):
    # d is the total number of features.
    d = true_d + noisy_d
    r = 0.0

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
                self.layers.append(nn.Linear(h_list[i - 1], h_list[i], bias=False))
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
            Batch normalization is applied before the activation.
            """
            n_samples = x.size(0)
            current = x
            for i, layer in enumerate(self.layers):
                # Compute keep probability p for layer i.
                p = (1 - r) * torch.sigmoid(self.c_list[i].detach()) + r * 0.5
                mask = torch.bernoulli(p.expand(n_samples, -1))
                current = current * mask
                current = layer(current)
                # If not the final layer, apply activation.
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
            r2 = r
            current = x
            for i, layer in enumerate(self.layers):
                p = torch.sigmoid(self.c_list[i])
                current = current * ((1 - r2) * p + r2 * 0.5)
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
            sum_cs = [torch.sum(torch.sigmoid(c)) for c in self.c_list]
            sizes = sum_cs + [1]
            param_count = 0.0
            for s0, s1 in zip(sizes[:-1], sizes[1:]):
                param_count += s0 * s1
            return k * param_count / n

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
    optimizer_weights = optim.AdamW(weights_params, lr=lr_weights, betas=(0.9, 0.999), eps=1e-8,
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
            batch_indices = permutation[i:i + batch_size]
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

        # Set the model to evaluation mode so that batch norm uses stored running statistics.
        model.eval()
        with torch.no_grad():
            logits_val = model.forward_network2(x_val)
            preds = (torch.sigmoid(logits_val) >= 0.5).float()
            accuracy = (preds == y_val).float().mean().item()
        tracker.val_acc_history.append(accuracy)
        # Return to training mode.
        model.train()

        if epoch % 10 == 0:
            # Compute training accuracy on the full training set using network 2.
            with torch.no_grad():
                logits_train = model.forward_network2(x)
                preds_train = (torch.sigmoid(logits_train) >= 0.5).float()
                train_acc = (preds_train == y).float().mean().item()
            print(
                f"Epoch {epoch}: loss1 = {loss1.item():.4f}, loss2 = {loss2.item():.4f}, train_acc = {train_acc:.4f}, val_acc = {accuracy:.4f}")

    # Final tracker update at end of training.
    tracker.update(model)
    # Plot the evolution of the keep probabilities, weights, and validation accuracy.
    tracker.plot()

    return model


if __name__ == '__main__':
    # torch.manual_seed(3994)
    n = 100
    n_test = 100  # Validation set size
    true_d = 2
    noisy_d = 20
    weight_decay = 0.0003
    batch_size = 25
    # Instead of a single integer h, we now set h_list. For a single hidden layer
    h_list = [12, 12]
    epochs = 3000
    k = 0.2
    lr_weights = 0.003
    lr_dropout = 0.003
    weight_decay = 0.001
    train_model(n, n_test, true_d, noisy_d, batch_size, h_list, epochs, k, lr_weights, lr_dropout, weight_decay, True)
