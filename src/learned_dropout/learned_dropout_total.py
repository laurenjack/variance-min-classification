import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src import dataset_creator
from src.learned_dropout.model_tracker import ModelTracker


def train_model(n, n_test, true_d, noisy_d, batch_size, h_list, epochs, lr_weights, lr_c, weight_decay, relus):
    # d is the total number of features.
    d = true_d + noisy_d

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

        def forward(self, x):
            """
            Forward pass with dropout.
            Returns:
              - logits: the network output.
              - dropout_masks: a list of dropout masks (one per dropout layer) used in this pass.
            The dropout masks are sampled using the current c parameters (via sigmoid) but using detached values so that
            the gradients do not update c.
            """
            dropout_masks = []
            current = x
            for i in range(len(self.layers)):
                # Compute keep probability for the current layer (detach so gradients do not flow to c).
                p = torch.sigmoid(self.c_list[i].detach())
                # Sample dropout mask.
                mask = torch.bernoulli(p.expand_as(current))
                dropout_masks.append(mask)
                current = current * mask
                current = self.layers[i](current)
                if i < len(self.layers) - 1:
                    if relus:
                        current = F.relu(current)
            logits = current.squeeze(1)
            return logits, dropout_masks

        def forward_eval(self, x):
            """
            Deterministic forward pass for evaluation.
            Instead of sampling dropout, it uses the expected value (i.e. sigmoid(c)) as a scaling factor.
            """
            current = x
            for i in range(len(self.layers)):
                p = torch.sigmoid(self.c_list[i])
                current = current * p
                current = self.layers[i](current)
                if i < len(self.layers) - 1:
                    if relus:
                        current = F.relu(current)
            logits = current.squeeze(1)
            return logits

    # Instantiate the model.
    model = MLP(d, h_list)

    # Define the loss function (we use reduction='none' to get per-sample losses).
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    # Separate parameters: weights.
    weights_params = []
    for layer in model.layers:
        weights_params += list(layer.parameters())

    # Create an AdamW optimizer for the weights.
    optimizer_weights = optim.AdamW(weights_params, lr=lr_weights, betas=(0.9, 0.999), eps=1e-8,
                                    weight_decay=weight_decay)

    # Instantiate the model tracker.
    tracker = ModelTracker()

    # Create the problem generator.
    problem = dataset_creator.HyperXorNormal(true_d, 0.8, noisy_d)
    x, y = problem.generate_dataset(n)  # x: [n, d], y: [n,]
    y = y.float()  # BCEWithLogitsLoss expects float labels
    x_val, y_val = problem.generate_dataset(n_test)  # x_val: [n_test, d], y_val: [n_test,]
    y_val = y_val.float()



    for epoch in range(epochs):
        tracker.update(model)
        # Shuffle the training data.
        permutation = torch.randperm(n)
        for i in range(0, n, batch_size):
            batch_indices = permutation[i:i + batch_size]
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]

            # Forward pass (with dropout sampling) and obtain dropout masks.
            logits, dropout_masks = model(x_batch)
            # Compute per-sample loss.
            loss_per_sample = loss_fn(logits, y_batch)  # shape: [batch_size]
            # Compute mean loss for weight update.
            loss = loss_per_sample.mean()

            optimizer_weights.zero_grad()
            loss.backward()
            optimizer_weights.step()

            # Manually update c parameters using the custom rule.
            # For each dropout layer and each node, compute:
            #   node_loss: average loss for samples where the node was active.
            #   node_anti_loss: average loss for samples where the node was dropped.
            # And update: c += lr_c * p * (1 - p) * (node_loss - node_anti_loss)
            with torch.no_grad():
                for layer_idx, mask in enumerate(dropout_masks):
                    # mask shape: [batch_size, num_nodes] for this dropout layer.
                    c_param = model.c_list[layer_idx]
                    # Compute current keep probabilities.
                    p = torch.sigmoid(c_param)
                    for node_idx in range(mask.size(1)):
                        node_mask = mask[:, node_idx]  # [batch_size]
                        active_indices = (node_mask == 1)
                        inactive_indices = (node_mask == 0)
                        if active_indices.sum() > 0:
                            node_loss = loss_per_sample[active_indices].mean()
                        else:
                            node_loss = 0.0
                        if inactive_indices.sum() > 0:
                            node_anti_loss = loss_per_sample[inactive_indices].mean()
                        else:
                            node_anti_loss = 0.0
                        update = lr_c * p[node_idx] * (1 - p[node_idx]) * (node_loss - node_anti_loss)
                        c_param[node_idx] -= update

        # Evaluate on the validation set using deterministic forward (no dropout sampling).
        model.eval()
        with torch.no_grad():
            logits_val = model.forward_eval(x_val)
            preds = (torch.sigmoid(logits_val) >= 0.5).float()
            accuracy = (preds == y_val).float().mean().item()
        tracker.val_acc_history.append(accuracy)
        model.train()

        if epoch % 10 == 0:
            with torch.no_grad():
                logits_train = model.forward_eval(x)
                preds_train = (torch.sigmoid(logits_train) >= 0.5).float()
                train_acc = (preds_train == y).float().mean().item()
            print(
                f"Epoch {epoch}: loss = {loss.item():.4f}, train_acc = {train_acc:.4f}, val_acc = {accuracy:.4f}")

    tracker.update(model)
    tracker.plot()

    return model


if __name__ == '__main__':
    torch.manual_seed(3991)
    n = 100
    n_test = 100  # Validation set size
    true_d = 2
    noisy_d = 1
    weight_decay = 0.001
    batch_size = 50
    h_list = [12]
    epochs = 500
    lr_weights = 0.01
    lr_c = 0.01  # Learning rate for updating c parameters
    train_model(n, n_test, true_d, noisy_d, batch_size, h_list, epochs, lr_weights, lr_c, weight_decay, True)
