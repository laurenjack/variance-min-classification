import torch
from torch import nn as nn, optim as optim

from src.learned_dropout.model_tracker import NullTracker


def polarise_dropout_masks(resnet, magnitude: float = 100.0) -> None:
    """
    Push every learnable dropout parameter c in-place toward ±`magnitude`:

        • if  c_i > 0  ⇒  c_i ←  +magnitude   (σ(c_i) ≈ 1.00)
        • if  c_i < 0  ⇒  c_i ←  -magnitude   (σ(c_i) ≈ 0.00)
        • if  c_i == 0 ⇒  unchanged           (σ(c_i) = 0.50)

    Parameters
    ----------
    resnet     : ResNet
        The network whose dropout masks are modified **in-place**.
    magnitude  : float, default=100.0
        Absolute value used for the pushed parameters.  |10| already gives
        σ(±10) ≈ 0.99995 / 0.000045; raise it if you need them even “closer”
        to hard 0-1 behaviour.
    """
    with torch.no_grad():
        # helper applied to every individual c tensor
        def _polarise(tensor: torch.Tensor) -> None:
            pos_mask = tensor >= 0
            neg_mask = tensor < 0
            tensor[pos_mask] =  magnitude
            tensor[neg_mask] = -magnitude

        # per-block masks
        for block in resnet.blocks:
            _polarise(block.c_hidden.data)
            _polarise(block.c_out.data)

        # final layer mask
        _polarise(resnet.c_final.data)

def train(dataset, model, batch_size, epochs, k, lr_weights, lr_dropout, weight_decay, do_track=False, track_weights=False):

    # Define the loss function.
    criterion = nn.BCEWithLogitsLoss()

    # Separate parameters: weights and dropout parameters.
    weights_params = model.get_weight_params()
    dropout_params = model.get_drop_out_params()

    # Create two Adam optimizers.
    optimizer_weights = optim.AdamW(weights_params, lr=lr_weights, betas=(0.9, 0.999), eps=1e-8,
                                    weight_decay=weight_decay)
    optimizer_dropout = optim.Adam(dropout_params, lr=lr_dropout)

    # Instantiate the model tracker.
    if do_track:
        tracker = model.get_tracker(track_weights)
    else:
        tracker = NullTracker()

    x, y, x_val, y_val = dataset
    n = x.shape[0]

    polarise_dropout_masks(model, magnitude=2.0)

    dropout_noise_scale =0.0
    for epoch in range(epochs):
        # if epoch == 500:
        #     dropout_noise_scale = 2.0
        # Set the model to evaluation mode so that batch norm uses stored running statistics.
        model.eval()
        with torch.no_grad():
            logits_val = model.forward_network2(x_val)
            preds = (torch.sigmoid(logits_val) >= 0.5).float()
            accuracy = (preds == y_val).float().mean().item()
        # Update tracker at the beginning of each epoch.
        tracker.update(model, accuracy)

        # Return to training mode.
        model.train()

        # Shuffle the dataset indices.
        permutation = torch.randperm(n)

        for i in range(0, n, batch_size):
            model.randomize_dropout_biases(dropout_noise_scale)
            batch_indices = permutation[i:i + batch_size]
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]

            ## Network 1: update weights only.
            logits1 = model.forward_network1(x_batch)
            loss1 = criterion(logits1, y_batch)
            optimizer_weights.zero_grad()
            loss1.backward()
            optimizer_weights.step()

            ## Network 2: update dropout parameters (c_list) only.
            logits2 = model.forward_network2(x_batch)
            loss2 = criterion(logits2, y_batch)
            var_term = model.var_network2(k)
            loss2 = loss2 + var_term
            if epoch >= 500:
                optimizer_dropout.zero_grad()
                loss2.backward()
                optimizer_dropout.step()



        if epoch % 10 == 0:
            # Compute training accuracy on the full training set using network 2.
            with torch.no_grad():
                logits_train = model.forward_network2(x)
                preds_train = (torch.sigmoid(logits_train) >= 0.5).float()
                train_acc = (preds_train == y).float().mean().item()
            print(
                f"Epoch {epoch}: loss1 = {loss1.item():.4f}, loss2 = {loss2.item():.4f}, var term = {var_term.item():.4f}, train_acc = {train_acc:.4f}, val_acc = {accuracy:.4f}")

    # Final tracker update at end of training.
    model.eval()
    with torch.no_grad():
        logits_val = model.forward_network2(x_val)
        preds = (torch.sigmoid(logits_val) >= 0.5).float()
        accuracy = (preds == y_val).float().mean().item()
    # Update tracker at the beginning of each epoch.
    tracker.update(model, accuracy)

    # Plot the evolution of the keep probabilities, weights, and validation accuracy.
    tracker.plot()

    return model


def train_standard(dataset, model, batch_size, epochs, lr_weights, weight_decay, do_track=False, track_weights=False, print_progress=True):
    """
    Trains a standard model (MLPStandard or ResNetStandard) that has no learned dropout parameters.
    This function uses a single AdamW optimizer on all weights and only the model.forward method.

    Parameters:
      - n: Number of training examples.
      - n_test: Number of validation examples.
      - problem: An object that supports generate_dataset(n) and returns (x, y) tensors.
      - model: A standard model instance (with a forward method) to train.
      - batch_size: Batch size used for mini-batch SGD.
      - epochs: Total number of epochs to train.
      - lr_weights: Learning rate for the optimizer.
      - weight_decay: Weight decay factor.
      - track_weights: (Boolean) A dummy parameter to match the interface.

    Returns:
      - The trained model.
    """
    # Instantiate the model tracker.
    if do_track:
        tracker = model.get_tracker(track_weights)
    else:
        tracker = NullTracker()

    # Define the loss function.
    criterion = nn.BCEWithLogitsLoss()

    # Use a single AdamW optimizer for all model parameters.
    optimizer = optim.AdamW(model.parameters(), lr=lr_weights, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=weight_decay)

    # Generate dataset.
    x, y, x_val, y_val = dataset
    n = x.shape[0]

    # Training loop.
    for epoch in range(epochs):
        # --- Evaluate on the validation set. ---
        model.eval()
        with torch.no_grad():
            logits_val = model.forward(x_val)
            preds_val = (torch.sigmoid(logits_val) >= 0.5).float()
            val_acc = (preds_val == y_val).float().mean().item()

        # Update the model tracker with the current weights and validation accuracy.
        tracker.update(model, val_acc)

        # Every 10 epochs, compute and print the training accuracy.
        if print_progress and epoch % 10 == 0:
            with torch.no_grad():
                logits_train = model.forward(x)
                preds_train = (torch.sigmoid(logits_train) >= 0.5).float()
                train_acc = (preds_train == y).float().mean().item()
            print(f"Epoch {epoch}: train_acc = {train_acc:.4f}, val_acc = {val_acc:.4f}")

        # --- Update model on mini-batches ---
        model.train()
        permutation = torch.randperm(n)
        for i in range(0, n, batch_size):
            batch_indices = permutation[i:i + batch_size]
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]

            optimizer.zero_grad()
            logits = model.forward(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    # Final evaluation on validation set.
    model.eval()
    with torch.no_grad():
        logits_val = model.forward(x_val)
        preds_val = (torch.sigmoid(logits_val) >= 0.5).float()
        final_val_acc = (preds_val == y_val).float().mean().item()
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")

    # Perform a final tracker update and plot the tracked histories.
    tracker.update(model, final_val_acc)
    tracker.plot()

    return model
