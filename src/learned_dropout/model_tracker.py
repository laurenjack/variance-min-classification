import matplotlib.pyplot as plt
import numpy as np
import torch
from abc import ABC, abstractmethod


class NullTracker:

    def update(self, model, val_acc):
        pass

    def plot(self):
        pass




class ModelTracker(ABC):
    def __init__(self, track_weights=True):
        """
        Parameters:
            track_weights (bool): If True the weight matrices are tracked; if False only the
                                  dropout keep probabilities (p values) and validation accuracy
                                  are tracked.
        """
        self.track_weights = track_weights
        # Each element is a list (one per epoch) of numpy arrays for the keep probabilities.
        self.keep_history = []
        # Each element is a list (one per epoch) of numpy arrays for the weight matrices.
        self.weight_history = []
        # Validation accuracy recorded at each epoch.
        self.val_acc_history = []

    @abstractmethod
    def update(self, model, val_acc):
        """
        Update the tracker with the current state of the model and validation accuracy.
        Should record all dropout parameters (after applying sigmoid) and, if enabled, all weight matrices.

        Parameters:
            model: The model instance (MLP or ResNet) to extract parameters from.
            val_acc: The validation accuracy (a float) for the current epoch.
        """
        pass

    @abstractmethod
    def _get_keep_titles(self):
        """
        Return a list of titles (one per dropout layer) to use when plotting keep probability histories.
        """
        pass

    @abstractmethod
    def _get_weight_titles(self):
        """
        Return a list of titles (one per weight matrix) to use when plotting weight histories.
        """
        pass

    def plot(self):
        """Generate plots for dropout keep probabilities, weight evolution (if tracked), and validation accuracy."""
        epochs = range(len(self.keep_history))
        # Plot keep probabilities for each dropout (c parameter) layer.
        keep_titles = self._get_keep_titles()
        for i in range(len(self.keep_history[0])):
            plt.figure()
            # Extract, for each epoch, the keep probabilities for layer i.
            keep_array = np.array([epoch_keep[i] for epoch_keep in self.keep_history])
            for j in range(keep_array.shape[1]):
                plt.plot(epochs, keep_array[:, j], label=f"keep[{i}][{j}]")
            plt.xlabel("Epoch")
            plt.ylabel("Keep Probability")
            plt.title(keep_titles[i])
            plt.legend()
            plt.show()

        # If weight tracking is enabled, plot each weight matrix.
        if self.track_weights and self.weight_history and self.weight_history[0]:
            weight_titles = self._get_weight_titles()
            for i in range(len(self.weight_history[0])):
                plt.figure()
                # Extract the weight matrix history for the i-th layer.
                weight_array = np.array([epoch_weights[i] for epoch_weights in self.weight_history])
                out_dim, in_dim = weight_array.shape[1], weight_array.shape[2]
                for row in range(out_dim):
                    for col in range(in_dim):
                        plt.plot(epochs, weight_array[:, row, col], label=f"w[{i}][{row},{col}]")
                plt.xlabel("Epoch")
                plt.ylabel("Weight Value")
                plt.title(weight_titles[i])
                plt.legend()
                plt.show()

        # Plot the validation accuracy (we start at epoch 1 as the first tracker update is before training).
        plt.figure()
        plt.plot(list(epochs), np.array(self.val_acc_history), label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy for Network 2")
        plt.legend()
        plt.show()


###############################################################################
# MLPTracker: Implements tracking for the MLP model.
###############################################################################

class MLPTracker(ModelTracker):
    def update(self, model, val_acc):
        """
        For an MLP model, record:
          - The keep probabilities for all dropout parameters contained in model.c_list.
          - The weight matrices for all linear layers (model.layers) if weight tracking is enabled.
          - The validation accuracy.
        """
        # Record p values computed from c parameters.
        self.keep_history.append(
            [torch.sigmoid(c.detach()).cpu().numpy() for c in model.c_list]
        )
        # Record weight matrices if weight tracking is turned on.
        if self.track_weights:
            self.weight_history.append(
                [layer.weight.detach().cpu().numpy().copy() for layer in model.layers]
            )
        else:
            self.weight_history.append([])
        self.val_acc_history.append(val_acc)

    def _get_keep_titles(self):
        """
        Return titles for the dropout layers of the MLP.
          - Index 0 is the Input Layer.
          - Subsequent indices are Hidden Layers.
        """
        titles = []
        n_layers = len(self.keep_history[0])
        for i in range(n_layers):
            if i == 0:
                titles.append("Keep Probabilities for Input Layer")
            else:
                titles.append(f"Keep Probabilities for Hidden Layer {i}")
        return titles

    def _get_weight_titles(self):
        """
        Return titles for the weight matrices of the MLP.
          - Assumes:
              • Layer 0: Input → Hidden 1,
              • Middle layers: Hidden i → Hidden i+1,
              • Final layer: Hidden → Output.
        """
        titles = []
        if self.weight_history and self.weight_history[0]:
            num_layers = len(self.weight_history[0])
            for i in range(num_layers):
                if i == 0:
                    titles.append("Weights for Linear Layer 0 (Input → Hidden 1)")
                elif i == num_layers - 1:
                    titles.append(f"Weights for Linear Layer {i} (Hidden {i} → Output)")
                else:
                    titles.append(f"Weights for Linear Layer {i} (Hidden {i} → Hidden {i + 1})")
        return titles


###############################################################################
# ResNetTracker: Implements tracking for the ResNet model.
###############################################################################

class ResNetTracker(ModelTracker):
    def update(self, model, val_acc):
        """
        For a ResNet model, record:
          - Keep probabilities from each residual block:
              For each block, record in order:
                  • c_in (input dropout for the block),
                  • c_hidden (dropout after weight_in/hidden activation),
                  • c_out (dropout after weight_out).
          - Finally, record the final layer’s input dropout parameter (c_final).
          - If weight tracking is enabled, record:
              • For each block: the weight_in and weight_out matrices,
              • The final layer's weight matrix.
          - Record the validation accuracy.
        """
        keep_list = []
        for block in model.blocks:
            keep_list.append(torch.sigmoid(block.c_in.detach()).cpu().numpy())
            keep_list.append(torch.sigmoid(block.c_hidden.detach()).cpu().numpy())
            keep_list.append(torch.sigmoid(block.c_out.detach()).cpu().numpy())
        # Append the final layer's dropout parameter.
        keep_list.append(torch.sigmoid(model.c_final.detach()).cpu().numpy())
        self.keep_history.append(keep_list)

        if self.track_weights:
            weight_list = []
            for block in model.blocks:
                weight_list.append(block.weight_in.weight.detach().cpu().numpy().copy())
                weight_list.append(block.weight_out.weight.detach().cpu().numpy().copy())
            weight_list.append(model.final_layer.weight.detach().cpu().numpy().copy())
            self.weight_history.append(weight_list)
        else:
            self.weight_history.append([])

        self.val_acc_history.append(val_acc)

    def _get_keep_titles(self):
        """
        Construct titles for each dropout parameter.
          - For each block i (starting with 1), there are three dropout titles:
                • "Block i Input Dropout"
                • "Block i Hidden Dropout"
                • "Block i Output Dropout"
          - The final dropout parameter corresponds to the final layer input:
                • "Final Layer Input Dropout"
        """
        # Determine number of blocks based on the number of dropout groups.
        n_total = len(self.keep_history[0])  # should equal 3 * num_blocks + 1
        num_blocks = (n_total - 1) // 3
        titles = []
        for i in range(num_blocks):
            titles.append(f"Block {i + 1} Input Dropout")
            titles.append(f"Block {i + 1} Hidden Dropout")
            titles.append(f"Block {i + 1} Output Dropout")
        titles.append("Final Layer Input Dropout")
        return titles

    def _get_weight_titles(self):
        """
        Construct titles for weight matrices in a ResNet:
          - For each block i, two weight matrices are recorded:
                • "Block i Weight_in (d→h)"
                • "Block i Weight_out (h→d)"
          - Finally, the final layer weight matrix:
                • "Final Layer Weight (d→1)"
        """
        titles = []
        if self.weight_history and self.weight_history[0]:
            num_weights = len(self.weight_history[0])
            # The final weight matrix is the last element; the remaining come in pairs.
            num_blocks = (num_weights - 1) // 2
            for i in range(num_blocks):
                titles.append(f"Block {i + 1} Weight_in (d→h)")
                titles.append(f"Block {i + 1} Weight_out (h→d)")
            titles.append("Final Layer Weight (d→1)")
        return titles
