import matplotlib.pyplot as plt
import numpy as np
import torch
from abc import ABC, abstractmethod


class ModelStandardTracker(ABC):
    def __init__(self, track_weights=True):
        """
        Parameters:
            track_weights (bool): If True the weight matrices are tracked; if False only the
                                  validation accuracy is tracked.
        """
        self.track_weights = track_weights
        # Each element is a list (one per epoch) of lists of numpy arrays for the weight matrices.
        self.weight_history = []
        # Validation accuracy recorded at each epoch.
        self.val_acc_history = []

    @abstractmethod
    def update(self, model, val_acc):
        """
        Update the tracker with the current state of the model and validation accuracy.
        Should record all weight matrices (if enabled).

        Parameters:
            model: The model instance (either MLPStandard or ResNetStandard) whose parameters are to be tracked.
            val_acc: The validation accuracy (a float) for the current epoch.
        """
        pass

    @abstractmethod
    def _get_weight_titles(self):
        """
        Return a list of titles (one per weight matrix) to use when plotting the weight history.
        """
        pass

    def plot(self):
        """Generate plots for weight evolution (if tracked) and validation accuracy."""
        epochs = range(len(self.val_acc_history))

        # Plot weight matrices if weight tracking is enabled.
        if self.track_weights and self.weight_history and self.weight_history[0]:
            weight_titles = self._get_weight_titles()
            for i in range(len(self.weight_history[0])):
                plt.figure()
                # Extract the history for the i-th weight matrix.
                weight_array = np.array([epoch_weights[i] for epoch_weights in self.weight_history])
                # weight_array has shape (num_epochs, output_dim, input_dim)
                out_dim, in_dim = weight_array.shape[1], weight_array.shape[2]
                for row in range(out_dim):
                    for col in range(in_dim):
                        plt.plot(epochs, weight_array[:, row, col],
                                 label=f"w[{i}][{row},{col}]")
                plt.xlabel("Epoch")
                plt.ylabel("Weight Value")
                plt.title(weight_titles[i])
                plt.legend()
                plt.show()

        # Plot validation accuracy.
        plt.figure()
        plt.plot(epochs, np.array(self.val_acc_history), label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.legend()
        plt.show()


###############################################################################
# MLPStandardTracker: Tracks standard MLP models that have a .layers attribute.
###############################################################################

class MLPStandardTracker(ModelStandardTracker):
    def update(self, model, val_acc):
        """
        For a standard MLP, record:
          - The weight matrices from all layers (contained in model.layers) if weight tracking is enabled.
          - The validation accuracy.
        """
        if self.track_weights:
            self.weight_history.append([layer.weight.detach().cpu().numpy().copy() for layer in model.layers])
        else:
            self.weight_history.append([])
        self.val_acc_history.append(val_acc)

    def _get_weight_titles(self):
        """
        Return titles for the weight matrices for a standard MLP.
         - Assumes that layer 0 is Input → Hidden,
         - Middle layers are Hidden → Hidden,
         - The final layer is Hidden → Output.
        """
        titles = []
        if self.track_weights and self.weight_history and self.weight_history[0]:
            num_layers = len(self.weight_history[0])
            for i in range(num_layers):
                if i == 0:
                    titles.append("Weights for Linear Layer 0 (Input → Hidden)")
                elif i == num_layers - 1:
                    titles.append(f"Weights for Linear Layer {i} (Hidden → Output)")
                else:
                    titles.append(f"Weights for Linear Layer {i} (Hidden → Hidden)")
        return titles


###############################################################################
# ResNetStandardTracker: Tracks standard ResNet models that have .blocks and .final_layer.
###############################################################################

class ResNetStandardTracker(ModelStandardTracker):
    def update(self, model, val_acc):
        """
        For a standard ResNet, record:
          - The weight matrices from each residual block (each block has two weight matrices: weight_in and weight_out)
          - The weight matrix from the final linear layer (model.final_layer)
          - The validation accuracy.
        """
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

    def _get_weight_titles(self):
        """
        Return titles for the weight matrices in a standard ResNet:
          - For each block, provide a title for the weight_in and weight_out matrices.
          - One title for the final layer weight.
        """
        titles = []
        if self.track_weights and self.weight_history and self.weight_history[0]:
            num_weights = len(self.weight_history[0])
            # The final layer is the last weight; the others come in pairs.
            num_blocks = (num_weights - 1) // 2
            for i in range(num_blocks):
                titles.append(f"Block {i + 1} Weight_in (d→h)")
                titles.append(f"Block {i + 1} Weight_out (h→d)")
            titles.append("Final Layer Weight (d→1)")
        return titles
