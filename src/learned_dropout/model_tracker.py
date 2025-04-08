import matplotlib.pyplot as plt
import numpy as np
import torch


class ModelTracker:
    def __init__(self):
        # Each element in these lists is a list (one per layer) of numpy arrays recorded at that epoch.
        self.keep_history = []    # List of lists; each inner list holds the keep probabilities for all layers (input and hidden).
        self.weight_history = []  # List of lists; each inner list holds the weight matrices for all linear layers.
        self.bn_history = []      # List of lists; each inner list holds a dict with batch norm parameters (weight and bias) for each BN layer.
        self.val_acc_history = [] # Validation accuracy for network2 over epochs.

    def update(self, model):
        # Record current keep probabilities for each layer.
        self.keep_history.append(
            [torch.sigmoid(c.detach()).cpu().numpy() for c in model.c_list]
        )
        # Record current weight matrices for each linear layer.
        self.weight_history.append(
            [layer.weight.detach().cpu().numpy().copy() for layer in model.layers]
        )
        # # Record current batch norm parameters (affine parameters) for each BN layer.
        # self.bn_history.append([
        #     {'weight': bn.weight.detach().cpu().numpy().copy(),
        #      'bias': bn.bias.detach().cpu().numpy().copy()}
        #     for bn in model.batch_norms
        # ])

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

        # Plot weights for each linear layer.
        for i in range(len(self.weight_history[0])):
            plt.figure()
            # For each epoch, extract the weight matrix for layer i.
            weight_array = np.array([epoch_weights[i] for epoch_weights in self.weight_history])
            out_dim, in_dim = weight_array.shape[1], weight_array.shape[2]
            for row in range(out_dim):
                for col in range(in_dim):
                    plt.plot(epochs, weight_array[:, row, col], label=f"w[{i}][{row},{col}]")
            plt.xlabel("Epoch")
            plt.ylabel("Weight Value")
            if i == 0:
                plt.title("Weights for Linear Layer 0 (Input → Hidden 1)")
            elif i == len(self.weight_history[0]) - 1:
                plt.title(f"Weights for Linear Layer {i} (Hidden {i} → Output)")
            else:
                plt.title(f"Weights for Linear Layer {i} (Hidden {i} → Hidden {i+1})")
            plt.legend()
            plt.show()

        # # Plot batch norm weight parameters for each BN layer.
        # for i in range(len(self.bn_history[0])):
        #     plt.figure()
        #     # For each epoch, extract the BN weight for layer i.
        #     bn_weight_array = np.array([epoch_bn[i]['weight'] for epoch_bn in self.bn_history])
        #     for j in range(bn_weight_array.shape[0] if bn_weight_array.ndim == 1 else bn_weight_array.shape[1]):
        #         plt.plot(epochs, bn_weight_array[:, j], label=f"BN Layer {i} weight[{j}]")
        #     plt.xlabel("Epoch")
        #     plt.ylabel("BN Weight Value")
        #     plt.title(f"Batch Norm Weight Parameters for Layer {i}")
        #     plt.legend()
        #     plt.show()
        #
        # # Plot batch norm bias parameters for each BN layer.
        # for i in range(len(self.bn_history[0])):
        #     plt.figure()
        #     # For each epoch, extract the BN bias for layer i.
        #     bn_bias_array = np.array([epoch_bn[i]['bias'] for epoch_bn in self.bn_history])
        #     for j in range(bn_bias_array.shape[0] if bn_bias_array.ndim == 1 else bn_bias_array.shape[1]):
        #         plt.plot(epochs, bn_bias_array[:, j], label=f"BN Layer {i} bias[{j}]")
        #     plt.xlabel("Epoch")
        #     plt.ylabel("BN Bias Value")
        #     plt.title(f"Batch Norm Bias Parameters for Layer {i}")
        #     plt.legend()
        #     plt.show()

        # Plot validation accuracy for Network 2.
        plt.figure()
        # Note: We start plotting at epoch 1 since the first tracker update was before any training.
        plt.plot(list(epochs)[1:], np.array(self.val_acc_history), label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy for Network 2")
        plt.legend()
        plt.show()
