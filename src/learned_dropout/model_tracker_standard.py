import matplotlib.pyplot as plt
import numpy as np
import torch


class ResNetStandardTracker:
    def __init__(self, track_weights=True):
        """
        Parameters:
            track_weights (bool): If True the weight matrices are tracked; if False only the
                                  accuracy is tracked.
        """
        self.track_weights = track_weights
        # Each element is a list (one per training step) of lists of numpy arrays for the linear layer weight matrices.
        self.weight_history = []
        # Each element is a list (one per training step) of lists of numpy arrays for the layer norm weight vectors.
        self.norm_history = []
        # Validation accuracy recorded at each training step.
        self.val_acc_history = []
        # Training accuracy recorded at each training step.
        self.train_acc_history = []
        # Validation loss recorded at each training step.
        self.val_loss_history = []
        # Training loss recorded at each training step.
        self.train_loss_history = []

    def update(self, model, val_acc, train_acc=None, val_loss=None, train_loss=None):
        """
        For a standard ResNet, record:
          - The weight matrices from each residual block (each block has two weight matrices: weight_in and weight_out)
          - The weight matrix from the down-rank layer (if it exists)
          - The weight matrix from the final linear layer (model.final_layer)
          - The layer norm weights from each residual block
          - The final layer norm weights
          - The validation accuracy.
          - The training accuracy (if provided).
          - The validation loss (if provided).
          - The training loss (if provided).
        """
        if self.track_weights:
            # Track linear layer weights
            linear_weights = []
            
            # Track weights from each residual block
            for block in model.blocks:
                linear_weights.append(block.weight_in.weight.detach().cpu().numpy().copy())
                linear_weights.append(block.weight_out.weight.detach().cpu().numpy().copy())
            
            # Track down-rank layer if it exists
            if model.down_rank_layer is not None:
                linear_weights.append(model.down_rank_layer.weight.detach().cpu().numpy().copy())
            
            # Track final layer weights
            linear_weights.append(model.final_layer.weight.detach().cpu().numpy().copy())
            
            self.weight_history.append(linear_weights)
            
            # Track layer norm weights
            norm_weights = []
            
            # Track layer norm weights from each residual block
            for block in model.blocks:
                norm_weights.append(block.layer_norm.weight.detach().cpu().numpy().copy())
            
            # Track final layer norm weights
            norm_weights.append(model.final_layer_norm.weight.detach().cpu().numpy().copy())
            
            self.norm_history.append(norm_weights)
        else:
            self.weight_history.append([])
            self.norm_history.append([])
        self.val_acc_history.append(val_acc)
        if train_acc is not None:
            self.train_acc_history.append(train_acc)
        if val_loss is not None:
            self.val_loss_history.append(val_loss)
        if train_loss is not None:
            self.train_loss_history.append(train_loss)

    def _get_weight_titles(self):
        """
        Return titles for the linear layer weight matrices in a standard ResNet.
        Distinguishes between hidden layers (blocks) and final layers (down-rank + output).
        """
        titles = []
        if self.track_weights and self.weight_history and self.weight_history[0]:
            num_weights = len(self.weight_history[0])
            
            # Look at the weight shapes to determine structure
            weight_shapes = [w.shape for w in self.weight_history[0]]
            
            # Process in pairs for blocks (these are hidden layers)
            weight_idx = 0
            block_count = 0
            
            # Each block has weight_in and weight_out (hidden layers)
            while weight_idx < len(weight_shapes) - 2:  # Leave room for potential down_rank + final
                if len(weight_shapes[weight_idx]) == 2 and len(weight_shapes[weight_idx + 1]) == 2:
                    titles.append(f"Hidden Layer: Block {block_count + 1} Weight_in (d→h)")
                    titles.append(f"Hidden Layer: Block {block_count + 1} Weight_out (h→d)")
                    block_count += 1
                    weight_idx += 2
                else:
                    break
            
            # Handle remaining weights (final layers)
            remaining = len(weight_shapes) - weight_idx
            if remaining == 2:  # down_rank + final
                titles.append("Final Layer: Down-rank (d→down_rank)")
                titles.append("Final Layer: Output (down_rank→1)")
            elif remaining == 1:  # just final
                titles.append("Final Layer: Output (d→1)")
                
        return titles

    def _get_norm_titles(self):
        """
        Return titles for the layer norm weight vectors in a standard ResNet.
        Distinguishes between hidden layer norms (blocks) and final layer norm.
        """
        titles = []
        if self.track_weights and self.norm_history and self.norm_history[0]:
            num_norms = len(self.norm_history[0])
            
            if num_norms == 0:
                return titles
            
            # All norms except the last are from blocks (hidden layers)
            for i in range(num_norms - 1):
                titles.append(f"Hidden Layer Norm: Block {i + 1}")
            
            # The last norm is the final layer norm
            titles.append("Final Layer Norm")
                
        return titles

    def plot(self):
        """Generate plots for weight evolution (if tracked) and validation accuracy."""
        training_steps = range(len(self.val_acc_history))

        # Plot linear layer weights if weight tracking is enabled.
        if self.track_weights and self.weight_history and self.weight_history[0]:
            weight_titles = self._get_weight_titles()
            for i in range(len(self.weight_history[0])):
                plt.figure()
                # Extract the history for the i-th weight matrix.
                weight_array = np.array([step_weights[i] for step_weights in self.weight_history])
                # weight_array has shape (num_training_steps, output_dim, input_dim)
                out_dim, in_dim = weight_array.shape[1], weight_array.shape[2]
                for row in range(out_dim):
                    for col in range(in_dim):
                        plt.plot(training_steps, weight_array[:, row, col],
                                 label=f"w[{row},{col}]")
                plt.xlabel("Training Step")
                plt.ylabel("Weight Value")
                plt.title(weight_titles[i])
                plt.legend()
                plt.show()

        # Plot layer norm weights if weight tracking is enabled.
        if self.track_weights and self.norm_history and self.norm_history[0]:
            norm_titles = self._get_norm_titles()
            for i in range(len(self.norm_history[0])):
                plt.figure()
                # Extract the history for the i-th norm weight vector.
                norm_array = np.array([step_norms[i] for step_norms in self.norm_history])
                # norm_array has shape (num_training_steps, dim)
                dim = norm_array.shape[1]
                for j in range(dim):
                    plt.plot(training_steps, norm_array[:, j], label=f"norm[{j}]")
                plt.xlabel("Training Step")
                plt.ylabel("Norm Weight Value")
                plt.title(norm_titles[i])
                plt.legend()
                plt.show()

        # Plot validation accuracy.
        plt.figure()
        plt.plot(training_steps, np.array(self.val_acc_history), label="Validation Accuracy")
        if self.train_acc_history:
            plt.plot(training_steps[:len(self.train_acc_history)], np.array(self.train_acc_history), label="Training Accuracy")
        plt.xlabel("Training Step")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.show()

        # Plot losses.
        if self.val_loss_history or self.train_loss_history:
            plt.figure()
            if self.val_loss_history:
                plt.plot(training_steps[:len(self.val_loss_history)], np.array(self.val_loss_history), label="Validation Loss")
            if self.train_loss_history:
                plt.plot(training_steps[:len(self.train_loss_history)], np.array(self.train_loss_history), label="Training Loss")
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.show()
