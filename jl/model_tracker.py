import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class ResnetTracker:
    def __init__(self, track_weights=True, num_layers=0, has_down_rank_layer=False, learnable_norm_parameters=True):
        """
        Parameters:
            track_weights (bool): If True the weight matrices are tracked; if False only the
                                  accuracy is tracked.
            num_layers (int): Number of residual blocks in the model.
            has_down_rank_layer (bool): Whether the model has a down-rank layer.
            learnable_norm_parameters (bool): Whether norm parameters are learnable (if False, skip norm plots).
        """
        self.track_weights = track_weights
        self.num_layers = num_layers
        self.has_down_rank_layer = has_down_rank_layer
        self.learnable_norm_parameters = learnable_norm_parameters
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
        For a Resnet, record:
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
            
            # Track RMS norm weights
            norm_weights = []
            
            # Track RMS norm weights from each residual block
            for block in model.blocks:
                norm_weights.append(block.rms_norm.weight.detach().cpu().numpy().copy())
            
            # Track final RMS norm weights
            norm_weights.append(model.final_rms_norm.weight.detach().cpu().numpy().copy())
            
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
        Return titles for the linear layer weight matrices in a Resnet.
        Distinguishes between hidden layers (blocks) and final layers (down-rank + output).
        Uses stored model structure information to correctly label layers.
        """
        titles = []
        if self.track_weights and self.weight_history and self.weight_history[0]:
            # Each residual block has weight_in and weight_out (hidden layers)
            for i in range(self.num_layers):
                titles.append(f"Hidden Layer: Block {i + 1} Weight_in (d→h)")
                titles.append(f"Hidden Layer: Block {i + 1} Weight_out (h→d)")
            
            # Down-rank layer (if exists)
            if self.has_down_rank_layer:
                titles.append("Final Layer: Down-rank (d→down_rank)")
            
            # Final output layer (always exists)
            titles.append("Final Layer: Output")
                
        return titles

    def _get_norm_titles(self):
        """
        Return titles for the layer norm weight vectors in a Resnet.
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
                        plt.plot(training_steps, weight_array[:, row, col])
                plt.xlabel("Training Step")
                plt.ylabel("Weight Value")
                plt.title(weight_titles[i])
                plt.show()

        # Plot layer norm weights if weight tracking is enabled and norms are learnable.
        if self.track_weights and self.learnable_norm_parameters and self.norm_history and self.norm_history[0]:
            norm_titles = self._get_norm_titles()
            for i in range(len(self.norm_history[0])):
                plt.figure()
                # Extract the history for the i-th norm weight vector.
                norm_array = np.array([step_norms[i] for step_norms in self.norm_history])
                # norm_array has shape (num_training_steps, dim)
                dim = norm_array.shape[1]
                for j in range(dim):
                    plt.plot(training_steps, norm_array[:, j])
                plt.xlabel("Training Step")
                plt.ylabel("Norm Weight Value")
                plt.title(norm_titles[i])
                plt.show()

        # Plot validation accuracy.
        plt.figure()
        if self.train_acc_history:
            plt.plot(training_steps[:len(self.train_acc_history)], np.array(self.train_acc_history), label="Training Accuracy", color='darkorange')
        plt.plot(training_steps, np.array(self.val_acc_history), label="Validation Accuracy", color='steelblue')
        plt.xlabel("Training Step")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.show()

        # Plot losses.
        if self.val_loss_history or self.train_loss_history:
            plt.figure()
            if self.train_loss_history:
                plt.plot(training_steps[:len(self.train_loss_history)], np.array(self.train_loss_history), label="Training Loss", color='darkorange')
            if self.val_loss_history:
                plt.plot(training_steps[:len(self.val_loss_history)], np.array(self.val_loss_history), label="Validation Loss", color='steelblue')
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.show()


class MLPTracker:
    def __init__(self, track_weights=True, num_layers=0, has_down_rank_layer=False, learnable_norm_parameters=True):
        """
        Parameters:
            track_weights (bool): If True the weight matrices are tracked; if False only the
                                  accuracy is tracked.
            num_layers (int): Number of hidden layers in the model.
            has_down_rank_layer (bool): Whether the model has a down-rank layer.
            learnable_norm_parameters (bool): Whether norm parameters are learnable (if False, skip norm plots).
        """
        self.track_weights = track_weights
        self.num_layers = num_layers
        self.has_down_rank_layer = has_down_rank_layer
        self.learnable_norm_parameters = learnable_norm_parameters
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
        For an MLP, record:
          - The weight matrices from hidden layers (two linear layers per hidden layer)
          - The weight matrix from the down-rank layer (if it exists)
          - The weight matrix from the final linear layer (model.final_layer)
          - The layer norm weights from each layer
          - The final layer norm weights
          - The validation accuracy.
          - The training accuracy (if provided).
          - The validation loss (if provided).
          - The training loss (if provided).
        """
        if self.track_weights:
            # Track linear layer weights
            linear_weights = []
            
            # Extract linear layers from hidden_linear1 and hidden_linear2
            for i in range(len(model.hidden_linear1)):
                linear_weights.append(model.hidden_linear1[i].weight.detach().cpu().numpy().copy())
                linear_weights.append(model.hidden_linear2[i].weight.detach().cpu().numpy().copy())
            
            # Track down-rank layer if it exists
            if model.down_rank_layer is not None:
                linear_weights.append(model.down_rank_layer.weight.detach().cpu().numpy().copy())
            
            # Track final layer weights
            linear_weights.append(model.final_layer.weight.detach().cpu().numpy().copy())
            
            self.weight_history.append(linear_weights)
            
            # Track RMS norm weights
            norm_weights = []
            
            # Extract norm layers from hidden_norms
            for norm in model.hidden_norms:
                if norm is not None:
                    norm_weights.append(norm.weight.detach().cpu().numpy().copy())
            
            # Track final RMS norm weights if it exists
            if model.final_rms_norm is not None:
                norm_weights.append(model.final_rms_norm.weight.detach().cpu().numpy().copy())
            
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
        Return titles for the linear layer weight matrices in an MLP.
        Uses stored model structure information to correctly label layers.
        """
        titles = []
        if self.track_weights and self.weight_history and self.weight_history[0]:
            # Hidden layers - each has 2 linear layers (Linear1 and Linear2)
            for i in range(self.num_layers):
                titles.append(f"Hidden Layer {i + 1} - Linear 1")
                titles.append(f"Hidden Layer {i + 1} - Linear 2")
            
            # Down-rank layer (if exists)
            if self.has_down_rank_layer:
                titles.append("Final Layer: Down-rank")
            
            # Final output layer (always exists)
            titles.append("Final Layer: Output")
                
        return titles

    def _get_norm_titles(self):
        """
        Return titles for the layer norm weight vectors in an MLP.
        Uses stored model structure information to correctly label norms.
        """
        titles = []
        if self.track_weights and self.norm_history and self.norm_history[0]:
            num_norms = len(self.norm_history[0])
            
            if num_norms == 0:
                return titles
            
            # Norms for hidden layers (one per hidden layer)
            for i in range(self.num_layers):
                titles.append(f"Hidden Layer {i + 1} Norm")
            
            # The last norm is the final layer norm (if it exists)
            if num_norms > self.num_layers:
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
                        plt.plot(training_steps, weight_array[:, row, col])
                plt.xlabel("Training Step")
                plt.ylabel("Weight Value")
                plt.title(weight_titles[i] if i < len(weight_titles) else f"Weight Matrix {i}")
                plt.show()

        # Plot layer norm weights if weight tracking is enabled and norms are learnable.
        if self.track_weights and self.learnable_norm_parameters and self.norm_history and self.norm_history[0]:
            norm_titles = self._get_norm_titles()
            for i in range(len(self.norm_history[0])):
                plt.figure()
                # Extract the history for the i-th norm weight vector.
                norm_array = np.array([step_norms[i] for step_norms in self.norm_history])
                # norm_array has shape (num_training_steps, dim)
                dim = norm_array.shape[1]
                for j in range(dim):
                    plt.plot(training_steps, norm_array[:, j])
                plt.xlabel("Training Step")
                plt.ylabel("Norm Weight Value")
                plt.title(norm_titles[i] if i < len(norm_titles) else f"Norm {i}")
                plt.show()

        # Plot validation accuracy.
        plt.figure()
        if self.train_acc_history:
            plt.plot(training_steps[:len(self.train_acc_history)], np.array(self.train_acc_history), label="Training Accuracy", color='darkorange')
        plt.plot(training_steps, np.array(self.val_acc_history), label="Validation Accuracy", color='steelblue')
        plt.xlabel("Training Step")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.show()

        # Plot losses.
        if self.val_loss_history or self.train_loss_history:
            plt.figure()
            if self.train_loss_history:
                plt.plot(training_steps[:len(self.train_loss_history)], np.array(self.train_loss_history), label="Training Loss", color='darkorange')
            if self.val_loss_history:
                plt.plot(training_steps[:len(self.val_loss_history)], np.array(self.val_loss_history), label="Validation Loss", color='steelblue')
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.show()


class MultiLinearTracker:
    def __init__(self, track_weights=True, num_layers=0, has_down_rank_layer=False):
        """
        Tracker for MultiLinear model that tracks only linear layer weights.
        Since MultiLinear uses RMSNorm with has_parameters=False (constant weights),
        we only track the trainable linear weights.
        
        Parameters:
            track_weights (bool): If True the weight matrices are tracked; if False only the
                                  accuracy is tracked.
            num_layers (int): Number of hidden layers in the model.
            has_down_rank_layer (bool): Whether the model has a down-rank layer.
        """
        self.track_weights = track_weights
        self.num_layers = num_layers
        self.has_down_rank_layer = has_down_rank_layer
        # Each element is a list (one per training step) of lists of numpy arrays for the linear layer weight matrices.
        self.weight_history = []
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
        For a MultiLinear model, record:
          - The weight matrices from the layers (linear layers only, skip norms)
          - The weight matrix from the down-rank layer (if it exists)
          - The weight matrix from the final linear layer (model.final_layer)
          - The validation accuracy.
          - The training accuracy (if provided).
          - The validation loss (if provided).
          - The training loss (if provided).
        """
        if self.track_weights:
            # Track linear layer weights only (skip RMSNorm layers which have constant weights)
            linear_weights = []
            
            # Extract linear layers from the Sequential module
            for layer in model.layers:
                if isinstance(layer, nn.Linear):
                    linear_weights.append(layer.weight.detach().cpu().numpy().copy())
            
            # Track down-rank layer if it exists
            if model.down_rank_layer is not None:
                linear_weights.append(model.down_rank_layer.weight.detach().cpu().numpy().copy())
            
            # Track final layer weights
            linear_weights.append(model.final_layer.weight.detach().cpu().numpy().copy())
            
            self.weight_history.append(linear_weights)
        else:
            self.weight_history.append([])
        
        self.val_acc_history.append(val_acc)
        if train_acc is not None:
            self.train_acc_history.append(train_acc)
        if val_loss is not None:
            self.val_loss_history.append(val_loss)
        if train_loss is not None:
            self.train_loss_history.append(train_loss)

    def _get_weight_titles(self):
        """
        Return titles for the linear layer weight matrices in a MultiLinear model.
        Uses stored model structure information to correctly label layers.
        """
        titles = []
        if self.track_weights and self.weight_history and self.weight_history[0]:
            # Hidden layers (from self.layers Sequential)
            for i in range(self.num_layers):
                titles.append(f"Hidden Layer {i + 1}")
            
            # Down-rank layer (if exists)
            if self.has_down_rank_layer:
                titles.append("Final Layer: Down-rank")
            
            # Final output layer (always exists)
            titles.append("Final Layer: Output")
                
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
                        plt.plot(training_steps, weight_array[:, row, col])
                plt.xlabel("Training Step")
                plt.ylabel("Weight Value")
                plt.title(weight_titles[i] if i < len(weight_titles) else f"Weight Matrix {i}")
                plt.show()

        # Plot validation accuracy.
        plt.figure()
        if self.train_acc_history:
            plt.plot(training_steps[:len(self.train_acc_history)], np.array(self.train_acc_history), label="Training Accuracy", color='darkorange')
        plt.plot(training_steps, np.array(self.val_acc_history), label="Validation Accuracy", color='steelblue')
        plt.xlabel("Training Step")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.show()

        # Plot losses.
        if self.val_loss_history or self.train_loss_history:
            plt.figure()
            if self.train_loss_history:
                plt.plot(training_steps[:len(self.train_loss_history)], np.array(self.train_loss_history), label="Training Loss", color='darkorange')
            if self.val_loss_history:
                plt.plot(training_steps[:len(self.val_loss_history)], np.array(self.val_loss_history), label="Validation Loss", color='steelblue')
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.show()


class PolynomialTracker:
    def __init__(self, track_weights=True):
        """
        Parameters:
            track_weights (bool): If True the coefficient matrix is tracked; if False only the
                                  accuracy is tracked.
        """
        self.track_weights = track_weights
        # Each element is a numpy array of the coefficient matrix at that training step.
        self.weight_history = []
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
        For a KPolynomial model, record:
          - The coefficient matrix (shape: d x k)
          - The validation accuracy.
          - The training accuracy (if provided).
          - The validation loss (if provided).
          - The training loss (if provided).
        """
        if self.track_weights:
            # Track the coefficient matrix
            coeff_matrix = model.coefficients.detach().cpu().numpy().copy()
            self.weight_history.append(coeff_matrix)
        else:
            self.weight_history.append(None)
        
        self.val_acc_history.append(val_acc)
        if train_acc is not None:
            self.train_acc_history.append(train_acc)
        if val_loss is not None:
            self.val_loss_history.append(val_loss)
        if train_loss is not None:
            self.train_loss_history.append(train_loss)

    def _get_weight_titles(self):
        """
        Return title for the polynomial coefficient matrix.
        """
        if self.track_weights and self.weight_history and self.weight_history[0] is not None:
            d, k = self.weight_history[0].shape
            return [f"Polynomial Coefficients (d={d}, k={k})"]
        return []

    def plot(self):
        """
        Plot:
          1. Validation and training accuracy vs training steps.
          2. Validation and training loss vs training steps.
          3. Evolution of polynomial coefficients (Frobenius norm) vs training steps.
        """
        training_steps = np.arange(len(self.val_acc_history))
        
        # Plot accuracies.
        plt.figure()
        plt.plot(training_steps, np.array(self.val_acc_history), label="Validation Accuracy", color='steelblue')
        if self.train_acc_history:
            plt.plot(training_steps[:len(self.train_acc_history)], np.array(self.train_acc_history), label="Training Accuracy", color='darkorange')
        plt.xlabel("Training Step")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.show()
        
        # Plot losses.
        if self.val_loss_history or self.train_loss_history:
            plt.figure()
            if self.train_loss_history:
                plt.plot(training_steps[:len(self.train_loss_history)], np.array(self.train_loss_history), label="Training Loss", color='darkorange')
            if self.val_loss_history:
                plt.plot(training_steps[:len(self.val_loss_history)], np.array(self.val_loss_history), label="Validation Loss", color='steelblue')
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.show()
        
        # Plot coefficient matrix evolution (Frobenius norm).
        if self.track_weights and self.weight_history:
            valid_history = [w for w in self.weight_history if w is not None]
            if valid_history:
                plt.figure()
                frobenius_norms = [np.linalg.norm(w, 'fro') for w in valid_history]
                plt.plot(training_steps[:len(frobenius_norms)], frobenius_norms, color='green')
                plt.xlabel("Training Step")
                plt.ylabel("Frobenius Norm")
                plt.title("Polynomial Coefficient Matrix Frobenius Norm")
                plt.show()
