import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from jl.config import Config


class TrackerInterface:
    """Interface for all model trackers. Can be used directly as a no-op tracker."""
    
    def update(self, model, val_acc, train_acc=None, val_loss=None, train_loss=None):
        """Record model state and metrics after a training step."""
        pass
    
    def plot(self):
        """Generate plots for tracked data."""
        pass


class BaseTracker(TrackerInterface):
    """Base class with shared functionality for all model trackers."""
    
    def __init__(self, c: Config):
        self.weight_tracker = c.weight_tracker  # 'accuracy', 'weight', or 'full_step'
        self.weight_history = []
        self.val_acc_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.train_loss_history = []
    
    def _tracks_weights(self):
        """Return True if tracking weights or full steps."""
        return self.weight_tracker in ('weight', 'full_step')

    def _get_tensor_data(self, param):
        """Get weight data or full step data depending on tracking mode."""
        if self.weight_tracker == 'weight':
            return param.detach().cpu().numpy().copy()
        elif self.weight_tracker == 'full_step':
            # 'full_step' mode tracks the optimizer step direction (Δw/lr), stored as param._step
            return param._step.detach().cpu().numpy().copy()
        return None

    def _get_tracking_suffix(self):
        """Return suffix for titles based on tracking mode."""
        return "Full Steps" if self.weight_tracker == 'full_step' else "Weights"

    def _get_y_label(self):
        """Return y-axis label for weight/step plots."""
        return "Full Step (Δw/lr)" if self.weight_tracker == 'full_step' else "Weight Value"

    def _get_norm_y_label(self):
        """Return y-axis label for norm weight/step plots."""
        return "Norm Full Step (Δw/lr)" if self.weight_tracker == 'full_step' else "Norm Weight Value"

    def _update_metrics(self, val_acc, train_acc=None, val_loss=None, train_loss=None):
        """Update accuracy and loss history."""
        self.val_acc_history.append(val_acc)
        if train_acc is not None:
            self.train_acc_history.append(train_acc)
        if val_loss is not None:
            self.val_loss_history.append(val_loss)
        if train_loss is not None:
            self.train_loss_history.append(train_loss)

    def _plot_accuracy(self, training_steps):
        """Plot training and validation accuracy."""
        plt.figure()
        if self.train_acc_history:
            plt.plot(training_steps[:len(self.train_acc_history)], np.array(self.train_acc_history), 
                     label="Training Accuracy", color='darkorange')
        plt.plot(training_steps, np.array(self.val_acc_history), label="Validation Accuracy", color='steelblue')
        plt.xlabel("Training Step")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.show()

    def _plot_loss(self, training_steps):
        """Plot training and validation loss."""
        if self.val_loss_history or self.train_loss_history:
            plt.figure()
            if self.train_loss_history:
                plt.plot(training_steps[:len(self.train_loss_history)], np.array(self.train_loss_history), 
                         label="Training Loss", color='darkorange')
            if self.val_loss_history:
                plt.plot(training_steps[:len(self.val_loss_history)], np.array(self.val_loss_history), 
                         label="Validation Loss", color='steelblue')
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.show()

    def _plot_weight_matrices(self, training_steps, titles):
        """Plot weight/step matrix evolution."""
        if not (self._tracks_weights() and self.weight_history and self.weight_history[0]):
            return
        
        y_label = self._get_y_label()
        for i in range(len(self.weight_history[0])):
            plt.figure()
            weight_array = np.array([step_weights[i] for step_weights in self.weight_history])
            out_dim, in_dim = weight_array.shape[1], weight_array.shape[2]
            for row in range(out_dim):
                for col in range(in_dim):
                    plt.plot(training_steps, weight_array[:, row, col])
            plt.xlabel("Training Step")
            plt.ylabel(y_label)
            plt.title(titles[i] if i < len(titles) else f"Weight Matrix {i}")
            plt.show()


class ResnetTracker(BaseTracker):
    def __init__(self, c: Config, num_layers: int):
        """
        Parameters:
            c: Config object containing training parameters.
            num_layers (int): Number of residual blocks in the model.
        """
        super().__init__(c)
        self.num_layers = num_layers
        self.learnable_norm_parameters = c.learnable_norm_parameters
        self.norm_history = []

    def update(self, model, val_acc, train_acc=None, val_loss=None, train_loss=None):
        """Record weights/steps, norms, and metrics for a Resnet."""
        if self._tracks_weights():
            # Track linear layer weights/steps
            linear_data = []
            for block in model.blocks:
                linear_data.append(self._get_tensor_data(block.weight_in.weight))
                linear_data.append(self._get_tensor_data(block.weight_out.weight))
            linear_data.append(self._get_tensor_data(model.final_layer.weight))
            self.weight_history.append(linear_data)
            
            # Track RMS norm weights/steps (only when learnable)
            if self.learnable_norm_parameters:
                norm_data = []
                for block in model.blocks:
                    norm_data.append(self._get_tensor_data(block.rms_norm.weight))
                norm_data.append(self._get_tensor_data(model.final_rms_norm.weight))
                self.norm_history.append(norm_data)
            else:
                self.norm_history.append([])
        else:
            self.weight_history.append([])
            self.norm_history.append([])
        
        self._update_metrics(val_acc, train_acc, val_loss, train_loss)

    def _get_weight_titles(self):
        """Return titles for the linear layer weight/step matrices."""
        titles = []
        suffix = self._get_tracking_suffix()
        if self._tracks_weights() and self.weight_history and self.weight_history[0]:
            for i in range(self.num_layers):
                titles.append(f"Hidden Layer: Block {i + 1} Weight_in (d→h) {suffix}")
                titles.append(f"Hidden Layer: Block {i + 1} Weight_out (h→d) {suffix}")
            titles.append(f"Final Layer: Output {suffix}")
        return titles

    def _get_norm_titles(self):
        """Return titles for the layer norm weight/step vectors."""
        titles = []
        suffix = self._get_tracking_suffix()
        if self._tracks_weights() and self.norm_history and self.norm_history[0]:
            num_norms = len(self.norm_history[0])
            if num_norms == 0:
                return titles
            for i in range(num_norms - 1):
                titles.append(f"Hidden Layer Norm: Block {i + 1} {suffix}")
            titles.append(f"Final Layer Norm {suffix}")
        return titles

    def plot(self):
        """Generate plots for weight/step evolution (if tracked) and metrics."""
        training_steps = range(len(self.val_acc_history))
        
        # Plot weight matrices
        self._plot_weight_matrices(training_steps, self._get_weight_titles())
        
        # Plot norm weights/steps if tracking is enabled and norms are learnable
        if self._tracks_weights() and self.learnable_norm_parameters and self.norm_history and self.norm_history[0]:
            norm_titles = self._get_norm_titles()
            norm_y_label = self._get_norm_y_label()
            for i in range(len(self.norm_history[0])):
                plt.figure()
                norm_array = np.array([step_norms[i] for step_norms in self.norm_history])
                dim = norm_array.shape[1]
                for j in range(dim):
                    plt.plot(training_steps, norm_array[:, j])
                plt.xlabel("Training Step")
                plt.ylabel(norm_y_label)
                plt.title(norm_titles[i])
                plt.show()
        
        self._plot_accuracy(training_steps)
        self._plot_loss(training_steps)


class MLPTracker(BaseTracker):
    def __init__(self, c: Config, num_layers: int):
        """
        Parameters:
            c: Config object containing training parameters.
            num_layers (int): Number of hidden layers in the model.
        """
        super().__init__(c)
        self.num_layers = num_layers
        self.learnable_norm_parameters = c.learnable_norm_parameters
        self.norm_history = []

    def update(self, model, val_acc, train_acc=None, val_loss=None, train_loss=None):
        """Record weights/steps, norms, and metrics for an MLP."""
        if self._tracks_weights():
            # Track linear layer weights/steps
            linear_data = []
            for i in range(len(model.hidden_linear1)):
                linear_data.append(self._get_tensor_data(model.hidden_linear1[i].weight))
                linear_data.append(self._get_tensor_data(model.hidden_linear2[i].weight))
            linear_data.append(self._get_tensor_data(model.final_layer.weight))
            self.weight_history.append(linear_data)
            
            # Track RMS norm weights/steps (only when learnable)
            if self.learnable_norm_parameters:
                norm_data = []
                for norm in model.hidden_norms:
                    if norm is not None:
                        norm_data.append(self._get_tensor_data(norm.weight))
                if model.final_rms_norm is not None:
                    norm_data.append(self._get_tensor_data(model.final_rms_norm.weight))
                self.norm_history.append(norm_data)
            else:
                self.norm_history.append([])
        else:
            self.weight_history.append([])
            self.norm_history.append([])
        
        self._update_metrics(val_acc, train_acc, val_loss, train_loss)

    def _get_weight_titles(self):
        """Return titles for the linear layer weight/step matrices."""
        titles = []
        suffix = self._get_tracking_suffix()
        if self._tracks_weights() and self.weight_history and self.weight_history[0]:
            for i in range(self.num_layers):
                titles.append(f"Hidden Layer {i + 1} - Linear 1 {suffix}")
                titles.append(f"Hidden Layer {i + 1} - Linear 2 {suffix}")
            titles.append(f"Final Layer: Output {suffix}")
        return titles

    def _get_norm_titles(self):
        """Return titles for the layer norm weight/step vectors."""
        titles = []
        suffix = self._get_tracking_suffix()
        if self._tracks_weights() and self.norm_history and self.norm_history[0]:
            num_norms = len(self.norm_history[0])
            if num_norms == 0:
                return titles
            for i in range(self.num_layers):
                titles.append(f"Hidden Layer {i + 1} Norm {suffix}")
            if num_norms > self.num_layers:
                titles.append(f"Final Layer Norm {suffix}")
        return titles

    def plot(self):
        """Generate plots for weight/step evolution (if tracked) and metrics."""
        training_steps = range(len(self.val_acc_history))
        
        # Plot weight matrices
        self._plot_weight_matrices(training_steps, self._get_weight_titles())
        
        # Plot norm weights/steps if tracking is enabled and norms are learnable
        if self._tracks_weights() and self.learnable_norm_parameters and self.norm_history and self.norm_history[0]:
            norm_titles = self._get_norm_titles()
            norm_y_label = self._get_norm_y_label()
            for i in range(len(self.norm_history[0])):
                plt.figure()
                norm_array = np.array([step_norms[i] for step_norms in self.norm_history])
                dim = norm_array.shape[1]
                for j in range(dim):
                    plt.plot(training_steps, norm_array[:, j])
                plt.xlabel("Training Step")
                plt.ylabel(norm_y_label)
                plt.title(norm_titles[i] if i < len(norm_titles) else f"Norm {i}")
                plt.show()
        
        self._plot_accuracy(training_steps)
        self._plot_loss(training_steps)


class MultiLinearTracker(BaseTracker):
    def __init__(self, c: Config, num_layers: int):
        """
        Tracker for MultiLinear model that tracks only linear layer weights/steps.
        Since MultiLinear uses RMSNorm with has_parameters=False (constant weights),
        we only track the trainable linear weights.
        
        Parameters:
            c: Config object containing training parameters.
            num_layers (int): Number of hidden layers in the model.
        """
        super().__init__(c)
        self.num_layers = num_layers

    def update(self, model, val_acc, train_acc=None, val_loss=None, train_loss=None):
        """Record weights/steps and metrics for a MultiLinear model."""
        if self._tracks_weights():
            # Track linear layer weights/steps only (skip RMSNorm layers which have constant weights)
            linear_data = []
            for layer in model.layers:
                if isinstance(layer, nn.Linear):
                    linear_data.append(self._get_tensor_data(layer.weight))
            linear_data.append(self._get_tensor_data(model.final_layer.weight))
            self.weight_history.append(linear_data)
        else:
            self.weight_history.append([])
        
        self._update_metrics(val_acc, train_acc, val_loss, train_loss)

    def _get_weight_titles(self):
        """Return titles for the linear layer weight/step matrices."""
        titles = []
        suffix = self._get_tracking_suffix()
        if self._tracks_weights() and self.weight_history and self.weight_history[0]:
            for i in range(self.num_layers):
                titles.append(f"Hidden Layer {i + 1} {suffix}")
            titles.append(f"Final Layer: Output {suffix}")
        return titles

    def plot(self):
        """Generate plots for weight/step evolution (if tracked) and metrics."""
        training_steps = range(len(self.val_acc_history))
        self._plot_weight_matrices(training_steps, self._get_weight_titles())
        self._plot_accuracy(training_steps)
        self._plot_loss(training_steps)


class PolynomialTracker(BaseTracker):
    def __init__(self, c: Config):
        """
        Parameters:
            c: Config object containing training parameters.
        """
        super().__init__(c)

    def update(self, model, val_acc, train_acc=None, val_loss=None, train_loss=None):
        """Record coefficient weights/steps and metrics for a KPolynomial model."""
        if self._tracks_weights():
            coeff_data = self._get_tensor_data(model.coefficients)
            self.weight_history.append(coeff_data)
        else:
            self.weight_history.append(None)
        
        self._update_metrics(val_acc, train_acc, val_loss, train_loss)

    def _get_weight_titles(self):
        """Return title for the polynomial coefficient matrix/step."""
        suffix = self._get_tracking_suffix()
        if self._tracks_weights() and self.weight_history and self.weight_history[0] is not None:
            d, k = self.weight_history[0].shape
            return [f"Polynomial Coefficients (d={d}, k={k}) {suffix}"]
        return []

    def plot(self):
        """
        Plot:
          1. Validation and training accuracy vs training steps.
          2. Validation and training loss vs training steps.
          3. Evolution of polynomial coefficients/steps (Frobenius norm) vs training steps.
        """
        training_steps = np.arange(len(self.val_acc_history))
        
        self._plot_accuracy(training_steps)
        self._plot_loss(training_steps)
        
        # Plot coefficient matrix/step evolution (Frobenius norm)
        if self._tracks_weights() and self.weight_history:
            valid_history = [w for w in self.weight_history if w is not None]
            if valid_history:
                plt.figure()
                frobenius_norms = [np.linalg.norm(w, 'fro') for w in valid_history]
                plt.plot(training_steps[:len(frobenius_norms)], frobenius_norms, color='green')
                plt.xlabel("Training Step")
                norm_label = "Full Step Frobenius Norm (Δw/lr)" if self.weight_tracker == 'full_step' else "Frobenius Norm"
                plt.ylabel(norm_label)
                title_suffix = self._get_tracking_suffix()
                plt.title(f"Polynomial Coefficient Matrix {title_suffix} Frobenius Norm")
                plt.show()


class SimpleMLPTracker(BaseTracker):
    """Tracker for SimpleMLP that only supports accuracy tracking (no weight tracking)."""

    def __init__(self, c: Config):
        super().__init__(c)

    def update(self, model, val_acc, train_acc=None, val_loss=None, train_loss=None):
        """Record metrics only (SimpleMLP doesn't support weight tracking)."""
        self._update_metrics(val_acc, train_acc, val_loss, train_loss)

    def plot(self):
        """Generate plots for accuracy and loss."""
        training_steps = range(len(self.val_acc_history))
        self._plot_accuracy(training_steps)
        self._plot_loss(training_steps)
