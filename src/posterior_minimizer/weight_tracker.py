import torch
import matplotlib.pyplot as plt

from src.hyper_parameters import DataParameters, HyperParameters
from src.posterior_minimizer import variance as v
from src.empirical import reporter


class WeightTracker:

    def update(self, model):
        pass

    def update_for_gradient(self, model, x, y):
        pass

    def show(self):
        pass

    def track_grad_at_zero(self, weight_grads, bias_grads):
        pass


def update_list(layer_lists, current_tensors, node_limit):
    for l, W in enumerate(current_tensors):
        nodes = W.shape[0]
        # Reduce nodes visualized so graph is not crazy big
        node_limit = node_limit if node_limit and node_limit < nodes else nodes
        W = W[0:node_limit]
        layer_lists[l].append(W)


class AllWeights(WeightTracker):

    def __init__(self, num_layers, node_limit=None):
        self.layers = []
        for l in range(num_layers):
            self.layers.append([])
        self.node_limit = node_limit

    def update(self, model):
        current_weights = [linear.weight.detach().clone() for linear in model.linears]
        update_list(self.layers, current_weights, self.node_limit)

    def show(self):
        plot_all_weights(self.layers, 'Weight')


class GradAtZeroTracker(WeightTracker):

    def __init__(self, num_layers, dp: DataParameters, hp: HyperParameters, variance : v.Variance, node_limit=None):
        self.node_limit = node_limit
        self.dp = dp
        self.hp = hp
        self.variance = variance
        self.grads_at_zero = []
        for l in range(num_layers):
            self.grads_at_zero.append([])
        # Global properties for tracking the maximum normalized gradient
        self.global_max = None  # will hold the single largest normalized value (a torch.Tensor scalar)
        self.global_max_grad = None  # will hold the corresponding element from weight_grads
        self.global_max_sd = None  # will hold the corresponding element from weight_sds

    def update_for_gradient(self, model, x, y):
        # Compute gradients and standard deviations
        if self.hp.implementation == 'old':
            weight_grads, _ = v.grad_at_zero(model, x, y, self.dp.percent_correct)
            weight_sds, _ = self.variance.calculate(model, x, y)
        else:
            weight_grads, weight_sds = reporter.grads_at_zero_and_regs(model, x, y, self.dp, self.hp)

        # Compute normalized absolute gradients
        normed_grads_at_zero = [
            torch.abs(grad / (sd + 1e-8))
            for grad, sd in zip(weight_grads, weight_sds)
        ]
        normed_grads_at_zero[0] = normed_grads_at_zero[0][:, 2:]

        # # Iterate over each tensor in the list to find the maximum element
        # for norm_tensor, grad_tensor, sd_tensor in zip(normed_grads_at_zero, weight_grads, weight_sds):
        #     # Flatten the tensor for easy maximum search
        #     flat_norm = norm_tensor.view(-1)
        #     max_val, max_idx = torch.max(flat_norm, dim=0)
        #
        #     # If no global max yet, or this max is larger, update global properties
        #     if (self.global_max is None) or (max_val > self.global_max):
        #         self.global_max = max_val
        #         # Get corresponding values from the grad and sd tensors.
        #         self.global_max_grad = grad_tensor.view(-1)[max_idx]
        #         self.global_max_sd = sd_tensor.view(-1)[max_idx]

        update_list(self.grads_at_zero, normed_grads_at_zero, self.node_limit)

    def show(self):
        print(f"Global Max abs normed grad: {self.global_max}")
        print(f"Grad: {self.global_max_grad}")
        print(f"SD: {self.global_max_sd}")
        first_weight = self.grads_at_zero[0][0]
        last_weight = self.grads_at_zero[0][-1]
        print(f"mean_grad_at_zero_first: {torch.mean(first_weight, dim=0)}")
        print(f"mean_grad_at_zero_last: {torch.mean(last_weight, dim=0)}")
        plot_all_weights(self.grads_at_zero, 'Gradient')


class SingleDimWeightTracker(WeightTracker):

    def __init__(self):
        self.hidden_weights = []
        self.single_dim_weights = []
        self.random_weights = []

    def update(self, model, x, y):
        self.hidden_weights.append(model.linears[1].weight[0].detach().clone())
        self.single_dim_weights.append(model.linears[0].weight[:, 0].detach().clone())
        self.random_weights.append(model.linears[0].weight[:, 1:].detach().clone())


    def show(self):
        plot_individual_weights(self.hidden_weights, 'Hidden Weights Over Time')
        plot_individual_weights(self.single_dim_weights, 'Single Dim Weights Over Time')
        plot_weight_trajectory(self.random_weights)


def plot_individual_weights(weight_list, title):
    num_weights = weight_list[0].shape[0]
    for i in range(num_weights):
        weights = [tensor[i].item() for tensor in weight_list]
        plt.plot(weights, label=f'Weight {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Weight Value')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_all_weights(layers, title):
    for l, layer in enumerate(layers):
        out_nodes, in_nodes = layer[0].shape
        for i in range(out_nodes):
            for j in range(in_nodes):
                weights = []
                for weight in layer:
                    weights.append(weight[i, j].item())
                plt.plot(weights, label=f'w {i}{j}')
        # plt.axhline(y=0, color='k', linestyle='--', label='Reference Line')
        plt.xlabel('Time')
        plt.ylabel(f'{title} Value')
        plt.title(f'Layer {l}')
        plt.legend()
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()



def plot_weights_with_gradients(weight_list, pre_grad_weight_list, post_grad_weight_list, title):
    num_weights = weight_list[0].shape[0]
    # plt.figure(figsize=(12, 6))
    plt.axhline(y=0, color='k', linestyle='--', label='Reference Line')
    for i in range(num_weights):
        weights = [tensor[i].item() for tensor in weight_list]
        pre_grad_weights = [tensor[i].item() for tensor in pre_grad_weight_list]
        post_grad_weights = [tensor[i].item() for tensor in post_grad_weight_list]

        plt.plot(weights, label=f'Original Weight {i + 1}')
        plt.plot(pre_grad_weights, label=f'Pre-Gradient Weight {i + 1}')
        plt.plot(post_grad_weights, label=f'Post-Gradient Weight {i + 1}')

    plt.xlabel('Time')
    plt.ylabel('Weight Value')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_average_weights(weight_list, title):

    avg_weights = [torch.abs(tensor).mean().item() for tensor in weight_list]
    plt.plot(avg_weights, label='Average Weight')
    plt.xlabel('Time')
    plt.ylabel('Average Weight Value')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_weight_trajectory(weights_list):
    """
    Function to plot the trajectory of the weights corresponding to the index
    of the largest absolute element along axis 1 from the last element in the weights_list.

    Parameters:
    weights_list (list of torch.Tensor): List of weight tensors over time.
    """
    if not weights_list:
        print("The weights list is empty.")
        return

    # Step 1: Take the last element of the weight list
    last_weights = weights_list[-1]

    # Step 2: Apply torch.abs
    abs_last_weights = torch.abs(last_weights)

    # Step 3: Find the index of the largest element along axis 1
    largest_indices = torch.argmax(abs_last_weights, dim=1)

    # Step 4: Graph each weight of the largest index over time
    num_indices = largest_indices.size(0)
    weight_trajectories = [[] for _ in range(num_indices)]

    for time_step, weights in enumerate(weights_list):
        for idx in range(num_indices):
            weight_trajectories[idx].append(weights[idx, largest_indices[idx]].item())

    # Plot the trajectories
    plt.figure(figsize=(10, 6))
    for idx, trajectory in enumerate(weight_trajectories):
        plt.plot(trajectory, label=f'Weight {idx+1}')

    plt.title('Weight Trajectories of the Largest Indices Over Time')
    plt.xlabel('Time')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True)
    plt.show()
