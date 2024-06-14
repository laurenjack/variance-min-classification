import torch
import matplotlib.pyplot as plt


class WeightTracker:

    def update(self, model):
        pass

    def show(self):
        pass

    def pre_reg(self, model):
        pass

    def post_reg(self, model):
        pass


class AllWeightsLinear(WeightTracker):

    def __init__(self):
        self.all_weights = []
        self.pre_reg_gradients = []
        self.post_reg_gradients = []

    def update(self, model):
        self.all_weights.append(model.linears[0].weight[0].detach().clone())

    def show(self):
        assert len(self.all_weights) == len(self.pre_reg_gradients) == len(self.post_reg_gradients)
        plot_weights_with_gradients(self.all_weights,self.pre_reg_gradients, self.post_reg_gradients, "All Linear")
        # plot_individual_weights(self.all_weights, "Linear Weights")

    def pre_reg(self, model):
        self.pre_reg_gradients.append(model.linears[0].weight.grad[0].clone())

    def post_reg(self, model):
        self.post_reg_gradients.append(model.linears[0].weight.grad[0].clone())


class SingleDimWeightTracker(WeightTracker):

    def __init__(self):
        self.hidden_weights = []
        self.single_dim_weights = []
        self.random_weights = []

    def update(self, model):
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
