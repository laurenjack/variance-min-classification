import torch
from torch import nn
import matplotlib.pyplot as plt

from jl.posterior_minimizer.hyper_parameters import HyperParameters


def run(model: nn.Module, image, label, hp: HyperParameters, c_values: list[float]):
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=hp.learning_rate,
                                weight_decay=hp.weight_decay,
                                momentum=hp.momentum)
    bce_with_sigmoid = nn.BCEWithLogitsLoss()

    # grads = []
    # totals = []
    # regs = []
    w0s = []
    w1s = []
    w2s = []
    final_weight = model.weight.data
    normed_weight = final_weight / torch.norm(final_weight)
    for c in c_values:
        # Update the weight
        model.weight.data = normed_weight * c
        # model.weight_mag.data = torch.tensor(c)
        logits = model(image)
        loss = bce_with_sigmoid(logits[:, 0], label.float())  # + 0.1 * model.weight_mag
        optimizer.zero_grad()
        loss.backward()
        # grad = model.weight_mag.grad.item()
        # grads.append(grad)
        grad_w = model.weight.grad  # + hp.post_constant * model.weight.data / c ** 2
        grad_w += 0.1 * torch.sign(normed_weight)
        grad_w0 = grad_w[0, 0].item()
        grad_w1 = grad_w[0, 1].item()
        grad_w2 = grad_w[0, 2].item()
        # reg =  hp.post_constant * model.weight.data[0, 0] / c ** 2


        # regs.append(reg)
        # total = grad + reg
        # totals.append(total)
        w0s.append(grad_w0)
        w1s.append(grad_w1)
        w2s.append(grad_w2)

    plt.figure(figsize=(10, 6))  # Creates a new figure with a specified size

    # plt.plot(c_values, grads, label='Grads')
    # plt.plot(c_values, totals, label='Totals')
    # plt.plot(c_values, regs, label='Regs')
    plt.plot(c_values, w0s, label='w0_grads')
    plt.plot(c_values, w1s, label='w1_grads')
    plt.plot(c_values, w2s, label='w2_grads')

    plt.xlabel('C Values')  # X-axis label
    plt.ylabel('Values')  # Y-axis label
    plt.title('Grads and Regs vs C Values')  # Chart title
    plt.legend()  # Adds a legend to specify which line corresponds to which dataset

    plt.grid(True)  # Adds a grid for easier reading
    plt.tight_layout()  # Adjusts subplot params so that the subplot(s) fits in to the figure area
    plt.show()


