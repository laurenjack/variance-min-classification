import torch
from torch import nn
import matplotlib.pyplot as plt

from src.hyper_parameters import HyperParameters


def run(model: nn.Module, image, label, hp: HyperParameters, c_values: list[float]):
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=hp.learning_rate,
                                weight_decay=hp.weight_decay,
                                momentum=hp.momentum)
    bce_with_sigmoid = nn.BCEWithLogitsLoss()

    grads = []
    regs = []
    totals = []
    w0s = []
    w2s = []
    for c in c_values:
        # Update the weight
        model.weight_mag.data = torch.tensor(c)
        logits = model(image)
        loss = bce_with_sigmoid(logits[:, 0], label)  # + 0.1 * model.weight_mag
        optimizer.zero_grad()
        loss.backward()
        grad = model.weight_mag.grad.item()
        grad_w = model.w.grad
        grad_w0 = grad_w[0, 0].item()
        grad_w2 = grad_w[0, 2].item()
        reg = hp.post_constant
        total = grad + reg
        grads.append(grad)
        regs.append(reg)
        totals.append(total)
        w0s.append(grad_w0)
        w2s.append(grad_w2)

    plt.figure(figsize=(10, 6))  # Creates a new figure with a specified size

    # Plotting both grads and regs against c_values
    plt.plot(c_values, grads, label='Grads', marker='o')  # Adds a line plot for grads
    plt.plot(c_values, regs, label='Regs', marker='x') # Adds a line plot for regs
    plt.plot(c_values, totals, label='Totals')
    plt.plot(c_values, w0s, label='w0_grads')
    plt.plot(c_values, w2s, label='w2_grads')

    plt.xlabel('C Values')  # X-axis label
    plt.ylabel('Values')  # Y-axis label
    plt.title('Grads and Regs vs C Values')  # Chart title
    plt.legend()  # Adds a legend to specify which line corresponds to which dataset

    plt.grid(True)  # Adds a grid for easier reading
    plt.tight_layout()  # Adjusts subplot params so that the subplot(s) fits in to the figure area
    plt.show()


