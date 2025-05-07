import torch

from src import dataset_creator
from src.learned_dropout.models import MLP, ResNet
from src.learned_dropout.models_standard import MLPStandard, ResNetStandard
from src.learned_dropout.trainer import train, train_standard
from src.learned_dropout.domain import Dataset


# torch.manual_seed(3997)
n = 1500
n_test = 100  # Validation set size
d = 20
batch_size = 200
h_list = [20, 20]
epochs = 1000
k = 0.25
lr_weights = 0.01
weight_decay = 0.01
relus = True
num_runs = 30

problem = dataset_creator.Gaussian(d)
    # Generate the validation dataset
x_val, y_val = problem.generate_dataset(n_test)  # x_val: [n_test, d], y_val: [n_test,]
y_val = y_val.float()

# Multiple training runs
num_layers = len(h_list) + 1  # if you have h_list hidden layers + final layer
# initialize accumulators
sum_w      = [0.0 for _ in range(num_layers)]
sum_w2     = [0.0 for _ in range(num_layers)]
sample_per = [None for _ in range(num_layers)]  # to record # weights per layer

for run in range(num_runs):
    print(f'Run: {run+1}')
    # 1) generate new data + model
    x, y = problem.generate_dataset(n)       # x: [n,d], y: [n,]
    y = y.float()
    dataset = Dataset(x, y, x_val, y_val)

    model = MLPStandard(d, h_list, relus=relus, layer_norm=True)
    train_standard(
        dataset, model,
        batch_size, epochs,
        lr_weights, weight_decay,
        do_track=False, track_weights=True, print_progress=False
    )

    # 2) accumulate sums
    for l, linear in enumerate(model.layers):
        w = linear.weight.detach()      # [d_in, d_out]
        flat = w.view(-1)
        if sample_per[l] is None:
            sample_per[l] = flat.numel()
        sum_w[l]  += flat.sum().item()
        sum_w2[l] += (flat * flat).sum().item()

# 3) final mean & variance
for l in range(num_layers):
    N = sample_per[l] * num_runs
    mean_all = sum_w[l] / N
    var_all  = sum_w2[l]/N - mean_all**2
    print(f"Layer {l:2d}:  mean={mean_all:.6f},  var={var_all:.6f},  sample_size={N}")