import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import binom

from src import hyper_parameters
from src import custom_modules as cm
from src.train import SigmoidBxeTrainer, BoundsAsParam, DirectReg, L1

from src import dataset_creator
from src.posterior_minimizer import weight_tracker as wt

torch.manual_seed(769)  # 392841 769

n = 100
n_test = 100
noisy_d = 1
d = noisy_d + 1
desired_success_rate = 0.728
bp = (1 - desired_success_rate ** (1/noisy_d)) / 2
print(bp)
threshold_success = binom.ppf(1 - bp, n, 0.5) # - 1
small_n = n // 5
smaller_thresh = binom.ppf(1 - bp, small_n, 0.5)
smaller_constant = smaller_thresh / small_n - 0.5
post_constant = threshold_success / n - 0.5
print(f"Regularization Constant: {post_constant}")
print(f"Smaller Constant: {smaller_thresh}")

# all_noise = dataset_creator.AllNoise(num_class=2, d=d)
bra = dataset_creator.BinaryRandomAssigned(2, 1, noisy_d=noisy_d, scale_by_root_d=False)
x, y = bra.generate_dataset(n, percent_correct=0.8, shuffle=True)
# x[1,0] = 1.0
# x[2,0] = -1.0
x_test, y_test = bra.generate_dataset(n_test, shuffle=True)

hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=400,
                                      learning_rate= 0.1,
                                      momentum=0.0,
                                      weight_decay=0.0,
                                      post_constant=post_constant,
                                      gamma=0.9,
                                      single_moving=None,
                                      is_adam=True,
                                      all_linear=True,
                                      print_epoch=False,
                                      print_batch=False)

train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, hp.batch_size)
test_set = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_set, n_test)

model = cm.Mlp([d, 1], is_bias=False, all_linear=hp.all_linear)
# with torch.no_grad():
#     model.linears[0].weight.copy_(1.0 * torch.ones([1, 1], dtype=torch.float32))

trainer = SigmoidBxeTrainer()
weight_tracker = wt.AllWeightsLinear()
l1 = L1(hp.post_constant, hp.single_moving)
trainer.run(model, train_loader, test_loader, hp, direct_reg=l1, weight_tracker=weight_tracker)
y_shift = y * 2 - 1
is_same0 = (y_shift * x[:, 0]) == 1
num_same0 = is_same0.sum().item()
if num_same0 < n - num_same0:
    num_correct = n - num_same0
    is_unexplained = is_same0
else:
    num_correct = num_same0
    is_unexplained = ~is_same0


def get_same_counts(subsample_selector=None):
    same_counts = []
    if subsample_selector is not None:
        x_sub = x[subsample_selector]
        y_sub = y_shift[subsample_selector]
    else:
        x_sub = x
        y_sub = y_shift
    for j in range(noisy_d):
        xj = x_sub[:, 1 + j]
        prod = y_sub * xj
        is_same = prod == 1
        same_count = is_same.sum().item()
        same_counts.append(same_count)
    return same_counts


print(f"Same Counts All x: {get_same_counts()}")
print(f"Same Counts Unexp: {get_same_counts(is_unexplained)}")


print(model.linears[0].weight)
sigmoid = nn.Sigmoid()
print(sigmoid(model(x[0])))
weight_tracker.show()