import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src import hyper_parameters
from src import custom_modules as cm
from src.train import SigmoidBxeTrainer, BoundsAsParam, DirectReg, L1

from src import dataset_creator

runs = 1000

n = 100
n_test = 100
d = 1

hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=300,
                                      learning_rate= 0.1,
                                      momentum=0.0,
                                      weight_decay=0.0,
                                      post_constant=0.5,
                                      gamma=0.9,
                                      is_adam=True,
                                      all_linear=False,
                                      print_epoch=False,
                                      print_batch=False)

non_zero_count = 0
for r in range(runs):
    all_noise = dataset_creator.AllNoise(num_class=2, d=d)
    x, y = all_noise.generate_dataset(n, shuffle=True)
    x_test, y_test = all_noise.generate_dataset(n_test, shuffle=True)
    train_set = TensorDataset(x, y)
    train_loader = DataLoader(train_set, hp.batch_size)
    test_set = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_set, n_test)

    model = cm.Mlp([d, 1], is_bias=False)
    trainer = SigmoidBxeTrainer()
    trainer.run(model, train_loader, test_loader, hp, direct_reg_constructor=L1)
    if abs(model.linears[0].weight[0,0].item()) > 0.01:
        non_zero_count += 1
    print(f"Run {r}: {non_zero_count}")

