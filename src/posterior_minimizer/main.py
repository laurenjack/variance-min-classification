from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src import hyper_parameters
from src.posterior_minimizer import data_generator

n_per_class = 5
test_n_per_class = 100
d = 3
p = 1.0


hp = hyper_parameters.HyperParameters(batch_size=40,
                                      epochs=30, # 300
                                      learning_rate=0.5, # 0.01
                                      momentum=0.2,
                                      weight_decay=0.0,
                                      print_epoch=False,
                                      print_batch=False)


x, y = data_generator.uniform_one_true_dim(n_per_class, d, p)
train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, hp.batch_size)
# The test set will allow us to evaluate whether the model is using spurious dimensions
x, y = data_generator.uniform_one_true_dim(n_per_class, d, 1.0)
model = nn.Linear(d, 1)
sigmoid_xe = nn.BCEWithLogitsLoss()

for epoch in range(hp.epochs):
