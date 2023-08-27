import torch
from torch.utils.data import DataLoader

from src.hyper_parameters import HyperParameters
from src import dataset_creator, train_val_split, train
from src.models import Mlp

torch.manual_seed(59310 + 1)

num_input = None
num_hidden = 10
total_n = 320
num_classes = 8
noisy_d = 20
percent_correct = 1.0

# dataset, input_shape = dataset_creator.binary_class_pattern_with_noise(total_n, num_classes, noisy_d, percent_correct)
dataset, input_shape = dataset_creator.binary_random_assigned(num_classes, 4, total_n)

hp = HyperParameters(batch_size=40,
                     epochs=100,
                     learning_rate=0.02,
                     momentum=0.9,
                     weight_decay=0.01,
                     print_epoch=True)

train_set, val_set = train_val_split.on_percentage(dataset)
train_loader = DataLoader(train_set, batch_size=hp.batch_size)
val_loader = DataLoader(val_set, batch_size=hp.batch_size)

sizes = [input_shape[1], num_hidden]
model = Mlp(sizes, num_classes)

train.run(model, train_loader, val_loader, hp)
