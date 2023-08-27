import copy
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from src import train_val_split, train, hyper_parameters
from src.models import Mlp
from src.dataset_creator import BinaryRandomAssigned

torch.manual_seed(837201)

num_classes = 8
signal_bits = 3
input_perms = 2 ** signal_bits
val_n = input_perms * 10
min_n = input_perms * 5
step_n = input_perms
max_n = 321

hp = hyper_parameters.HyperParameters(batch_size=40,
                                      epochs=40,
                                      learning_rate=0.05,
                                      momentum=0.9,
                                      weight_decay=0.00,
                                      gradient='purity-scaled',
                                      print_epoch=False,
                                      print_batch=False)

# hps = [hp]
hps = hyper_parameters.with_different_gradients(hp)

num_hidden = 10


binary_random_assigned = BinaryRandomAssigned(num_classes, signal_bits, noisy_d=50, percent_correct=1.0)
# We'll always use the same test set, to keep a consistent measuring stick as we increase n
test_loader = DataLoader(binary_random_assigned.generate_dataset(val_n), batch_size=hp.batch_size)
num_input = binary_random_assigned.num_input

sizes = [num_input]  # [num_input, num_hidden]
model = Mlp(sizes, num_classes)
# Take a copy of the models initial parameters, so we can re-initialize it each training run, to keep a consistent init.
initial_params = copy.deepcopy(model.state_dict())

for n in range(min_n, max_n, step_n):
    # n is the current size of the train set.
    train_set = binary_random_assigned.generate_dataset(n)
    train_loader = DataLoader(train_set, batch_size=hp.batch_size) # TODO (Make batch size a dataset param)
    for hp in hps:
        model.load_state_dict(initial_params)
        result = train.run(model, train_loader, test_loader, hp, num_classes)
        print(f'{n}, {hp.gradient}: {result}')


