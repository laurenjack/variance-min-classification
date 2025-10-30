import copy
import torch
from torch.utils.data import DataLoader

from jl.posterior_minimizer import train, hyper_parameters, helpers
from jl.posterior_minimizer.models import Mlp
from jl.posterior_minimizer.dataset_creator import BinaryRandomAssigned

torch.manual_seed(837211)

num_classes = 2 # 64
signal_bits = 1 # 8
input_perms = 2 ** signal_bits
val_n = input_perms * 50
min_n = input_perms * 1
step_n = input_perms * 1 # 1
max_n = input_perms * 6 + 1  # 20

hp = hyper_parameters.HyperParameters(batch_size=40,
                                      epochs=30, # 300
                                      learning_rate=0.5, # 0.01
                                      momentum=0.2,
                                      weight_decay=0.0,
                                      print_epoch=False,
                                      print_batch=False)

# hps = [hp]
# hps = hyper_parameters.with_different_k(hp, [0.0, 0.2, 0.4])
hps = hyper_parameters.with_different_weight_decay(hp, [0.0, 3.0, 10.0])

num_hidden = 20 # 20


binary_random_assigned = BinaryRandomAssigned(num_classes, signal_bits, noisy_d=10, percent_correct=1.0)
# We'll always use the same test set, to keep a consistent measuring stick as we increase n
validation_set = binary_random_assigned.generate_dataset(val_n)
num_input = binary_random_assigned.num_input

sizes = [num_input]  # [num_input, num_hidden, num_hidden]
model = Mlp(sizes, num_classes, hp.is_bias)
# Take a copy of the models initial parameters, so we can re-initialize it each training run, to keep a consistent init.
initial_params = copy.deepcopy(model.state_dict())

training_run_seed = 51477

for n in range(min_n, max_n, step_n):
    # n is the current size of the train set.
    train_set = binary_random_assigned.generate_dataset(n)
    # helpers.report_patternwise_accurarices(train_set, signal_bits, num_classes)
    purity = helpers.calc_dimension_purity(train_set)
    print(f'{n}: {purity}')
    for hp in hps:
        # This is set so that each iteration goes through the same batches in the same order, reducing variance
        torch.manual_seed(training_run_seed)
        train_loader = DataLoader(train_set, batch_size=hp.batch_size)  # TODO (Make batch size a dataset param)
        test_loader = DataLoader(validation_set, batch_size=hp.batch_size)
        model.load_state_dict(initial_params)
        result = train.run(model, train_loader, test_loader, num_classes, hp)
        print(f'    wd={hp.weight_decay}: {result}')
    training_run_seed += 1




