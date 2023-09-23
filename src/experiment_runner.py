import copy
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from src import train, hyper_parameters, helpers, gradient
from src.models import Mlp
from src.dataset_creator import BinaryRandomAssigned

torch.manual_seed(837203)

num_classes = 8
signal_bits = 3
input_perms = 2 ** signal_bits
val_n = input_perms * 50
min_n = input_perms * 10
step_n = input_perms
max_n = 321

hp = hyper_parameters.HyperParameters(batch_size=40,
                                      epochs=2,
                                      learning_rate=0.05,
                                      momentum=0.9,
                                      weight_decay=0.00,
                                      gradient='xe-single',
                                      print_epoch=False,
                                      print_batch=False)

# hps = [hp]
# hps = hyper_parameters.with_different_gradients(hp)
# hps = hyper_parameters.with_different_purity_components(hp)
hps = hyper_parameters.with_different_single_calculators(hp)

num_hidden = 10


binary_random_assigned = BinaryRandomAssigned(num_classes, signal_bits, noisy_d=20, percent_correct=0.8)
# We'll always use the same test set, to keep a consistent measuring stick as we increase n
validation_set = binary_random_assigned.generate_dataset(val_n)
num_input = binary_random_assigned.num_input

sizes = [num_input]  # [num_input, num_hidden]
model = Mlp(sizes, num_classes, hp.is_bias)
# Take a copy of the models initial parameters, so we can re-initialize it each training run, to keep a consistent init.
initial_params = copy.deepcopy(model.state_dict())

training_run_seed = 51471

for n in range(min_n, max_n, step_n):
    # n is the current size of the train set.
    train_set = binary_random_assigned.generate_dataset(n)
    # helpers.report_patternwise_accurarices(train_set, signal_bits, num_classes)
    for hp in hps:
        # This is set so that each iteration goes through the same batches in the same order, reducing variance
        torch.manual_seed(training_run_seed)
        train_loader = DataLoader(train_set, batch_size=hp.batch_size)  # TODO (Make batch size a dataset param)
        test_loader = DataLoader(validation_set, batch_size=hp.batch_size)
        model.load_state_dict(initial_params)
        gradient_modifier = gradient.create_gradient(hp, model)
        result = train.run(model, train_loader, test_loader, hp, num_classes, gradient_modifier)
        # print(f'{n}, {hp.purity_components}: {result}')
        # print(f'{n}, {hp.gradient}: {result}')
        print(f'{n}, {hp.single_calculator}: {result}')
    training_run_seed += 1


