import ssl
import torch
from torch import nn
from src import dataset_creator, splitter

# Set the seed for reproduceability
torch.manual_seed(59284)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Fix issues with ssl certificates
ssl._create_default_https_context = ssl._create_unverified_context

m = 10  # The number of models
batch_size = 40  # The batch size when training or evaluating the network
total_n = 160
num_classes = 8
noisy_d = 20
percent_correct = 1.0

binary_pattern_dataset = dataset_creator.class_pattern_with_noise(total_n, num_classes, noisy_d, percent_correct, 1)
train_loader, val_loader = splitter.train_val_split(binary_pattern_dataset, batch_size, m)

num_input = train_loader.num_input()
num_hidden = 10
num_epochs = 100
learning_rate = 0.3
momentum = 0.9
epsilon = 0.00000001

class Mlp(nn.Module):

    def __init__(self, num_input, num_hidden, num_classes, m):
        super(Mlp, self).__init__()
        networks = []
        for j in range(m):
            network = nn.Sequential(nn.Linear(num_input, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_classes))
            networks.append(network)
        self.networks = nn.ParameterList(networks)

    def forward(self, x, j):
        network = self.networks[j]
        return network(x)

mlp = Mlp(num_input, num_hidden, num_classes, m)

# Loss and optimizer
softmax = nn.Softmax(dim=1)
log_softmax = nn.LogSoftmax(dim=1)
optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate, momentum = momentum)

def eval_train_accuracy(y_train, label):
    _, predicted = torch.max(y_train, 2)
    total = y_train.size(0) * y_train.size(1)
    correct = (predicted == label).sum().item()
    return correct / total

def eval_accuracy(data_loader, model):
    with torch.no_grad():
        total = 0.0
        correct = 0.0
        for example in data_loader:
            image = example[0]
            label = example[1]
            y_bar_sum = 0.0
            for j in range(m):
                # TODO(Jack) replace with eval method
                output = model(image, j)
                y_bar = softmax(output)
                y_bar_sum += y_bar
            _, predicted = torch.max(y_bar_sum, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        return correct / total


for epoch in range(num_epochs):
    print('Epoch {}'.format(epoch))
    for image, label, train_with, train_without in train_loader:
        current_batch_size = len(label)
        indices_for_batch = torch.arange(current_batch_size, dtype=torch.int64)
        one_hot = torch.zeros(current_batch_size, num_classes)
        one_hot[indices_for_batch, label] = 1.0
        y_hats = []
        for j in range(m):
            y_hats.append(mlp(image, j))
        y_hat = torch.stack(y_hats) # Shape (m, batch_size, num_classes)
        # Separate into the models to be trained, and those to be used as variance minimizing means
        model_indices = torch.transpose(train_with, 0, 1)

        y_hat_train = y_hat[model_indices, indices_for_batch] # Shape (m // 2, batch_size, num_classes)
        loss = 0.0
        for j in range(m // 2):
            ls = log_softmax(y_hat_train[j])
            loss_per_example = one_hot * ls # + (1 - one_hot) * (1 - ls)
            loss -= torch.mean(torch.sum(loss_per_example, axis=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss.item())
        print('Train Acc: {}'.format(eval_train_accuracy(y_hat_train, label)))
    print('Val Acc: {}'.format(eval_accuracy(val_loader, mlp)))
    print('')
        # y_hat_without = y_hat[train_without]  # Shape (m // 2, batch_size, num_classes)
        # targets = torch.nn.functional.one_hot(targets, num_classes=num_classes) # Shape (batch_size, num_classes)
        # # We need to detach as to not back-propagate signals to models that are not supposed to receive them.
        # y_bar = torch.mean(y_hat_without, axis=0, keepdim=True).detach()



