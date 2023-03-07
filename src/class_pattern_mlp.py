import ssl
import torch
from torch import nn
from src import dataset_creator, splitter
from torchmetrics import AveragePrecision
# from sklearn.metrics import precision_recall_curve
# import matplotlib.pyplot as plt
import torch.nn.functional as F

# Set the seed for reproduceability
torch.manual_seed(59308)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Fix issues with ssl certificates
ssl._create_default_https_context = ssl._create_unverified_context

m = 10  # The number of models
batch_size = 40  # The batch size when training or evaluating the network
total_n = 160
num_classes = 8
noisy_d = 30
percent_correct = 1.0

binary_pattern_dataset = dataset_creator.class_pattern_with_noise(total_n, num_classes, noisy_d, percent_correct, 1)
train_loader, val_loader = splitter.train_val_split(binary_pattern_dataset, batch_size, m)

num_input = train_loader.num_input()
num_hidden = 10
num_epochs = 100
learning_rate = 0.1
momentum = 0.9
epsilon = 0.00000001
is_learned = False
weight_decay = 0.01

def create_params(base_weight, base_bias):
    """Take a base_weight and a base_bias tensor, create multi-model versions of the same shape, and then return the
    original tensors plus the multi-model tensors as parameters"""
    weight_shape = [m] + list(base_weight.shape)
    bias_shape = [m] + list(base_bias.shape)
    weight = torch.zeros(*weight_shape)
    bias = torch.zeros(*bias_shape)
    return nn.Parameter(base_weight), nn.Parameter(base_bias), nn.Parameter(weight), nn.Parameter(bias)

def create_batch_norm_params(num_features):
    base_weight = torch.ones(num_features)
    base_bias = torch.zeros(num_features)
    return create_params(base_weight, base_bias)

def create_conv_params(c_in, out, k):
    scaler = 1 / (c_in * k ** 2) ** 0.5
    base_weight = 2 * (torch.rand(m, out, c_in, k, k, requires_grad=True) - 0.5) * scaler
    base_bias = 2 * (torch.rand(m, out, requires_grad=True) - 0.5) * scaler
    return create_params(base_weight, base_bias)

def create_linear_params(in_dim, out):
    scaler = 1 / in_dim ** 0.5
    base_weight = 2 * (torch.rand(out, in_dim, requires_grad=True) - 0.5)  * scaler
    base_bias = 2 * (torch.rand(out, requires_grad=True) - 0.5) * scaler
    return create_params(base_weight, base_bias)

class Linear(nn.Module):

    def __init__(self, c_in, out):
        super().__init__()
        self.base_weight, self.base_bias, self.weight, self.bias = create_linear_params(c_in, out)

    def forward(self, x, j):
        w = self.base_weight
        b = self.base_bias
        if j is not None:
            w = w.detach() + self.weight[j]
            b = b.detach() + self.bias[j]
        return F.linear(x, weight=w, bias=b)

    def reg_loss(self):
        reg = torch.sum(self.base_bias ** 2)
        if is_learned:
            w_mean = torch.mean(self.weight, axis=0, keepdim=True)
            var = torch.mean((self.weight - w_mean) ** 2, axis=0)
            reg = var.detach() * torch.abs(self.base_weight)
        return torch.sum(reg)

class Sequential(nn.Sequential):

    def forward(self, x, j):
        for module in self:
            x = module(x, j)
        return x

class Mlp(nn.Module):

    def __init__(self, num_input, num_hidden, num_classes):
        super(Mlp, self).__init__()
        self.hidden_layer = Linear(num_input, num_hidden)
        self.relu = nn.ReLU()
        self.hidden_layer2 = Linear(10, 10)
        self.relu2 = nn.ReLU()
        self.hidden_layer3 = Linear(10, 10)
        self.relu3 = nn.ReLU()
        self.output_layer = Linear(num_hidden, num_classes)

    def forward(self, x, j):
        a = self.hidden_layer(x, j)
        a = self.relu(a)
        a = self.hidden_layer2(a, j)
        a = self.relu2(a)
        a = self.hidden_layer3(a, j)
        a = self.relu3(a)
        return self.output_layer(a, j)

    def reg_loss(self):
        return self.hidden_layer.reg_loss() + self.output_layer.reg_loss() + self.hidden_layer2.reg_loss() + self.hidden_layer3.reg_loss()

mlp = Mlp(num_input, num_hidden, num_classes)

# Loss and optimizer
softmax = nn.Softmax(dim=1)
softmax_second_dim = nn.Softmax(dim=2)
log_softmax = nn.LogSoftmax(dim=1)
log_softmax_without = nn.LogSoftmax(dim=1)
optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate, weight_decay=0.00, momentum = momentum)
ap = AveragePrecision(task='multiclass', num_classes=num_classes)

def eval_train_accuracy(y_train, label):
    _, predicted = torch.max(y_train, 2)
    total = y_train.size(0) * y_train.size(1)
    correct = (predicted == label).sum().item()
    return correct / total

def eval_accuracy(data_loader, model):
    with torch.no_grad():
        total = 0.0
        correct = 0.0
        y_bar_sums = []
        targets = []
        for example in data_loader:
            image = example[0]
            label = example[1]
            # y_bar_sum = 0.0
            # for j in range(m):
            #     # TODO(Jack) replace with eval method
            #     output = model(image, j)
            #     y_bar = softmax(output)
            #     y_bar_sum += y_bar / m
            y_bar_sum = model(image, None)
            y_bar_sums.append(y_bar_sum)
            targets.append(label)
            _, predicted = torch.max(y_bar_sum, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        y_bar_sums = torch.cat(y_bar_sums)
        targets = torch.cat(targets)
        average_precision = ap(y_bar_sums, targets)
        return correct / total, average_precision, y_bar_sums, targets  #, auc


for epoch in range(num_epochs):
    print('Epoch {}'.format(epoch))
    for image, label, train_with, train_without in train_loader:
        # Grab just half the models
        # train_with = train_with[:, torch.randperm(m // 2)[: m // 4]]
        # train_without = train_without[:, torch.randperm(m // 2)[: m // 4]]
        current_batch_size = len(label)
        indices_for_batch = torch.arange(current_batch_size, dtype=torch.int64)
        one_hot = torch.zeros(current_batch_size, num_classes)
        one_hot[indices_for_batch, label] = 1.0

        # Firstly, update the base model
        base_y_hat = mlp(image, None)
        ls = log_softmax(base_y_hat)
        loss = -torch.mean(torch.sum(one_hot * ls, axis=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Now update the additional models
        y_hats = []
        for j in range(m):
            y_hats.append(mlp(image, j))

        y_hat = torch.stack(y_hats) # Shape (m, batch_size, num_classes)
        # Separate into the models to be trained, and those to be used as variance minimizing means
        train_indices = torch.transpose(train_with, 0, 1)
        without_indices = torch.transpose(train_without, 0, 1)
        y_hat_train = y_hat[train_indices, indices_for_batch] # Shape (m // 2, batch_size, num_classes)
        #y_hat_without = y_hat[without_indices, indices_for_batch] # Shape (m // 2, batch_size, num_classes)
        # y_hat_probs = softmax_second_dim(y_hat_without)
        # This must be detached, so that the gradients do not back-propagate
        # y_hat_mean = torch.mean(y_hat_probs, axis=0).detach()

        loss = 0.0
        variance = 0.0
        for j in range(m // 2):
            ls = log_softmax(y_hat_train[j])
            # ls_without = log_softmax(y_hat_without[j])
            loss -= torch.mean(torch.sum(one_hot * ls, axis=1))
            loss += weight_decay * mlp.reg_loss()
            # loss -= 5.0 * torch.mean(torch.sum(y_hat_mean * ls_without, axis=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Train Acc: {}'.format(eval_train_accuracy(y_hat_train, label)))
        # print('Without Acc: {}'.format(eval_train_accuracy(y_hat_without, label)))
        print(label[0])
        # print(y_hat_probs[:, 0, :])

    acc, average_precision, probs, targets = eval_accuracy(val_loader, mlp)
    print('Val Acc: {}'.format(acc))
    print('Average Precision: {}'.format(average_precision))
    print('')

print(mlp.hidden_layer.base_weight[0])

# fig, ax = plt.subplots()
# colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
# for i in range(num_classes):
#     precision, recall, _ = precision_recall_curve(targets == i, probs[:, i])
#     ax.plot(recall, precision, color=colors[i], label=f"Class {i}")
# ax.set_xlabel('Recall')
# ax.set_ylabel('Precision')
# ax.legend()
# plt.show()



