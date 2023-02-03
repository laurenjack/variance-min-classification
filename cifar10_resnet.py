import math
import copy
import ssl
import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

torch.manual_seed(575)
BIG_BATCH = 1000
CIFAR_NUM_CLASSES = 10


def cifar10(data_dir, examples_per_class, batch_size, m):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    original_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    data_loader = DataLoader(original_dataset, batch_size=BIG_BATCH, shuffle=True)

    # Get the first BIG_BATCH images and labels
    for images, labels in data_loader:
        break

    indices = []
    # For each class get every index where it occurs
    per_class_indices_list = []
    for c in range(CIFAR_NUM_CLASSES):
        is_c = torch.eq(labels, c)
        indices_of_c = torch.nonzero(is_c)
        per_class_indices_list.append(indices_of_c)
    # Now take from each class one by one
    total_examples_per_class = examples_per_class * 3
    for i in range(total_examples_per_class):
        for c in range(CIFAR_NUM_CLASSES):
            index = per_class_indices_list[c][i]
            indices.append(index)

    balanced_images = images[indices]
    balanced_labels = labels[indices]
    balanced_dataset = TensorDataset(balanced_images, balanced_labels)

    total_n = total_examples_per_class * CIFAR_NUM_CLASSES
    n = total_n // 3
    indices = list(range(total_n))
    all_train_idx, valid_idx = indices[n:], indices[:n]

    splits = m // 2
    train_idx_list = []
    for s in range(splits):
        np.random.shuffle(all_train_idx)
        train_idx_list.append(all_train_idx[:n])
        train_idx_list.append(all_train_idx[n:])

    train_loaders = []
    for train_idx in train_idx_list:
        train_sampler = SubsetRandomSampler(train_idx)
        train_loader = DataLoader(balanced_dataset, batch_size=batch_size, sampler=train_sampler)
        train_loaders.append(train_loader)

    valid_sampler = SubsetRandomSampler(valid_idx)
    valid_loader = torch.utils.data.DataLoader(balanced_dataset, batch_size=batch_size, sampler=valid_sampler)
    return train_loaders, valid_loader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fix issues with ssl certificates
ssl._create_default_https_context = ssl._create_unverified_context


data_dir = './data'  # Where should the cifar10 data be downloaded?
m = 10  # The number of models
examples_per_class = 20  # The number of examples per class in the training set, and also the validation set
batch_size = 50  # The batch size when training or evaluating the network
num_classes = 10  # The number of classes in cifar10
num_epochs = 100
learning_rate = 0.01
n = examples_per_class * num_classes
num_batches = math.ceil(n / batch_size)
reg = 0.02
reg_power = 0.15
epsilon = 0.00000001

# Data loader for train and validation
train_loaders, valid_loader = cifar10(data_dir, examples_per_class, batch_size, m)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# base_model = ResNet(ResidualBlock, [3, 4, 6, 3])  # .to(device)
# models = [base_model]
# for j in range(m-1):
#     models.append(copy.deepcopy(base_model))
# models = [model.to(device) for model in models]
models = [ResNet(ResidualBlock, [2, 2, 2, 2]).to(device) for j in range(m)]

all_params = []  # The params from all the models in a single list, passed to the optimizer
params_lists = []  # The params from each model in their own list, used for variance minimization
for model in models:
    params = list(model.parameters())
    all_params.extend(params)
    params_lists.append(params)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(all_params, lr=learning_rate, momentum = 0.9) # weight_decay = 0.001,


def evaluate_accuracy(data_loader, models, name):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logit_sum = 0.0
            for model in models:
                outputs = model(images)
                logits = outputs.data
                logit_sum += logits
            _, predicted = torch.max(logit_sum, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} images (average): {} %'.format(name, 100 * correct / total))


mean = torch.tensor([0.0], device=device) # mean of the distribution
std = torch.tensor([1.0], device=device) # standard deviation of the distribution
normal = torch.distributions.Normal(mean, std)

t = 0
for epoch in range(num_epochs):
    print('Epoch {}'.format(epoch))
    train_iterators = [iter(train_loader) for train_loader in train_loaders]
    for b in range(num_batches):
        classification_loss = 0.0
        mean_params = [0.0] * len(params_lists[0])
        for train_iterator, model, params in zip(train_iterators, models, params_lists):
            images, labels = next(train_iterator)
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            classification_loss += loss
            # Increment the mean of each param
            for i, param in enumerate(params):
                mean_params[i] += param.detach() / m
        # increment the standard deviation of each param
        total_squared_error = [0.0] * len(params_lists[0])
        for params in params_lists:
            for i, param in enumerate(params):
                total_squared_error[i] += (param.detach() - mean_params[i]) ** 2

        d_regs = [1.0 - normal.cdf(reg_power * torch.abs(mean_param) / (mse ** 0.5 / m + epsilon))
                        for mean_param, mse in zip(mean_params, total_squared_error)]
        # d_regs = [torch.abs(mean_param) / (mse ** 0.5 / m + epsilon)
        #                 for mean_param, mse in zip(mean_params, total_squared_error)]

        reg_loss = 0
        for params in params_lists:
            for param, d_reg in zip(params, d_regs):
                reg_loss += torch.sum(reg * d_reg * torch.abs(param))
        classification_loss /= m
        reg_loss /= m
        total_loss = classification_loss + reg_loss
        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # del images, labels, outputs
        # torch.cuda.empty_cache()
        # gc.collect()

        t += 1
        print('    Iteration {}, Classification Loss: {:.4f}'
              .format(t, classification_loss.item()))
        print('    Iteration {}, Reg Loss: {:.4f}'
              .format(t, reg_loss.item()))

    # for train_loader, model in zip(train_loaders, models):
    evaluate_accuracy(train_loaders[0], [models[0]], 'train')
    evaluate_accuracy(valid_loader, models, 'validation')
