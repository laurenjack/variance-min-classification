import math
import copy
import ssl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

torch.manual_seed(575)
BIG_BATCH = 400
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
m = 2  # The number of models
examples_per_class = 10  # The number of examples per class in the training set, and also the validation set
batch_size = 50  # The batch size when training or evaluating the network
num_classes = 10  # The number of classes in cifar10
num_epochs = 100
learning_rate = 0.01
n = examples_per_class * num_classes
num_batches = math.ceil(n / batch_size)
reg = 0.02
reg_power = 0.15
epsilon = 0.00000001
mag_epsilon = 0.00000001

# Data loader for train and validation
train_loaders, valid_loader = cifar10(data_dir, examples_per_class, batch_size, m)

def norm2(tensor):
    return torch.sum(tensor ** 2, axis=0) ** 0.5

def create_normed_params(weight, bias):
    weight_mag = norm2(weight)
    bias_mag = norm2(bias)
    return nn.Parameter(weight), nn.Parameter(bias), nn.Parameter(weight_mag), nn.Parameter(bias_mag)


def create_batch_norm_params(num_features):
    weight = torch.ones(m, num_features)
    bias = torch.zeros(m, num_features)
    return create_normed_params(weight, bias)

def create_conv_params(c_in, out, k):
    scaler = 1 / (c_in * k ** 2) ** 0.5
    weight = 2 * (torch.rand(m, out, c_in, k, k, requires_grad=True) - 0.5) * scaler
    bias = 2 * (torch.rand(m, out, requires_grad=True) - 0.5) * scaler
    return create_normed_params(weight, bias)


def create_linear_params(in_dim, out):
    scaler = 1 / in_dim ** 0.5
    weight = 2 * (torch.rand(m, out, in_dim, requires_grad=True) - 0.5)  * scaler
    bias = 2 * (torch.rand(m, out, requires_grad=True) - 0.5) * scaler
    return create_normed_params(weight, bias)

def get_applied(param, param_magnitude, j):
    param_j = param[j]
    norm = norm2(param.detach()) # Need to detach the norm, so that gradients are not shared across datasets.
    normed_param = param_j / (norm + mag_epsilon)
    return normed_param * param_magnitude

class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight, self.bias, self.weight_mag, self.bias_mag = create_batch_norm_params(num_features)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x, j):
        w = get_applied(self.weight, self.weight_mag, j)
        b = get_applied(self.bias, self.bias_mag, j)
        y = F.batch_norm(x, self.running_mean, self.running_var, weight=w, bias=b, training=self.training,
                         momentum=self.momentum, eps=self.eps)
        return y



class ConvBlock(nn.Module):
    def __init__(self, c_in, out, k, image_width, stride, padding=0, with_relu=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight, self.bias, self.weight_mag, self.bias_mag = create_conv_params(c_in, out, k)
        self.layer_norm = BatchNorm2d(out)
        self.with_relu = with_relu

    def forward(self, x, j):
        w = get_applied(self.weight, self.weight_mag, j)
        b = get_applied(self.bias, self.bias_mag, j)
        a = F.conv2d(x, weight=w, bias=b, stride=self.stride, padding=self.padding)
        a = self.layer_norm(a, j)
        if self.with_relu:
            a = F.relu(a)
        return a


class Linear(nn.Module):

    def __init__(self, c_in, out):
        super().__init__()
        self.weight, self.bias, self.weight_mag, self.bias_mag = create_linear_params(out, c_in)

    def forward(self, x, j):
        w = get_applied(self.weight, self.weight_mag, j)
        b = get_applied(self.bias, self.bias_mag, j)
        return F.linear(x, weight=w, bias=b)

class Sequential(nn.Sequential):

    def forward(self, x, j):
        for module in self:
            x = module(x, j)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, output_width, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU())
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(out_channels))
        self.conv1 = ConvBlock(in_channels, out_channels, 3, output_width, stride, padding=1, with_relu=True)
        self.conv2 = ConvBlock(out_channels, out_channels, 3, output_width, 1, padding=1, with_relu=False)
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x, j):
        residual = x
        out = self.conv1(x, j)
        out = self.conv2(out, j)
        if self.downsample:
            residual = self.downsample(x, j)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, layers, num_classes=10, base_model=None):
        super(ResNet, self).__init__()
        self.inplanes = 64
        k = 7
        c_in = 3
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU())
        self.conv1 = ConvBlock(c_in, self.inplanes, k, 112, 2, 3, True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(64, 56,  layers[0], stride=1)
        self.layer1 = self._make_layer(128, 28, layers[1], stride=2)
        self.layer2 = self._make_layer(256, 14, layers[2], stride=2)
        self.layer3 = self._make_layer(512, 7, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = Linear(num_classes, 512)

    def _make_layer(self, planes, output_width, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
            #     nn.BatchNorm2d(planes),
            # )
            downsample = ConvBlock(self.inplanes, planes, 1, output_width, stride, with_relu=False)
        layers = [ResidualBlock(self.inplanes, planes, output_width, stride, downsample)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes, output_width))

        return Sequential(*layers)

    def forward(self, x, j):
        x = self.conv1(x, j)
        x = self.maxpool(x)
        x = self.layer0(x, j)
        x = self.layer1(x, j)
        x = self.layer2(x, j)
        x = self.layer3(x, j)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, j)

        return x


# base_model = ResNet([3, 4, 6, 3])  # .to(device)
# models = [base_model]
# for j in range(m-1):
#     models.append(copy.deepcopy(base_model))
# models = [model.to(device) for model in models]
models = [ResNet([2, 2, 2, 2]).to(device) for j in range(m)]

# # Create a base model, this won't be trained but is used to initialize the magnitudes
# base_model = ResNet([2, 2, 2, 2])
# # Create the models that will be trained, their initial values will be used as the normalized weight tensors
# models = [ResNet([2, 2, 2, 2]) for j in range(m)]

all_params = []  # The params from all the models in a single list, passed to the optimizer
params_lists = []  # The params from each model in their own list, used for variance minimization
for model in models:
    params = list(model.parameters())
    all_params.extend(params)
    params_lists.append(params)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(all_params, lr=learning_rate, momentum = 0.9) # , weight_decay = 0.001


def evaluate_accuracy(data_loader, models, name):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logit_sum = 0.0
            for j, model in enumerate(models):
                outputs = model(images, j)
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

def make_positive(module, grad_input, grad_output):
    return [torch.abs(grad) for grad in grad_input]


t = 0
for epoch in range(num_epochs):
    print('Epoch {}'.format(epoch))
    train_iterators = [iter(train_loader) for train_loader in train_loaders]
    validation_iterator = iter(valid_loader)
    for b in range(num_batches):
        classification_loss = 0.0
        # Update based on the classification error
        for j, train_iterator, model in zip(range(len(models)), train_iterators, models):
            images, labels = next(train_iterator)
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images, j)
            loss = criterion(outputs, labels)
            classification_loss += loss
        # Backward and optimize
        optimizer.zero_grad()
        classification_loss.backward()
        optimizer.step()

        # # Calculate the regularization error
        # val_images, val_labels = next(validation_iterator)
        # val_images = val_images.to(device)
        # val_labels = val_labels.to(device)
        # outputs_per_model = []
        # mean_output = 0
        # for model in models:
        #     outputs = model(val_images)
        #     outputs_per_model.append(outputs)
        #     mean_output += outputs / m
        # regularization_loss = 0.0
        # for outputs in outputs_per_model:
        #     delta = outputs - mean_output
        #     delta.register_hook(make_positive)
        #     regularization_loss += delta ** 2
        # # Backward and optimize
        # optimizer.zero_grad()
        # regularization_loss.backward()
        # optimizer.step()

        # del images, labels, outputs
        # torch.cuda.empty_cache()
        # gc.collect()

        t += 1
        print('    Iteration {}, Classification Loss: {:.4f}'
              .format(t, classification_loss.item()))
        # print('    Iteration {}, Reg Loss: {:.4f}'
        #       .format(t, regularization_loss.item()))

    # for train_loader, model in zip(train_loaders, models):
    evaluate_accuracy(train_loaders[0], [models[0]], 'train')
    evaluate_accuracy(valid_loader, models, 'validation')
