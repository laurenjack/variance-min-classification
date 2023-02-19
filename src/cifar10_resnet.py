import math
import ssl
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(575)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fix issues with ssl certificates
ssl._create_default_https_context = ssl._create_unverified_context


data_dir = './data'  # Where should the cifar10 data be downloaded?
m = 10  # The number of models
examples_per_class = 10  # The number of examples per class in the training set, and also the validation set
batch_size = 50  # The batch size when training or evaluating the network
num_classes = 10  # The number of classes in cifar10
num_epochs = 300
learning_rate = 0.01
n = examples_per_class * num_classes
num_batches = math.ceil(n / batch_size)
reg = 0.02
reg_power = 0.15
epsilon = 0.00000001
mag_epsilon = 0.00000001

# Data loader for train and validation
import dataset_creator
train_loaders, valid_loader = dataset_creator.cifar10(data_dir, examples_per_class, batch_size, m)

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

def get_applied(param, param_magnitude, j, is_variance):
    param_j = param[j]
    norm = norm2(param.detach()) # Need to detach the norm, so that gradients are not shared across datasets.
    normed_param = param_j / (norm + mag_epsilon)
    # Detach the entire normed weight if this is the variance forward pass, because we cannot minimize variance using
    # the weight parameters (the signal will disappear)
    if is_variance:
        normed_param = normed_param.detach()
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

    def forward(self, x, j, is_variance):
        w = get_applied(self.weight, self.weight_mag, j, is_variance)
        b = get_applied(self.bias, self.bias_mag, j, is_variance)
        # w = self.weight[j]
        # b = self.bias[j]
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

    def forward(self, x, j, is_variance):
        w = get_applied(self.weight, self.weight_mag, j, is_variance)
        b = get_applied(self.bias, self.bias_mag, j, is_variance)
        # w = self.weight[j]
        # b = self.bias[j]
        a = F.conv2d(x, weight=w, bias=b, stride=self.stride, padding=self.padding)
        a = self.layer_norm(a, j, is_variance)
        if self.with_relu:
            a = F.relu(a)
        return a


class Linear(nn.Module):

    def __init__(self, c_in, out):
        super().__init__()
        self.weight, self.bias, self.weight_mag, self.bias_mag = create_linear_params(out, c_in)

    def forward(self, x, j, is_variance):
        w = get_applied(self.weight, self.weight_mag, j, is_variance)
        b = get_applied(self.bias, self.bias_mag, j, is_variance)
        # w = self.weight[j]
        # b = self.bias[j]
        return F.linear(x, weight=w, bias=b)

class Sequential(nn.Sequential):

    def forward(self, x, j, is_variance):
        for module in self:
            x = module(x, j, is_variance)
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

    def forward(self, x, j, is_variance):
        residual = x
        out = self.conv1(x, j, is_variance)
        out = self.conv2(out, j, is_variance)
        if self.downsample:
            residual = self.downsample(x, j, is_variance)
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

        # self.shared_parameters = []
        # self.non_shared_parameters = []
        # for name, parameter in self.named_parameters():
        #     if name.endswith('_mag'):
        #         self.shared_parameters.append(parameter)
        #     else:
        #         self.non_shared_parameters.append(parameter)


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

    def forward(self, x, j, is_variance):
        x = self.conv1(x, j, is_variance)
        x = self.maxpool(x)
        x = self.layer0(x, j, is_variance)
        x = self.layer1(x, j, is_variance)
        x = self.layer2(x, j, is_variance)
        x = self.layer3(x, j, is_variance)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, j, is_variance)

        return x


# base_model = ResNet([3, 4, 6, 3])  # .to(device)
# models = [base_model]
# for j in range(m-1):
#     models.append(copy.deepcopy(base_model))
# models = [model.to(device) for model in models]
# models = [ResNet([2, 2, 2, 2]).to(device) for j in range(m)]

# # Create a base model, this won't be trained but is used to initialize the magnitudes
# base_model = ResNet([2, 2, 2, 2])
# # Create the models that will be trained, their initial values will be used as the normalized weight tensors
# models = [ResNet([2, 2, 2, 2]) for j in range(m)]


# all_params = []  # The params from all the models in a single list, passed to the optimizer
# params_lists = []  # The params from each model in their own list, used for variance minimization
# for model in models:
#     params = list(model.parameters())
#     all_params.extend(params)
#     params_lists.append(params)

model = ResNet([2, 2, 2, 2]).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)
# optimizer_shared = torch.optim.SGD(model.shared_parameters, lr=learning_rate, momentum = 0.9) # , weight_decay = 0.001
# optimizer_non_shared = torch.optim.SGD(model.non_shared_parameters, lr=learning_rate, momentum = 0.9)


def evaluate_accuracy(data_loader, name):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            if name == 'train':
                # TODO(Jack) replace with eval method
                outputs = model(images, 0, False) # Is variance flag is irrelevant here
                logit_sum = outputs.data
            else:
                logit_sum = 0.0
                for j in range(m):
                    # TODO(Jack) replace with eval method
                    outputs = model(images, j, False) # Is variance flag is irrelevant here
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

def bias(batch_per_model):
    classification_loss = 0.0
    for j, batch in enumerate(batch_per_model):
        images, labels = batch
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images, j, False)
        loss = criterion(outputs, labels)
        classification_loss += loss
    return classification_loss


def variance(images):
    output_list = []
    output_mean = 0.0
    for j in range(m):
        output = model(images, j, True)
        output_mean += output / m
        output_list.append(output)
    variance_loss = 0.0
    for output in output_list:
        delta = output - output_mean
        delta_square = delta ** 2
        variance_loss += torch.mean(torch.sum(delta_square, axis=1), axis=0)
    return 5.0 * variance_loss





t = 0
for epoch in range(num_epochs):
    print('Epoch {}'.format(epoch))
    train_iterators = [iter(train_loader) for train_loader in train_loaders]
    validation_iterator = iter(valid_loader)
    for b in range(num_batches):
        # optimizer_non_shared.zero_grad()
        # optimizer_shared.zero_grad()
        optimizer.zero_grad()
        # Get the batch for each model
        batch_per_model = [next(train_iterator) for train_iterator in train_iterators]
        # Apply just the bias loss to the weights
        first_bias_loss = bias(batch_per_model)
        first_bias_loss.backward()
        images, _ = next(validation_iterator)
        images = images.to(device)
        variance_loss = variance(images)
        variance_loss.backward()
        optimizer.step()
        # Now apply the combined bias and variance loss to the magnitude
        # second_bias_loss = bias(batch_per_model)
        # images, _ = next(validation_iterator)
        # images = images.to(device)
        # # variance_loss = variance(images)
        # total_loss = second_bias_loss # + variance_loss
        # total_loss.backward()

        # optimizer_non_shared.step()
        # optimizer_shared.step()

        t += 1
        print('    Iteration {}, First Bias Loss: {:.4f}'
              .format(t, first_bias_loss.item()))
        # print('    Iteration {}, Second Bias Loss: {:.4f}'
        #       .format(t, second_bias_loss.item()))
        print('    Iteration {}, Variance Loss: {:.4f}'
              .format(t, variance_loss.item()))

    # for train_loader, model in zip(train_loaders, models):
    evaluate_accuracy(train_loaders[0], 'train')
    evaluate_accuracy(valid_loader, 'validation')
