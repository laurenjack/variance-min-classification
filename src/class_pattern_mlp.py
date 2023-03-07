import ssl
import torch
from torch import nn
from src import dataset_creator, splitter
from torchmetrics import AveragePrecision
# from sklearn.metrics import precision_recall_curve
# import matplotlib.pyplot as plt
import torch.nn.functional as F

# Set the seed for reproduceability
torch.manual_seed(59309)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Fix issues with ssl certificates
ssl._create_default_https_context = ssl._create_unverified_context

data_dir = './data'  # Where should the cifar10 data be downloaded?
examples_per_class = 20
m = 10  # The number of models
batch_size = 40  # The batch size when training or evaluating the network
total_n = 160
num_classes =  8 # 10
noisy_d = 30
percent_correct = 1.0

dataset = dataset_creator.class_pattern_with_noise(total_n, num_classes, noisy_d, percent_correct, 1)
# dataset = dataset_creator.cifar10(data_dir, examples_per_class, batch_size, m)
train_loader, val_loader = splitter.train_val_split(dataset, batch_size, m)

num_input = train_loader.num_input()
num_hidden = 10
num_epochs = 100
learning_rate = 0.1
momentum = 0.9
epsilon = 0.00000001
is_learned = True
weight_decay = 1.0

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
    base_weight = 2 * (torch.rand(out, c_in, k, k, requires_grad=True) - 0.5) * scaler
    base_bias = 2 * (torch.rand(out, requires_grad=True) - 0.5) * scaler
    return create_params(base_weight, base_bias)

def create_linear_params(in_dim, out):
    scaler = 1 / in_dim ** 0.5
    base_weight = 2 * (torch.rand(out, in_dim, requires_grad=True) - 0.5)  * scaler
    base_bias = 2 * (torch.rand(out, requires_grad=True) - 0.5) * scaler
    return create_params(base_weight, base_bias)

def get_reg(tensor, base_tensor):
    w_mean = torch.mean(tensor, axis=0, keepdim=True)
    var = torch.mean((tensor - w_mean) ** 2, axis=0)
    return var.detach() * torch.abs(base_tensor)


class Sequential(nn.Sequential):

    def forward(self, x, j):
        for module in self:
            x = module(x, j)
        return x

class MultiModule(nn.Module):

    def __init__(self):
        super().__init__()

    def get_params(self, j):
        w = self.base_weight
        b = self.base_bias
        if j is not None:
            w = w.detach() + self.weight[j]
            b = b.detach() + self.bias[j]
        return w, b

    def reg_loss(self):
        w_reg = torch.sum(self.base_weight ** 2)
        b_reg = torch.sum(self.base_bias ** 2)
        if is_learned:
            w_reg = get_reg(self.weight, self.base_weight)
            b_reg = get_reg(self.bias, self.base_bias)
        return torch.sum(w_reg) + torch.sum(b_reg)


class Linear(MultiModule):

    def __init__(self, c_in, out):
        super().__init__()
        self.base_weight, self.base_bias, self.weight, self.bias = create_linear_params(c_in, out)

    def forward(self, x, j):
        w, b = self.get_params(j)
        return F.linear(x, weight=w, bias=b)



class BatchNorm2d(MultiModule):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.base_weight, self.base_bias, self.weight, self.bias = create_batch_norm_params(num_features)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x, j):
        w, b = self.get_params(j)
        y = F.batch_norm(x, self.running_mean, self.running_var, weight=w, bias=b, training=self.training,
                         momentum=self.momentum, eps=self.eps)
        return y

class ConvBlock(MultiModule):
    def __init__(self, c_in, out, k, image_width, stride, padding=0, with_relu=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.base_weight, self.base_bias, self.weight, self.bias = create_conv_params(c_in, out, k)
        self.batch_norm = BatchNorm2d(out)
        self.with_relu = with_relu

    def forward(self, x, j):
        w, b = self.get_params(j)
        a = F.conv2d(x, weight=w, bias=b, stride=self.stride, padding=self.padding)
        a = self.batch_norm(a, j)
        if self.with_relu:
            a = F.relu(a)
        return a

    def reg_loss(self):
        return super.reg_loss() + self.batch_norm.reg_loss()

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, output_width, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
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

    def reg_loss(self):
        total = self.conv1.reg_loss() + self.conv2.reg_loss()
        if self.downsample:
            total += self.downsample.reg_loss()
        return total

class ResNet(nn.Module):

    def __init__(self, layers, num_classes=10, base_model=None):
        super(ResNet, self).__init__()
        self.all_residual_blocks = []
        self.inplanes = 64
        k = 7
        c_in = 3
        self.conv1 = ConvBlock(c_in, self.inplanes, k, 112, 2, 3, True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(64, 56,  layers[0], stride=1)
        self.layer1 = self._make_layer(128, 28, layers[1], stride=2)
        self.layer2 = self._make_layer(256, 14, layers[2], stride=2)
        self.layer3 = self._make_layer(512, 7, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = Linear(512, num_classes)


    def _make_layer(self, planes, output_width, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = ConvBlock(self.inplanes, planes, 1, output_width, stride, with_relu=False)
        layers = [ResidualBlock(self.inplanes, planes, output_width, stride, downsample)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes, output_width))
        self.all_residual_blocks.extend(layers)
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

    def reg_loss(self):
        return self.conv1.reg_loss() + sum([r.reg_loss() for r in self.all_residual_blocks]) + self.fc.reg_loss()

class Mlp(nn.Module):

    def __init__(self, num_input, num_hidden, num_classes):
        super(Mlp, self).__init__()
        self.hidden_layer = Linear(num_input, num_hidden)
        self.relu = nn.ReLU()
        # self.hidden_layer2 = Linear(10, 10)
        # self.relu2 = nn.ReLU()
        # self.hidden_layer3 = Linear(10, 10)
        # self.relu3 = nn.ReLU()
        self.output_layer = Linear(num_hidden, num_classes)

    def forward(self, x, j):
        a = self.hidden_layer(x, j)
        a = self.relu(a)
        # a = self.hidden_layer2(a, j)
        # a = self.relu2(a)
        # a = self.hidden_layer3(a, j)
        # a = self.relu3(a)
        return self.output_layer(a, j)

    def reg_loss(self):
        return self.hidden_layer.reg_loss() + self.output_layer.reg_loss() # + self.hidden_layer2.reg_loss() + self.hidden_layer3.reg_loss()

model = Mlp(num_input, num_hidden, num_classes).to(device)
# model = ResNet([2, 2, 2, 2]).to(device)

# Loss and optimizer
softmax = nn.Softmax(dim=1)
softmax_second_dim = nn.Softmax(dim=2)
log_softmax = nn.LogSoftmax(dim=1)
log_softmax_without = nn.LogSoftmax(dim=1)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.00, momentum = momentum)
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
            image = example[0].to(device)
            label = example[1].to(device)
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
        image = image.to(device)
        label = label.to(device)
        # Grab just half the models
        # train_with = train_with[:, torch.randperm(m // 2)[: m // 4]]
        # train_without = train_without[:, torch.randperm(m // 2)[: m // 4]]
        current_batch_size = len(label)
        indices_for_batch = torch.arange(current_batch_size, dtype=torch.int64)
        one_hot = torch.zeros(current_batch_size, num_classes).to(device)
        one_hot[indices_for_batch, label] = 1.0

        # Firstly, update the base model
        base_y_hat = model(image, None)
        ls = log_softmax(base_y_hat)
        loss = -torch.mean(torch.sum(one_hot * ls, axis=1))
        loss += weight_decay * model.reg_loss()
        print('Loss: {}'.format(loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Now update the additional models
        y_hats = []
        for j in range(m):
            y_hats.append(model(image, j))

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
            # loss += weight_decay * model.reg_loss()
            # loss -= 5.0 * torch.mean(torch.sum(y_hat_mean * ls_without, axis=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Train Acc: {}'.format(eval_train_accuracy(y_hat_train, label)))
        # print('Without Acc: {}'.format(eval_train_accuracy(y_hat_without, label)))
        # print(label[0])
        # print(y_hat_probs[:, 0, :])

    acc, average_precision, probs, targets = eval_accuracy(val_loader, model)
    print('Val Acc: {}'.format(acc))
    print('Average Precision: {}'.format(average_precision))
    print('')

# print(mlp.hidden_layer.base_weight[0])




