import ssl
import torch
from torch import nn
from src import dataset_creator, train_val_split
from torchmetrics import AveragePrecision
from models import MultiMlp, ResNet


# Set the seed for reproduce-ability
torch.manual_seed(59310 + 1)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Fix issues with ssl certificates
ssl._create_default_https_context = ssl._create_unverified_context


# Set these
m = 6  # The number of models
batch_size = 40  # The batch size when training or evaluating the network
num_epochs = 100
learning_rate = 0.02
momentum = 0.9
epsilon = 0.00000001
weight_decay = 0.01
reg_const = 0.0
# Must be set when running the CIFAR10 dataset  + resnet
data_dir = './data' # Where should the cifar10 data be downloaded?
examples_per_class = 160
# Must be set when using the binary class pattern dataset + MLP
num_input = 23
num_hidden = 10
total_n = 320
num_classes = 8
noisy_d = 20
percent_correct = 1.0


def eval_train_accuracy(y_train, label):
    _, predicted = torch.max(y_train, 2)
    total = y_train.size(0) * y_train.size(1)
    correct = (predicted == label).sum().item()
    return correct / total


def eval_accuracy(data_loader, model):
    softmax = nn.Softmax(dim=1)
    ap = AveragePrecision(task='multiclass', num_classes=num_classes)
    with torch.no_grad():
        total = 0.0
        correct = 0.0
        y_bar_sums = []
        targets = []
        for example in data_loader:
            image = example[0].to(device)
            label = example[1].to(device)
            y_bar_sum = 0.0
            for j in range(m):
                # TODO(Jack) replace with eval method
                output = model(image, j, True) # is_variance flag is irrelevant here
                y_bar = softmax(output)
                y_bar_sum += y_bar / m
            # y_bar_sum = model(image, None)
            y_bar_sums.append(y_bar_sum)
            targets.append(label)
            _, predicted = torch.max(y_bar_sum, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        y_bar_sums = torch.cat(y_bar_sums)
        targets = torch.cat(targets)
        average_precision = ap(y_bar_sums, targets)
        return correct / total, average_precision, y_bar_sums, targets  #, auc

def train():
    # dataset = dataset_creator.cifar10(data_dir, examples_per_class)
    dataset, _ = dataset_creator.binary_class_pattern_with_noise(total_n, num_classes, noisy_d, percent_correct)
    train_loader, val_loader = train_val_split.on_percentage(dataset, batch_size, m)

    # model = ResNet(m, [2, 2, 2, 2]).to(device)
    model = MultiMlp(m, num_input, num_hidden, num_classes).to(device)

    # Loss and optimizer
    softmax_second_dim = nn.Softmax(dim=2)
    log_softmax = nn.LogSoftmax(dim=1)
    log_softmax_without = nn.LogSoftmax(dim=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    for epoch in range(num_epochs):
        print('Epoch {}'.format(epoch))
        for image, label, train_with, train_without in train_loader:
            image = image.to(device)
            label = label.to(device)
            current_batch_size = len(label)
            indices_for_batch = torch.arange(current_batch_size, dtype=torch.int64)
            one_hot = torch.zeros(current_batch_size, num_classes).to(device)
            one_hot[indices_for_batch, label] = 1.0

            # Firstly, update the base model
            # base_y_hat = model(image, None)
            # ls = log_softmax(base_y_hat)
            # class_loss = -torch.mean(torch.sum(one_hot * ls, axis=1))
            # reg_loss = weight_decay * model.reg_loss()
            # print('Classification Loss: {}'.format(class_loss.item()))
            # print('Reg Loss: {}'.format(reg_loss.item()))
            # loss = class_loss + reg_loss

            z_hat_train_list = []
            z_hat_without_list = []
            for j in range(m):
                z_hat_train_list.append(model(image, j, False))
                z_hat_without_list.append(model(image, j, True))
            z_hat_train = torch.stack(z_hat_train_list)
            z_hat_without = torch.stack(z_hat_without_list)
            # Separate into the models to be trained, and those to be used as variance minimizing means
            train_indices = torch.transpose(train_with, 0, 1)
            without_indices = torch.transpose(train_without, 0, 1)
            z_hat_train = z_hat_train[train_indices, indices_for_batch]  # Shape (m // 2, batch_size, num_classes)
            z_hat_without = z_hat_without[without_indices, indices_for_batch] # Shape (m // 2, batch_size, num_classes)
            y_hat_probs = softmax_second_dim(z_hat_without)
            y_hat_mean = torch.mean(y_hat_probs, axis=0, keepdim=True).detach()

            # z_hat_mean = torch.mean(z_hat_without, axis=0, keepdim=True)
            # z_delta = z_hat_without - z_hat_mean
            # var_delta = y_hat_probs - y_hat_mean
            # var_delta = var_delta.detach()

            loss = 0.0
            for j in range(m // 2):
                ls = log_softmax(z_hat_train[j])
                ls_without = log_softmax_without(z_hat_without[j])
                loss -= torch.mean(torch.sum(one_hot * ls, axis=1))
                loss -= reg_const * torch.mean(torch.sum(y_hat_mean * ls_without, axis=1))
                # loss += reg_const * torch.mean(torch.sum(var_delta[j] * z_delta[j]))
                # loss += reg_const * torch.mean(torch.sum(var_delta ** 2, axis=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Train Acc: {}'.format(eval_train_accuracy(z_hat_train, label)))
            # print('Without Acc: {}'.format(eval_train_accuracy(y_hat_without, label)))
            # print(label[0])
            # print(y_hat_probs[:, 0, :])

        base_train_acc, _, _, _ = eval_accuracy(train_loader, model)
        print('Base Train Acc: {}', base_train_acc)
        acc, _, _, _ = eval_accuracy(val_loader, model)
        print('Val Acc: {}'.format(acc))
        # print('Average Precision: {}'.format(average_precision))

        print('')


if __name__ == '__main__':
    train()




