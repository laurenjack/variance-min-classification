import torch
from torch import nn
from torch.utils.data import DataLoader

from src.hyper_parameters import HyperParameters


def run(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, hp: HyperParameters, num_class, gradient):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=hp.learning_rate,
                                weight_decay=hp.weight_decay,
                                momentum=hp.momentum)
    softmax_cross_entropy = nn.CrossEntropyLoss()

    for epoch in range(hp.epochs):
        if hp.print_epoch:
            print('Before epoch {}'.format(epoch))
            val_accuracy = _evaluate_accuracy(model, validation_loader, device)
            print('Validation accuracy: {}\n'.format(val_accuracy))

        gradient.reset_purity()

        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            logits = model(image)
            logits = gradient.logit_clipper(logits, label)

            loss = softmax_cross_entropy(logits, label)

            optimizer.zero_grad()
            loss.backward()
            gradient.modify_gradient(hp, label)
            optimizer.step()
            if hp.print_batch:
                print(f'Batch train loss: {loss.item()}')
    return _evaluate_accuracy(model, train_loader, device), _evaluate_accuracy(model, validation_loader, device)


def _evaluate_accuracy(model: nn.Module, data_loader: DataLoader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for image, label in data_loader:
            image = image.to(device)
            label = label.to(device)
            logits = model(image)
            _, predicted = torch.max(logits, dim=1)
            is_correct = predicted == label
            correct += is_correct.sum().item()
            total += label.size(0)
        return correct / total

