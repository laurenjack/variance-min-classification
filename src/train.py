import torch
from torch import nn
from torch.utils.data import DataLoader

from src.hyper_parameters import HyperParameters


def run(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, num_classes, hp: HyperParameters):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=hp.learning_rate,
                                weight_decay=hp.weight_decay,
                                momentum=hp.momentum)
    softmax_cross_entropy = nn.CrossEntropyLoss()
    # n = len(train_loader.dataset)

    for epoch in range(hp.epochs):
        if hp.print_epoch:
            print('Before epoch {}'.format(epoch))
            r = _evaluate_accuracy(model, train_loader, device), _evaluate_accuracy(model, validation_loader, device)
            print('Accuracy: {}\n'.format(r))

        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            logits = model(image)

            current_batch_size = label.shape[0]
            indices_for_batch = torch.arange(current_batch_size, dtype=torch.int64)
            one_hot = torch.zeros(current_batch_size, num_classes).to(device)
            one_hot[indices_for_batch, label] = 1.0
            # alpha = hp.k / (num_classes * n)
            one_hot += hp.k
            one_hot /= torch.sum(one_hot, axis=1, keepdim=True)

            loss = softmax_cross_entropy(logits, one_hot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if hp.print_batch:
                print(f'Batch train loss: {loss.item()}')
    return _evaluate_accuracy(model, train_loader, device), _evaluate_accuracy(model, validation_loader, device)


def _evaluate_accuracy(model: nn.Module, data_loader: DataLoader, device):
    model.eval()
    with torch.no_grad():
        softmax = torch.nn.Softmax()
        correct = 0
        total = 0
        for image, label in data_loader:
            image = image.to(device)
            label = label.to(device)
            logits = model(image)
            # if image.shape[0] <= 12:
            #     a = softmax(logits)
            #     print(a)
            _, predicted = torch.max(logits, dim=1)
            is_correct = predicted == label
            correct += is_correct.sum().item()
            total += label.size(0)
        return correct / total

