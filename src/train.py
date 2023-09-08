import torch
from torch import nn
from torch.utils.data import DataLoader

from src.hyper_parameters import HyperParameters


def run(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, hp: HyperParameters, num_class: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=hp.learning_rate,
                                weight_decay=hp.weight_decay,
                                momentum=hp.momentum)
    softmax_cross_entropy = nn.CrossEntropyLoss()

    # Setup for calculating gradient purity
    d = model.num_input
    numerator = torch.zeros(num_class, d)
    denominator = torch.zeros(num_class, d)
    previous_numerator = torch.zeros(num_class, d)
    previous_denominator = torch.zeros(num_class, d)

    for epoch in range(hp.epochs):
        if hp.print_epoch:
            print('Before epoch {}'.format(epoch))
            val_accuracy = _evaluate_accuracy(model, validation_loader, device)
            print('Validation accuracy: {}\n'.format(val_accuracy))

        numerator -= previous_numerator
        denominator -= previous_denominator
        previous_numerator = numerator.clone()
        previous_denominator = denominator.clone()


        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            logits = model(image)
            logits.retain_grad() # Retain grad for future calculation
            loss = softmax_cross_entropy(logits, label)

            batch_size = image.shape[0]
            batch_indices = torch.range(0, batch_size - 1, dtype=torch.int64)

            optimizer.zero_grad()
            loss.backward()

            if hp.gradient == 'xe-single' or hp.gradient == 'purity-scaled':
                x = image.detach()
                y = label.detach()
                logit_gradient = logits.grad.detach()[batch_indices, y].unsqueeze(1)
                # .unsqueeze(1)
                element_wise_grad = logit_gradient * x
                new_grad = torch.zeros(num_class, d)
                part_denominator = torch.zeros(num_class, d)
                for i in range(d):
                    new_grad[:, i] = torch.zeros(num_class).scatter_add_(0, y, element_wise_grad[:, i])
                    part_denominator[:, i] = torch.zeros(num_class).scatter_add_(0, y, torch.abs(element_wise_grad[:, i]))
                if hp.gradient == 'purity-scaled':
                    numerator += new_grad
                    denominator += part_denominator
                    gradient_purity = torch.abs(numerator) / (denominator + 0.0000001)
                    gradient_filter = (gradient_purity > 0.5).float()
                    new_grad *= gradient_filter
                old_grad = model.linear_layers[-1].weight.grad
                model.linear_layers[-1].weight.grad = new_grad
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

