import torch
from torch import nn
from torch.utils.data import DataLoader

from src.hyper_parameters import HyperParameters


class Trainer(object):

    def run(self, model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, hp: HyperParameters):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=hp.learning_rate,
                                    weight_decay=hp.weight_decay,
                                    momentum=hp.momentum)

        for epoch in range(hp.epochs):
            if hp.print_epoch:
                print('Before epoch {}'.format(epoch))
                r = self._eval_accuracy(model, train_loader, device), self._eval_accuracy(model, validation_loader,
                                                                                          device)
                print('Accuracy: {}\n'.format(r))

            for image, label in train_loader:
                image = image.to(device)
                label = label.to(device)
                logits = model(image)
                loss = self.loss(logits, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if hp.print_batch:
                    print(f'Batch train loss: {loss.item()}')
        return self._eval_accuracy(model, train_loader, device), self._eval_accuracy(model, validation_loader, device)

    def _eval_accuracy(self, model: nn.Module, data_loader: DataLoader, device):
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for image, label in data_loader:
                image = image.to(device)
                label = label.to(device)
                logits = model(image)
                predicted = self.predict(logits)
                is_correct = predicted == label
                correct += is_correct.sum().item()
                total += label.size(0)
            return correct / total


class SoftmaxXeTrainer(Trainer):

    def __init__(self):
        self.xe = None  # TODO(Jack) add correct loss

    def loss(self, logits, labels):
        return self.xe(logits, labels)  # TODO(Jack) check correct

    def predict(self, logits):
        return torch.max(logits, dim=1)[1]  # TODO(Jack) check correct


class SigmoidBxeTrainer(Trainer):

    def __init__(self):
        self.bce_with_sigmoid = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def loss(self, logits, labels):
        return self.bce_with_sigmoid(logits[:, 0], labels)

    def predict(self, logits):
        probs = self.sigmoid(logits[:, 0])
        return torch.round(probs, decimals=0)




