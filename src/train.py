import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from src.hyper_parameters import HyperParameters
from src.posterior_minimizer import numeric_integrals, weight_tracker as wt
from src.posterior_minimizer import regularizer as reg


class Trainer(object):

    def run(self, model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, hp: HyperParameters,
            direct_reg, weight_tracker=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        if hp.is_adam:
            optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=hp.weight_decay)
            #optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=hp.learning_rate,
                                        weight_decay=hp.weight_decay,
                                        momentum=hp.momentum)

        sigmoid = nn.Sigmoid()
        scheduler = StepLR(optimizer, step_size=7, gamma=hp.gamma)
        if weight_tracker is None:
            weight_tracker = wt.WeightTracker()
        weight_tracker.update(model)

        for epoch in range(hp.epochs):
            if hp.print_epoch:
                print('Before epoch {}'.format(epoch))
                r = self._eval_accuracy(model, train_loader, device), self._eval_accuracy(model, validation_loader,
                                                                                          device)
                print('Accuracy: {}\n'.format(r))
                # print(model.linears[0].weight[:, 0])

            for image, label in train_loader:
                image = image.to(device)
                label = label.to(device)
                loss = self.loss(model, image, label) # + hp.post_constant * model.weight_mag

                optimizer.zero_grad()
                loss.backward()
                # reg_term = hp.post_constant * model.weight / torch.sum(model.weight ** 2)
                # print(reg_term)
                with torch.no_grad():
                    weight_tracker.pre_reg(model)
                    direct_reg.apply(model, image, label, epoch)
                    weight_tracker.post_reg(model)
                optimizer.step()
                if hp.print_batch:
                    print(f'Batch train loss: {loss.item()}')
            weight_tracker.update(model)
            scheduler.step()
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
                # w = torch.tensor([[1.0, 1.0, 1.0]])
                # logits = F.linear(image, weight=w)
                predicted = self.predict(model, image)
                is_correct = predicted == label
                correct += is_correct.sum().item()
                total += label.size(0)
            return correct / total


class SoftmaxXeTrainer(Trainer):

    def __init__(self):
        self.xe = None  # TODO(Jack) add correct loss

    def loss(self, model, x, y):
        logits = model(x)
        return self.xe(logits, y)  # TODO(Jack) check correct

    def predict(self, model, x):
        logits = model(x)
        return torch.max(logits, dim=1)[1]  # TODO(Jack) check correct


class SigmoidBxeTrainer(Trainer):

    def __init__(self):
        self.bce_with_sigmoid = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def loss(self, model, x, y):
        logits = model(x)
        return self.bce_with_sigmoid(logits[:, 0], y.float())

    def predict(self, model, x):
        logits = model(x)
        probs = self.sigmoid(logits[:, 0])
        return torch.round(probs, decimals=0)

class IdentityMseTrainer(Trainer):

    def __init__(self):
        self.mse = nn.MSELoss()

    def loss(self, model, x, y):
        logits = model(x)
        return self.mse(logits[:, 0], y.float())

    def predict(self, model, x):
        logits = model(x)
        return (logits[:, 0] > 0).float() # * 2 - 1


class DirectMeanTrainer(IdentityMseTrainer):

    def loss(self, model, x, y):
        n, d = x.shape
        y_shift = y.view(n, 1) * 2.0 - 1
        mean = model(torch.eye(d)).t()
        # target = y_shift.view(n, 1) @ mean.t()
        return torch.sum(torch.mean((2 * x * y_shift - mean) ** 2, axis=0)) / 2



