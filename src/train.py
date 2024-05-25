import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

from src.hyper_parameters import HyperParameters
from src.posterior_minimizer import numeric_integrals


class Trainer(object):

    def run(self, model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, hp: HyperParameters, direct_reg_constructor=None):
        if direct_reg_constructor is None:
            direct_reg_constructor = DirectReg
        direct_reg = direct_reg_constructor(hp.post_constant)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        # optimizer = torch.optim.SGD(model.parameters(),
        #                             lr=hp.learning_rate,
        #                             weight_decay=hp.weight_decay,
        #                             momentum=hp.momentum)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)

        for epoch in range(hp.epochs):
            if hp.print_epoch:
                print('Before epoch {}'.format(epoch))
                r = self._eval_accuracy(model, train_loader, device), self._eval_accuracy(model, validation_loader,
                                                                                          device)
                print('Accuracy: {}\n'.format(r))
                # if epoch >= 75:
                #     print(model.weight)
                #     print(model.weight_mag)
                #     print(model.w)

            for image, label in train_loader:
                image = image.to(device)
                label = label.to(device)
                logits = model(image)
                loss = self.loss(logits, label) # + hp.post_constant * model.weight_mag

                optimizer.zero_grad()
                loss.backward()
                # reg_term = hp.post_constant * model.weight / torch.sum(model.weight ** 2)
                # print(reg_term)
                with torch.no_grad():
                    direct_reg.apply(model, image, label)
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
                # w = torch.tensor([[1.0, 1.0, 1.0]])
                # logits = F.linear(image, weight=w)
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
        return self.bce_with_sigmoid(logits[:, 0], labels.float())

    def predict(self, logits):
        probs = self.sigmoid(logits[:, 0])
        return torch.round(probs, decimals=0)


class DirectReg:
    """
    Modify the gradients of a model directly, to apply a regularizer.
    """

    def __init__(self, post_constant):
        self.post_constant = post_constant

    def apply(cls, model, x, y):
        pass


class L1(DirectReg):

    def apply(self, model, x, y):
        n = x.shape[0]
        for linear in model.linears:
            # if linear.weight.shape[1] == 8:
            #     print(linear.weight)
            #     print(linear.weight.grad)
            linear.weight.grad += self.post_constant / n ** 0.5 * torch.sign(linear.weight.data)


class BoxScaled(DirectReg):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def apply(self, model, x, y):
        w = model.weight.data
        model.weight.grad += w



class BoundsAsParam(DirectReg):

    def apply(self, model, x, y):
        w = model.weight.data
        posterior_grad = self.post_constant * numeric_integrals.d_posterior(x.numpy(), y.numpy(), w.numpy()[0])
        print(model.weight)
        print(model.weight.grad)
        print(posterior_grad)
        model.weight.grad += posterior_grad.astype(np.float32)
        print(model.weight.grad)







