import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR

from src.hyper_parameters import HyperParameters
from src.posterior_minimizer import numeric_integrals, weight_tracker as wt

class Trainer(object):

    def run(self, model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, hp: HyperParameters, direct_reg_constructor=None, weight_tracker=None):
        if direct_reg_constructor is None:
            direct_reg_constructor = DirectReg
        direct_reg = direct_reg_constructor(hp.post_constant)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        if hp.is_adam:
            optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=hp.learning_rate,
                                        weight_decay=hp.weight_decay,
                                        momentum=hp.momentum)

        scheduler = StepLR(optimizer, step_size=10, gamma=hp.gamma)
        if weight_tracker is None:
            weight_tracker = wt.WeightTracker()
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
                logits = model(image)
                loss = self.loss(logits, label) # + hp.post_constant * model.weight_mag

                optimizer.zero_grad()
                loss.backward()
                # reg_term = hp.post_constant * model.weight / torch.sum(model.weight ** 2)
                # print(reg_term)
                with torch.no_grad():
                    weight_tracker.pre_reg(model)
                    direct_reg.apply(model, image, label)
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
        d0 = x.shape[1]
        L =len(model.linears)
        for l, linear in enumerate(model.linears):
            dl_plus, dl = linear.weight.shape
            d_scale = dl_plus # dl_plus * dl
            # if l < L-1:
            #     d_scale /= 2 ** 0.5
            # if l > 0:
            #     d_scale *= 2
            # else:
            #     d_scale /= 2 ** 0.5
            # grads = linear.weight.grad[0, 1:]
            # var = torch.sum(grads ** 2) / n
            # print(var ** 0.5)
            # import matplotlib.pyplot as plt
            #
            # plt.hist(grads, bins=50, edgecolor='black')
            # plt.title('Histogram of grads array')
            # plt.xlabel('Value')
            # plt.ylabel('Frequency')
            # plt.grid(True)
            # plt.show()
            # break
            linear.weight.grad += self.post_constant / (n * d_scale) ** 0.5 * torch.sign(linear.weight.data)



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




