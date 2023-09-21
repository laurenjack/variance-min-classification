import torch

from hyper_parameters import HyperParameters
from torch.autograd import Function

class Xe:

    def __init__(self, model):
        self.model = model

    def reset_purity(self):
        pass

    def modify_gradient(self, hp: HyperParameters, image, label, batch_indices):
        pass


class SingleXe(Xe):

    def __init__(self, model):
        super().__init__(model)

    def single_logit_gradient(self, y, batch_indices):
        logit_gradient = self.model.layers.recent_activations[-1].grad.detach()
        single_logit_gradient = torch.zeros(*logit_gradient.shape)
        single_logit_gradient[batch_indices, y] = logit_gradient[batch_indices, y]
        return single_logit_gradient

    def modify_gradient(self, hp: HyperParameters, image, label, batch_indices):
        x = image.detach()
        y = label.detach()
        grad = self.single_logit_gradient(y, batch_indices)
        self.model.linear_layers[-1].weight.grad = grad.t() @ x


class Pure(SingleXe):

    def __init__(self, model):
        super().__init__(model)
        self.numerators = []
        self.denominators = []
        self.previous_numerators = []
        self.previous_denominators = []
        for linear in model.linear_layers:
            shape = linear.weight.shape
            self.numerators.append(torch.zeros(*shape))
            self.denominators.append(torch.zeros(*shape))
            self.previous_numerators.append(torch.zeros(*shape))
            self.previous_denominators.append(torch.zeros(*shape))

    def reset_purity(self):
        for i in range(len(self.numerators)):
            self.numerators[i] -= self.previous_numerators[i]
            self.denominators[i] -= self.previous_denominators[i]
            self.previous_numerators[i] = self.numerators[i].clone()
            self.previous_denominators[i] = self.denominators[i].clone()

    def modify_gradient(self, hp: HyperParameters, image, label, batch_indices):
        x = image.detach()
        y = label.detach()
        grad = self.single_logit_gradient(y, batch_indices)
        for i in range(len(self.model.linear_layers)):
            grad_t = grad.t()
            weight_grad =grad_t  @ x
            self.numerators[i] += weight_grad
            self.denominators[i] += torch.abs(grad_t) @ torch.abs(x)
            if hp.purity_components == 'leading':
                gradient_purity = torch.abs(self.numerators[0]) / (self.denominators[0] + 0.0000001)
            elif hp.purity_components == 'lagging':
                gradient_purity = torch.abs(self.previous_numerators[0]) / (self.previous_denominators[0] + 0.0000001)
            gradient_filter = (gradient_purity > hp.purity_threshold).float()
            new_grad = weight_grad * gradient_filter
            self.model.linear_layers[i].weight.grad = new_grad


def create_gradient(hp: HyperParameters, model):
    if hp.gradient == 'xe':
        return Xe(model)
    if hp.gradient == 'xe-single':
        return SingleXe(model)
    if hp.gradient == 'purity-scaled':
        return Pure(model)
    raise ValueError(f'Unknown gradient type specified: {hp.gradient}')

