import torch
from torch import nn
import torch.nn.functional as F

from src.learned_dropout.legacy.model_tracker import MLPTracker, ResNetTracker


class MLP(nn.Module):
    def __init__(self, d, n, h_list, relus):
        super(MLP, self).__init__()
        self.n = n
        L = len(h_list)
        self.relus = relus
        # Build L+1 linear layers.
        self.layers = nn.ModuleList()
        # First linear layer: from input (d) to first hidden layer.
        self.layers.append(nn.Linear(d, h_list[0], bias=False))
        # Hidden layers (if L > 1).
        for i in range(1, L):
            self.layers.append(nn.Linear(h_list[i - 1], h_list[i], bias=False))
        # Final linear layer: from last hidden layer to output (1).
        self.layers.append(nn.Linear(h_list[-1], 1, bias=False))

        # Create dropout parameters (c_list) for input and each hidden layer (L+1 in total).
        self.c_list = nn.ParameterList()
        self.c_list.append(nn.Parameter(torch.zeros(d)))  # For input.
        for i in range(L):
            self.c_list.append(nn.Parameter(torch.zeros(h_list[i])))


    @staticmethod
    def get_tracker(track_weights):
        return MLPTracker(track_weights)

    def forward_network1(self, x):
        """
        Forward pass for Network 1.
        Applies dropout by sampling binary masks using keep probabilities computed from c_list (using detached versions).
        Batch normalization is applied before the activation.
        """
        n_samples = x.size(0)
        current = x
        for i, layer in enumerate(self.layers):
            # Compute keep probability p for layer i.
            p = torch.sigmoid(self.c_list[i].detach())
            mask = torch.bernoulli(p.expand(n_samples, -1))
            current = current * mask
            current = layer(current)
            # If not the final layer, apply activation.
            if i < len(self.layers) - 1:
                if self.relus:
                    current = F.relu(current)
        logits = current.squeeze(1)
        return logits

    def forward_network2(self, x):
        """
        Forward pass for Network 2.
        Scales the inputs and hidden activations by differentiable keep probabilities (from c_list) and uses detached weights.
        """
        current = x
        for i, layer in enumerate(self.layers):
            p = torch.sigmoid(self.c_list[i])
            current = current * p
            current = F.linear(current, layer.weight.detach())
            if i < len(self.layers) - 1:
                if self.relus:
                    current = F.relu(current)
        logits = current.squeeze(1)
        return logits

    def var_network2(self, k):
        """
        Defines a variance term as k multiplied by the product of the sums of the differentiable keep probabilities.
        """
        sum_cs = [torch.sum(torch.sigmoid(c)) for c in self.c_list]
        sizes = sum_cs + [1]
        param_count = 0.0
        for s0, s1 in zip(sizes[:-1], sizes[1:]):
            param_count += s0 * s1
        return k * param_count / self.n

    def get_weight_params(self):
        """
        Returns a list of all weight parameters from the MLP layers.
        """
        params = []
        for layer in self.layers:
            params += list(layer.parameters())
        return params

    def get_drop_out_params(self):
        """
        Returns a list of all dropout (learned c) parameters of the MLP.
        """
        return list(self.c_list)


class ResidualBlock(nn.Module):
    """
    A residual block with *two* learnable dropout masks:
      • c_hidden  – applied to the hidden activations (size =h)
      • c_out     – applied to the block output   (size =d)

    The forward paths come in two flavours:
      • forward_network1 – stochastic (Bernoulli-sampled) dropout
      • forward_network2 – deterministic scaling with σ(c)
    """
    def __init__(self, d: int, h: int, relus: bool, layer_norm: bool) -> None:
        super().__init__()
        self.d = d
        self.relus = relus
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.layer_norm_layer = nn.LayerNorm(d)

        # weights
        self.weight_in  = nn.Linear(d, h, bias=False)
        self.weight_out = nn.Linear(h, d, bias=False)

        # learnable dropout parameters
        self.c_hidden = nn.Parameter(torch.zeros(h))   # hidden mask
        self.c_out    = nn.Parameter(torch.zeros(d))   # output mask
        self.cb_hidden = torch.zeros(h)
        self.cb_out = torch.zeros(d)

    # ────────────────────────── forward paths ──────────────────────────

    def forward_network1(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bernoulli-sampled dropout on both the hidden and the output.
        """
        if self.layer_norm:
            x = self.layer_norm_layer(x)

        n = x.size(0)

        # hidden transformation + dropout
        hidden = self.weight_in(x)
        if self.relus:
            hidden = F.relu(hidden)

        p_h   = torch.sigmoid(self.c_hidden.detach())
        maskh = torch.bernoulli(p_h.expand(n, -1))
        hidden = hidden * maskh

        # output transformation + dropout
        out = self.weight_out(hidden)
        p_o   = torch.sigmoid(self.c_out.detach())
        masko = torch.bernoulli(p_o.expand(n, -1))
        return out * masko

    def forward_network2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable (deterministic) dropout on hidden & output.
        """
        if self.layer_norm:
            x = self.layer_norm_layer(x)

        hidden = F.linear(x, self.weight_in.weight.detach())
        if self.relus:
            hidden = F.relu(hidden)
        hidden = hidden * torch.sigmoid(self.c_hidden)

        out = F.linear(hidden, self.weight_out.weight.detach())
        return out * torch.sigmoid(self.c_out)

    def param_count(self):
        return (self.d + torch.sigmoid(self.c_out)) * torch.sigmoid(self.c_in)


    def randomize_dropout_biases(self, scale):
        c_hidden = self.c_hidden
        c_out = self.c_out
        self.cb_hidden = c_hidden + scale * torch.randn(c_hidden.shape)
        self.cb_out = c_out + scale * torch.randn(c_out.shape)


class ResNet(nn.Module):
    """
    ResNet with learnable dropout masks on every hidden & output layer
    (but **no** dropout on the raw input features).
    """
    def __init__(self,
                 d: int,
                 n: int,
                 h_list: list[int],
                 relus: bool,
                 layer_norm: bool) -> None:
        super().__init__()
        self.d = d                     # input feature width
        self.n = n                     # dataset size (for variance formula)

        # residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(d, h, relus, layer_norm) for h in h_list
        ])

        # final linear layer + its dropout mask
        self.final_layer = nn.Linear(d, 1, bias=False)
        self.c_final     = nn.Parameter(torch.zeros(d))     # size =d
        self.cb_final = torch.zeros(d)

    # ────────────────────────── forward paths ──────────────────────────

    def forward_network1(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stochastic (Bernoulli) dropout:
        – no mask on the raw input
        – dropout in every block (handled inside the block)
        – dropout before final layer
        """
        current = x
        for block in self.blocks:
            current = current + block.forward_network1(current)

        n      = x.size(0)
        p_f    = torch.sigmoid(self.c_final.detach())
        mask_f = torch.bernoulli(p_f.expand(n, -1))
        final_in = current * mask_f
        return self.final_layer(final_in).squeeze(1)

    @staticmethod
    def get_tracker(track_weights):
        return ResNetTracker(track_weights)

    def forward_network2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic (σ-scaled) dropout:
        – no mask on the raw input
        – σ(c_hidden) / σ(c_out) in every block
        – σ(c_final) before final linear layer
        """
        current = x
        for block in self.blocks:
            current = current + block.forward_network2(current)

        final_in = current * torch.sigmoid(self.c_final)
        return F.linear(final_in, self.final_layer.weight.detach()).squeeze(1)

    def param_count(self):
        total_param_count = 0.0
        for block in self.blocks:
            total_param_count += block.param_count()
        total_param_count += torch.sigmoid(self.c_final)
        return total_param_count

    # ─────────────────────────── variance term ──────────────────────────

    def var_network2(self, k: float) -> torch.Tensor:
        """
        k/n · Σ_i d_i · Σ_{j≠i} d_j   with

        • Hidden-layer connections  (block i) :
              (d + Σ_{t<i} Σ(p_t_out)) · Σ(p_i_hidden)
        • Output-layer connections  (block i) :
              (Σ(p_final) + Σ_{t>i} Σ(p_t_hidden)) · Σ(p_i_out)
        • Within-block connections  (block i) :
              Σ(p_i_hidden) · Σ(p_i_out)
        • Final layer connections   :
              (d + Σ_{t=1..B} Σ(p_t_out)) · 1
        """
        # device, dtype = self.c_final.device, self.c_final.dtype
        # d_val = torch.tensor(self.d, device=device, dtype=dtype)
        #
        # # σ(c) totals for each block
        # p_h_sum = [torch.sum(torch.sigmoid(b.c_hidden)) for b in self.blocks]
        # p_o_sum = [torch.sum(torch.sigmoid(b.c_out)) for b in self.blocks]
        # p_final = torch.sum(torch.sigmoid(self.c_final))
        #
        # # hidden-layer connections (forward cumulative p_out)
        # hidden_conns = []
        # cum_out = torch.tensor(0.0, device=device, dtype=dtype)
        # for h_sum, o_sum in zip(p_h_sum, p_o_sum):
        #     hidden_conns.append((d_val + cum_out) * h_sum)
        #     cum_out = cum_out + o_sum
        #
        # # output-layer connections (reverse cumulative p_hidden)
        # output_conns = [None] * len(self.blocks)
        # cum_hidden = torch.tensor(0.0, device=device, dtype=dtype)
        # for idx in reversed(range(len(self.blocks))):
        #     output_conns[idx] = (p_final + cum_hidden) * p_o_sum[idx]
        #     cum_hidden = cum_hidden + p_h_sum[idx]
        #
        # # within-block connections
        # inside_conns = [h * o for h, o in zip(p_h_sum, p_o_sum)]
        #
        # # final layer (scalar output)
        # final_conns = (d_val + torch.sum(torch.stack(p_o_sum))) * torch.tensor(
        #     1.0, device=device, dtype=dtype
        # )
        #
        # # aggregate all connection counts
        # all_conns = hidden_conns + output_conns + inside_conns + [final_conns]
        # total_conn = torch.sum(torch.stack(all_conns))
        param_count = self.param_count()
        return (k * param_count) / self.n

    def randomize_dropout_biases(self, scale):
        for block in self.blocks:
            block.randomize_dropout_biases(scale)
        c = self.c_final
        self.cb_final = c + scale * torch.randn(c.shape)


    # ────────────────────── helpers for optimisation ────────────────────

    def get_weight_params(self) -> list[nn.Parameter]:
        params = []
        for block in self.blocks:
            params.extend(block.weight_in.parameters())
            params.extend(block.weight_out.parameters())
        params.extend(self.final_layer.parameters())
        return params

    def get_drop_out_params(self) -> list[nn.Parameter]:
        params = []
        for block in self.blocks:
            params.append(block.c_hidden)
            params.append(block.c_out)
        params.append(self.c_final)
        return params


