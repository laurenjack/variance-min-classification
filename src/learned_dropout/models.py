import torch
from torch import nn
import torch.nn.functional as F

from model_tracker import MLPTracker, ResNetTracker


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
    def __init__(self, d, h, relus, layer_norm):
        """
        d: Dimension of the residual stream.
        h: Hidden dimension for this block.
        relus: If True, apply ReLU after the weight-in linear layer.

        Each block contains:
          - A weight_in layer: transforms from d to h.
          - A weight_out layer: transforms from h to d.
          - Three learned dropout parameters (c parameters):
              • c_in: applied to the block input (shape: d)
              • c_hidden: applied after the weight_in layer (shape: h)
              • c_out: applied after the weight_out layer (shape: d)
        """
        super(ResidualBlock, self).__init__()
        self.relus = relus
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.layer_norm_layer = nn.LayerNorm(d)
        self.weight_in = nn.Linear(d, h, bias=False)
        self.weight_out = nn.Linear(h, d, bias=False)

        # Learned dropout (keep probability) parameters.
        # Note: no dropout is applied globally on the network input.
        self.c_in = nn.Parameter(torch.zeros(d))  # For the block’s input.
        self.c_hidden = nn.Parameter(torch.zeros(h))  # For the hidden activation.
        self.c_out = nn.Parameter(torch.zeros(d))  # For the block output.


    def forward_network1(self, x):
        """
        Non-differentiable dropout:
          - For each dropout position, sample a binary mask using the detached keep probabilities.
          - The order is: first apply dropout to the block input, then after weight_in (and optional ReLU),
            then after weight_out.
          - Finally, the block output is returned.
        """
        n_samples = x.size(0)
        if self.layer_norm:
            x = self.layer_norm_layer(x)
        # 1. Apply dropout to the block input.
        p_in = 0.8 * torch.sigmoid(self.c_in.detach()) + 0.2
        mask_in = torch.bernoulli(p_in.expand(n_samples, x.size(1)))
        x_dropped = x * mask_in

        # 2. Compute hidden activation.
        hidden = self.weight_in(x_dropped)
        if self.relus:
            hidden = F.relu(hidden)
        # Apply dropout on hidden activation.
        p_hidden = torch.sigmoid(self.c_hidden.detach())
        mask_hidden = torch.bernoulli(p_hidden.expand(n_samples, hidden.size(1)))
        hidden_dropped = hidden * mask_hidden

        # 3. Compute block output.
        out = self.weight_out(hidden_dropped)
        # Apply dropout on block output.
        p_out = torch.sigmoid(self.c_out.detach())
        mask_out = torch.bernoulli(p_out.expand(n_samples, out.size(1)))
        out_dropped = out * mask_out

        return out_dropped

    def forward_network2(self, x):
        """
        Differentiable dropout:
          - Instead of sampling masks, scale each activation by the keep probabilities (computed via sigmoid)
          - Weight matrices are used in a detached manner.
        """
        if self.layer_norm:
            x = self.layer_norm_layer(x)
        x_scaled = x * torch.sigmoid(self.c_in) / 0.8
        hidden = F.linear(x_scaled, self.weight_in.weight.detach())
        if self.relus:
            hidden = F.relu(hidden)
        hidden = hidden * torch.sigmoid(self.c_hidden)
        out = F.linear(hidden, self.weight_out.weight.detach())
        out = out * torch.sigmoid(self.c_out)
        return out

    def effective_param_count(self):
        """
        Computes the effective parameter count for this block.
        Here we consider the two weight matrices.
          - For weight_in: effective count = sum(sigmoid(c_in)) * sum(sigmoid(c_hidden))
          - For weight_out: effective count = sum(sigmoid(c_hidden)) * sum(sigmoid(c_out))
        The block's effective parameter count is the sum of these two contributions.
        """
        count_in = torch.sum(torch.sigmoid(self.c_in)) * torch.sum(torch.sigmoid(self.c_hidden))
        count_out = torch.sum(torch.sigmoid(self.c_hidden)) * torch.sum(torch.sigmoid(self.c_out))
        return count_in + count_out


class ResNet(nn.Module):
    def __init__(self, d, n, h_list, relus, layer_norm):
        """
        d: Dimension of the residual stream.
        n: Total number of training examples (used to scale the variance term).
        h_list: List of hidden dimensions for each residual block.
                The number of blocks is len(h_list).
        relus: Whether to apply ReLU activation after the weight_in linear layer.

        The network consists of a series of residual blocks and a final layer.
          - There is no dropout at the global input.
          - Each residual block applies dropout to its own input, hidden activation, and output.
          - After all blocks, a final linear layer (d -> 1) is applied.
            This final layer uses a learned dropout on its input (c_final).
        """
        super(ResNet, self).__init__()
        self.d = d
        self.n = n  # total number of training examples
        self.L = len(h_list)  # number of residual blocks
        self.blocks = nn.ModuleList([ResidualBlock(d, h, relus, layer_norm) for h in h_list])

        # Final linear layer mapping from d to 1.
        self.final_layer = nn.Linear(d, 1, bias=False)
        # Learned dropout parameter for the input of the final layer.
        self.c_final = nn.Parameter(torch.zeros(d))

    @staticmethod
    def get_tracker(track_weights):
        return ResNetTracker(track_weights)

    def forward_network1(self, x):
        """
        Forward pass using non-differentiable (sampled) dropout.
          - The input x is fed directly into the first residual block.
          - Each residual block computes its output with dropout and that output
            is added to the residual stream.
          - After all blocks, a final dropout mask (sampled from c_final) is applied
            to the residual stream before the final linear layer produces the output.
        """
        current = x  # No dropout applied to the very beginning.
        for block in self.blocks:
            block_out = block.forward_network1(current)
            current = current + block_out

        n_samples = current.size(0)
        # Apply dropout for final layer input.
        p_final = torch.sigmoid(self.c_final.detach())
        mask_final = torch.bernoulli(p_final.expand(n_samples, self.d))
        final_input = current * mask_final
        output = self.final_layer(final_input).squeeze(1)
        return output

    def forward_network2(self, x):
        """
        Forward pass using differentiable dropout:
          - Each dropout position scales the activation by the corresponding keep probability.
          - Weight matrices are used in a detached manner.
        """
        current = x  # Start with the raw input.
        for block in self.blocks:
            block_out = block.forward_network2(current)
            current = current + block_out

        # Apply differentiable dropout to final layer input.
        final_input = current * torch.sigmoid(self.c_final)
        output = F.linear(final_input, self.final_layer.weight.detach()).squeeze(1)
        return output

    def var_network2(self, k):
        """
        Computes a variance term proportional to the effective parameter count.
          - For each residual block, the effective parameter count is computed
            as described in ResidualBlock.effective_param_count().
          - The final layer's effective parameter count is the sum of its input keep probabilities:
              • count_final = sum(sigmoid(c_final))
          - The total is scaled by the constant k and divided by the total number of training examples n.
        """
        total_count = 0.0
        for block in self.blocks:
            total_count = total_count + block.effective_param_count()

        # Final layer effective parameter count:
        count_final = torch.sum(torch.sigmoid(self.c_final))
        total_count = total_count + count_final

        return k * total_count / self.n

    def get_weight_params(self):
        """
        Returns a list of all weight parameters from the ResNet.
        It collects the weights from all residual blocks (both weight_in and weight_out)
        and from the final linear layer.
        """
        params = []
        for block in self.blocks:
            params += list(block.weight_in.parameters())
            params += list(block.weight_out.parameters())
        params += list(self.final_layer.parameters())
        return params

    def get_drop_out_params(self):
        """
        Returns a list of all learned dropout (c) parameters for the ResNet.
        It collects the dropout parameters from each residual block (c_in, c_hidden, c_out)
        and the dropout parameter for the final layer (c_final).
        """
        params = []
        for block in self.blocks:
            params.append(block.c_in)
            params.append(block.c_hidden)
            params.append(block.c_out)
        params.append(self.c_final)
        return params


