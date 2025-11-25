import unittest
import torch
import torch.nn as nn
from jl.config import Config
from jl.models import MLP, Resnet, MultiLinear, create_model

class TestFrobeniusReg(unittest.TestCase):
    def test_mlp_linear_equivalent(self):
        # Simple MLP: input=2, h=2, layers=1 -> 1 hidden layer
        # Layers: Input(2->2) -> Hidden(2->2) -> Final(2->1 or 2->2)
        # Actually MLP structure in models.py:
        # if num_layers > 0:
        #   Input (Linear)
        #   Hidden (Linear) * (num_layers - 1)
        # Final (Linear)
        
        c = Config(model_type='mlp', d=2, n_val=10, n=10, batch_size=5, lr=0.01, epochs=1, weight_decay=0.0,
                   num_layers=2, num_class=2, h=2, is_norm=False, frobenius_reg_k=0.1)
        
        model = MLP(c)
        
        # Set weights to known values
        with torch.no_grad():
            # layers[0] is input linear
            model.layers[0].weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]])) # 2*I
            # layers[2] is hidden linear (index 1 is ReLU - skipped) - wait, index 1 is ReLU?
            # Let's check structure: Linear, RMS(opt), ReLU, Linear...
            # is_norm=False. Linear, ReLU, Linear, ReLU...
            # layers[0]: Linear
            # layers[1]: ReLU
            # layers[2]: Linear
            # layers[3]: ReLU
            model.layers[2].weight.copy_(torch.tensor([[3.0, 0.0], [0.0, 3.0]])) # 3*I
            
            # Final layer
            model.final_layer.weight.copy_(torch.tensor([[0.5, 0.5]])) 
            
        # Expected linear product: Final @ Hidden @ Input
        # Final(1x2) @ (3*I) @ (2*I) = Final @ (6*I) = 6 * Final
        # Final = [[0.5, 0.5]]
        # Result = [[3.0, 3.0]]
        # Frobenius norm squared = 3^2 + 3^2 = 18
        
        device = torch.device('cpu')
        A = model.get_linear_equivalent(device)
        expected_A = torch.tensor([[3.0, 3.0]])
        
        self.assertTrue(torch.allclose(A, expected_A), f"Expected {expected_A}, got {A}")
        
        reg = (A ** 2).sum()
        self.assertAlmostEqual(reg.item(), 18.0)

    def test_mlp_squared_path_sums(self):
        # Test squared path sums method for MLP
        c = Config(model_type='mlp', d=2, n_val=10, n=10, batch_size=5, lr=0.01, epochs=1, weight_decay=0.0,
                   num_layers=2, num_class=2, h=2, is_norm=False, frobenius_reg_k=0.1)
        
        model = MLP(c)
        
        # Set weights to known values
        with torch.no_grad():
            model.layers[0].weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]])) # 2*I
            model.layers[2].weight.copy_(torch.tensor([[3.0, 0.0], [0.0, 3.0]])) # 3*I
            model.final_layer.weight.copy_(torch.tensor([[0.5, 0.5]])) 
            
        # Expected squared path sums: Final^2 @ Hidden^2 @ Input^2
        # Input^2: [[4, 0], [0, 4]]
        # Hidden^2: [[9, 0], [0, 9]]
        # Final^2: [[0.25, 0.25]]
        # Result: [[0.25, 0.25]] @ [[9, 0], [0, 9]] @ [[4, 0], [0, 4]]
        #       = [[0.25, 0.25]] @ [[36, 0], [0, 36]]
        #       = [[9.0, 9.0]]
        # Sum = 18.0
        
        device = torch.device('cpu')
        squared_sums = model.get_squared_path_sums(device)
        expected_sum = 18.0
        
        self.assertAlmostEqual(squared_sums.sum().item(), expected_sum)

    def test_resnet_linear_equivalent(self):
        c = Config(model_type='resnet', d=2, n_val=10, n=10, batch_size=5, lr=0.01, epochs=1, weight_decay=0.0,
                   num_layers=1, num_class=2, h=2, is_norm=False, frobenius_reg_k=0.1)
        model = Resnet(c)
        
        # Resnet: InputProj(opt) -> Block( x + W2 W1 x ) -> Final
        # d=2, d_model=None => d_model=2. InputProj is None.
        
        with torch.no_grad():
            # Block 0
            model.blocks[0].weight_in.weight.copy_(torch.eye(2))
            model.blocks[0].weight_out.weight.copy_(torch.eye(2))
            # Block transform: I + I@I = 2*I
            
            # Final layer
            model.final_layer.weight.copy_(torch.tensor([[1.0, 1.0]]))
            
        # Expected: Final @ (2*I) = [[2.0, 2.0]]
        device = torch.device('cpu')
        A = model.get_linear_equivalent(device)
        expected_A = torch.tensor([[2.0, 2.0]])
        self.assertTrue(torch.allclose(A, expected_A), f"Expected {expected_A}, got {A}")

    def test_resnet_squared_path_sums(self):
        # Test squared path sums method for Resnet
        c = Config(model_type='resnet', d=2, n_val=10, n=10, batch_size=5, lr=0.01, epochs=1, weight_decay=0.0,
                   num_layers=1, num_class=2, h=2, is_norm=False, frobenius_reg_k=0.1)
        model = Resnet(c)
        
        with torch.no_grad():
            # Block 0
            model.blocks[0].weight_in.weight.copy_(torch.eye(2))
            model.blocks[0].weight_out.weight.copy_(torch.eye(2))
            # Block transform with squared weights: I + I^2@I^2 = 2*I
            
            # Final layer
            model.final_layer.weight.copy_(torch.tensor([[1.0, 1.0]]))
            
        # Expected squared path sums: Final^2 @ (I + I^2@I^2)
        # I^2 = I (for identity)
        # Block: I + I@I = 2*I
        # Final^2: [[1.0, 1.0]]
        # Result: [[1.0, 1.0]] @ 2*I = [[2.0, 2.0]]
        # Sum = 4.0
        
        device = torch.device('cpu')
        squared_sums = model.get_squared_path_sums(device)
        expected_sum = 4.0
        
        self.assertAlmostEqual(squared_sums.sum().item(), expected_sum)

if __name__ == '__main__':
    unittest.main()


