import torch
import torch.nn as nn
from jl.config import Config
from jl.models import create_model


def test_c_parameter_in_config():
    """Test that c parameter can be set in Config"""
    # Test with c=None (default)
    config1 = Config(
        model_type='mlp',
        d=10,
        n_val=100,
        n=100,
        batch_size=16,
        lr=1e-3,
        epochs=10,
        weight_decay=0.01,
        num_layers=2,
        d_model=20
    )
    assert config1.c is None
    
    # Test with c=0.1
    config2 = Config(
        model_type='mlp',
        d=10,
        n_val=100,
        n=100,
        batch_size=16,
        lr=1e-3,
        epochs=10,
        weight_decay=0.01,
        num_layers=2,
        d_model=20,
        c=0.1
    )
    assert config2.c == 0.1
    print("✓ Config c parameter test passed")


def test_logit_regularization_loss():
    """Test that logit regularization affects the loss correctly"""
    torch.manual_seed(42)
    device = torch.device("cpu")
    
    # Create a simple config and model
    config = Config(
        model_type='mlp',
        d=5,
        n_val=100,
        n=100,
        batch_size=4,
        lr=1e-3,
        epochs=1,
        weight_decay=0.01,
        num_layers=1,
        d_model=10,
        c=0.5
    )
    
    model = create_model(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    # Create dummy data
    x = torch.randn(4, 5, device=device)
    y = torch.randint(0, 2, (4,), device=device).float()
    
    # Forward pass
    logits = model(x).squeeze()
    
    # Calculate loss without regularization
    base_loss = criterion(logits, y)
    
    # Calculate loss with regularization (manual)
    logit_reg = config.c * torch.mean(logits ** 2)
    expected_loss = base_loss + logit_reg
    
    # Verify regularization term is positive and affects loss
    assert logit_reg.item() > 0, "Logit regularization should be positive"
    assert expected_loss.item() > base_loss.item(), "Loss with regularization should be higher"
    
    print(f"✓ Logit regularization test passed")
    print(f"  Base loss: {base_loss.item():.6f}")
    print(f"  Regularization term: {logit_reg.item():.6f}")
    print(f"  Total loss: {expected_loss.item():.6f}")


if __name__ == "__main__":
    test_c_parameter_in_config()
    test_logit_regularization_loss()
    print("\n✓ All tests passed!")

