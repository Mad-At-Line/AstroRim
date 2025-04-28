import torch
import torch.nn as nn
from torchsummary import summary

class RIMCell(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv_gate = nn.Sequential(
            nn.Conv2d(1 + 1 + hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, 2, 3, padding=1)
        )
        self.conv_candidate = nn.Sequential(
            nn.Conv2d(1 + 1 + hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        )

    def forward(self, x, grad, h):
        combined = torch.cat([x, grad, h], dim=1)
        gates = torch.sigmoid(self.conv_gate(combined))
        update_gate, reset_gate = gates.chunk(2, dim=1)
        combined_reset = torch.cat([x, grad, reset_gate * h], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_reset))
        return (1 - update_gate) * h + update_gate * candidate

class RIM(nn.Module):
    def __init__(self, n_iter=10, hidden_dim=64):
        super().__init__()
        self.n_iter = n_iter
        self.hidden_dim = hidden_dim
        self.cell = RIMCell(hidden_dim)
        self.final_conv = nn.Conv2d(hidden_dim, 1, 3, padding=1)

    def forward(self, y, forward_operator):
        B, _, H, W = y.shape
        h = torch.zeros(B, self.hidden_dim, H, W, device=y.device)
        x = y.clone()

        for _ in range(self.n_iter):
            if self.training:
                x = x.detach().clone().requires_grad_(True)
                y_sim = forward_operator(x)
                loss = F.mse_loss(y_sim, y)
                grad = torch.autograd.grad(loss, x, create_graph=False)[0]
            else:
                with torch.no_grad():
                    y_sim = forward_operator(x)
                    grad = torch.zeros_like(x)

            h = self.cell(x, grad, h)
            x = self.final_conv(h)

        return x

    def summary_forward(self, x):
        # Dummy forward pass for summary
        return self.final_conv(x[:, :1].repeat(1, self.hidden_dim, 1, 1))

# Updated based on the latest error messages showing:
# - net.9.weight: [64,128,3,3] (expected [1,128,3,3])
# - net.10.* layers exist
# - net.12.* layers exist
class DifferentiableLensing(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 64, 9, padding=4),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            # Second conv block
            nn.Conv2d(64, 128, 7, padding=3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            # Third conv block
            nn.Conv2d(128, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            # Additional blocks based on error messages
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            # Final conv
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

def analyze_model(model_path, model_type='auto'):
    """Analyze a saved PyTorch model and print its architecture"""
    try:
        # Load the state dict
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Handle cases where model is saved in different formats
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Determine model type automatically if not specified
        if model_type == 'auto':
            model_type = guess_model_type(state_dict)
        
        print(f"\n{'='*50}")
        print(f"Analyzing model: {model_path}")
        print(f"Detected model type: {model_type}")
        print(f"{'='*50}\n")
        
        # Reconstruct the model architecture
        if model_type == 'forward_operator':
            model = DifferentiableLensing()
        elif model_type == 'rim':
            model = RIM()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Try to load state dict
        try:
            model.load_state_dict(state_dict)
            print("✅ Successfully loaded state dict (exact match)")
        except Exception as e:
            print(f"⚠️ Could not load exact state dict: {str(e)}")
            print("⚠️ Trying partial load...")
            try:
                model.load_state_dict(state_dict, strict=False)
                print("⚠️ Partial load completed (some layers may not match)")
            except Exception as e:
                print(f"❌ Failed to even partially load state dict: {str(e)}")
                return None
        
        # Print model summary
        print("\nModel Architecture:")
        print(model)
        
        # Print layer details
        print("\nLayer Details:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.shape}")
        
        # Try to print summary with input shape
        try:
            print("\nModel Summary:")
            input_shape = (1, 256, 256)  # Default guess for lensing images
            if model_type == 'rim':
                # Create a temporary forward operator for summary
                dummy_op = DifferentiableLensing()
                summary(model, input_shape, device='cpu', 
                       forward_args=(dummy_op,))
            else:
                summary(model, input_shape, device='cpu')
        except Exception as e:
            print(f"\nCould not generate full summary: {str(e)}")
        
        return model
        
    except Exception as e:
        print(f"\n❌ Error analyzing model: {e}")
        import traceback
        traceback.print_exc()
        return None

def guess_model_type(state_dict):
    """Guess whether the model is a forward operator or RIM based on its layers"""
    rim_keys = ['cell.conv_gate', 'cell.conv_candidate', 'final_conv']
    if any(k in ' '.join(state_dict.keys()) for k in rim_keys):
        return 'rim'
    return 'forward_operator'

if __name__ == '__main__':
    # Define the paths to your model files
    rim_model_path = r"C:\Users\mythi\.astropy\Code\Pipeline4\results\checkpoint_rim_epoch100.pt"
    operator_model_path = r"C:\Users\mythi\.astropy\Code\Pipeline4\results\checkpoint_operator_epoch100.pt"
    
    print("="*80)
    print("Analyzing RIM Model")
    print("="*80)
    rim_model = analyze_model(rim_model_path, 'rim')
    
    print("\n" + "="*80)
    print("Analyzing Forward Operator Model")
    print("="*80)
    operator_model = analyze_model(operator_model_path, 'forward_operator')
    
    print("\nAnalysis complete for both models.")