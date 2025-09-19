import torch
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EnhancedMNISTCNN(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(EnhancedMNISTCNN, self).__init__()
        
        # Initial feature extraction with wider channels
        # Input: 28x28x1 -> Output: 26x26x16
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Deeper feature extraction with increased channels
        # Input: 26x26x16 -> Output: 24x24x32
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Transition block with pooling and channel reduction
        # Input: 24x24x32 -> MaxPool -> 12x12x32 -> 1x1 Conv -> 12x12x16
        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # Enhanced feature extraction on reduced feature map
        # Input: 12x12x16 -> Output: 10x10x32
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Deep feature interaction layer
        # Input: 10x10x32 -> Output: 8x8x32
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Global Average Pooling for spatial dimension reduction
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final classification layer using 1x1 convolution
        self.classifier = nn.Conv2d(32, 10, kernel_size=1, padding=0, bias=False)
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.transition1(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.gap(x)
        x = self.classifier(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

def print_model_summary(model):
    """Print a simple model summary"""
    print("Model Architecture:")
    print("=" * 50)
    total_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and hasattr(module, 'weight'):
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"{name}: {params:,} parameters")
                total_params += params
    
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    return total_params

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedMNISTCNN().to(device)
    
    total_params = print_model_summary(model)
    print(f"Parameter constraint (<20k): {'PASS' if total_params < 20000 else 'FAIL'}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
