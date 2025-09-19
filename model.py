import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ImprovedOptimizedCNN(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(ImprovedOptimizedCNN, self).__init__()
        
        # Initial feature extraction block
        # Input: 28x28x1 -> Output: 26x26x16
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Dimension reduction and channel transition
        # Input: 26x26x16 -> Output: 13x13x10
        self.transition1 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=1, padding=0, bias=False),
            nn.MaxPool2d(2, 2)
        )
        
        # Enhanced feature extraction
        # Input: 13x13x10 -> Output: 11x11x16
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Final feature refinement
        # Input: 11x11x16 -> Output: 9x9x16
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Global Average Pooling for spatial dimension reduction
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final classification layer using 1x1 convolution
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=1, padding=0, bias=False)
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.transition1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.gap(x)
        x = self.conv_block4(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

def count_parameters(model):
    """Count total parameters in the model"""
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedOptimizedCNN().to(device)
    
    # Display model architecture
    summary(model, input_size=(1, 28, 28))
    
    # Check parameter constraint
    total_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Parameter constraint (<20k): {'PASS' if total_params < 20000 else 'FAIL'}")
