import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import json
from model import ImprovedOptimizedCNN, count_parameters

class MNISTTrainer:
    def __init__(self, model, device='cpu', learning_rate=0.003, batch_size=128):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.NLLLoss()  # Using NLLLoss with log_softmax
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.1)
        self.batch_size = batch_size
        self.training_logs = []
        
    def prepare_data(self):
        """Load and prepare MNIST dataset with required split"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load full datasets
        full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        # Split training set: 50k train, 10k validation as per assignment
        train_dataset, val_dataset = random_split(full_train, [50000, 10000])
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        return train_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        """Evaluate on validation set"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return val_loss / len(self.val_loader), 100. * correct / total
    
    def train(self, epochs=20, target_accuracy=99.4):
        """Complete training loop"""
        print("Starting training...")
        print(f"Target: {target_accuracy}% accuracy in <{epochs} epochs")
        
        best_accuracy = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Log training metrics
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': round(train_loss, 4),
                'train_accuracy': round(train_acc, 2),
                'val_loss': round(val_loss, 4),
                'val_accuracy': round(val_acc, 2),
                'learning_rate': round(self.optimizer.param_groups[0]['lr'], 6)
            }
            self.training_logs.append(log_entry)
            
            print(f"Epoch {epoch+1:2d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            self.scheduler.step()
            
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                self.save_model('best_model.pth')
            
            # Early stopping if target reached
            if val_acc >= target_accuracy:
                print(f"Target accuracy {target_accuracy}% reached at epoch {epoch+1}!")
                break
        
        print(f"Training completed. Best validation accuracy: {best_accuracy:.2f}%")
        return best_accuracy
    
    def save_logs(self, filename='training_logs.json'):
        """Save training logs to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.training_logs, f, indent=2)
        print(f"Training logs saved to {filename}")
    
    def save_model(self, filename='final_model.pth'):
        """Save model state dict"""
        torch.save(self.model.state_dict(), filename)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model and check parameters
    model = ImprovedOptimizedCNN(dropout_rate=0.1)
    total_params = count_parameters(model)
    
    print(f"\nModel Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Parameter constraint (<20k): {'PASS' if total_params < 20000 else 'FAIL'}")
    
    # Setup trainer and data
    trainer = MNISTTrainer(model, device, learning_rate=0.003, batch_size=128)
    trainer.prepare_data()
    
    # Train model
    best_accuracy = trainer.train(epochs=20, target_accuracy=99.4)
    
    # Save results
    trainer.save_logs()
    trainer.save_model()
    
    print(f"\nTraining Summary:")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Target achieved: {'YES' if best_accuracy >= 99.4 else 'NO'}")
    print(f"Parameter count: {total_params:,}/20,000")

if __name__ == "__main__":
    main()
