# In train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import json
from model import EnhancedMNISTCNN, count_parameters

class MNISTTrainer:
    def __init__(self, model, device='cpu', learning_rate=0.002, batch_size=128):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.criterion = nn.NLLLoss()
        
        # Using Adam optimizer which was part of the successful 99.31% run
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Using the reliable StepLR scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        
        self.training_logs = []
        
    def prepare_data(self):
        """Load and prepare MNIST dataset with slightly enhanced, stable augmentations."""
        
        # --- CHANGE 1: Slightly stronger but stable augmentation ---
        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        full_train = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
        
        train_dataset, val_dataset = random_split(full_train, [50000, 10000])
        
        # Make sure the validation set does not use augmentations
        val_dataset.dataset.transform = test_transform
        
        use_pin_memory = torch.cuda.is_available()
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=use_pin_memory)
        self.val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=use_pin_memory)
        self.test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=use_pin_memory)
        
    def train_epoch(self):
        """Train for one epoch with simple logging."""
        self.model.train()
        train_loss, correct, total = 0, 0, 0
        
        # --- CHANGE 2: Removed tqdm for cleaner logs ---
        for data, target in self.train_loader:
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
        """Evaluate on validation set."""
        self.model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100. * correct / len(self.val_loader.dataset)
        return avg_loss, accuracy

    def train(self, epochs=20, target_accuracy=99.4):
        """Complete training loop."""
        print("Starting training with refined strategy...")
        best_accuracy = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()
            
            # Simple, clean log for each epoch
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            log_entry = { 'epoch': epoch + 1, 'train_loss': round(train_loss, 4), 'train_accuracy': round(train_acc, 2),
                          'val_loss': round(val_loss, 4), 'val_accuracy': round(val_acc, 2), 'learning_rate': self.optimizer.param_groups[0]['lr']}
            self.training_logs.append(log_entry)
            
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                print(f"  -> New best validation accuracy! Saving model to 'best_model.pth'")
                self.save_model('best_model.pth')
            
            if val_acc >= target_accuracy:
                print(f"\nTARGET ACCURACY {target_accuracy}% REACHED!")
                break
        
        print(f"\nTraining completed. Best validation accuracy: {best_accuracy:.2f}%")
        return best_accuracy

    def save_logs(self, filename='training_logs.json'):
        with open(filename, 'w') as f: json.dump(self.training_logs, f, indent=2)
        print(f"Training logs saved to {filename}")
    
    def save_model(self, filename='final_model.pth'):
        torch.save(self.model.state_dict(), filename)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = EnhancedMNISTCNN(dropout_rate=0.05)
    total_params = count_parameters(model)
    
    print(f"\nModel Information: Total parameters: {total_params:,}")
    print(f"Parameter constraint (<20k): {'PASS' if total_params < 20000 else 'FAIL'}")
    
    # --- CHANGE 3: Higher initial learning rate for a stronger start ---
    trainer = MNISTTrainer(model, device, learning_rate=0.002, batch_size=128)
    trainer.prepare_data()
    best_accuracy = trainer.train(epochs=20, target_accuracy=99.4)
    
    trainer.save_logs()
    
    print(f"\n--- FINAL SUMMARY ---")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"Target Achieved: {'YES' if best_accuracy >= 99.4 else 'NO'}")
    print(f"Parameter Count: {total_params:,}/20,000")

if __name__ == "__main__":
    main()