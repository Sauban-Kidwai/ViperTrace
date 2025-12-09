"""
ViperTrace Model Training Script (PyTorch)

Trains a dual-input CNN model for malware detection:
- Input 1: 512-byte sequences (1D CNN)
- Input 2: Shannon entropy (scalar)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time
from process_data import generate_dataset


class ViperNet(nn.Module):
    """
    Dual-input CNN architecture for malware detection
    
    Inputs:
        - x_bytes: (batch_size, 512) - Normalized byte sequences
        - x_entropy: (batch_size, 1) - Shannon entropy values
    
    Output:
        - (batch_size, 1) - Malware probability [0-1]
    """
    
    def __init__(self, chunk_size: int = 512):
        super(ViperNet, self).__init__()
        
        # 1D CNN for byte sequence processing
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_conv = nn.Dropout(0.3)
        
        # Calculate flattened size after convolutions
        # After 3 pooling layers: 512 -> 256 -> 128 -> 64
        self.cnn_output_size = 128 * (chunk_size // 8)
        
        # Fully connected layers (CNN output + entropy)
        self.fc1 = nn.Linear(self.cnn_output_size + 1, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.dropout_fc = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x_bytes, x_entropy):
        """
        Forward pass with dual inputs
        
        Args:
            x_bytes: (batch_size, 512) - Byte sequences
            x_entropy: (batch_size, 1) - Entropy values
        
        Returns:
            (batch_size, 1) - Malware probability
        """
        # Reshape for 1D CNN: (batch, channels=1, sequence_length=512)
        x = x_bytes.unsqueeze(1)
        
        # Convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout_conv(x)
        
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout_conv(x)
        
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout_conv(x)
        
        # Flatten CNN output
        x = x.view(x.size(0), -1)
        
        # Concatenate with entropy
        x = torch.cat([x, x_entropy], dim=1)
        
        # Fully connected layers
        x = self.dropout_fc(self.relu(self.fc1(x)))
        x = self.dropout_fc(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        
        return x


class MalwareDataset(Dataset):
    """PyTorch Dataset for malware detection"""
    
    def __init__(self, X_bytes, X_entropy, y_labels):
        self.X_bytes = torch.FloatTensor(X_bytes)
        self.X_entropy = torch.FloatTensor(X_entropy)
        self.y_labels = torch.FloatTensor(y_labels)
    
    def __len__(self):
        return len(self.X_bytes)
    
    def __getitem__(self, idx):
        return self.X_bytes[idx], self.X_entropy[idx], self.y_labels[idx]


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_bytes, X_entropy, y_labels in dataloader:
        X_bytes = X_bytes.to(device)
        X_entropy = X_entropy.to(device)
        y_labels = y_labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_bytes, X_entropy)
        loss = criterion(outputs, y_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = (outputs > 0.5).float()
        correct += (predictions == y_labels).sum().item()
        total += y_labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_bytes, X_entropy, y_labels in dataloader:
            X_bytes = X_bytes.to(device)
            X_entropy = X_entropy.to(device)
            y_labels = y_labels.to(device)
            
            outputs = model(X_bytes, X_entropy)
            loss = criterion(outputs, y_labels)
            
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == y_labels).sum().item()
            total += y_labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_model(epochs: int = 20, batch_size: int = 32, learning_rate: float = 0.001):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Generate or load data
    print("\n📦 Generating training data...")
    X_bytes_train, X_entropy_train, y_train = generate_dataset(n_samples=5000, seed=42)
    
    print("📦 Generating validation data...")
    X_bytes_val, X_entropy_val, y_val = generate_dataset(n_samples=1000, seed=123)
    
    # Create datasets and dataloaders
    train_dataset = MalwareDataset(X_bytes_train, X_entropy_train, y_train)
    val_dataset = MalwareDataset(X_bytes_val, X_entropy_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Batch size: {batch_size}")
    
    # Initialize model
    print("\n🔧 Creating ViperNet model...")
    model = ViperNet(chunk_size=512).to(device)
    
    # Print model architecture
    print("\n📊 Model Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📈 Total parameters: {total_params:,}")
    print(f"📈 Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n🚀 Starting training...\n")
    print("=" * 70)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch [{epoch+1:2d}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'vipertrace.pth')
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.4f})")
    
    print("=" * 70)
    print(f"\n✅ Training complete!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    print(f"💾 Model saved as: vipertrace.pth")
    
    return model


if __name__ == "__main__":
    print("=" * 70)
    print("                  ViperTrace Model Training (PyTorch)")
    print("=" * 70)
    
    # Train the model
    model = train_model(epochs=25, batch_size=32, learning_rate=0.001)
    
    print("\n" + "=" * 70)
    print("✅ All done!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. The model is saved as 'vipertrace.pth'")
    print("2. Run 'python main.py' or 'uvicorn main:app --reload' to start the API")
    print("3. The API will automatically load this model on startup")

