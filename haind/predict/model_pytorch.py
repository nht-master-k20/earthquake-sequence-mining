"""
PyTorch LSTM Model for Earthquake Prediction
Multi-output model: time_to_next, next_mag, next_mag_binary

Author: haind
Date: 2025-03-25
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm

from config import MODEL_DIR, LOG_DIR, SEQUENCE_LENGTH, BATCH_SIZE


# =============================================================================
# PYTORCH DATASET
# =============================================================================
class EarthquakeDataset(Dataset):
    """PyTorch Dataset for earthquake sequences"""

    def __init__(self, X, y):
        """
        Args:
            X: Input sequences (n_samples, seq_length, n_features)
            y: Targets (n_samples, 3) - [time_to_next, next_mag, next_mag_binary]
        """
        self.X = torch.FloatTensor(X)
        # Ensure y is float type and convert properly
        self.y = torch.FloatTensor(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# PYTORCH LSTM MODEL
# =============================================================================
class EarthquakeLSTM(nn.Module):
    """
    Multi-output LSTM model for earthquake prediction

    Outputs:
        - time_to_next: Regression (seconds)
        - next_mag: Regression (magnitude)
        - next_mag_binary: Classification (M5+ or not)
    """

    def __init__(self, n_features, lstm_hidden=[128, 64], dropout=0.3):
        """
        Initialize LSTM model

        Args:
            n_features: Number of input features
            lstm_hidden: List of hidden units for each LSTM layer
            dropout: Dropout rate
        """
        super(EarthquakeLSTM, self).__init__()

        self.n_features = n_features
        self.lstm_hidden = lstm_hidden

        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        lstm_input_size = n_features

        for i, hidden_size in enumerate(lstm_hidden):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=dropout if i < len(lstm_hidden) - 1 else 0
                )
            )
            lstm_input_size = hidden_size

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Dense layers
        self.fc1 = nn.Linear(lstm_hidden[-1], 64)
        self.relu = nn.ReLU()
        # Removed BatchNorm1d to avoid issues with batch size=1
        # self.bn1 = nn.BatchNorm1d(64)

        # Output heads
        self.output_time = nn.Linear(64, 1)  # Time to next (positive)
        self.output_mag = nn.Linear(64, 1)   # Next magnitude
        self.output_binary = nn.Linear(64, 1) # Binary classification

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, seq_length, n_features)

        Returns:
            Tuple of (time_to_next, next_mag, next_mag_binary)
        """
        batch_size = x.size(0)

        # Pass through LSTM layers
        lstm_out = x
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(lstm_out)

        # Take last timestep output
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # Dense layers
        out = self.fc1(lstm_out)
        out = self.relu(out)
        # Removed BatchNorm1d
        # out = self.bn1(out)
        out = self.dropout(out)

        # Multi-output
        time_to_next = torch.relu(self.output_time(out))  # Ensure positive
        next_mag = self.output_mag(out).squeeze(-1)      # No activation for regression
        next_mag_binary = torch.sigmoid(self.output_binary(out)).squeeze(-1)  # Binary

        return time_to_next.squeeze(-1), next_mag, next_mag_binary


# =============================================================================
# TRAINING CLASS
# =============================================================================
class EarthquakeTrainer:
    """Training class for PyTorch LSTM model"""

    def __init__(self, model, device='cpu', learning_rate=0.001):
        """
        Initialize trainer

        Args:
            model: PyTorch model
            device: 'cpu' or 'cuda'
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate

        # Loss functions
        self.criterion_mse = nn.MSELoss()
        self.criterion_bce = nn.BCELoss()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_time_mae': [],
            'val_time_mae': [],
            'train_mag_mae': [],
            'val_mag_mae': [],
            'train_binary_acc': [],
            'val_binary_acc': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train_epoch(self, train_loader):
        """Train for one epoch with progress bar"""
        self.model.train()

        total_loss = 0
        total_time_mae = 0
        total_mag_mae = 0
        total_binary_correct = 0
        total_samples = 0

        # Progress bar for batches
        pbar = tqdm(train_loader, desc="  Training", leave=False,
                   unit="batch")

        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            time_pred, mag_pred, binary_pred = self.model(X_batch)

            # Calculate losses
            loss_time = self.criterion_mse(time_pred, y_batch[:, 0])
            loss_mag = self.criterion_mse(mag_pred, y_batch[:, 1])
            loss_binary = self.criterion_bce(binary_pred, y_batch[:, 2])

            # Combined loss
            loss = loss_time + loss_mag + loss_binary

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item() * X_batch.size(0)
            total_time_mae += torch.abs(time_pred - y_batch[:, 0]).sum().item()
            total_mag_mae += torch.abs(mag_pred - y_batch[:, 1]).sum().item()
            total_binary_correct += ((binary_pred > 0.5) == (y_batch[:, 2] > 0.5)).sum().item()
            total_samples += X_batch.size(0)

            # Update progress bar with running loss
            pbar.set_postfix({
                'loss': f'{total_loss/total_samples:.2f}',
                'time_mae': f'{total_time_mae/total_samples:.1f}s'
            })

        pbar.close()

        return {
            'loss': total_loss / total_samples,
            'time_mae': total_time_mae / total_samples,
            'mag_mae': total_mag_mae / total_samples,
            'binary_acc': total_binary_correct / total_samples
        }

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()

        total_loss = 0
        total_time_mae = 0
        total_mag_mae = 0
        total_binary_correct = 0
        total_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                time_pred, mag_pred, binary_pred = self.model(X_batch)

                # Calculate losses
                loss_time = self.criterion_mse(time_pred, y_batch[:, 0])
                loss_mag = self.criterion_mse(mag_pred, y_batch[:, 1])
                loss_binary = self.criterion_bce(binary_pred, y_batch[:, 2])

                loss = loss_time + loss_mag + loss_binary

                # Metrics
                total_loss += loss.item() * X_batch.size(0)
                total_time_mae += torch.abs(time_pred - y_batch[:, 0]).sum().item()
                total_mag_mae += torch.abs(mag_pred - y_batch[:, 1]).sum().item()
                total_binary_correct += ((binary_pred > 0.5) == (y_batch[:, 2] > 0.5)).sum().item()
                total_samples += X_batch.size(0)

        return {
            'loss': total_loss / total_samples,
            'time_mae': total_time_mae / total_samples,
            'mag_mae': total_mag_mae / total_samples,
            'binary_acc': total_binary_correct / total_samples
        }

    def train(self, train_loader, val_loader, epochs, early_stopping_patience=15):
        """
        Full training loop with progress bars

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
        """
        print(f"\n{'='*70}")
        print(" TRAINING STARTED")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Train samples: {len(train_loader.dataset):,}")
        print(f"Val samples: {len(val_loader.dataset):,}")
        print(f"{'='*70}\n")

        patience_counter = 0

        # Progress bar for epochs
        from tqdm import tqdm

        pbar = tqdm(range(epochs), desc="Training", unit="epoch")

        for epoch in pbar:
            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_time_mae'].append(train_metrics['time_mae'])
            self.history['val_time_mae'].append(val_metrics['time_mae'])
            self.history['train_mag_mae'].append(train_metrics['mag_mae'])
            self.history['val_mag_mae'].append(val_metrics['mag_mae'])
            self.history['train_binary_acc'].append(train_metrics['binary_acc'])
            self.history['val_binary_acc'].append(val_metrics['binary_acc'])

            # Update progress bar description with current metrics
            pbar.set_description(
                f"Epoch {epoch+1}/{epochs} - "
                f"Val Loss: {val_metrics['loss']:.4f} - "
                f"Val Time MAE: {val_metrics['time_mae']:.0f}s - "
                f"Val Mag MAE: {val_metrics['mag_mae']:.2f} - "
                f"Val Acc: {val_metrics['binary_acc']:.2%}"
            )

            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])

            # Early stopping and best model saving
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                pbar.write(f"  ✓ New best model! (val_loss: {val_metrics['loss']:.4f})")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                pbar.write(f"\nEarly stopping at epoch {epoch+1}")
                pbar.close()
                break

        # Close progress bar
        pbar.close()

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✓ Best model loaded with val_loss: {self.best_val_loss:.4f}")

        return self.history

    def predict(self, X):
        """
        Make predictions

        Args:
            X: Input tensor (batch_size, seq_length, n_features)

        Returns:
            Dictionary with predictions
        """
        self.model.eval()
        X = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            time_pred, mag_pred, binary_pred = self.model(X)

        return {
            'time_to_next': time_pred.cpu().numpy(),
            'next_mag': mag_pred.cpu().numpy(),
            'next_mag_binary': binary_pred.cpu().numpy()
        }

    def save_model(self, filepath=None):
        """Save model checkpoint"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = MODEL_DIR / f'pytorch_model_{timestamp}.pt'

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'n_features': self.model.n_features,
            'lstm_hidden': self.model.lstm_hidden
        }

        torch.save(checkpoint, filepath)
        print(f"Model saved to: {filepath}")

        return filepath

    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Model loaded from: {filepath}")

        return checkpoint


# =============================================================================
# UTILITIES
# =============================================================================
def save_scaler(scaler, filepath=None):
    """Save scaler for preprocessing"""
    import pickle

    if filepath is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = MODEL_DIR / f'scaler_{timestamp}.pkl'

    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Scaler saved to: {filepath}")
    return filepath


def load_scaler(filepath):
    """Load scaler"""
    import pickle

    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)

    print(f"Scaler loaded from: {filepath}")
    return scaler


def save_training_history(history, filepath=None):
    """Save training history to JSON"""
    if filepath is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = MODEL_DIR / f'history_{timestamp}.json'

    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"History saved to: {filepath}")
    return filepath


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("Testing PyTorch LSTM model...")

    from config import INPUT_FEATURES

    n_features = len(INPUT_FEATURES) - 1  # Exclude 'time'

    # Create model
    model = EarthquakeLSTM(n_features=n_features)

    # Test forward pass
    batch_size = 4
    seq_length = 5
    X_dummy = torch.randn(batch_size, seq_length, n_features)

    time_pred, mag_pred, binary_pred = model(X_dummy)

    print(f"\nInput shape: {X_dummy.shape}")
    print(f"Time prediction shape: {time_pred.shape}")
    print(f"Mag prediction shape: {mag_pred.shape}")
    print(f"Binary prediction shape: {binary_pred.shape}")

    print("\n✓ PyTorch model test completed")
