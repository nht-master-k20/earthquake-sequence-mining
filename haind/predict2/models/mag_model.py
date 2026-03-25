"""
MagLSTM Model for Earthquake Magnitude Prediction
Regression: Predict next earthquake magnitude

Author: haind
Date: 2025-03-25
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import json
import time

from sklearn.metrics import mean_absolute_error, mean_squared_error

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_DIR, LOG_DIR


class MagDataset(Dataset):
    """PyTorch Dataset for magnitude prediction"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MagLSTM(nn.Module):
    """LSTM model for earthquake magnitude prediction (regression)"""

    def __init__(self, n_features, lstm_hidden=[128, 64], dropout=0.3):
        super(MagLSTM, self).__init__()
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

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden[-1], 64)
        self.relu = nn.ReLU()
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        # Pass through LSTM layers
        lstm_out = x
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(lstm_out)

        # Take last timestep output
        lstm_out = lstm_out[:, -1, :]

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # Dense layers
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout(out)

        # Output (magnitude)
        mag = self.output(out)
        return mag.squeeze(-1)


class MagTrainer:
    """Training class for MagLSTM model (Regression)"""

    def __init__(self, model, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate

        # Loss function - MSE for regression
        self.criterion = nn.MSELoss()

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
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': [],
            'learning_rate': [],
            'epoch_time': []
        }

        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.best_epoch = 0

    def compute_metrics(self, y_true, y_pred):
        """Compute MAE, RMSE from predictions and true values"""
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        mae = mean_absolute_error(y_true_np, y_pred_np)
        rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))

        return mae, rmse

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        all_preds = []
        all_labels = []
        total_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            pred = self.model(X_batch)
            loss = self.criterion(pred, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            all_preds.append(pred.detach())
            all_labels.append(y_batch.detach())
            total_samples += X_batch.size(0)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        mae, rmse = self.compute_metrics(all_labels, all_preds)

        return {
            'loss': total_loss / total_samples,
            'mae': mae,
            'rmse': rmse
        }

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()

        total_loss = 0
        all_preds = []
        all_labels = []
        total_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)

                total_loss += loss.item() * X_batch.size(0)
                all_preds.append(pred)
                all_labels.append(y_batch)
                total_samples += X_batch.size(0)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        mae, rmse = self.compute_metrics(all_labels, all_preds)

        return {
            'loss': total_loss / total_samples,
            'mae': mae,
            'rmse': rmse
        }

    def train(self, train_loader, val_loader, epochs, early_stopping_patience=15):
        """Full training loop"""
        print(f"\n{'='*80}")
        print(" MAG MODEL TRAINING (Regression)".center(80))
        print(f"{'='*80}")
        print(f"{'Device:':<15} {self.device}")
        print(f"{'Epochs:':<15} {epochs}")
        print(f"{'Batch Size:':<15} {train_loader.batch_size}")
        print(f"{'Train Samples:':<15} {len(train_loader.dataset):,}")
        print(f"{'Val Samples:':<15} {len(val_loader.dataset):,}")
        print(f"{'Learning Rate:':<15} {self.learning_rate}")
        print(f"{'Patience:':<15} {early_stopping_patience}")
        print(f"{'='*80}\n")

        patience_counter = 0
        start_time = time.time()

        print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Train MAE':<10} "
              f"{'Val MAE':<10} {'Val RMSE':<10} {'LR':<10} {'Time':<8}")
        print("-" * 80)

        for epoch in range(epochs):
            epoch_start = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['train_rmse'].append(train_metrics['rmse'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)

            is_best = val_metrics['loss'] < self.best_val_loss
            best_marker = " *" if is_best else ""

            print(f"{epoch+1:<6} {train_metrics['loss']:<12.4f} {val_metrics['loss']:<12.4f} "
                  f"{train_metrics['mae']:<10.3f} {val_metrics['mae']:<10.3f} "
                  f"{val_metrics['rmse']:<10.3f} {current_lr:<10.2e} {epoch_time:<6.1f}s{best_marker}")

            self.scheduler.step(val_metrics['loss'])

            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                self.best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                total_time = time.time() - start_time
                print(f"\n{'='*80}")
                print(f" EARLY STOPPING at epoch {epoch+1}".center(80))
                print(f"{'='*80}")
                break

        total_time = time.time() - start_time

        print(f"\n{'='*80}")
        print(" TRAINING SUMMARY".center(80))
        print(f"{'='*80}")
        print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Epochs Completed: {epoch + 1}/{epochs}")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Val MAE: {self.history['val_mae'][self.best_epoch-1]:.3f}")
        print(f"Best Val RMSE: {self.history['val_rmse'][self.best_epoch-1]:.3f}")

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        print(f"{'='*80}\n")

        return self.history

    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        X = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            pred = self.model(X)

        return pred.cpu().numpy()

    def save_model(self, filepath=None, scaler=None, test_indices=None, metadata=None):
        """Save model checkpoint"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = MODEL_DIR / f'mag_model_{timestamp}.pt'

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'n_features': self.model.n_features,
            'lstm_hidden': self.model.lstm_hidden,
            'model_type': 'MagLSTM',
            'scaler': scaler,
            'test_indices': test_indices,
            'metadata': metadata
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


def save_training_history(history, filepath=None):
    """Save training history to JSON"""
    if filepath is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = MODEL_DIR / f'mag_history_{timestamp}.json'

    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"History saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    print("Testing MagLSTM model (Regression)...")

    n_features = 26
    model = MagLSTM(n_features=n_features)

    batch_size = 4
    seq_length = 5
    X_dummy = torch.randn(batch_size, seq_length, n_features)

    pred = model(X_dummy)

    print(f"\nInput shape: {X_dummy.shape}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Predictions: {pred.detach().numpy()}")

    print("\nMagLSTM model test completed")
