"""
=============================================================================
LSTM MODEL - EARTHQUAKE PREDICTION (PROPER FIX)
=============================================================================

Mục tiêu:
    - Train LSTM model to predict time & magnitude of next earthquake
    - Clean architecture with proper evaluation

Proper Fix - Cải tiến:
    - Simplified but effective LSTM architecture
    - Consistent model design across training and evaluation
    - Better training stability

Input:
    - data/train_sequences.npz, val_sequences.npz, test_sequences.npz

Output:
    - models/lstm_model.pth: Trained model
    - models/lstm_config.json: Model configuration
    - models/lstm_results.json: Training results

Tác giả: Haind
Ngày tạo: 2025-03-20
Cập nhật: 2025-03-20 (Proper Fix)
=============================================================================
"""

import numpy as np
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tqdm import tqdm


def print_step(step, total, description):
    print(f"\n[{step}/{total}] {description}...")


def print_success(message):
    print(f"  ✓ {message}")


def print_info(message):
    print(f"  ℹ {message}")


# ============================================================================
# DATASET
# ============================================================================
class EarthquakeDataset(Dataset):
    def __init__(self, sequences, time_targets, mag_targets):
        self.X = torch.FloatTensor(sequences)
        self.y_time = torch.FloatTensor(time_targets)
        self.y_mag = torch.FloatTensor(mag_targets)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_time[idx], self.y_mag[idx]


# ============================================================================
# LSTM MODEL
# ============================================================================
class EarthquakeLSTM(nn.Module):
    """
    Simple but effective LSTM for earthquake prediction
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(EarthquakeLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layers - 2 separate heads
        self.fc_time = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.fc_mag = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # LSTM: (batch, seq_len, features) -> (batch, seq_len, hidden)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)

        # Dropout
        last_hidden = self.dropout(last_hidden)

        # Predictions
        pred_time = self.fc_time(last_hidden).squeeze(-1)  # (batch,)
        pred_mag = self.fc_mag(last_hidden).squeeze(-1)    # (batch,)

        return pred_time, pred_mag


# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    pred_times, pred_mags = [], []
    true_times, true_mags = [], []

    criterion = nn.MSELoss()

    for X_batch, y_time_batch, y_mag_batch in dataloader:
        X_batch = X_batch.to(device)
        y_time_batch = y_time_batch.to(device)
        y_mag_batch = y_mag_batch.to(device)

        # Forward
        pred_time, pred_mag = model(X_batch)

        # Loss
        loss_time = criterion(pred_time, y_time_batch)
        loss_mag = criterion(pred_mag, y_mag_batch)
        loss = loss_time + loss_mag

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # Store predictions
        pred_times.extend(pred_time.detach().cpu().numpy())
        pred_mags.extend(pred_mag.detach().cpu().numpy())
        true_times.extend(y_time_batch.cpu().numpy())
        true_mags.extend(y_mag_batch.cpu().numpy())

    mae_time = mean_absolute_error(true_times, pred_times)
    mae_mag = mean_absolute_error(true_mags, pred_mags)

    return total_loss / len(dataloader), mae_time, mae_mag


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    pred_times, pred_mags = [], []
    true_times, true_mags = [], []

    for X_batch, y_time_batch, y_mag_batch in dataloader:
        X_batch = X_batch.to(device)

        pred_time, pred_mag = model(X_batch)

        pred_times.extend(pred_time.cpu().numpy())
        pred_mags.extend(pred_mag.cpu().numpy())
        true_times.extend(y_time_batch.numpy())
        true_mags.extend(y_mag_batch.numpy())

    # Convert back from log scale
    pred_times_orig = np.expm1(pred_times)
    true_times_orig = np.exp(true_times) - 1

    # Metrics
    r2_time = r2_score(true_times, pred_times)
    mae_time = mean_absolute_error(true_times_orig, pred_times_orig)
    rmse_time = np.sqrt(mean_squared_error(true_times_orig, pred_times_orig))

    r2_mag = r2_score(true_mags, pred_mags)
    mae_mag = mean_absolute_error(true_mags, pred_mags)
    rmse_mag = np.sqrt(mean_squared_error(true_mags, pred_mags))

    return {
        'time': {'r2': r2_time, 'mae': mae_time, 'rmse': rmse_time},
        'mag': {'r2': r2_mag, 'mae': mae_mag, 'rmse': rmse_mag}
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print(" LSTM MODEL - PROPER FIX ".center(70))
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ============================================================================
    # 1. LOAD DATA
    # ============================================================================
    print_step(1, 5, "Loading sequences")

    data_dir = Path('/home/haind/Desktop/earthquake-sequence-mining/haind/data')

    train_data = np.load(data_dir / 'train_sequences.npz')
    X_train, y_train_time, y_train_mag = train_data['X'], train_data['y_time'], train_data['y_mag']

    val_data = np.load(data_dir / 'val_sequences.npz')
    X_val, y_val_time, y_val_mag = val_data['X'], val_data['y_time'], val_data['y_mag']

    test_data = np.load(data_dir / 'test_sequences.npz')
    X_test, y_test_time, y_test_mag = test_data['X'], test_data['y_time'], test_data['y_mag']

    print_success("Loaded sequences")
    print_info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    print_info(f"Shape: {X_train.shape}")

    # ============================================================================
    # 2. CREATE DATALOADERS
    # ============================================================================
    print_step(2, 5, "Creating dataloaders")

    BATCH_SIZE = 256

    train_dataset = EarthquakeDataset(X_train, y_train_time, y_train_mag)
    val_dataset = EarthquakeDataset(X_val, y_val_time, y_val_mag)
    test_dataset = EarthquakeDataset(X_test, y_test_time, y_test_mag)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    print_success("Created dataloaders")
    print_info(f"Batch size: {BATCH_SIZE}")

    # ============================================================================
    # 3. CREATE MODEL
    # ============================================================================
    print_step(3, 5, "Creating model")

    INPUT_SIZE = X_train.shape[2]
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3

    model = EarthquakeLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    print_success("Created model")
    print_info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ============================================================================
    # 4. TRAIN
    # ============================================================================
    print_step(4, 5, "Training")

    NUM_EPOCHS = 50
    PATIENCE = 10

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_mae_time': [], 'val_mae_mag': []}

    print(f"Training for max {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_mae_time, train_mae_mag = train_epoch(
            model, train_loader, optimizer, device
        )

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(train_loss)

        history['train_loss'].append(train_loss)
        history['val_mae_time'].append(val_metrics['time']['mae'])
        history['val_mae_mag'].append(val_metrics['mag']['mae'])

        val_loss = val_metrics['time']['mae'] + val_metrics['mag']['mae']

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: "
              f"Loss={train_loss:.4f}, "
              f"Val Time MAE={val_metrics['time']['mae']:.4f}, "
              f"Val Mag MAE={val_metrics['mag']['mae']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            model_dir = Path('/home/haind/Desktop/earthquake-sequence-mining/haind/models')
            model_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_dir / 'lstm_model.pth')
            print_success("  → Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ============================================================================
    # 5. EVALUATE ON TEST SET
    # ============================================================================
    print_step(5, 5, "Final evaluation")

    model.load_state_dict(torch.load('/home/haind/Desktop/earthquake-sequence-mining/haind/models/lstm_model.pth'))
    test_metrics = evaluate(model, test_loader, device)

    # Save results
    model_dir = Path('/home/haind/Desktop/earthquake-sequence-mining/haind/models')

    config = {
        'input_size': INPUT_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'batch_size': BATCH_SIZE
    }

    results = {
        'test': {
            'time_r2': float(test_metrics['time']['r2']),
            'time_mae': float(test_metrics['time']['mae']),
            'time_rmse': float(test_metrics['time']['rmse']),
            'mag_r2': float(test_metrics['mag']['r2']),
            'mag_mae': float(test_metrics['mag']['mae']),
            'mag_rmse': float(test_metrics['mag']['rmse'])
        },
        'history': history
    }

    with open(model_dir / 'lstm_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    with open(model_dir / 'lstm_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print_success("Saved config and results")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 70)
    print(" TEST RESULTS ".center(70))
    print("=" * 70)
    print(f"\nTime Prediction:")
    print(f"  R² (log):    {test_metrics['time']['r2']:.4f}")
    print(f"  MAE (days):  {test_metrics['time']['mae']:.4f}")
    print(f"  RMSE (days): {test_metrics['time']['rmse']:.4f}")
    print(f"\nMagnitude Prediction:")
    print(f"  R²:    {test_metrics['mag']['r2']:.4f}")
    print(f"  MAE:   {test_metrics['mag']['mae']:.4f}")
    print(f"  RMSE:  {test_metrics['mag']['rmse']:.4f}")
    print(f"\nFiles:")
    print(f"  - {model_dir / 'lstm_model.pth'}")
    print(f"  - {model_dir / 'lstm_config.json'}")
    print(f"  - {model_dir / 'lstm_results.json'}")
    print(f"\nNext step:")
    print(f"  python 05_lstm_evaluation.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
