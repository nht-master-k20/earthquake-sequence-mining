"""
=============================================================================
LSTM MODEL EVALUATION - EARTHQUAKE PREDICTION (PROPER FIX)
=============================================================================

Mục tiêu:
    - Evaluate trained LSTM model
    - Create visualizations
    - Generate analysis report

Proper Fix - Cải tiến:
    - Consistent model architecture with training script
    - Clean visualization code
    - Comprehensive evaluation metrics

Input:
    - models/lstm_model.pth: Trained model
    - models/lstm_config.json: Model configuration
    - data/test_sequences.npz: Test data

Output:
    - models/lstm_evaluation.json: Evaluation metrics
    - models/lstm_plots.png: Visualization plots
    - models/lstm_predictions.csv: Sample predictions

Tác giả: Haind
Ngày tạo: 2025-03-20
Cập nhật: 2025-03-20 (Proper Fix)
=============================================================================
"""

import numpy as np
import json
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


def print_step(step, total, description):
    print(f"\n[{step}/{total}] {description}...")


def print_success(message):
    print(f"  ✓ {message}")


def print_info(message):
    print(f"  ℹ {message}")


# ============================================================================
# DATASET & MODEL (SAME AS TRAINING)
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


class EarthquakeLSTM(nn.Module):
    """Same architecture as training script"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(EarthquakeLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

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
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        pred_time = self.fc_time(last_hidden).squeeze(-1)
        pred_mag = self.fc_mag(last_hidden).squeeze(-1)
        return pred_time, pred_mag


@torch.no_grad()
def predict(model, dataloader, device):
    """Get predictions from model"""
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

    return {
        'pred_time': np.array(pred_times),
        'pred_mag': np.array(pred_mags),
        'true_time': np.array(true_times_orig),
        'true_mag': np.array(true_mags),
        'pred_time_log': np.array(pred_times),
        'true_time_log': np.array(true_times)
    }


def create_plots(results, output_path):
    """Create evaluation plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('LSTM Earthquake Prediction - Evaluation Results', fontsize=16, fontweight='bold')

    # Time: Predicted vs Actual
    ax = axes[0, 0]
    ax.scatter(results['true_time'], results['pred_time'], alpha=0.3, s=5)
    max_val = max(results['true_time'].max(), results['pred_time'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual (days)')
    ax.set_ylabel('Predicted (days)')
    ax.set_title('Time Prediction: Predicted vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Time: Residuals
    ax = axes[0, 1]
    residuals_time = results['pred_time'] - results['true_time']
    ax.scatter(results['pred_time'], residuals_time, alpha=0.3, s=5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted (days)')
    ax.set_ylabel('Residual (days)')
    ax.set_title(f'Time Residuals (MAE={np.mean(np.abs(residuals_time)):.3f})')
    ax.grid(True, alpha=0.3)

    # Time: Error Distribution
    ax = axes[0, 2]
    ax.hist(residuals_time, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    ax.axvline(x=np.mean(residuals_time), color='g', linestyle='-', lw=2, label=f'Mean={np.mean(residuals_time):.2f}')
    ax.set_xlabel('Error (days)')
    ax.set_ylabel('Frequency')
    ax.set_title('Time Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mag: Predicted vs Actual
    ax = axes[1, 0]
    ax.scatter(results['true_mag'], results['pred_mag'], alpha=0.3, s=5)
    min_val, max_val = results['true_mag'].min(), results['true_mag'].max()
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual Magnitude')
    ax.set_ylabel('Predicted Magnitude')
    ax.set_title('Magnitude: Predicted vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mag: Residuals
    ax = axes[1, 1]
    residuals_mag = results['pred_mag'] - results['true_mag']
    ax.scatter(results['pred_mag'], residuals_mag, alpha=0.3, s=5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Magnitude')
    ax.set_ylabel('Residual')
    ax.set_title(f'Magnitude Residuals (MAE={np.mean(np.abs(residuals_mag)):.3f})')
    ax.grid(True, alpha=0.3)

    # Mag: Error Distribution
    ax = axes[1, 2]
    ax.hist(residuals_mag, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    ax.axvline(x=np.mean(residuals_mag), color='g', linestyle='-', lw=2, label=f'Mean={np.mean(residuals_mag):.3f}')
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Magnitude Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print(" LSTM EVALUATION - PROPER FIX ".center(70))
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ============================================================================
    # 1. LOAD MODEL & DATA
    # ============================================================================
    print_step(1, 4, "Loading model and data")

    model_dir = Path('/home/haind/Desktop/earthquake-sequence-mining/haind/models')
    data_dir = Path('/home/haind/Desktop/earthquake-sequence-mining/haind/data')

    # Load config
    with open(model_dir / 'lstm_config.json', 'r') as f:
        config = json.load(f)

    # Load test data
    test_data = np.load(data_dir / 'test_sequences.npz')
    X_test, y_test_time, y_test_mag = test_data['X'], test_data['y_time'], test_data['y_mag']

    # Create dataloader
    test_dataset = EarthquakeDataset(X_test, y_test_time, y_test_mag)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Load model
    model = EarthquakeLSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    model.load_state_dict(torch.load(model_dir / 'lstm_model.pth'))
    model.eval()

    print_success("Loaded model and data")
    print_info(f"Test samples: {len(X_test):,}")

    # ============================================================================
    # 2. GET PREDICTIONS
    # ============================================================================
    print_step(2, 4, "Getting predictions")

    results = predict(model, test_loader, device)

    # Calculate metrics
    r2_time = r2_score(results['true_time_log'], results['pred_time_log'])
    mae_time = mean_absolute_error(results['true_time'], results['pred_time'])
    rmse_time = np.sqrt(mean_squared_error(results['true_time'], results['pred_time']))

    r2_mag = r2_score(results['true_mag'], results['pred_mag'])
    mae_mag = mean_absolute_error(results['true_mag'], results['pred_mag'])
    rmse_mag = np.sqrt(mean_squared_error(results['true_mag'], results['pred_mag']))

    print_success("Calculated metrics")

    # ============================================================================
    # 3. CREATE PLOTS
    # ============================================================================
    print_step(3, 4, "Creating visualization plots")

    plot_path = model_dir / 'lstm_evaluation_plots.png'
    create_plots(results, plot_path)

    print_success(f"Saved plots: {plot_path}")

    # ============================================================================
    # 4. SAVE RESULTS
    # ============================================================================
    print_step(4, 4, "Saving results")

    # Save evaluation metrics
    eval_results = {
        'time': {
            'r2_log': float(r2_time),
            'mae_days': float(mae_time),
            'rmse_days': float(rmse_time)
        },
        'magnitude': {
            'r2': float(r2_mag),
            'mae': float(mae_mag),
            'rmse': float(rmse_mag)
        },
        'num_samples': int(len(results['true_time']))
    }

    with open(model_dir / 'lstm_evaluation.json', 'w') as f:
        json.dump(eval_results, f, indent=2)

    # Save sample predictions
    sample_size = min(1000, len(results['true_time']))
    sample_indices = np.random.choice(len(results['true_time']), sample_size, replace=False)

    predictions_df = pd.DataFrame({
        'true_time_days': results['true_time'][sample_indices],
        'pred_time_days': results['pred_time'][sample_indices],
        'time_error_days': results['pred_time'][sample_indices] - results['true_time'][sample_indices],
        'true_mag': results['true_mag'][sample_indices],
        'pred_mag': results['pred_mag'][sample_indices],
        'mag_error': results['pred_mag'][sample_indices] - results['true_mag'][sample_indices]
    })

    predictions_df.to_csv(model_dir / 'lstm_predictions.csv', index=False)

    print_success("Saved evaluation results")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 70)
    print(" EVALUATION RESULTS ".center(70))
    print("=" * 70)
    print(f"\nTime Prediction:")
    print(f"  R² (log scale):  {r2_time:.4f}")
    print(f"  MAE:            {mae_time:.4f} days")
    print(f"  RMSE:           {rmse_time:.4f} days")
    print(f"\nMagnitude Prediction:")
    print(f"  R²:             {r2_mag:.4f}")
    print(f"  MAE:            {mae_mag:.4f}")
    print(f"  RMSE:           {rmse_mag:.4f}")
    print(f"\nFiles Created:")
    print(f"  - {model_dir / 'lstm_evaluation.json'}")
    print(f"  - {model_dir / 'lstm_evaluation_plots.png'}")
    print(f"  - {model_dir / 'lstm_predictions.csv'}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
