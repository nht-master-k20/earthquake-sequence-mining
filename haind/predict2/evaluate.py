"""
Evaluate trained models on test set with advanced metrics
Binary Classification for Time Model + Regression for Mag Model

Metrics:
- Brier Score (probability prediction)
- ROC-AUC (classification)
- Precision/Recall at different thresholds
- Poisson baseline comparison

Usage:
    python predict2/evaluate.py
    python predict2/evaluate.py --time-model path/to/time_model.pt --mag-model path/to/mag_model.pt

Author: haind
Date: 2025-03-25
"""

import argparse
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    brier_score_loss, roc_curve, precision_recall_curve, confusion_matrix
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import SEQUENCE_LENGTH, TIME_FEATURES, MAG_FEATURES, MODEL_DIR
from models.time_model import TimeLSTM
from models.mag_model import MagLSTM
from dashboard_utils import TrainingDashboard, create_combined_report


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate trained models on test set',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--time-model', type=str, default=None,
                       help='Path to time model')
    parser.add_argument('--mag-model', type=str, default=None,
                       help='Path to mag model')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')

    return parser.parse_args()


def get_device(device_arg='auto'):
    """Get device"""
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def find_latest_models():
    """Find latest trained models"""
    import glob

    time_models = glob.glob(str(MODEL_DIR / 'time_model_*.pt'))
    mag_models = glob.glob(str(MODEL_DIR / 'mag_model_*.pt'))

    if not time_models:
        raise FileNotFoundError("No time model found")
    if not mag_models:
        raise FileNotFoundError("No mag model found")

    time_models.sort(reverse=True)
    mag_models.sort(reverse=True)

    return time_models[0], mag_models[0]


def find_scalers(model_path, model_type):
    """Find corresponding scaler"""
    timestamp = os.path.basename(model_path).replace(f'{model_type}_', '').replace('.pt', '')
    scaler_path = MODEL_DIR / f'{model_type}_scaler_{timestamp}.pkl'

    if os.path.exists(scaler_path):
        return scaler_path
    return None


def load_model_and_scaler(model_path, scaler_path, model_class, device):
    """Load model, scaler, and test indices from checkpoint"""
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = model_class(
        n_features=checkpoint['n_features'],
        lstm_hidden=checkpoint['lstm_hidden']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load scaler from checkpoint (preferred)
    scaler = checkpoint.get('scaler', None)

    # If not in checkpoint, try loading from separate file
    if scaler is None and scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    # Get test indices
    test_indices = checkpoint.get('test_indices', None)

    return model, scaler, checkpoint, test_indices


def compute_binary_metrics(y_true, y_proba, thresholds=[0.3, 0.5, 0.7]):
    """
    Compute comprehensive binary classification metrics

    Args:
        y_true: True labels (0/1)
        y_proba: Predicted probabilities
        thresholds: List of thresholds for precision/recall

    Returns:
        Dictionary of metrics
    """
    # Brier Score
    brier_score = brier_score_loss(y_true, y_proba)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except:
        roc_auc = 0.5

    # Default threshold (0.5) metrics
    y_pred = (y_proba >= 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        'brier_score': brier_score,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'threshold_metrics': {}
    }

    # Metrics at different thresholds
    for thresh in thresholds:
        y_pred_t = (y_proba >= thresh).astype(int)
        prec_t, rec_t, f1_t, _ = precision_recall_fscore_support(
            y_true, y_pred_t, average='binary', zero_division=0
        )
        metrics['threshold_metrics'][f'thresh_{thresh}'] = {
            'precision': prec_t,
            'recall': rec_t,
            'f1': f1_t
        }

    return metrics


def poisson_baseline(y_true, constant_rate=True):
    """
    Poisson process baseline for earthquake prediction

    Args:
        y_true: True binary labels
        constant_rate: If True, use constant rate (positive ratio)
                      If False, use time-varying rate (not implemented)

    Returns:
        Dictionary with baseline metrics
    """
    n = len(y_true)
    n_pos = y_true.sum()

    if constant_rate:
        # Constant rate = positive ratio
        lambda_poisson = n_pos / n
        # All predictions = same probability
        y_proba_poisson = np.full(n, lambda_poisson)
    else:
        # Time-varying rate could be implemented here
        y_proba_poisson = np.full(n, n_pos / n)

    # Compute metrics
    y_pred_poisson = (y_proba_poisson >= 0.5).astype(int)
    accuracy_poisson = accuracy_score(y_true, y_pred_poisson)

    try:
        auc_poisson = roc_auc_score(y_true, y_proba_poisson)
    except:
        auc_poisson = 0.5

    brier_poisson = brier_score_loss(y_true, y_proba_poisson)

    return {
        'accuracy': accuracy_poisson,
        'auc': auc_poisson,
        'brier_score': brier_poisson,
        'rate': n_pos / n,
        'predictions': y_proba_poisson
    }


def print_time_metrics(metrics, baseline_metrics=None):
    """Print time model evaluation results"""
    print(f"\n{'='*80}")
    print(" TIME MODEL RESULTS (Binary Classification)".center(80))
    print(f"{'='*80}")

    print(f"\nPrimary Metrics:")
    print(f"  ROC-AUC:       {metrics['roc_auc']:.4f}")
    print(f"  Brier Score:   {metrics['brier_score']:.4f} (lower is better)")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  F1 Score:      {metrics['f1']:.4f}")

    print(f"\nPrecision/Recall:")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")

    print(f"\nConfusion Matrix (threshold=0.5):")
    print(f"  True Positives:    {metrics['tp']:,}")
    print(f"  True Negatives:    {metrics['tn']:,}")
    print(f"  False Positives:   {metrics['fp']:,}")
    print(f"  False Negatives:   {metrics['fn']:,}")

    print(f"\nPrecision/Recall at Different Thresholds:")
    print(f"  {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 50)
    for thresh_name, thresh_metrics in metrics['threshold_metrics'].items():
        thresh_val = thresh_name.replace('thresh_', '')
        print(f"  {thresh_val:<12} {thresh_metrics['precision']:<12.4f} "
              f"{thresh_metrics['recall']:<12.4f} {thresh_metrics['f1']:<12.4f}")

    if baseline_metrics:
        print(f"\nBaseline Comparison (Poisson Process):")
        print(f"  {'Metric':<20} {'Model':<12} {'Baseline':<12} {'Improvement':<12}")
        print("-" * 60)

        auc_improve = ((metrics['roc_auc'] - baseline_metrics['auc']) /
                       (baseline_metrics['auc'] + 1e-8)) * 100
        brier_improve = ((baseline_metrics['brier_score'] - metrics['brier_score']) /
                         (baseline_metrics['brier_score'] + 1e-8)) * 100

        print(f"  {'ROC-AUC':<20} {metrics['roc_auc']:<12.4f} {baseline_metrics['auc']:<12.4f} "
              f"{auc_improve:+.1f}%")
        print(f"  {'Brier Score':<20} {metrics['brier_score']:<12.4f} {baseline_metrics['brier_score']:<12.4f} "
              f"{brier_improve:+.1f}%")
        print(f"  {'Baseline Rate':<20} {'-':<12} {baseline_metrics['rate']:<12.4f} {'-':<12}")


def evaluate_on_test():
    """Evaluate both models on test set with advanced metrics"""

    print(f"\n{'='*80}")
    print(" TEST SET EVALUATION - COMPREHENSIVE REPORT ".center(80))
    print(f"{'='*80}")

    # Get device
    device = get_device()

    # Find models
    time_model_path, mag_model_path = find_latest_models()

    # Find scalers
    time_scaler_path = find_scalers(time_model_path, 'time_model')
    mag_scaler_path = find_scalers(mag_model_path, 'mag_model')

    print(f"\nModels:")
    print(f"  Time: {os.path.basename(time_model_path)}")
    print(f"  Mag:  {os.path.basename(mag_model_path)}")

    # Load models and test indices from checkpoint
    print(f"\nLoading models...")
    time_model, time_scaler, time_checkpoint, time_test_indices = load_model_and_scaler(
        time_model_path, time_scaler_path, TimeLSTM, device
    )
    mag_model, mag_scaler, mag_checkpoint, mag_test_indices = load_model_and_scaler(
        mag_model_path, mag_scaler_path, MagLSTM, device
    )

    # Verify test_indices exist in checkpoint
    if time_test_indices is None:
        raise ValueError(f"time_model checkpoint missing 'test_indices'. Please retrain the model.")
    if mag_test_indices is None:
        raise ValueError(f"mag_model checkpoint missing 'test_indices'. Please retrain the model.")

    print(f"  Time test indices: {len(time_test_indices)} samples")
    print(f"  Mag test indices: {len(mag_test_indices)} samples")

    # Load test data using test_indices from checkpoint
    print(f"\nLoading test data using test_indices from checkpoint...")
    from data.time_data import TimeDataPreparer, TimeDataset as TimeDataset
    from data.mag_data import MagDataPreparer, MagDataset as MagDataset
    from config import FEATURES_TIME_FILE, FEATURES_MAG_FILE, TIME_TARGET, MAG_TARGET

    # Prepare time data - load all sequences, then extract test set using indices
    time_preparer = TimeDataPreparer()
    time_preparer.load_data()
    X_time, y_time = time_preparer.prepare_sequences(time_preparer.data, for_training=True)

    # Use test_indices to get exact test set
    X_time_test = X_time[time_test_indices]
    y_time_test = y_time[time_test_indices]

    # Prepare mag data - load all sequences, then extract test set using indices
    mag_preparer = MagDataPreparer()
    mag_preparer.load_data()
    X_mag, y_mag = mag_preparer.prepare_sequences(mag_preparer.data, for_training=True)

    # Use test_indices to get exact test set
    X_mag_test = X_mag[mag_test_indices]
    y_mag_test = y_mag[mag_test_indices]

    # Scale test data using scalers from checkpoint
    print(f"  Scaling test data using scalers from checkpoint...")

    # Reshape test data for scaling
    time_shape = X_time_test.shape
    X_time_test_2d = X_time_test.reshape(-1, X_time_test.shape[-1])
    X_time_test_scaled = time_scaler.transform(X_time_test_2d)
    X_time_test_scaled = X_time_test_scaled.reshape(time_shape)

    mag_shape = X_mag_test.shape
    X_mag_test_2d = X_mag_test.reshape(-1, X_mag_test.shape[-1])
    X_mag_test_scaled = mag_scaler.transform(X_mag_test_2d)
    X_mag_test_scaled = X_mag_test_scaled.reshape(mag_shape)

    print(f"  Time test samples: {len(X_time_test_scaled)}")
    print(f"  Mag test samples: {len(X_mag_test_scaled)}")

    # Show class distribution for time model
    pos_count = y_time_test.sum()
    neg_count = len(y_time_test) - pos_count
    print(f"\n  Test set class distribution:")
    print(f"    Positive: {pos_count:,} ({pos_count/len(y_time_test):.2%})")
    print(f"    Negative: {neg_count:,} ({neg_count/len(y_time_test):.2%})")

    # Create dataloaders
    time_test_dataset = TimeDataset(X_time_test_scaled, y_time_test)
    mag_test_dataset = MagDataset(X_mag_test_scaled, y_mag_test)

    time_test_loader = DataLoader(time_test_dataset, batch_size=64, shuffle=False)
    mag_test_loader = DataLoader(mag_test_dataset, batch_size=64, shuffle=False)

    # ============================================
    # EVALUATE TIME MODEL (Binary Classification)
    # ============================================
    print(f"\n{'='*80}")
    print(" EVALUATING TIME MODEL (Binary Classification)".center(80))
    print(f"{'='*80}")

    time_model.eval()
    y_time_logits = []
    with torch.no_grad():
        for X_batch, _ in time_test_loader:
            X_batch = X_batch.to(device)
            logits = time_model(X_batch)
            y_time_logits.extend(logits.cpu().numpy())

    y_time_logits = np.array(y_time_logits)
    y_time_proba = 1 / (1 + np.exp(-y_time_logits))  # Sigmoid

    # Compute comprehensive metrics
    time_metrics = compute_binary_metrics(y_time_test, y_time_proba)

    # Compute Poisson baseline
    baseline_metrics = poisson_baseline(y_time_test)

    # Print results
    print_time_metrics(time_metrics, baseline_metrics)

    # ============================================
    # EVALUATE MAG MODEL (Regression)
    # ============================================
    print(f"\n{'='*80}")
    print(" EVALUATING MAG MODEL (Regression)".center(80))
    print(f"{'='*80}")

    mag_model.eval()
    y_mag_pred = []
    with torch.no_grad():
        for X_batch, _ in mag_test_loader:
            X_batch = X_batch.to(device)
            pred = mag_model(X_batch)
            y_mag_pred.extend(pred.cpu().numpy())

    y_mag_pred = np.array(y_mag_pred)

    # Mag metrics
    mag_mae = np.mean(np.abs(y_mag_pred - y_mag_test))
    mag_rmse = np.sqrt(np.mean((y_mag_pred - y_mag_test) ** 2))
    mag_mse = np.mean((y_mag_pred - y_mag_test) ** 2)

    print(f"Test MAE:  {mag_mae:.3f}")
    print(f"Test RMSE: {mag_rmse:.3f}")

    mag_test_metrics = {
        'loss': mag_mse,
        'mae': mag_mae,
        'rmse': mag_rmse
    }

    # ============================================
    # CREATE VISUALIZATIONS
    # ============================================
    print(f"\n{'='*80}")
    print(" CREATING VISUALIZATIONS ".center(80))
    print(f"{'='*80}")

    output_dir = Path('predict2/dashboard')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Time model visualizations
    time_dashboard = TrainingDashboard('TimeLSTM-Binary', output_dir=str(output_dir))

    # ROC Curve
    save_roc_curve(y_time_test, y_time_proba, output_dir, timestamp)

    # Precision-Recall Curve
    save_precision_recall_curve(y_time_test, y_time_proba, output_dir, timestamp)

    # Mag model visualizations
    mag_dashboard = TrainingDashboard('MagLSTM', output_dir=str(output_dir))
    mag_dashboard.save_comparison_plot(y_mag_test, y_mag_pred, 'Test')
    mag_dashboard.save_error_distribution(y_mag_test, y_mag_pred, 'Test')

    # Combined comparison
    print(f"\nCreating combined report...")
    # Note: create_combined_report expects loss/mae/rmse, so we adapt
    time_metrics_for_dashboard = {
        'loss': time_metrics['brier_score'],
        'mae': 1 - time_metrics['accuracy'],
        'rmse': 1 - time_metrics['roc_auc']
    }
    create_combined_report(time_dashboard, mag_dashboard, time_metrics_for_dashboard, mag_test_metrics)

    # Create metrics table for slides
    create_metrics_table_for_slides(time_metrics, mag_test_metrics, baseline_metrics,
                                    time_checkpoint, mag_checkpoint,
                                    output_dir, timestamp)

    print(f"\n{'='*80}")
    print(" EVALUATION COMPLETED ".center(80))
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - {output_dir}/plots/roc_curve_{timestamp}.png")
    print(f"  - {output_dir}/plots/precision_recall_curve_{timestamp}.png")
    print(f"  - {output_dir}/plots/test_metrics_table_{timestamp}.png")
    print()


def save_roc_curve(y_true, y_proba, output_dir, timestamp):
    """Save ROC curve plot"""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Earthquake Prediction (7 days)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    filepath = output_dir / 'plots' / f'roc_curve_{timestamp}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def save_precision_recall_curve(y_true, y_proba, output_dir, timestamp):
    """Save Precision-Recall curve"""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AP = {ap:.4f})')

    # Baseline (positive ratio)
    positive_ratio = y_true.mean()
    ax.axhline(y=positive_ratio, color='r', linestyle='--', linewidth=1,
               label=f'Baseline ({positive_ratio:.2%})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve - Earthquake Prediction (7 days)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    filepath = output_dir / 'plots' / f'precision_recall_curve_{timestamp}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def create_metrics_table_for_slides(time_metrics, mag_test_metrics, baseline_metrics,
                                    time_checkpoint, mag_checkpoint,
                                    output_dir, timestamp):
    """Create a formatted metrics table for presentation slides"""

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    # Get training info
    time_epochs = len(time_checkpoint['history']['train_loss'])
    mag_epochs = len(mag_checkpoint['history']['train_loss'])

    # Prepare table data
    table_data = [
        ['Metric', 'TimeLSTM (Binary)', 'Poisson Baseline', 'MagLSTM (Regression)', 'Unit'],
        ['=' * 35, '=' * 30, '=' * 30, '=' * 30, '=' * 10],
        # Training info
        ['Epochs Trained', f"{time_epochs}", '-', f"{mag_epochs}", ''],
        ['Best Val Loss', f"{time_checkpoint['best_val_loss']:.4f}", '-',
         f"{mag_checkpoint['best_val_loss']:.4f}", ''],
        ['-' * 35, '-' * 30, '-' * 30, '-' * 30, '-' * 10],
        # Time model metrics
        ['ROC-AUC', f"{time_metrics['roc_auc']:.4f}", f"{baseline_metrics['auc']:.4f}", '-', 'score'],
        ['Brier Score', f"{time_metrics['brier_score']:.4f}", f"{baseline_metrics['brier_score']:.4f}", '-', 'lower=better'],
        ['Accuracy (t=0.5)', f"{time_metrics['accuracy']:.4f}", f"{baseline_metrics['accuracy']:.4f}", '-', 'score'],
        ['Precision', f"{time_metrics['precision']:.4f}", '-', '-', 'score'],
        ['Recall', f"{time_metrics['recall']:.4f}", '-', '-', 'score'],
        ['F1 Score', f"{time_metrics['f1']:.4f}", '-', '-', 'score'],
        ['-' * 35, '-' * 30, '-' * 30, '-' * 30, '-' * 10],
        # Time model: thresholds
        ['Precision @ t=0.3', f"{time_metrics['threshold_metrics']['thresh_0.3']['precision']:.4f}", '-', '-', 'score'],
        ['Recall @ t=0.3', f"{time_metrics['threshold_metrics']['thresh_0.3']['recall']:.4f}", '-', '-', 'score'],
        ['Precision @ t=0.7', f"{time_metrics['threshold_metrics']['thresh_0.7']['precision']:.4f}", '-', '-', 'score'],
        ['Recall @ t=0.7', f"{time_metrics['threshold_metrics']['thresh_0.7']['recall']:.4f}", '-', '-', 'score'],
        ['-' * 35, '-' * 30, '-' * 30, '-' * 30, '-' * 10],
        # Mag model metrics
        ['MAE (magnitude)', '-', '-', f"{mag_test_metrics['mae']:.3f}", 'mag'],
        ['RMSE (magnitude)', '-', '-', f"{mag_test_metrics['rmse']:.3f}", 'mag'],
        ['-' * 35, '-' * 30, '-' * 30, '-' * 30, '-' * 10],
        # Model architecture
        ['Input Features', f"{time_checkpoint['n_features']}", '-', f"{mag_checkpoint['n_features']}", ''],
        ['LSTM Layers', '2', '-', '2', ''],
        ['Hidden Units', f"{time_checkpoint['lstm_hidden']}", '-', f"{mag_checkpoint['lstm_hidden']}", ''],
    ]

    table = ax.table(cellText=table_data, cellLoc='center',
                     loc='center', colWidths=[0.22, 0.20, 0.18, 0.20, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#1F4E78')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style separator rows
    for i, row in enumerate(table_data):
        if row[0] in ['=' * 35, '-' * 35]:
            for j in range(5):
                table[(i, j)].set_facecolor('#D9E1F2')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0 and table_data[i-1][0] not in ['=' * 35, '-' * 35]:
            for j in range(5):
                table[(i, j)].set_facecolor('#F2F2F2')

    ax.set_title('Model Performance Comparison - Test Set (Binary + Regression)',
                fontsize=16, fontweight='bold', pad=20,
                color='#1F4E78')

    # Save
    filepath = output_dir / 'plots' / f'test_metrics_table_{timestamp}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filepath}")

    return filepath


def main():
    """Main function"""
    args = parse_args()

    evaluate_on_test()


if __name__ == "__main__":
    main()
