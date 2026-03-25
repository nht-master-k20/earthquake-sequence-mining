"""
Dashboard and Visualization Utilities for Training
Save training curves, metrics, and test results for reports

Author: haind
Date: 2025-03-25
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class TrainingDashboard:
    """Dashboard for tracking and visualizing training progress"""

    def __init__(self, model_type, output_dir=None):
        """
        Initialize dashboard

        Args:
            model_type: 'TimeLSTM' or 'MagLSTM'
            output_dir: Directory to save outputs
        """
        self.model_type = model_type
        self.output_dir = Path(output_dir) if output_dir else Path('predict2/dashboard')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)

        self.metrics_dir = self.output_dir / 'metrics'
        self.metrics_dir.mkdir(exist_ok=True)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_acc': [],
            'val_acc': [],
            'train_auc': [],
            'val_auc': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rate': [],
            'epoch_time': []
        }

        # Best metrics
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        # Timestamp
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def update(self, epoch, train_metrics, val_metrics, lr, epoch_time):
        """Update dashboard with epoch metrics"""
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['val_loss'].append(val_metrics['loss'])

        # Always update all available metrics
        for key in ['mae', 'rmse', 'acc', 'auc', 'f1']:
            if key in train_metrics:
                self.history[f'train_{key}'].append(train_metrics[key])
            if key in val_metrics:
                self.history[f'val_{key}'].append(val_metrics[key])

        self.history['learning_rate'].append(lr)
        self.history['epoch_time'].append(epoch_time)

        # Update best
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            self.best_epoch = epoch

    def save_training_curves(self):
        """Save training curves plot"""
        # Detect if binary classification or regression (check if lists have data)
        has_acc = bool(self.history.get('train_acc'))
        has_auc = bool(self.history.get('train_auc'))

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.model_type} Training Curves', fontsize=16, fontweight='bold')

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
        axes[0, 0].axvline(self.best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch ({self.best_epoch})')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss over Epochs')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        if has_acc and has_auc:
            # Binary classification: Accuracy and AUC
            axes[0, 1].plot(epochs, self.history['train_acc'], 'b-o', label='Train Acc', linewidth=2)
            axes[0, 1].plot(epochs, self.history['val_acc'], 'r-s', label='Val Acc', linewidth=2)
            axes[0, 1].axvline(self.best_epoch, color='g', linestyle='--', alpha=0.5)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Accuracy over Epochs')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].plot(epochs, self.history['train_auc'], 'b-o', label='Train AUC', linewidth=2)
            axes[1, 0].plot(epochs, self.history['val_auc'], 'r-s', label='Val AUC', linewidth=2)
            axes[1, 0].axvline(self.best_epoch, color='g', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].set_title('AUC over Epochs')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            # Regression: MAE and RMSE
            axes[0, 1].plot(epochs, self.history['train_mae'], 'b-o', label='Train MAE', linewidth=2)
            axes[0, 1].plot(epochs, self.history['val_mae'], 'r-s', label='Val MAE', linewidth=2)
            axes[0, 1].axvline(self.best_epoch, color='g', linestyle='--', alpha=0.5)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].set_title('MAE over Epochs')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].plot(epochs, self.history['train_rmse'], 'b-o', label='Train RMSE', linewidth=2)
            axes[1, 0].plot(epochs, self.history['val_rmse'], 'r-s', label='Val RMSE', linewidth=2)
            axes[1, 0].axvline(self.best_epoch, color='g', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('RMSE')
            axes[1, 0].set_title('RMSE over Epochs')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate
        axes[1, 1].plot(epochs, self.history['learning_rate'], 'g-o', label='Learning Rate', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')

        plt.tight_layout()

        # Save
        filepath = self.plots_dir / f'{self.model_type}_training_curves_{self.timestamp}.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return filepath

    def save_metrics_table(self, test_metrics=None):
        """Save metrics summary table as image"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')

        # Detect if binary classification or regression based on test_metrics
        is_binary = test_metrics and 'acc' in test_metrics

        # Prepare data
        if self.model_type == 'TimeLSTM':
            unit = 'seconds'
        else:
            unit = 'magnitude'

        # Build metrics data
        metrics_data = []
        metrics_data.append(['Train Loss', f"{self.history['train_loss'][-1]:.2f}"])
        metrics_data.append(['Val Loss', f"{self.history['val_loss'][-1]:.2f}"])
        metrics_data.append(['Best Val Loss', f"{self.best_val_loss:.2f}"])
        metrics_data.append(['Best Epoch', f"{self.best_epoch}"])

        if is_binary:
            # Binary classification metrics
            metrics_data.append(['Train Accuracy', f"{self.history['train_acc'][-1]:.3f}"])
            metrics_data.append(['Val Accuracy', f"{self.history['val_acc'][-1]:.3f}"])
            metrics_data.append(['Train AUC', f"{self.history['train_auc'][-1]:.3f}"])
            metrics_data.append(['Val AUC', f"{self.history['val_auc'][-1]:.3f}"])
        else:
            # Regression metrics
            metrics_data.append(['Train MAE', f"{self.history['train_mae'][-1]:.2f} {unit}"])
            metrics_data.append(['Val MAE', f"{self.history['val_mae'][-1]:.2f} {unit}"])
            metrics_data.append(['Train RMSE', f"{self.history['train_rmse'][-1]:.2f} {unit}"])
            metrics_data.append(['Val RMSE', f"{self.history['val_rmse'][-1]:.2f} {unit}"])

        if test_metrics:
            metrics_data.append(['-' * 20, '-' * 20])
            metrics_data.append(['Test Loss', f"{test_metrics['loss']:.2f}"])
            if is_binary:
                metrics_data.append(['Test Accuracy', f"{test_metrics['acc']:.3f}"])
                metrics_data.append(['Test AUC', f"{test_metrics['auc']:.3f}"])
                metrics_data.append(['Test F1', f"{test_metrics['f1']:.3f}"])
            else:
                metrics_data.append(['Test MAE', f"{test_metrics['mae']:.2f} {unit}"])
                metrics_data.append(['Test RMSE', f"{test_metrics['rmse']:.2f} {unit}"])

        # Create table
        table = ax.table(cellText=metrics_data, colWidths=[0.4, 0.4],
                         cellLoc='left', loc='center',
                         colLabels=['Metric', 'Value'])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)

        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(metrics_data) + 1):
            if i % 2 == 0:
                for j in range(2):
                    table[(i, j)].set_facecolor('#E7E6E6')

        ax.set_title(f'{self.model_type} Metrics Summary',
                    fontsize=14, fontweight='bold', pad=20)

        # Save
        filepath = self.plots_dir / f'{self.model_type}_metrics_table_{self.timestamp}.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return filepath

    def save_comparison_plot(self, y_true, y_pred, dataset='Test'):
        """Save prediction vs actual comparison plot"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{self.model_type} - {dataset} Set: Predicted vs Actual',
                    fontsize=14, fontweight='bold')

        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=30)
        axes[0].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)],
                    'r--', linewidth=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Scatter Plot')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Add metrics text
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

        textstr = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.3f}'
        axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Residual plot
        residuals = y_pred - y_true
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=30)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        filepath = self.plots_dir / f'{self.model_type}_{dataset.lower()}_comparison_{self.timestamp}.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return filepath

    def save_error_distribution(self, y_true, y_pred, dataset='Test'):
        """Save error distribution plot"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{self.model_type} - {dataset} Set: Error Distribution',
                    fontsize=14, fontweight='bold')

        errors = y_pred - y_true
        abs_errors = np.abs(errors)

        # Error histogram
        axes[0].hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].axvline(x=np.mean(errors), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
        axes[0].set_xlabel('Error (Predicted - Actual)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Error Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Absolute error histogram
        axes[1].hist(abs_errors, bins=50, alpha=0.7, color='coral', edgecolor='black')
        axes[1].axvline(x=np.mean(abs_errors), color='r', linestyle='--', linewidth=2,
                       label=f'MAE: {np.mean(abs_errors):.2f}')
        axes[1].axvline(x=np.median(abs_errors), color='g', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(abs_errors):.2f}')
        axes[1].set_xlabel('Absolute Error')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Absolute Error Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        filepath = self.plots_dir / f'{self.model_type}_{dataset.lower()}_error_dist_{self.timestamp}.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return filepath

    def save_metrics_json(self, test_metrics=None):
        """Save all metrics to JSON file"""
        # Build training metrics dict - only include non-empty lists
        training_metrics = {
            'epochs_trained': len(self.history['train_loss']),
            'best_epoch': self.best_epoch,
            'best_val_loss': float(self.best_val_loss),
            'final_train_loss': float(self.history['train_loss'][-1]),
            'final_val_loss': float(self.history['val_loss'][-1]),
        }

        # Add optional metrics if available
        for key in ['mae', 'rmse', 'acc', 'auc', 'f1']:
            if self.history[f'train_{key}']:
                training_metrics[f'final_train_{key}'] = float(self.history[f'train_{key}'][-1])
            if self.history[f'val_{key}']:
                training_metrics[f'final_val_{key}'] = float(self.history[f'val_{key}'][-1])

        metrics = {
            'model_type': self.model_type,
            'timestamp': self.timestamp,
            'training': training_metrics,
            'history': {k: [float(v) for v in vals] for k, vals in self.history.items()}
        }

        if test_metrics:
            test_dict = {'loss': float(test_metrics['loss'])}
            for key in ['mae', 'rmse', 'acc', 'auc', 'f1']:
                if key in test_metrics:
                    test_dict[key] = float(test_metrics[key])
            metrics['test'] = test_dict

        # Save
        filepath = self.metrics_dir / f'{self.model_type}_metrics_{self.timestamp}.json'
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        return filepath

    def create_summary_report(self, test_metrics=None):
        """Create a summary report with all plots"""
        print(f"\n{'='*70}")
        print(f" CREATING DASHBOARD - {self.model_type} ".center(70))
        print(f"{'='*70}")

        plot_files = []

        # 1. Training curves
        print("Saving training curves...")
        filepath = self.save_training_curves()
        plot_files.append(filepath)
        print(f"  Saved: {filepath}")

        # 2. Metrics table
        print("Saving metrics table...")
        filepath = self.save_metrics_table(test_metrics)
        plot_files.append(filepath)
        print(f"  Saved: {filepath}")

        # 3. Metrics JSON
        print("Saving metrics JSON...")
        filepath = self.save_metrics_json(test_metrics)
        print(f"  Saved: {filepath}")

        print(f"\n{'='*70}")
        print(f" DASHBOARD CREATED ".center(70))
        print(f"{'='*70}")
        print(f"Output directory: {self.output_dir}")

        return plot_files


def create_combined_report(time_dashboard, mag_dashboard, time_test_metrics, mag_test_metrics):
    """Create combined report for both models"""

    output_dir = time_dashboard.output_dir

    # Create summary comparison table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Prepare comparison data
    comparison_data = [
        ['Metric', 'TimeLSTM', 'MagLSTM'],
        ['-' * 20, '-' * 20, '-' * 20],
        ['Train MAE', f"{time_dashboard.history['train_mae'][-1]:.1f}s",
         f"{mag_dashboard.history['train_mae'][-1]:.3f}"],
        ['Val MAE', f"{time_dashboard.history['val_mae'][-1]:.1f}s",
         f"{mag_dashboard.history['val_mae'][-1]:.3f}"],
        ['Test MAE', f"{time_test_metrics['mae']:.1f}s",
         f"{mag_test_metrics['mae']:.3f}"],
        ['Train RMSE', f"{time_dashboard.history['train_rmse'][-1]:.1f}s",
         f"{mag_dashboard.history['train_rmse'][-1]:.3f}"],
        ['Val RMSE', f"{time_dashboard.history['val_rmse'][-1]:.1f}s",
         f"{mag_dashboard.history['val_rmse'][-1]:.3f}"],
        ['Test RMSE', f"{time_test_metrics['rmse']:.1f}s",
         f"{mag_test_metrics['rmse']:.3f}"],
        ['-' * 20, '-' * 20, '-' * 20],
        ['Best Epoch', f"{time_dashboard.best_epoch}",
         f"{mag_dashboard.best_epoch}"],
        ['Epochs Trained', f"{len(time_dashboard.history['train_loss'])}",
         f"{len(mag_dashboard.history['train_loss'])}"],
    ]

    table = ax.table(cellText=comparison_data, cellLoc='left',
                     loc='center', colWidths=[0.3, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(comparison_data) + 1):
        if i % 2 == 0:
            for j in range(3):
                table[(i, j)].set_facecolor('#E7E6E6')

    ax.set_title('Model Comparison Summary',
                fontsize=16, fontweight='bold', pad=20)

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = output_dir / 'plots' / f'model_comparison_{timestamp}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nCombined comparison saved: {filepath}")

    return filepath
