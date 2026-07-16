# src/visualization.py
"""
Advanced visualization functions for SDSS classification analysis.

Features:
  - Distribution plots
  - Confusion matrix heatmaps
  - ROC curves
  - Learning curves
  - Feature importance plots
  - Manifold projections
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def set_style(style='seaborn-v0_8-darkgrid', palette='husl'):
    """Set matplotlib and seaborn styling."""
    plt.style.use(style)
    sns.set_palette(palette)


def save_figure(fig, filepath, dpi=300, bbox_inches='tight'):
    """Save figure to file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    print(f"✓ Figure saved: {filepath}")


def plot_class_distribution(data, target_col, save_path=None):
    """Plot class distribution."""
    class_counts = data[target_col].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    class_counts.plot(kind='bar', ax=axes[0], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0].set_title('Class Distribution (Counts)', fontweight='bold')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=0)
    
    axes[1].pie(class_counts, labels=class_counts.index, autopct='%1.1f%%',
               colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1].set_title('Class Distribution (%)', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    plt.show()


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', 
                         normalized=False, save_path=None):
    """Plot confusion matrix heatmap."""
    if normalized:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.3f'
    else:
        cm_display = cm
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues', ax=ax,
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Count' if not normalized else 'Proportion'})
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc, class_name, save_path=None):
    """Plot single ROC curve."""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})', color='#2E86AB')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curve - {class_name}', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    plt.show()


def plot_learning_curves(history, save_path=None):
    """Plot training and validation curves."""
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = range(1, len(history_df) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Accuracy
    axes[0].plot(history_df['epoch'], history_df['accuracy'], 'o-', 
               label='training', linewidth=2)
    axes[0].plot(history_df['epoch'], history_df['val_accuracy'], 's--',
               label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history_df['epoch'], history_df['loss'], 'o-',
               label='Training', linewidth=2)
    axes[1].plot(history_df['epoch'], history_df['val_loss'], 's--',
               label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    plt.show()


def plot_feature_importance(importance_df, top_n=15, save_path=None):
    """Plot feature importance bars."""
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_features)), top_features['importance_mean'],
           xerr=top_features['importance_std'], color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Most Important Features', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    plt.show()


def plot_model_comparison(results_dict, metric='accuracy', save_path=None):
    """Compare multiple models."""
    models = list(results_dict.keys())
    metrics_val = [results_dict[m][metric] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette('husl', len(models))
    bars = ax.bar(range(len(models)), metrics_val, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel(metric.title())
    ax.set_title(f'Model Comparison - {metric.title()}', fontweight='bold')
    ax.set_ylim([0.7, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, metrics_val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    plt.show()


def plot_uncertainty_analysis(uncertainty_df, save_path=None):
    """Plot uncertainty quantification analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Confidence by correctness
    correct_conf = uncertainty_df[uncertainty_df['correct']]['confidence']
    incorrect_conf = uncertainty_df[~uncertainty_df['correct']]['confidence']
    
    axes[0, 0].hist([correct_conf, incorrect_conf], label=['Correct', 'Incorrect'],
                   bins=20, alpha=0.7, color=['green', 'red'])
    axes[0, 0].set_xlabel('Confidence')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Confidence Distribution by Correctness', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Uncertainty by correctness
    correct_unc = uncertainty_df[uncertainty_df['correct']]['uncertainty']
    incorrect_unc = uncertainty_df[~uncertainty_df['correct']]['uncertainty']
    
    axes[0, 1].hist([correct_unc, incorrect_unc], label=['Correct', 'Incorrect'],
                   bins=20, alpha=0.7, color=['green', 'red'])
    axes[0, 1].set_xlabel('Uncertainty')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Uncertainty Distribution by Correctness', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy vs confidence threshold
    thresholds = np.linspace(0, 1, 50)
    accuracies = []
    coverage = []
    
    for thresh in thresholds:
        mask = uncertainty_df['confidence'] >= thresh
        if mask.sum() > 0:
            acc = uncertainty_df[mask]['correct'].mean()
            accuracies.append(acc)
            coverage.append(mask.sum() / len(uncertainty_df))
        else:
            accuracies.append(np.nan)
            coverage.append(0)
    
    axes[1, 0].plot(thresholds, accuracies, 'o-', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('Confidence Threshold')
    axes[1, 0].set_ylabel('Accuracy (on retained samples)')
    axes[1, 0].set_title('Accuracy vs Confidence Threshold', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Coverage vs threshold
    axes[1, 1].plot(thresholds, coverage, 'o-', linewidth=2, markersize=4, color='#A23B72')
    axes[1, 1].set_xlabel('Confidence Threshold')
    axes[1, 1].set_ylabel('Coverage')
    axes[1, 1].set_title('Coverage vs Confidence Threshold', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Uncertainty Quantification Analysis', fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    plt.show()


def plot_manifold_projection(X_proj, y_true, y_pred=None, class_names=None,
                            method='UMAP', save_path=None):
    """Plot 2D manifold projection."""
    if class_names is None:
        class_names = [f"Class {i}" for i in np.unique(y_true)]
    
    fig, axes = plt.subplots(1, 2 if y_pred is not None else 1, figsize=(14, 5))
    
    if y_pred is None:
        axes = [axes]
    
    # True labels
    for class_id in np.unique(y_true):
        mask = y_true == class_id
        axes[0].scatter(X_proj[mask, 0], X_proj[mask, 1], 
                       label=class_names[class_id], s=50, alpha=0.7, edgecolors='black')
    
    axes[0].set_xlabel(f'{method} Component 1')
    axes[0].set_ylabel(f'{method} Component 2')
    axes[0].set_title('True Classes', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Predictions
    if y_pred is not None:
        for class_id in np.unique(y_pred):
            mask = y_pred == class_id
            axes[1].scatter(X_proj[mask, 0], X_proj[mask, 1],
                          label=class_names[class_id], s=50, alpha=0.7, edgecolors='black')
        
        axes[1].set_xlabel(f'{method} Component 1')
        axes[1].set_ylabel(f'{method} Component 2')
        axes[1].set_title('Predictions', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{method} Projection', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    plt.show()
