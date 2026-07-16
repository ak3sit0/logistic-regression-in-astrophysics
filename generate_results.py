#!/usr/bin/env python
"""
SDSS Classification - Complete Analysis Pipeline for Results Generation
Generates publication-quality figures for README and results directory
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, '/home/joseangel/Documents/Proyectos/Regresion_redes_neuronales_y_astrofisica')

from src.preprocessing import prepare_data
from src.models import build_neural_network, compile_model, train_model, train_on_fold, MCDropoutModel, create_callbacks
from src.evaluation import ClassificationMetrics, ROCAnalysis, ConfidenceIntervals
from src.visualization import set_style, save_figure, plot_uncertainty_analysis, plot_manifold_projection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
import tensorflow as tf
from tensorflow import keras

print("=" * 100)
print("SDSS ASTRONOMICAL CLASSIFICATION - RESULTS GENERATION PIPELINE")
print("=" * 100)

# Setup
set_style('seaborn-v0_8-darkgrid', 'husl')
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

# ============================================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================================
print("\n[1/8] Loading and preparing data...")

# Prefer the real downloaded sample if available, otherwise fall back to original CSV
real_csv = 'data/sdss_real_sample.csv'
default_csv = 'data/Skyserver_SQL2_27_2018 6_51_39 PM.csv'
if os.path.exists(real_csv):
    csv_path = real_csv
else:
    csv_path = default_csv

if not os.path.exists(csv_path):
    print(f"❌ CSV not found at {csv_path}")
    sys.exit(1)

data = prepare_data(csv_path, test_size=0.2, random_state=42)
X_train = data['X_train_scaled']
X_test = data['X_test_scaled']
y_train = data['y_train']
y_test = data['y_test']
class_names = data['class_names']
feature_names = data['feature_names']

print(f"✓ Data loaded: {len(y_test)} test samples, {len(y_train)} training samples")
print(f"✓ Classes: {list(class_names)}")

# ============================================================================
# 2. BASELINE MODELS TRAINING
# ============================================================================
print("\n[2/8] Training baseline models...")

baseline_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, max_depth=20),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42, learning_rate=0.05),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
}

baseline_results = {}
for name, model in baseline_models.items():
    print(f"  Training {name}...", end=' ')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()
    print(f"✓ {acc:.4f}")
    baseline_results[name] = {
        'model': model,
        'y_pred': y_pred,
        'accuracy': acc,
        'y_pred_proba': model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    }

best_baseline_name = max(baseline_results.keys(), key=lambda x: baseline_results[x]['accuracy'])
print(f"✓ Best baseline: {best_baseline_name} ({baseline_results[best_baseline_name]['accuracy']:.4f})")

# ============================================================================
# 3. NEURAL NETWORK TRAINING
# ============================================================================
print("\n[3/8] Training neural network...")

np.random.seed(42)
tf.random.set_seed(42)

num_features = X_train.shape[1]
num_classes = len(class_names)

model = build_neural_network(num_features, num_classes, l2_reg=0.001, dropout_rate=0.2)
model = compile_model(model, learning_rate=1e-3)

callbacks = create_callbacks(early_stopping_patience=15, reduce_lr=True, monitor='val_accuracy')
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

y_pred_proba = model.predict(X_test, verbose=0)
y_pred_nn = np.argmax(y_pred_proba, axis=1)
nn_accuracy = (y_pred_nn == y_test).mean()
print(f"✓ Neural Network accuracy: {nn_accuracy:.4f}")

# ============================================================================
# 4. GENERATE CONFUSION MATRICES FIGURE
# ============================================================================
print("\n[4/8] Generating confusion matrices figure...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# Baseline models
for idx, (name, result) in enumerate(baseline_results.items()):
    cm = confusion_matrix(y_test, result['y_pred'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[idx],
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
    axes[idx].set_title(f'{name}\n(Acc: {result["accuracy"]:.3f})', fontweight='bold', fontsize=11)
    axes[idx].set_ylabel('True Label', fontsize=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=10)

# Neural Network
cm_nn = confusion_matrix(y_test, y_pred_nn)
cm_nn_norm = cm_nn.astype('float') / cm_nn.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_nn_norm, annot=True, fmt='.2f', cmap='Greens', ax=axes[5],
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
axes[5].set_title(f'Deep Neural Network\n(Acc: {nn_accuracy:.3f})', fontweight='bold', fontsize=11, color='#2E7D32')
axes[5].set_ylabel('True Label', fontsize=10)
axes[5].set_xlabel('Predicted Label', fontsize=10)

plt.suptitle('Confusion Matrices - All Models (Normalized)', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
save_figure(fig, results_dir / 'confusion_matrices_all_models.png')
print(f"✓ Saved: confusion_matrices_all_models.png")

# ============================================================================
# 5. TRAINING HISTORY FIGURE
# ============================================================================
print("\n[5/8] Generating training history figure...")

history_df = pd.DataFrame(history.history)
history_df['epoch'] = range(1, len(history_df) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Accuracy
axes[0].plot(history_df['epoch'], history_df['accuracy'], 'o-', label='Training',
             linewidth=2.5, markersize=4, color='#1976D2')
axes[0].plot(history_df['epoch'], history_df['val_accuracy'], 's--', label='Validation',
             linewidth=2.5, markersize=4, color='#D32F2F')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy - Neural Network Training', fontweight='bold', fontsize=12)
axes[0].legend(fontsize=11, loc='lower right')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0.85, 1.0])

# Loss
axes[1].plot(history_df['epoch'], history_df['loss'], 'o-', label='Training',
             linewidth=2.5, markersize=4, color='#1976D2')
axes[1].plot(history_df['epoch'], history_df['val_loss'], 's--', label='Validation',
             linewidth=2.5, markersize=4, color='#D32F2F')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Model Loss - Neural Network Training', fontweight='bold', fontsize=12)
axes[1].legend(fontsize=11, loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Training History - Deep Learning Model', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
save_figure(fig, results_dir / 'training_history.png')
print(f"✓ Saved: training_history.png")

# ============================================================================
# 6. MODEL PERFORMANCE COMPARISON
# ============================================================================
print("\n[6/8] Generating model comparison figure...")

baseline_accs = [baseline_results[name]['accuracy'] for name in baseline_results.keys()]
all_accs = baseline_accs + [nn_accuracy]
all_names = list(baseline_results.keys()) + ['Deep Neural\nNetwork']
colors = ['#1976D2'] * len(baseline_results) + ['#2E7D32']

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy bars
bars = axes[0].bar(range(len(all_accs)), all_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_xticks(range(len(all_names)))
axes[0].set_xticklabels(all_names, fontsize=10)
axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Test Set Accuracy Comparison', fontweight='bold', fontsize=12)
axes[0].set_ylim([0.85, 1.0])
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, all_accs)):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Performance metrics table
metrics_data = []
for name in baseline_results.keys():
    y_pred = baseline_results[name]['y_pred']
    y_proba = baseline_results[name]['y_pred_proba']
    metrics = ClassificationMetrics(y_test, y_pred, y_proba, class_names)
    metrics_data.append({
        'Model': name,
        'Accuracy': f"{metrics.accuracy:.4f}",
        'Precision': f"{metrics.precision_macro:.4f}",
        'Recall': f"{metrics.recall_macro:.4f}",
        'F1': f"{metrics.f1_macro:.4f}"
    })

metrics_df = pd.DataFrame(metrics_data)
nn_metrics = ClassificationMetrics(y_test, y_pred_nn, y_pred_proba, class_names)
nn_row = pd.DataFrame([{
    'Model': 'Deep Neural Network',
    'Accuracy': f"{nn_metrics.accuracy:.4f}",
    'Precision': f"{nn_metrics.precision_macro:.4f}",
    'Recall': f"{nn_metrics.recall_macro:.4f}",
    'F1': f"{nn_metrics.f1_macro:.4f}"
}])

axes[1].axis('tight')
axes[1].axis('off')

# Create table
table_data = pd.concat([metrics_df, nn_row]).values
table = axes[1].table(cellText=table_data,
                     colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Color header
for i in range(5):
    table[(0, i)].set_facecolor('#34495E')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows
for i in range(1, len(metrics_df) + 1):
    for j in range(5):
        table[(i, j)].set_facecolor('#ECF0F1')

# Highlight DNN row
for j in range(5):
    table[(len(metrics_df) + 1, j)].set_facecolor('#C8E6C9')
    table[(len(metrics_df) + 1, j)].set_text_props(weight='bold')

axes[1].set_title('Performance Metrics Summary', fontweight='bold', fontsize=12, pad=20)

plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
save_figure(fig, results_dir / 'model_comparison.png')
print(f"✓ Saved: model_comparison.png")

# ============================================================================
# 7. ROC CURVES FIGURE
# ============================================================================
print("\n[7/8] Generating ROC curves figure...")

fig, axes = plt.subplots(1, num_classes, figsize=(15, 4))
if num_classes == 1:
    axes = [axes]

for i in range(num_classes):
    y_binary = (y_test == i).astype(int)
    y_proba_i = y_pred_proba[:, i]
    
    fpr, tpr, _ = roc_curve(y_binary, y_proba_i)
    roc_auc = auc(fpr, tpr)
    
    axes[i].plot(fpr, tpr, lw=2.5, label=f'ROC (AUC = {roc_auc:.3f})',
                color='#2E86AB', zorder=5)
    axes[i].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier', alpha=0.5)
    axes[i].fill_between(fpr, tpr, alpha=0.15, color='#2E86AB')
    
    axes[i].set_xlim([0.0, 1.0])
    axes[i].set_ylim([0.0, 1.05])
    axes[i].set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    axes[i].set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    axes[i].set_title(f'{class_names[i]} (One-vs-Rest)', fontsize=12, fontweight='bold')
    axes[i].legend(loc='lower right', fontsize=10)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('ROC Curves - Neural Network Classifier', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
save_figure(fig, results_dir / 'roc_curves.png')
print(f"✓ Saved: roc_curves.png")

# ============================================================================
# 8. FEATURE IMPORTANCE FIGURE
# ============================================================================
print("\n[8/8] Performing feature importance analysis...")

# Use Random Forest for feature importance
rf_for_importance = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=20, n_jobs=-1)
rf_for_importance.fit(X_train, y_train)

perm = permutation_importance(rf_for_importance, X_test, y_test, n_repeats=30, 
                             random_state=42, n_jobs=-1)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': perm.importances_mean,
    'std': perm.importances_std
}).sort_values('importance', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top 10 features
top_10 = importance_df.head(10)
axes[0].barh(range(len(top_10)), top_10['importance'],
            xerr=top_10['std'], color='#1976D2', alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_yticks(range(len(top_10)))
axes[0].set_yticklabels(top_10['feature'], fontsize=11)
axes[0].set_xlabel('Permutation Importance', fontsize=12, fontweight='bold')
axes[0].set_title('Top 10 Most Important Features', fontweight='bold', fontsize=12)
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# All features
all_features = importance_df.head(17)
colors_gradient = plt.cm.viridis(np.linspace(0, 1, len(all_features)))
axes[1].barh(range(len(all_features)), all_features['importance'],
            color=colors_gradient, alpha=0.8, edgecolor='black', linewidth=0.5)
axes[1].set_yticks(range(len(all_features)))
axes[1].set_yticklabels(all_features['feature'], fontsize=9)
axes[1].set_xlabel('Permutation Importance', fontsize=12, fontweight='bold')
axes[1].set_title('All 17 Features Ranked', fontweight='bold', fontsize=12)
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

plt.suptitle('Feature Importance Analysis (Permutation Method)', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
save_figure(fig, results_dir / 'feature_importance.png')
print(f"✓ Saved: feature_importance.png")

# ============================================================================
# 9. STRATIFIED 5-FOLD CROSS-VALIDATION (NEURAL NETWORK)
# ============================================================================
print("\n[9/11] Running stratified 5-fold cross-validation...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    model_fn = lambda: compile_model(
        build_neural_network(num_features, num_classes, l2_reg=0.001, dropout_rate=0.2)
    )
    fold_callbacks = create_callbacks(early_stopping_patience=15, reduce_lr=True, monitor='val_accuracy')
    _, _, y_fold_pred, _ = train_on_fold(
        model_fn, X_fold_train, y_fold_train, X_fold_val, y_fold_val,
        epochs=150, batch_size=32, verbose=0, callbacks=fold_callbacks
    )
    fold_acc = (y_fold_pred == y_fold_val).mean()
    fold_accuracies.append(fold_acc)
    print(f"  Fold {fold_idx + 1}/5: accuracy = {fold_acc:.4f}")

fold_accuracies = np.array(fold_accuracies)
mean_acc, ci_lower, ci_upper = ConfidenceIntervals.normal_ci(fold_accuracies, ci=0.95)
print(f"✓ Mean CV accuracy: {mean_acc:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, 6), fold_accuracies, color='#1976D2', alpha=0.8, edgecolor='black')
ax.axhline(mean_acc, color='#D32F2F', linestyle='--', linewidth=2,
           label=f'Mean = {mean_acc:.4f}')
ax.fill_between([0.5, 5.5], ci_lower, ci_upper, color='#D32F2F', alpha=0.15,
                 label=f'95% CI [{ci_lower:.4f}, {ci_upper:.4f}]')
ax.set_xticks(range(1, 6))
ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Stratified 5-Fold Cross-Validation (Neural Network)', fontweight='bold', fontsize=12)
ax.set_xlim([0.5, 5.5])
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_figure(fig, results_dir / 'kfold_results.png')
print(f"✓ Saved: kfold_results.png")

# ============================================================================
# 10. MC DROPOUT UNCERTAINTY QUANTIFICATION
# ============================================================================
print("\n[10/11] Running MC Dropout uncertainty quantification...")

mc_model = MCDropoutModel(model, n_iterations=50)
mc_mean_pred, mc_uncertainty, mc_entropy = mc_model.predict_with_uncertainty(X_test)
y_pred_mc = np.argmax(mc_mean_pred, axis=1)

max_entropy = np.log(num_classes)
mc_confidence = 1.0 - (mc_entropy / max_entropy)
mc_uncertainty_per_sample = mc_uncertainty[np.arange(len(y_pred_mc)), y_pred_mc]

uncertainty_df = pd.DataFrame({
    'confidence': mc_confidence,
    'uncertainty': mc_uncertainty_per_sample,
    'correct': (y_pred_mc == y_test)
})

plot_uncertainty_analysis(uncertainty_df, save_path=results_dir / 'uncertainty_analysis.png')
plt.close('all')
print(f"✓ Saved: uncertainty_analysis.png")
print(f"✓ MC Dropout accuracy: {(y_pred_mc == y_test).mean():.4f} "
      f"(mean confidence: {mc_confidence.mean():.4f})")

# ============================================================================
# 11. UMAP / PCA PROJECTION
# ============================================================================
print("\n[11/11] Generating 2D manifold projection...")

try:
    from umap import UMAP
    reducer = UMAP(random_state=42)
    method = 'UMAP'
except ImportError:
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2, random_state=42)
    method = 'PCA'

X_test_proj = reducer.fit_transform(X_test)
plot_manifold_projection(X_test_proj, y_test, y_pred=y_pred_nn, class_names=class_names,
                          method=method, save_path=results_dir / 'umap_projection.png')
plt.close('all')
print(f"✓ Saved: umap_projection.png (method: {method})")

# ============================================================================
# 12. GENERATE SUMMARY RESULTS FIGURE
# ============================================================================
print("\n[Summary] Generating comprehensive results figure...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

# 1. Class distribution
ax1 = fig.add_subplot(gs[0, 0])
class_counts = pd.Series(y_test).value_counts().sort_index()
colors_class = ['#FF6B6B', '#4ECDC4', '#45B7D1']
ax1.bar(range(len(class_counts)), class_counts.values, color=colors_class, alpha=0.8, edgecolor='black')
ax1.set_xticks(range(len(class_names)))
ax1.set_xticklabels(class_names, fontsize=10)
ax1.set_ylabel('Count', fontsize=10, fontweight='bold')
ax1.set_title('Test Set Class Distribution', fontweight='bold', fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')

# 2. Model accuracy comparison
ax2 = fig.add_subplot(gs[0, 1:])
bars = ax2.bar(range(len(all_accs)), all_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xticks(range(len(all_names)))
ax2.set_xticklabels(all_names, fontsize=10, rotation=15, ha='right')
ax2.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
ax2.set_ylim([0.85, 1.0])
ax2.set_title('Model Performance Comparison', fontweight='bold', fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, all_accs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

# 3. NN Confusion matrix
ax3 = fig.add_subplot(gs[1, 0])
cm_norm_small = cm_nn.astype('float') / cm_nn.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm_small, annot=True, fmt='.2f', cmap='Greens', ax=ax3,
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1, square=True)
ax3.set_title('NN Confusion Matrix', fontweight='bold', fontsize=11)
ax3.set_ylabel('True', fontsize=9)
ax3.set_xlabel('Predicted', fontsize=9)

# 4. ROC curve (first class)
ax4 = fig.add_subplot(gs[1, 1])
y_binary = (y_test == 0).astype(int)
y_proba_0 = y_pred_proba[:, 0]
fpr, tpr, _ = roc_curve(y_binary, y_proba_0)
roc_auc = auc(fpr, tpr)
ax4.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}', color='#2E86AB')
ax4.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax4.fill_between(fpr, tpr, alpha=0.15, color='#2E86AB')
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel('FPR', fontsize=9, fontweight='bold')
ax4.set_ylabel('TPR', fontsize=9, fontweight='bold')
ax4.set_title(f'ROC - {class_names[0]}', fontweight='bold', fontsize=11)
ax4.legend(fontsize=9, loc='lower right')
ax4.grid(True, alpha=0.3)

# 5. Training accuracy curve
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(history_df['epoch'], history_df['accuracy'], 'o-', label='Train', 
        linewidth=2, markersize=3, color='#1976D2')
ax5.plot(history_df['epoch'], history_df['val_accuracy'], 's--', label='Val',
        linewidth=2, markersize=3, color='#D32F2F')
ax5.set_xlabel('Epoch', fontsize=9, fontweight='bold')
ax5.set_ylabel('Accuracy', fontsize=9, fontweight='bold')
ax5.set_title('NN Training History', fontweight='bold', fontsize=11)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_ylim([0.85, 1.0])

# 6. Feature importance (top 8)
ax6 = fig.add_subplot(gs[2, :])
top_8 = importance_df.head(8)
bars_ft = ax6.barh(range(len(top_8)), top_8['importance'],
                   xerr=top_8['std'], color='#1976D2', alpha=0.8, edgecolor='black', linewidth=1.5)
ax6.set_yticks(range(len(top_8)))
ax6.set_yticklabels(top_8['feature'], fontsize=10)
ax6.set_xlabel('Permutation Importance', fontsize=10, fontweight='bold')
ax6.set_title('Top 8 Most Important Features', fontweight='bold', fontsize=11)
ax6.invert_yaxis()
ax6.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, imp) in enumerate(zip(bars_ft, top_8['importance'])):
    ax6.text(imp + 0.001, bar.get_y() + bar.get_height()/2,
            f'{imp:.4f}', ha='left', va='center', fontsize=9)

plt.suptitle('SDSS Classification Results Summary', fontsize=16, fontweight='bold', y=0.995)
save_figure(fig, results_dir / '00_results_summary.png')
print(f"✓ Saved: 00_results_summary.png (MAIN OVERVIEW)")

# ============================================================================
# 10. DOWNLOAD SDSS IMAGES
# ============================================================================
print("\n[Extra] Downloading SDSS example images...")

import urllib.request
import warnings
warnings.filterwarnings('ignore')

# Load original data to get coordinates
sdss_full = pd.read_csv(csv_path)
sdss_full_with_class = sdss_full.copy()

images_dir = results_dir / 'sdss_examples'
images_dir.mkdir(exist_ok=True)

# Get one example from each class
for class_id, class_name in enumerate(class_names):
    print(f"  Downloading {class_name} example...", end=' ')

    df_full = sdss_full_with_class
    if 'class' not in df_full.columns:
        print("❌ 'class' column missing in data; skipping image downloads")
        break

    mask = df_full['class'] == class_name
    if mask.sum() <= 0:
        print("❌ (no objects found)")
        continue

    # Pick a random matching row and safely access columns
    idx = int(np.random.choice(np.where(mask)[0]))
    row = df_full.iloc[idx]
    ra = row.get('ra') if hasattr(row, 'get') else row['ra']
    dec = row.get('dec') if hasattr(row, 'get') else row['dec']
    objid = row.get('objid') if hasattr(row, 'get') else row.get('specobjid', None)

    if pd.isna(ra) or pd.isna(dec):
        print("❌ (missing coordinates)")
        continue

    url = (f'https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg?'
           f'ra={ra}&dec={dec}&scale=0.3&width=256&height=256')

    if objid is None or pd.isna(objid):
        filename = f'{class_id:02d}_{class_name.lower()}_{idx}.jpg'
    else:
        filename = f'{class_id:02d}_{class_name.lower()}_objid_{objid}.jpg'

    filepath = images_dir / filename

    try:
        urllib.request.urlretrieve(url, str(filepath))
        print(f"✓ ({ra:.2f}, {dec:.2f}) -> {filename}")
    except Exception:
        print("❌ (network error)")

print(f"✓ SDSS images saved in: {images_dir}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 100)
print("RESULTS GENERATION COMPLETE!")
print("=" * 100)

print(f"\n📊 GENERATED FIGURES:")
print(f"  • confusion_matrices_all_models.png - All 6 confusion matrices")
print(f"  • training_history.png - Training and validation curves")
print(f"  • model_comparison.png - Performance metrics table")
print(f"  • roc_curves.png - ROC curves for 3 classes")
print(f"  • feature_importance.png - Top 10 and all 17 features")
print(f"  • kfold_results.png - Stratified 5-fold CV accuracy with 95% CI")
print(f"  • uncertainty_analysis.png - MC Dropout (n=50) uncertainty quantification")
print(f"  • umap_projection.png - 2D manifold projection ({method})")
print(f"  • 00_results_summary.png - COMPREHENSIVE 6-panel overview")

print(f"\n🖼️  SDSS EXAMPLE IMAGES:")
import glob
example_images = glob.glob(str(images_dir / '*.jpg'))
print(f"  • {len(example_images)} SDSS object cutouts downloaded")
for img in sorted(example_images):
    print(f"    - {Path(img).name}")

print(f"\n✅ All results saved to: {results_dir.absolute()}")
print(f"\n💡 Next step: Update README.md with these figures!")
print("=" * 100)
