#!/usr/bin/env python
"""
Fase 1 del plan de rescate (plan.md): compara accuracy entre tres regímenes de
features para exponer si el ~90% de accuracy del pipeline "de libro" viene de
física (magnitudes fotométricas) o de metadatos de observación (ra/dec/mjd/
plate/fiberid) que correlacionan con la clase por cómo SDSS asigna targets
espectroscópicos.

Regímenes:
  (a) all               -> pipeline actual (incluye metadatos de leakage)
  (b) photometry        -> solo u,g,r,i,z + errores (honesto)
  (c) photometry_color  -> (b) + índices de color u-g, g-r, r-i, i-z
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import DataConfig, select_feature_regime
from src.models import build_neural_network, compile_model, create_callbacks, train_model, train_on_fold
from src.evaluation import ClassificationMetrics
from src.visualization import set_style, save_figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.inspection import permutation_importance
import tensorflow as tf

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)
tf.random.set_seed(SEED)
set_style('seaborn-v0_8-darkgrid', 'husl')

results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

real_csv = 'data/sdss_real_sample.csv'
default_csv = 'data/Skyserver_SQL2_27_2018 6_51_39 PM.csv'
csv_path = real_csv if os.path.exists(real_csv) else default_csv

raw = pd.read_csv(csv_path)
config = DataConfig()
clean = raw.drop(columns=[c for c in config.drop_columns if c in raw.columns])

REGIMES = ['all', 'photometry', 'photometry_color']
REGIME_LABELS = {
    'all': 'All\n(+ metadata)',
    'photometry': 'Photometry only\n(honest)',
    'photometry_color': 'Photometry\n+ color',
}

print("=" * 100)
print("FASE 1 — EXPERIMENTO DE FUGA DE METADATOS (3 REGÍMENES)")
print("=" * 100)

regime_results = {}
importance_by_regime = {}

for regime in REGIMES:
    print(f"\n[{regime}] Preparando datos...")
    df_regime = select_feature_regime(clean, regime)
    feature_names = [c for c in df_regime.columns if c != 'class']

    X = df_regime[feature_names].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_regime['class'].values)
    class_names = label_encoder.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Features ({len(feature_names)}): {feature_names}")

    # Random Forest (rápido, referencia para importancia de permutación)
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, max_depth=20, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = ClassificationMetrics(y_test, rf_pred, rf.predict_proba(X_test), class_names)
    print(f"  Random Forest: accuracy={rf_metrics.accuracy:.4f}  macro-F1={rf_metrics.f1_macro:.4f}")

    # Red neuronal — stratified K-fold sobre X_train: cada fold varía tanto el split
    # (qué filas caen en train/val) como la inicialización/entrenamiento de la red,
    # a diferencia de repetir con distintas seeds sobre un único split fijo.
    num_features = X_train.shape[1]
    num_classes = len(class_names)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    nn_accuracies = []
    nn_f1s = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        model_fn = lambda: compile_model(
            build_neural_network(num_features, num_classes, l2_reg=0.001, dropout_rate=0.2)
        )
        fold_callbacks = create_callbacks(early_stopping_patience=15, reduce_lr=True, monitor='val_accuracy')
        _, _, y_fold_pred, y_fold_proba = train_on_fold(
            model_fn, X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx],
            epochs=150, batch_size=32, verbose=0, callbacks=fold_callbacks
        )
        fold_metrics = ClassificationMetrics(y_train[val_idx], y_fold_pred, y_fold_proba, class_names)
        nn_accuracies.append(fold_metrics.accuracy)
        nn_f1s.append(fold_metrics.f1_macro)
        print(f"  Red Neuronal (fold {fold_idx + 1}/{N_FOLDS}): accuracy={fold_metrics.accuracy:.4f}  "
              f"macro-F1={fold_metrics.f1_macro:.4f}")

    nn_accuracies = np.array(nn_accuracies)
    nn_f1s = np.array(nn_f1s)
    print(f"  Red Neuronal (media de {N_FOLDS} folds): "
          f"accuracy={nn_accuracies.mean():.4f} ± {nn_accuracies.std():.4f}  "
          f"macro-F1={nn_f1s.mean():.4f} ± {nn_f1s.std():.4f}")

    regime_results[regime] = {
        'rf_accuracy': rf_metrics.accuracy, 'rf_f1_macro': rf_metrics.f1_macro,
        'nn_accuracy_mean': nn_accuracies.mean(), 'nn_accuracy_std': nn_accuracies.std(),
        'nn_f1_macro_mean': nn_f1s.mean(), 'nn_f1_macro_std': nn_f1s.std(),
        'nn_accuracies_all': nn_accuracies.tolist(),
    }

    perm = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=SEED, n_jobs=-1)
    importance_by_regime[regime] = pd.DataFrame({
        'feature': feature_names,
        'importance': perm.importances_mean,
        'std': perm.importances_std,
    }).sort_values('importance', ascending=False)

# ============================================================================
# FIGURA 1: accuracy por régimen (RF y NN)
# ============================================================================
print("\nGenerando figura de comparación de regímenes...")

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(REGIMES))
width = 0.35

rf_accs = [regime_results[r]['rf_accuracy'] for r in REGIMES]
nn_means = [regime_results[r]['nn_accuracy_mean'] for r in REGIMES]
nn_stds = [regime_results[r]['nn_accuracy_std'] for r in REGIMES]

bars_rf = ax.bar(x - width / 2, rf_accs, width, label='Random Forest', color='#1976D2',
                  alpha=0.85, edgecolor='black')
bars_nn = ax.bar(x + width / 2, nn_means, width, yerr=nn_stds, capsize=5,
                  label=f'Neural Network (mean of {N_FOLDS} folds)', color='#2E7D32',
                  alpha=0.85, edgecolor='black')

for bar in bars_rf:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f'{h:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar, std in zip(bars_nn, nn_stds):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + std + 0.005, f'{h:.3f}±{std:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([REGIME_LABELS[r] for r in REGIMES], fontsize=11)
ax.set_ylabel('Accuracy (test set)', fontsize=12, fontweight='bold')
ax.set_title(f'Accuracy by feature regime (NN: mean ± std over {N_FOLDS} folds)',
              fontweight='bold', fontsize=13)
ax.set_ylim([0.5, 1.0])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_figure(fig, results_dir / 'leakage_regime_comparison.png')
print("✓ Saved: leakage_regime_comparison.png")

# ============================================================================
# FIGURA 2: importancia por permutación del régimen "all", metadatos en rojo
# ============================================================================
print("Generando figura de importancia con metadatos resaltados...")

imp_all = importance_by_regime['all']
metadata_cols = {'ra', 'dec', 'mjd', 'plate', 'fiberid'}
colors = ['#D32F2F' if f in metadata_cols else '#1976D2' for f in imp_all['feature']]

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(range(len(imp_all)), imp_all['importance'], xerr=imp_all['std'],
        color=colors, alpha=0.85, edgecolor='black')
ax.set_yticks(range(len(imp_all)))
ax.set_yticklabels(imp_all['feature'], fontsize=10)
ax.set_xlabel('Permutation Importance (Random Forest)', fontsize=11, fontweight='bold')
ax.set_title('Feature importance — red = observational metadata (leakage risk)',
              fontweight='bold', fontsize=12)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
save_figure(fig, results_dir / 'leakage_permutation_importance.png')
print("✓ Saved: leakage_permutation_importance.png")

# ============================================================================
# RESUMEN
# ============================================================================
print("\n" + "=" * 100)
print("RESUMEN — ACCURACY POR RÉGIMEN")
print("=" * 100)
summary_df = pd.DataFrame({
    'Régimen': [REGIME_LABELS[r].replace(chr(10), ' ') for r in REGIMES],
    'RF Accuracy': [f"{regime_results[r]['rf_accuracy']:.4f}" for r in REGIMES],
    'RF Macro-F1': [f"{regime_results[r]['rf_f1_macro']:.4f}" for r in REGIMES],
    f'NN Accuracy (media±std, {N_FOLDS} folds)': [
        f"{regime_results[r]['nn_accuracy_mean']:.4f}±{regime_results[r]['nn_accuracy_std']:.4f}"
        for r in REGIMES
    ],
    f'NN Macro-F1 (media±std, {N_FOLDS} folds)': [
        f"{regime_results[r]['nn_f1_macro_mean']:.4f}±{regime_results[r]['nn_f1_macro_std']:.4f}"
        for r in REGIMES
    ],
})
print(summary_df.to_string(index=False))
summary_df.to_csv(results_dir / 'leakage_regime_summary.csv', index=False)
print(f"\n✓ Saved: results/leakage_regime_summary.csv")

drop_rf = regime_results['all']['rf_accuracy'] - regime_results['photometry']['rf_accuracy']
drop_nn = regime_results['all']['nn_accuracy_mean'] - regime_results['photometry']['nn_accuracy_mean']
print(f"\nCaída de accuracy al quitar metadatos (RF): {drop_rf:+.4f}")
print(f"Caída de accuracy al quitar metadatos (NN, media de {N_FOLDS} folds): {drop_nn:+.4f}")
print("\nAccuracy de NN por fold y régimen (para ver la dispersión entre folds):")
for r in REGIMES:
    print(f"  {r}: {regime_results[r]['nn_accuracies_all']}")
print("\nTop 5 features más importantes (régimen 'all'):")
print(imp_all.head(5).to_string(index=False))
