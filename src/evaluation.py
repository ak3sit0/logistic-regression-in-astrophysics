# src/evaluation.py
"""
Comprehensive evaluation metrics and statistical testing.

Features:
  - Detailed classification metrics
  - Confidence interval calculation
  - ROC-AUC analysis
  - Statistical significance testing
  - Per-class performance analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve, auc,
    classification_report
)


class ClassificationMetrics:
    """Comprehensive classification metrics."""
    
    def __init__(self, y_true, y_pred, y_pred_proba=None, class_names=None):
        """
        Initialize metrics calculator.
        
        Parameters
        ----------
        y_true : ndarray
            True labels
        y_pred : ndarray
            Predicted labels
        y_pred_proba : ndarray, optional
            Prediction probabilities
        class_names : list, optional
            Class names
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.class_names = class_names if class_names is not None else [f"Class_{i}" for i in range(len(np.unique(y_true)))]
        
        self._compute_metrics()
    
    def _compute_metrics(self):
        """Compute all metrics."""
        # Overall accuracy
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        
        # Per-class metrics (macro average)
        self.precision, self.recall, self.f1, self.support = (
            precision_recall_fscore_support(
                self.y_true, self.y_pred,
                average=None,
                zero_division=0
            )
        )
        
        # Weighted averages
        self.precision_weighted, self.recall_weighted, self.f1_weighted, _ = (
            precision_recall_fscore_support(
                self.y_true, self.y_pred,
                average='weighted',
                zero_division=0
            )
        )
        
        # Macro averages
        self.precision_macro = self.precision.mean()
        self.recall_macro = self.recall.mean()
        self.f1_macro = self.f1.mean()
        
        # Confusion matrix
        self.cm = confusion_matrix(self.y_true, self.y_pred)
        
        # ROC-AUC (if probabilities provided)
        try:
            if self.y_pred_proba is not None:
                self.roc_auc = roc_auc_score(
                    self.y_true, self.y_pred_proba,
                    multi_class='ovr', average='weighted'
                )
            else:
                self.roc_auc = np.nan
        except:
            self.roc_auc = np.nan
    
    def summary_dict(self):
        """Return metrics as dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision_macro': self.precision_macro,
            'precision_weighted': self.precision_weighted,
            'recall_macro': self.recall_macro,
            'recall_weighted': self.recall_weighted,
            'f1_macro': self.f1_macro,
            'f1_weighted': self.f1_weighted,
            'roc_auc': self.roc_auc,
        }
    
    def per_class_summary(self):
        """Return per-class metrics as DataFrame."""
        return pd.DataFrame({
            'Class': self.class_names,
            'Precision': self.precision,
            'Recall': self.recall,
            'F1-Score': self.f1,
            'Support': self.support
        })
    
    def classification_report(self):
        """Return classification report string."""
        return classification_report(
            self.y_true, self.y_pred,
            target_names=self.class_names,
            zero_division=0
        )


class ConfidenceIntervals:
    """Calculate confidence intervals for metrics."""
    
    @staticmethod
    def bootstrap_ci(scores, ci=0.95, n_resamples=1000, random_state=42):
        """
        Calculate bootstrap confidence intervals.
        
        Parameters
        ----------
        scores : ndarray
            Array of scores
        ci : float
            Confidence level
        n_resamples : int
            Number of bootstrap resamples
        random_state : int
            Random seed
        
        Returns
        -------
        tuple
            (mean, lower_bound, upper_bound)
        """
        np.random.seed(random_state)
        means = []
        
        for _ in range(n_resamples):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            means.append(sample.mean())
        
        means = np.array(means)
        alpha = 1 - ci
        lower = np.percentile(means, alpha/2 * 100)
        upper = np.percentile(means, (1 - alpha/2) * 100)
        
        return scores.mean(), lower, upper
    
    @staticmethod
    def normal_ci(scores, ci=0.95):
        """
        Calculate normal approximation confidence interval.
        
        Parameters
        ----------
        scores : ndarray
            Array of scores
        ci : float
            Confidence level
        
        Returns
        -------
        tuple
            (mean, lower_bound, upper_bound)
        """
        mean = scores.mean()
        std = scores.std()
        z = stats.norm.ppf((1 + ci) / 2)
        margin = z * std / np.sqrt(len(scores))
        
        return mean, mean - margin, mean + margin


class StatisticalTests:
    """Statistical significance testing."""
    
    @staticmethod
    def paired_ttest(scores_model1, scores_model2, alpha=0.05):
        """
        Paired t-test for model comparison.
        
        Parameters
        ----------
        scores_model1 : ndarray
            Cross-validation scores from model 1
        scores_model2 : ndarray
            Cross-validation scores from model 2
        alpha : float
            Significance level
        
        Returns
        -------
        dict
            Test results
        """
        t_stat, p_value = stats.ttest_rel(scores_model1, scores_model2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'mean_diff': scores_model1.mean() - scores_model2.mean(),
        }
    

class ROCAnalysis:
    """ROC curve and AUC analysis."""
    
    def __init__(self, y_true, y_pred_proba, class_names=None):
        """
        Initialize ROC analysis.
        
        Parameters
        ----------
        y_true : ndarray
            True binary or multiclass labels
        y_pred_proba : ndarray
            Prediction probabilities
        class_names : list, optional
            Class names
        """
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.n_classes = y_pred_proba.shape[1]
        self.class_names = class_names or [f"Class_{i}" for i in range(self.n_classes)]
        
        self.fpr = {}
        self.tpr = {}
        self.roc_auc = {}
        self._compute_roc()
    
    def _compute_roc(self):
        """Compute ROC curves for each class (one-vs-rest)."""
        for i in range(self.n_classes):
            # Binary classification: class i vs rest
            y_binary = (self.y_true == i).astype(int)
            y_proba_i = self.y_pred_proba[:, i]
            
            self.fpr[i], self.tpr[i], _ = roc_curve(y_binary, y_proba_i)
            self.roc_auc[i] = auc(self.fpr[i], self.tpr[i])
    
    def get_curve(self, class_id):
        """Get ROC curve for specific class."""
        return self.fpr[class_id], self.tpr[class_id], self.roc_auc[class_id]
    
    def summary_df(self):
        """Return AUC summary as DataFrame."""
        return pd.DataFrame({
            'Class': self.class_names,
            'AUC': [self.roc_auc[i] for i in range(self.n_classes)]
        })


def evaluate_model(model, X_test, y_test, model_name="", class_names=None):
    """
    Comprehensive model evaluation.
    
    Parameters
    ----------
    model : scikit-learn model
        Trained model with predict() and predict_proba()
    X_test : ndarray
        Test features
    y_test : ndarray
        Test labels
    model_name : str
        Model name for reporting
    class_names : list, optional
        Class names
    
    Returns
    -------
    dict
        Comprehensive evaluation results
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Metrics
    metrics = ClassificationMetrics(y_test, y_pred, y_pred_proba, class_names)
    
    # ROC analysis
    if y_pred_proba is not None:
        roc = ROCAnalysis(y_test, y_pred_proba, class_names)
    else:
        roc = None
    
    return {
        'model_name': model_name,
        'metrics': metrics,
        'roc': roc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'classification_report': metrics.classification_report(),
    }
