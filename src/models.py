# src/models.py
"""
Model architectures and training utilities for SDSS classification.

Features:
  - Neural network architectures
  - Model building utilities
  - Training wrappers
  - Checkpoint management
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_neural_network(num_features, num_classes, l2_reg=0.001, 
                        dropout_rate=0.4, include_batch_norm=True):
    """
    Build a deep neural network with regularization and dropout.
    
    Parameters
    ----------
    num_features : int
        Number of input features
    num_classes : int
        Number of output classes
    l2_reg : float
        L2 regularization coefficient
    dropout_rate : float
        Initial dropout rate
    include_batch_norm : bool
        Whether to include batch normalization
    
    Returns
    -------
    keras.Model
        Compiled neural network model
    """
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(num_features,)))
    
    # First dense block
    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(l2_reg)))
    if include_batch_norm:
        model.add(layers.BatchNormalization(momentum=0.99, epsilon=0.001))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_rate))
    
    # Second dense block
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(l2_reg)))
    if include_batch_norm:
        model.add(layers.BatchNormalization(momentum=0.99, epsilon=0.001))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_rate - 0.1))
    
    # Third dense block
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(l2_reg)))
    if include_batch_norm:
        model.add(layers.BatchNormalization(momentum=0.99, epsilon=0.001))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_rate - 0.2))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


def compile_model(model, learning_rate=1e-3):
    """
    Compile model with Adam optimizer.
    
    Parameters
    ----------
    model : keras.Model
        Model to compile
    learning_rate : float
        Adam learning rate
    
    Returns
    -------
    keras.Model
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_callbacks(early_stopping_patience=15, reduce_lr=False, monitor='val_loss'):
    """
    Create training callbacks.

    Parameters
    ----------
    early_stopping_patience : int
        Early stopping patience
    reduce_lr : bool
        Whether to include learning rate reduction
    monitor : str
        Metric to monitor for early stopping and LR reduction

    Returns
    -------
    list
        List of keras callbacks
    """
    mode = 'max' if 'acc' in monitor else 'min'

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
    ]

    if reduce_lr:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                mode=mode,
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        )
    
    return callbacks


class MCDropoutModel:
    """
    Monte Carlo Dropout wrapper for uncertainty estimation.
    
    Performs multiple forward passes with dropout enabled to estimate
    epistemic uncertainty through the variance of predictions.
    """
    
    def __init__(self, model, n_iterations=50):
        """
        Initialize MC Dropout model.
        
        Parameters
        ----------
        model : keras.Model
            Trained model (with dropout layers)
        n_iterations : int
            Number of MC iterations
        """
        self.model = model
        self.n_iterations = n_iterations
    
    def predict_with_uncertainty(self, X):
        """
        Get predictions with uncertainty estimates using MC Dropout.
        
        Parameters
        ----------
        X : ndarray
            Input features
        
        Returns
        -------
        tuple
            (predictions, uncertainty, entropy)
            - predictions: mean predictions across MC samples
            - uncertainty: standard deviation across MC samples
            - entropy: predictive entropy of mean predictions
        """
        predictions_mc = np.array([
            self.model(X, training=True).numpy()
            for _ in range(self.n_iterations)
        ])
        
        # Mean prediction
        mean_predictions = predictions_mc.mean(axis=0)
        
        # Epistemic uncertainty (standard deviation across samples)
        uncertainty = predictions_mc.std(axis=0)
        
        # Predictive entropy (confidence measure)
        entropy = -np.sum(
            mean_predictions * np.log(mean_predictions + 1e-10),
            axis=1
        )
        
        return mean_predictions, uncertainty, entropy
    
    def predict_class(self, X):
        """Get predicted classes with confidence."""
        mean_pred, _, entropy = self.predict_with_uncertainty(X)
        y_pred = np.argmax(mean_pred, axis=1)
        
        # Normalize entropy to [0, 1] confidence
        n_classes = mean_pred.shape[1]
        max_entropy = np.log(n_classes)
        confidence = 1.0 - (entropy / max_entropy)
        
        return y_pred, confidence


def train_model(model, X_train, y_train, epochs=150, batch_size=32,
                validation_split=0.2, callbacks=None, verbose=1):
    """
    Train model with validation.
    
    Parameters
    ----------
    model : keras.Model
        Model to train
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
    epochs : int
        Number of epochs
    batch_size : int
        Batch size
    validation_split : float
        Validation split fraction
    callbacks : list, optional
        Training callbacks
    verbose : int
        Verbosity level
    
    Returns
    -------
    keras.callbacks.History
        Training history
    """
    if callbacks is None:
        callbacks = create_callbacks()
    
    history = model.fit(
        X_train, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return history


def train_on_fold(model_class, X_fold_train, y_fold_train, X_fold_val, y_fold_val,
                  epochs=150, batch_size=32, verbose=0, callbacks=None):
    """
    Train model on a single fold.

    Parameters
    ----------
    model_class : callable
        Function that returns a compiled model
    X_fold_train : ndarray
        Fold training features
    y_fold_train : ndarray
        Fold training labels
    X_fold_val : ndarray
        Fold validation features
    y_fold_val : ndarray
        Fold validation labels
    epochs : int
        Number of epochs
    batch_size : int
        Batch size
    verbose : int
        Verbosity
    callbacks : list, optional
        Training callbacks. Defaults to create_callbacks(early_stopping_patience=15)
        (monitor='val_loss', no LR reduction) if not provided.

    Returns
    -------
    tuple
        (model, history, predictions, uncertainty)
    """
    model = model_class()

    if callbacks is None:
        callbacks = create_callbacks(early_stopping_patience=15)

    history = model.fit(
        X_fold_train, y_fold_train,
        validation_data=(X_fold_val, y_fold_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    
    y_pred_proba = model.predict(X_fold_val, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    return model, history, y_pred, y_pred_proba
