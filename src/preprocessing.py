# src/preprocessing.py
"""
Data preprocessing and preparation utilities for SDSS object classification.

Features:
  - Data loading with validation
  - Missing value handling
  - Feature engineering
  - Train-test split with stratification
  - Feature normalization
  - EDA utilities
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold


PHOTOMETRIC_BANDS = ['u', 'g', 'r', 'i', 'z']
PHOTOMETRIC_ERRORS = ['err_u', 'err_g', 'err_r', 'err_i', 'err_z']
METADATA_LEAKAGE_COLUMNS = ['ra', 'dec', 'mjd', 'plate', 'fiberid']


def add_color_indices(df):
    """
    Add physical color-index features (u-g, g-r, r-i, i-z) from SDSS magnitudes.

    These are the quantities astronomers actually use to separate stars,
    galaxies and quasars (e.g. the stellar locus in color-color space),
    as opposed to raw per-band magnitudes.
    """
    df = df.copy()
    df['u_g'] = df['u'] - df['g']
    df['g_r'] = df['g'] - df['r']
    df['r_i'] = df['r'] - df['i']
    df['i_z'] = df['i'] - df['z']
    return df


def select_feature_regime(df, regime):
    """
    Select feature columns for one of three leakage-audit regimes.

    Parameters
    ----------
    df : DataFrame
        Data already stripped of `objid`/`specobjid`/`rerun` (but still
        containing `class` plus whatever metadata columns remain).
    regime : str
        - 'all': everything, including ra/dec/mjd/plate/fiberid
          (the inflated baseline — includes metadata leakage risk)
        - 'photometry': only photometric bands + their errors (honest)
        - 'photometry_color': photometry + physical color indices (honest)

    Returns
    -------
    DataFrame
        Feature columns for the requested regime, plus `class`.
    """
    if regime == 'all':
        return df.copy()

    if regime == 'photometry':
        cols = [c for c in PHOTOMETRIC_BANDS + PHOTOMETRIC_ERRORS if c in df.columns]
        return df[cols + ['class']].copy()

    if regime == 'photometry_color':
        cols = [c for c in PHOTOMETRIC_BANDS + PHOTOMETRIC_ERRORS if c in df.columns]
        df_colors = add_color_indices(df[cols + ['class']])
        return df_colors

    raise ValueError(f"Unknown regime: {regime}")


class DataConfig:
    """Configuration for data handling."""
    def __init__(self, test_size=0.2, validation_size=0.2, random_state=42):
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.drop_columns = ['objid', 'specobjid', 'rerun']
        self.target_column = 'class'


class SDSSDataLoader:
    """Load and validate SDSS dataset."""
    
    def __init__(self, csv_path, config=None):
        """
        Initialize data loader.
        
        Parameters
        ----------
        csv_path : str
            Path to SDSS CSV file
        config : DataConfig, optional
            Configuration object
        """
        self.csv_path = csv_path
        self.config = config or DataConfig()
        self.data_raw = None
        self.data_clean = None
        self.features = None
        self.target = None
        self.label_encoder = LabelEncoder()
    
    def load(self):
        """Load CSV file with validation."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")
        
        self.data_raw = pd.read_csv(self.csv_path)
        print(f"✓ Dataset loaded: {self.data_raw.shape}")
        return self
    
    def validate(self):
        """Validate data integrity."""
        if self.data_raw is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        # Check for missing values
        missing = self.data_raw.isna().sum()
        if missing.sum() > 0:
            print(f"⚠ Missing values detected:\n{missing[missing > 0]}")
        else:
            print("✓ No missing values")
        
        # Check class distribution
        class_dist = self.data_raw[self.config.target_column].value_counts()
        print(f"✓ Class distribution:\n{class_dist}")
        
        return self
    
    def preprocess(self):
        """Clean and preprocess data."""
        self.data_clean = self.data_raw.copy()
        
        # Drop unnecessary columns
        self.data_clean = self.data_clean.drop(
            columns=[col for col in self.config.drop_columns if col in self.data_clean.columns]
        )
        
        # Extract features and target
        self.features = self.data_clean.drop(columns=[self.config.target_column]).values
        target_raw = self.data_clean[self.config.target_column].values
        
        # Encode target labels
        self.target = self.label_encoder.fit_transform(target_raw)
        
        print(f"✓ Features shape: {self.features.shape}")
        print(f"✓ Classes: {list(self.label_encoder.classes_)}")
        
        return self
    
    @property
    def feature_names(self):
        """Get feature column names."""
        return self.data_clean.drop(columns=[self.config.target_column]).columns.tolist()
    
    @property
    def class_names(self):
        """Get decoded class names."""
        return self.label_encoder.classes_
    
    @property
    def n_features(self):
        """Get number of features."""
        return self.features.shape[1] if self.features is not None else None
    
    @property
    def n_classes(self):
        """Get number of classes."""
        return len(self.label_encoder.classes_)


class DataSplitter:
    """Handle train-test-validation splitting with stratification."""
    
    def __init__(self, features, target, test_size=0.2, validation_size=0.2, 
                 random_state=42):
        """
        Initialize splitter.
        
        Parameters
        ----------
        features : ndarray
            Input features
        target : ndarray
            Target labels
        test_size : float
            Test set fraction
        validation_size : float
            Validation set fraction (of training data)
        random_state : int
            Random seed
        """
        self.features = features
        self.target = target
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def split(self):
        """Perform stratified train-test split."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.target
        )
        
        print(f"✓ Data split: Train {self.X_train.shape[0]}, Test {self.X_test.shape[0]}")
        return self
    
    def get_train_test(self):
        """Return train and test sets."""
        if self.X_train is None:
            self.split()
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_kfold_generator(self, n_splits=5):
        """Get k-fold stratified generator."""
        return StratifiedKFold(n_splits=n_splits, shuffle=True, 
                             random_state=self.random_state).split(self.X_train, self.y_train)


class FeatureNormalizer:
    """Handle feature normalization with consistent scaling."""
    
    def __init__(self, method='StandardScaler'):
        """
        Initialize normalizer.
        
        Parameters
        ----------
        method : str
            'StandardScaler' or 'MinMaxScaler'
        """
        self.method = method
        if method == 'StandardScaler':
            self.scaler = StandardScaler()
        elif method == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit(self, X_train):
        """Fit scaler on training data."""
        self.scaler.fit(X_train)
        return self
    
    def transform(self, X):
        """Transform data using fitted scaler."""
        return self.scaler.transform(X)
    
    def fit_transform(self, X_train, X_test):
        """Fit and transform training and test data."""
        self.fit(X_train)
        X_train_scaled = self.transform(X_train)
        X_test_scaled = self.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def inverse_transform(self, X_scaled):
        """Reverse normalization."""
        return self.scaler.inverse_transform(X_scaled)


def prepare_data(csv_path, test_size=0.2, validation_size=0.2, 
                 random_state=42, scaler_method='StandardScaler'):
    """
    Complete data preparation pipeline.
    
    Returns
    -------
    dict
        Contains X_train_scaled, X_test_scaled, y_train, y_test, 
        scaler, data_loader, and metadata
    """
    # Load data
    loader = SDSSDataLoader(csv_path, DataConfig(test_size, validation_size, random_state))
    loader.load().validate().preprocess()
    
    # Split data
    splitter = DataSplitter(loader.features, loader.target, 
                           test_size, validation_size, random_state)
    splitter.split()
    X_train, X_test, y_train, y_test = splitter.get_train_test()
    
    # Normalize features
    normalizer = FeatureNormalizer(scaler_method)
    X_train_scaled, X_test_scaled = normalizer.fit_transform(X_train, X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': normalizer.scaler,
        'data_loader': loader,
        'feature_names': loader.feature_names,
        'class_names': loader.class_names,
    }
