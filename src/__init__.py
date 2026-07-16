# src/__init__.py
"""
SDSS Astronomical Classification Research Package

Modules:
  - preprocessing: Data loading and preprocessing utilities
  - models: Model architectures and training utilities
  - evaluation: Comprehensive evaluation metrics and statistical tests
  - visualization: Advanced visualization functions
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__date__ = "2026-04-12"

from .preprocessing import *
from .models import *
from .evaluation import *
from .visualization import *

__all__ = [
    'preprocessing',
    'models',
    'evaluation',
    'visualization'
]
