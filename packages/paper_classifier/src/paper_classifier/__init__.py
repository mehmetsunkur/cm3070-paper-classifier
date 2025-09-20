"""
Paper Classifier Package

A clean implementation of the Mamba-based text classification system
for academic paper classification.
"""

__version__ = "0.1.0"

from .main import main
from .pipeline import run_pipeline, TrainingPipeline
from .subset_generator import ParquetSubsetGenerator

__all__ = [
    "main",
    "run_pipeline", 
    "TrainingPipeline",
    "ParquetSubsetGenerator"
]