#!/usr/bin/env python3
"""
Simplified Model Manager for Paper Classifier project.

This module provides a ModelManager class that discovers trained models,
extracts metrics, and provides analysis capabilities while hiding the 
complexity of run directories from the user.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd
from io import StringIO


@dataclass
class ModelInfo:
    """Information about a trained model."""
    name: str
    path: Path
    final_model_path: Path
    run_name: str
    has_final_model: bool
    has_label_mappings: bool
    config: Optional[Dict] = None
    training_config: Optional[Dict] = None
    label_mappings: Optional[Dict] = None
    metrics: Optional[Dict] = None
    dataset_info: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['path'] = str(data['path'])
        data['final_model_path'] = str(data['final_model_path'])
        return data


class ModelManager:
    """
    Simplified Model Manager for Paper Classifier.
    
    This class discovers trained models and provides analysis capabilities
    while abstracting away the complexity of run directories.
    """
    
    def __init__(self, 
                 models_dir: str = "trained_models",
                 metric_preference: str = "eval_loss",
                 verbose: bool = False):
        """
        Initialize the ModelManager.
        
        Args:
            models_dir: Directory containing trained models
            metric_preference: Preferred metric for selecting best runs
            verbose: Enable verbose output
        """
        self.models_dir = Path(models_dir)
        self.metric_preference = metric_preference
        self.verbose = verbose
        
        if not self.models_dir.exists():
            if self.verbose:
                print(f"Models directory does not exist: {self.models_dir}")
    
    def _load_json_file(self, file_path: Path) -> Optional[Dict]:
        """Load JSON file safely."""
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            if self.verbose:
                print(f"Error loading {file_path}: {e}")
        return None
    
    def _extract_metrics_from_trainer_state(self, trainer_state: Dict) -> Dict:
        """Extract relevant metrics from trainer state."""
        metrics = {
            'best_metric': trainer_state.get('best_metric'),
            'global_step': trainer_state.get('global_step'),
            'epoch': trainer_state.get('epoch'),
            'eval_loss': None,
            'eval_accuracy': None,
            'eval_f1': None,
            'eval_precision': None,
            'eval_recall': None
        }
        
        # Extract latest eval metrics from log history
        if 'log_history' in trainer_state:
            eval_entries = [entry for entry in trainer_state['log_history'] 
                          if 'eval_loss' in entry or 'eval_accuracy' in entry]
            if eval_entries:
                latest_eval = eval_entries[-1]
                metrics.update({
                    'eval_loss': latest_eval.get('eval_loss'),
                    'eval_accuracy': latest_eval.get('eval_accuracy'),
                    'eval_f1': latest_eval.get('eval_f1'),
                    'eval_precision': latest_eval.get('eval_precision'),
                    'eval_recall': latest_eval.get('eval_recall')
                })
        
        return metrics
    
    def _validate_final_model(self, final_model_dir: Path) -> bool:
        """
        Check if a final_model directory represents a fully trained model.
        
        Requires: config.json, pytorch_model.bin, label_mappings.json
        """
        required_files = [
            "config.json",
            "pytorch_model.bin", 
            "label_mappings.json"
        ]
        
        for file_name in required_files:
            if not (final_model_dir / file_name).exists():
                if self.verbose:
                    print(f"Missing {file_name} in {final_model_dir}")
                return False
        
        # Validate label_mappings.json content
        label_mappings = self._load_json_file(final_model_dir / "label_mappings.json")
        if not label_mappings:
            if self.verbose:
                print(f"Invalid label_mappings.json in {final_model_dir}")
            return False
            
        # Check for required structure
        for label_type, mapping in label_mappings.items():
            if not all(key in mapping for key in ['num_classes', 'label_names']):
                if self.verbose:
                    print(f"Invalid label mapping structure in {final_model_dir}")
                return False
        
        return True
    
    def _find_best_run(self, model_dir: Path) -> Optional[Path]:
        """
        Find the best run directory based on metrics or timestamp.
        
        Args:
            model_dir: Path to model directory
            
        Returns:
            Path to best run directory or None
        """
        run_dirs = [d for d in model_dir.iterdir() 
                   if d.is_dir() and d.name.startswith("run_")]
        
        if not run_dirs:
            return None
        
        # Filter runs that have valid final models
        valid_runs = []
        for run_dir in run_dirs:
            final_model_dir = run_dir / "final_model"
            if final_model_dir.exists() and self._validate_final_model(final_model_dir):
                valid_runs.append(run_dir)
        
        if not valid_runs:
            return None
        
        # If only one valid run, return it
        if len(valid_runs) == 1:
            return valid_runs[0]
        
        # Try to find best run based on metrics
        best_run = None
        best_value = None
        
        for run_dir in valid_runs:
            final_model_dir = run_dir / "final_model"
            trainer_state = self._load_json_file(final_model_dir / "trainer_state.json")
            if trainer_state:
                metrics = self._extract_metrics_from_trainer_state(trainer_state)
                
                if self.metric_preference == "eval_loss" and metrics['eval_loss'] is not None:
                    if best_value is None or metrics['eval_loss'] < best_value:
                        best_value = metrics['eval_loss']
                        best_run = run_dir
                elif self.metric_preference == "eval_accuracy" and metrics['eval_accuracy'] is not None:
                    if best_value is None or metrics['eval_accuracy'] > best_value:
                        best_value = metrics['eval_accuracy']
                        best_run = run_dir
        
        # If no metrics found, return the latest run by timestamp
        if best_run is None:
            best_run = max(valid_runs, key=lambda x: x.name)
        
        return best_run
    
    def _analyze_model(self, model_dir: Path) -> Optional[ModelInfo]:
        """
        Analyze a single model directory.
        
        Args:
            model_dir: Path to model directory
            
        Returns:
            ModelInfo object or None if no valid final model found
        """
        best_run = self._find_best_run(model_dir)
        if not best_run:
            return None
            
        final_model_dir = best_run / "final_model"
        if not final_model_dir.exists() or not self._validate_final_model(final_model_dir):
            return None
        
        # Load configuration files
        config = self._load_json_file(final_model_dir / "config.json")
        training_config = self._load_json_file(best_run / "training_config.json")
        dataset_info = self._load_json_file(best_run / "dataset_info.json")
        label_mappings = self._load_json_file(final_model_dir / "label_mappings.json")
        
        # Load and extract metrics - try multiple locations
        trainer_state = None
        
        # First try final_model directory
        trainer_state = self._load_json_file(final_model_dir / "trainer_state.json")
        
        # If not found, try run directory
        if not trainer_state:
            trainer_state = self._load_json_file(best_run / "trainer_state.json")
        
        # If still not found, try the latest checkpoint
        if not trainer_state:
            checkpoints = sorted([d for d in best_run.iterdir() 
                                if d.is_dir() and d.name.startswith("checkpoint-")],
                               key=lambda x: int(x.name.split("-")[1]), reverse=True)
            for checkpoint in checkpoints:
                trainer_state = self._load_json_file(checkpoint / "trainer_state.json")
                if trainer_state:
                    break
        
        metrics = None
        if trainer_state:
            metrics = self._extract_metrics_from_trainer_state(trainer_state)
        
        return ModelInfo(
            name=model_dir.name,
            path=model_dir,
            final_model_path=final_model_dir,
            run_name=best_run.name,
            has_final_model=True,
            has_label_mappings=label_mappings is not None,
            config=config,
            training_config=training_config,
            label_mappings=label_mappings,
            metrics=metrics,
            dataset_info=dataset_info
        )
    
    def discover_models(self, name_pattern: str = "model_*") -> List[ModelInfo]:
        """
        Discover all trained models with valid final_model directories.
        
        Args:
            name_pattern: Glob pattern for model directory names
            
        Returns:
            List of ModelInfo objects
        """
        if not self.models_dir.exists():
            return []
        
        models = []
        
        for model_dir in sorted(self.models_dir.glob(name_pattern)):
            if not model_dir.is_dir():
                continue
                
            model_info = self._analyze_model(model_dir)
            if model_info:
                models.append(model_info)
                if self.verbose:
                    print(f"Found model: {model_info.name} (run: {model_info.run_name})")
        
        return models
    
    def get_model_summary(self, name_pattern: str = "model_*") -> Dict[str, Any]:
        """
        Get summary statistics about discovered models.
        
        Args:
            name_pattern: Glob pattern for model directory names
            
        Returns:
            Dictionary with summary statistics
        """
        models = self.discover_models(name_pattern)
        
        # Categorize models
        by_label_type = {}
        by_size = {}
        by_sample_count = {}
        
        for model in models:
            name = model.name
            
            # Extract label type (discipline, field, method)
            if "label_discipline" in name:
                label_type = "discipline"
            elif "label_field" in name:
                label_type = "field"
            elif "label_method" in name:
                label_type = "method"
            else:
                label_type = "other"
            
            by_label_type[label_type] = by_label_type.get(label_type, 0) + 1
            
            # Extract size info (1_4k, 4_8k, etc.)
            for size in ["1_4k", "4_8k", "8_16k", "16_32k", "32_64k", "64_128k"]:
                if size in name:
                    by_size[size] = by_size.get(size, 0) + 1
                    break
            
            # Extract sample count (s1000, s5000, etc.)
            for sample in ["s1000", "s5000", "s10000"]:
                if sample in name:
                    by_sample_count[sample] = by_sample_count.get(sample, 0) + 1
                    break
        
        # Calculate metrics summary
        models_with_metrics = [m for m in models if m.metrics and m.metrics.get('eval_loss')]
        avg_eval_loss = None
        avg_eval_accuracy = None
        
        if models_with_metrics:
            losses = [m.metrics['eval_loss'] for m in models_with_metrics 
                     if m.metrics['eval_loss'] is not None]
            accuracies = [m.metrics['eval_accuracy'] for m in models_with_metrics 
                         if m.metrics['eval_accuracy'] is not None]
            
            if losses:
                avg_eval_loss = sum(losses) / len(losses)
            if accuracies:
                avg_eval_accuracy = sum(accuracies) / len(accuracies)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "models_dir": str(self.models_dir),
            "total_models": len(models),
            "models_with_metrics": len(models_with_metrics),
            "models_with_label_mappings": len([m for m in models if m.has_label_mappings]),
            "by_label_type": by_label_type,
            "by_size": by_size,
            "by_sample_count": by_sample_count,
            "avg_eval_loss": avg_eval_loss,
            "avg_eval_accuracy": avg_eval_accuracy
        }
    
    def get_best_models(self, 
                       metric: str = "eval_loss", 
                       top_k: int = 10,
                       name_pattern: str = "model_*") -> List[ModelInfo]:
        """
        Get the best performing models based on a metric.
        
        Args:
            metric: Metric to sort by ('eval_loss' or 'eval_accuracy')
            top_k: Number of top models to return
            name_pattern: Glob pattern for model directory names
            
        Returns:
            List of top ModelInfo objects
        """
        models = self.discover_models(name_pattern)
        
        # Filter models with the requested metric
        models_with_metric = [
            m for m in models 
            if m.metrics and m.metrics.get(metric) is not None
        ]
        
        # Sort based on metric
        reverse = metric == "eval_accuracy"  # Higher is better for accuracy
        sorted_models = sorted(
            models_with_metric,
            key=lambda x: x.metrics[metric],
            reverse=reverse
        )
        
        return sorted_models[:top_k]
    
    def search_models(self, 
                     label_type: Optional[str] = None,
                     size_range: Optional[str] = None, 
                     sample_count: Optional[str] = None,
                     name_pattern: str = "model_*") -> List[ModelInfo]:
        """
        Search for models matching specific criteria.
        
        Args:
            label_type: Filter by label type ('discipline', 'field', 'method')
            size_range: Filter by size range ('1_4k', '4_8k', etc.)
            sample_count: Filter by sample count ('s1000', 's5000', etc.)
            name_pattern: Glob pattern for model directory names
            
        Returns:
            List of matching ModelInfo objects
        """
        models = self.discover_models(name_pattern)
        
        filtered_models = []
        for model in models:
            name = model.name.lower()
            
            # Check label type
            if label_type:
                if f"label_{label_type.lower()}" not in name:
                    continue
            
            # Check size range
            if size_range:
                if size_range.lower() not in name:
                    continue
            
            # Check sample count
            if sample_count:
                if sample_count.lower() not in name:
                    continue
            
            filtered_models.append(model)
        
        return filtered_models
    
    def get_model_by_name(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get a specific model by name.
        
        Args:
            model_name: Name of the model to find
            
        Returns:
            ModelInfo object or None if not found
        """
        model_dir = self.models_dir / model_name
        if not model_dir.exists():
            return None
        
        return self._analyze_model(model_dir)
    
    def print_summary(self, name_pattern: str = "model_*"):
        """Print a formatted summary of all models."""
        summary = self.get_model_summary(name_pattern)
        models = self.discover_models(name_pattern)
        
        print("\n" + "=" * 60)
        print("PAPER CLASSIFIER MODEL SUMMARY")
        print("=" * 60)
        print(f"Generated: {summary['timestamp']}")
        print(f"Models Directory: {summary['models_dir']}")
        print()
        
        print("OVERVIEW")
        print("-" * 40)
        print(f"Total Models: {summary['total_models']}")
        print(f"Models with Metrics: {summary['models_with_metrics']}")
        print(f"Models with Label Mappings: {summary['models_with_label_mappings']}")
        if summary['avg_eval_loss']:
            print(f"Average Eval Loss: {summary['avg_eval_loss']:.4f}")
        if summary['avg_eval_accuracy']:
            print(f"Average Eval Accuracy: {summary['avg_eval_accuracy']:.4f}")
        print()
        
        print("BY LABEL TYPE")
        print("-" * 40)
        for label_type, count in summary['by_label_type'].items():
            print(f"  {label_type}: {count}")
        print()
        
        print("BY SIZE RANGE")
        print("-" * 40)
        for size, count in summary['by_size'].items():
            print(f"  {size}: {count}")
        print()
        
        print("BY SAMPLE COUNT")
        print("-" * 40)
        for sample, count in summary['by_sample_count'].items():
            print(f"  {sample}: {count}")
        print()
        
        # Show top 5 models by eval_loss
        best_models = self.get_best_models("eval_loss", 5, name_pattern)
        if best_models:
            print("TOP 5 MODELS (by eval_loss)")
            print("-" * 40)
            for i, model in enumerate(best_models, 1):
                loss = model.metrics['eval_loss']
                acc = model.metrics.get('eval_accuracy', 'N/A')
                if isinstance(acc, float):
                    acc = f"{acc:.4f}"
                print(f"{i}. {model.name}")
                print(f"   Loss: {loss:.4f}, Accuracy: {acc}")
                print(f"   Run: {model.run_name}")
        print()
    
    def export_to_dataframe(self, name_pattern: str = "model_*") -> pd.DataFrame:
        """
        Export model information to a pandas DataFrame.
        
        Args:
            name_pattern: Glob pattern for model directory names
            
        Returns:
            DataFrame with model information
        """
        models = self.discover_models(name_pattern)
        
        data = []
        for model in models:
            row = {
                'model_name': model.name,
                'run_name': model.run_name,
                'final_model_path': str(model.final_model_path),
                'has_config': model.config is not None,
                'has_training_config': model.training_config is not None,
                'has_label_mappings': model.has_label_mappings,
                'has_metrics': model.metrics is not None
            }
            
            # Add configuration info
            if model.config:
                row.update({
                    'num_classes': model.config.get('num_classes'),
                    'd_model': model.config.get('d_model'),
                    'n_layer': model.config.get('n_layer'),
                    'vocab_size': model.config.get('vocab_size')
                })
            
            # Add training config info
            if model.training_config:
                row.update({
                    'model_name_config': model.training_config.get('model_name'),
                    'learning_rate': model.training_config.get('training_learning_rate'),
                    'batch_size': model.training_config.get('training_batch_size_train'),
                    'num_epochs': model.training_config.get('training_num_epochs'),
                    'max_seq_length': model.training_config.get('data_max_seq_length'),
                    'label_column': model.training_config.get('subset_generation_label_column')
                })
            
            # Add label mappings info
            if model.label_mappings:
                for label_type, mapping in model.label_mappings.items():
                    row[f'{label_type}_num_classes'] = mapping.get('num_classes')
                    row[f'{label_type}_label_column'] = mapping.get('label_column')
            
            # Add metrics
            if model.metrics:
                row.update({
                    'eval_loss': model.metrics.get('eval_loss'),
                    'eval_accuracy': model.metrics.get('eval_accuracy'),
                    'eval_f1': model.metrics.get('eval_f1'),
                    'eval_precision': model.metrics.get('eval_precision'),
                    'eval_recall': model.metrics.get('eval_recall'),
                    'best_metric': model.metrics.get('best_metric'),
                    'global_step': model.metrics.get('global_step'),
                    'epoch': model.metrics.get('epoch')
                })
            
            # Add dataset info
            if model.dataset_info:
                row.update({
                    'train_samples': model.dataset_info.get('train_samples'),
                    'val_samples': model.dataset_info.get('val_samples'),
                    'test_samples': model.dataset_info.get('test_samples')
                })
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def export_to_json(self, 
                      output_path: str, 
                      name_pattern: str = "model_*",
                      include_summary: bool = True):
        """
        Export model information to JSON file.
        
        Args:
            output_path: Path to output JSON file
            name_pattern: Glob pattern for model directory names
            include_summary: Whether to include summary statistics
        """
        models = self.discover_models(name_pattern)
        
        data = {
            "models": [model.to_dict() for model in models]
        }
        
        if include_summary:
            data["summary"] = self.get_model_summary(name_pattern)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        if self.verbose:
            print(f"Exported {len(models)} models to {output_path}")
    
    def export_to_csv(self, 
                     output_path: str, 
                     name_pattern: str = "model_*"):
        """
        Export model information to CSV file.
        
        Args:
            output_path: Path to output CSV file
            name_pattern: Glob pattern for model directory names
        """
        df = self.export_to_dataframe(name_pattern)
        df.to_csv(output_path, index=False)
        
        if self.verbose:
            print(f"Exported {len(df)} models to {output_path}")


def main():
    """Command-line interface for ModelManager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage and analyze paper classifier models")
    parser.add_argument("--models-dir", default="trained_models", 
                       help="Directory containing trained models")
    parser.add_argument("--pattern", default="model_*", 
                       help="Model directory pattern")
    parser.add_argument("--metric", default="eval_loss", 
                       choices=["eval_loss", "eval_accuracy"],
                       help="Preferred metric for model selection")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose output")
    
    # Actions
    parser.add_argument("--summary", action="store_true", 
                       help="Print model summary")
    parser.add_argument("--best", type=int, metavar="N",
                       help="Show top N best models")
    parser.add_argument("--search-label", choices=["discipline", "field", "method"],
                       help="Filter by label type")
    parser.add_argument("--search-size", 
                       help="Filter by size range (e.g., '4_8k')")
    parser.add_argument("--search-samples",
                       help="Filter by sample count (e.g., 's5000')")
    parser.add_argument("--export-json", 
                       help="Export to JSON file")
    parser.add_argument("--export-csv",
                       help="Export to CSV file")
    parser.add_argument("--get-model", 
                       help="Get details for a specific model by name")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = ModelManager(
        models_dir=args.models_dir,
        metric_preference=args.metric,
        verbose=args.verbose
    )
    
    try:
        if args.get_model:
            model = manager.get_model_by_name(args.get_model)
            if model:
                print(f"\nModel: {model.name}")
                print(f"Run: {model.run_name}")
                print(f"Path: {model.final_model_path}")
                print(f"Has label mappings: {model.has_label_mappings}")
                if model.metrics:
                    print(f"Eval loss: {model.metrics.get('eval_loss', 'N/A')}")
                    print(f"Eval accuracy: {model.metrics.get('eval_accuracy', 'N/A')}")
                if model.label_mappings:
                    for label_type, mapping in model.label_mappings.items():
                        print(f"{label_type} classes: {mapping.get('num_classes', 'N/A')}")
            else:
                print(f"Model '{args.get_model}' not found")
        
        elif args.summary:
            manager.print_summary(args.pattern)
        
        elif args.best:
            models = manager.get_best_models(args.metric, args.best, args.pattern)
            print(f"\nTop {args.best} models by {args.metric}:")
            print("-" * 50)
            for i, model in enumerate(models, 1):
                metric_val = model.metrics.get(args.metric, 'N/A')
                print(f"{i}. {model.name}")
                print(f"   {args.metric}: {metric_val}")
                print(f"   Run: {model.run_name}")
                print()
        
        elif any([args.search_label, args.search_size, args.search_samples]):
            models = manager.search_models(
                label_type=args.search_label,
                size_range=args.search_size,
                sample_count=args.search_samples,
                name_pattern=args.pattern
            )
            print(f"\nFound {len(models)} matching models:")
            print("-" * 50)
            for model in models:
                print(f"  {model.name} (run: {model.run_name})")
        
        elif args.export_json:
            manager.export_to_json(args.export_json, args.pattern)
        
        elif args.export_csv:
            manager.export_to_csv(args.export_csv, args.pattern)
        
        else:
            # Default: show summary
            manager.print_summary(args.pattern)
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())