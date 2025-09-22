#!/usr/bin/env python3
"""
Tests for the ModelManager class.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from paper_classifier.model_manager import ModelManager, ModelInfo


class TestModelManager(unittest.TestCase):
    """Test cases for ModelManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "trained_models"
        self.models_dir.mkdir(parents=True)
        
        # Create test model structure
        self.test_model_dir = self.models_dir / "model_label_discipline_test"
        self.test_run_dir = self.test_model_dir / "run_20250101_000000"
        self.test_final_model_dir = self.test_run_dir / "final_model"
        
        self.test_final_model_dir.mkdir(parents=True)
        
        # Create required files
        self._create_test_files()
        
        self.manager = ModelManager(
            models_dir=str(self.models_dir),
            verbose=False
        )
    
    def _create_test_files(self):
        """Create test files for model validation."""
        # config.json
        config = {
            "d_model": 768,
            "n_layer": 12,
            "num_classes": 7,
            "vocab_size": 50000
        }
        with open(self.test_final_model_dir / "config.json", 'w') as f:
            json.dump(config, f)
        
        # pytorch_model.bin (empty file)
        (self.test_final_model_dir / "pytorch_model.bin").touch()
        
        # label_mappings.json
        label_mappings = {
            "label_discipline": {
                "label2id": {"0": 0, "1": 1, "2": 2},
                "id2label": {"0": 0, "1": 1, "2": 2},
                "label_names": {
                    "0": "Computer Science",
                    "1": "Engineering", 
                    "2": "Mathematics"
                },
                "num_classes": 3,
                "label_column": "label_discipline"
            }
        }
        with open(self.test_final_model_dir / "label_mappings.json", 'w') as f:
            json.dump(label_mappings, f)
        
        # training_config.json
        training_config = {
            "model_name": "test-model",
            "training_learning_rate": 1e-5,
            "training_batch_size_train": 16,
            "training_num_epochs": 10
        }
        with open(self.test_run_dir / "training_config.json", 'w') as f:
            json.dump(training_config, f)
        
        # trainer_state.json
        trainer_state = {
            "best_metric": 0.5,
            "global_step": 1000,
            "epoch": 10,
            "log_history": [
                {"step": 100, "eval_loss": 0.8, "eval_accuracy": 0.6},
                {"step": 200, "eval_loss": 0.6, "eval_accuracy": 0.7},
                {"step": 300, "eval_loss": 0.5, "eval_accuracy": 0.75}
            ]
        }
        with open(self.test_final_model_dir / "trainer_state.json", 'w') as f:
            json.dump(trainer_state, f)
    
    def test_initialization(self):
        """Test ModelManager initialization."""
        self.assertEqual(str(self.manager.models_dir), str(self.models_dir))
        self.assertEqual(self.manager.metric_preference, "eval_loss")
        self.assertFalse(self.manager.verbose)
    
    def test_validate_final_model_valid(self):
        """Test validation of a valid final model."""
        result = self.manager._validate_final_model(self.test_final_model_dir)
        self.assertTrue(result)
    
    def test_validate_final_model_missing_files(self):
        """Test validation fails with missing files."""
        # Remove required file
        (self.test_final_model_dir / "pytorch_model.bin").unlink()
        
        result = self.manager._validate_final_model(self.test_final_model_dir)
        self.assertFalse(result)
    
    def test_validate_final_model_invalid_label_mappings(self):
        """Test validation fails with invalid label mappings."""
        # Create invalid label mappings
        invalid_mappings = {"invalid": {"no_required_fields": True}}
        with open(self.test_final_model_dir / "label_mappings.json", 'w') as f:
            json.dump(invalid_mappings, f)
        
        result = self.manager._validate_final_model(self.test_final_model_dir)
        self.assertFalse(result)
    
    def test_find_best_run(self):
        """Test finding the best run directory."""
        best_run = self.manager._find_best_run(self.test_model_dir)
        self.assertEqual(best_run, self.test_run_dir)
    
    def test_analyze_model(self):
        """Test analyzing a model directory."""
        model_info = self.manager._analyze_model(self.test_model_dir)
        
        self.assertIsInstance(model_info, ModelInfo)
        self.assertEqual(model_info.name, "model_label_discipline_test")
        self.assertEqual(model_info.run_name, "run_20250101_000000")
        self.assertTrue(model_info.has_final_model)
        self.assertTrue(model_info.has_label_mappings)
        
        # Check loaded data
        self.assertIsNotNone(model_info.config)
        self.assertEqual(model_info.config['num_classes'], 7)
        
        self.assertIsNotNone(model_info.training_config)
        self.assertEqual(model_info.training_config['training_learning_rate'], 1e-5)
        
        self.assertIsNotNone(model_info.label_mappings)
        self.assertEqual(model_info.label_mappings['label_discipline']['num_classes'], 3)
        
        self.assertIsNotNone(model_info.metrics)
        self.assertEqual(model_info.metrics['eval_loss'], 0.5)
        self.assertEqual(model_info.metrics['eval_accuracy'], 0.75)
    
    def test_discover_models(self):
        """Test discovering models."""
        models = self.manager.discover_models()
        
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].name, "model_label_discipline_test")
    
    def test_get_model_summary(self):
        """Test getting model summary."""
        summary = self.manager.get_model_summary()
        
        self.assertEqual(summary['total_models'], 1)
        self.assertEqual(summary['models_with_metrics'], 0)  # No valid eval_loss
        self.assertEqual(summary['models_with_label_mappings'], 1)
        self.assertEqual(summary['by_label_type']['discipline'], 1)
    
    def test_search_models(self):
        """Test searching models."""
        # Search by label type
        models = self.manager.search_models(label_type="discipline")
        self.assertEqual(len(models), 1)
        
        # Search by non-existent label type
        models = self.manager.search_models(label_type="field")
        self.assertEqual(len(models), 0)
    
    def test_get_model_by_name(self):
        """Test getting model by name."""
        model = self.manager.get_model_by_name("model_label_discipline_test")
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "model_label_discipline_test")
        
        # Test non-existent model
        model = self.manager.get_model_by_name("non_existent_model")
        self.assertIsNone(model)
    
    def test_export_to_dataframe(self):
        """Test exporting to DataFrame."""
        try:
            import pandas as pd
            df = self.manager.export_to_dataframe()
            
            self.assertEqual(len(df), 1)
            self.assertIn('model_name', df.columns)
            self.assertIn('has_label_mappings', df.columns)
            self.assertEqual(df.iloc[0]['model_name'], "model_label_discipline_test")
            self.assertTrue(df.iloc[0]['has_label_mappings'])
        except ImportError:
            self.skipTest("pandas not available")
    
    def test_export_to_json(self):
        """Test exporting to JSON."""
        output_file = Path(self.temp_dir) / "test_export.json"
        
        self.manager.export_to_json(str(output_file))
        
        self.assertTrue(output_file.exists())
        
        with open(output_file) as f:
            data = json.load(f)
        
        self.assertIn('models', data)
        self.assertIn('summary', data)
        self.assertEqual(len(data['models']), 1)
        self.assertEqual(data['models'][0]['name'], "model_label_discipline_test")
    
    def test_extract_metrics_from_trainer_state(self):
        """Test extracting metrics from trainer state."""
        trainer_state = {
            "best_metric": 0.3,
            "global_step": 500,
            "epoch": 5,
            "log_history": [
                {"step": 100, "eval_loss": 0.8},
                {"step": 200, "eval_loss": 0.6, "eval_accuracy": 0.7},
                {"step": 300, "eval_loss": 0.4, "eval_accuracy": 0.8}
            ]
        }
        
        metrics = self.manager._extract_metrics_from_trainer_state(trainer_state)
        
        self.assertEqual(metrics['best_metric'], 0.3)
        self.assertEqual(metrics['global_step'], 500)
        self.assertEqual(metrics['epoch'], 5)
        self.assertEqual(metrics['eval_loss'], 0.4)
        self.assertEqual(metrics['eval_accuracy'], 0.8)
    
    def test_model_info_to_dict(self):
        """Test ModelInfo to_dict method."""
        model_info = ModelInfo(
            name="test_model",
            path=Path("/test/path"),
            final_model_path=Path("/test/final"),
            run_name="test_run",
            has_final_model=True,
            has_label_mappings=True
        )
        
        data = model_info.to_dict()
        
        self.assertEqual(data['name'], "test_model")
        self.assertEqual(data['path'], "/test/path")
        self.assertEqual(data['final_model_path'], "/test/final")
        self.assertTrue(data['has_final_model'])
        self.assertTrue(data['has_label_mappings'])
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)


class TestModelManagerCLI(unittest.TestCase):
    """Test cases for ModelManager CLI functionality."""
    
    def test_main_function_import(self):
        """Test that main function can be imported."""
        from paper_classifier.model_manager import main
        self.assertTrue(callable(main))
    
    @patch('sys.argv', ['model_manager.py', '--help'])
    def test_cli_help(self):
        """Test CLI help functionality."""
        from paper_classifier.model_manager import main
        
        with self.assertRaises(SystemExit):
            main()


if __name__ == '__main__':
    unittest.main()