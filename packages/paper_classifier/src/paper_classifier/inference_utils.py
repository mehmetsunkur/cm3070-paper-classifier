#!/usr/bin/env python3
"""
Inference utilities for paper classifier.
Handles label mapping loading, model discovery, and output formatting.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer

from .model_manager import ModelManager
from .mamba.model import MambaTextClassification


@dataclass
class ClassificationResult:
    """Single classification result with confidence scores."""
    label_type: str  # 'discipline', 'field', 'method'
    predicted_id: int
    predicted_label: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    model_info: Optional[Dict[str, str]] = None


@dataclass
class MultiLabelResult:
    """Combined results for all label types."""
    discipline: Optional[ClassificationResult] = None
    field: Optional[ClassificationResult] = None
    method: Optional[ClassificationResult] = None
    text_length: Optional[int] = None
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for label_type in ['discipline', 'field', 'method']:
            classification = getattr(self, label_type)
            if classification:
                result[label_type] = {
                    'label': classification.predicted_label,
                    'confidence': classification.confidence,
                    'id': classification.predicted_id
                }
                if classification.probabilities:
                    result[label_type]['top_probabilities'] = classification.probabilities
                if classification.model_info:
                    result[label_type]['model'] = classification.model_info
        
        if self.text_length:
            result['text_length'] = self.text_length
        if self.processing_time:
            result['processing_time_seconds'] = self.processing_time
            
        return result


class LabelMappingLoader:
    """Utility for loading and managing label mappings."""
    
    @staticmethod
    def load_label_mappings(model_path: Path) -> Dict[str, Dict[str, Any]]:
        """
        Load label mappings from a model directory.
        
        Args:
            model_path: Path to model directory (should contain label_mappings.json)
            
        Returns:
            Dictionary with label mappings
        """
        label_mappings_path = model_path / "label_mappings.json"
        if not label_mappings_path.exists():
            raise FileNotFoundError(f"No label_mappings.json found in {model_path}")
        
        with open(label_mappings_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def extract_human_readable_mapping(label_mappings: Dict[str, Any], label_type: str) -> Dict[int, str]:
        """
        Extract human-readable id2label mapping from label_mappings.
        
        Args:
            label_mappings: Raw label mappings from JSON
            label_type: Label type key (e.g., 'label_discipline')
            
        Returns:
            Dictionary mapping label IDs to human-readable names
        """
        if label_type not in label_mappings:
            raise KeyError(f"Label type '{label_type}' not found in mappings")
        
        label_data = label_mappings[label_type]
        if 'label_names' not in label_data:
            raise KeyError(f"No 'label_names' found for label type '{label_type}'")
        
        # Convert string keys to integers and return mapping
        label_names = label_data['label_names']
        return {int(k): v for k, v in label_names.items()}
    
    @staticmethod
    def get_all_human_mappings(label_mappings: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
        """
        Extract all human-readable mappings from label_mappings.
        
        Returns:
            Dictionary with mappings for each label type
        """
        result = {}
        for label_type in label_mappings.keys():
            try:
                result[label_type] = LabelMappingLoader.extract_human_readable_mapping(
                    label_mappings, label_type
                )
            except KeyError as e:
                print(f"Warning: Could not extract mapping for {label_type}: {e}")
        return result


class ModelInferenceLoader:
    """Utility for loading models with proper label mappings."""
    
    def __init__(self, models_dir: str = "trained_models"):
        self.model_manager = ModelManager(models_dir=models_dir, verbose=False)
        self._loaded_models = {}  # Cache for loaded models
        self._tokenizer = None
    
    def get_tokenizer(self):
        """Get shared tokenizer instance."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        return self._tokenizer
    
    def estimate_text_size_rank(self, text: str) -> str:
        """
        Estimate the appropriate size rank for the given text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Size rank string (e.g., '1_4k', '4_8k', etc.)
        """
        tokenizer = self.get_tokenizer()
        tokens = tokenizer(text, truncation=False, return_tensors=None)
        token_count = len(tokens['input_ids'])
        
        # Map token count to size ranks
        if token_count <= 4000:
            return "1_4k"
        elif token_count <= 8000:
            return "4_8k"
        elif token_count <= 16000:
            return "8_16k"
        elif token_count <= 32000:
            return "16_32k"
        elif token_count <= 64000:
            return "32_64k"
        else:
            return "64_128k"
    
    def find_best_model(self, label_type: str, size_rank: str = None) -> Optional[Path]:
        """
        Find the best model for a given label type and size rank.
        
        Args:
            label_type: One of 'discipline', 'field', 'method'
            size_rank: Size rank or None for auto-detection
            
        Returns:
            Path to the best model's final_model directory
        """
        # Search for models matching the criteria
        models = self.model_manager.search_models(
            label_type=label_type,
            size_range=size_rank
        )
        
        if not models:
            return None
        
        # Get the best model by evaluation loss
        best_models = self.model_manager.get_best_models(
            metric="eval_loss", 
            top_k=1, 
            name_pattern=f"*{label_type}*"
        )
        
        # Filter by size rank if specified
        if size_rank:
            best_models = [m for m in best_models if size_rank in m.name]
        
        if best_models:
            return best_models[0].final_model_path
        elif models:
            # Fall back to first available model
            return models[0].final_model_path
        
        return None
    
    def load_model_with_mappings(self, model_path: Path) -> Tuple[MambaTextClassification, Dict[int, str], Dict]:
        """
        Load a model with its human-readable label mappings.
        
        Args:
            model_path: Path to model's final_model directory
            
        Returns:
            Tuple of (model, id2label_mapping, model_info)
        """
        # Load label mappings
        label_mappings = LabelMappingLoader.load_label_mappings(model_path)
        
        # Determine label type from mappings
        label_type = list(label_mappings.keys())[0]
        id2label = LabelMappingLoader.extract_human_readable_mapping(label_mappings, label_type)
        
        # Load model configuration to get num_classes
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            num_classes = config.get('num_classes', len(id2label))
        else:
            num_classes = len(id2label)
        
        # Load the model
        model = MambaTextClassification.from_pretrained(
            str(model_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_classes=num_classes
        )
        
        # Model info for result metadata
        model_info = {
            'path': str(model_path),
            'label_type': label_type,
            'num_classes': num_classes
        }
        
        return model, id2label, model_info


class InferenceEngine:
    """Main inference engine for paper classification."""
    
    def __init__(self, models_dir: str = "trained_models"):
        self.loader = ModelInferenceLoader(models_dir)
        self._model_cache = {}  # Cache loaded models
    
    def classify_text(self, text: str, label_types: List[str] = None, size_rank: str = None) -> MultiLabelResult:
        """
        Classify text for specified label types.
        
        Args:
            text: Input text to classify
            label_types: List of label types to classify ['discipline', 'field', 'method']
            size_rank: Force specific size rank or None for auto-detection
            
        Returns:
            MultiLabelResult with classifications
        """
        import time
        start_time = time.time()
        
        if label_types is None:
            label_types = ['discipline', 'field', 'method']
        
        # Auto-detect size rank if not provided
        if size_rank is None:
            size_rank = self.loader.estimate_text_size_rank(text)
        
        result = MultiLabelResult()
        result.text_length = len(text)
        
        tokenizer = self.loader.get_tokenizer()
        
        # Classify for each label type
        for label_type in label_types:
            try:
                # Find best model for this label type and size
                model_path = self.loader.find_best_model(label_type, size_rank)
                if not model_path:
                    print(f"Warning: No model found for {label_type} with size {size_rank}")
                    continue
                
                # Load model with mappings
                model, id2label, model_info = self.loader.load_model_with_mappings(model_path)
                
                # Run inference
                model.eval()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                with torch.no_grad():
                    input_ids = torch.tensor(tokenizer(text)['input_ids'], device=device)[None]
                    logits = model.forward(input_ids).logits[0]
                    
                    # Get prediction and confidence
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    predicted_id = int(np.argmax(probs))
                    confidence = float(probs[predicted_id])
                    
                    # Get human-readable label
                    predicted_label = id2label.get(predicted_id, f"Unknown_{predicted_id}")
                    
                    # Get top probabilities for additional context
                    top_indices = np.argsort(probs)[-3:][::-1]  # Top 3
                    top_probs = {
                        id2label.get(int(idx), f"Unknown_{idx}"): float(probs[idx])
                        for idx in top_indices
                    }
                
                # Create classification result
                classification = ClassificationResult(
                    label_type=label_type,
                    predicted_id=predicted_id,
                    predicted_label=predicted_label,
                    confidence=confidence,
                    probabilities=top_probs,
                    model_info=model_info
                )
                
                # Set result
                setattr(result, label_type, classification)
                
            except Exception as e:
                print(f"Error classifying {label_type}: {e}")
                continue
        
        result.processing_time = time.time() - start_time
        return result
    
    def classify_pdf(self, pdf_path: Union[str, Path], label_types: List[str] = None) -> MultiLabelResult:
        """
        Classify a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            label_types: List of label types to classify
            
        Returns:
            MultiLabelResult with classifications
        """
        # Import here to avoid circular imports and handle missing dependency
        try:
            import sys
            import os
            
            # Add paper_dataset package to path if it exists
            paper_dataset_path = Path(__file__).parent.parent.parent.parent / "paper_dataset" / "src"
            if paper_dataset_path.exists():
                sys.path.insert(0, str(paper_dataset_path))
            
            from paper_dataset.extractors.pdf_processor import PDFExtractor
        except ImportError as e:
            raise ImportError(
                f"paper_dataset package required for PDF processing: {e}\n"
                "Install with: pip install -e packages/paper_dataset\n"
                "Or ensure paper_dataset package is in Python path"
            )
        
        # Extract text from PDF
        extractor = PDFExtractor(use_ocr=True, extract_metadata=False)
        text, _ = extractor.extract(Path(pdf_path))  # Explicitly ignore metadata
        
        if not text.strip():
            raise ValueError(f"No text extracted from PDF: {pdf_path}")
        
        # Classify the extracted text
        return self.classify_text(text, label_types)


def format_result_human_readable(result: MultiLabelResult, include_confidence: bool = True) -> str:
    """
    Format MultiLabelResult as human-readable text.
    
    Args:
        result: Classification result
        include_confidence: Whether to include confidence scores
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("📄 Paper Classification Results")
    lines.append("=" * 40)
    
    for label_type in ['discipline', 'field', 'method']:
        classification = getattr(result, label_type)
        if classification:
            label_name = label_type.title()
            lines.append(f"\n🏷️  {label_name}: {classification.predicted_label}")
            
            if include_confidence:
                lines.append(f"   Confidence: {classification.confidence:.2%}")
                
                if classification.probabilities:
                    lines.append("   Top alternatives:")
                    for label, prob in list(classification.probabilities.items())[1:3]:
                        lines.append(f"     • {label}: {prob:.2%}")
    
    if result.text_length:
        lines.append(f"\n📊 Text length: {result.text_length:,} characters")
    
    if result.processing_time:
        lines.append(f"⏱️  Processing time: {result.processing_time:.2f} seconds")
    
    return "\n".join(lines)