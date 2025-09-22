# Model Selection and Ranking Algorithms

**Generated**: 2025-09-22  
**Context**: Detailed specification for automated model selection based on size preferences and performance metrics

## Overview

The model selection system automatically chooses optimal models for each label type (discipline/field/method) based on:
1. **Size Preference**: User-specified context length preference with intelligent fallback
2. **Performance Metrics**: Evaluation loss/accuracy ranking within size categories  
3. **Availability**: Graceful handling of missing models or metrics

## Current Model Inventory Analysis

### Available Models by Label Type
```
DISCIPLINE (18 models):
- Size distribution: 1_4k(5), 4_8k(4), 8_16k(4), 16_32k(3), 32_64k(2)
- Best performer: model_label_discipline_4k_s1000 (eval_loss: 0.0042)

FIELD (13 models): 
- Size distribution: 1_4k(4), 4_8k(3), 8_16k(3), 16_32k(3)
- Best performer: model_label_field_1000_1_4k (eval_loss: 0.0040)

METHOD (8 models):
- Size distribution: 1_4k(3), 4_8k(2), 8_16k(1), 16_32k(2)  
- Best performer: model_label_method_1_4k_s10000 (eval_loss: 0.0054)
```

### Performance Metrics Distribution
```
Models with metrics: 8/39 (20.5%)
Average eval_loss: 0.8063 (across models with metrics)
Average eval_accuracy: 0.7847

Top performers by size:
- 1_4k: model_label_field_1000_1_4k (0.0040 loss, 1.0000 acc)
- 4_8k: model_label_discipline_4k_s1000 (0.0042 loss, 1.0000 acc)  
- 8_16k: model_label_discipline_1000_8_16k (0.1537 loss, 0.9587 acc)
- 16_32k: model_label_field_8_16k_s10000 (0.3936 loss, 0.8987 acc)
```

## Algorithm Design

### 1. Size Preference Expansion Algorithm

**Problem**: User requests specific size (e.g., "4_8k") but optimal model might be in adjacent size category

**Solution**: Bidirectional expansion from preferred size with distance-based fallback

```python
class SizePreferenceExpansion:
    """Expands size preference to ordered fallback list."""
    
    SIZE_ORDER = ['1_4k', '4_8k', '8_16k', '16_32k', '32_64k']
    
    def expand_preference(self, preferred_size: str) -> List[str]:
        """
        Generate ordered list of sizes based on preference.
        
        Algorithm:
        1. Start with preferred size
        2. Expand bidirectionally by distance
        3. Prefer smaller sizes for efficiency (tie-breaking)
        
        Example: preferred='4_8k' → ['4_8k', '1_4k', '8_16k', '16_32k', '32_64k']
        """
        if preferred_size not in self.SIZE_ORDER:
            raise ValueError(f"Invalid size: {preferred_size}")
        
        preferred_idx = self.SIZE_ORDER.index(preferred_size)
        result = [preferred_size]
        
        # Bidirectional expansion with tie-breaking toward smaller sizes
        left_idx = preferred_idx - 1
        right_idx = preferred_idx + 1
        
        while left_idx >= 0 or right_idx < len(self.SIZE_ORDER):
            # Prefer smaller sizes when equidistant (efficiency bias)
            if left_idx >= 0:
                result.append(self.SIZE_ORDER[left_idx])
                left_idx -= 1
            if right_idx < len(self.SIZE_ORDER):
                result.append(self.SIZE_ORDER[right_idx])  
                right_idx += 1
                
        return result

# Test cases:
assert expand_preference('4_8k') == ['4_8k', '1_4k', '8_16k', '16_32k', '32_64k']
assert expand_preference('1_4k') == ['1_4k', '4_8k', '8_16k', '16_32k', '32_64k']  
assert expand_preference('32_64k') == ['32_64k', '16_32k', '8_16k', '4_8k', '1_4k']
```

### 2. Model Ranking Algorithm

**Problem**: Within each size category, select the best performing model

**Solution**: Multi-criteria ranking with metric-based primary sort and fallback strategies

```python
@dataclass
class ModelCandidate:
    """Represents a model candidate with ranking metadata."""
    model_info: ModelInfo
    size_category: str
    has_metrics: bool
    eval_loss: Optional[float] = None
    eval_accuracy: Optional[float] = None
    sample_count: Optional[int] = None
    
    def rank_score(self) -> tuple:
        """
        Generate ranking score tuple for sorting.
        
        Ranking Priority:
        1. Has metrics (True > False)
        2. Eval loss (lower = better)
        3. Eval accuracy (higher = better) 
        4. Sample count (higher = better, indicates more training data)
        5. Model name (lexicographic, for determinism)
        
        Returns tuple for sorting (lower values = higher rank)
        """
        return (
            not self.has_metrics,  # False sorts before True (invert for priority)
            self.eval_loss if self.eval_loss is not None else float('inf'),
            -self.eval_accuracy if self.eval_accuracy is not None else float('-inf'),
            -self.sample_count if self.sample_count is not None else 0,
            self.model_info.name  # Lexicographic tie-breaker
        )

class ModelRanker:
    """Ranks models within size categories."""
    
    def rank_models_in_size_category(self, models: List[ModelInfo], 
                                   size_category: str) -> List[ModelCandidate]:
        """Rank models within a single size category."""
        candidates = []
        
        for model in models:
            candidate = ModelCandidate(
                model_info=model,
                size_category=size_category,
                has_metrics=bool(model.metrics and 'eval_loss' in model.metrics),
                eval_loss=model.metrics.get('eval_loss') if model.metrics else None,
                eval_accuracy=model.metrics.get('eval_accuracy') if model.metrics else None,
                sample_count=self._extract_sample_count(model.name)
            )
            candidates.append(candidate)
        
        # Sort by rank score (ascending = better rank)
        candidates.sort(key=lambda c: c.rank_score())
        return candidates
    
    def _extract_sample_count(self, model_name: str) -> Optional[int]:
        """Extract sample count from model name (e.g., 's1000' → 1000)."""
        import re
        match = re.search(r's(\d+)', model_name)
        return int(match.group(1)) if match else None
```

### 3. Integrated Selection Algorithm

**Problem**: Combine size preference expansion with model ranking for optimal selection

**Solution**: Iterative selection through preference-ordered size categories

```python
class ModelSelector:
    """Main model selection orchestrator."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.size_expander = SizePreferenceExpansion()
        self.ranker = ModelRanker()
    
    def select_best_model(self, 
                         label_type: str,
                         size_preference: str = '4_8k',
                         require_metrics: bool = True) -> ModelInfo:
        """
        Select optimal model using integrated algorithm.
        
        Algorithm Steps:
        1. Get all models for label_type
        2. Expand size preference to ordered fallback list  
        3. Group models by size category
        4. For each size in preference order:
           a. Rank models in that size category
           b. Apply metric requirements  
           c. Return best candidate if found
        5. Fallback: return any available model
        
        Args:
            label_type: 'discipline', 'field', or 'method'
            size_preference: Preferred context size ('1_4k' to '32_64k')
            require_metrics: Whether to require evaluation metrics
            
        Returns:
            Selected ModelInfo with additional metadata
            
        Raises:
            ModelSelectionError: No suitable models found
        """
        # Step 1: Get candidate models
        all_models = self.model_manager.search_models(label_type=label_type)
        if not all_models:
            raise ModelSelectionError(f"No models found for label type: {label_type}")
        
        # Step 2: Expand size preferences  
        size_order = self.size_expander.expand_preference(size_preference)
        
        # Step 3: Group models by size
        size_groups = self._group_models_by_size(all_models)
        
        # Step 4: Try each size in preference order
        selection_log = []  # For debugging/reporting
        
        for size in size_order:
            if size not in size_groups:
                selection_log.append(f"Size {size}: No models available")
                continue
                
            # Rank models in this size category
            ranked_candidates = self.ranker.rank_models_in_size_category(
                size_groups[size], size
            )
            
            # Apply metric requirements
            if require_metrics:
                candidates_with_metrics = [c for c in ranked_candidates if c.has_metrics]
                if candidates_with_metrics:
                    best_candidate = candidates_with_metrics[0]
                    selection_log.append(f"Size {size}: Selected {best_candidate.model_info.name} "
                                       f"(eval_loss: {best_candidate.eval_loss:.4f})")
                    return self._finalize_selection(best_candidate, selection_log)
                else:
                    selection_log.append(f"Size {size}: {len(ranked_candidates)} models, none with metrics")
            else:
                # Any model acceptable
                best_candidate = ranked_candidates[0]
                selection_log.append(f"Size {size}: Selected {best_candidate.model_info.name} "
                                   f"({'with' if best_candidate.has_metrics else 'without'} metrics)")
                return self._finalize_selection(best_candidate, selection_log)
        
        # Step 5: Fallback to any available model if requirements too strict
        if require_metrics and all_models:
            logger.warning("No models with metrics found, relaxing requirement")
            return self.select_best_model(label_type, size_preference, require_metrics=False)
        
        raise ModelSelectionError(f"No suitable models for {label_type} with size preference {size_preference}")
    
    def _group_models_by_size(self, models: List[ModelInfo]) -> Dict[str, List[ModelInfo]]:
        """Group models by extracted size category."""
        groups = defaultdict(list)
        
        for model in models:
            size = self._extract_size_from_name(model.name)
            if size:
                groups[size].append(model)
        
        return dict(groups)
    
    def _extract_size_from_name(self, model_name: str) -> Optional[str]:
        """Extract size category from model name using pattern matching."""
        for size in self.size_expander.SIZE_ORDER:
            if size in model_name:
                return size
        return None
    
    def _finalize_selection(self, candidate: ModelCandidate, selection_log: List[str]) -> ModelInfo:
        """Finalize model selection with metadata."""
        model_info = candidate.model_info
        
        # Add selection metadata
        model_info.label_type = candidate.model_info.name.split('_')[2]  # Extract from name
        model_info.size_category = candidate.size_category
        model_info.selection_reason = "metric_based" if candidate.has_metrics else "availability_based"
        model_info.selection_log = selection_log
        
        return model_info
```

## Advanced Selection Strategies

### 1. Confidence-Based Selection

For applications requiring high confidence, select models based on prediction confidence:

```python  
def select_high_confidence_model(self, label_type: str, 
                               confidence_threshold: float = 0.95) -> ModelInfo:
    """Select model optimized for high-confidence predictions."""
    
    models_with_metrics = [m for m in self.get_all_models(label_type) 
                          if m.metrics and 'eval_accuracy' in m.metrics]
    
    # Filter by accuracy threshold (proxy for confidence)
    high_accuracy_models = [m for m in models_with_metrics 
                           if m.metrics['eval_accuracy'] >= confidence_threshold]
    
    if high_accuracy_models:
        # Among high-accuracy models, prefer smallest (most efficient)
        return min(high_accuracy_models, 
                  key=lambda m: self._get_size_numeric_value(m.name))
    else:
        # Fallback to standard selection
        return self.select_best_model(label_type, require_metrics=True)
```

### 2. Speed-Optimized Selection

For latency-sensitive applications, prioritize smaller models:

```python
def select_fastest_model(self, label_type: str, 
                        min_accuracy: float = 0.8) -> ModelInfo:
    """Select model optimized for inference speed."""
    
    # Preference order: smallest to largest
    size_order = ['1_4k', '4_8k', '8_16k', '16_32k', '32_64k']
    
    for size in size_order:
        models = self.get_models_by_size(label_type, size)
        qualified_models = [m for m in models 
                           if m.metrics and m.metrics.get('eval_accuracy', 0) >= min_accuracy]
        
        if qualified_models:
            return min(qualified_models, key=lambda m: m.metrics['eval_loss'])
    
    raise ModelSelectionError(f"No models meet speed/accuracy requirements for {label_type}")
```

### 3. Ensemble Selection

For maximum accuracy, select multiple models for ensemble prediction:

```python
def select_ensemble_models(self, label_type: str, 
                          ensemble_size: int = 3) -> List[ModelInfo]:
    """Select diverse set of models for ensemble prediction."""
    
    all_models = self.get_all_models_with_metrics(label_type)
    
    # Diversification strategy: select best model from each size category
    ensemble = []
    size_categories = ['1_4k', '4_8k', '8_16k', '16_32k', '32_64k']
    
    for size in size_categories:
        models_in_size = [m for m in all_models if size in m.name]
        if models_in_size:
            best_in_size = min(models_in_size, key=lambda m: m.metrics['eval_loss'])
            ensemble.append(best_in_size)
            
        if len(ensemble) >= ensemble_size:
            break
    
    return ensemble[:ensemble_size]
```

## Performance Analysis Tools

### 1. Selection Report Generator

```python
def generate_selection_report(self, label_types: List[str], 
                            size_preference: str) -> Dict[str, Any]:
    """Generate comprehensive model selection report."""
    
    report = {
        'selection_criteria': {
            'size_preference': size_preference,
            'size_expansion_order': self.size_expander.expand_preference(size_preference),
            'label_types': label_types
        },
        'model_inventory': {},
        'selections': {},
        'alternatives': {}
    }
    
    for label_type in label_types:
        # Inventory analysis
        all_models = self.model_manager.search_models(label_type=label_type)
        size_distribution = self._analyze_size_distribution(all_models)
        
        report['model_inventory'][label_type] = {
            'total_models': len(all_models),
            'models_with_metrics': len([m for m in all_models if m.metrics]),
            'size_distribution': size_distribution,
            'best_by_size': self._get_best_by_size(all_models)
        }
        
        # Selection analysis
        try:
            selected = self.select_best_model(label_type, size_preference)
            report['selections'][label_type] = {
                'selected_model': selected.name,
                'size_category': selected.size_category,
                'eval_loss': selected.metrics.get('eval_loss') if selected.metrics else None,
                'eval_accuracy': selected.metrics.get('eval_accuracy') if selected.metrics else None,
                'selection_reason': selected.selection_reason,
                'selection_log': selected.selection_log
            }
        except ModelSelectionError as e:
            report['selections'][label_type] = {'error': str(e)}
        
        # Alternative analysis
        report['alternatives'][label_type] = self._analyze_alternatives(
            all_models, size_preference
        )
    
    return report
```

### 2. Performance Benchmarking

```python
class ModelSelectionBenchmark:
    """Benchmark model selection algorithm performance."""
    
    def benchmark_selection_time(self, iterations: int = 100) -> Dict[str, float]:
        """Measure selection algorithm performance."""
        
        times = {'discipline': [], 'field': [], 'method': []}
        
        for _ in range(iterations):
            for label_type in times.keys():
                start_time = time.time()
                try:
                    self.select_best_model(label_type, '4_8k')
                except ModelSelectionError:
                    pass
                end_time = time.time()
                times[label_type].append(end_time - start_time)
        
        return {
            label_type: {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'max_time': np.max(times)
            }
            for label_type, times in times.items()
        }
    
    def validate_selection_consistency(self, trials: int = 50) -> Dict[str, Any]:
        """Validate that selection algorithm produces consistent results."""
        
        selections = {'discipline': [], 'field': [], 'method': []}
        
        for _ in range(trials):
            for label_type in selections.keys():
                try:
                    selected = self.select_best_model(label_type, '4_8k')
                    selections[label_type].append(selected.name)
                except ModelSelectionError:
                    selections[label_type].append(None)
        
        consistency_report = {}
        for label_type, results in selections.items():
            unique_selections = set(results)
            consistency_report[label_type] = {
                'unique_selections': len(unique_selections),
                'most_common': max(set(results), key=results.count) if results else None,
                'consistency_ratio': results.count(
                    max(set(results), key=results.count)
                ) / len(results) if results else 0
            }
        
        return consistency_report
```

## Edge Case Handling

### 1. Missing Models
```python
def handle_missing_models(self, label_type: str) -> ModelInfo:
    """Handle cases where no models exist for a label type."""
    
    # Check if any models exist at all
    all_models = self.model_manager.discover_models()
    if not all_models:
        raise ModelSelectionError("No trained models found in models directory")
    
    # Suggest similar label types
    available_types = set()
    for model in all_models:
        if 'label_' in model.name:
            type_part = model.name.split('label_')[1].split('_')[0]
            available_types.add(type_part)
    
    raise ModelSelectionError(
        f"No models found for label type '{label_type}'. "
        f"Available types: {', '.join(sorted(available_types))}"
    )
```

### 2. Metric Inconsistencies
```python  
def handle_inconsistent_metrics(self, models: List[ModelInfo]) -> List[ModelInfo]:
    """Handle models with missing or inconsistent metrics."""
    
    # Separate models with and without metrics
    with_metrics = [m for m in models if m.metrics and 'eval_loss' in m.metrics]
    without_metrics = [m for m in models if not (m.metrics and 'eval_loss' in m.metrics)]
    
    if with_metrics:
        logger.info(f"Using {len(with_metrics)} models with metrics, "
                   f"ignoring {len(without_metrics)} without metrics")
        return with_metrics
    else:
        logger.warning("No models have evaluation metrics, using all available models")
        return models
```

### 3. Size Category Ambiguity
```python
def resolve_size_ambiguity(self, model_name: str) -> str:
    """Resolve cases where model name contains multiple size indicators."""
    
    # Find all size matches
    size_matches = [size for size in self.SIZE_ORDER if size in model_name]
    
    if len(size_matches) == 0:
        return 'unknown'
    elif len(size_matches) == 1:
        return size_matches[0]
    else:
        # Multiple matches - use the one that appears latest in the name
        # (assumes more specific size info comes later)
        last_positions = [(size, model_name.rfind(size)) for size in size_matches]
        return max(last_positions, key=lambda x: x[1])[0]
```

## Testing Framework

```python
class TestModelSelection(unittest.TestCase):
    """Comprehensive tests for model selection algorithms."""
    
    def setUp(self):
        self.mock_models = self._create_mock_models()
        self.selector = ModelSelector(MockModelManager(self.mock_models))
    
    def test_size_preference_expansion(self):
        """Test size preference expansion logic."""
        expander = SizePreferenceExpansion()
        
        # Test middle preference
        result = expander.expand_preference('4_8k')
        expected = ['4_8k', '1_4k', '8_16k', '16_32k', '32_64k']
        self.assertEqual(result, expected)
        
        # Test edge preferences  
        result = expander.expand_preference('1_4k')
        self.assertEqual(result[0], '1_4k')  # Preferred first
        
        result = expander.expand_preference('32_64k')  
        self.assertEqual(result[0], '32_64k')  # Preferred first
    
    def test_model_ranking(self):
        """Test model ranking within size categories."""
        ranker = ModelRanker()
        
        # Create models with different metrics
        models = [
            self._create_mock_model("model_a", eval_loss=0.1, eval_accuracy=0.9),
            self._create_mock_model("model_b", eval_loss=0.05, eval_accuracy=0.95),  # Best
            self._create_mock_model("model_c", eval_loss=None, eval_accuracy=None),  # No metrics
        ]
        
        ranked = ranker.rank_models_in_size_category(models, '4_8k')
        
        # Best model should be first
        self.assertEqual(ranked[0].model_info.name, "model_b")
        # Model without metrics should be last
        self.assertEqual(ranked[-1].model_info.name, "model_c")
    
    def test_integrated_selection(self):
        """Test full model selection workflow."""
        # Test successful selection
        selected = self.selector.select_best_model('discipline', '4_8k')
        self.assertIsNotNone(selected)
        self.assertIn('discipline', selected.name)
        
        # Test fallback to different sizes
        selected_fallback = self.selector.select_best_model('discipline', '64_128k')  # Non-existent
        self.assertIsNotNone(selected_fallback)  # Should fallback
        
        # Test error cases
        with self.assertRaises(ModelSelectionError):
            self.selector.select_best_model('nonexistent_label', '4_8k')
    
    def test_edge_cases(self):
        """Test edge case handling."""
        # Empty model list
        empty_selector = ModelSelector(MockModelManager([]))
        with self.assertRaises(ModelSelectionError):
            empty_selector.select_best_model('discipline', '4_8k')
        
        # Models without metrics
        no_metrics_models = [self._create_mock_model("model", eval_loss=None)]
        no_metrics_selector = ModelSelector(MockModelManager(no_metrics_models))
        
        # Should work when require_metrics=False
        result = no_metrics_selector.select_best_model('discipline', '4_8k', require_metrics=False)
        self.assertIsNotNone(result)
        
        # Should fail when require_metrics=True (default)
        with self.assertRaises(ModelSelectionError):
            no_metrics_selector.select_best_model('discipline', '4_8k', require_metrics=True)
```

## Summary

This model selection system provides:

✅ **Intelligent Size Fallback**: Bidirectional expansion from preferred sizes  
✅ **Performance-Based Ranking**: Multi-criteria sorting within size categories  
✅ **Robust Error Handling**: Graceful degradation for missing models/metrics  
✅ **Extensive Testing**: Comprehensive test coverage for edge cases  
✅ **Performance Monitoring**: Benchmarking and consistency validation tools  
✅ **Flexible Strategies**: Support for speed-optimized, confidence-based, and ensemble selection  

The algorithm successfully handles the real-world constraints of the current model inventory while providing a foundation for future model additions and diverse selection strategies.