#!/usr/bin/env python3
"""
Test script for the inference pipeline.
"""

import tempfile
from pathlib import Path

def test_text_classification():
    """Test basic text classification."""
    print("🧪 Testing text classification...")
    
    from packages.paper_classifier.src.paper_classifier.inference_utils import InferenceEngine
    
    # Sample academic text
    test_text = """
    This paper presents a novel deep learning approach for natural language processing.
    We propose a transformer-based architecture that improves upon existing methods
    for text classification tasks. Our experimental evaluation on benchmark datasets
    shows significant improvements in accuracy and computational efficiency.
    """
    
    try:
        engine = InferenceEngine(models_dir="packages/paper_classifier/trained_models")
        result = engine.classify_text(test_text, label_types=['discipline'])
        
        print("✅ Text classification successful!")
        print(f"   Discipline: {result.discipline.predicted_label if result.discipline else 'None'}")
        confidence = result.discipline.confidence if result.discipline else 0
        print(f"   Confidence: {confidence:.2%}" if result.discipline else "   Confidence: N/A")
        
        return True
        
    except Exception as e:
        print(f"❌ Text classification failed: {e}")
        return False


def test_cli_interface():
    """Test CLI interface."""
    print("\n🧪 Testing CLI interface...")
    
    # Create a temporary text file
    test_text = "This is a machine learning paper about computer vision algorithms."
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_text)
        temp_file = Path(f.name)
    
    try:
        import subprocess
        
        # Test classify-paper command
        result = subprocess.run([
            'classify-paper', 'text', 
            '--text-file', str(temp_file),
            '--discipline', '--json'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ CLI interface working!")
            print(f"   Output: {result.stdout[:100]}...")
            return True
        else:
            print(f"❌ CLI failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False
    finally:
        # Clean up
        temp_file.unlink()


def test_model_listing():
    """Test model listing functionality."""
    print("\n🧪 Testing model listing...")
    
    try:
        from packages.paper_classifier.src.paper_classifier.model_manager import ModelManager
        
        manager = ModelManager(models_dir="packages/paper_classifier/trained_models")
        models = manager.discover_models()
        
        print(f"✅ Found {len(models)} trained models")
        
        # Show some examples
        for i, model in enumerate(models[:3]):
            print(f"   {i+1}. {model.name}")
        
        if len(models) > 3:
            print(f"   ... and {len(models) - 3} more")
            
        return len(models) > 0
        
    except Exception as e:
        print(f"❌ Model listing failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Paper Classification Inference Pipeline")
    print("=" * 50)
    
    tests = [
        test_model_listing,
        test_text_classification,
        test_cli_interface,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Inference pipeline is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above.")


if __name__ == "__main__":
    main()