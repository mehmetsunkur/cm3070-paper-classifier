#!/usr/bin/env python3
"""
Example usage of the paper classification inference system.
"""

def main():
    print("📄 Paper Classification Inference Examples")
    print("=" * 50)
    
    print("\n1. 📋 List available models:")
    print("   classify-paper list-models --summary")
    
    print("\n2. 📝 Classify text directly:")
    print('   classify-paper text --text "This paper presents a deep learning approach for computer vision tasks."')
    
    print("\n3. 📄 Classify text from file:")
    print("   classify-paper text --text-file paper.txt")
    
    print("\n4. 📑 Classify PDF file:")
    print("   classify-paper pdf paper.pdf")
    
    print("\n5. 🏷️  Classify specific label types only:")
    print("   classify-paper pdf paper.pdf --discipline --field")
    
    print("\n6. 📊 Get JSON output:")
    print("   classify-paper pdf paper.pdf --json")
    
    print("\n7. 📁 Batch process directory:")
    print("   classify-paper batch /path/to/pdfs --output results.json")
    
    print("\n8. 🔍 Search models by criteria:")
    print("   classify-paper list-models --search-label discipline --search-size 4_8k")
    
    print("\n9. 🚀 Force specific size rank:")
    print("   classify-paper text --text-file large_paper.txt --size-rank 16_32k")
    
    print("\n📚 Available Commands:")
    print("   • text      - Classify text input")
    print("   • pdf       - Classify PDF file") 
    print("   • batch     - Batch classify PDF files")
    print("   • list-models - List available models")
    
    print("\n🏷️  Label Types:")
    print("   • discipline - Academic discipline (Computer Science, etc.)")
    print("   • field      - Research field (AI, Machine Learning, etc.)")
    print("   • method     - Research method (Empirical Study, etc.)")
    
    print("\n📏 Size Ranks (auto-detected):")
    print("   • 1_4k      - Up to 4,000 tokens")
    print("   • 4_8k      - 4,000-8,000 tokens")
    print("   • 8_16k     - 8,000-16,000 tokens")
    print("   • 16_32k    - 16,000-32,000 tokens")
    print("   • 32_64k    - 32,000-64,000 tokens")
    print("   • 64_128k   - 64,000+ tokens")

if __name__ == "__main__":
    main()