#!/usr/bin/env python3
"""
Command-line interface for paper classification inference.
Provides CLI commands for classifying text and PDF files.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

from .inference_utils import InferenceEngine, format_result_human_readable


def classify_text_command(args):
    """Handle text classification command."""
    engine = InferenceEngine(models_dir=args.models_dir)
    
    # Read text from file or stdin
    if args.text_file:
        text_path = Path(args.text_file)
        if not text_path.exists():
            print(f"Error: Text file not found: {text_path}", file=sys.stderr)
            sys.exit(1)
        
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("Reading text from stdin... (Press Ctrl+D when done)")
        text = sys.stdin.read()
    
    if not text.strip():
        print("Error: No text provided for classification", file=sys.stderr)
        sys.exit(1)
    
    # Determine label types to classify
    label_types = []
    if args.discipline:
        label_types.append('discipline')
    if args.field:
        label_types.append('field')
    if args.method:
        label_types.append('method')
    
    if not label_types:
        label_types = ['discipline', 'field', 'method']  # Default: all
    
    try:
        # Run classification
        result = engine.classify_text(
            text=text,
            label_types=label_types,
            size_rank=args.size_rank
        )
        
        # Output results
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(format_result_human_readable(result, include_confidence=not args.no_confidence))
            
    except Exception as e:
        print(f"Error during classification: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def classify_pdf_command(args):
    """Handle PDF classification command."""
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    
    engine = InferenceEngine(models_dir=args.models_dir)
    
    # Determine label types to classify
    label_types = []
    if args.discipline:
        label_types.append('discipline')
    if args.field:
        label_types.append('field')
    if args.method:
        label_types.append('method')
    
    if not label_types:
        label_types = ['discipline', 'field', 'method']  # Default: all
    
    try:
        # Run PDF classification
        result = engine.classify_pdf(
            pdf_path=pdf_path,
            label_types=label_types
        )
        
        # Output results
        if args.json:
            output = result.to_dict()
            output['pdf_file'] = str(pdf_path)
            print(json.dumps(output, indent=2))
        else:
            print(f"📁 PDF File: {pdf_path.name}")
            print(format_result_human_readable(result, include_confidence=not args.no_confidence))
            
    except Exception as e:
        print(f"Error during PDF classification: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def batch_classify_command(args):
    """Handle batch classification command."""
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Find PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files to process...")
    
    engine = InferenceEngine(models_dir=args.models_dir)
    
    # Determine label types to classify
    label_types = []
    if args.discipline:
        label_types.append('discipline')
    if args.field:
        label_types.append('field')
    if args.method:
        label_types.append('method')
    
    if not label_types:
        label_types = ['discipline', 'field', 'method']  # Default: all
    
    results = []
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\nProcessing {i}/{len(pdf_files)}: {pdf_file.name}")
        
        try:
            result = engine.classify_pdf(
                pdf_path=pdf_file,
                label_types=label_types
            )
            
            output = result.to_dict()
            output['pdf_file'] = str(pdf_file)
            results.append(output)
            
            if not args.quiet:
                print(format_result_human_readable(result, include_confidence=False))
                
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            continue
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {output_path}")
    else:
        print(f"\n📊 Batch Results Summary:")
        print(json.dumps(results, indent=2))


def list_models_command(args):
    """Handle list models command."""
    from .model_manager import ModelManager
    
    manager = ModelManager(models_dir=args.models_dir, verbose=args.verbose)
    
    if args.summary:
        manager.print_summary()
    elif args.search_label or args.search_size or args.search_samples:
        models = manager.search_models(
            label_type=args.search_label,
            size_range=args.search_size,
            sample_count=args.search_samples
        )
        print(f"\nFound {len(models)} matching models:")
        for model in models:
            print(f"  • {model.name}")
    else:
        models = manager.discover_models()
        print(f"Available models ({len(models)}):")
        for model in models:
            print(f"  • {model.name}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Paper Classification CLI - Classify academic papers by discipline, field, and method",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify text from file
  classify-paper --text-file paper.txt
  
  # Classify PDF with specific labels
  classify-paper --pdf paper.pdf --discipline --field
  
  # Batch process directory
  classify-paper --batch /path/to/pdfs --output results.json
  
  # Classify text with JSON output
  echo "This paper presents a machine learning approach..." | classify-paper --json
        """
    )
    
    # Global options
    parser.add_argument('--models-dir', default='trained_models',
                       help='Directory containing trained models (default: trained_models)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Text classification command
    text_parser = subparsers.add_parser('text', help='Classify text input')
    text_parser.add_argument('--text', type=str, help='Text to classify')
    text_parser.add_argument('--text-file', type=str, help='File containing text to classify')
    text_parser.add_argument('--size-rank', type=str, 
                            choices=['1_4k', '4_8k', '8_16k', '16_32k', '32_64k', '64_128k'],
                            help='Force specific size rank')
    
    # PDF classification command  
    pdf_parser = subparsers.add_parser('pdf', help='Classify PDF file')
    pdf_parser.add_argument('pdf_file', type=str, help='PDF file to classify')
    
    # Batch classification command
    batch_parser = subparsers.add_parser('batch', help='Batch classify PDF files')
    batch_parser.add_argument('input_dir', type=str, help='Directory containing PDF files')
    batch_parser.add_argument('--output', type=str, help='Output JSON file for results')
    batch_parser.add_argument('--quiet', action='store_true', help='Suppress individual results')
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available models')
    list_parser.add_argument('--summary', action='store_true', help='Show model summary')
    list_parser.add_argument('--search-label', choices=['discipline', 'field', 'method'],
                            help='Filter by label type')
    list_parser.add_argument('--search-size', help='Filter by size range (e.g., "4_8k")')
    list_parser.add_argument('--search-samples', help='Filter by sample count (e.g., "s5000")')
    list_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Add common options to classification commands
    for cmd_parser in [text_parser, pdf_parser, batch_parser]:
        cmd_parser.add_argument('--discipline', action='store_true',
                               help='Classify discipline only')
        cmd_parser.add_argument('--field', action='store_true', 
                               help='Classify field only')
        cmd_parser.add_argument('--method', action='store_true',
                               help='Classify method only')
        cmd_parser.add_argument('--json', action='store_true',
                               help='Output results as JSON')
        cmd_parser.add_argument('--no-confidence', action='store_true',
                               help='Hide confidence scores')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'text':
        classify_text_command(args)
    elif args.command == 'pdf':
        classify_pdf_command(args)
    elif args.command == 'batch':
        batch_classify_command(args)
    elif args.command == 'list-models':
        list_models_command(args)
    else:
        # Default behavior: if no subcommand, try to infer from arguments
        # Check for direct file argument
        if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
            potential_file = Path(sys.argv[1])
            if potential_file.exists():
                if potential_file.suffix.lower() == '.pdf':
                    # Treat as PDF classification
                    args.pdf_file = str(potential_file)
                    args.command = 'pdf'
                    # Set defaults for missing attributes
                    for attr in ['discipline', 'field', 'method', 'json', 'no_confidence']:
                        if not hasattr(args, attr):
                            setattr(args, attr, False)
                    classify_pdf_command(args)
                else:
                    # Treat as text file
                    args.text_file = str(potential_file)
                    args.text = None
                    args.command = 'text'
                    # Set defaults for missing attributes
                    for attr in ['discipline', 'field', 'method', 'json', 'no_confidence', 'size_rank']:
                        if not hasattr(args, attr):
                            setattr(args, attr, False if attr != 'size_rank' else None)
                    classify_text_command(args)
            else:
                parser.print_help()
        else:
            parser.print_help()


if __name__ == "__main__":
    main()