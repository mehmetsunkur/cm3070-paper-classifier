#!/bin/bash

# train.sh
# Training script for the paper classifier package
# Simplified version of the original train_untrained_configs.sh

set -euo pipefail

# ============================================================================
# Configuration Section
# ============================================================================

# Default directories
CONFIG_DIR="${PAPER_CLASSIFIER_CONFIG_DIR:-configs}"
MODEL_DIR="${PAPER_CLASSIFIER_MODEL_DIR:-trained_models}"
LOG_DIR="${PAPER_CLASSIFIER_LOG_DIR:-logs}"

# Training defaults
DEFAULT_GPU_COUNT="${PAPER_CLASSIFIER_GPU_COUNT:-1}"
TRAINER_MODULE="paper_classifier.main"

# Script settings
SCRIPT_NAME="$(basename "$0")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Command-line Arguments
# ============================================================================

DRY_RUN=false
GPU_COUNT="$DEFAULT_GPU_COUNT"
VERBOSE=false
CONFIG_FILE=""

usage() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS] --config CONFIG_FILE

Train a paper classifier model using the specified configuration.

OPTIONS:
    -h, --help              Show this help message
    -c, --config FILE       Path to YAML config file (required)
    -d, --dry-run          Show what would be executed without running
    -g, --gpu-count N      Number of GPUs to use (default: $DEFAULT_GPU_COUNT)
    -v, --verbose          Enable verbose output
    --config-dir PATH      Custom config directory (default: $CONFIG_DIR)
    --model-dir PATH       Custom models directory (default: $MODEL_DIR)

EXAMPLES:
    # Train with a specific config
    $SCRIPT_NAME --config configs/example_config.yaml

    # Train with custom GPU count
    $SCRIPT_NAME --config configs/example_config.yaml --gpu-count 2

    # Dry run to see what would be executed
    $SCRIPT_NAME --config configs/example_config.yaml --dry-run

EOF
    exit 0
}

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    
    case "$level" in
        INFO)
            echo -e "${GREEN}[INFO]${NC} $message"
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} $message"
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
        DEBUG)
            if [ "$VERBOSE" = true ]; then
                echo -e "${BLUE}[DEBUG]${NC} $message"
            fi
            ;;
    esac
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -g|--gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# ============================================================================
# Validation
# ============================================================================

if [ -z "$CONFIG_FILE" ]; then
    log ERROR "Config file is required. Use --config to specify a configuration file."
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    log ERROR "Config file does not exist: $CONFIG_FILE"
    exit 1
fi

# Verify required commands
if ! command -v uv &> /dev/null; then
    log ERROR "uv command not found. Please ensure uv is installed."
    exit 1
fi

# ============================================================================
# Training Execution
# ============================================================================

log INFO "=========================================="
log INFO "Starting Paper Classifier Training"
log INFO "Timestamp: $TIMESTAMP"
log INFO "=========================================="

log INFO "Configuration:"
log INFO "  Config file: $CONFIG_FILE"
log INFO "  GPU count: $GPU_COUNT"
log INFO "  Model directory: $MODEL_DIR"
log INFO "  Log directory: $LOG_DIR"

# Create necessary directories
mkdir -p "$MODEL_DIR" "$LOG_DIR"

# Set environment variables
export TRAIN_UNTRAINED_MODEL_DIR="$MODEL_DIR"
export TRAIN_UNTRAINED_LOG_DIR="$LOG_DIR"

if [ "$DRY_RUN" = true ]; then
    log INFO "[DRY RUN] Would execute:"
    log INFO "  uv run accelerate launch --num_processes=$GPU_COUNT -m $TRAINER_MODULE --config $CONFIG_FILE"
    exit 0
fi

# Execute training
log INFO "Starting training..."
training_log="${LOG_DIR}/training_${TIMESTAMP}.log"

if uv run accelerate launch \
    --num_processes="$GPU_COUNT" \
    -m "$TRAINER_MODULE" \
    --config "$CONFIG_FILE" \
    2>&1 | tee "$training_log"; then
    
    log INFO "Training completed successfully!"
    log INFO "Training log: $training_log"
else
    log ERROR "Training failed!"
    log ERROR "Check log file: $training_log"
    exit 1
fi