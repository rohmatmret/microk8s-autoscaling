#!/bin/bash

# KEDA vs HPA Comparison Runner Script
# This script runs the KEDA comparison from the root directory

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_status "Running KEDA vs HPA comparison from: $PROJECT_ROOT"

# Change to the scripts directory and run the comparison
cd "$SCRIPT_DIR"

# Check if the comparison script exists
if [ ! -f "keda-comparison-script.sh" ]; then
    print_error "KEDA comparison script not found!"
    exit 1
fi

# Make sure the script is executable
chmod +x keda-comparison-script.sh

# Run the comparison script with all arguments
./keda-comparison-script.sh "$@"

print_success "KEDA comparison completed!" 