#!/bin/bash

# External Load Testing Script for HPA
# This script generates load from outside the cluster to test HPA scaling

set -e  # Exit on any error

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

# Check if URL is provided
if [ $# -eq 0 ]; then
    print_error "Usage: $0 <target_url> [duration_seconds] [concurrent_requests]"
    print_status "Example: $0 http://192.168.64.8:32420 300 50"
    exit 1
fi

TARGET_URL=$1
DURATION=${2:-300}  # Default 5 minutes
CONCURRENT=${3:-50}  # Default 50 concurrent requests

print_status "Starting external load test..."
print_status "Target URL: $TARGET_URL"
print_status "Duration: $DURATION seconds"
print_status "Concurrent requests: $CONCURRENT"

# Check if curl is available
if ! command -v curl >/dev/null 2>&1; then
    print_error "curl is not installed. Please install curl first."
    exit 1
fi

# Check for load testing tools and provide installation hints
print_status "Checking available load testing tools..."
if command -v k6 >/dev/null 2>&1; then
    print_success "k6 is available"
else
    print_warning "k6 not found. To install k6:"
    echo "  macOS: brew install k6"
    echo "  Ubuntu/Debian: sudo apt-get install k6"
    echo "  Or download from: https://k6.io/docs/getting-started/installation/"
fi

# if command -v wrk >/dev/null 2>&1; then
#     print_success "wrk is available"
# else
#     print_warning "wrk not found. To install wrk:"
#     echo "  macOS: brew install wrk"
#     echo "  Ubuntu/Debian: sudo apt-get install wrk"
# fi

# if command -v ab >/dev/null 2>&1; then
#     print_success "Apache Bench (ab) is available"
# else
#     print_warning "Apache Bench not found. To install:"
#     echo "  macOS: brew install httpd"
#     echo "  Ubuntu/Debian: sudo apt-get install apache2-utils"
# fi

# Test if target is reachable
print_status "Testing connectivity to $TARGET_URL..."
if curl -s --max-time 5 "$TARGET_URL" >/dev/null; then
    print_success "Target is reachable!"
else
    print_error "Cannot reach target URL. Please check the URL and try again."
    exit 1
fi

# Function to generate load with curl
generate_load() {
    local url=$1
    local duration=$2
    local concurrent=$3
    
    print_status "Starting load generation..."
    print_status "Press Ctrl+C to stop early"
    
    # Calculate total requests (roughly 10 requests per second per concurrent connection)
    local requests_per_second=$((concurrent * 10))
    local total_requests=$((requests_per_second * duration))
    
    print_status "Estimated total requests: $total_requests"
    
    # Start time
    local start_time=$(date +%s)
    
    # Generate load using curl in parallel
    for ((i=1; i<=concurrent; i++)); do
        (
            while true; do
                curl -s "$url" >/dev/null 2>&1
                sleep 0.1  # Small delay to prevent overwhelming
            done
        ) &
    done
    
    # Store background process IDs
    local pids=($!)
    
    # Wait for specified duration
    sleep "$duration"
    
    # Stop all background processes
    print_status "Stopping load generation..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    
    # Wait for processes to finish
    wait 2>/dev/null || true
    
    local end_time=$(date +%s)
    local actual_duration=$((end_time - start_time))
    
    print_success "Load test completed in ${actual_duration} seconds"
}

# Function to generate load with Apache Bench (if available)
generate_load_ab() {
    local url=$1
    local duration=$2
    local concurrent=$3
    
    if command -v ab >/dev/null 2>&1; then
        print_status "Using Apache Bench for load testing..."
        local requests=$((concurrent * duration * 10))  # Rough estimate
        
        ab -n "$requests" -c "$concurrent" -t "$duration" "$url"
    else
        print_warning "Apache Bench not found, using curl method..."
        generate_load "$url" "$duration" "$concurrent"
    fi
}

# Function to generate load with k6 (if available)
generate_load_k6() {
    local url=$1
    local duration=$2
    local concurrent=$3
    
    if command -v k6 >/dev/null 2>&1; then
        print_status "Using k6 for load testing..."
        
        # Create a temporary k6 script
        cat > /tmp/k6_load_test.js << EOF
import http from 'k6/http';
import { sleep } from 'k6';

export let options = {
  stages: [
    { duration: '30s', target: $((concurrent / 4)) },     // Ramp-up
    { duration: '${duration}s', target: $concurrent },    // Sustained load
    { duration: '30s', target: 0 },                       // Ramp-down
  ],
  thresholds: {
    http_req_duration: ['p(95)<200'],    // Target latency <200ms
    http_req_failed: ['rate<0.1'],       // Error rate <10%
  },
};

export default function () {
  http.get('$url');
  sleep(0.1);
}
EOF
        
        print_status "Running k6 load test..."
        k6 run /tmp/k6_load_test.js
        
        # Clean up
        rm -f /tmp/k6_load_test.js
    else
        print_warning "k6 not found, trying wrk..."
        generate_load_wrk "$url" "$duration" "$concurrent"
    fi
}

# Function to generate load with wrk (if available)
generate_load_wrk() {
    local url=$1
    local duration=$2
    local concurrent=$3
    
    if command -v wrk >/dev/null 2>&1; then
        print_status "Using wrk for load testing..."
        wrk -t12 -c"$concurrent" -d"$duration"s "$url"
    else
        print_warning "wrk not found, trying Apache Bench..."
        generate_load_ab "$url" "$duration" "$concurrent"
    fi
}

# Main execution
print_status "Choose load testing method:"
echo "1. k6 (recommended - modern load testing)"
echo "2. wrk (high-performance HTTP benchmarking)"
echo "3. Apache Bench (ab - traditional load testing)"
echo "4. curl (fallback - always available)"
echo "5. Manual curl loop (for simple testing)"

read -p "Enter choice (1-5): " choice

case $choice in
    1)
        generate_load_k6 "$TARGET_URL" "$DURATION" "$CONCURRENT"
        ;;
    2)
        generate_load_wrk "$TARGET_URL" "$DURATION" "$CONCURRENT"
        ;;
    3)
        generate_load_ab "$TARGET_URL" "$DURATION" "$CONCURRENT"
        ;;
    4)
        generate_load "$TARGET_URL" "$DURATION" "$CONCURRENT"
        ;;
    5)
        print_status "Manual curl loop mode"
        print_status "Run this command in another terminal to monitor HPA:"
        echo "kubectl get hpa -w"
        echo ""
        print_status "Press Ctrl+C to stop the load test"
        while true; do
            curl -s "$TARGET_URL" >/dev/null 2>&1
            sleep 0.1
        done
        ;;
    *)
        print_error "Invalid choice. Using k6 method."
        generate_load_k6 "$TARGET_URL" "$DURATION" "$CONCURRENT"
        ;;
esac

print_success "Load test completed!"
print_status "Check HPA scaling with: kubectl get hpa"
print_status "Monitor resource usage with: kubectl top pods" 