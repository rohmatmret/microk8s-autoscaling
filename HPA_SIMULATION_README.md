# HPA Simulation with MicroK8s

This project demonstrates Horizontal Pod Autoscaler (HPA) functionality using MicroK8s, nginx, and k6 load testing.

## Overview

The simulation includes:
- **Nginx Deployment**: Web server with Prometheus metrics exporter
- **HPA Configuration**: CPU-based autoscaling (70% threshold)
- **External Load Testing**: Tools and scripts for testing from outside the cluster
- **Monitoring**: Real-time resource monitoring and logging
- **Ingress**: External access configuration

## Prerequisites

1. **MicroK8s** installed on your system
2. **kubectl** command-line tool
3. **Bash** shell environment

### Installing Prerequisites

#### macOS
```bash
# Install MicroK8s
brew install microk8s

# Install kubectl
brew install kubectl

# Install k6 (recommended for load testing)
brew install k6
```

#### Ubuntu/Debian
```bash
# Install MicroK8s
sudo snap install microk8s --classic

# Install kubectl
sudo apt-get update && sudo apt-get install -y kubectl

# Install k6 (recommended for load testing)
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update && sudo apt-get install k6
```

#### Automatic Installation
```bash
# Use the provided installation script
./scripts/install_k6.sh
```

## Quick Start

### 1. Run the Complete Simulation

```bash
# Make scripts executable (if not already done)
chmod +x scripts/run_hpa_simulation.sh
chmod +x scripts/cleanup_hpa_simulation.sh
chmod +x scripts/external_load_test.sh

# Run the simulation
./scripts/run_hpa_simulation.sh
```

### 2. Test HPA Scaling (External)

```bash
# Use the provided external URL and run load testing
./scripts/external_load_test.sh <external_url> [duration] [concurrent_requests]

# Example:
./scripts/external_load_test.sh http://192.168.64.6:30000 300 50
```

### 3. Clean Up Resources

```bash
# Clean up all resources
./scripts/cleanup_hpa_simulation.sh
```

## What the Script Does

### 1. **Prerequisites Check**
- Verifies MicroK8s and kubectl are installed
- Starts MicroK8s if not running

### 2. **Addon Configuration**
- Enables required MicroK8s addons:
  - `ingress`: For external access
  - `metrics-server`: For resource monitoring
  - `prometheus`: For metrics collection
  - `dashboard`: For web-based monitoring

### 3. **Resource Deployment**
- Creates nginx ConfigMap with status endpoint
- Deploys nginx with Prometheus exporter
- Applies HPA configuration (CPU-based, 70% threshold)
- Sets up ingress for external access
- Creates k6 load test ConfigMap

### 4. **External Testing Setup**
- Provides external access URL for testing
- Offers multiple load testing options:
  - curl-based testing
  - Apache Bench (ab)
  - wrk load testing tool
  - Manual curl loops
- Interactive testing with real-time monitoring

### 5. **Monitoring**
- Real-time monitoring of:
  - Pod status and scaling
  - HPA metrics
  - Resource usage
  - Service status

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   External      │    │   Ingress       │    │   Nginx         │
│   Load Test     │───▶│   Controller    │───▶│   Deployment    │
│   (curl/wrk/ab) │    │                 │    │   (HPA)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Metrics       │    │   Prometheus    │
                       │   Server        │    │   Exporter      │
                       └─────────────────┘    └─────────────────┘
```

## Key Components

### 1. **Nginx Deployment** (`deployments/nginx-deployment.yaml`)
- **Image**: `nginx:latest`
- **Resources**: 100m CPU request, 500m CPU limit
- **Sidecar**: nginx-prometheus-exporter for metrics
- **Config**: Custom nginx.conf with status endpoint

### 2. **HPA Configuration** (`deployments/nginx-hpa.yaml`)
- **Target**: CPU utilization at 70%
- **Scaling**: 1-5 replicas
- **Metrics**: Resource-based CPU metrics

### 3. **External Load Testing** (`scripts/external_load_test.sh`)
- **Multiple Tools**: k6 (recommended), wrk, Apache Bench, curl
- **Flexible Configuration**: Duration, concurrent requests, stages
- **External Access**: Testing from outside the cluster
- **Real-time Monitoring**: Interactive HPA observation
- **Professional Features**: Thresholds, metrics, reporting

### 4. **Ingress** (`deployments/ingres.yaml`)
- **Class**: nginx
- **Path**: Root path (/) to nginx service
- **Access**: External traffic routing

## Monitoring and Debugging

### Real-time Monitoring
```bash
# Watch HPA scaling
kubectl get hpa -w

# Monitor resource usage
kubectl top pods

# View pod logs
kubectl logs -f deployment/nginx

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp

# Access Grafana dashboard
./scripts/access_observability.sh
```

### Log Files
- **Monitoring Logs**: `logs/hpa_monitoring.log`
- **Nginx Logs**: Available via kubectl logs
- **Observability**: Grafana, Prometheus, Alertmanager via access script

### Useful Commands
```bash
# Get external access URL
kubectl get ingress nginx-ingress

# Check HPA status
kubectl describe hpa nginx-hpa

# View service endpoints
kubectl get endpoints

# Monitor scaling events
kubectl get events --field-selector involvedObject.kind=HorizontalPodAutoscaler

# Access observability stack
./scripts/access_observability.sh
```

## Troubleshooting

### Common Issues

1. **MicroK8s Not Starting**
   ```bash
   microk8s status
   microk8s start
   ```

2. **Addons Not Ready**
   ```bash
   microk8s enable ingress
   microk8s enable metrics-server
   ```

3. **HPA Not Scaling**
   ```bash
   # Check metrics server
   kubectl top pods
   
   # Check HPA events
   kubectl describe hpa nginx-hpa
   ```

4. **External Load Test Failing**
   ```bash
   # Test connectivity manually
   curl <external_url>
   
   # Verify service connectivity
   kubectl get svc nginx
   
   # Check ingress status
   kubectl get ingress nginx-ingress
   ```

### Reset Everything
```bash
# Complete reset
microk8s reset
microk8s start
```

## Expected Behavior

### Scaling Timeline
1. **Initial State**: 1 nginx pod running
2. **Load Increase**: External tools generate traffic
3. **CPU Spike**: CPU usage exceeds 70% threshold
4. **HPA Trigger**: HPA scales up to 2-3 replicas
5. **Load Decrease**: Traffic reduces
6. **Scale Down**: HPA scales back to 1 replica

### Performance Metrics
- **Target Latency**: <200ms (95th percentile)
- **Error Rate**: <10%
- **Scaling Time**: 1-3 minutes for scale-up

## Customization

### Modify HPA Threshold
Edit `deployments/nginx-hpa.yaml`:
```yaml
target:
  type: Utilization
  averageUtilization: 50  # Change from 70 to 50
```

### Adjust Load Test
Modify the external load test parameters:
```bash
# Using k6 (recommended)
./scripts/external_load_test.sh <url> 600 100  # 10 minutes, 100 concurrent

# Using wrk directly
wrk -t12 -c200 -d300s <url>  # 200 concurrent for 5 minutes

# Using k6 with custom script
k6 run --vus 50 --duration 300s <script.js>
```

### Change Resource Limits
Edit the deployment resources:
```yaml
resources:
  requests:
    cpu: "200m"    # Increase request
  limits:
    cpu: "1000m"   # Increase limit
```

## Cleanup

To remove all resources:
```bash
./scripts/cleanup_hpa_simulation.sh
```

This will delete:
- All deployments and services
- HPA configuration
- ConfigMaps
- Ingress resources

## Next Steps

After running the simulation, you can:
1. Analyze the monitoring logs
2. Experiment with different HPA thresholds
3. Test with different load patterns
4. Integrate with Prometheus for advanced metrics
5. Add custom metrics for more sophisticated scaling

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the monitoring logs
3. Verify MicroK8s addon status
4. Check Kubernetes events for errors 