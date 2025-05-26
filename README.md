# MicroK8s Adaptive Autoscaling with Reinforcement Learning

<p align="center">
  <img src="https://raw.githubusercontent.com/kubernetes/kubernetes/master/logo/logo.png" alt="Kubernetes" height="48" style="vertical-align:middle; margin-left:10px;"/>
    <img src="https://www.tensorflow.org/images/tf_logo_social.png" alt="TensorFlow" height="48" style="vertical-align:middle; margin-left:10px;"/>
    <img src="https://site.wandb.ai/wp-content/uploads/2023/05/wb-cw.svg" alt="Weights & Biases" height="48" style="vertical-align:middle; margin-left:10px;"/>
</p>

**MicroK8s** | **Reinforcement Learning** | **k6 Load Testing**

Cost-Efficient Solution for Startup Scalability  
This project integrates **MicroK8s** (lightweight Kubernetes) with **Reinforcement Learning (RL)** for adaptive autoscaling in startups, reducing cloud costs by up to 30% compared to traditional solutions (HPA/CA).

---

## 📋 Key Features
- 🚀 **Autoscaling** (Pod ) based on RL (DQN/PPO)  
- 📉 Optimization for **latency (<200ms)** and **resource efficiency (CPU/memory <85%)**  
- 💡 Local simulation using **k6** and monitoring via **Prometheus+Grafana** for cost-saving

## 🔄 Autoscaling Solutions Comparison

| Functional Aspect | Traditional HPA | KEDA (Kubernetes Event-Driven Autoscaler) | RL Adaptive (This Thesis: DQN + PPO) |
|-------------------|----------------|------------------------------------------|--------------------------------------|
| **Scaling Paradigm** | Reactive based on static thresholds | Reactive based on event triggers | Proactive and adaptive based on policy learning |
| **Scaling Triggers** | Internal system metrics (CPU, memory) | External events from message queues, databases, Prometheus metrics, etc. | Simulation environment state: CPU, latency, queue, and dynamically evaluated rewards |
| **Decision-Making Strategy** | Fixed interval evaluation and metric-based averaging | Event detection through manually configured scalers and triggers | Decision-making based on estimated values (Q-values) and long-term reward optimization |
| **Workload Adaptation Flexibility** | Limited to stable and repetitive load scenarios | More flexible, but still based on static rules | Highly adaptive to dynamic loads and real-time workload pattern changes |
| **Learning Model** | None (rule-based logic) | None (event mapping) | Deep Reinforcement Learning: combination of DQN (decision-making) and PPO (policy optimization) |
| **Scaling Control Granularity** | Limited to pod count | Supports granular triggers (e.g., queues, connections) | Policy-based considering multi-metric state, including queues and latency |
| **Latency Sensitivity** | Not sensitive to application latency | Responsive to events but doesn't consider latency rewards | Explicit reward function considers latency and throughput as primary components |
| **Configuration & Operation Complexity** | Low; simple YAML-based configuration | Medium; requires integration with external event sources | High; involves RL model training, agent coordination, and hyperparameter tuning |
| **Generalization & Transferability** | Low; difficult to adapt to new patterns | Limited to predefined events | High; ability to generalize from previous experiences to new workload patterns |
| **System Overhead** | Minimal; efficient for simple applications | Low to medium depending on trigger complexity | Medium to high; overhead in training and model inference phases |
| **Additional Infrastructure Dependencies** | No additional components required | Requires relevant scalers and triggers | Requires monitoring pipeline (e.g., Prometheus), logging, and RL framework integration |

---

## 🛠 Requirements
### Hardware
| Component | Minimum Specs            | Notes                       |
| --------- | ------------------------ | --------------------------- |
| OS        | Ubuntu 20.04+/Debian 11+ | WSL2/Docker (Windows/macOS) |
| CPU       | 2 cores                  | For MicroK8s + RL           |
| RAM       | 4 GB                     | 2 GB MicroK8s, 2 GB app     |

### This project implements an adaptive autoscaling solution for startups using:
- **MicroK8s** (lightweight Kubernetes distribution)
- **Reinforcement Learning** (DQN/PPO algorithms)
- **k6** for load testing
- **Prometheus + Grafana** for monitoring
- **Python** for RL agent implementation
- **Wandb** for experiment tracking


# Infrastructure Architecture 

![Local Diagram Infrastructure architecture ](/out/architecture/MicroK8s-Full-Architecture.svg)


### Software Dependencies
- MicroK8s: `snap install microk8s --classic`
- k6: `sudo apt-get install k6`
- Python 3.8+ with packages:


# Setup Instructions
1. **Install MicroK8s**: Follow the [MicroK8s installation guide](https://microk8s.io/docs/installing-on-linux) for your OS.
2. **Install k6**: Use the package manager for your OS (e.g., `brew install k6` for macOS).
3. **Install Python dependencies**: Use `pip install -r requirements.txt` to install the required Python packages.
4. **Install Prometheus and Grafana**: Use the following commands to deploy Prometheus and Grafana on MicroK8s:
```bash
microk8s enable prometheus
microk8s enable grafana
```
5. **Deploy the application**: Use the provided Kubernetes YAML files to deploy your application on MicroK8s.
6. **Run k6 load tests**: Use the provided k6 scripts to simulate load on your application and collect performance metrics.
7. **Monitor with Prometheus and Grafana**: Access the Grafana dashboard to visualize performance metrics and monitor the autoscaling behavior of your application.
8. **Run the RL agent**: Use the provided Python scripts to train and run the reinforcement learning agent for autoscaling.
9. **Test the autoscaling**: Simulate load on your application and observe the autoscaling behavior in real-time using Grafana.
10. **Optimize the RL agent**: Fine-tune the RL agent's hyperparameters and training process to improve its performance and adaptability to changing workloads.
11. **Deploy to production**: Once the RL agent is trained and optimized, deploy it to your production environment for adaptive autoscaling.
12. **Monitor and iterate**: Continuously monitor the performance of your application and the RL agent's autoscaling decisions, making adjustments as necessary to improve efficiency and cost-effectiveness.
13. **Documentation**: Refer to the provided documentation for detailed instructions on each step, including configuration files, deployment scripts, and performance metrics.
14. **Contribute**: If you find this project useful, consider contributing by submitting issues, pull requests, or feedback to improve the solution further.
----
# For MacOs user 

## Prerequisites

Before starting, ensure you have the following:

- macOS system with internet access
- Homebrew installed (`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)
- At least 4GB RAM, 2 CPUs, and 20GB disk available for the Multipass VM
- Basic familiarity with terminal commands and Kubernetes concepts (e.g., pods, deployments, services)


>[!NOTE]
> **Note**: For MacOS users, you may need to install MicroK8s using Multipass or Docker Desktop. Follow the instructions below for your OS.


1. Install Multipass (macOS Virtualization Layer)
```sh
# Install via Homebrew (recommended)
brew install --cask multipass

# Verify installation
multipass version
```
2. Launch a MicroK8s VM with Multipass
```sh
# Create a dedicated VM (4GB RAM, 20GB disk)
multipass launch --name microk8s-vm --cpus 2 --memory 4G --disk 20G

# Install MicroK8s inside the VM
multipass exec microk8s-vm -- sudo snap install microk8s --classic

# Add your user to the microk8s group
multipass exec microk8s-vm -- sudo usermod -a -G microk8s ubuntu

```
## Expected Output:
```sh
Name            State             IPv4
microk8s-vm     Running           192.168.64.x

// If not running: Start it with command

multipass start microk8s-vm

```
3. Install MicroK8s in the VM
Install MicroK8s using snap inside the VM.

```sh
multipass shell microk8s-vm
sudo snap install microk8s --classic

```
### Verify microk8s
```sh
microk8s version
```
Expected Output: Version information (e.g., MicroK8s v1.x.x).

If it fails: Check internet connectivity in the VM (ping google.com) and retry.

Add the ubuntu user to the microk8s group:

```sh
sudo usermod -a -G microk8s ubuntu
```
Log out and back in to apply group changes:

```sh
exit
multipass shell microk8s-vm
```

# Konfigurasi Kubeconfig untuk Akses dari Host
Setup Config kubectl from local machine, copy file config from VM

Get Config from VM
```sh
multipass exec microk8s-vm -- /snap/bin/microk8s config > ~/.kube/microk8s-config
```

## Gabungkan dengan kubeconfig lokal:
```sh
KUBECONFIG=~/.kube/config:~/.kube/microk8s-config kubectl config view --flatten > ~/.kube/merged_kubeconfig
mv ~/.kube/merged_kubeconfig ~/.kube/config
```

## Uji akses dari macOS:
```sh
kubectl get nodes
```



## Project Structure
```plaintext
microk8s-rl-autoscaling/  
├── .github/workflows/          # CI/CD (opsional)  
│   └── test.yaml               # Workflow for CI/CD
├── agent/                      # Reinforcement Learning Agent
│   ├── __init__.py             # Inisialisasi package
│   ├── dqn.py                  # implementasi DQN
│   ├── ppo.py                  # implementasi PPO
│   └── kubernetes_api.py       # API Kubernetes 
│   └── environment.py          # Environment for RL
├── deployments/                # Konfigurasi Kubernetes  
│   ├── nginx-deployment.yaml   # Deployment Nginx 
├── load-test/                  # load testing for k6
│   └── loadtest.js             # Simulasi lonjakan trafik  
├── monitoring/                 # Konfigurasi Prometheus/Grafana  
│   ├── prometheus.yaml         # Custom rules (opsional)  
│   └── nginx_rules.yaml        # Custom rules (opsional)
├── requirements.txt            # Dependensi Python  
├── README.md                   # Documentation 
└── scripts/                    # Skrip utilitas  
    ├── install_microk8s.sh     # Auto-install MicroK8s  
    └── run_simulation.sh     # Jalankan simulasi end-to-end  
    ├── stop_simulation.sh     # Stop Simulasi
|── .gitignore                  # Ignore files for Git
|- Makefile                    # Build automation (opsional)
└── LICENSE                     # Lisensi proyek
```
----
# Tutorial
## Read more Code Base [Here](index.md)

----

## Solution Manually Install Grafana (If Addon is Missing)
If Grafana is still not available as an addon, you can deploy it manually using Helm or YAML.

Option A: Install Grafana via Helm

1. Enable Helm in MicroK8s:

```sh
microk8s enable helm
```

2. Add the Grafana Helm repo:

```sh
microk8s helm repo add grafana https://grafana.github.io/helm-charts
microk8s helm repo update
```

3. Install Grafana:
```sh
microk8s helm install grafana grafana/grafana -n monitoring --create-namespace
```

4. Get the admin password:

```sh
microk8s kubectl get secret -n monitoring grafana -o jsonpath='{.data.admin-password}' | base64 --decode
```
5. Port-forward to access Grafana:
```sh
microk8s kubectl port-forward -n monitoring svc/grafana 3000:80
```

Access at http://localhost:3000 (Username: admin, Password from Step 4).


## Manual Load Test

```sh
kubectl create configmap k6-load-script --from-file=load-test/load-test.js
```


## 🧩 Ensure Components Are Running

| Component                     | Namespace   | Check Status Command                           | Criteria / Ideal Status                     |
|-------------------------------|-------------|------------------------------------------------|---------------------------------------------|
| ✅ Ingress Controller Pod     | ingress     | `kubectl -n ingress get pods --show-labels`   | Status = `Running`, READY = `1/1`, label `name=nginx-ingress-microk8s` |
| ✅ Ingress Controller Service | ingress     | `kubectl -n ingress get svc nginx-ingress-controller` | Type = `NodePort`, has `Endpoints` |
| ✅ Ingress Endpoints          | ingress     | `kubectl -n ingress get endpoints nginx-ingress-controller` | Should show backend Pod IP:Port |
| ✅ Ingress Resource           | app-specific| `kubectl get ingress -A`<br>`kubectl describe ingress <name>` | Host/path/backend correct, no errors |
| ✅ Backend App Pod            | default/app | `kubectl get pods -n <namespace>`             | Status = `Running`                         |
| ✅ Backend Service            | default/app | `kubectl get svc -n <namespace>`              | Type = ClusterIP/NodePort, matches Ingress |
| ✅ HPA (Autoscaler)           | default/app | `kubectl get hpa -n <namespace>`              | Active, target matches Pod                |
| ✅ Metrics Server/Prometheus  | monitoring  | `kubectl get pods -n monitoring`              | Status = `Running`                        |
| ✅ Network Access             | -           | `curl http://<hostname>` / browser            | Returns OK response                       |

---

# Quick Troubleshooting

| Symptom                     | Check                                 | Solution                                           |
|-----------------------------|---------------------------------------|---------------------------------------------------|
| Ingress inaccessible        | `nginx-ingress-controller` Service    | Verify `selector`, ensure endpoints appear       |
| 404 from Ingress            | Ingress resource or backend service   | Check path, host, target service, and port       |
| `port-forward` fails        | Empty endpoints                       | Verify label/selector matches                    |
| Autoscaling not working     | HPA + Metrics server                  | Ensure metrics available and target resource matches |
| Cannot access from host     | Node IP, NodePort port, firewall      | Use `/etc/hosts` or `port-forward` from host     |
| Ingress log errors          | Controller logs                       | `kubectl -n ingress logs <nginx-pod>`            |


## Validate yaml Client 

```sh
kubectl apply -f simulation --dry-run=client 
```

# MakeFile

1. Start Training Agent
   
```
sudo make  train-simulation AGENT=ppo ENV_MODE=simulate TIMESTEPS=100000 EVAL_EPISODES=100
```

2. Start Training Agent with DQN
```
sudo make  train-simulation AGENT=dqn ENV_MODE=simulate TIMESTEPS=100000 EVAL_EPISODES=100
```
