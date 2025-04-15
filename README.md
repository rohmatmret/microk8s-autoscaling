# MicroK8s Adaptive Autoscaling with Reinforcement Learning

**MicroK8s** | **Reinforcement Learning** | **k6 Load Testing**

Cost-Efficient Solution for Startup Scalability  
This project integrates **MicroK8s** (lightweight Kubernetes) with **Reinforcement Learning (RL)** for adaptive autoscaling in startups, reducing cloud costs by up to 30% compared to traditional solutions (HPA/CA).

---

## 📋 Key Features
- 🚀 **Two-layer autoscaling** (Pod + Node) based on RL (DQN/PPO)  
- 📉 Optimization for **latency (<200ms)** and **resource efficiency (CPU/memory <85%)**  
- 💡 Local simulation using **k6** and monitoring via **Prometheus+Grafana** for cost-saving

---

## 🛠 Requirements
### Hardware
| Component | Minimum Specs            | Notes                        |
| --------- | ------------------------ | ---------------------------- |
| OS        | Ubuntu 20.04+/Debian 11+ | WSL2/Docker (Windows/macOS)  |
| CPU       | 2 cores                  | For MicroK8s + RL            |
| RAM       | 4 GB                     | 2 GB MicroK8s, 2 GB app      |

### This project implements an adaptive autoscaling solution for startups using:
- **MicroK8s** (lightweight Kubernetes distribution)
- **Reinforcement Learning** (DQN/PPO algorithms)
- **k6** for load testing
- **Prometheus + Grafana** for monitoring

### Software Dependencies
- MicroK8s: `snap install microk8s --classic`
- k6: `sudo apt-get install k6`
- Python 3.8+ with packages:

### Software
```bash
# MicroK8s
snap install microk8s --classic

# k6
sudo apt-get install k6

# macOS
brew install k6

# Windows
choco install k6

# Python
pip install tensorflow pytorch kubernetes
```

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
multipass launch --name microk8s-vm --mem 4G --disk 20G

# Install MicroK8s inside the VM
multipass exec microk8s-vm -- sudo snap install microk8s --classic

# Add your user to the microk8s group
multipass exec microk8s-vm -- sudo usermod -a -G microk8s ubuntu

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
│   └── prometheus.yml          # Deployment Prometheus
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
|── .gitignore                  # Ignore files for Git
|- Makefile                    # Build automation (opsional)
└── LICENSE                     # Lisensi proyek
```


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
