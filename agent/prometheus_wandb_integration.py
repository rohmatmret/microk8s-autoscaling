import wandb
import prometheus_client
import requests
import time
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

# Callback untuk logging ke wandb
class WandbPrometheusCallback(BaseCallback):
    def __init__(self, prometheus_url="http://localhost:9090", verbose=0):
        super(WandbPrometheusCallback, self).__init__(verbose=verbose)
        self.prometheus_url = prometheus_url
    
    def _on_step(self) -> bool:
        # Query metrik dari Prometheus
        try:
            # Contoh query untuk latensi (sesuaikan dengan metrik Anda)
            latency_query = 'rate(http_request_duration_seconds_sum[1m])/rate(http_request_duration_seconds_count[1m])'
            latency_response = requests.get(f"{self.prometheus_url}/api/v1/query", params={"query": latency_query})
            latency_ms = float(latency_response.json()["data"]["result"][0]["value"][1]) * 1000  # Konversi ke ms

            # Query untuk penggunaan CPU
            cpu_query = 'rate(container_cpu_usage_seconds_total[1m])'
            cpu_response = requests.get(f"{self.prometheus_url}/api/v1/query", params={"query": cpu_query})
            cpu_percent = float(cpu_response.json()["data"]["result"][0]["value"][1]) * 100

            # Query untuk throughput
            throughput_query = 'rate(http_requests_total[1m])'
            throughput_response = requests.get(f"{self.prometheus_url}/api/v1/query", params={"query": throughput_query})
            throughput = float(throughput_response.json()["data"]["result"][0]["value"][1])

            # Log metrik ke wandb
            wandb.log({
                "latency_ms": latency_ms,
                "cpu_usage_percent": cpu_percent,
                "throughput_req_per_sec": throughput,
                "episode_reward": self.locals["rewards"][-1] if self.locals["rewards"] else 0,
                "q_value_mean": self.model.q_net.q_values.mean().item()
            })
        except Exception as e:
            print(f"Error querying Prometheus: {e}")
        return True

# Inisialisasi wandb
wandb.init(
    project="microk8s-autoscaling",
    group="scenario-sudden-spike",
    name="dqn-run-1",
    config={
        "learning_rate": 0.001,
        "buffer_size": 10000,
        "gamma": 0.99
    }
)

# Setup lingkungan simulasi
env = gym.make("CustomMicroK8sEnv-v0")  # Ganti dengan environment Anda

# Inisialisasi model DQN
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.001,
    buffer_size=10000,
    gamma=0.99,
    verbose=1
)

# Latih model dengan callback wandb
model.learn(total_timesteps=50000, callback=WandbPrometheusCallback(prometheus_url="http://localhost:9090"))

# Simpan model
model.save("dqn_microk8s")

# Selesai logging wandb
wandb.finish()