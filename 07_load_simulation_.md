# Chapter 7: Load Simulation

Welcome back! We've covered quite a journey so far. We started with the blueprints for our system ([Kubernetes Configuration](01_kubernetes_configuration_.md)), learned how to see what's happening inside ([Observability & Metrics](02_observability___metrics_.md)), understood how our code can control the cluster ([Kubernetes Interaction (API)](03_kubernetes_interaction___api__.md)), saw how scripts automate the whole process ([Project Orchestration Scripts](04_project_orchestration_scripts_.md)), and then dived into the "world" where our RL agent learns ([RL Environment](05_rl_environment_.md)) and the "brain" that does the learning ([RL Agent (DQN/PPO)](06_rl_agent__dqn_ppo__.md)).

In the previous chapter, we saw that the [RL Environment](05_rl_environment_.md) needs to provide the agent with observations (metrics) and a reward based on its actions and the resulting state. For the simulated environment (`MicroK8sEnvSimulated`), this state isn't coming from a real cluster; it's being *simulated*.

But what drives the changes in the simulated cluster's performance? What causes the CPU usage to go up, latency to increase, and pods to become overwhelmed? It's the **traffic load** hitting the application!

If we're simulating the environment, we also need to simulate the incoming traffic. This is where **Load Simulation** comes in.

Think of Load Simulation as a **digital traffic generator**. Instead of waiting for real users to visit our application (which might be slow or unpredictable, especially if we're trying to test extreme scaling scenarios), we create artificial demand. We generate fake "requests" hitting the application to see how it performs and how the autoscaling system (the HPA or our RL agent) reacts.

Load simulation is absolutely crucial for training and evaluating an RL agent in a controlled setting:

*   **Training:** The agent needs to experience a wide variety of load patterns (sudden spikes, gradual increases, steady load, drops) to learn how to respond effectively. Running these scenarios on a real system for millions of training steps would be impossible due to cost and time. Simulation makes it fast and cheap.
*   **Evaluation:** Once trained, we can test the agent against specific, repeatable load patterns to measure its performance objectively and compare it to other agents or traditional autoscalers.
*   **Safety:** You can test how the system behaves under extreme load without risking downtime or performance issues for real users.

In our project, Load Simulation is the component that creates this artificial demand, making the simulated [RL Environment](05_rl_environment_.md) dynamic and realistic.

## Two Ways to Simulate (or Generate) Load

The project uses different approaches depending on whether you're running a full simulation or testing against a real cluster:

1.  **Custom Python Simulators:** Used primarily within the `MicroK8sEnvSimulated` ([RL Environment](05_rl_environment_.md)) for fast, integrated simulation of load patterns. These are the `TrafficSimulator` and `HybridTrafficSimulator` classes.
2.  **`k6` Load Testing Tool:** A real-world open-source tool used to generate actual HTTP traffic against a *real* deployed application (like our Nginx service).

Let's look at the Python simulators first, as they are core to the training simulation environment.

## The Python Traffic Simulators (`TrafficSimulator`, `HybridTrafficSimulator`)

The files `agent/traffic_simulation.py` and `agent/hybird_simulation.py` contain Python classes that generate numerical representations of "load" over time steps. They don't send actual network requests; they just produce a number (e.g., representing "requests per second") that the `MockKubernetesAPI` uses to calculate simulated resource usage and latency.

### `TrafficSimulator`: Creating Basic Patterns

The `TrafficSimulator` (`agent/traffic_simulation.py`) is designed to create common load patterns you might see in a real application:

*   **Base Load:** A constant minimum level of traffic.
*   **Daily Variation:** A smooth, repeating pattern (like higher traffic during the day, lower at night).
*   **Random Spikes:** Sudden, short bursts of extra traffic.
*   **Scheduled Events:** Specific, planned increases in traffic at certain times (like a marketing campaign).
*   **Noise:** Small, random fluctuations to make it more realistic.

The simulator's main job is to provide a function, typically called `get_load(step)`, which returns the simulated load value for the current time `step`.

Here's a *simplified* look at how it might generate a load value:

```python
# Simplified concept from TrafficSimulator.get_load()

class TrafficSimulator:
    # ... init and other methods ...

    def get_load(self, step: int) -> float:
        # 1. Base Load + Daily Pattern
        daily_variation = self.base_load * self.daily_amplitude * np.sin(...) # Sinusoidal pattern
        load = self.base_load + daily_variation

        # 2. Add Random Spikes
        if self._is_random_spike_active(): # Check if a random spike is happening
             load += self._get_random_spike_effect()

        # 3. Add Scheduled Events
        for event in self.events:
             if event.start <= step < event.start + event.duration:
                 load *= event.intensity # Apply event multiplier

        # 4. Add Noise
        noise = self.rng.normal(0, self.base_load * 0.05)
        load += noise

        # Ensure minimum load
        final_load = max(self.min_load, load)

        return final_load
```

**Explanation:**

*   The simulator takes the current `step` (representing time) as input.
*   It calculates a base load that includes a daily cycle using a sine wave.
*   It checks if a random spike is currently active or should start and adds its effect.
*   It iterates through a list of pre-defined `events` (`self.events`) and applies any active event's intensity multiplier.
*   It adds a small amount of random `noise`.
*   Finally, it ensures the load doesn't drop below a minimum value.

This calculated `final_load` number is then used by the simulation environment.

You can define scheduled events when initializing the simulator or by calling the `add_event` method:

```python
# Simplified from agent/traffic_simulation.py or environment_simulated.py

simulator = TrafficSimulator(base_load=100, daily_amplitude=0.4)

# Add a specific event: traffic spike starts at step 500, lasts 100 steps, increases load by 5x
simulator.add_event(start=500, duration=100, intensity=5.0, event_type="marketing_push")

# The environment will then call simulator.get_load(step) in its step() method.
```

This allows you to set up specific, repeatable traffic scenarios for training or testing.

### `HybridTrafficSimulator`: Potentially More Complex Patterns

The `HybridTrafficSimulator` (`agent/hybird_simulation.py`) appears to offer similar functionality but might incorporate different types of events or more complex patterns (like weekly variations) or different simulation logic for event types (sudden spike, gradual increase, drastic drop as shown in the code). The core concept is the same: generate a numerical load value for a given time step. The specific implementation details might differ, offering alternative or richer load dynamics.

While the specific *reward function* logic in `MicroK8sEnvSimulated` refers to load mean and gradient from `self.load_history`, the underlying `MockKubernetesAPI` gets the current load directly via a function reference, which could point to either simulator's `get_load` method.

## Connecting Simulation to the Environment

How does this simulated load affect the simulated environment? The `MockKubernetesAPI` ([Kubernetes Interaction (API)](03_kubernetes_interaction___api__.md)) uses the simulated load value generated by the `TrafficSimulator` (or `HybridTrafficSimulator`) to calculate the simulated metrics (CPU, Memory, Latency, etc.).

Recall the `MockKubernetesAPI.get_cluster_state()` method from [Chapter 3: Kubernetes Interaction (API)](03_kubernetes_interaction___api__.md). It receives a reference to the traffic simulator's `get_load` function during its initialization.

Here's how the connection works:

```python
# Simplified from MockKubernetesAPI.__init__
class MockKubernetesAPI:
    def __init__(self, max_pods=10, traffic_simulator=None):
        # ... other init ...
        # Store the function provided by the traffic simulator
        self.traffic_simulator_func = traffic_simulator or self._default_traffic_simulator()
        self.current_step = 0 # Keep track of simulation step

# Simplified from MockKubernetesAPI.get_cluster_state()
    def get_cluster_state(self) -> Dict[str, Any]:
        self.current_step += 1

        # Get the simulated load value for this step
        current_load = self.traffic_simulator_func(self.current_step)

        # Calculate simulated metrics based on load and active pods
        effective_capacity = max(1, self.active_pods * self.pod_capacity) # pod_capacity is internal mock value
        load_ratio = current_load / effective_capacity

        # Higher load ratio -> higher simulated CPU, Memory, Latency
        cpu_util = min(1.0, 0.2 + 0.8 * load_ratio + ...) # Add some noise/variation
        memory_util = min(..., (current_load * 0.3) + ...)
        latency = max(0.05, self.base_latency + (load_ratio ** 2) + ...)

        # ... logic for autoscaling inside the mock API (if enabled for comparison) ...
        # ... simulate pod scaling effects (active_pods changing slowly) ...

        return {
            "pods": self.active_pods,
            "cpu": cpu_util,
            "memory": memory_util,
            "latency": latency,
            # ... other simulated metrics ...
        }

```

**Explanation:**

*   The `MockKubernetesAPI` stores the `get_load` function from the traffic simulator.
*   Every time the `get_cluster_state()` method is called by the [RL Environment](05_rl_environment_.md)'s `step()` method, the mock API increments its internal step counter.
*   It calls the stored `traffic_simulator_func` with the current step to get the simulated load value.
*   It then uses this `current_load` value, combined with the current number of `active_pods` (simulated pods that are ready to serve traffic), to calculate realistic-looking values for CPU utilization, memory usage, latency, etc. A higher load relative to capacity leads to higher resource usage and latency in the simulation.
*   These calculated metrics are returned as part of the simulated cluster state dictionary.

This state dictionary is what the [RL Environment](05_rl_environment_.md) uses to form the **observation** that is passed to the [RL Agent (DQN/PPO)](06_rl_agent__dqn_ppo__.md), and also to calculate the **reward**. The simulated load is thus the root cause driving the dynamics the agent needs to learn to manage.

## `k6`: Real-World Load Generation

While the Python simulators are for the simulated environment, when testing the RL agent or the HPA against a *real* MicroK8s cluster, you need to generate actual network traffic. This is where a tool like `k6` is used.

`k6` is an open-source load testing tool that lets you write test scripts (in JavaScript) to define how many virtual users should make requests, how often, and to what endpoint.

Our project includes an example of using `k6` by defining a Kubernetes `Job` resource ([Kubernetes Configuration](01_kubernetes_configuration_.md)) that runs a `k6` container.

Here's the Kubernetes `Job` definition (`deployments/k6-job.yaml`):

```yaml
# Simplified from deployments/k6-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: k6-loadtest # Name of the Kubernetes Job
spec:
  template:
    spec:
      containers:
      - name: k6 # Container running the k6 tool
        image: grafana/k6:latest # Use the official k6 image
        command: ["k6", "run", "/scripts/load-test.js"] # Command to run k6 with our script
        volumeMounts: # Mount the script ConfigMap
        - name: k6-script
          mountPath: /scripts
      restartPolicy: Never # Don't restart the job if it finishes
      volumes: # Define the volume that gets the script from a ConfigMap
      - name: k6-script
        configMap:
          name: k6-load-script # Name of the ConfigMap containing the k6 script
```

**Explanation:**

*   This defines a Kubernetes `Job` that creates a single Pod.
*   The Pod runs a container using the `grafana/k6` image.
*   The container's command tells `k6` to `run` a script located at `/scripts/load-test.js`.
*   The `volumeMounts` and `volumes` section shows that the script comes from a `ConfigMap` named `k6-load-script`. This `ConfigMap` would contain the actual JavaScript code for the `k6` test.

The `k6-load-script` ConfigMap (not shown in the provided snippets but implied by the `Job` definition) would contain JavaScript like this:

```javascript
// Simplified concept of load-test.js script
import http from 'k6/http';
import { sleep } from 'k6';

// Define the load test scenario
export let options = {
  stages: [
    { duration: '1m', target: 50 }, // Ramp up to 50 virtual users over 1 minute
    { duration: '5m', target: 50 }, // Stay at 50 virtual users for 5 minutes
    { duration: '1m', target: 0 },  // Ramp down to 0 users over 1 minute
  ],
};

export default function () {
  // This function runs repeatedly for each virtual user
  http.get('http://nginx-service.default.svc.cluster.local/'); // Make request to our Nginx Service
  sleep(1); // Wait 1 second between requests (can be randomized)
}
```

**Explanation:**

*   The `options` define the load profile: how many virtual users (`target`) are active over what `duration`. This script ramps up to 50 users, stays there, then ramps down.
*   The `default` function is the core test loop. Each virtual user executes this code.
*   `http.get(...)` makes an HTTP request to the target. Notice it uses the internal Kubernetes Service name (`nginx-service.default.svc.cluster.local`) to reach the Nginx pods.
*   `sleep(1)` makes the virtual user wait for 1 second before making the next request.

When you run `kubectl apply -f deployments/k6-job.yaml` (often orchestrated by the [Project Orchestration Scripts](04_project_orchestration_scripts_.md)), Kubernetes creates the `k6` Pod, it runs the script, and `k6` starts sending HTTP requests to your running Nginx application. This generates real load, which causes real CPU/Memory usage and latency, which is then collected by Prometheus ([Observability & Metrics](02_observability___metrics_.md)) and observed by the real `MicroK8sEnv` and the RL agent.

## Orchestration of Load Simulation

As mentioned in [Chapter 4: Project Orchestration Scripts](04_project_orchestration_scripts_.md), scripts like `run_simulation.sh` handle starting the RL agent. If the script is configured for simulation (`--simulate` flag), it will instantiate the `MicroK8sEnvSimulated` environment, which internally uses the `TrafficSimulator` or `HybridTrafficSimulator` and the `MockKubernetesAPI`.

If the script were configured to run against a real cluster (which isn't the primary focus of the provided `run_simulation.sh` but is the goal for real-world testing), it would instantiate the `MicroK8sEnv` (using the real `KubernetesAPI`) and potentially start the `k6` Job using `kubectl apply`.

The scripts ensure that the correct load generation mechanism is active depending on the environment mode (simulated vs. real).

## Visualizing Simulated Load and Response

Since the simulated environment (or the real environment with `k6`) produces metrics (either simulated or real), and the agent's callback logs these metrics to WandB ([Observability & Metrics](02_observability___metrics_.md)), you can use the WandB dashboards to visualize:

*   The simulated traffic load pattern over time.
*   The resulting simulated/real CPU utilization, memory, and latency.
*   The agent's scaling actions (change in pod count).
*   How well the agent's actions keep performance metrics within desired ranges under different load conditions.

The example scripts like `examples/traffic_simulation.py`, `examples/hybrid_autoscaling_comparison.py`, and `examples/simulator_comparison.py` also demonstrate how to use `matplotlib` to plot the output of the load simulators and simulated autoscaling responses, providing local visualization even without WandB.

Here's a simplified flow:

```mermaid
graph LR
    A[Python Traffic Simulator] --> B{Mock Kubernetes API};
    B --> C[Simulated RL Environment];
    C --> D[RL Agent];
    D --> E[WandB Logging];
    subgraph Real Cluster
        F[k6 Load Generator] --> G[Real Kubernetes Cluster (Nginx)];
        G --> H[Prometheus Metrics];
        H --> I[Real RL Environment (Observing)];
        I --> D;
    end
    C -- Sim State --> D;
    I -- Real State --> D;
    D -- Actions --> C;
    D -- Actions --> I;
```

This diagram shows that both the simulated and real paths involve a load generation component (`Traffic Simulator` or `k6`) that drives the state changes observed by the environment and the agent.

## Summary of Load Simulation Components

| Component Type        | Specific Tool/Class           | Purpose                                                                  | Used In                 | Links to Concepts                                       |
| :-------------------- | :---------------------------- | :----------------------------------------------------------------------- | :---------------------- | :------------------------------------------------------ |
| **Python Simulator**  | `TrafficSimulator`            | Generate numerical load values with daily, random, scheduled patterns.   | `MicroK8sEnvSimulated`  | [RL Environment](05_rl_environment_.md), [Kubernetes Interaction (API)](03_kubernetes_interaction___api__.md) (via Mock) |
| **Python Simulator**  | `HybridTrafficSimulator`      | Generate numerical load values with potentially more complex/varied patterns. | `MicroK8sEnvSimulated`  | [RL Environment](05_rl_environment_.md), [Kubernetes Interaction (API)](03_kubernetes_interaction___api__.md) (via Mock) |
| **Python Integration**| `MockKubernetesAPI`           | Uses simulated load to calculate simulated cluster metrics (CPU, Latency). | `MicroK8sEnvSimulated`  | [Kubernetes Interaction (API)](03_kubernetes_interaction___api__.md), [RL Environment](05_rl_environment_.md) |
| **Real-World Tool**   | `k6`                          | Generate actual HTTP traffic against a live application endpoint.        | Real Cluster Testing    | [Kubernetes Configuration](01_kubernetes_configuration_.md) (as K8s Job), [Observability & Metrics](02_observability___metrics_.md) (Prometheus scrapes real metrics) |
| **K8s Resource**      | `Job` (for k6)                | Defines how to run the `k6` load generation container in the cluster.    | Real Cluster Testing    | [Kubernetes Configuration](01_kubernetes_configuration_.md) |
| **K8s Resource**      | `ConfigMap` (for k6 script)   | Stores the `k6` test script for the `Job` to use.                        | Real Cluster Testing    | [Kubernetes Configuration](01_kubernetes_configuration_.md) |

Load simulation, whether using Python classes for fast training simulations or a tool like `k6` for realistic testing, is fundamental to generating the dynamic conditions needed to train and evaluate the autoscaling logic developed in this project.

## Conclusion

In this chapter, you learned that **Load Simulation** is the process of generating artificial traffic or demand to test the autoscaling system. You saw how the project uses custom Python simulators (`TrafficSimulator`, `HybridTrafficSimulator`) within the `MicroK8sEnvSimulated` and `MockKubernetesAPI` to create dynamic scenarios for fast RL agent training, and how these simulators provide numerical load values that drive the calculation of simulated metrics. You also learned about `k6` as a tool for generating real HTTP traffic when testing against a live cluster, typically run as a Kubernetes `Job`. Load simulation is the engine that creates the varied conditions necessary for the agent to learn robust autoscaling behavior.

This concludes our introductory chapters on the core components of the `microk8s-autoscaling` project. You now have a foundational understanding of the Kubernetes environment, how we monitor it, how our code interacts with it, how we orchestrate the project, the world where the RL agent lives, the agent's learning process, and finally, how we simulate the critical load that drives the need for autoscaling.

You are now equipped to dive deeper into the project's code, understand how these pieces fit together in detail, run simulations, and potentially contribute to extending its capabilities!

---

<sub><sup>Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge).</sup></sub> <sub><sup>**References**: [[1]](https://github.com/rohmatmret/microk8s-autoscaling/blob/ff93765af606c718dc57fc58e4284e10f9ff1560/agent/environment_simulated.py), [[2]](https://github.com/rohmatmret/microk8s-autoscaling/blob/ff93765af606c718dc57fc58e4284e10f9ff1560/agent/hybird_simulation.py), [[3]](https://github.com/rohmatmret/microk8s-autoscaling/blob/ff93765af606c718dc57fc58e4284e10f9ff1560/agent/mock_kubernetes_api.py), [[4]](https://github.com/rohmatmret/microk8s-autoscaling/blob/ff93765af606c718dc57fc58e4284e10f9ff1560/agent/traffic_simulation.py), [[5]](https://github.com/rohmatmret/microk8s-autoscaling/blob/ff93765af606c718dc57fc58e4284e10f9ff1560/deployments/k6-job.yaml), [[6]](https://github.com/rohmatmret/microk8s-autoscaling/blob/ff93765af606c718dc57fc58e4284e10f9ff1560/examples/autoscaling_simulation.py), [[7]](https://github.com/rohmatmret/microk8s-autoscaling/blob/ff93765af606c718dc57fc58e4284e10f9ff1560/examples/hybrid_autoscaling_comparison.py), [[8]](https://github.com/rohmatmret/microk8s-autoscaling/blob/ff93765af606c718dc57fc58e4284e10f9ff1560/examples/hybrid_traffic_simulation.py), [[9]](https://github.com/rohmatmret/microk8s-autoscaling/blob/ff93765af606c718dc57fc58e4284e10f9ff1560/examples/simulator_comparison.py)</sup></sub>