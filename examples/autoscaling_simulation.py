"""Example script demonstrating traffic simulation with autoscaling."""

import numpy as np
import matplotlib.pyplot as plt
from agent.traffic_simulation import TrafficSimulator

def simulate_autoscaling():
    """Simulate autoscaling behavior with realistic traffic patterns."""
    # Initialize traffic simulator with realistic settings
    simulator = TrafficSimulator(
        base_load=100,          # Base requests per second
        max_spike=50,           # Maximum spike intensity
        daily_amplitude=0.3,    # Daily variation amplitude
        spike_probability=0.005, # Probability of random spikes
        min_spike_duration=10,  # Minimum spike duration
        max_spike_duration=60   # Maximum spike duration
    )
    
    # Add some scheduled events
    simulator.add_event(start=1000, duration=300, intensity=40, event_type="burst")
    simulator.add_event(start=5000, duration=1000, intensity=20, event_type="periodic")
    
    # Simulation parameters
    steps = 10080  # 7 days
    current_pods = 1
    max_pods = 10
    min_pods = 1
    target_cpu = 0.7
    pod_capacity = 100  # Requests per pod
    
    # Metrics collection
    loads = []
    cpu_utils = []
    pod_counts = []
    scaling_events = []
    
    # Run simulation
    for step in range(steps):
        # Get current load
        load = simulator.get_load(step)
        loads.append(load)
        
        # Calculate CPU utilization
        cpu_util = min(1.0, load / (current_pods * pod_capacity))
        cpu_utils.append(cpu_util)
        
        # Autoscaling logic
        old_pods = current_pods
        if cpu_util > target_cpu and current_pods < max_pods:
            current_pods += 1
            scaling_events.append(("scale_up", step))
        elif cpu_util < target_cpu * 0.5 and current_pods > min_pods:
            current_pods -= 1
            scaling_events.append(("scale_down", step))
            
        pod_counts.append(current_pods)
    
    # Visualization
    plt.figure(figsize=(15, 12))
    
    # Plot traffic load
    plt.subplot(3, 1, 1)
    plt.plot(loads, label='Traffic Load', color='blue')
    plt.title('Traffic Pattern and Autoscaling Simulation')
    plt.xlabel('Time Steps')
    plt.ylabel('Load (req/s)')
    plt.legend()
    
    # Plot CPU utilization
    plt.subplot(3, 1, 2)
    plt.plot(cpu_utils, label='CPU Utilization', color='green')
    plt.axhline(y=target_cpu, color='r', linestyle='--', label='Target CPU')
    plt.axhline(y=target_cpu * 0.5, color='r', linestyle=':', label='Scale Down Threshold')
    plt.xlabel('Time Steps')
    plt.ylabel('CPU Utilization')
    plt.legend()
    
    # Plot pod count
    plt.subplot(3, 1, 3)
    plt.plot(pod_counts, label='Pod Count', color='purple')
    for event_type, step in scaling_events:
        color = 'g' if event_type == 'scale_up' else 'r'
        plt.scatter(step, pod_counts[step], color=color, marker='^' if event_type == 'scale_up' else 'v')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Pods')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('autoscaling_simulation.png')
    plt.close()
    
    # Print statistics
    print("\nSimulation Statistics:")
    print(f"Total Steps: {steps}")
    print(f"Average Load: {np.mean(loads):.2f} req/s")
    print(f"Max Load: {max(loads):.2f} req/s")
    print(f"Average CPU: {np.mean(cpu_utils):.2f}")
    print(f"Average Pods: {np.mean(pod_counts):.2f}")
    print(f"Scaling Events: {len(scaling_events)}")
    print(f"Scale Up Events: {sum(1 for e, _ in scaling_events if e == 'scale_up')}")
    print(f"Scale Down Events: {sum(1 for e, _ in scaling_events if e == 'scale_down')}")

if __name__ == "__main__":
    simulate_autoscaling() 