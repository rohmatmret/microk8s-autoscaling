"""Example script demonstrating the comparison between hybrid traffic simulation and autoscaling behavior."""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.traffic_simulation import TrafficSimulator
from agent.hybird_simulation import HybridTrafficSimulator

def compare_hybrid_autoscaling():
    """Compare and visualize hybrid traffic patterns with autoscaling behavior."""
    # Initialize both simulators
    hybrid_sim = HybridTrafficSimulator(
        base_load=100,
        event_frequency=0.005,
        min_intensity=5,
        max_intensity=50,
        min_duration=10,
        max_duration=200
    )
    
    traffic_sim = TrafficSimulator(
        base_load=100,
        max_spike=30,
        daily_amplitude=0.3,
        spike_probability=0.005,
        min_spike_duration=10,
        max_spike_duration=30
    )
    
    # Add some scheduled events to TrafficSimulator
    traffic_sim.add_event(start=1000, duration=300, intensity=2.0, event_type="scheduled")
    traffic_sim.add_event(start=2000, duration=200, intensity=1.5, event_type="periodic")
    
    # Simulation parameters
    steps = 4320  # 3 days
    current_pods = 1
    max_pods = 10
    min_pods = 1
    target_cpu = 0.7
    pod_capacity = 100  # Requests per pod
    
    # Metrics collection
    hybrid_loads = []
    traffic_loads = []
    hybrid_events = []
    traffic_events = []
    cpu_utils = []
    pod_counts = []
    scaling_events = []
    
    # Run simulation
    for step in range(steps):
        # Get loads from both simulators
        hybrid_load = hybrid_sim.get_load(step)
        traffic_load = traffic_sim.get_load(step)
        
        hybrid_loads.append(hybrid_load)
        traffic_loads.append(traffic_load)
        
        # Track events
        hybrid_events.append(hybrid_sim.event_history[-1] if hybrid_sim.event_history else "none")
        traffic_events.append(traffic_sim.event_history[-1] if traffic_sim.event_history else "none")
        
        # Calculate CPU utilization based on hybrid load
        cpu_util = min(1.0, hybrid_load / (current_pods * pod_capacity))
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
    
    # Create visualization
    plt.figure(figsize=(15, 15))
    
    # Plot loads comparison
    plt.subplot(4, 1, 1)
    plt.plot(hybrid_loads, label='Hybrid Traffic', color='blue', alpha=0.7)
    plt.plot(traffic_loads, label='Standard Traffic', color='red', alpha=0.7)
    plt.title('Traffic Load Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Load (req/s)')
    plt.legend()
    
    # Plot CPU utilization
    plt.subplot(4, 1, 2)
    plt.plot(cpu_utils, label='CPU Utilization', color='green')
    plt.axhline(y=target_cpu, color='r', linestyle='--', label='Target CPU')
    plt.axhline(y=target_cpu * 0.5, color='r', linestyle=':', label='Scale Down Threshold')
    plt.title('CPU Utilization')
    plt.xlabel('Time Steps')
    plt.ylabel('CPU Utilization')
    plt.legend()
    
    # Plot pod count
    plt.subplot(4, 1, 3)
    plt.plot(pod_counts, label='Pod Count', color='purple')
    for event_type, step in scaling_events:
        color = 'g' if event_type == 'scale_up' else 'r'
        plt.scatter(step, pod_counts[step], color=color, marker='^' if event_type == 'scale_up' else 'v')
    plt.title('Pod Scaling')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Pods')
    plt.legend()
    
    # Plot events
    plt.subplot(4, 1, 4)
    event_types = set(hybrid_events)
    colors = {
        'sudden_spike': 'red',
        'gradual_increase': 'green',
        'drastic_drop': 'orange',
        'none': 'gray'
    }
    
    for event_type in event_types:
        event_steps = [i for i, e in enumerate(hybrid_events) if e == event_type]
        plt.scatter(event_steps, [1] * len(event_steps),
                   label=event_type, color=colors.get(event_type, 'gray'),
                   alpha=0.5)
    plt.title('Event Distribution')
    plt.xlabel('Time Steps')
    plt.ylabel('Event Type')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hybrid_autoscaling_comparison.png')
    plt.close()
    
    # Print statistics
    print("\nSimulation Statistics:")
    print("\nHybrid Traffic:")
    print(f"Average Load: {np.mean(hybrid_loads):.2f} req/s")
    print(f"Max Load: {max(hybrid_loads):.2f} req/s")
    print(f"Min Load: {min(hybrid_loads):.2f} req/s")
    print(f"Load Std Dev: {np.std(hybrid_loads):.2f}")
    
    print("\nStandard Traffic:")
    print(f"Average Load: {np.mean(traffic_loads):.2f} req/s")
    print(f"Max Load: {max(traffic_loads):.2f} req/s")
    print(f"Min Load: {min(traffic_loads):.2f} req/s")
    print(f"Load Std Dev: {np.std(traffic_loads):.2f}")
    
    print("\nAutoscaling Metrics:")
    print(f"Average CPU: {np.mean(cpu_utils):.2f}")
    print(f"Average Pods: {np.mean(pod_counts):.2f}")
    print(f"Scaling Events: {len(scaling_events)}")
    print(f"Scale Up Events: {sum(1 for e, _ in scaling_events if e == 'scale_up')}")
    print(f"Scale Down Events: {sum(1 for e, _ in scaling_events if e == 'scale_down')}")
    
    # Event statistics
    hybrid_event_counts = {event_type: hybrid_events.count(event_type) 
                          for event_type in event_types}
    print("\nHybrid Event Statistics:")
    for event_type, count in hybrid_event_counts.items():
        print(f"{event_type}: {count} occurrences ({count/steps*100:.1f}%)")

if __name__ == "__main__":
    compare_hybrid_autoscaling() 