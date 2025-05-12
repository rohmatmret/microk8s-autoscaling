"""Test script for traffic patterns and their impact on autoscaling simulation."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from agent.traffic_simulation import TrafficSimulator

def test_basic_traffic_patterns():
    """Test basic traffic patterns and their characteristics."""
    # Initialize simulator with moderate settings
    simulator = TrafficSimulator(
        base_load=100,
        max_spike=30,
        daily_amplitude=0.3,
        spike_probability=0.005,
        min_spike_duration=10,
        max_spike_duration=30
    )
    
    # Simulate for 7 days (10080 steps)
    steps = 10080
    loads = []
    events = []
    
    for step in range(steps):
        load = simulator.get_load(step)
        loads.append(load)
        events.append(simulator.event_history[-1] if simulator.event_history else "none")
    
    # Basic validation
    assert len(loads) == steps
    assert min(loads) >= simulator.min_load
    assert any(load > simulator.base_load * 2 for load in loads)  # Verify spikes exist
    assert any(load < simulator.base_load * 0.8 for load in loads)  # Verify daily patterns

def test_autoscaling_scenarios():
    """Test different autoscaling scenarios with traffic patterns."""
    # Initialize simulator with more extreme settings
    simulator = TrafficSimulator(
        base_load=100,
        max_spike=50,  # Higher spikes
        daily_amplitude=0.5,  # More pronounced daily pattern
        spike_probability=0.01,  # More frequent spikes
        min_spike_duration=5,
        max_spike_duration=60  # Longer spikes
    )
    
    # Add specific events for testing
    simulator.add_event(start=1000, duration=300, intensity=40, event_type="burst")
    simulator.add_event(start=5000, duration=1000, intensity=20, event_type="periodic")
    
    # Simulate and collect metrics
    steps = 10080  # 7 days
    loads = []
    cpu_utils = []
    pod_counts = []
    
    # Simulate autoscaling behavior
    current_pods = 1
    max_pods = 10
    target_cpu = 0.7
    
    for step in range(steps):
        # Get current load
        load = simulator.get_load(step)
        loads.append(load)
        
        # Simulate CPU utilization based on load and pods
        cpu_util = min(1.0, load / (current_pods * 100))
        cpu_utils.append(cpu_util)
        
        # Simple autoscaling logic
        if cpu_util > target_cpu and current_pods < max_pods:
            current_pods += 1
        elif cpu_util < target_cpu * 0.5 and current_pods > 1:
            current_pods -= 1
            
        pod_counts.append(current_pods)
    
    # Validate autoscaling behavior
    assert max(pod_counts) <= max_pods
    assert min(pod_counts) >= 1
    assert any(pod_counts[i] != pod_counts[i-1] for i in range(1, len(pod_counts)))  # Verify scaling occurred

def visualize_traffic_patterns():
    """Visualize traffic patterns and their impact on autoscaling."""
    simulator = TrafficSimulator(
        base_load=100,
        max_spike=30,
        daily_amplitude=0.3,
        spike_probability=0.005
    )
    
    # Simulate for 3 days
    steps = 4320
    loads = []
    events = []
    
    for step in range(steps):
        load = simulator.get_load(step)
        loads.append(load)
        events.append(simulator.event_history[-1] if simulator.event_history else "none")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot load
    plt.subplot(2, 1, 1)
    plt.plot(loads, label='Traffic Load')
    plt.title('Traffic Pattern Simulation')
    plt.xlabel('Time Steps')
    plt.ylabel('Load')
    plt.legend()
    
    # Plot events
    plt.subplot(2, 1, 2)
    event_types = set(events)
    for event_type in event_types:
        event_steps = [i for i, e in enumerate(events) if e == event_type]
        plt.scatter(event_steps, [1] * len(event_steps), label=event_type, alpha=0.5)
    plt.title('Event Distribution')
    plt.xlabel('Time Steps')
    plt.ylabel('Event Type')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('traffic_patterns.png')
    plt.close()

def test_traffic_characteristics():
    """Test specific characteristics of the traffic patterns."""
    simulator = TrafficSimulator(
        base_load=100,
        max_spike=30,
        daily_amplitude=0.3
    )
    
    # Test daily pattern
    daily_loads = [simulator.get_load(step) for step in range(1440)]
    assert abs(max(daily_loads) - min(daily_loads)) > simulator.base_load * 0.2  # Verify daily variation
    
    # Test weekly pattern
    weekly_loads = [simulator.get_load(step) for step in range(10080)]
    weekday_avg = np.mean(weekly_loads[:7200])  # First 5 days
    weekend_avg = np.mean(weekly_loads[7200:])  # Last 2 days
    assert weekday_avg > weekend_avg  # Verify weekly pattern
    
    # Test spike behavior
    spike_loads = []
    for _ in range(1000):
        load = simulator.get_load(0)  # Reset step to increase spike probability
        spike_loads.append(load)
    assert max(spike_loads) > simulator.base_load * 2  # Verify spikes occur

if __name__ == "__main__":
    # Run visualization
    visualize_traffic_patterns()
    
    # Run tests
    test_basic_traffic_patterns()
    test_autoscaling_scenarios()
    test_traffic_characteristics()
    
    print("All tests passed!") 