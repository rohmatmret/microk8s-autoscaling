"""Example script demonstrating the differences between TrafficSimulator and HybridTrafficSimulator."""

import numpy as np
import matplotlib.pyplot as plt
from agent.traffic_simulation import TrafficSimulator, HybridTrafficSimulator

def compare_simulators():
    """Compare and visualize the differences between both traffic simulators."""
    # Initialize both simulators with similar base settings
    traffic_sim = TrafficSimulator(
        base_load=100,
        max_spike=30,
        daily_amplitude=0.3,
        spike_probability=0.005,
        min_spike_duration=10,
        max_spike_duration=30
    )
    
    hybrid_sim = HybridTrafficSimulator(
        base_load=100,
        event_frequency=0.005,
        min_intensity=5,
        max_intensity=50,
        min_duration=10,
        max_duration=200
    )
    
    # Add some scheduled events to TrafficSimulator
    traffic_sim.add_event(start=1000, duration=300, intensity=2.0, event_type="scheduled")
    traffic_sim.add_event(start=2000, duration=200, intensity=1.5, event_type="periodic")
    
    # Simulate for 3 days (4320 steps)
    steps = 4320
    traffic_loads = []
    hybrid_loads = []
    traffic_events = []
    hybrid_events = []
    
    for step in range(steps):
        # Get loads from both simulators
        traffic_load = traffic_sim.get_load(step)
        hybrid_load = hybrid_sim.get_load(step)
        
        traffic_loads.append(traffic_load)
        hybrid_loads.append(hybrid_load)
        
        # Track events
        traffic_events.append(traffic_sim.event_history[-1] if traffic_sim.event_history else "none")
        hybrid_events.append(hybrid_sim.event_history[-1] if hybrid_sim.event_history else "none")
    
    # Create visualization
    plt.figure(figsize=(15, 15))
    
    # Plot loads comparison
    plt.subplot(3, 1, 1)
    plt.plot(traffic_loads, label='TrafficSimulator', color='blue', alpha=0.7)
    plt.plot(hybrid_loads, label='HybridTrafficSimulator', color='red', alpha=0.7)
    plt.title('Load Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Load (req/s)')
    plt.legend()
    
    # Plot TrafficSimulator events
    plt.subplot(3, 1, 2)
    traffic_event_types = set(traffic_events)
    traffic_colors = {
        'scheduled': 'green',
        'spike': 'red',
        'burst': 'orange',
        'periodic': 'purple',
        'none': 'gray'
    }
    
    for event_type in traffic_event_types:
        event_steps = [i for i, e in enumerate(traffic_events) if e == event_type]
        plt.scatter(event_steps, [1] * len(event_steps),
                   label=event_type, color=traffic_colors.get(event_type, 'gray'),
                   alpha=0.5)
    plt.title('TrafficSimulator Events')
    plt.xlabel('Time Steps')
    plt.ylabel('Event Type')
    plt.legend()
    
    # Plot HybridTrafficSimulator events
    plt.subplot(3, 1, 3)
    hybrid_event_types = set(hybrid_events)
    hybrid_colors = {
        'sudden_spike': 'red',
        'gradual_increase': 'green',
        'drastic_drop': 'orange',
        'none': 'gray'
    }
    
    for event_type in hybrid_event_types:
        event_steps = [i for i, e in enumerate(hybrid_events) if e == event_type]
        plt.scatter(event_steps, [1] * len(event_steps),
                   label=event_type, color=hybrid_colors.get(event_type, 'gray'),
                   alpha=0.5)
    plt.title('HybridTrafficSimulator Events')
    plt.xlabel('Time Steps')
    plt.ylabel('Event Type')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('simulator_comparison.png')
    plt.close()
    
    # Print comparison statistics
    print("\nSimulator Comparison Statistics:")
    print("\nTrafficSimulator:")
    print(f"Average Load: {np.mean(traffic_loads):.2f} req/s")
    print(f"Max Load: {max(traffic_loads):.2f} req/s")
    print(f"Min Load: {min(traffic_loads):.2f} req/s")
    print(f"Load Std Dev: {np.std(traffic_loads):.2f}")
    
    traffic_event_counts = {event_type: traffic_events.count(event_type) 
                          for event_type in traffic_event_types}
    print("\nTrafficSimulator Events:")
    for event_type, count in traffic_event_counts.items():
        print(f"{event_type}: {count} occurrences ({count/steps*100:.1f}%)")
    
    print("\nHybridTrafficSimulator:")
    print(f"Average Load: {np.mean(hybrid_loads):.2f} req/s")
    print(f"Max Load: {max(hybrid_loads):.2f} req/s")
    print(f"Min Load: {min(hybrid_loads):.2f} req/s")
    print(f"Load Std Dev: {np.std(hybrid_loads):.2f}")
    
    hybrid_event_counts = {event_type: hybrid_events.count(event_type) 
                          for event_type in hybrid_event_types}
    print("\nHybridTrafficSimulator Events:")
    for event_type, count in hybrid_event_counts.items():
        print(f"{event_type}: {count} occurrences ({count/steps*100:.1f}%)")
    
    # Calculate and print key differences
    print("\nKey Differences:")
    print(f"Average Load Difference: {abs(np.mean(traffic_loads) - np.mean(hybrid_loads)):.2f} req/s")
    print(f"Max Load Difference: {abs(max(traffic_loads) - max(hybrid_loads)):.2f} req/s")
    print(f"Load Variability (TrafficSimulator): {np.std(traffic_loads)/np.mean(traffic_loads):.2f}")
    print(f"Load Variability (HybridTrafficSimulator): {np.std(hybrid_loads)/np.mean(hybrid_loads):.2f}")

if __name__ == "__main__":
    compare_simulators() 