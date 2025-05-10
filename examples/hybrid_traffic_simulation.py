"""Example script demonstrating hybrid traffic simulation with visualization."""

import numpy as np
import matplotlib.pyplot as plt
from agent.traffic_simulation import HybridTrafficSimulator

def simulate_hybrid_traffic():
    """Simulate and visualize hybrid traffic patterns."""
    # Initialize simulator with realistic settings
    simulator = HybridTrafficSimulator(
        base_load=100,          # Base requests per second
        seed=42,                # For reproducibility
        event_frequency=0.005,  # 0.5% chance of event per step
        min_intensity=5,        # Minimum event intensity
        max_intensity=50,       # Maximum event intensity
        min_duration=10,        # Minimum event duration
        max_duration=200        # Maximum event duration
    )
    
    # Simulate for 3 days (4320 steps)
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
    plt.plot(loads, label='Traffic Load', color='blue')
    plt.title('Hybrid Traffic Pattern Simulation')
    plt.xlabel('Time Steps')
    plt.ylabel('Load (req/s)')
    plt.legend()
    
    # Plot events
    plt.subplot(2, 1, 2)
    event_types = set(events)
    colors = {'sudden_spike': 'red', 'gradual_increase': 'green', 
              'drastic_drop': 'orange', 'none': 'gray'}
    
    for event_type in event_types:
        event_steps = [i for i, e in enumerate(events) if e == event_type]
        plt.scatter(event_steps, [1] * len(event_steps), 
                   label=event_type, color=colors[event_type], alpha=0.5)
    
    plt.title('Event Distribution')
    plt.xlabel('Time Steps')
    plt.ylabel('Event Type')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hybrid_traffic_simulation.png')
    plt.close()
    
    # Print statistics
    print("\nSimulation Statistics:")
    print(f"Total Steps: {steps}")
    print(f"Average Load: {np.mean(loads):.2f} req/s")
    print(f"Max Load: {max(loads):.2f} req/s")
    print(f"Min Load: {min(loads):.2f} req/s")
    
    # Event statistics
    event_counts = {event_type: events.count(event_type) for event_type in event_types}
    print("\nEvent Statistics:")
    for event_type, count in event_counts.items():
        print(f"{event_type}: {count} occurrences ({count/steps*100:.1f}%)")

if __name__ == "__main__":
    simulate_hybrid_traffic() 