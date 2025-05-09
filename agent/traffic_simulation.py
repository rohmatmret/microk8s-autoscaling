#agent/traffic_simulation.py
import numpy as np

class TrafficSimulator:
    def __init__(self, base_load=100, max_spike=30, seed=None):
        self.rng = np.random.default_rng(seed)
        self.base_load = base_load
        self.max_spike = max_spike
        self.current_load = base_load
        self.spike_duration = 0
        self.current_spike = 0  
        
        # Event schedule (simulated marketplace events)
        self.events = [
            {"start": 1000, "duration": 300, "intensity": 25},
            {"start": 1500, "duration": 300, "intensity": 20},
            {"start": 2000, "duration": 300, "intensity": 15},
            {"start": 2500, "duration": 600, "intensity": 30}
        ]

    def get_load(self, step):
        """Generate realistic traffic pattern with base + spikes + events and noise."""
        # Daily pattern: sinusoidal variation (e.g., day/night cycle)
        daily_variation = self.base_load * 0.3 * np.sin(2 * np.pi * step / 1440)

        # Scheduled events: simulate marketplace surges
        event_multiplier = 1
        for event in self.events:
            if event["start"] <= step < event["start"] + event["duration"]:
                event_multiplier = event["intensity"]
                break

        # Random spikes: rare, short, high spikes (flash sales, outages, etc.)
        spike_prob = 0.005
        if self.spike_duration > 0:
            # Continue current spike
            spike_effect = self.current_spike * self.rng.uniform(0.8, 1.2)
            self.spike_duration -= 1
        elif self.rng.random() < spike_prob:
            # Start new spike
            self.current_spike = self.base_load * self.rng.uniform(2, 5)
            self.spike_duration = self.rng.integers(10, 30)
            spike_effect = self.current_spike * self.rng.uniform(0.8, 1.2)
        else:
            spike_effect = 0
            self.current_spike = 0

        # Additive random noise for realism
        noise = self.rng.normal(0, self.base_load * 0.05)

        # Final load calculation
        load = (self.base_load + daily_variation) * event_multiplier + spike_effect + noise

        # Clamp to minimum realistic value
        return max(10, load)