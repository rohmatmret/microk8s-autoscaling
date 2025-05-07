import numpy as np

class TrafficSimulator:
    def __init__(self, base_load=100, max_spike=30, seed=None):
        self.rng = np.random.default_rng(seed)
        self.base_load = base_load
        self.max_spike = max_spike
        self.current_load = base_load
        self.spike_duration = 0
        
        # Event schedule (simulated marketplace events)
        self.events = [
            {"start": 1000, "duration": 300, "intensity": 25},
            {"start": 1500, "duration": 300, "intensity": 20},
            {"start": 2000, "duration": 300, "intensity": 15},
            {"start": 2500, "duration": 600, "intensity": 30}
        ]

    def get_load(self, step):
        """Generate realistic traffic pattern with base + spikes + events"""
        # Base load with daily pattern
        daily_variation = self.base_load * 0.3 * np.sin(2 * np.pi * step / 1440)
        
        # Random spikes (Poisson process)
        spike_prob = 0.005
        spike =0
        if self.rng.random() < spike_prob:
            spike = self.base_load * self.rng.uniform(2, 5)
            duration = self.rng.integers(10, 30)
            self.spike_duration = duration
        
        # Handle scheduled events
        event_multiplier = 1
        for event in self.events:
            if event["start"] <= step < event["start"] + event["duration"]:
                event_multiplier = event["intensity"]
                break
        
        # Calculate final load
        if self.spike_duration > 0:
            spike_effect = spike * (self.spike_duration / duration)
            self.spike_duration -= 1
        else:
            spike_effect = 0
            
        return max(10, (
            (self.base_load + daily_variation) * event_multiplier +
            spike_effect * self.rng.uniform(0.8, 1.2)
        ))