"""Hybrid traffic simulation module for generating realistic workload patterns.

This module provides traffic simulation capabilities for testing and evaluating
autoscaling algorithms with various traffic patterns including daily variations,
special events, and random noise.
"""

import numpy as np
from typing import List, Tuple

class HybridTrafficSimulator:
    """Simulates realistic traffic patterns with sudden spikes, gradual increases, drastic drops, and hybrid patterns."""
    
    def __init__(self, base_load=100, seed=None, event_frequency=0.005, min_intensity=5, max_intensity=50, min_duration=10, max_duration=200):
        """
        Initialize the traffic simulator with configurable parameters.
        
        Args:
            base_load (float): Base traffic load.
            seed (int): Random seed for reproducibility.
            event_frequency (float): Probability of an event occurring per step.
            min_intensity (float): Minimum intensity multiplier for events.
            max_intensity (float): Maximum intensity multiplier for events.
            min_duration (int): Minimum duration of events in steps.
            max_duration (int): Maximum duration of events in steps.
        """
        self.rng = np.random.default_rng(seed)
        self.base_load = base_load
        self.event_frequency = event_frequency
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.current_load = base_load
        self.active_event = None
        self.event_end_step = 0
        self.load_history = []  # Track load for visualization
        self.event_history = []  # Track event types for visualization
        
        # Possible event types
        self.event_types = ["sudden_spike", "gradual_increase", "drastic_drop"]

    def get_load(self, step):
        """Generate traffic load based on daily variation, events, and noise."""
        # Daily variation: sinusoidal pattern (day/night cycle)
        daily_variation = self.base_load * 0.3 * np.sin(2 * np.pi * step / 1440)
        load = self.base_load + daily_variation

        # Handle active event
        event_type = "none"
        if self.active_event and step < self.event_end_step:
            event = self.active_event
            event_type = event["type"]
            if event["type"] == "sudden_spike":
                load *= event["intensity"]
            elif event["type"] == "gradual_increase":
                progress = (step - event["start"]) / event["duration"]
                load *= (1 + event["intensity"] * progress)
            elif event["type"] == "drastic_drop":
                load *= event["intensity"]
        else:
            # Check for new event
            if self.rng.random() < self.event_frequency:
                self.active_event = {
                    "type": self.rng.choice(self.event_types),
                    "start": step,
                    "duration": self.rng.integers(self.min_duration, self.max_duration),
                    "intensity": self.rng.uniform(self.min_intensity, self.max_intensity)
                    if self.rng.choice(self.event_types) in ["sudden_spike", "gradual_increase"]
                    else self.rng.uniform(0.05, 0.2)
                }
                self.event_end_step = step + self.active_event["duration"]
                event_type = self.active_event["type"]

        # Add random noise for realism
        noise = self.rng.normal(0, self.base_load * 0.05)
        self.current_load = max(10, load + noise)
        
        # Track load and event for visualization
        self.load_history.append(self.current_load)
        self.event_history.append(event_type)
        if len(self.load_history) > 1000:  # Limit history to save memory
            if self.load_history:  # Check if list is not empty
                self.load_history.pop(0)
            if self.event_history:  # Check if list is not empty
                self.event_history.pop(0)

        return self.current_load

    def get_load_history(self, window=50):
        """Return average load over the last window steps."""
        if not self.load_history:
            return 0.0
        return np.mean(self.load_history[-window:])

    def get_visualization_data(self, max_steps: int = 1000) -> Tuple[List[int], List[float], List[str]]:
        """Get visualization data for plotting."""
        # Ensure we have data to return
        if not self.load_history or not self.event_history:
            return [], [], []
            
        # Get the last max_steps entries
        steps = list(range(len(self.load_history)))
        loads = self.load_history[-max_steps:] if len(self.load_history) > max_steps else self.load_history
        events = self.event_history[-max_steps:] if len(self.event_history) > max_steps else self.event_history
        
        # Ensure all lists have the same length
        min_length = min(len(steps), len(loads), len(events))
        return steps[:min_length], loads[:min_length], events[:min_length]