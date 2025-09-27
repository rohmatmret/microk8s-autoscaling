"""Hybrid traffic simulation module for generating realistic workload patterns.

This module provides traffic simulation capabilities for testing and evaluating
autoscaling algorithms with various traffic patterns including daily variations,
special events, and random noise.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional

class HybridTrafficSimulator:
    """Simulates realistic traffic patterns with sudden spikes, gradual increases, drastic drops, and hybrid patterns."""
    
    def __init__(self, base_load: float = 100, seed: Optional[int] = None, 
                 event_frequency: float = 0.005, min_intensity: float = 5, 
                 max_intensity: float = 50, min_duration: int = 10, 
                 max_duration: int = 200) -> None:
        """
        Initialize the traffic simulator with configurable parameters.
        
        Args:
            base_load: Base traffic load.
            seed: Random seed for reproducibility.
            event_frequency: Probability of an event occurring per step.
            min_intensity: Minimum intensity multiplier for events.
            max_intensity: Maximum intensity multiplier for events.
            min_duration: Minimum duration of events in steps.
            max_duration: Maximum duration of events in steps.
        """
        self.rng = np.random.default_rng(seed)
        self.base_load = base_load
        self.event_frequency = event_frequency
        # Ensure min_intensity < max_intensity to avoid "high - low < 0" error
        if min_intensity >= max_intensity:
            self.min_intensity = max_intensity * 0.3
            self.max_intensity = max_intensity
        else:
            self.min_intensity = min_intensity
            self.max_intensity = max_intensity
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.current_load = base_load
        self.active_event: Optional[Dict] = None
        self.event_end_step = 0
        self.load_history: List[float] = []  # Track load for visualization
        self.event_history: List[str] = []  # Track event types for visualization
        
        # Possible event types
        self.event_types = ["sudden_spike", "gradual_increase", "drastic_drop"]

    def get_load(self, step: int) -> float:
        """
        Generate traffic load based on daily variation, events, and noise.
        
        Args:
            step: Current simulation step.
            
        Returns:
            float: Current traffic load.
        """
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
                duration = max(1, event["duration"])  # Prevent division by zero
                progress = (step - event["start"]) / duration
                load *= (1 + event["intensity"] * progress)
            elif event["type"] == "drastic_drop":
                load *= event["intensity"]
        else:
            # Check for new event
            if self.rng.random() < self.event_frequency:
                event_type = self.rng.choice(self.event_types)
                duration = self.rng.integers(self.min_duration, self.max_duration)
                intensity = (self.rng.uniform(self.min_intensity, self.max_intensity)
                           if event_type in ["sudden_spike", "gradual_increase"]
                           else self.rng.uniform(0.05, 0.2))
                
                self.active_event = {
                    "type": event_type,
                    "start": step,
                    "duration": duration,
                    "intensity": intensity
                }
                self.event_end_step = step + duration

        # Add random noise for realism
        noise = self.rng.normal(0, self.base_load * 0.05)
        self.current_load = max(10, load + noise)
        
        # Track load and event for visualization
        self.load_history.append(self.current_load)
        self.event_history.append(event_type)
        
        # Limit history size
        if len(self.load_history) > 1000:
            self.load_history.pop(0)
            self.event_history.pop(0)

        return self.current_load

    def get_load_history(self, window: int = 50) -> float:
        """
        Return average load over the last window steps.
        
        Args:
            window: Number of steps to average over.
            
        Returns:
            float: Average load over the window.
        """
        if not self.load_history:
            return 0.0
        return float(np.mean(self.load_history[-window:]))

    def get_visualization_data(self, max_steps: int = 1000) -> Tuple[List[int], List[float], List[str]]:
        """
        Get visualization data for plotting.
        
        Args:
            max_steps: Maximum number of steps to include.
            
        Returns:
            Tuple containing steps, loads, and event types.
        """
        if not self.load_history or not self.event_history:
            return [], [], []
            
        # Get the last max_steps entries
        steps = list(range(len(self.load_history)))
        loads = self.load_history[-max_steps:] if len(self.load_history) > max_steps else self.load_history
        events = self.event_history[-max_steps:] if len(self.event_history) > max_steps else self.event_history
        
        # Ensure all lists have the same length
        min_length = min(len(steps), len(loads), len(events))
        return steps[:min_length], loads[:min_length], events[:min_length]