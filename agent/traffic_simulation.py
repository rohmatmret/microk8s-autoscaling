"""Traffic simulation module for generating realistic workload patterns.

This module provides traffic simulation capabilities for testing and evaluating
autoscaling algorithms with various traffic patterns including daily variations,
special events, and random noise.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

@dataclass
class TrafficEvent:
    """Represents a traffic event with timing and intensity."""
    start: int
    duration: int
    intensity: float
    event_type: str = "scheduled"  # scheduled, spike, burst, periodic

class TrafficSimulator:
    """Simulates realistic traffic patterns with multiple event types and patterns."""
    
    def __init__(self, 
                 base_load: float = 50,
                 max_spike: float = 30,
                 seed: Optional[int] = None,
                 daily_amplitude: float = 0.5,
                 spike_probability: float = 0.005,
                 min_spike_duration: int = 10,
                 max_spike_duration: int = 30,
                 min_load: float = 5,
                 history_size: int = 1000) -> None:
        """
        Initialize the traffic simulator with configurable parameters.
        
        Args:
            base_load: Base traffic load.
            max_spike: Maximum spike intensity multiplier.
            seed: Random seed for reproducibility.
            daily_amplitude: Amplitude of daily variation (0-1).
            spike_probability: Probability of random spike per step.
            min_spike_duration: Minimum duration of random spikes.
            max_spike_duration: Maximum duration of random spikes.
            min_load: Minimum allowed load.
            history_size: Maximum size of load history to maintain.
        """
        self.rng = np.random.default_rng(seed)
        self.base_load = base_load
        self.max_spike = max_spike
        self.current_load = base_load
        self.daily_amplitude = daily_amplitude
        self.spike_probability = spike_probability
        self.min_spike_duration = min_spike_duration
        self.max_spike_duration = max_spike_duration
        self.min_load = min_load
        self.history_size = history_size
        
        # Active events tracking
        self.active_spike: Optional[Dict] = None
        self.spike_duration = 0
        
        # History tracking
        self.load_history: List[float] = []
        self.event_history: List[str] = []
        
        # Event schedule (simulated marketplace events)
        self.events = [
            TrafficEvent(start=1000, duration=10, intensity=25, event_type="scheduled"),
            TrafficEvent(start=100, duration=10, intensity=25, event_type="scheduled"),
            TrafficEvent(start=1500, duration=300, intensity=20, event_type="scheduled"),
            TrafficEvent(start=50, duration=10, intensity=25, event_type="scheduled"),
            TrafficEvent(start=2500, duration=600, intensity=30, event_type="scheduled")
        ]
        
        # Periodic patterns (e.g., weekly patterns)
        self.weekly_pattern = self._generate_weekly_pattern()

    def _generate_weekly_pattern(self) -> List[float]:
        """Generate a weekly traffic pattern."""
        # More pronounced weekly pattern with lower weekend loads
        pattern = [1.0] * 5  # Weekdays
        pattern.extend([0.4, 0.3])  # Weekend with lower loads
        return pattern

    def get_load(self, step: int) -> float:
        """
        Generate realistic traffic pattern with multiple components.
        
        Args:
            step: Current simulation step.
            
        Returns:
            float: Current traffic load.
        """
        # 1. Base load with daily pattern
        daily_variation = self.base_load * self.daily_amplitude * np.sin(2 * np.pi * step / 1440)
        
        # 2. Weekly pattern
        day_of_week = (step // 1440) % 7
        weekly_multiplier = self.weekly_pattern[day_of_week]
        
        # 3. Random spikes (Poisson process)
        spike_effect = 0
        if self.active_spike and self.spike_duration > 0:
            spike_effect = self.active_spike["intensity"] * (self.spike_duration / self.active_spike["duration"])
            self.spike_duration -= 1
        elif self.rng.random() < self.spike_probability:
            spike_intensity = self.base_load * self.rng.uniform(2, self.max_spike)
            spike_duration = self.rng.integers(self.min_spike_duration, self.max_spike_duration)
            self.active_spike = {
                "intensity": spike_intensity,
                "duration": spike_duration
            }
            self.spike_duration = spike_duration
            spike_effect = spike_intensity
        
        # 4. Handle scheduled events
        event_multiplier = 1.0
        event_type = "none"
        for event in self.events:
            if event.start <= step < event.start + event.duration:
                event_multiplier = event.intensity
                event_type = event.event_type
                break
        
        # 5. Add random noise for realism
        noise = self.rng.normal(0, self.base_load * 0.05)
        
        # Calculate final load with more variation
        load = max(self.min_load, (
            (self.base_load + daily_variation) * weekly_multiplier * event_multiplier +
            spike_effect * self.rng.uniform(0.8, 1.2) +
            noise
        ))
        
        # Track history
        self.load_history.append(load)
        self.event_history.append(event_type)
        if len(self.load_history) > self.history_size:
            self.load_history.pop(0)
            self.event_history.pop(0)
        
        return load

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
            
        steps = list(range(len(self.load_history)))
        loads = self.load_history[-max_steps:] if len(self.load_history) > max_steps else self.load_history
        events = self.event_history[-max_steps:] if len(self.event_history) > max_steps else self.event_history
        
        min_length = min(len(steps), len(loads), len(events))
        return steps[:min_length], loads[:min_length], events[:min_length]

    def add_event(self, start: int, duration: int, intensity: float, event_type: str = "scheduled") -> None:
        """
        Add a new traffic event to the schedule.
        
        Args:
            start: Start step of the event.
            duration: Duration of the event in steps.
            intensity: Intensity multiplier of the event.
            event_type: Type of the event.
        """
        self.events.append(TrafficEvent(start=start, duration=duration, intensity=intensity, event_type=event_type))

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