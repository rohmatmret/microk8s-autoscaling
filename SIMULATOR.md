# key differences between HybridTrafficSimulator and TrafficSimulator:
1. Event Generation:
   - HybridTrafficSimulator: Uses event-based approach with explicit event_frequency parameter
   - TrafficSimulator: Uses spike_probability for random events and adds daily patterns

   ```python
   # HybridTrafficSimulator
   self.event_frequency = 0.005  # Event probability per step
   
   # TrafficSimulator
   self.spike_probability = 0.005  # Spike chance
   self.daily_pattern = np.sin(2 * np.pi * step / steps_per_day) * self.daily_amplitude
   ```

2. Load Parameters:
   - HybridTrafficSimulator:
     - min_intensity/max_intensity for event load range
     - min_duration/max_duration for event duration
   - TrafficSimulator: 
     - max_spike for spike magnitude
     - min_spike_duration/max_spike_duration for spike length
     - daily_amplitude for daily pattern variation
     - min_load as absolute minimum

   ```python
   # HybridTrafficSimulator
   self.min_intensity = 5
   self.max_intensity = 50
   self.min_duration = 10
   self.max_duration = 200

   # TrafficSimulator
   self.max_spike = 30
   self.min_spike_duration = 10
   self.max_spike_duration = 30
   self.daily_amplitude = 0.3
   self.min_load = 10
   ```

3. Pattern Generation:
   - HybridTrafficSimulator: Focuses on discrete events
   - TrafficSimulator: Combines:
     - Daily cyclic patterns (morning/afternoon/evening peaks)
     - Random spikes
     - Base load variations

   ```python
   # HybridTrafficSimulator
   if random.random() < self.event_frequency:
       intensity = random.uniform(self.min_intensity, self.max_intensity)
       duration = random.randint(self.min_duration, self.max_duration)
       
   # TrafficSimulator
   load = self.base_load + self.daily_pattern
   if random.random() < self.spike_probability:
       load += random.uniform(0, self.max_spike)
   ```

4. Configurability:
   - HybridTrafficSimulator: More focused on event parameters
   - TrafficSimulator: More parameters for natural traffic patterns

   ```python
   # HybridTrafficSimulator initialization
   simulator = HybridTrafficSimulator(
       base_load=100,
       event_frequency=0.005,
       min_intensity=5,
       max_intensity=50
   )

   # TrafficSimulator initialization
   simulator = TrafficSimulator(
       base_load=100,
       max_spike=30,
       daily_amplitude=0.3,
       spike_probability=0.005
   )
   ```

5. History Management:
   - HybridTrafficSimulator: Basic event tracking
   - TrafficSimulator: Built-in history size management with visualization support

   ```python
   # TrafficSimulator history management
   self.load_history = deque(maxlen=history_size)
   self.event_history = deque(maxlen=history_size)
   
   def get_visualization_data(self):
       return list(range(len(self.load_history))), \
              list(self.load_history), \
              list(self.event_history)
   ```

6. Use Case Focus:
   - HybridTrafficSimulator: Better for testing specific event responses
   - TrafficSimulator: Better for realistic daily traffic patterns

