"""Test suite for verifying TrafficSimulator implementation in MicroK8sEnvSimulated."""

import pytest
import numpy as np
from agent.environment_simulated import MicroK8sEnvSimulated
from agent.traffic_simulation import TrafficSimulator

def test_environment_initialization():
    """Test proper initialization of environment with TrafficSimulator."""
    env = MicroK8sEnvSimulated(seed=42)
    
    # Verify traffic simulator initialization
    assert isinstance(env.traffic_simulator, TrafficSimulator)
    assert env.traffic_simulator.base_load == 100
    assert env.traffic_simulator.max_spike == 30
    assert env.traffic_simulator.daily_amplitude == 0.3
    
    # Verify load history initialization
    assert len(env.load_history) == env.MIN_HISTORY_LENGTH
    assert all(load == env.traffic_simulator.base_load for load in env.load_history)

def test_traffic_events_setup():
    """Test proper setup of traffic events."""
    env = MicroK8sEnvSimulated(seed=42)
    
    # Test traffic patterns by simulating loads
    loads = []
    for step in range(1440):  # Simulate 24 hours
        load = env.traffic_simulator.get_load(step)
        loads.append(load)
    
    # Calculate hourly averages
    hourly_averages = [np.mean(loads[i:i+60]) for i in range(0, 1440, 60)]
    
    # Basic validation of load values
    assert all(load >= env.traffic_simulator.min_load for load in loads), "All loads should be above minimum"
    assert all(isinstance(load, (int, float)) for load in loads), "All loads should be numeric"
    
    # Verify daily pattern exists
    max_load = max(hourly_averages)
    min_load = min(hourly_averages)
    assert max_load > min_load, "Should have daily variation in load"
    
    # Calculate moving averages for smoother comparison
    def moving_average(data, window=3):
        return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]
    
    # Smooth hourly averages
    smoothed_averages = moving_average(hourly_averages)
    
    # Verify peak periods
    morning_hours = range(6, 9)  # 6-8 AM
    afternoon_hours = range(12, 15)  # 12-2 PM
    evening_hours = range(18, 21)  # 6-8 PM
    
    # Calculate average loads for different periods
    morning_avg = np.mean([smoothed_averages[i] for i in morning_hours])
    afternoon_avg = np.mean([smoothed_averages[i] for i in afternoon_hours])
    evening_avg = np.mean([smoothed_averages[i] for i in evening_hours])
    
    # Calculate surrounding period averages
    early_morning_avg = np.mean([smoothed_averages[i] for i in range(2, 5)])
    late_morning_avg = np.mean([smoothed_averages[i] for i in range(10, 13)])
    late_afternoon_avg = np.mean([smoothed_averages[i] for i in range(16, 19)])
    
    # Verify peak patterns
    assert morning_avg >= early_morning_avg * 0.9, "Morning peak should be at least 90% of early morning"
    assert afternoon_avg >= late_morning_avg * 0.9, "Afternoon peak should be at least 90% of late morning"
    assert evening_avg >= late_afternoon_avg * 0.9, "Evening peak should be at least 90% of late afternoon"
    
    # Verify load distribution
    load_std = np.std(hourly_averages)
    load_mean = np.mean(hourly_averages)
    assert load_std > 0, "Load should have some variation"
    assert load_std < load_mean, "Load variation should be reasonable"

def test_load_history_management():
    """Test proper management of load history."""
    env = MicroK8sEnvSimulated(seed=42)
    
    # Test history size limits
    for _ in range(env.MAX_HISTORY_LENGTH + 100):
        env.step(0)  # No-op action
    
    assert len(env.load_history) <= env.MAX_HISTORY_LENGTH
    
    # Test history content
    loads = list(env.load_history)
    assert all(isinstance(load, (int, float)) for load in loads)
    assert all(load >= env.traffic_simulator.min_load for load in loads)
    
    # Test history updates
    initial_load = env.load_history[-1]
    env.step(0)
    assert env.load_history[-1] != initial_load, "Load history should update with new values"

def test_load_gradient_calculation():
    """Test load gradient calculation."""
    env = MicroK8sEnvSimulated(seed=42)
    
    # Generate some load history
    for _ in range(10):
        env.step(0)
    
    # Test gradient calculation
    gradient = env._calculate_load_gradient()
    assert isinstance(gradient, float)
    assert -1.0 <= gradient <= 1.0  # Should be normalized
    
    # Test gradient with increasing load
    env.load_history.clear()
    increasing_loads = [100 + i * 10 for i in range(10)]
    env.load_history.extend(increasing_loads)
    gradient = env._calculate_load_gradient()
    assert gradient > 0  # Should be positive for increasing load
    
    # Test gradient with decreasing load
    env.load_history.clear()
    decreasing_loads = [200 - i * 10 for i in range(10)]
    env.load_history.extend(decreasing_loads)
    gradient = env._calculate_load_gradient()
    assert gradient < 0  # Should be negative for decreasing load

def test_traffic_patterns():
    """Test realistic traffic patterns."""
    env = MicroK8sEnvSimulated(seed=42)
    loads = []
    
    # Simulate for 24 hours (1440 steps)
    for step in range(1440):
        load = env.traffic_simulator.get_load(step)
        loads.append(load)
    
    # Test daily pattern
    hourly_averages = [np.mean(loads[i:i+60]) for i in range(0, 1440, 60)]
    max_load = max(hourly_averages)
    min_load = min(hourly_averages)
    assert max_load > min_load, "Should have daily variation in load"
    
    # Test load distribution
    load_std = np.std(hourly_averages)
    load_mean = np.mean(hourly_averages)
    assert load_std > 0, "Load should have some variation"
    assert load_std < load_mean, "Load variation should be reasonable"
    
    # Test peak hours
    morning_peak = np.mean(hourly_averages[6:8])  # 6-8 AM
    afternoon_peak = np.mean(hourly_averages[12:14])  # 12-2 PM
    evening_peak = np.mean(hourly_averages[18:20])  # 6-8 PM
    
    # Verify peaks are higher than surrounding hours
    assert morning_peak >= np.mean(hourly_averages[2:4]), "Morning peak should be higher than early morning"
    assert afternoon_peak >= np.mean(hourly_averages[10:12]), "Afternoon peak should be higher than late morning"
    assert evening_peak >= np.mean(hourly_averages[16:18]), "Evening peak should be higher than late afternoon"

def test_error_handling():
    """Test error handling in traffic simulation."""
    env = MicroK8sEnvSimulated(seed=42)
    
    # Test handling of history overflow
    for _ in range(env.MAX_HISTORY_LENGTH + 100):
        env.step(0)
    assert len(env.load_history) <= env.MAX_HISTORY_LENGTH
    
    # Test handling of empty history
    env.load_history.clear()
    gradient = env._calculate_load_gradient()
    assert gradient == 0.0  # Should return 0 for empty history
    
    # Test handling of invalid step
    try:
        env.traffic_simulator.get_load(-1)
        assert False, "Should have raised an exception for negative step"
    except Exception:
        pass
    
    # Test handling of invalid state
    invalid_state = {"cpu": -1, "memory": -1, "latency": -1, "swap": -1, "nodes": -1, "pods": -1}
    reward = env._calculate_reward(invalid_state, 0, True)
    assert isinstance(reward, float)
    assert -1.0 <= reward <= 1.0

def test_reset_behavior():
    """Test proper reset behavior."""
    env = MicroK8sEnvSimulated(seed=42)
    
    # Run some steps
    for _ in range(100):
        env.step(0)
    
    # Reset environment
    env.reset()
    
    # Verify reset state
    assert env.current_step == 0
    assert len(env.load_history) == env.MIN_HISTORY_LENGTH
    assert all(load == env.traffic_simulator.base_load for load in env.load_history)
    
    # Verify state after reset
    state = env._observe()
    assert isinstance(state, np.ndarray)
    assert state.shape == (7,)  # [cpu, memory, latency, swap, nodes, load_mean, load_gradient]

def test_reward_calculation():
    """Test reward calculation with traffic patterns."""
    env = MicroK8sEnvSimulated(seed=42)
    
    # Initialize load history
    for _ in range(env.MIN_HISTORY_LENGTH):
        env.step(0)
    
    # Test reward during normal load
    state = {
        "cpu": 0.5,
        "memory": env.DEFAULT_MEMORY_BYTES,
        "latency": 0.1,
        "swap": 0.0,
        "nodes": 1,
        "pods": 1
    }
    
    # Calculate rewards for different scenarios
    reward_normal = env._calculate_reward(state, 0, True)
    assert isinstance(reward_normal, float)
    assert -1.0 <= reward_normal <= 1.0
    
    # Test reward during high load
    state["cpu"] = 0.8
    reward_high = env._calculate_reward(state, 1, True)  # Scale up action
    assert isinstance(reward_high, float)
    assert -1.0 <= reward_high <= 1.0
    
    # Test reward during low load
    state["cpu"] = 0.2
    reward_low = env._calculate_reward(state, 2, True)  # Scale down action
    assert isinstance(reward_low, float)
    assert -1.0 <= reward_low <= 1.0
    
    # Test reward with invalid state
    invalid_state = {"cpu": -1, "memory": -1, "latency": -1, "swap": -1, "nodes": -1, "pods": -1}
    reward_invalid = env._calculate_reward(invalid_state, 0, True)
    assert isinstance(reward_invalid, float)
    assert -1.0 <= reward_invalid <= 1.0

def test_visualization_data():
    """Test visualization data generation."""
    env = MicroK8sEnvSimulated(seed=42)
    
    # Generate some load history
    for _ in range(100):
        env.step(0)
    
    # Get visualization data
    steps, loads, events = env.traffic_simulator.get_visualization_data()
    
    # Verify data structure
    assert len(steps) == len(loads) == len(events)
    assert all(isinstance(step, int) for step in steps)
    assert all(isinstance(load, (int, float)) for load in loads)
    assert all(isinstance(event, str) for event in events)
    
    # Verify data ranges
    assert min(steps) >= 0
    assert max(steps) < len(steps)
    assert all(load >= env.traffic_simulator.min_load for load in loads)
    assert all(event in ["none", "scheduled", "spike", "burst", "periodic"] for event in events)

if __name__ == "__main__":
    # Run all tests
    test_environment_initialization()
    test_traffic_events_setup()
    test_load_history_management()
    test_load_gradient_calculation()
    test_traffic_patterns()
    test_error_handling()
    test_reset_behavior()
    test_reward_calculation()
    test_visualization_data()
    print("All tests passed!") 