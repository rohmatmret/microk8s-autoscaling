import pytest
from agent.traffic_simulation import TrafficSimulator

@pytest.fixture
def simulator():
    return TrafficSimulator(base_load=100, max_spike=30, seed=42)

def test_initialization(simulator):
    assert simulator.base_load == 100
    assert simulator.max_spike == 30 
    assert simulator.current_load == 100
    assert simulator.spike_duration == 0
    assert len(simulator.events) == 5

def test_load_minimum(simulator):
    # Test that load never goes below 10
    for step in range(1000):
        load = simulator.get_load(step)
        print(f"Step {step}: Load = {load}")
        assert load >= 10

def test_event_multiplier(simulator):
    # Test load during an event
    load_at_event = simulator.get_load(100)  # Step 100 has an event
    load_no_event = simulator.get_load(200)  # Step 200 has no event
    assert load_at_event != load_no_event

def test_daily_variation(simulator):
    # Test that load varies throughout the day
    loads = [simulator.get_load(step) for step in range(0, 1440, 120)]
    assert any(loads[i] != loads[i+1] for i in range(len(loads)-1))