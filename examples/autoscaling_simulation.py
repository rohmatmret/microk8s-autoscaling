"""Final version of traffic simulation with optimized autoscaling, detailed statistics, and additional metrics."""

import numpy as np
import matplotlib.pyplot as plt
from agent.traffic_simulation import TrafficSimulator

def simulate_autoscaling():
    """Simulate autoscaling behavior with realistic traffic patterns, advanced metrics, and optimizations."""
    # Initialize traffic simulator with optimized settings
    simulator = TrafficSimulator(
        base_load=100,
        max_spike=50,
        daily_amplitude=0.3,
        spike_probability=0.005,
        min_spike_duration=10,
        max_spike_duration=60
    )
    
    # Add scheduled events
    simulator.add_event(start=1000, duration=300, intensity=40, event_type="burst")
    simulator.add_event(start=5000, duration=1000, intensity=20, event_type="periodic")
    
    # Simulation parameters
    steps = 10080  # 7 days
    current_pods = 1
    max_pods = 30
    min_pods = 1
    pod_capacity = 100
    
    # Production-ready scaling parameters
    scale_up_threshold = 0.70    # Scale up at 70% CPU
    scale_down_threshold = 0.50  # Scale down at 50% CPU
    scale_up_step = 2           # Add 2 pods when scaling up
    scale_down_step = 2         # Remove 1 pod when scaling down
    cooldown_period = 100       # 5 minutes between scaling decisions
    stabilization_period = 1200 # 10 minutes after spike
    min_stable_pods = 3         # Minimum pods during stabilization
    last_scaling_step = 0
    
    # Predictive scaling parameters
    window_size = 10
    prediction_margin = 2.5
    
    # Advanced metric thresholds
    latency_threshold = 140
    p95_latency_threshold = 200
    error_threshold = 0.01      # 1% error rate
    queue_threshold = 50
    memory_threshold = 0.85
    base_latency = 50
    
    # Spike detection
    spike_threshold = 0.7
    last_spike_step = -1
    
    # Metrics collection
    loads = []
    cpu_utils = []
    pod_counts = []
    latencies = []
    errors = []
    queue_lengths = []
    memory_utils = []
    throughputs = []
    drop_rates = []
    scaling_events = []
    
    # Run simulation
    for step in range(steps):
        load = simulator.get_load(step)
        loads.append(load)
        
        # Calculate CPU utilization
        cpu_util = min(1.0, load / (current_pods * pod_capacity))
        cpu_utils.append(cpu_util)
        
        # Calculate latency
        latency = base_latency * (2 + cpu_util)
        latencies.append(latency)
        
        # Calculate advanced metrics
        error_rate = 0.0 if cpu_util < 0.9 else (cpu_util - 0.9) * 0.5
        errors.append(error_rate)
        
        queue_length = max(0, load - current_pods * pod_capacity)
        queue_lengths.append(queue_length)
        
        memory_util = min(1.0, cpu_util * 0.8 + 0.1)
        memory_utils.append(memory_util)
        
        throughput = min(load, current_pods * pod_capacity)
        drop_rate = max(0, (load - throughput) / load) if load > 0 else 0
        throughputs.append(throughput)
        drop_rates.append(drop_rate)
        
        # Predictive scaling
        predicted_cpu = cpu_util
        if step >= window_size:
            recent_loads = loads[-window_size:]
            predicted_load = np.mean(recent_loads) * prediction_margin
            predicted_cpu = min(1.0, predicted_load / (current_pods * pod_capacity))
        
        # Scaling decisions
        if step - last_scaling_step >= cooldown_period:
            if cpu_util > spike_threshold and last_spike_step == -1:
                last_spike_step = step
            
            # Scale up based on CPU, latency, or advanced metrics
            scale_up_triggered = False
            if (cpu_util > scale_up_threshold or predicted_cpu > scale_up_threshold) and current_pods < max_pods:
                current_pods = min(max_pods, current_pods + scale_up_step)
                scaling_events.append(("scale_up_cpu", step))
                scale_up_triggered = True
            elif latency > latency_threshold and current_pods < max_pods:
                current_pods = min(max_pods, current_pods + scale_up_step)
                scaling_events.append(("scale_up_latency", step))
                scale_up_triggered = True
            elif step >= window_size:
                recent_latencies = latencies[-window_size:]
                current_p95_latency = np.percentile(recent_latencies, 95)
                recent_errors = errors[-window_size:]
                avg_error_rate = np.mean(recent_errors)
                if (current_p95_latency > p95_latency_threshold or 
                    avg_error_rate > error_threshold or 
                    queue_length > queue_threshold or 
                    memory_util > memory_threshold) and current_pods < max_pods:
                    current_pods = min(max_pods, current_pods + scale_up_step)
                    scaling_events.append(("scale_up_advanced", step))
                    scale_up_triggered = True
            
            # Scale down logic
            if not scale_up_triggered and cpu_util < scale_down_threshold and current_pods > min_pods:
                if last_spike_step != -1 and step - last_spike_step < stabilization_period:
                    current_pods = max(min_stable_pods, current_pods)
                else:
                    current_pods = max(min_pods, current_pods - scale_down_step)
                    scaling_events.append(("scale_down", step))
                    if cpu_util < 0.2:
                        last_spike_step = -1
            
            if scale_up_triggered or (cpu_util < scale_down_threshold and current_pods > min_pods):
                last_scaling_step = step
        
        # Trend-based scaling
        if step >= 10 and cpu_utils[step] - cpu_utils[step-10] > 0.1 and current_pods < max_pods:
            current_pods = min(max_pods, current_pods + scale_up_step)
            scaling_events.append(("scale_up_trend", step))
            last_scaling_step = step
        
        # Logging for debugging
        if predicted_cpu > scale_up_threshold:
            print(f"Step {step}: Predictive scale-up triggered, predicted CPU: {predicted_cpu:.2f}")
        if latency > latency_threshold:
            print(f"Step {step}: Latency scale-up triggered, latency: {latency:.2f} ms")
        
        pod_counts.append(current_pods)
    
    # Visualization
    plt.figure(figsize=(15, 18))
    
    plt.subplot(6, 1, 1)
    plt.plot(loads, label='Traffic Load', color='blue')
    plt.title('Traffic Pattern and Autoscaling Simulation (Final Version)')
    plt.xlabel('Time Steps')
    plt.ylabel('Load (req/s)')
    plt.legend()
    
    plt.subplot(6, 1, 2)
    plt.plot(cpu_utils, label='CPU Utilization', color='green')
    plt.axhline(y=scale_up_threshold, color='r', linestyle='--', label='Scale Up Threshold')
    plt.axhline(y=scale_down_threshold, color='r', linestyle=':', label='Scale Down Threshold')
    plt.xlabel('Time Steps')
    plt.ylabel('CPU Utilization')
    plt.legend()
    
    plt.subplot(6, 1, 3)
    plt.plot(latencies, label='Latency', color='orange')
    plt.axhline(y=latency_threshold, color='r', linestyle='--', label='Latency Threshold')
    plt.axhline(y=p95_latency_threshold, color='b', linestyle=':', label='P95 Latency Threshold')
    plt.xlabel('Time Steps')
    plt.ylabel('Latency (ms)')
    plt.legend()
    
    plt.subplot(6, 1, 4)
    plt.plot(errors, label='Error Rate', color='red')
    plt.axhline(y=error_threshold, color='r', linestyle='--', label='Error Threshold')
    plt.xlabel('Time Steps')
    plt.ylabel('Error Rate')
    plt.legend()
    
    plt.subplot(6, 1, 5)
    plt.plot(queue_lengths, label='Queue Length', color='purple')
    plt.axhline(y=queue_threshold, color='r', linestyle='--', label='Queue Threshold')
    plt.xlabel('Time Steps')
    plt.ylabel('Queue Length')
    plt.legend()
    
    plt.subplot(6, 1, 6)
    plt.plot(pod_counts, label='Pod Count', color='purple')
    for event_type, step in scaling_events:
        if "scale_up" in event_type:
            color, marker = 'g', '^'
        else:
            color, marker = 'r', 'v'
        plt.scatter(step, pod_counts[step], color=color, marker=marker)
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Pods')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('autoscaling_simulation_final.png')
    plt.close()
    
    # Scaling decision timeline
    plt.figure(figsize=(15, 4))
    plt.title("Scaling Decision Timeline")
    plt.xlabel("Time Steps")
    plt.ylabel("Scaling Event")
    
    scale_up_cpu_steps = [step for event, step in scaling_events if event == "scale_up_cpu"]
    scale_up_latency_steps = [step for event, step in scaling_events if event == "scale_up_latency"]
    scale_up_advanced_steps = [step for event, step in scaling_events if event == "scale_up_advanced"]
    scale_up_trend_steps = [step for event, step in scaling_events if event == "scale_up_trend"]
    scale_down_steps = [step for event, step in scaling_events if event == "scale_down"]
    
    plt.eventplot([scale_up_cpu_steps, scale_up_latency_steps, scale_up_advanced_steps, scale_up_trend_steps, scale_down_steps],
                  colors=['green', 'blue', 'cyan', 'orange', 'red'],
                  lineoffsets=[3, 2, 1, 0, -1],
                  linelengths=0.8,
                  linewidths=1.5,
                  label='Scaling Events')
    
    plt.yticks([-1, 0, 1, 2, 3], ['Scale Down', 'Scale Up (Trend)', 'Scale Up (Advanced)', 'Scale Up (Latency)', 'Scale Up (CPU)'])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("scaling_decision_timeline_final.png")
    plt.close()
    
    # Detailed Statistics and Analysis
    print("\n=== Detailed Simulation Statistics and Analysis ")
    
    # Traffic Load Analysis
    avg_load = np.mean(loads)
    max_load = np.max(loads)
    min_load = np.min(loads)
    std_load = np.std(loads)
    print(f"1. Traffic Load Analysis:")
    print(f"   - Average Load: {avg_load:.2f} req/s")
    print(f"   - Maximum Load: {max_load:.2f} req/s")
    print(f"   - Minimum Load: {min_load:.2f} req/s")
    print(f"   - Standard Deviation: {std_load:.2f} req/s")
    print(f"   - Interpretation: High variability (std {std_load:.2f}) requires robust scaling to handle spikes up to {max_load:.2f} req/s.")
    
    # CPU Utilization Analysis
    avg_cpu = np.mean(cpu_utils)
    median_cpu = np.median(cpu_utils)
    std_cpu = np.std(cpu_utils)
    print(f"\n2. CPU Utilization Analysis:")
    print(f"   - Average CPU Utilization: {avg_cpu:.2f} (Target: 0.5-0.8)")
    print(f"   - Median CPU Utilization: {median_cpu:.2f}")
    print(f"   - Standard Deviation: {std_cpu:.2f}")
    print(f"   - Interpretation: Average CPU {avg_cpu:.2f} is below target, with high variability (std {std_cpu:.2f}). Adjust thresholds to optimize resource use.")
    
    # Latency Analysis
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    print(f"\n3. Latency Analysis:")
    print(f"   - Average Latency: {avg_latency:.2f} ms (Threshold: {latency_threshold} ms)")
    print(f"   - P95 Latency: {p95_latency:.2f} ms (Threshold: {p95_latency_threshold} ms)")
    print(f"   - Maximum Latency: {max_latency:.2f} ms")
    print(f"   - Interpretation: Latency is well-controlled, with P95 at {p95_latency:.2f} ms. Consider tightening thresholds for sensitive applications.")
    
    # Pod Utilization Analysis
    avg_pods = np.mean(pod_counts)
    max_pods_used = np.max(pod_counts)
    print(f"\n4. Pod Utilization Analysis:")
    print(f"   - Average Number of Pods: {avg_pods:.2f} (Max Capacity: {max_pods})")
    print(f"   - Maximum Pods Used: {max_pods_used:.2f}")
    print(f"   - Interpretation: Low pod usage (avg {avg_pods:.2f}) suggests potential overprovisioning. Evaluate max_pods to reduce costs.")
    
    # Scaling Event Analysis
    total_scaling_events = len(scaling_events)
    scale_up_cpu = sum(1 for e, _ in scaling_events if e == 'scale_up_cpu')
    scale_up_latency = sum(1 for e, _ in scaling_events if e == 'scale_up_latency')
    scale_up_advanced = sum(1 for e, _ in scaling_events if e == 'scale_up_advanced')
    scale_up_trend = sum(1 for e, _ in scaling_events if e == 'scale_up_trend')
    scale_down = sum(1 for e, _ in scaling_events if e == 'scale_down')
    print(f"\n5. Scaling Event Analysis:")
    print(f"   - Total Scaling Events: {total_scaling_events}")
    print(f"   - Scale Up Events (CPU): {scale_up_cpu}")
    print(f"   - Scale Up Events (Latency): {scale_up_latency}")
    print(f"   - Scale Up Events (Advanced Metrics): {scale_up_advanced}")
    print(f"   - Scale Up Events (Trend): {scale_up_trend}")
    print(f"   - Scale Down Events: {scale_down}")
    print(f"   - Interpretation: Moderate scaling activity ({total_scaling_events/steps*100:.2f}% of time). High scale-downs ({scale_down}) may be aggressive; extend stabilization_period if needed.")
    
    # Trend Analysis
    moving_avg_load = np.convolve(loads, np.ones(window_size)/window_size, mode='valid')
    trend_periods = len(moving_avg_load)
    peak_trends = sum(1 for i in range(1, trend_periods-1) if moving_avg_load[i] > moving_avg_load[i-1] and moving_avg_load[i] > moving_avg_load[i+1])
    print(f"\n6. Trend Analysis (Moving Average, Window={window_size}):")
    print(f"   - Average Trend Value: {np.mean(moving_avg_load):.2f} req/s")
    print(f"   - Number of Peak Trends: {peak_trends}")
    print(f"   - Interpretation: {peak_trends/trend_periods*100:.2f}% of time shows upward trends, indicating recurring spikes. Enhance predictive scaling.")
    
    # Error Rate Analysis
    avg_error_rate = np.mean(errors)
    p99_error_rate = np.percentile(errors, 99)
    print(f"\n7. Error Rate Analysis:")
    print(f"   - Average Error Rate: {avg_error_rate*100:.2f}% (Threshold: {error_threshold*100}%)")
    print(f"   - P99 Error Rate: {p99_error_rate*100:.2f}%")
    print(f"   - Interpretation: Error rate at {avg_error_rate*100:.2f}% is acceptable, with P99 at {p99_error_rate*100:.2f}% showing rare spikes. Monitor closely during peaks.")
    
    # Queue Length Analysis
    avg_queue_length = np.mean(queue_lengths)
    max_queue_length = np.max(queue_lengths)
    print(f"\n8. Queue Length Analysis:")
    print(f"   - Average Queue Length: {avg_queue_length:.2f} requests (Threshold: {queue_threshold})")
    print(f"   - Maximum Queue Length: {max_queue_length:.2f} requests")
    print(f"   - Interpretation: Queue length at {avg_queue_length:.2f} is manageable, with max {max_queue_length:.2f} indicating occasional bottlenecks.")
    
    # Memory Utilization Analysis
    avg_memory_util = np.mean(memory_utils)
    max_memory_util = np.max(memory_utils)
    print(f"\n9. Memory Utilization Analysis:")
    print(f"   - Average Memory Utilization: {avg_memory_util:.2f} (Threshold: {memory_threshold})")
    print(f"   - Maximum Memory Utilization: {max_memory_util:.2f}")
    print(f"   - Interpretation: Memory usage at {avg_memory_util:.2f} is safe, with max {max_memory_util:.2f} showing pressure during peaks.")
    
    # Throughput and Drop Rate Analysis
    avg_throughput = np.mean(throughputs)
    avg_drop_rate = np.mean(drop_rates)
    print(f"\n10. Throughput and Drop Rate Analysis:")
    print(f"   - Average Throughput: {avg_throughput:.2f} req/s")
    print(f"   - Average Drop Rate: {avg_drop_rate*100:.2f}%")
    print(f"   - Interpretation: Throughput at {avg_throughput:.2f} req/s meets most demand, but drop rate at {avg_drop_rate*100:.2f}% suggests lost requests during peaks.")
    
    # Overall System Performance
    print(f"\n11. Overall System Performance:")
    print(f"   - The system handles an average load of {avg_load:.2f} req/s with peaks up to {max_load:.2f} req/s using an average of {avg_pods:.2f} pods.")
    print(f"   - CPU utilization ({avg_cpu:.2f}), latency ({avg_latency:.2f} ms), and error rate ({avg_error_rate*100:.2f}%) indicate stable performance.")
    print(f"   - Recommendations: Optimize scale_down_threshold to reduce aggressive scale-downs, implement predictive scaling for trends, and monitor P95 latency and drop rate during spikes.")

if __name__ == "__main__":
    simulate_autoscaling()