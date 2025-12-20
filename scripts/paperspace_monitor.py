#!/usr/bin/env python3
"""Monitor Paperspace training progress in real-time."""

import time
import json
from pathlib import Path
import sys

def monitor_training(log_file="training.log", refresh_interval=30):
    """Monitor training progress and display metrics."""

    print("=" * 60)
    print("Monitoring Paperspace Training Progress")
    print("=" * 60)
    print("Press Ctrl+C to stop monitoring")
    print("")

    last_steps = 0
    start_time = time.time()

    while True:
        try:
            # Read latest metrics from training summary
            summary_path = Path("models/hybrid/training_summary.yaml")

            if summary_path.exists():
                try:
                    import yaml
                    with open(summary_path) as f:
                        summary = yaml.safe_load(f)

                    steps = summary.get('total_steps', 0)
                    episodes = summary.get('final_episode_count', 0)
                    elapsed = time.time() - start_time

                    # Calculate steps per second
                    if elapsed > 0:
                        steps_delta = steps - last_steps
                        sps = steps_delta / refresh_interval if steps_delta > 0 else 0
                    else:
                        sps = 0

                    # Display progress
                    print(f"\rSteps: {steps:>6} | Episodes: {episodes:>4} | "
                          f"Steps/sec: {sps:.1f} | Elapsed: {elapsed/60:.1f}m", end="")

                    last_steps = steps

                except Exception as e:
                    print(f"\rError reading summary: {e}", end="")
            else:
                print(f"\rWaiting for training to start...", end="")

            time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(refresh_interval)

if __name__ == "__main__":
    monitor_training()
