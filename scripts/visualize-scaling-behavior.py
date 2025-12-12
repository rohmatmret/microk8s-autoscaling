#!/usr/bin/env python3
"""
Scaling Behavior Visualization Script

Creates comprehensive visualizations for autoscaling agent behavior comparison.
Best visualizations for scaling behavior:
1. Time-series: Pod count changes over time (shows scaling decisions)
2. Action distribution: Pie/bar charts (shows decision patterns)
3. Multi-metric comparison: Radar charts (shows overall performance)
4. Response metrics: Line charts (shows system behavior)
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from collections import Counter

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_data(file_path):
    """Load scaling decision data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Handle nested structure
        if 'results' in data:
            return data['results']
        return data
    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON file: {file_path}")
        sys.exit(1)


def extract_time_series(agent_data, scenario_name):
    """Extract time series data from scenario steps."""
    if scenario_name not in agent_data or not agent_data[scenario_name]:
        return None

    scenario_data = agent_data[scenario_name]

    return {
        'steps': list(range(len(scenario_data))),
        'pod_count': [step['pod_count'] for step in scenario_data],
        'target_pod_count': [step['target_pod_count'] for step in scenario_data],
        'cpu_utilization': [step['cpu_utilization'] for step in scenario_data],
        'response_time': [step['response_time'] for step in scenario_data],
        'throughput': [step['throughput'] for step in scenario_data],
        'sla_violations': [step['sla_violations'] for step in scenario_data],
        'resource_cost': [step['resource_cost'] for step in scenario_data],
    }


def extract_action_distribution(agent_data):
    """Extract cumulative action distribution across all scenarios."""
    total_actions = {'scale_up': 0, 'scale_down': 0, 'no_change': 0}

    for scenario_name, scenario_data in agent_data.items():
        if scenario_name in ['analysis', 'timestamp'] or not scenario_data:
            continue

        # Get the last step which has cumulative counts
        if len(scenario_data) > 0:
            last_step = scenario_data[-1]
            if 'action_distribution' in last_step:
                actions = last_step['action_distribution']
                total_actions['scale_up'] += actions.get('scale_up', 0)
                total_actions['scale_down'] += actions.get('scale_down', 0)
                total_actions['no_change'] += actions.get('no_change', 0)

    return total_actions


def plot_pod_count_timeline(ax, data, agents, scenario_name):
    """
    Plot 1: Pod count over time (BEST for showing scaling decisions)
    Shows actual scaling behavior with target pod count overlay.
    """
    ax.set_title(f'Pod Count Timeline - {scenario_name.replace("_", " ").title()}',
                 fontweight='bold', pad=10)

    colors = {'hybrid_dqn_ppo': '#2E86AB', 'k8s_hpa': '#A23B72'}
    styles = {'hybrid_dqn_ppo': '-', 'k8s_hpa': '--'}

    for agent_name in agents:
        if agent_name not in data:
            continue

        ts_data = extract_time_series(data[agent_name], scenario_name)
        if ts_data is None:
            continue

        label = agent_name.replace('_', ' ').title()
        color = colors.get(agent_name, 'gray')

        # Plot actual pod count
        ax.plot(ts_data['steps'], ts_data['pod_count'],
               color=color, linestyle=styles[agent_name], linewidth=2.5,
               label=f'{label} (Actual)', marker='o', markersize=4, alpha=0.8)

        # Plot target pod count with dashed line
        ax.plot(ts_data['steps'], ts_data['target_pod_count'],
               color=color, linestyle=':', linewidth=1.5,
               label=f'{label} (Target)', alpha=0.5)

    ax.set_xlabel('Time Step', fontweight='bold')
    ax.set_ylabel('Pod Count', fontweight='bold')

    # Only show legend if there are plots
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', framealpha=0.9)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)


def plot_cpu_utilization(ax, data, agents, scenario_name):
    """
    Plot 2: CPU utilization over time
    Shows resource usage patterns that trigger scaling.
    """
    ax.set_title(f'CPU Utilization - {scenario_name.replace("_", " ").title()}',
                 fontweight='bold', pad=10)

    colors = {'hybrid_dqn_ppo': '#2E86AB', 'k8s_hpa': '#A23B72'}

    for agent_name in agents:
        if agent_name not in data:
            continue

        ts_data = extract_time_series(data[agent_name], scenario_name)
        if ts_data is None:
            continue

        label = agent_name.replace('_', ' ').title()
        color = colors.get(agent_name, 'gray')

        ax.plot(ts_data['steps'], [cpu * 100 for cpu in ts_data['cpu_utilization']],
               color=color, linewidth=2, label=label, marker='s', markersize=3, alpha=0.7)

    # Add HPA target line (70%)
    if ts_data:
        ax.axhline(y=70, color='red', linestyle='--', linewidth=1.5,
                  label='HPA Target (70%)', alpha=0.7)

    ax.set_xlabel('Time Step', fontweight='bold')
    ax.set_ylabel('CPU Utilization (%)', fontweight='bold')

    # Only show legend if there are plots
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', framealpha=0.9)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)


def plot_action_distribution_comparison(ax, data, agents):
    """
    Plot 3: Action distribution comparison (EXCELLENT for showing decision patterns)
    Side-by-side bar chart showing scaling decision frequencies.
    """
    ax.set_title('Scaling Action Distribution Comparison', fontweight='bold', pad=10)

    action_data = {}
    for agent_name in agents:
        if agent_name not in data:
            continue
        action_data[agent_name] = extract_action_distribution(data[agent_name])

    # Prepare data for grouped bar chart
    action_types = ['Scale Up', 'Scale Down', 'No Change']
    x = np.arange(len(action_types))
    width = 0.35

    colors = {'hybrid_dqn_ppo': '#2E86AB', 'k8s_hpa': '#A23B72'}

    for i, agent_name in enumerate(agents):
        if agent_name not in action_data:
            continue

        actions = action_data[agent_name]
        counts = [actions['scale_up'], actions['scale_down'], actions['no_change']]
        total = sum(counts)

        if total == 0:
            continue

        # Calculate percentages
        percentages = [(count / total) * 100 for count in counts]

        label = agent_name.replace('_', ' ').title()
        offset = width * (i - 0.5)

        bars = ax.bar(x + offset, percentages, width,
                     label=label, color=colors.get(agent_name, 'gray'), alpha=0.8)

        # Add percentage labels on bars
        for j, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%\n({counts[j]:,})',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Action Type', fontweight='bold')
    ax.set_ylabel('Percentage of Total Actions', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(action_types)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)  # Extra space for labels


def plot_response_time(ax, data, agents, scenario_name):
    """
    Plot 4: Response time comparison
    Shows performance impact of scaling decisions.
    """
    ax.set_title(f'Response Time - {scenario_name.replace("_", " ").title()}',
                 fontweight='bold', pad=10)

    colors = {'hybrid_dqn_ppo': '#2E86AB', 'k8s_hpa': '#A23B72'}

    for agent_name in agents:
        if agent_name not in data:
            continue

        ts_data = extract_time_series(data[agent_name], scenario_name)
        if ts_data is None:
            continue

        label = agent_name.replace('_', ' ').title()
        color = colors.get(agent_name, 'gray')

        ax.plot(ts_data['steps'], [rt * 1000 for rt in ts_data['response_time']],
               color=color, linewidth=2, label=label, marker='d', markersize=3, alpha=0.7)

    # Add SLA threshold line (200ms)
    if ts_data:
        ax.axhline(y=200, color='red', linestyle='--', linewidth=1.5,
                  label='SLA Threshold (200ms)', alpha=0.7)

    ax.set_xlabel('Time Step', fontweight='bold')
    ax.set_ylabel('Response Time (ms)', fontweight='bold')

    # Only show legend if there are plots
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', framealpha=0.9)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)


def plot_performance_radar(ax, data, agents):
    """
    Plot 5: Radar chart for multi-metric comparison (EXCELLENT for publication)
    Shows overall agent performance across multiple dimensions.
    """
    if 'analysis' not in data or 'agent_comparison' not in data['analysis']:
        ax.text(0.5, 0.5, 'No analysis data available',
               ha='center', va='center', fontsize=12)
        return

    agent_comparison = data['analysis']['agent_comparison']

    # Define metrics (normalized to 0-1 scale, higher is better)
    metrics = ['CPU Efficiency', 'Response Time', 'Cost Efficiency',
              'Scaling Efficiency', 'SLA Compliance']

    agent_scores = {}

    for agent_name in agents:
        if agent_name not in agent_comparison:
            continue

        metrics_data = agent_comparison[agent_name]

        # Normalize metrics (0-1, higher is better)
        cpu_eff = metrics_data.get('avg_cpu_utilization', 0)
        # Normalize CPU: optimal is 60-80%, scale to 0-1
        cpu_score = 1 - abs(cpu_eff - 0.7) / 0.7

        # Response time: lower is better, invert and normalize
        response_time = metrics_data.get('avg_response_time', 0.2)
        response_score = max(0, 1 - (response_time / 0.2))

        # Cost: lower is better, normalize relative to max
        cost = metrics_data.get('total_cost', 0)

        # Scaling efficiency: higher is better
        scaling_eff = metrics_data.get('scaling_efficiency', 0)

        # SLA compliance: fewer violations is better
        sla_violations = metrics_data.get('total_sla_violations', 0)
        sla_score = max(0, 1 - (sla_violations / 2000000))  # Normalize

        agent_scores[agent_name] = [
            cpu_score,
            response_score,
            0.5,  # Placeholder for cost (will calculate relative)
            min(scaling_eff / 3000, 1),  # Normalize scaling efficiency
            sla_score
        ]

    # Calculate relative cost scores
    if len(agent_scores) > 1:
        costs = [data['analysis']['agent_comparison'][agent]['total_cost']
                for agent in agent_scores.keys()]
        max_cost = max(costs)
        for i, agent_name in enumerate(agent_scores.keys()):
            cost = data['analysis']['agent_comparison'][agent_name]['total_cost']
            agent_scores[agent_name][2] = 1 - (cost / max_cost)

    # Number of variables
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)

    colors = {'hybrid_dqn_ppo': '#2E86AB', 'k8s_hpa': '#A23B72'}

    for agent_name, scores in agent_scores.items():
        scores += scores[:1]  # Complete the circle
        label = agent_name.replace('_', ' ').title()
        color = colors.get(agent_name, 'gray')

        ax.plot(angles, scores, 'o-', linewidth=2, label=label,
               color=color, markersize=6)
        ax.fill(angles, scores, alpha=0.15, color=color)

    ax.set_title('Multi-Metric Performance Comparison',
                fontweight='bold', pad=20, fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)


def plot_cost_comparison(ax, data, agents):
    """
    Plot 6: Cost comparison bar chart
    Shows total resource cost for each agent.
    """
    if 'analysis' not in data or 'agent_comparison' not in data['analysis']:
        ax.text(0.5, 0.5, 'No analysis data available',
               ha='center', va='center', fontsize=12)
        return

    agent_comparison = data['analysis']['agent_comparison']

    ax.set_title('Total Resource Cost Comparison', fontweight='bold', pad=10)

    agent_names = []
    costs = []
    colors_list = []

    colors = {'hybrid_dqn_ppo': '#2E86AB', 'k8s_hpa': '#A23B72'}

    for agent_name in agents:
        if agent_name not in agent_comparison:
            continue

        cost = agent_comparison[agent_name].get('total_cost', 0)
        agent_names.append(agent_name.replace('_', ' ').title())
        costs.append(cost)
        colors_list.append(colors.get(agent_name, 'gray'))

    bars = ax.bar(agent_names, costs, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        # AWS equivalent cost
        aws_cost = cost * 0.00005556
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'${cost:,.0f}\n(AWS: ${aws_cost:.2f})',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Calculate savings
    if len(costs) == 2:
        savings = ((costs[1] - costs[0]) / costs[1]) * 100
        ax.text(0.5, max(costs) * 0.9,
               f'DQN-PPO saves {savings:.1f}%',
               ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
               transform=ax.transData)

    ax.set_ylabel('Total Cost (Simulation Units)', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(costs) * 1.15)


def create_comprehensive_visualization(data_file, output_dir=None):
    """
    Create comprehensive scaling behavior visualization.

    Args:
        data_file: Path to scaling decision JSON file
        output_dir: Output directory for plots (default: same as data file)
    """
    print("\n" + "="*80)
    print("📊 SCALING BEHAVIOR VISUALIZATION")
    print("="*80)

    # Load data
    data = load_data(data_file)

    # Identify agents (exclude analysis and timestamp)
    agents = [k for k in data.keys() if k not in ['analysis', 'timestamp']]
    print(f"\n✅ Found agents: {', '.join([a.replace('_', ' ').title() for a in agents])}")

    # Identify scenarios with data
    scenarios_with_data = []
    for agent in agents:
        if isinstance(data[agent], dict):
            for scenario, scenario_data in data[agent].items():
                if scenario not in ['analysis', 'timestamp'] and scenario_data:
                    if scenario not in scenarios_with_data:
                        scenarios_with_data.append(scenario)

    print(f"✅ Scenarios with data: {', '.join([s.replace('_', ' ').title() for s in scenarios_with_data])}")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Pod count timeline (first scenario with data)
    if scenarios_with_data:
        ax1 = fig.add_subplot(gs[0, :2])
        plot_pod_count_timeline(ax1, data, agents, scenarios_with_data[0])

        # Plot 2: CPU utilization
        ax2 = fig.add_subplot(gs[0, 2])
        plot_cpu_utilization(ax2, data, agents, scenarios_with_data[0])

    # Plot 3: Action distribution (MOST IMPORTANT for your question)
    ax3 = fig.add_subplot(gs[1, :2])
    plot_action_distribution_comparison(ax3, data, agents)

    # Plot 4: Response time
    if scenarios_with_data:
        ax4 = fig.add_subplot(gs[1, 2])
        plot_response_time(ax4, data, agents, scenarios_with_data[0])

    # Plot 5: Radar chart (multi-metric comparison)
    ax5 = fig.add_subplot(gs[2, 0], projection='polar')
    plot_performance_radar(ax5, data, agents)

    # Plot 6: Cost comparison
    ax6 = fig.add_subplot(gs[2, 1])
    plot_cost_comparison(ax6, data, agents)

    # Add timestamp and metadata
    timestamp = data.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
    fig.suptitle(f'Autoscaling Behavior Analysis - {timestamp}',
                fontsize=16, fontweight='bold', y=0.995)

    # Add summary text
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    summary_text = "Visualization Summary\n" + "="*30 + "\n\n"

    if 'analysis' in data and 'agent_comparison' in data['analysis']:
        for agent in agents:
            if agent in data['analysis']['agent_comparison']:
                metrics = data['analysis']['agent_comparison'][agent]
                agent_label = agent.replace('_', ' ').title()
                summary_text += f"{agent_label}:\n"
                summary_text += f"  CPU: {metrics.get('avg_cpu_utilization', 0):.1%}\n"
                summary_text += f"  Response: {metrics.get('avg_response_time', 0):.3f}s\n"
                summary_text += f"  Pods: {metrics.get('avg_pod_count', 0):.1f}\n"
                summary_text += f"  Cost: ${metrics.get('total_cost', 0):,.0f}\n"
                summary_text += f"  SLA Violations: {metrics.get('total_sla_violations', 0):,}\n\n"

    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Save figure
    if output_dir is None:
        output_dir = Path(data_file).parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"scaling_behavior_visualization_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Visualization saved: {output_file}")

    # Also save as PDF for publications
    output_pdf = output_dir / f"scaling_behavior_visualization_{timestamp}.pdf"
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"✅ PDF saved: {output_pdf}")

    plt.close()

    return str(output_file)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python visualize-scaling-behavior.py <scaling_decision.json> [output_dir]")
        print("\nExample:")
        print("  python visualize-scaling-behavior.py scaling_decision.json")
        print("  python visualize-scaling-behavior.py scaling_decision.json ./visualizations")
        sys.exit(1)

    data_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        output_file = create_comprehensive_visualization(data_file, output_dir)
        print(f"\n🎉 Visualization complete!")
        print(f"📁 Output: {output_file}")
    except Exception as e:
        print(f"\n❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
