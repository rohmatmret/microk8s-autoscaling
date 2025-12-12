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
            # Preserve top-level analysis and timestamp if they exist
            result = data['results'].copy()
            if 'analysis' in data and 'analysis' not in result:
                result['analysis'] = data['analysis']
            if 'timestamp' in data and 'timestamp' not in result:
                result['timestamp'] = data['timestamp']
            return result
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


def extract_action_distribution_by_scenario(agent_data, scenarios):
    """
    Extract action distribution per scenario.

    Returns:
        dict: {scenario_name: {'scale_up': count, 'scale_down': count, 'no_change': count}}
    """
    scenario_actions = {}

    for scenario_name in scenarios:
        if scenario_name not in agent_data or not agent_data[scenario_name]:
            continue

        scenario_data = agent_data[scenario_name]

        if len(scenario_data) > 0:
            last_step = scenario_data[-1]
            if 'action_distribution' in last_step:
                scenario_actions[scenario_name] = last_step['action_distribution'].copy()
            else:
                scenario_actions[scenario_name] = {'scale_up': 0, 'scale_down': 0, 'no_change': 0}

    return scenario_actions


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


def plot_pod_count_timeline_all_scenarios(axes, data, agents, scenarios):
    """
    Plot pod count timelines for all scenarios in a grid layout.
    Shows agent behavior across different workload patterns.

    Args:
        axes: Array of matplotlib axes (one per scenario)
        data: Full data dictionary
        agents: List of agent names
        scenarios: List of scenario names
    """
    colors = {'hybrid_dqn_ppo': '#2E86AB', 'k8s_hpa': '#A23B72'}
    styles = {'hybrid_dqn_ppo': '-', 'k8s_hpa': '--'}

    for idx, scenario_name in enumerate(scenarios):
        ax = axes[idx] if len(scenarios) > 1 else axes

        ax.set_title(f'{scenario_name.replace("_", " ").title()}',
                     fontweight='bold', pad=8, fontsize=10)

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
                   color=color, linestyle=styles[agent_name], linewidth=2,
                   label=f'{label}', marker='o', markersize=2, alpha=0.8)

        ax.set_xlabel('Time Step', fontsize=9)
        ax.set_ylabel('Pod Count', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)

        # Only show legend on first subplot
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='best', framealpha=0.9, fontsize=8)


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


def plot_action_distribution_by_scenario(ax, data, agents, scenarios):
    """
    Plot action distribution grouped by scenario.
    Shows scaling decision patterns across different workload types.

    Creates a grouped bar chart where:
    - X-axis: Scenarios (baseline_steady, gradual_ramp, etc.)
    - Y-axis: Action count
    - Bars: Grouped by agent (different colors)
    - Stacked: Actions types (scale_up, scale_down, no_change)
    """
    ax.set_title('Scaling Actions by Scenario', fontweight='bold', pad=10)

    colors = {'hybrid_dqn_ppo': '#2E86AB', 'k8s_hpa': '#A23B72'}
    action_colors = {'scale_up': '#4CAF50', 'scale_down': '#FF6B6B', 'no_change': '#FFA726'}

    # Extract data for all agents and scenarios
    agent_scenario_data = {}
    for agent_name in agents:
        if agent_name not in data:
            continue
        agent_scenario_data[agent_name] = extract_action_distribution_by_scenario(
            data[agent_name], scenarios
        )

    # Filter scenarios that have data
    scenarios_with_data = [s for s in scenarios
                          if any(s in agent_scenario_data.get(agent, {})
                                for agent in agents)]

    if not scenarios_with_data:
        ax.text(0.5, 0.5, 'No action distribution data available',
               ha='center', va='center', fontsize=12)
        return

    # Prepare data for grouped bar chart
    x = np.arange(len(scenarios_with_data))
    width = 0.35
    num_agents = len(agents)

    for agent_idx, agent_name in enumerate(agents):
        if agent_name not in agent_scenario_data:
            continue

        label = agent_name.replace('_', ' ').title()
        offset = width * (agent_idx - (num_agents - 1) / 2)

        # Prepare data for this agent across scenarios
        scale_up_counts = []
        scale_down_counts = []
        no_change_counts = []

        for scenario in scenarios_with_data:
            actions = agent_scenario_data[agent_name].get(scenario,
                                                          {'scale_up': 0, 'scale_down': 0, 'no_change': 0})
            scale_up_counts.append(actions.get('scale_up', 0))
            scale_down_counts.append(actions.get('scale_down', 0))
            no_change_counts.append(actions.get('no_change', 0))

        # Create stacked bars
        agent_color = colors.get(agent_name, 'gray')

        # Stack: scale_up at bottom, scale_down in middle, no_change on top
        p1 = ax.bar(x + offset, scale_up_counts, width,
                   label=f'{label} - Scale Up' if agent_idx == 0 else '',
                   color=action_colors['scale_up'], alpha=0.8, edgecolor='black', linewidth=0.5)

        p2 = ax.bar(x + offset, scale_down_counts, width, bottom=scale_up_counts,
                   label=f'{label} - Scale Down' if agent_idx == 0 else '',
                   color=action_colors['scale_down'], alpha=0.8, edgecolor='black', linewidth=0.5)

        bottom_for_no_change = [up + down for up, down in zip(scale_up_counts, scale_down_counts)]
        p3 = ax.bar(x + offset, no_change_counts, width, bottom=bottom_for_no_change,
                   label=f'{label} - No Change' if agent_idx == 0 else '',
                   color=action_colors['no_change'], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Scenario', fontweight='bold')
    ax.set_ylabel('Action Count', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios_with_data],
                       rotation=15, ha='right', fontsize=8)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=7, ncol=2)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')


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

    # Create figure with new layout to accommodate all scenarios
    num_scenarios = len(scenarios_with_data)
    fig_height = 16 if num_scenarios >= 5 else 14

    fig = plt.figure(figsize=(22, fig_height))

    # Create grid layout:
    # - Top section (rows 0-1): Pod count timelines for all scenarios
    # - Middle section (row 2): Action distribution by scenario
    # - Bottom section (row 3): Radar chart, Cost comparison, Summary
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3,
                          height_ratios=[2, 2, 1.5, 1])

    # Plot pod count timelines for all scenarios (arranged in grid)
    if scenarios_with_data:
        # Arrange scenarios in a grid (up to 3 per row)
        num_cols = min(3, num_scenarios)
        num_rows = (num_scenarios + num_cols - 1) // num_cols  # Ceiling division

        for idx, scenario in enumerate(scenarios_with_data):
            row = idx // num_cols
            col = idx % num_cols

            if row == 0:
                ax_pod = fig.add_subplot(gs[row, col])
            else:
                ax_pod = fig.add_subplot(gs[row, col])

            plot_pod_count_timeline(ax_pod, data, agents, scenario)

    # Plot action distribution by scenario (spans full width of row 2)
    ax_action = fig.add_subplot(gs[2, :])
    plot_action_distribution_by_scenario(ax_action, data, agents, scenarios_with_data)

    # Plot radar chart (bottom left)
    ax_radar = fig.add_subplot(gs[3, 0], projection='polar')
    plot_performance_radar(ax_radar, data, agents)

    # Plot cost comparison (bottom middle)
    ax_cost = fig.add_subplot(gs[3, 1])
    plot_cost_comparison(ax_cost, data, agents)

    # Add timestamp and metadata
    timestamp = data.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
    fig.suptitle(f'Autoscaling Behavior Analysis (Grouped by Scenario) - {timestamp}',
                fontsize=16, fontweight='bold', y=0.995)

    # Add summary text (bottom right)
    ax_summary = fig.add_subplot(gs[3, 2])
    ax_summary.axis('off')

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

    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
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
