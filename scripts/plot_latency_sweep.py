#!/usr/bin/env python3
"""
Plot number of successes vs latency for each task in the latency sweep experiment.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define the base directory containing the latency sweep data
BASE_DIR = Path("data/libero/latency_sweep_rtc_off")

# Define latency values to analyze
LATENCIES = [0.0, 10.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0]

def find_results_file(latency):
    """Find the results_summary.csv file for a given latency value."""
    pattern = f"*rtcFalse_lat{latency}_hrzn10_parallel"
    matching_dirs = list(BASE_DIR.glob(pattern))
    
    if not matching_dirs:
        print(f"Warning: No directory found for latency {latency}")
        return None
    
    results_file = matching_dirs[0] / "horizon_10" / "results_summary.csv"
    
    if not results_file.exists():
        print(f"Warning: Results file not found at {results_file}")
        return None
    
    return results_file

def load_all_data():
    """Load all results_summary.csv files and combine into a single dataframe."""
    all_data = []
    
    for latency in LATENCIES:
        results_file = find_results_file(latency)
        
        if results_file is None:
            continue
        
        df = pd.read_csv(results_file)
        # Filter out the OVERALL row
        df = df[df['task_id'] != 'OVERALL']
        # Convert task_id to int
        df['task_id'] = df['task_id'].astype(int)
        all_data.append(df)
    
    if not all_data:
        raise ValueError("No data found!")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def plot_successes_vs_latency(df, output_path=None):
    """Create a line plot of successes vs latency for each task."""
    # Set up the plot with a nice style
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # Get unique tasks
    tasks = sorted(df['task_id'].unique())
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))
    
    # Plot each task
    for i, task_id in enumerate(tasks):
        task_data = df[df['task_id'] == task_id].sort_values('latency_ms')
        
        # Get task description (truncate if too long)
        task_desc = task_data.iloc[0]['task_description']
        if len(task_desc) > 50:
            task_desc = task_desc[:47] + "..."
        
        label = f"Task {task_id}: {task_desc}"
        
        plt.plot(task_data['latency_ms'], 
                task_data['successes'], 
                marker='o', 
                linewidth=2,
                markersize=8,
                color=colors[i],
                label=label,
                alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Latency (ms)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Successes (out of 20)', fontsize=14, fontweight='bold')
    plt.title('Task Success vs Latency', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Set y-axis limits and ticks
    plt.ylim(-1, 21)
    plt.yticks(range(0, 21, 2))
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend (below the plot area)
    plt.legend(bbox_to_anchor=(0.5, -0.15), 
              loc='upper center', 
              fontsize=9,
              framealpha=0.9,
              ncol=2)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def print_summary_statistics(df):
    """Print summary statistics for the data."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for task_id in sorted(df['task_id'].unique()):
        task_data = df[df['task_id'] == task_id].sort_values('latency_ms')
        task_desc = task_data.iloc[0]['task_description']
        
        print(f"\nTask {task_id}: {task_desc}")
        print(f"  Latency range: {task_data['latency_ms'].min():.0f} - {task_data['latency_ms'].max():.0f} ms")
        print(f"  Success range: {task_data['successes'].min()} - {task_data['successes'].max()} (out of 20)")
        print(f"  Success rate range: {task_data['success_rate'].min():.2%} - {task_data['success_rate'].max():.2%}")
        
        # Calculate degradation
        if len(task_data) > 1:
            initial_success = task_data.iloc[0]['successes']
            final_success = task_data.iloc[-1]['successes']
            degradation = initial_success - final_success
            print(f"  Degradation (0ms → 400ms): {degradation} successes")
    
    print("\n" + "="*80)

def main():
    """Main function to load data and create plots."""
    print("Loading data from latency sweep experiments...")
    
    try:
        df = load_all_data()
        print(f"Successfully loaded data for {len(df)} task-latency combinations")
        print(f"Unique latencies: {sorted(df['latency_ms'].unique())}")
        print(f"Unique tasks: {sorted(df['task_id'].unique())}")
        
        # Print summary statistics
        print_summary_statistics(df)
        
        # Create the plot
        output_dir = Path("scripts")
        output_path = output_dir / "latency_sweep_successes_plot.png"
        
        print(f"\nCreating plot...")
        plot_successes_vs_latency(df, output_path)
        
        # Also create a PDF version
        output_path_pdf = output_dir / "latency_sweep_successes_plot.pdf"
        plot_successes_vs_latency(df, output_path_pdf)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

