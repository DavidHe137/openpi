#!/usr/bin/env python3
"""
Script to verify roofline calculations for model inference.

Roofline formula for BF16 GEMM (N×K×M):
    t_roofline = max(2*K*M/T_bandwidth, N*K*M/T_compute)

Where:
- 2*K*M accounts for reading the weight matrix (2 bytes per BF16 element)
- N*K*M is the number of MAC operations
- T_bandwidth = memory bandwidth in bytes/s
- T_compute = compute throughput in MAC/s
"""

import pandas as pd
import numpy as np
import argparse

BYTES_PER_BF16 = 2

# Global variables that will be set by command-line args
T_BANDWIDTH = None
T_COMPUTE = None

def calculate_roofline(N, K, M, bandwidth, compute, is_attention=False):
    """
    Calculate roofline lower bound for a GEMM operation.

    Args:
        N, K, M: Matrix dimensions (N×K) @ (K×M) = (N×M)
        bandwidth: Memory bandwidth in bytes/s
        compute: Compute throughput in MAC/s
        is_attention: If True, use attention-specific calculation

    Returns:
        Roofline time in milliseconds
    """
    if is_attention:
        # For attention: heads × seq_len² × head_dim
        # The key operations are Q@K^T (produces seq×seq) and (attn)@V
        # Memory: read Q (seq×head_dim), K (seq×head_dim), V (seq×head_dim)
        # Compute: 2 * seq² * head_dim per head
        heads = N
        seq_len_squared = K  # This is already seq²
        head_dim = M
        seq_len = int(seq_len_squared ** 0.5)

        # Memory: read Q, K, V matrices (3 * seq * head_dim per head)
        memory_bytes = heads * 3 * seq_len * head_dim * BYTES_PER_BF16
        memory_time = memory_bytes / bandwidth

        # Compute: Q@K^T and attn@V (2 * seq² * head_dim per head)
        mac_ops = heads * 2 * seq_len * seq_len * head_dim
        compute_time = mac_ops / compute

        roofline_seconds = max(memory_time, compute_time)
    else:
        # Memory-bound time: reading K×M weight matrix
        memory_time = (2 * K * M) / bandwidth

        # Compute-bound time: N*K*M MAC operations
        compute_time = (N * K * M) / compute

        # Roofline is the max of the two bottlenecks
        roofline_seconds = max(memory_time, compute_time)

    return roofline_seconds * 1000  # Convert to milliseconds

def main(csv_path, bandwidth, compute):
    # Read the CSV data
    df = pd.read_csv(csv_path)

    # Calculate roofline for each single operation
    df['calculated_single_op_ms'] = df.apply(
        lambda row: calculate_roofline(row['N'], row['K'], row['M'],
                                       bandwidth, compute,
                                       is_attention=(row['shape_type'] == 'attn')),
        axis=1
    )

    # Calculate total time (multiply by times)
    # Note: roofline_ms in the table is already total time (times * single_op)
    df['calculated_total_ms'] = df['calculated_single_op_ms'] * df['times']

    # Calculate the difference
    df['diff_ms'] = df['calculated_total_ms'] - df['roofline_ms']
    df['diff_percent'] = (df['diff_ms'] / df['roofline_ms'] * 100).abs()

    # Display results
    print("=" * 100)
    print("Roofline Verification")
    print("=" * 100)
    print(f"\nHardware specs:")
    print(f"  Bandwidth: {bandwidth/1e12:.2f} TB/s")
    print(f"  Compute:   {compute/1e12:.1f} TMAC/s")
    print("\n")

    # Show per-operation comparison
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)

    display_cols = ['component', 'times', 'N', 'K', 'M',
                    'calculated_single_op_ms', 'roofline_ms', 'calculated_total_ms', 'diff_ms', 'diff_percent']
    print(df[display_cols].to_string(index=False))

    # Summary by component
    print("\n" + "=" * 100)
    print("Summary by Component")
    print("=" * 100)

    summary = df.groupby('component').agg({
        'roofline_ms': 'sum',
        'calculated_total_ms': 'sum'
    }).reset_index()

    summary['diff_ms'] = summary['calculated_total_ms'] - summary['roofline_ms']
    summary['diff_percent'] = (summary['diff_ms'] / summary['roofline_ms'] * 100).abs()

    print(summary.to_string(index=False))

    # Overall total
    print("\n" + "=" * 100)
    print("Overall Total")
    print("=" * 100)
    total_roofline = summary['roofline_ms'].sum()
    total_calculated = summary['calculated_total_ms'].sum()
    total_diff = total_calculated - total_roofline
    total_diff_pct = abs(total_diff / total_roofline * 100)

    print(f"  Table roofline:      {total_roofline:.3f} ms")
    print(f"  Calculated roofline: {total_calculated:.3f} ms")
    print(f"  Difference:          {total_diff:.3f} ms ({total_diff_pct:.2f}%)")

    # Check which operations are memory-bound vs compute-bound
    print("\n" + "=" * 100)
    print("Bottleneck Analysis")
    print("=" * 100)

    def calc_memory_time(row):
        if row['shape_type'] == 'attn':
            heads = row['N']
            seq_len = int(row['K'] ** 0.5)
            head_dim = row['M']
            return heads * 3 * seq_len * head_dim * BYTES_PER_BF16 / bandwidth * 1000
        else:
            return (2 * row['K'] * row['M']) / bandwidth * 1000

    def calc_compute_time(row):
        if row['shape_type'] == 'attn':
            heads = row['N']
            seq_len = int(row['K'] ** 0.5)
            head_dim = row['M']
            return heads * 2 * seq_len * seq_len * head_dim / compute * 1000
        else:
            return (row['N'] * row['K'] * row['M']) / compute * 1000

    df['memory_time_ms'] = df.apply(calc_memory_time, axis=1)
    df['compute_time_ms'] = df.apply(calc_compute_time, axis=1)
    df['bottleneck'] = df.apply(
        lambda row: 'memory' if row['memory_time_ms'] > row['compute_time_ms'] else 'compute',
        axis=1
    )

    bottleneck_cols = ['component', 'shape_type', 'times', 'N', 'K', 'M',
                       'memory_time_ms', 'compute_time_ms', 'bottleneck']
    print(df[bottleneck_cols].to_string(index=False))

    print("\n" + "=" * 100)
    memory_bound = (df['bottleneck'] == 'memory').sum()
    compute_bound = (df['bottleneck'] == 'compute').sum()
    print(f"Memory-bound operations: {memory_bound}/{len(df)}")
    print(f"Compute-bound operations: {compute_bound}/{len(df)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Verify roofline calculations for model inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default RTX 4090 specs
  python verify_roofline.py

  # Specify custom hardware (H100: 3.35 TB/s, 989.4 TFLOP/s BF16)
  python verify_roofline.py --bandwidth 3.35 --compute 494.7

  # A100 specs (1.55 TB/s, 312 TFLOP/s BF16)
  python verify_roofline.py --bandwidth 1.55 --compute 156
        """
    )
    parser.add_argument(
        '--bandwidth', '-b',
        type=float,
        default=1.01,
        help='Memory bandwidth in TB/s (default: 1.01 for RTX 4090)'
    )
    parser.add_argument(
        '--compute', '-c',
        type=float,
        default=91.4,
        help='Compute throughput in TMAC/s for BF16 (default: 91.4 for RTX 4090 boosted)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='roofline_data.csv',
        help='Path to CSV file with roofline data (default: roofline_data.csv)'
    )

    args = parser.parse_args()

    # Set global variables
    T_BANDWIDTH = args.bandwidth * 1e12  # Convert TB/s to bytes/s
    T_COMPUTE = args.compute * 1e12      # Convert TMAC/s to MAC/s

    main(args.csv, T_BANDWIDTH, T_COMPUTE)
