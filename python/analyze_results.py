#!/usr/bin/env python3
"""
SpMV Performance Analysis Script

Reads JSON results from test_runner and generates visualizations and reports.
"""

import json
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_results(json_file):
    """Load performance results from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['results']

def create_dataframe(results):
    """Convert results to pandas DataFrame for analysis."""
    rows = []
    for r in results:
        row = {
            'rows': r['matrix']['rows'],
            'cols': r['matrix']['cols'],
            'nnz': r['matrix']['nnz'],
            'sparsity': r['matrix']['sparsity'],
            'device_id': r['device']['id'],
            'warp_size': r['device']['warpSize'],
            'peak_bw': r['device']['peakBandwidthGBs'],
            'time_ms': r['performance']['timeMs'],
            'gflops': r['performance']['gflops'],
            'bandwidth_gbs': r['performance']['bandwidthGBs'],
            'bw_utilization': r['performance']['bandwidthUtilization'],
            'correctness': r['correctnessPassed']
        }
        rows.append(row)
    return pd.DataFrame(rows)

def plot_bandwidth_vs_sparsity(df, output_dir):
    """Plot bandwidth utilization vs sparsity."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by warp size (GPU type)
    for warp_size in df['warp_size'].unique():
        subset = df[df['warp_size'] == warp_size]
        ax.scatter(subset['sparsity'] * 100, subset['bw_utilization'],
                   label=f'Warp={warp_size}', s=100, alpha=0.7)

    ax.set_xlabel('Sparsity (%)')
    ax.set_ylabel('Bandwidth Utilization (%)')
    ax.set_title('SpMV Bandwidth Utilization vs Matrix Sparsity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bandwidth_vs_sparsity.png'), dpi=150)
    plt.close()

def plot_gflops_vs_nnz(df, output_dir):
    """Plot GFLOPS vs number of non-zeros."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for warp_size in df['warp_size'].unique():
        subset = df[df['warp_size'] == warp_size]
        ax.scatter(subset['nnz'], subset['gflops'],
                   label=f'Warp={warp_size}', s=100, alpha=0.7)

    ax.set_xlabel('Number of Non-Zeros')
    ax.set_ylabel('GFLOPS')
    ax.set_title('SpMV Performance vs Matrix Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gflops_vs_nnz.png'), dpi=150)
    plt.close()

def plot_time_breakdown(df, output_dir):
    """Plot time distribution across tests."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time vs NNZ
    ax1 = axes[0]
    for warp_size in df['warp_size'].unique():
        subset = df[df['warp_size'] == warp_size]
        ax1.scatter(subset['nnz'], subset['time_ms'],
                    label=f'Warp={warp_size}', s=100, alpha=0.7)
    ax1.set_xlabel('Number of Non-Zeros')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Execution Time vs Matrix Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Bar chart for summary
    ax2 = axes[1]
    summary = df.groupby('warp_size').agg({
        'gflops': 'mean',
        'bw_utilization': 'mean',
        'time_ms': 'mean'
    }).round(2)

    x = np.arange(len(summary))
    width = 0.25

    ax2.bar(x - width, summary['gflops'], width, label='GFLOPS')
    ax2.bar(x, summary['bw_utilization'], width, label='BW Util (%)')
    ax2.bar(x + width, summary['time_ms'], width, label='Time (ms)')

    ax2.set_xlabel('Warp Size')
    ax2.set_ylabel('Value')
    ax2.set_title('Average Performance by GPU Type')
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary.index)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_breakdown.png'), dpi=150)
    plt.close()

def generate_report(df, output_dir):
    """Generate text summary report."""
    report_file = os.path.join(output_dir, 'performance_report.txt')

    with open(report_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SpMV Performance Analysis Report\n")
        f.write("=" * 60 + "\n\n")

        f.write("Test Summary:\n")
        f.write(f"  Total tests: {len(df)}\n")
        f.write(f"  All correctness checks passed: {df['correctness'].all()}\n\n")

        for warp_size in df['warp_size'].unique():
            subset = df[df['warp_size'] == warp_size]
            gpu_type = "Domestic (Mars X201)" if warp_size == 64 else "NVIDIA (RTX 4090)"

            f.write(f"\n--- GPU: {gpu_type} (Warp={warp_size}) ---\n")
            f.write(f"  Tests: {len(subset)}\n")
            f.write(f"  Average GFLOPS: {subset['gflops'].mean():.2f}\n")
            f.write(f"  Average Bandwidth Util: {subset['bw_utilization'].mean():.1f}%\n")
            f.write(f"  Peak Bandwidth: {subset['peak_bw'].mean():.2f} GB/s\n")
            f.write(f"  Best GFLOPS: {subset['gflops'].max():.2f}\n")
            f.write(f"  Best BW Util: {subset['bw_utilization'].max():.1f}%\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Detailed Results:\n")
        f.write("=" * 60 + "\n\n")

        for _, row in df.iterrows():
            f.write(f"Matrix: {row['rows']}x{row['cols']}, NNZ={row['nnz']}\n")
            f.write(f"  Sparsity: {row['sparsity']*100:.4f}%\n")
            f.write(f"  Time: {row['time_ms']:.4f} ms\n")
            f.write(f"  GFLOPS: {row['gflops']:.2f}\n")
            f.write(f"  Bandwidth: {row['bandwidth_gbs']:.2f} GB/s ({row['bw_utilization']:.1f}%)\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description='SpMV Performance Analysis')
    parser.add_argument('results_file', help='JSON results file from test_runner')
    parser.add_argument('--output-dir', default='analysis_output',
                        help='Output directory for plots and report')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)

    print(f"Found {len(results)} test results")

    # Create DataFrame
    df = create_dataframe(results)

    print("\nSummary statistics:")
    print(df.describe())

    # Generate plots
    print("\nGenerating visualizations...")
    plot_bandwidth_vs_sparsity(df, args.output_dir)
    plot_gflops_vs_nnz(df, args.output_dir)
    plot_time_breakdown(df, args.output_dir)

    # Generate report
    print("Generating report...")
    generate_report(df, args.output_dir)

    print(f"\nOutput saved to: {args.output_dir}")
    print("Files generated:")
    print(f"  - bandwidth_vs_sparsity.png")
    print(f"  - gflops_vs_nnz.png")
    print(f"  - time_breakdown.png")
    print(f"  - performance_report.txt")

    # Save DataFrame to CSV
    csv_file = os.path.join(args.output_dir, 'results.csv')
    df.to_csv(csv_file, index=False)
    print(f"  - results.csv")

if __name__ == '__main__':
    main()