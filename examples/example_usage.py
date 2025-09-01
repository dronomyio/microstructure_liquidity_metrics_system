#!/usr/bin/env python3

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import sys
sys.path.append('../src/python')

from liquidity_api import MetricsRequest
from polygon_client import PolygonClient
import liquidity_metrics

async def example_calculation():
    """
    Example of calculating liquidity metrics for a stock.
    """
    
    # Initialize Polygon client
    async with PolygonClient() as client:
        # Fetch data for Apple
        symbol = "AAPL"
        date = "2024-01-15"
        
        print(f"Fetching data for {symbol} on {date}...")
        trades = await client.get_trades(symbol, date)
        quotes = await client.get_quotes(symbol, date)
        
        print(f"Retrieved {len(trades)} trades and {len(quotes)} quotes")
        
        # Convert to numpy arrays for C++ processing
        trades_array = trades.to_records(index=False)
        quotes_array = quotes.to_records(index=False)
        
        # Calculate metrics using GPU
        print("\nCalculating metrics using GPU...")
        gpu_results = liquidity_metrics.calculate_gpu(
            trades_array,
            quotes_array,
            ["amihud", "duration", "hazard", "holes", "spread"]
        )
        
        # Calculate metrics using CPU (SIMD)
        print("Calculating metrics using CPU (SIMD)...")
        cpu_results = liquidity_metrics.calculate_cpu(
            trades_array,
            quotes_array,
            ["amihud", "duration", "hazard", "holes", "spread"]
        )
        
        # Compare results
        print("\n=== Results Comparison ===")
        for metric in gpu_results:
            gpu_val = np.mean(gpu_results[metric])
            cpu_val = np.mean(cpu_results[metric])
            diff = abs(gpu_val - cpu_val) / cpu_val * 100
            print(f"{metric}:")
            print(f"  GPU: {gpu_val:.6f}")
            print(f"  CPU: {cpu_val:.6f}")
            print(f"  Difference: {diff:.2f}%")
        
        # Visualize results
        visualize_metrics(gpu_results)

def visualize_metrics(results):
    """
    Create visualizations of the liquidity metrics.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Liquidity Metrics Analysis', fontsize=16)
    
    # Amihud illiquidity over time
    if 'amihud' in results:
        ax = axes[0, 0]
        ax.plot(results['amihud'], alpha=0.7)
        ax.set_title('Amihud Illiquidity')
        ax.set_xlabel('Time')
        ax.set_ylabel('Illiquidity')
        ax.grid(True, alpha=0.3)
    
    # Duration distribution
    if 'duration' in results:
        ax = axes[0, 1]
        ax.hist(results['duration'], bins=50, alpha=0.7, edgecolor='black')
        ax.set_title('Inter-trade Duration Distribution')
        ax.set_xlabel('Duration (ns)')
        ax.set_ylabel('Frequency')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # Hazard rates
    if 'hazard' in results:
        ax = axes[0, 2]
        ax.plot(results['hazard'], alpha=0.7, color='orange')
        ax.set_title('Hazard Rates')
        ax.set_xlabel('Time')
        ax.set_ylabel('Hazard Rate')
        ax.grid(True, alpha=0.3)
    
    # Liquidity holes
    if 'holes' in results:
        ax = axes[1, 0]
        holes_binary = (np.array(results['holes']) > 0).astype(int)
        ax.scatter(range(len(holes_binary)), holes_binary, alpha=0.5, s=1)
        ax.set_title('Liquidity Holes')
        ax.set_xlabel('Time')
        ax.set_ylabel('Hole Indicator')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
    
    # Bid-ask spread
    if 'spread' in results:
        ax = axes[1, 1]
        ax.plot(results['spread'], alpha=0.7, color='green')
        ax.set_title('Bid-Ask Spread')
        ax.set_xlabel('Time')
        ax.set_ylabel('Spread ($)')
        ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    stats_text = "Summary Statistics\n" + "="*20 + "\n"
    for metric, values in results.items():
        arr = np.array(values)
        stats_text += f"\n{metric.upper()}:\n"
        stats_text += f"  Mean: {np.mean(arr):.6f}\n"
        stats_text += f"  Std:  {np.std(arr):.6f}\n"
        stats_text += f"  Min:  {np.min(arr):.6f}\n"
        stats_text += f"  Max:  {np.max(arr):.6f}\n"
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('liquidity_metrics.png', dpi=150)
    plt.show()

async def benchmark_performance():
    """
    Benchmark the performance difference between CPU and GPU.
    """
    print("\n=== Performance Benchmark ===")
    
    sizes = [10_000, 100_000, 1_000_000, 10_000_000]
    cpu_times = []
    gpu_times = []
    
    for size in sizes:
        print(f"\nTesting with {size:,} trades...")
        
        # Generate synthetic data
        trades = np.zeros(size, dtype=[
            ('timestamp_ns', 'i8'),
            ('price', 'f8'),
            ('volume', 'f8'),
            ('conditions', 'i4'),
            ('exchange', 'S1')
        ])
        
        trades['timestamp_ns'] = np.arange(size) * 1000000  # 1ms intervals
        trades['price'] = 100 + np.random.randn(size) * 0.1
        trades['volume'] = np.random.exponential(1000, size)
        
        # Benchmark CPU
        import time
        start = time.perf_counter()
        cpu_result = liquidity_metrics.calculate_cpu(trades, None, ["amihud"])
        cpu_time = time.perf_counter() - start
        cpu_times.append(cpu_time)
        
        # Benchmark GPU
        start = time.perf_counter()
        gpu_result = liquidity_metrics.calculate_gpu(trades, None, ["amihud"])
        gpu_time = time.perf_counter() - start
        gpu_times.append(gpu_time)
        
        speedup = cpu_time / gpu_time
        print(f"  CPU Time: {cpu_time:.4f}s")
        print(f"  GPU Time: {gpu_time:.4f}s")
        print(f"  Speedup:  {speedup:.2f}x")
    
    # Plot performance comparison
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.loglog(sizes, cpu_times, 'o-', label='CPU (SIMD)', linewidth=2)
    plt.loglog(sizes, gpu_times, 's-', label='GPU (CUDA)', linewidth=2)
    plt.xlabel('Number of Trades')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Processing Time Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    speedups = [c/g for c, g in zip(cpu_times, gpu_times)]
    plt.semilogx(sizes, speedups, 'o-', color='green', linewidth=2)
    plt.xlabel('Number of Trades')
    plt.ylabel('Speedup Factor')
    plt.title('GPU Speedup over CPU')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('performance_benchmark.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    # Run examples
    asyncio.run(example_calculation())
    asyncio.run(benchmark_performance())
