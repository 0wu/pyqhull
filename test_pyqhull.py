import numpy as np
import pyqhull
import time
import matplotlib.pyplot as plt

def test_basic_functionality():
    # Define cube vertices
    cube_points = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
        [-1, 1, 1],
        [0, 0, 0]  # Adding an extra point inside the cube
    ], dtype=np.float64)  # changed to float64
    
    # Create batch
    batch_size = 2
    points = np.tile(cube_points[np.newaxis, :, :], (batch_size, 1, 1))
    
    # Add some noise to second batch
    points[1] += np.random.randn(*points[1].shape).astype(np.float64) * 0.1  # changed to float64
    
    # Compute convex hull
    pyqhull.set_threadpool_size(1)  # Set threadpool size for parallel processing
    result = pyqhull.convex_hull_batch(points)
    print(result)
    
    # Validate the result
    assert result is not None, "Convex hull computation failed"
    assert len(result) > 0, "Convex hull result is empty"

def benchmark_threadpool_scaling(batch_sizes, n_points, n_trials, threadpool_sizes):
    results = {tp_size: {} for tp_size in threadpool_sizes}
    for tp_size in threadpool_sizes:
        for batch_size in batch_sizes:
            times = []
            for trial in range(n_trials):
                points = np.random.randn(batch_size, n_points, 3).astype(np.float64)
                pyqhull.set_threadpool_size(tp_size)
                start_time = time.time()
                result = pyqhull.convex_hull_batch(points)
                end_time = time.time()
                elapsed = end_time - start_time
                times.append(elapsed)
                assert result.shape == (batch_size, n_points), f"Expected shape {(batch_size, n_points)}, got {result.shape}"
            times = np.array(times)
            avg_time = np.mean(times)
            std_time = np.std(times)
            std_err = std_time / np.sqrt(len(times))
            results[tp_size][batch_size] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'std_err': std_err,
                'times': times
            }
            print(f"Threadpool {tp_size:2d} | Batch size {batch_size:4d}: {avg_time:.6f} ± {std_err:.6f} seconds (stderr)")
    return results

def plot_scaling(results, batch_sizes, threadpool_sizes, filename="scaling.svg", n_points=None, n_trials=None):
    plt.figure(figsize=(8, 6))
    for tp_size in threadpool_sizes:
        avg_times = [results[tp_size][b]['avg_time'] for b in batch_sizes]
        std_errs = [results[tp_size][b]['std_err'] for b in batch_sizes]
        plt.errorbar(batch_sizes, avg_times, yerr=std_errs, marker='o', label=f"Threadpool size={tp_size}", capsize=3)
    plt.xlabel("Batch Size")
    plt.ylabel("Average Time (s)")
    title = "Convex Hull Batch Timing vs Batch Size"
    if n_points is not None and n_trials is not None:
        title += f"\n(n_points={n_points}, n_trials={n_trials})"
    plt.title(title)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

def test_parallel_scaling():
    print("\n=== Parallel Scaling Test ===")
    n_points = 24
    batch_sizes = [2**i for i in range(0, 13)]
    n_trials = 10
    threadpool_sizes = [1, 4, 16, 64]
    results = benchmark_threadpool_scaling(batch_sizes, n_points, n_trials, threadpool_sizes)
    print("\n=== Summary ===")
    print("Threadpool | Batch Size | Avg Time (s) | Stderr Time (s)")
    print("-" * 55)
    for tp_size in threadpool_sizes:
        for batch_size in batch_sizes:
            avg_time = results[tp_size][batch_size]['avg_time']
            std_err = results[tp_size][batch_size]['std_err']
            print(f"{tp_size:9d} | {batch_size:9d} | {avg_time:11.6f} | {std_err:14.6f}")
    plot_scaling(results, batch_sizes, threadpool_sizes, n_points=n_points, n_trials=n_trials)
    return results

if __name__ == "__main__":
    test_basic_functionality()
    test_parallel_scaling()
    print("All tests passed!")