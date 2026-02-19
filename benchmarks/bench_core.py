"""Benchmark core monitoring operations."""

import time

import numpy as np
import pandas as pd

from data_pipeline_monitor import DriftDetector, MonitoringConfig, PipelineMonitor

# Setup
np.random.seed(42)
baseline = pd.DataFrame({
    "f1": np.random.normal(100, 15, 1000),
    "f2": np.random.normal(50, 5, 1000),
    "f3": np.random.choice(["A", "B", "C"], 1000),
})
current = pd.DataFrame({
    "f1": np.random.normal(100, 15, 500),
    "f2": np.random.normal(50, 5, 500),
    "f3": np.random.choice(["A", "B", "C"], 500),
})

print("╔════════════════════════════════════════════════╗")
print("║     Data Pipeline Monitor Benchmarks           ║")
print("╚════════════════════════════════════════════════╝\n")

# Benchmark 1: Drift detection
print("1. Numerical Drift Detection (500 samples)")
detector = DriftDetector()
start = time.time()
for _ in range(100):
    detector.detect_numerical_drift(baseline["f1"], current["f1"], "f1")
elapsed = time.time() - start
avg_ms = (elapsed / 100) * 1000
print(f"   Avg: {avg_ms:.2f}ms | Total: {elapsed:.2f}s for 100 iterations\n")

# Benchmark 2: Data quality check
print("2. Data Quality Check (500 samples, 3 features)")
config = MonitoringConfig(pipeline_name="bench", min_baseline_samples=50)
monitor = PipelineMonitor(config)
monitor.set_baseline(baseline)

start = time.time()
for _ in range(100):
    monitor.check_data_quality(current)
elapsed = time.time() - start
avg_ms = (elapsed / 100) * 1000
print(f"   Avg: {avg_ms:.2f}ms | Total: {elapsed:.2f}s for 100 iterations\n")

# Benchmark 3: Multivariate drift detection
print("3. Multivariate Drift Detection (500 samples, 3 features)")
start = time.time()
for _ in range(50):
    monitor.detect_feature_drift(current)
elapsed = time.time() - start
avg_ms = (elapsed / 50) * 1000
print(f"   Avg: {avg_ms:.2f}ms | Total: {elapsed:.2f}s for 50 iterations\n")

# Benchmark 4: Large batch processing
print("4. Large Batch Processing (10K samples, 5 features)")
large_baseline = pd.DataFrame({
    f"feature_{i}": np.random.normal(100, 15, 5000)
    for i in range(5)
})
large_current = pd.DataFrame({
    f"feature_{i}": np.random.normal(100, 15, 10000)
    for i in range(5)
})

config_large = MonitoringConfig(
    pipeline_name="large",
    min_baseline_samples=100,
    features_to_monitor=[f"feature_{i}" for i in range(5)],
)
monitor_large = PipelineMonitor(config_large)
monitor_large.set_baseline(large_baseline)

start = time.time()
for _ in range(10):
    monitor_large.check_data_quality(large_current)
    monitor_large.detect_feature_drift(large_current)
elapsed = time.time() - start
avg_ms = (elapsed / 10) * 1000
print(f"   Avg: {avg_ms:.2f}ms | Total: {elapsed:.2f}s for 10 iterations\n")

print("✓ Benchmarks complete!")
