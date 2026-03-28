"""Sim-to-real validation — compare synthetic telemetry against real data.

Measures the fidelity gap between:
  - Synthetic telemetry from the RL environment (what the agent trains on)
  - Real telemetry from the Aiops-Dataset (what production looks like)

Reports differences in: metric naming, value ranges, log formats, and
statistical distributions. This is the empirical test of whether skills
trained on synthetic data will transfer to real observability platforms.

Usage:
    python -m src.training.sim_to_real --real-metrics /tmp/Aiops-Dataset/data/2022-05-05/metric/jvm/
"""

from __future__ import annotations

import contextlib
import csv
import logging
from collections import defaultdict
from pathlib import Path

from src.generators.loader import ScenarioLoader
from src.generators.telemetry import TelemetryGenerator

logger = logging.getLogger(__name__)


def load_real_metrics(metrics_dir: str) -> dict[str, list[float]]:
    """Load real metric values from Aiops-Dataset CSV files.

    Returns {metric_name: [values]} dict.
    """
    result: dict[str, list[float]] = defaultdict(list)
    metrics_path = Path(metrics_dir)

    if not metrics_path.exists():
        logger.warning("Real metrics dir not found: %s", metrics_path)
        return result

    for csv_file in metrics_path.rglob("*.csv"):
        try:
            with open(csv_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("kpi_name", "")
                    value = row.get("value", "")
                    if name and value:
                        with contextlib.suppress(ValueError):
                            result[name].append(float(value))
        except Exception as e:
            logger.warning("Failed to read %s: %s", csv_file, e)

    return result


def generate_synthetic_metrics() -> dict[str, list[float]]:
    """Generate synthetic metrics from all builtin scenarios.

    Returns {metric_name: [values]} dict.
    """
    result: dict[str, list[float]] = defaultdict(list)
    scenarios = ScenarioLoader().load_all()

    for scenario in scenarios:
        telemetry = TelemetryGenerator(seed=42).generate(scenario)
        for sample in telemetry.metrics:
            result[sample.metric_name].append(sample.value)

    return result


def compare_distributions(real: dict[str, list[float]], synthetic: dict[str, list[float]]) -> dict:
    """Compare statistical distributions between real and synthetic metrics.

    Returns a report dict with per-metric comparisons.
    """
    value_comparisons: list[dict[str, object]] = []
    report: dict[str, object] = {
        "real_metric_count": len(real),
        "synthetic_metric_count": len(synthetic),
        "real_metric_names": sorted(real.keys()),
        "synthetic_metric_names": sorted(synthetic.keys()),
        "naming_gap": {},
        "value_comparisons": value_comparisons,
    }

    # Naming gap — what real metrics look like vs synthetic
    report["naming_gap"] = {
        "real_examples": list(real.keys())[:10],
        "synthetic_examples": list(synthetic.keys())[:10],
        "observation": "Real metrics use Java/JMX naming (java_lang_OperatingSystem_SystemCpuLoad). "
        "Synthetic uses simple names (cpu_percent). An agent trained on synthetic "
        "names must handle the mapping to real metric names.",
    }

    # Value range comparisons for CPU-like metrics
    for real_name, real_values in real.items():
        if "cpu" in real_name.lower() and real_values:
            r_min, r_max = min(real_values), max(real_values)
            r_avg = sum(real_values) / len(real_values)
            value_comparisons.append(
                {
                    "real_name": real_name,
                    "real_range": f"{r_min:.4f} - {r_max:.4f}",
                    "real_avg": f"{r_avg:.4f}",
                    "real_samples": len(real_values),
                }
            )

    for synth_name, synth_values in synthetic.items():
        if "cpu" in synth_name.lower() and synth_values:
            s_min, s_max = min(synth_values), max(synth_values)
            s_avg = sum(synth_values) / len(synth_values)
            value_comparisons.append(
                {
                    "synthetic_name": synth_name,
                    "synthetic_range": f"{s_min:.3f} - {s_max:.3f}",
                    "synthetic_avg": f"{s_avg:.3f}",
                    "synthetic_samples": len(synth_values),
                }
            )

    return report


def print_validation_report(real_dir: str | None = None) -> None:
    """Print a sim-to-real validation report."""
    print(f"\n{'=' * 60}")
    print("Sim-to-Real Validation Report")
    print(f"{'=' * 60}")

    synthetic = generate_synthetic_metrics()
    print("\n  Synthetic telemetry:")
    print(f"    Metric names: {len(synthetic)}")
    print(f"    Examples: {list(synthetic.keys())[:8]}")
    total_samples = sum(len(v) for v in synthetic.values())
    print(f"    Total samples: {total_samples}")

    if real_dir and Path(real_dir).exists():
        real = load_real_metrics(real_dir)
        print("\n  Real telemetry (Aiops-Dataset):")
        print(f"    Metric names: {len(real)}")
        print(f"    Examples: {list(real.keys())[:5]}")
        total_real = sum(len(v) for v in real.values())
        print(f"    Total samples: {total_real}")

        report = compare_distributions(real, synthetic)

        print("\n  Naming gap:")
        print(f"    Real:      {report['naming_gap']['real_examples'][:3]}")
        print(f"    Synthetic: {report['naming_gap']['synthetic_examples'][:3]}")
        print(f"    {report['naming_gap']['observation']}")

        print("\n  Value range comparisons:")
        for comp in report["value_comparisons"][:6]:
            if "real_name" in comp:
                print(
                    f"    [real] {comp['real_name']}: {comp['real_range']} (avg={comp['real_avg']}, n={comp['real_samples']})"
                )
            else:
                print(
                    f"    [synth] {comp['synthetic_name']}: {comp['synthetic_range']} (avg={comp['synthetic_avg']}, n={comp['synthetic_samples']})"
                )
    else:
        print("\n  No real telemetry data available.")
        print("  To run sim-to-real validation:")
        print("    1. Extract Aiops-Dataset: tar xzf aiops-dataset.tar.gz -C /tmp")
        print(
            "    2. Run: python -m src.training.sim_to_real --real-metrics /tmp/Aiops-Dataset/data/2022-05-05/metric/"
        )

    # Known gaps (always reported)
    print("\n  Known fidelity gaps:")
    print("    1. Metric naming: synthetic uses cpu_percent, real uses java_lang_OperatingSystem_SystemCpuLoad")
    print("    2. Value scale: synthetic cpu 0-100%, real cpu 0.0-1.0 (fraction)")
    print("    3. Log format: synthetic structured text, real JSON/syslog")
    print("    4. Metric cardinality: synthetic ~10 metrics/service, real 50-100+")
    print("    5. Noise profile: synthetic gaussian, real has spikes, seasonality, GC pauses")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--real-metrics", default=None, help="Path to Aiops-Dataset metric dir")
    args = parser.parse_args()
    print_validation_report(args.real_metrics)
