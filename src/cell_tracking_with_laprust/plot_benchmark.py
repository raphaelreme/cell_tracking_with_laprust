"""Plot LAP backend runtime against the number of targets.

Reads the ``results/benchmark.json`` produced by the ``run_*`` scripts and draws one line per
backend: runtime per frame (y) versus the average number of detections per frame (x), with one
point per dataset. Both axes are logarithmic since datasets and runtimes each span two orders of
magnitude.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from cell_tracking_with_laprust.benchmark import DEFAULT_RESULTS_PATH

DEFAULT_PLOT_PATH = Path("results/runtime_vs_targets.png")

# Fixed per-backend style: a categorical colour (validated CVD-safe) and a distinct marker, so
# identity survives colour-blindness and greyscale printing. Colour follows the backend, never its
# rank, so adding a dataset never repaints a series.
BACKEND_STYLE: dict[str, tuple[str, str]] = {
    "laprust": ("#2a78d6", "D"),  # blue   — our Rust solver (the pylapy default)
    "scipy": ("#008300", "o"),  # green  — scipy dense LAP
    "lapmod": ("#eda100", "s"),  # yellow — lapx sparse LAP
    "csgraph": ("#1baf7a", "^"),  # aqua   — scipy sparse LAP
}


def load_series(results: dict[str, dict]) -> dict[str, list[tuple[float, float, str]]]:
    """Group the results into ``backend -> [(n_targets, ms_per_frame, dataset), ...]``.

    Points are sorted by ``n_targets`` so each backend's line reads left to right.
    """
    series: dict[str, list[tuple[float, float, str]]] = {name: [] for name in BACKEND_STYLE}
    for dataset, record in results.items():
        n_targets = record["n_targets"]
        for backend, timing in record["backends"].items():
            if backend not in series:  # Unknown backend not in our fixed style table.
                series[backend] = []
            series[backend].append((n_targets, timing["mean_s_per_frame"] * 1e3, dataset))
    for points in series.values():
        points.sort()
    return series


def plot(results: dict[str, dict], output: Path) -> None:
    """Draw the runtime-vs-targets figure and save it to ``output``."""
    series = load_series(results)

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    for backend, points in series.items():
        if not points:
            continue
        xs = [x for x, _, _ in points]
        ys = [y for _, y, _ in points]
        colour, marker = BACKEND_STYLE.get(backend, ("#52514e", "x"))
        ax.plot(
            xs,
            ys,
            marker=marker,
            color=colour,
            label=backend,
            linewidth=2,
            markersize=8,
            markeredgecolor="white",  # 2px surface ring keeps overlapping markers legible.
            markeredgewidth=1.5,
            zorder=3 if backend == "laprust" else 2,  # Keep the hero series on top.
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Targets per frame (avg. detections/frame)")
    ax.set_ylabel("Runtime (ms per frame)")
    ax.set_title("LAP backend runtime vs. number of targets")
    ax.grid(visible=True, which="both", color="#e6e6e3", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(title="Backend", frameon=False)

    fig.savefig(output, dpi=200)
    print(f"Saved plot to {output}")


def main() -> None:
    """Load the benchmark results and render the runtime-vs-targets plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH, help="Results JSON path")
    parser.add_argument("--output", type=Path, default=DEFAULT_PLOT_PATH, help="Output image path")
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(results, args.output)


if __name__ == "__main__":
    main()
