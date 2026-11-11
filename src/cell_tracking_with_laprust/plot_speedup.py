"""Plot the laprust-vs-scipy speedup against the number of targets.

Reads the ``results/benchmark.json`` produced by the ``run_*`` scripts and draws a single line:
the speedup of our Rust solver over scipy (``scipy`` runtime / ``laprust`` runtime) versus the
average number of detections per frame, with one point per dataset. A dashed line marks parity
(ratio 1): points above it are datasets where laprust is faster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from cell_tracking_with_laprust.benchmark import DEFAULT_RESULTS_PATH

DEFAULT_PLOT_PATH = Path("results/speedup_vs_targets.png")

LAPRUST_BLUE = "#2a78d6"  # Same hue/marker laprust carries in the runtime plot.


def load_speedups(results: dict[str, dict]) -> list[tuple[float, float, str]]:
    """Return ``[(n_targets, scipy/laprust ratio, dataset), ...]`` sorted by ``n_targets``.

    Datasets missing either the ``scipy`` or ``laprust`` backend are skipped.
    """
    points: list[tuple[float, float, str]] = []
    for dataset, record in results.items():
        backends = record["backends"]
        if "scipy" not in backends or "laprust" not in backends:
            continue
        ratio = backends["scipy"]["mean_s_per_frame"] / backends["laprust"]["mean_s_per_frame"]
        points.append((record["n_targets"], ratio, dataset))
    points.sort()
    return points


def plot(results: dict[str, dict], output: Path) -> None:
    """Draw the speedup-vs-targets figure and save it to ``output``."""
    points = load_speedups(results)
    if not points:
        msg = "No dataset has both 'scipy' and 'laprust' backends."
        raise ValueError(msg)

    xs = [x for x, _, _ in points]
    ys = [y for _, y, _ in points]

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ax.axhline(1.0, color="#52514e", linewidth=1, linestyle="--", zorder=1)
    ax.plot(
        xs,
        ys,
        marker="D",
        color=LAPRUST_BLUE,
        linewidth=2,
        markersize=8,
        markeredgecolor="white",  # 2px surface ring keeps markers legible where points crowd.
        markeredgewidth=1.5,
        zorder=3,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Targets per frame (avg. detections/frame)")
    ax.set_ylabel("Speedup (scipy runtime / laprust runtime)")
    ax.set_title("laprust speedup over scipy vs. number of targets")
    # Annotate which side of parity means what, in muted ink rather than the series colour.
    ax.text(0.015, 0.97, "laprust faster ↑", transform=ax.transAxes, color="#52514e", va="top")
    ax.grid(visible=True, which="both", color="#e6e6e3", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.savefig(output, dpi=200)
    print(f"Saved plot to {output}")


def main() -> None:
    """Load the benchmark results and render the speedup-vs-targets plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH, help="Results JSON path")
    parser.add_argument("--output", type=Path, default=DEFAULT_PLOT_PATH, help="Output image path")
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(results, args.output)


if __name__ == "__main__":
    main()
