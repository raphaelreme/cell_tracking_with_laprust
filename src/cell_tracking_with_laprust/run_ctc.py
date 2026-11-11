"""Benchmark LAP backends on Cell Tracking Challenge (CTC) datasets.

The provided (silver-truth) segmentation masks are loaded as detections — the linker ignores
the raw video — and each dataset's first sequence is benchmarked. Run from the project root so
that the ``data/CTC`` symlink resolves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import byotrack.dataset.ctc

from cell_tracking_with_laprust.benchmark import DEFAULT_RESULTS_PATH, benchmark_dataset, save_result

DEFAULT_DATASETS = ("PhC-C2DL-PSC", "BF-C2DL-HSC")


def main() -> None:
    """Benchmark the requested CTC datasets and store their timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), help="CTC dataset names")
    parser.add_argument("--seq", type=int, default=2, help="Sequence index to benchmark")
    parser.add_argument("--data-dir", type=Path, default=Path("data/CTC"), help="Directory holding the CTC datasets")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH, help="Results JSON path")
    args = parser.parse_args()

    for name in args.datasets:
        seg_dir = args.data_dir / name / f"{args.seq:02}_ERR_SEG"
        print(f"Loading detections from {seg_dir}")
        detections = byotrack.dataset.ctc.load_detections(seg_dir)[-100:]
        record = benchmark_dataset(name, detections, source="ctc", sequence=args.seq)
        save_result(name, record, args.results)


if __name__ == "__main__":
    main()
