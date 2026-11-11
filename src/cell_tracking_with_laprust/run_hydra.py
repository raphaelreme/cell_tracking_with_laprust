"""Benchmark LAP backends on the Hydra neurons example video.

Wavelet detection is run once on the full 1000-frame Hydra sequence (downloaded on first use),
then the LAP backends are benchmarked on the resulting detections.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import byotrack
import byotrack.example_data
import torch
from byotrack.implementation.detector.wavelet import WaveletDetector

from cell_tracking_with_laprust.benchmark import DEFAULT_RESULTS_PATH, benchmark_dataset, save_result


def main() -> None:
    """Detect on the Hydra video and benchmark the LAP backends on its detections."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH, help="Results JSON path")
    args = parser.parse_args()

    video = byotrack.example_data.hydra_neurons()[130:230]  # 100 frames with hydra contraction
    video.add_preprocessor(byotrack.video.ChannelProjection("mean"))
    video = video.normalize(q_min=0.01, q_max=0.999, smooth_clip=1.0)

    detector = WaveletDetector(scale=1, k=2.5, min_area=5, batch_size=20, device=torch.device("cpu"))
    print("Running wavelet detection on the Hydra video ...")
    detections = detector.run(video)

    record = benchmark_dataset("hydra", detections, source="hydra")
    save_result("hydra", record, args.results)


if __name__ == "__main__":
    main()
