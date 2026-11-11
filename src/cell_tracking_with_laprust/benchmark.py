"""Benchmark byotrack's ``KalmanLinker`` across Linear Assignment Problem (LAP) backends.

The same detections are linked three times, swapping only the LAP solver used by ``byotrack``
through ``pylapy``:

- ``laprust``: dense LAP, our new Rust solver (the ``pylapy`` default).
- ``scipy``: dense LAP, ``scipy.optimize.linear_sum_assignment``.
- ``lapmod``: sparse LAP (lapjv/lapmod from ``lapx``).

``byotrack`` instantiates ``pylapy.LapSolver()`` with no argument internally, so the dense
backend is selected by mutating the ``pylapy.LapSolver.implementations`` class variable. The
sparse backend is reached through the ``SPARSE_OPT_ALT`` association method: ``pylapy`` then
auto-picks ``lapmod`` from its ``sparse_implementations`` registry, independently of the dense
one. The original dense registry is captured at import and restored after every run.

Requires a ``byotrack`` providing the ``OPT_ALT`` / ``SPARSE_OPT_ALT`` association methods.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import byotrack
import pylapy
import torch
from byotrack.implementation.linker.frame_by_frame.kalman_linker import KalmanLinker, KalmanLinkerParameters

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

# Default backends registered by pylapy, in priority order (laprust first => the default).
_ORIGINAL_IMPLEMENTATIONS: dict[str, str] = dict(pylapy.LapSolver.implementations)
_ORIGINAL_SPARSE_IMPLEMENTATIONS: dict[str, str] = dict(pylapy.LapSolver.sparse_implementations)

DEFAULT_RESULTS_PATH = Path("results/benchmark.json")


@dataclass(frozen=True)
class Backend:
    """A LAP backend to benchmark.

    Attributes:
        name: Identifier stored in the results.
        association: byotrack association method driving the LAP solve.
        implementations: Dense backends to force onto ``pylapy.LapSolver``, or ``None`` to
            restore the pylapy defaults.
        sparse_implementations: Sparse backend to force onto ``pylapy.LapSolver``, or ``None`` to
            restore the pylapy defaults.
    """

    name: str
    association: str
    implementations: dict[str, str] | None
    sparse_implementations: dict[str, str] | None


BACKENDS: tuple[Backend, ...] = (
    Backend("scipy", "OPT_ALT", {"scipy": "scipy.optimize"}, None),
    Backend("lapmod", "SPARSE_OPT_ALT", None, {"lapmod": "lap"}),
    Backend("csgraph", "SPARSE_OPT_ALT", None, {"csgraph": "scipy"}),
    Backend("laprust", "OPT_ALT", {"laprust": "lap"}, None),
)


@dataclass
class Timing:
    """Per-frame runtime of one backend over the timed runs."""

    association: str
    mean_s_per_frame: float
    min_s_per_frame: float
    runs_s_per_frame: list[float]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of this timing."""
        return {
            "association": self.association,
            "mean_s_per_frame": self.mean_s_per_frame,
            "min_s_per_frame": self.min_s_per_frame,
            "runs_s_per_frame": self.runs_s_per_frame,
        }


def time_backend(
    detections: Sequence[byotrack.Detections],
    backend: Backend,
    *,
    repeats: int = 3,
) -> tuple[Timing, Collection[byotrack.Track]]:
    """Time the ``KalmanLinker`` on ``detections`` for a single backend.

    One untimed warmup run (on the first ``warmup_frames`` frames) compiles numba kernels and
    warms caches, then ``repeats`` full runs are timed with ``time.perf_counter``.

    Returns:
        A ``(timing, tracks)`` pair: the per-frame timing and the tracks of the first timed run
        (used for the cross-backend equality check).
    """
    pylapy.LapSolver.implementations = dict(backend.implementations or _ORIGINAL_IMPLEMENTATIONS)
    pylapy.LapSolver.sparse_implementations = dict(backend.sparse_implementations or _ORIGINAL_SPARSE_IMPLEMENTATIONS)

    specs = KalmanLinkerParameters(association_method=backend.association)
    specs.estimate(detections)
    linker = KalmanLinker(specs)

    linker.run(None, detections)  # Warmup (result discarded).

    n_frames = len(detections)
    runs_s_per_frame: list[float] = []
    tracks: Collection[byotrack.Track] | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        current = linker.run(None, detections)
        runs_s_per_frame.append((time.perf_counter() - start) / n_frames)
        if tracks is None:
            tracks = current

    assert tracks is not None  # repeats >= 1
    timing = Timing(
        association=backend.association,
        mean_s_per_frame=sum(runs_s_per_frame) / len(runs_s_per_frame),
        min_s_per_frame=min(runs_s_per_frame),
        runs_s_per_frame=runs_s_per_frame,
    )
    return timing, tracks


def assert_same_tracks(reference: Collection[byotrack.Track], other: Collection[byotrack.Track]) -> None:
    """Assert two track sets match on their non-NaN entries (should hold whatever the solver)."""
    ref = byotrack.Track.tensorize(reference)
    oth = byotrack.Track.tensorize(other)
    assert (ref[~torch.isnan(ref)] == oth[~torch.isnan(oth)]).all()


def benchmark_dataset(
    name: str,
    detections: Sequence[byotrack.Detections],
    *,
    source: str,
    sequence: int | None = None,
    repeats: int = 3,
    check: bool = True,
) -> dict[str, object]:
    """Benchmark every backend in ``BACKENDS`` on one dataset and build its results record.

    Args:
        name: Dataset name (results key).
        detections: Per-frame detections to link.
        source: Dataset origin, e.g. ``"ctc"`` or ``"hydra"``.
        sequence: CTC sequence index, if any.
        repeats: Number of timed full runs to average.
        check: If set, assert all backends produce the same tracks.

    Returns:
        The dataset record following the ``results/benchmark.json`` schema.
    """
    n_frames = len(detections)
    n_targets = sum(len(detection) for detection in detections) / n_frames
    print(f"[{name}] {n_frames} frames, {n_targets:.1f} targets/frame on average")

    backends: dict[str, object] = {}
    reference: Collection[byotrack.Track] | None = None
    try:
        for backend in BACKENDS:
            print(f"  {backend.name} ({backend.association}) ...")
            timing, tracks = time_backend(detections, backend, repeats=repeats)
            print(f"    mean {timing.mean_s_per_frame * 1e3:.3f} ms/frame (min {timing.min_s_per_frame * 1e3:.3f})")
            backends[backend.name] = timing.as_dict()
            if check and reference is not None:
                assert_same_tracks(reference, tracks)
            reference = reference or tracks
    finally:
        pylapy.LapSolver.implementations = dict(_ORIGINAL_IMPLEMENTATIONS)

    record: dict[str, object] = {"source": source, "n_frames": n_frames, "n_targets": n_targets}
    if sequence is not None:
        record["sequence"] = sequence
    record["backends"] = backends
    return record


def save_result(name: str, record: dict[str, object], path: Path = DEFAULT_RESULTS_PATH) -> None:
    """Merge ``record`` under ``name`` into the JSON results file (read-modify-write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = json.loads(path.read_text()) if path.exists() else {}
    results[name] = record
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved '{name}' to {path}")
