#!/usr/bin/env python3
"""Solve an exact global action-block budget from a calibrated cosine matrix."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from fastwam.acceleration.schedule import ActionSchedule, LAYERS, STEPS
from fastwam.acceleration.schedule_solver import solve_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--similarity", type=Path, required=True, help="Float .npy array [layer, source step, target step]")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--no-terminal-anchor", action="store_true")
    parser.add_argument("--allow-empty-step", action="store_true")
    args = parser.parse_args()

    similarity = np.load(args.similarity, allow_pickle=False)
    if similarity.shape != (LAYERS, STEPS, STEPS):
        parser.error(f"--similarity must have shape ({LAYERS}, {STEPS}, {STEPS})")
    constraints = dict(
        prefix_layers=3,
        require_step_coverage=not args.allow_empty_step,
        terminal_anchor=not args.no_terminal_anchor,
    )
    result = solve_schedule(similarity, args.budget, time_limit_seconds=args.time_limit, **constraints)
    mask = result.pop("mask")
    schedule = ActionSchedule(args.name or args.output.stem, tuple(tuple(row) for row in mask))
    payload = dict(**schedule.metadata(), mask=mask, constraints=constraints, solver=result,
                   similarity_sha256=sha256(args.similarity.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(schedule.metadata(), indent=2))


if __name__ == "__main__":
    main()
