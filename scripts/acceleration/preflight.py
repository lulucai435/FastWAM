#!/usr/bin/env python3
"""Check the fixed action schedule bundle and optionally verify CUDA."""
import argparse
import json
import sys

from fastwam.acceleration.schedule import load_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Also initialize CUDA and perform a device operation")
    args = parser.parse_args()
    expected = {
        "terminal97": (97, "1cb29bfe1cec4bf24c0e7c4054b07ba4d5188a5b4524f94d013607944cd959ed"),
        "global50": (165, "0ee838856939dce145f2acdf030a07fb728d74be12b9732c3bc06e1861bad684"),
    }
    report = dict(python=sys.version.split()[0], schedules={})
    for name in ("full300", *expected):
        schedule = load_schedule(name)
        if name in expected and (schedule.calls, schedule.mask_sha256) != expected[name]:
            raise RuntimeError(f"Unexpected packaged schedule: {name}")
        report["schedules"][name] = schedule.metadata()
    if args.cuda:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA initialization failed")
        value = torch.ones(8, device="cuda").sum().item()
        if value != 8:
            raise RuntimeError(f"Unexpected CUDA computation result: {value}")
        report.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      gpu=torch.cuda.get_device_name(0))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
