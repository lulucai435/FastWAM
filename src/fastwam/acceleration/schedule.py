"""Read and validate the 10-step, 30-layer action execution tables."""
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path

STEPS = 10
LAYERS = 30


@dataclass(frozen=True)
class ActionSchedule:
    name: str
    mask: tuple[tuple[bool, ...], ...]

    def __post_init__(self):
        if len(self.mask) != STEPS or any(len(row) != LAYERS for row in self.mask):
            raise ValueError(f"Expected a [{STEPS}, {LAYERS}] action mask")
        if any(type(skip) is not bool for row in self.mask for skip in row):
            raise ValueError("Mask values must be JSON booleans (true means reuse)")
        if any(self.mask[0]):
            raise ValueError("All layers must be computed at the first denoising step")

    @property
    def calls_per_step(self) -> tuple[int, ...]:
        return tuple(LAYERS - sum(row) for row in self.mask)

    @property
    def calls(self) -> int:
        return sum(self.calls_per_step)

    @property
    def mask_sha256(self) -> str:
        return sha256(bytes(skip for row in self.mask for skip in row)).hexdigest()

    def metadata(self) -> dict:
        return dict(name=self.name, calls=self.calls, calls_per_step=list(self.calls_per_step),
                    mask_sha256=self.mask_sha256)


def load_schedule(name_or_path: str | Path) -> ActionSchedule:
    if name_or_path == "full300":
        return ActionSchedule("full300", ((False,) * LAYERS,) * STEPS)
    if name_or_path in ("terminal97", "global50"):
        source = files("fastwam.acceleration").joinpath("schedules", f"{name_or_path}.json")
    else:
        source = Path(name_or_path)
    data = json.loads(source.read_text(encoding="utf-8"))
    schedule = ActionSchedule(data["name"], tuple(tuple(row) for row in data["mask"]))
    for key in ("calls", "calls_per_step", "mask_sha256"):
        if data[key] != schedule.metadata()[key]:
            raise ValueError(f"Inconsistent schedule {key} in {source}")
    return schedule
