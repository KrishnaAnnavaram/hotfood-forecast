"""Settings from environment variables. Every value has a safe offline default."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .schema import CATEGORIES


def read_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Read ``KEY=VALUE`` lines from a local ``.env`` file. Real environment variables win."""
    path = Path(path)
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                out[key.strip()] = value.strip()
    return out


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def _mapping(text: str) -> dict[str, float]:
    """Parse ``"Chicken=0.9,Otherfood=0.6"`` into a dict. Unknown categories are an error."""
    out: dict[str, float] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"expected NAME=VALUE, got {part!r}")
        name, value = (s.strip() for s in part.split("=", 1))
        if name not in CATEGORIES:
            raise ValueError(f"unknown category {name!r}; use one of {', '.join(CATEGORIES)}")
        out[name] = float(value)
    return out


@dataclass(frozen=True)
class Settings:
    data_path: Path = Path("data/sales.csv")
    output_dir: Path = Path("reports")
    seed: int = 42
    horizon: int = 7
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    model: str = "gbm"
    holidays: str = "us_federal"
    cost_ratio: float = 0.4
    unit_cost: dict[str, float] = field(default_factory=dict)
    unit_price: dict[str, float] = field(default_factory=dict)
    disposal_cost: float = 0.05
    threads: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.horizon <= 28:
            raise ValueError("horizon must be between 1 and 28 days")
        if not self.quantiles or any(not 0 < q < 1 for q in self.quantiles):
            raise ValueError("quantiles must be in the open interval (0, 1)")
        if 0.5 not in self.quantiles:
            raise ValueError("quantiles must include 0.5 (the point forecast)")
        if not 0 < self.cost_ratio < 1:
            raise ValueError("cost_ratio must be between 0 and 1")
        if self.threads < 1:
            raise ValueError("threads must be at least 1")
        if self.disposal_cost < 0:
            raise ValueError("disposal_cost must not be negative")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        e = {**read_dotenv(), **os.environ} if env is None else env
        d = cls()
        return cls(
            data_path=Path(e.get("HOTFOOD_DATA", str(d.data_path))),
            output_dir=Path(e.get("HOTFOOD_OUTPUT_DIR", str(d.output_dir))),
            seed=int(e.get("HOTFOOD_SEED", d.seed)),
            horizon=int(e.get("HOTFOOD_HORIZON", d.horizon)),
            quantiles=tuple(sorted(_floats(e["HOTFOOD_QUANTILES"]))) if e.get("HOTFOOD_QUANTILES") else d.quantiles,
            model=e.get("HOTFOOD_MODEL", d.model),
            holidays=e.get("HOTFOOD_HOLIDAYS", d.holidays),
            cost_ratio=float(e.get("HOTFOOD_COST_RATIO", d.cost_ratio)),
            unit_cost=_mapping(e.get("HOTFOOD_UNIT_COST", "")),
            unit_price=_mapping(e.get("HOTFOOD_UNIT_PRICE", "")),
            disposal_cost=float(e.get("HOTFOOD_DISPOSAL_COST", d.disposal_cost)),
            threads=int(e.get("HOTFOOD_THREADS", d.threads)),
        )
