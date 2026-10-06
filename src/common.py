"""Shared plumbing: config loading, path resolution, seeding, results output."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "base.yaml"

# Env vars that override config paths, so Colab can point at Drive without editing the config.
PATH_ENV = {
    "raw_root": "UVH_RAW_ROOT",
    "data_root": "UVH_DATA_ROOT",
    "runs_root": "UVH_RUNS_ROOT",
    "weights_dir": "UVH_WEIGHTS_DIR",
    "datasets_root": "UVH_DATASETS_ROOT",
    "labels_root": "UVH_LABELS_ROOT",
    "results_dir": "UVH_RESULTS_DIR",
}


def load_config(path: str | Path = DEFAULT_CONFIG, smoke: bool = False) -> dict[str, Any]:
    """Load the YAML config and resolve every entry in `paths` to an absolute Path.

    smoke=True shrinks the pool to `pool.smoke_size` and moves prepared data, splits and
    results under <data_root>/../smoke/, so CPU smoke tests never touch the real outputs.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg["paths"].items():
        value = os.environ.get(PATH_ENV.get(key, ""), value)
        p = Path(value).expanduser()
        cfg["paths"][key] = p if p.is_absolute() else ROOT / p
    if smoke:
        smoke_dir = cfg["paths"]["data_root"].parent / "smoke"
        scale = cfg["pool"]["smoke_size"] / cfg["pool"]["size"]
        cfg["pool"]["size"] = cfg["pool"]["smoke_size"]
        cfg["split"]["min_test_images"] = int(cfg["split"]["min_test_images"] * scale)
        cfg["paths"].update(data_root=smoke_dir / "prepared", splits_dir=smoke_dir / "splits",
                            results_dir=smoke_dir / "results", datasets_root=smoke_dir / "datasets",
                            labels_root=smoke_dir / "labels",
                            runs_root=smoke_dir / "runs")
        cfg["train"].update(epochs=2, pilot_epochs=3)  # CPU smoke: exercise the code, not the model
    cfg["smoke"] = smoke
    return cfg


def add_common_args(ap) -> None:
    """CLI flags shared by every src/ script."""
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--smoke", action="store_true", help="tiny pool, outputs under data/smoke/")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA). Ultralytics gets the seed via train args."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_result(name: str, data: dict[str, Any], cfg: dict[str, Any] | None = None) -> Path:
    """Write `data` to results/<name>.json so the number can be quoted in the report."""
    cfg = cfg or load_config()
    out = cfg["paths"]["results_dir"] / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return out


def file_sha256(path: Path) -> str:
    """sha256 of a file, read in chunks (weights are ~22 MB)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def library_versions() -> dict[str, str]:
    """Versions of every library the project uses, for the notebook's first cell."""
    import importlib.metadata as md
    import platform

    pkgs = ["numpy", "matplotlib", "torch", "torchvision", "opencv-python", "transformers",
            "ultralytics", "huggingface-hub", "pandas", "pyyaml", "tqdm"]
    versions = {"python": platform.python_version()}
    for p in pkgs:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = "not installed"
    return versions
