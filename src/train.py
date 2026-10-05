"""Fine-tune COCO-pretrained YOLOv8 on the frozen split: baseline (MV), ablation (STAPLE), freeze pilot.

Baseline and ablation differ ONLY in --labels (mv | st): same config file, seed, epochs, split and
weights. The weights are always configs/base.yaml train.model (yolov8s.pt) loaded by explicit path
from weights_dir, so Ultralytics never substitutes its default model (YOLO11/YOLO26). Ultralytics'
AMP self-check may still fetch a tiny default model to test mixed precision; it is not trained.

Dataset view, built under paths.datasets_root (local disk on Colab, rebuilt each session):
    images/<stem>.jpg                         prepared JPEGs (hard links, else copies)
    uvh26_<labels>/images/<split> -> ../../images   symlink; Ultralytics swaps /images/ for /labels/
    uvh26_<labels>/labels/<split>/<stem>.txt  the chosen label set
    uvh26_<labels>/<split>.txt                image paths for train / val / test
    uvh26_<labels>.yaml                       lists train and val only; the sealed test split is absent

Outputs: runs_root/<name>/ (Ultralytics run, Drive on Colab) and results/train_<name>.json.

Usage:
    python -m src.train --labels mv                  # baseline
    python -m src.train --labels st                  # ablation
    python -m src.train --labels mv --pilot frozen   # freeze pilot: pilot_epochs, freeze=10
    python -m src.train --labels mv --pilot full     # freeze pilot: pilot_epochs, all layers trainable
    python -m src.train --pilot-summary              # minutes per epoch, epochs that fit the budget
    python -m src.train --labels mv --resume         # continue after a Colab disconnect
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.common import add_common_args, load_config, set_seed, write_result

LABEL_SETS = ("mv", "st")
PILOTS = {"frozen": 10, "full": None}  # freeze value per pilot arm (10 = the YOLOv8 backbone)
SPLITS = ("train", "val", "test")


def weights_path(cfg: dict[str, Any]) -> Path:
    """The COCO-pretrained YOLOv8 checkpoint, by explicit path (downloaded once into weights_dir)."""
    name = cfg["train"]["model"]
    if not name.startswith("yolov8"):
        raise ValueError(f"train.model must be a YOLOv8 checkpoint (yolov8*.pt), got {name}")
    path = cfg["paths"]["weights_dir"] / name
    if not path.exists():
        from ultralytics.utils.downloads import attempt_download_asset

        path.parent.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(str(path))
    return path


def default_device() -> str:
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def build_dataset(cfg: dict[str, Any], label_set: str) -> Path:
    """Materialise the Ultralytics view of the frozen split for one label set; return its data yaml."""
    from src.split import load as load_split

    if label_set not in LABEL_SETS:
        raise ValueError(f"label set must be one of {LABEL_SETS}")
    root, src = cfg["paths"]["datasets_root"], cfg["paths"]["data_root"]
    split = load_split(cfg)
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)

    def place(stem: str) -> None:
        dst = images / f"{stem}.jpg"
        if not dst.exists():
            try:
                os.link(src / "images" / f"{stem}.jpg", dst)
            except OSError:  # different filesystem (Drive -> Colab local disk)
                shutil.copyfile(src / "images" / f"{stem}.jpg", dst)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(place, split.index))

    view = root / f"uvh26_{label_set}"
    shutil.rmtree(view / "labels", ignore_errors=True)  # no stale labels or label caches
    for s in SPLITS:
        link = view / "images" / s
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(images.resolve(), target_is_directory=True)
        labels = view / "labels" / s
        labels.mkdir(parents=True)
        stems = sorted(split.index[split == s])
        for stem in stems:
            shutil.copyfile(cfg["paths"]["labels_root"] / f"labels_{label_set}" / f"{stem}.txt", labels / f"{stem}.txt")
        (view / f"{s}.txt").write_text("\n".join(str(link / f"{stem}.jpg") for stem in stems) + "\n")

    data = root / f"uvh26_{label_set}.yaml"
    data.write_text(yaml.safe_dump({
        "path": str(view), "train": "train.txt", "val": "val.txt",
        "names": dict(enumerate(cfg["dataset"]["classes"])),
    }, sort_keys=False))
    return data


def run_name(label_set: str, pilot: str | None = None) -> str:
    return f"pilot_{pilot}_{label_set}" if pilot else label_set


def train(cfg: dict[str, Any], label_set: str, pilot: str | None = None, resume: bool = False,
          device: str | None = None) -> Path:
    """One Ultralytics training run. Everything but `label_set` (and the pilot arm) comes from the config."""
    from ultralytics import YOLO

    set_seed(cfg["seed"])
    t, name = cfg["train"], run_name(label_set, pilot)
    run_dir = cfg["paths"]["runs_root"] / name
    data = build_dataset(cfg, label_set)  # also needed on resume: the view lives on local disk

    if resume:
        YOLO(str(run_dir / "weights" / "last.pt")).train(resume=True)
    else:
        if run_dir.exists():
            raise FileExistsError(f"{run_dir} exists: pass --resume to continue it, or remove it to retrain")
        model = YOLO(str(weights_path(cfg)))
        model.train(
            data=str(data), epochs=t["pilot_epochs"] if pilot else t["epochs"],
            freeze=PILOTS[pilot] if pilot else t["freeze"],
            imgsz=t["imgsz"], batch=t["batch"], patience=t["patience"], workers=t["workers"], cache=t["cache"],
            cls_remap=t["cls_remap"],
            seed=cfg["seed"], deterministic=t["deterministic"], save_period=t["save_period"],
            project=str(cfg["paths"]["runs_root"]), name=name, exist_ok=True,
            device=device or default_device(), plots=True,
        )
    summary = run_summary(run_dir)
    write_result(f"train_{name}", summary, cfg)
    return run_dir


def run_state(run_dir: Path) -> str:
    """'new', 'unfinished' (last.pt still carries an optimizer) or 'finished' (Ultralytics stripped it)."""
    last = run_dir / "weights" / "last.pt"
    if not last.exists():
        return "unfinished" if run_dir.exists() else "new"
    import torch

    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    return "finished" if ckpt.get("optimizer") is None else "unfinished"


def ensure_run(cfg: dict[str, Any], label_set: str, pilot: str | None = None, device: str | None = None) -> Path:
    """Notebook entry point: train a new run, resume an interrupted one, or skip a finished one."""
    run_dir = cfg["paths"]["runs_root"] / run_name(label_set, pilot)
    state = run_state(run_dir)
    print(f"{run_dir.name}: {state}")
    if state == "finished":
        write_result(f"train_{run_dir.name}", run_summary(run_dir), cfg)
        return run_dir
    if state == "unfinished" and not (run_dir / "weights" / "last.pt").exists():
        raise RuntimeError(f"{run_dir} exists without weights/last.pt: remove it and rerun")
    return train(cfg, label_set, pilot=pilot, resume=state == "unfinished", device=device)


def run_summary(run_dir: Path) -> dict[str, Any]:
    """Timing and val metrics from an Ultralytics run's results.csv and args.yaml."""
    df = pd.read_csv(run_dir / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    t = df["time"].to_numpy()  # cumulative seconds since training started, at the end of each epoch
    per_epoch = float((t[-1] - t[0]) / (len(t) - 1)) if len(t) > 1 else float(t[0])  # excludes epoch-1 warm-up
    m = {"precision": "metrics/precision(B)", "recall": "metrics/recall(B)",
         "mAP50": "metrics/mAP50(B)", "mAP50_95": "metrics/mAP50-95(B)"}
    best = df.loc[df[m["mAP50_95"]].idxmax()]
    args = yaml.safe_load((run_dir / "args.yaml").read_text())
    return {
        "run": run_dir.name, "epochs_run": int(df["epoch"].iloc[-1]),
        "minutes_total": round(float(t[-1]) / 60, 2), "minutes_first_epoch": round(float(t[0]) / 60, 2),
        "minutes_per_epoch": round(per_epoch / 60, 2),
        "val_last": {k: round(float(df[c].iloc[-1]), 4) for k, c in m.items()},
        "val_best": {"epoch": int(best["epoch"]), **{k: round(float(best[c]), 4) for k, c in m.items()}},
        "args": {k: args.get(k) for k in ("model", "data", "epochs", "freeze", "imgsz", "batch", "seed",
                                          "deterministic", "patience", "cache", "cls_remap")},
    }


def epochs_for_budget(summary: dict[str, Any], budget_minutes: float, margin: float = 0.10) -> int:
    """Epochs that fit the budget: (budget * (1 - margin) - first-epoch overhead) / minutes per epoch.

    The margin covers setup, the final validation with best.pt, plots and Drive writes.
    """
    per, extra = summary["minutes_per_epoch"], summary["minutes_first_epoch"] - summary["minutes_per_epoch"]
    return max(1, math.floor((budget_minutes * (1 - margin) - max(extra, 0)) / per))


def pilot_summary(cfg: dict[str, Any], label_set: str = "mv") -> dict[str, Any]:
    """Compare the two pilot arms on val and timing; write results/pilot.json."""
    out = {"label_set": label_set, "pilot_epochs": cfg["train"]["pilot_epochs"],
           "budget_minutes": cfg["train"]["budget_minutes"], "arms": {}}
    for arm in PILOTS:
        run_dir = cfg["paths"]["runs_root"] / run_name(label_set, arm)
        if (run_dir / "results.csv").exists():
            s = run_summary(run_dir)
            s["epochs_within_budget"] = epochs_for_budget(s, cfg["train"]["budget_minutes"])
            out["arms"][arm] = s
    write_result("pilot", out, cfg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", choices=LABEL_SETS, help="training label set: the only baseline/ablation difference")
    ap.add_argument("--pilot", choices=list(PILOTS), help="Phase 2 freeze pilot arm")
    ap.add_argument("--pilot-summary", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default=None, help="default: GPU 0 if available, else cpu")
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)

    if args.pilot_summary:
        print(json.dumps(pilot_summary(cfg), indent=2))
        return
    if not args.labels:
        ap.error("--labels is required")
    run_dir = train(cfg, args.labels, pilot=args.pilot, resume=args.resume, device=args.device)
    print(json.dumps(run_summary(run_dir), indent=2))


if __name__ == "__main__":
    main()
