"""Detection evaluation outside the Ultralytics trainer.

Phase 2 (this file so far): the COCO-as-is context run. COCO-pretrained YOLOv8 is applied to val
without fine-tuning, and both its predictions and the UVH-26 labels are mapped to shared coarse
classes, following the plan:
    Hatchback, Sedan, SUV, MUV -> car     Bus, Mini-bus -> bus     Truck, LCV -> truck
    Two-wheeler -> motorcycle             Bicycle -> bicycle
Three-wheeler, Tempo-traveller, Van and Others have no COCO class: they stay in the ground truth as
classes no prediction can match, so they count as misses (AP 0). mAP is reported over all coarse
classes (the honest number) and over the five mappable ones.

Matching mirrors Ultralytics' validator (greedy, IoU thresholds 0.50:0.95) and AP comes from
ultralytics.utils.metrics.ap_per_class, so these numbers are comparable with fine-tuned runs.

The test split is sealed: every entry point refuses split="test" unless allow_test=True (Phase 5).

Usage:
    python -m src.evaluate coco-context [--smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.common import add_common_args, load_config, write_result

IOUV = np.linspace(0.5, 0.95, 10)
COARSE = ["car", "bus", "truck", "motorcycle", "bicycle", "three-wheeler", "tempo-traveller", "van", "others"]
UVH_TO_COARSE = {
    "Hatchback": "car", "Sedan": "car", "SUV": "car", "MUV": "car", "Bus": "bus", "Mini-bus": "bus",
    "Truck": "truck", "LCV": "truck", "Two-wheeler": "motorcycle", "Bicycle": "bicycle",
    "Three-wheeler": "three-wheeler", "Tempo-traveller": "tempo-traveller", "Van": "van", "Others": "others",
}
COCO_TO_COARSE = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # COCO80 class ids
MAPPABLE = ["car", "bus", "truck", "motorcycle", "bicycle"]


def split_stems(cfg: dict[str, Any], split: str, allow_test: bool = False) -> list[str]:
    if split == "test" and not allow_test:
        raise PermissionError("the test split is sealed until Phase 5 (allow_test=True)")
    return (cfg["paths"]["splits_dir"] / f"{split}.txt").read_text().split()


def load_gt(cfg: dict[str, Any], stems: list[str], label_set: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """YOLO labels per image as (class ids, normalised xyxy boxes)."""
    out = {}
    for stem in stems:
        a = np.loadtxt(cfg["paths"]["data_root"] / f"labels_{label_set}" / f"{stem}.txt", ndmin=2)
        a = a.reshape(-1, 5)
        cx, cy, w, h = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
        out[stem] = (a[:, 0].astype(int), np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1))
    return out


def predict(weights: Path, image_paths: list[Path], imgsz: int, conf: float = 0.001, iou: float = 0.7,
            classes: list[int] | None = None, device: str | None = None) -> dict[str, tuple[np.ndarray, ...]]:
    """Per image (stem): (class ids, confidences, normalised xyxy boxes). Ultralytics val-style settings."""
    from ultralytics import YOLO

    from src.train import default_device

    model, out = YOLO(str(weights)), {}
    for r in model.predict(source=[str(p) for p in image_paths], imgsz=imgsz, conf=conf, iou=iou, classes=classes,
                           max_det=300, device=device or default_device(), stream=True, verbose=False):
        b = r.boxes
        out[Path(r.path).stem] = (b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy(), b.xyxyn.cpu().numpy())
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix (len(a) x len(b)) for xyxy boxes. Normalised coordinates give the pixel IoU."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def match(pred_cls: np.ndarray, gt_cls: np.ndarray, iou: np.ndarray) -> np.ndarray:
    """(n_pred, 10) correct matrix; the greedy matching of Ultralytics' BaseValidator.match_predictions."""
    correct = np.zeros((len(pred_cls), len(IOUV)), bool)
    iou = iou * (gt_cls[:, None] == pred_cls[None, :])  # rows: ground truth, columns: predictions
    for k, thr in enumerate(IOUV):
        m = np.array(np.nonzero(iou >= thr)).T
        if len(m):
            if len(m) > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1], k] = True
    return correct


def detection_metrics(preds: dict, gts: dict, names: list[str]) -> dict[str, Any]:
    """mAP@0.50, mAP@0.50:0.95 and per-class AP over every class that has ground truth."""
    from ultralytics.utils.metrics import ap_per_class

    tp, conf, pcls, tcls = [], [], [], []
    for stem, (g_cls, g_box) in gts.items():
        p_cls, p_conf, p_box = preds.get(stem, (np.zeros(0, int), np.zeros(0), np.zeros((0, 4))))
        tp.append(match(p_cls, g_cls, box_iou(g_box, p_box)) if len(p_cls) else np.zeros((0, len(IOUV)), bool))
        conf.append(p_conf), pcls.append(p_cls), tcls.append(g_cls)
    tp, conf, pcls, tcls = map(np.concatenate, (tp, conf, pcls, tcls))
    _, _, p, r, _, ap, classes, *_ = ap_per_class(tp, conf, pcls, tcls, names=dict(enumerate(names)))
    per_class = {names[c]: {"AP50": round(float(ap[i, 0]), 4), "AP50_95": round(float(ap[i].mean()), 4),
                            "instances": int((tcls == c).sum())} for i, c in enumerate(classes)}
    return {"mAP50": round(float(ap[:, 0].mean()), 4), "mAP50_95": round(float(ap.mean()), 4),
            "per_class": per_class}


def coco_context(cfg: dict[str, Any], split: str = "val", label_sets=("mv", "st")) -> dict[str, Any]:
    """COCO-pretrained YOLOv8 as-is on `split`, scored in the shared coarse classes."""
    from src.train import weights_path

    stems = split_stems(cfg, split)
    paths = [cfg["paths"]["data_root"] / "images" / f"{s}.jpg" for s in stems]
    raw = predict(weights_path(cfg), paths, cfg["train"]["imgsz"], classes=sorted(COCO_TO_COARSE))
    cid = {n: i for i, n in enumerate(COARSE)}
    coco_map = {k: cid[v] for k, v in COCO_TO_COARSE.items()}
    preds = {s: (np.array([coco_map[c] for c in cls], int), cf, bx) for s, (cls, cf, bx) in raw.items()}
    uvh = np.array([cid[UVH_TO_COARSE[n]] for n in cfg["dataset"]["classes"]])

    out = {"weights": cfg["train"]["model"], "split": split, "images": len(stems), "imgsz": cfg["train"]["imgsz"],
           "mapping": UVH_TO_COARSE, "coco_classes_used": {v: k for k, v in COCO_TO_COARSE.items()}}
    for ls in label_sets:
        gts = {s: (uvh[c], b) for s, (c, b) in load_gt(cfg, stems, ls).items()}
        m = detection_metrics(preds, gts, COARSE)
        mappable = [m["per_class"][c] for c in MAPPABLE if c in m["per_class"]]
        m["mAP50_mappable_only"] = round(float(np.mean([c["AP50"] for c in mappable])), 4)
        m["mAP50_95_mappable_only"] = round(float(np.mean([c["AP50_95"] for c in mappable])), 4)
        out[ls] = m
    write_result(f"coco_context_{split}", out, cfg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["coco-context"])
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    if args.what == "coco-context":
        res = coco_context(cfg)
        print(json.dumps({k: v for k, v in res.items() if k != "mapping"}, indent=2))


if __name__ == "__main__":
    main()
