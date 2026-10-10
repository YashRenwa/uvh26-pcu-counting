"""Resize pool images to 1280 px JPEG and write YOLO labels twice (MV and STAPLE).

Outputs:
    data_root/images/<stem>.jpg       long side = prepare.long_side, JPEG quality = prepare.jpeg_quality
    labels_root/labels_mv/<stem>.txt  YOLO format: class cx cy w h, normalised to [0, 1]
    labels_root/labels_st/<stem>.txt
    labels_root/instance_counts.csv   boxes per class, per image and label set (used by split.py)

The JPEGs are slow to make and live on Drive (data_root) on Colab. The labels take seconds to
rebuild from the JSONs, so labels_root stays on local disk: thousands of small writes to Drive
take most of an hour.

YOLO coordinates are normalised, so the resize needs no separate box scaling: dividing the
COCO pixel boxes by the original width and height gives the same numbers at any resolution.
MV and ST files use different image ids, so labels are joined on file_name.

Usage:
    python -m src.prepare [--smoke]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common import add_common_args, load_config, write_result
from src.download import download_annotations, select_pool

LABEL_SETS = ("mv", "st")
Box = tuple[int, float, float, float, float]  # yolo class, x, y, w, h in original pixels


def load_labels(cfg: dict[str, Any]) -> tuple[dict[str, dict[str, list[Box]]], dict[str, tuple[int, int]]]:
    """Boxes per label set and file_name, plus each image's original (width, height)."""
    boxes: dict[str, dict[str, list[Box]]] = {k: defaultdict(list) for k in LABEL_SETS}
    sizes: dict[str, tuple[int, int]] = {}
    for split, paths in download_annotations(cfg).items():
        for k in LABEL_SETS:
            with open(paths[k]) as f:
                coco = json.load(f)
            files = {im["id"]: im["file_name"] for im in coco["images"]}
            for im in coco["images"]:
                sizes[im["file_name"]] = (im["width"], im["height"])
                boxes[k].setdefault(im["file_name"], [])
            for a in coco["annotations"]:
                boxes[k][files[a["image_id"]]].append((a["category_id"] - 1, *a["bbox"]))
    return boxes, sizes


def yolo_lines(boxes: list[Box], width: int, height: int) -> list[str]:
    """COCO pixel boxes -> YOLO 'class cx cy w h' lines, clipped to the frame, exact duplicates dropped.

    The STAPLE JSONs contain exact duplicate boxes (~0.8%, mostly Sedan and LCV; MV has none).
    Ultralytics drops identical label rows when it loads a dataset, so training and mAP never see them;
    dropping them here keeps the counts and PCU totals on the same labels.
    """
    lines = []
    for cls, x, y, w, h in boxes:
        x1, y1 = max(x, 0.0), max(y, 0.0)
        x2, y2 = min(x + w, width), min(y + h, height)
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return list(dict.fromkeys(lines))


def resize_image(src: Path, dst: Path, long_side: int, quality: int) -> int:
    """Downscale so the long side is `long_side` (never upscale) and save as JPEG. Returns bytes."""
    if not dst.exists():
        im = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(src)
        scale = long_side / max(im.shape[:2])
        if scale < 1:
            im = cv2.resize(im, (round(im.shape[1] * scale), round(im.shape[0] * scale)),
                            interpolation=cv2.INTER_AREA)
        tmp = dst.with_suffix(".tmp.jpg")
        cv2.imwrite(str(tmp), im, [cv2.IMWRITE_JPEG_QUALITY, quality])
        tmp.rename(dst)  # atomic, so a disconnect never leaves a half-written JPEG
    return dst.stat().st_size


def run(cfg: dict[str, Any], pool: pd.DataFrame, workers: int = 8) -> dict[str, Any]:
    """Prepare every pool image and both label sets; return the summary written to results."""
    root, lab, classes = cfg["paths"]["data_root"], cfg["paths"]["labels_root"], cfg["dataset"]["classes"]
    (root / "images").mkdir(parents=True, exist_ok=True)
    for k in LABEL_SETS:
        (lab / f"labels_{k}").mkdir(parents=True, exist_ok=True)

    raw = cfg["paths"]["raw_root"]
    p = cfg["prepare"]
    jobs = [(raw / rp, root / "images" / f"{s}.jpg") for s, rp in zip(pool["stem"], pool["repo_path"])]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        nbytes = list(tqdm(ex.map(lambda j: resize_image(*j, p["long_side"], p["jpeg_quality"]), jobs),
                           total=len(jobs), desc="resize", mininterval=10))

    boxes, sizes = load_labels(cfg)
    rows, duplicates = [], {k: 0 for k in LABEL_SETS}
    for stem, fname, split in zip(pool["stem"], pool["file_name"], pool["official_split"]):
        w, h = sizes[fname]
        for k in LABEL_SETS:
            if fname not in boxes[k]:
                raise KeyError(f"{fname} has no {k.upper()} labels; the pool must use images with both")
            lines = yolo_lines(boxes[k][fname], w, h)
            duplicates[k] += len(boxes[k][fname]) - len(lines)  # (degenerate boxes would count too; there are none)
            (lab / f"labels_{k}" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            counts = np.bincount([int(line.split()[0]) for line in lines], minlength=len(classes))
            rows.append({"stem": stem, "official_split": split, "label_set": k, **dict(zip(classes, counts))})
    counts = pd.DataFrame(rows)
    counts.to_csv(lab / "instance_counts.csv", index=False)

    per_set = counts.groupby("label_set")[classes].sum()
    out_h, out_w = cv2.imread(str(jobs[0][1])).shape[:2]
    summary = {
        "images": len(pool),
        "official_split": pool["official_split"].value_counts().to_dict(),
        "subfolders": {f"{s}/{d}": int(n) for (s, d), n in
                       pool.groupby(["official_split", "subfolder"]).size().items()},
        "output_size_wh": [out_w, out_h],
        "jpeg_total_mb": round(sum(nbytes) / 1e6, 1),
        "boxes": {k: int(per_set.loc[k].sum()) for k in LABEL_SETS},
        "boxes_per_image": {k: round(float(per_set.loc[k].sum()) / len(pool), 2) for k in LABEL_SETS},
        "exact_duplicate_boxes_dropped": duplicates,
        "class_counts": {k: {c: int(v) for c, v in per_set.loc[k].items()} for k in LABEL_SETS},
    }
    write_result("prepare", summary, cfg)
    return summary


def draw_labels(cfg: dict[str, Any], stem: str, out: Path) -> Path:
    """Side-by-side MV | ST boxes on one prepared image: a visual check of the conversion."""
    root, classes = cfg["paths"]["data_root"], cfg["dataset"]["classes"]
    base = cv2.imread(str(root / "images" / f"{stem}.jpg"))
    h, w = base.shape[:2]
    rng = np.random.default_rng(0)
    colours = [tuple(int(c) for c in rng.integers(60, 255, 3)) for _ in classes]
    panels = []
    for k in LABEL_SETS:
        im = base.copy()
        lines = (cfg["paths"]["labels_root"] / f"labels_{k}" / f"{stem}.txt").read_text().split()
        for i in range(0, len(lines), 5):
            c, cx, cy, bw, bh = int(lines[i]), *map(float, lines[i + 1:i + 5])
            p1 = (round((cx - bw / 2) * w), round((cy - bh / 2) * h))
            p2 = (round((cx + bw / 2) * w), round((cy + bh / 2) * h))
            cv2.rectangle(im, p1, p2, colours[c], 2)
            cv2.putText(im, classes[c], (p1[0], max(p1[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colours[c], 1)
        cv2.rectangle(im, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(im, f"{k.upper()}: {len(lines) // 5} boxes  ({stem})", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        panels.append(im)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def main() -> None:
    """CLI: `python -m src.prepare` resizes the pool, writes both label sets and a label check figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)

    pool = select_pool(cfg)
    summary = run(cfg, pool, workers=args.workers)
    print(json.dumps({k: v for k, v in summary.items() if k != "class_counts"}, indent=2))
    print(pd.DataFrame(summary["class_counts"]).to_string())
    fig = draw_labels(cfg, pool["stem"].iloc[0], cfg["paths"]["results_dir"] / "figures" / "labels_mv_vs_st.jpg")
    print("label check figure:", fig)


if __name__ == "__main__":
    main()
