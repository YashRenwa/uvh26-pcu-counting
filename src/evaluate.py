"""Detection evaluation: COCO-as-is context run and Ultralytics validation of trained models.

detector_metrics (Phases 2-5): a trained model's mAP@0.50, mAP@0.50:0.95, precision, recall and
per-class AP on one split, scored against one label set (MV or ST). Phase 2 uses it on train and val
for the generalisation gap; Phase 3 for the 2 x 2 trained-on x scored-against table.

coco_context (Phase 2): COCO-pretrained YOLOv8 is applied to val without fine-tuning, and both its
predictions and the UVH-26 labels are mapped to shared coarse classes, following the plan:
    Hatchback, Sedan, SUV, MUV -> car     Bus, Mini-bus -> bus     Truck, LCV -> truck
    Two-wheeler -> motorcycle             Bicycle -> bicycle
Three-wheeler, Tempo-traveller, Van and Others have no COCO class: they stay in the ground truth as
classes no prediction can match, so they count as misses (AP 0). mAP is reported over all coarse
classes (the honest number) and over the five mappable ones. Matching mirrors Ultralytics' validator
(greedy, IoU 0.50:0.95) and AP comes from ultralytics.utils.metrics.ap_per_class.

The test split is sealed: every entry point refuses split="test" unless allow_test=True (Phase 5).

Usage:
    python -m src.evaluate coco-context [--smoke]
    python -m src.evaluate detector --run mv --labels mv --split val [--smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.common import add_common_args, file_sha256, load_config, write_result

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
        a = np.loadtxt(cfg["paths"]["labels_root"] / f"labels_{label_set}" / f"{stem}.txt", ndmin=2)
        a = a.reshape(-1, 5)
        cx, cy, w, h = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
        out[stem] = (a[:, 0].astype(int), np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1))
    return out


def predict(weights: Path, image_paths: list[Path], imgsz: int, conf: float = 0.001, iou: float = 0.7,
            classes: list[int] | None = None, device: str | None = None, batch: int = 16,
            agnostic_nms: bool = False) -> dict[str, tuple[np.ndarray, ...]]:
    """Per image (stem): (class ids, confidences, normalised xyxy boxes). Ultralytics val-style settings.

    Images go to the model `batch` at a time: Ultralytics runs a list source as ONE batch, so passing
    every path at once loads them all and exhausts GPU memory (452 val images needed ~13 GB on a T4).
    """
    from ultralytics import YOLO

    from src.train import default_device

    model, out, device = YOLO(str(weights)), {}, device or default_device()
    for i in range(0, len(image_paths), batch):
        chunk = [str(p) for p in image_paths[i:i + batch]]
        for r in model.predict(source=chunk, imgsz=imgsz, conf=conf, iou=iou, classes=classes, max_det=300,
                               agnostic_nms=agnostic_nms, device=device, stream=True, verbose=False):
            b = r.boxes
            out[Path(r.path).stem] = (b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy(), b.xyxyn.cpu().numpy())
    if len(out) != len(image_paths):
        raise RuntimeError(f"got predictions for {len(out)} of {len(image_paths)} images")
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


def coarse_metrics(cfg: dict[str, Any], raw: dict, pred_to_coarse: dict[int, int], stems: list[str],
                   label_sets=("mv", "st")) -> dict[str, Any]:
    """Score raw predictions in the shared coarse classes, against each label set.

    Predictions are mapped to coarse classes and NMS is re-run per coarse class (IoU evaluate.iou_nms),
    so a fine-tuned model is not penalised for, e.g., a Hatchback and a Sedan box on one car.
    """
    import torch
    from torchvision.ops import batched_nms

    cid = {n: i for i, n in enumerate(COARSE)}
    preds = {}
    for stem, (cls, cf, bx) in raw.items():
        c = np.array([pred_to_coarse[int(k)] for k in cls], int)
        if len(c):
            keep = batched_nms(torch.as_tensor(bx, dtype=torch.float32), torch.as_tensor(cf, dtype=torch.float32),
                               torch.as_tensor(c), cfg["evaluate"]["iou_nms"]).numpy()[:300]
            cls_, cf, bx = c[keep], cf[keep], bx[keep]
        else:
            cls_ = c
        preds[stem] = (cls_, cf, bx)
    uvh = np.array([cid[UVH_TO_COARSE[n]] for n in cfg["dataset"]["classes"]])
    out = {}
    for ls in label_sets:
        gts = {st: (uvh[c], b) for st, (c, b) in load_gt(cfg, stems, ls).items()}
        m = detection_metrics(preds, gts, COARSE)
        mappable = [m["per_class"][c] for c in MAPPABLE if c in m["per_class"]]
        m["mAP50_mappable_only"] = round(float(np.mean([c["AP50"] for c in mappable])), 4)
        m["mAP50_95_mappable_only"] = round(float(np.mean([c["AP50_95"] for c in mappable])), 4)
        out[ls] = m
    return out


def coco_context(cfg: dict[str, Any], split: str = "val", label_sets=("mv", "st")) -> dict[str, Any]:
    """COCO-pretrained YOLOv8 as-is on `split`, scored in the shared coarse classes."""
    from src.train import weights_path

    stems = split_stems(cfg, split)
    paths = [cfg["paths"]["data_root"] / "images" / f"{s}.jpg" for s in stems]
    raw = predict(weights_path(cfg), paths, cfg["train"]["imgsz"], classes=sorted(COCO_TO_COARSE))
    cid = {n: i for i, n in enumerate(COARSE)}
    out = {"weights": cfg["train"]["model"], "split": split, "images": len(stems), "imgsz": cfg["train"]["imgsz"],
           "mapping": UVH_TO_COARSE, "coco_classes_used": {v: k for k, v in COCO_TO_COARSE.items()},
           **coarse_metrics(cfg, raw, {k: cid[v] for k, v in COCO_TO_COARSE.items()}, stems, label_sets)}
    write_result(f"coco_context_{split}", out, cfg)
    return out


def finetuned_coarse(cfg: dict[str, Any], weights: Path, name: str, split: str = "val",
                     label_sets=("mv", "st")) -> dict[str, Any]:
    """A fine-tuned 14-class model scored in the same coarse classes as the COCO-as-is context run."""
    stems = split_stems(cfg, split)
    paths = [cfg["paths"]["data_root"] / "images" / f"{s}.jpg" for s in stems]
    raw = predict(weights, paths, cfg["train"]["imgsz"])
    cid = {n: i for i, n in enumerate(COARSE)}
    pred_map = {i: cid[UVH_TO_COARSE[n]] for i, n in enumerate(cfg["dataset"]["classes"])}
    out = {"weights": str(weights), "split": split, "images": len(stems),
           **coarse_metrics(cfg, raw, pred_map, stems, label_sets)}
    write_result(f"coarse_{name}_{split}", out, cfg)
    return out


def detector_metrics(cfg: dict[str, Any], weights: Path, label_set: str, split: str, name: str,
                     allow_test: bool = False, device: str | None = None) -> dict[str, Any]:
    """Ultralytics validation of `weights` on `split`, scored against `label_set` labels.

    Same settings as Ultralytics' own validation (conf 0.001, NMS IoU evaluate.iou_nms, imgsz from the
    config). Plots (PR curve, confusion matrix) go to runs_root/eval/<name>/; numbers to
    results/eval_<name>.json.
    """
    from ultralytics import YOLO

    from src.train import build_dataset, default_device

    split_stems(cfg, split, allow_test)  # sealed-test guard
    settings = {"weights_sha256": file_sha256(weights), "label_set": label_set, "split": split,
                "imgsz": cfg["train"]["imgsz"], "iou_nms": cfg["evaluate"]["iou_nms"], "conf": 0.001}
    cached = cfg["paths"]["results_dir"] / f"eval_{name}.json"
    if cached.exists() and (cfg["paths"]["runs_root"] / "eval" / name / "per_image.npz").exists():
        old = json.loads(cached.read_text())
        if old.get("settings") == settings:
            print(f"{name}: unchanged weights and settings, using the saved evaluation")
            return old
    data = build_dataset(cfg, label_set, include_test=split == "test")
    captured: dict[str, Any] = {k: [] for k in ("tp", "conf", "pred_cls", "target_cls")}

    def grab_batch(v) -> None:  # per-image match statistics for the paired bootstrap; Ultralytics
        st = v.metrics.stats    # clears them before on_val_end, so copy the new entries each batch
        for k in captured:
            captured[k].extend(st[k][len(captured[k]):])

    def grab_names(v) -> None:
        captured["stems"] = [Path(f).stem for f in v.dataloader.dataset.im_files]

    model = YOLO(str(weights))
    model.add_callback("on_val_batch_end", grab_batch)
    model.add_callback("on_val_end", grab_names)
    r = model.val(
        data=str(data), split=split, imgsz=cfg["train"]["imgsz"], batch=16, conf=0.001,
        iou=cfg["evaluate"]["iou_nms"], device=device or default_device(), plots=True, verbose=False,
        project=str(cfg["paths"]["runs_root"] / "eval"), name=name, exist_ok=True,
    )
    save_per_image(captured, cfg["paths"]["runs_root"] / "eval" / name / "per_image.npz")
    rows = r.summary()
    out = {
        "weights": str(weights), "label_set": label_set, "split": split, "settings": settings,
        "images": len(split_stems(cfg, split, allow_test)),
        "mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
        "per_class": {row["Class"]: {"AP50": round(float(row["mAP50"]), 4), "AP50_95": round(float(row["mAP50-95"]), 4),
                                     "P": round(float(row["Box-P"]), 4), "R": round(float(row["Box-R"]), 4),
                                     "instances": int(row["Instances"])} for row in rows},
    }
    write_result(f"eval_{name}", out, cfg)
    return out


def save_per_image(c: dict[str, Any], path: Path) -> None:
    """Ragged per-image arrays -> flat arrays plus image indices, in one compressed file."""
    if len(c["tp"]) != len(c["stems"]):
        raise RuntimeError(f"per-image stats ({len(c['tp'])}) do not match images ({len(c['stems'])})")
    n_pred = [len(x) for x in c["conf"]]
    n_gt = [len(x) for x in c["target_cls"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, stems=np.array(c["stems"]),
        tp=np.concatenate(c["tp"]).astype(bool).reshape(-1, len(IOUV)), conf=np.concatenate(c["conf"]),
        pred_cls=np.concatenate(c["pred_cls"]).astype(int), pred_img=np.repeat(np.arange(len(n_pred)), n_pred),
        target_cls=np.concatenate(c["target_cls"]).astype(int), target_img=np.repeat(np.arange(len(n_gt)), n_gt))


def _resampled_ap(z, counts: np.ndarray) -> tuple[float, float, dict[int, float]]:
    """mAP50, mAP50-95 and per-class AP50-95 with image i included counts[i] times.

    Averages are over the classes present in the resample, as in Ultralytics.
    """
    from ultralytics.utils.metrics import ap_per_class

    wp, wt = counts[z["pred_img"]], counts[z["target_img"]]
    p, t = np.repeat(np.arange(len(wp)), wp), np.repeat(np.arange(len(wt)), wt)
    r = ap_per_class(z["tp"][p], z["conf"][p], z["pred_cls"][p], z["target_cls"][t])
    ap, classes = r[5], r[6]
    return float(ap[:, 0].mean()), float(ap.mean()), {int(c): float(a) for c, a in zip(classes, ap.mean(1))}


def _resampled_map(z, counts: np.ndarray) -> tuple[float, float]:
    return _resampled_ap(z, counts)[:2]


def paired_bootstrap(cfg: dict[str, Any], name_a: str, name_b: str, n_boot: int = 1000) -> dict[str, Any]:
    """95% CI for (model B - model A) mAP on the same images, resampling images with replacement.

    Both evaluations must cover the same images with the same labels (e.g. MV-trained vs ST-trained,
    both scored against MV val labels). This captures evaluation-set sampling noise, not seed noise.
    """
    za, zb = (dict(np.load(cfg["paths"]["runs_root"] / "eval" / n / "per_image.npz")) for n in (name_a, name_b))
    if za["stems"].tolist() != zb["stems"].tolist():
        raise ValueError("the two evaluations cover different images or orders")
    if not (np.array_equal(za["target_cls"], zb["target_cls"]) and np.array_equal(za["target_img"], zb["target_img"])):
        raise ValueError("the two evaluations use different ground truth")
    n = len(za["stems"])
    full = np.ones(n, int)
    point = {k: _resampled_ap(z, full) for k, z in (("a", za), ("b", zb))}
    rng = np.random.default_rng(cfg["seed"])
    d50, d95, dcls = [], [], {c: [] for c in point["a"][2]}
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        (a50, a95, acls), (b50, b95, bcls) = _resampled_ap(za, counts), _resampled_ap(zb, counts)
        d50.append(b50 - a50)
        d95.append(b95 - a95)
        for c in dcls:
            if c in acls and c in bcls:  # same ground truth, so present in both or neither
                dcls[c].append(bcls[c] - acls[c])
    d50, d95 = np.array(d50), np.array(d95)
    names = cfg["dataset"]["classes"]
    per_class = {}
    for c, d in dcls.items():
        d = np.array(d)
        per_class[names[c]] = {
            "a_AP50_95": round(point["a"][2][c], 4), "b_AP50_95": round(point["b"][2][c], 4),
            "diff": round(point["b"][2][c] - point["a"][2][c], 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "share_b_better": round(float((d > 0).mean()), 3), "resamples_with_class": len(d),
        }
    out = {"a": name_a, "b": name_b, "images": n, "n_boot": n_boot, "seed": cfg["seed"],
           "a_mAP50": round(point["a"][0], 4), "a_mAP50_95": round(point["a"][1], 4),
           "b_mAP50": round(point["b"][0], 4), "b_mAP50_95": round(point["b"][1], 4)}
    for k, d, (pa, pb) in (("mAP50", d50, (point["a"][0], point["b"][0])), ("mAP50_95", d95, (point["a"][1], point["b"][1]))):
        out[f"diff_{k}"] = round(pb - pa, 4)
        out[f"diff_{k}_ci95"] = [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
        out[f"share_b_better_{k}"] = round(float((d > 0).mean()), 3)
    out["per_class_AP50_95"] = per_class
    write_result(f"bootstrap_{name_b}_vs_{name_a}", out, cfg)
    return out


def pr_curve(cfg: dict[str, Any], name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Precision at IoU 0.5 on a recall grid, averaged over classes, from runs_root/eval/<name>/per_image.npz."""
    from ultralytics.utils.metrics import ap_per_class

    z = np.load(cfg["paths"]["runs_root"] / "eval" / name / "per_image.npz")
    r = ap_per_class(z["tp"], z["conf"], z["pred_cls"], z["target_cls"])
    ap, recall_grid, precision = r[5], r[10], np.asarray(r[11])
    return recall_grid, precision.mean(0), float(ap[:, 0].mean())


def plot_pr_curves(cfg: dict[str, Any], curves: dict[str, str], title: str, out: Path) -> Path:
    """One precision-recall line per model (IoU 0.5, mean over classes); curves maps label -> eval name."""
    from src import plotting

    fig, ax = plotting.plt.subplots(figsize=(5.2, 4.2))
    for i, (label, name) in enumerate(curves.items()):
        rec, prec, map50 = pr_curve(cfg, name)
        ax.plot(rec, prec, color=plotting.SERIES[i], label=f"{label} (mAP50 {map50:.3f})")
    ax.set(title=title, xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower left")
    return plotting.save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["coco-context", "detector"])
    ap.add_argument("--run", help="detector: run name under runs_root (uses weights/best.pt)")
    ap.add_argument("--labels", choices=["mv", "st"], default="mv", help="detector: label set to score against")
    ap.add_argument("--split", choices=["train", "val"], default="val", help="detector: test is Phase 5 only")
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    if args.what == "coco-context":
        res = coco_context(cfg)
        print(json.dumps({k: v for k, v in res.items() if k != "mapping"}, indent=2))
    else:
        weights = cfg["paths"]["runs_root"] / args.run / "weights" / "best.pt"
        res = detector_metrics(cfg, weights, args.labels, args.split, f"{args.run}__{args.split}__{args.labels}")
        print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
