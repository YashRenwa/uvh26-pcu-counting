"""Per-image vehicle counts, Passenger Car Units (PCU) and the operating confidence threshold. Phase 4.

Counting needs one detection per vehicle, so it has its own prediction pass. Ultralytics' validator
keeps several class labels per box (multi-label NMS): right for mAP, but it would count one vehicle two
or three times. Here each box keeps its top class and NMS is class-agnostic (evaluate.count_agnostic_nms),
at confidence >= the lowest sweep value; predictions are cached on runs_root/eval/<model>__<split>__counts/.

PCU factors follow IRC:106-1990, Table 1: a factor per vehicle type at 5% and at 10%-and-above share of
the traffic stream, interpolated linearly in between. Shares come from the training split's labels
(pcu.composition), never from val or test.

The threshold is chosen on val only: the sweep value with the lowest mean absolute PCU error per image,
separately for each model, against evaluate.count_reference labels. It is frozen in evaluate.conf before
the test set is opened (Phase 5).

Usage:
    python -m src.counts [--smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import plotting
from src.common import add_common_args, file_sha256, load_config, write_result
from src.evaluate import predict, split_stems

MODELS = ("mv", "st")


def pcu_factors(cfg: dict[str, Any]) -> dict[str, Any]:
    """Factor per UVH-26 class, from the IRC categories' shares of the training traffic."""
    p, classes = cfg["pcu"], cfg["dataset"]["classes"]
    comp = p["composition"]
    stems = set(split_stems(cfg, comp["split"]))
    counts = pd.read_csv(cfg["paths"]["labels_root"] / "instance_counts.csv", dtype={"stem": str})
    totals = counts[(counts["label_set"] == comp["label_set"]) & counts["stem"].isin(stems)][classes].sum()
    share = (totals.groupby(p["class_map"]).sum() / totals.sum() * 100).to_dict()
    category_factor = {}
    for cat, (f5, f10) in p["factors"].items():
        s = share.get(cat, 0.0)
        category_factor[cat] = round(f5 + (f10 - f5) * min(max((s - 5) / 5, 0), 1), 4)
    return {
        "source": "IRC:106-1990 Table 1, factors at 5% and >=10% share, linear interpolation",
        "composition": comp, "category_share_pct": {k: round(v, 2) for k, v in share.items()},
        "category_factor": category_factor,
        "class_category": p["class_map"],
        "class_factor": {c: category_factor[p["class_map"][c]] for c in classes},
    }


def true_counts(cfg: dict[str, Any], stems: list[str], label_set: str) -> np.ndarray:
    """(images, classes) labelled boxes per class."""
    counts = pd.read_csv(cfg["paths"]["labels_root"] / "instance_counts.csv", dtype={"stem": str})
    c = counts[counts["label_set"] == label_set].set_index("stem")
    return c.loc[stems, cfg["dataset"]["classes"]].to_numpy()


def count_predictions(cfg: dict[str, Any], model: str, split: str = "val", allow_test: bool = False) -> dict:
    """One-box-per-vehicle predictions of runs_root/<model>/weights/best.pt on `split`, cached."""
    stems = split_stems(cfg, split, allow_test)
    weights = cfg["paths"]["runs_root"] / model / "weights" / "best.pt"
    ev = cfg["evaluate"]
    settings = {"weights_sha256": file_sha256(weights), "split": split, "imgsz": cfg["train"]["imgsz"],
                "iou_nms": ev["iou_nms"], "conf_min": ev["conf_sweep"]["start"], "agnostic_nms": ev["count_agnostic_nms"],
                "boxes": True}
    cache = cfg["paths"]["runs_root"] / "eval" / f"{model}__{split}__counts" / "predictions.npz"
    if cache.exists():
        z = dict(np.load(cache, allow_pickle=False))
        if json.loads(str(z["settings"])) == settings and z["stems"].tolist() == stems:
            return z
    raw = predict(weights, [cfg["paths"]["data_root"] / "images" / f"{s}.jpg" for s in stems], cfg["train"]["imgsz"],
                  conf=settings["conf_min"], iou=ev["iou_nms"], agnostic_nms=ev["count_agnostic_nms"])
    n = [len(raw[s][0]) for s in stems]
    z = {"stems": np.array(stems), "settings": np.array(json.dumps(settings)),
         "pred_cls": np.concatenate([raw[s][0] for s in stems]).astype(int),
         "conf": np.concatenate([raw[s][1] for s in stems]).astype(np.float32),
         "boxes": np.concatenate([raw[s][2] for s in stems]).astype(np.float32).reshape(-1, 4),  # xyxy, normalised
         "pred_img": np.repeat(np.arange(len(stems)), n)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **z)
    return z


def predicted_counts(z: dict, n_classes: int, threshold: float) -> np.ndarray:
    """(images, classes) predictions with confidence >= threshold."""
    keep = z["conf"] >= threshold
    out = np.zeros((len(z["stems"]), n_classes), int)
    np.add.at(out, (z["pred_img"][keep], z["pred_cls"][keep]), 1)
    return out


def match_image(gt_cls: np.ndarray, gt_box: np.ndarray, pred_cls: np.ndarray, pred_conf: np.ndarray,
                pred_box: np.ndarray, iou_min: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one matching of predictions to labelled vehicles, ignoring class.

    Predictions are taken in decreasing confidence; each takes the unmatched label with the highest IoU,
    if that IoU >= iou_min. Returns (gt -> pred index or -1, pred -> gt index or -1).
    """
    from src.evaluate import box_iou

    gt_to_pred, pred_to_gt = np.full(len(gt_cls), -1), np.full(len(pred_cls), -1)
    if len(gt_cls) and len(pred_cls):
        iou = box_iou(gt_box, pred_box)
        for p in np.argsort(-pred_conf, kind="stable"):
            cand = np.where(gt_to_pred < 0, iou[:, p], -1.0)
            g = int(cand.argmax())
            if cand[g] >= iou_min:
                gt_to_pred[g], pred_to_gt[p] = p, g
    return gt_to_pred, pred_to_gt


def matched_split(cfg: dict[str, Any], model: str, split: str, label_set: str, threshold: float,
                  allow_test: bool = False) -> list[dict[str, Any]]:
    """Per image: labelled vehicles, counting predictions at `threshold`, and their matching."""
    from src.evaluate import load_gt

    z = count_predictions(cfg, model, split, allow_test)
    stems = z["stems"].tolist()
    gts = load_gt(cfg, stems, label_set)
    out = []
    for i, stem in enumerate(stems):
        k = (z["pred_img"] == i) & (z["conf"] >= threshold)
        g_cls, g_box = gts[stem]
        p_cls, p_conf, p_box = z["pred_cls"][k], z["conf"][k], z["boxes"][k]
        g2p, p2g = match_image(g_cls, g_box, p_cls, p_conf, p_box)
        out.append({"stem": stem, "gt_cls": g_cls, "gt_box": g_box, "pred_cls": p_cls, "pred_conf": p_conf,
                    "pred_box": p_box, "gt_to_pred": g2p, "pred_to_gt": p2g})
    return out


def confusion(cfg: dict[str, Any], model: str, split: str, label_set: str | None = None,
              allow_test: bool = False) -> dict[str, Any]:
    """Confusion of the counting pipeline at the frozen threshold: one box per vehicle, IoU >= 0.5.

    Rows: labelled class plus "no vehicle" (unmatched predictions). Columns: predicted class plus
    "missed" (unmatched labels). Also aggregated to IRC categories, the level PCU cares about.
    """
    label_set = label_set or cfg["evaluate"]["count_reference"]
    classes, t = cfg["dataset"]["classes"], cfg["evaluate"]["conf"][model]
    nc = len(classes)
    m = np.zeros((nc + 1, nc + 1), int)
    for im in matched_split(cfg, model, split, label_set, t, allow_test):
        for g, p in enumerate(im["gt_to_pred"]):
            m[im["gt_cls"][g], im["pred_cls"][p] if p >= 0 else nc] += 1
        for p in np.nonzero(im["pred_to_gt"] < 0)[0]:
            m[nc, im["pred_cls"][p]] += 1
    cats = list(cfg["pcu"]["factors"])
    cat_of = [cats.index(cfg["pcu"]["class_map"][c]) for c in classes] + [len(cats)]
    mc = np.zeros((len(cats) + 1, len(cats) + 1), int)
    np.add.at(mc, (np.repeat(cat_of, nc + 1), np.tile(cat_of, nc + 1)), m.ravel())
    out = {"model": model, "split": split, "label_set": label_set, "threshold": t,
           "classes": classes, "matrix": m.tolist(), "categories": cats, "category_matrix": mc.tolist()}
    write_result(f"confusion_{model}_{split}_{label_set}", out, cfg)
    return out


def plot_confusion(res: dict[str, Any], out: Path, level: str = "class") -> Path:
    """Row-normalised heatmap (share of each labelled class), plus a bottom row of raw false-positive counts."""
    from matplotlib.colors import LinearSegmentedColormap

    names = res["classes"] if level == "class" else res["categories"]
    m = np.array(res["matrix"] if level == "class" else res["category_matrix"], float)
    n = len(names)
    shares = m[:n] / np.maximum(m[:n].sum(1, keepdims=True), 1)
    grid = np.ma.masked_invalid(np.vstack([shares, np.full((1, n + 1), np.nan)]))
    ramp = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    ramp.set_bad(plotting.GRID)
    fig, ax = plotting.plt.subplots(figsize=(8.6, 7.6) if level == "class" else (5.8, 4.9))
    ax.imshow(grid, cmap=ramp, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    fs = 7 if level == "class" else 8
    for i in range(n):
        for j in range(n + 1):
            v = shares[i, j]
            if v >= 0.005:
                ax.text(j, i, "1" if v >= 0.995 else f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=fs,
                        color="white" if v > 0.55 else plotting.INK)
    for j, v in enumerate(m[n, :n].astype(int)):
        if v:
            ax.text(j, n, str(v), ha="center", va="center", fontsize=fs, color=plotting.INK_2)
    ax.set_xticks(range(n + 1), [*names, "missed"], rotation=60, ha="right")
    ax.set_yticks(range(n + 1), [*(f"{c} ({int(m[i].sum())})" for i, c in enumerate(names)), "false positives (count)"])
    ax.set_xlabel("predicted (counting pipeline: top class, class-agnostic NMS)")
    ax.set_ylabel("labelled class (instances)")
    ax.set_title(f"{'Classes' if level == 'class' else 'IRC categories'}: {res['split']}, "
                 f"{res['model'].upper()}-trained @ {res['threshold']}, vs {res['label_set'].upper()} labels")
    return plotting.save(fig, out)


def thresholds(cfg: dict[str, Any]) -> list[float]:
    s = cfg["evaluate"]["conf_sweep"]
    return [round(float(t), 2) for t in np.arange(s["start"], s["stop"] + 1e-9, s["step"])]


def error_summary(pred: np.ndarray, true: np.ndarray, factor: np.ndarray) -> dict[str, float]:
    """Per-image PCU and total-count errors (pred - true)."""
    pcu_err = pred @ factor - true @ factor
    n_err = pred.sum(1) - true.sum(1)
    return {"pcu_mae": float(np.abs(pcu_err).mean()), "pcu_bias": float(pcu_err.mean()),
            "pcu_rel_mae": float(np.abs(pcu_err).mean() / (true @ factor).mean()),
            "count_mae": float(np.abs(n_err).mean()), "count_bias": float(n_err.mean())}


def per_class_errors(pred: np.ndarray, true: np.ndarray, classes: list[str]) -> pd.DataFrame:
    err = pred - true
    return pd.DataFrame({"true per image": true.mean(0), "count MAE": np.abs(err).mean(0),
                         "signed bias": err.mean(0), "true total": true.sum(0), "predicted total": pred.sum(0)},
                        index=pd.Index(classes, name="class")).round(3)


def run(cfg: dict[str, Any], split: str = "val", models=MODELS) -> dict[str, Any]:
    """Threshold sweep on val for each model; results/counts_<split>.json and the sweep figure."""
    if split != "val":
        raise ValueError("the threshold is chosen on val only; test uses the frozen evaluate.conf in Phase 5")
    classes, ref = cfg["dataset"]["classes"], cfg["evaluate"]["count_reference"]
    fac = pcu_factors(cfg)
    factor = np.array([fac["class_factor"][c] for c in classes])
    stems = split_stems(cfg, split)
    truth = {ls: true_counts(cfg, stems, ls) for ls in ("mv", "st")}
    out = {"split": split, "images": len(stems), "reference_labels": ref, "pcu": fac,
           "true_pcu_per_image_mean": round(float((truth[ref] @ factor).mean()), 3), "models": {}}
    for m in models:
        z = count_predictions(cfg, m, split)
        rows = [{"threshold": t, **error_summary(predicted_counts(z, len(classes), t), truth[ref], factor)}
                for t in thresholds(cfg)]
        sweep = pd.DataFrame(rows)
        best = sweep.loc[sweep["pcu_mae"].idxmin()]
        t = float(best["threshold"])
        pred = predicted_counts(z, len(classes), t)
        out["models"][m] = {
            "sweep": sweep.round(4).to_dict(orient="records"),
            "chosen_threshold": t,
            "at_threshold": {ls: {k: round(v, 4) for k, v in error_summary(pred, truth[ls], factor).items()}
                             for ls in ("mv", "st")},
            "per_class_vs_reference": per_class_errors(pred, truth[ref], classes).to_dict(orient="index"),
            "frozen_conf": cfg["evaluate"]["conf"].get(m),
        }
    write_result(f"counts_{split}", out, cfg)
    plot_sweep(out, cfg["paths"]["results_dir"] / "figures" / f"pcu_threshold_sweep_{split}.png")
    return out


def evaluate_frozen(cfg: dict[str, Any], split: str = "test", allow_test: bool = False,
                    models=MODELS) -> dict[str, Any]:
    """Count and PCU errors at the thresholds frozen in evaluate.conf (Phase 5: split="test", allow_test=True)."""
    conf = cfg["evaluate"]["conf"]
    if any(conf.get(m) is None for m in models):
        raise RuntimeError("evaluate.conf is not frozen; choose the thresholds on val (counts.run) first")
    classes, ref = cfg["dataset"]["classes"], cfg["evaluate"]["count_reference"]
    fac = pcu_factors(cfg)
    factor = np.array([fac["class_factor"][c] for c in classes])
    stems = split_stems(cfg, split, allow_test)
    truth = {ls: true_counts(cfg, stems, ls) for ls in ("mv", "st")}
    out = {"split": split, "images": len(stems), "reference_labels": ref, "pcu": fac,
           "true_pcu_per_image_mean": {ls: round(float((truth[ls] @ factor).mean()), 3) for ls in ("mv", "st")},
           "models": {}}
    for m in models:
        pred = predicted_counts(count_predictions(cfg, m, split, allow_test), len(classes), conf[m])
        out["models"][m] = {
            "threshold": conf[m],
            "errors": {ls: {k: round(v, 4) for k, v in error_summary(pred, truth[ls], factor).items()} for ls in ("mv", "st")},
            "per_class_vs_reference": per_class_errors(pred, truth[ref], classes).to_dict(orient="index"),
        }
    write_result(f"counts_{split}", out, cfg)
    return out


def plot_sweep(res: dict[str, Any], path: Path) -> Path:
    """Mean absolute and mean signed PCU error per image vs confidence threshold, one line per model."""
    names = {"mv": "trained on MV", "st": "trained on STAPLE"}
    fig, axes = plotting.plt.subplots(1, 2, figsize=(9.6, 3.4))
    for i, (m, r) in enumerate(res["models"].items()):
        sw = pd.DataFrame(r["sweep"])
        colour = plotting.SERIES[i]
        t = r["chosen_threshold"]
        best = sw.set_index("threshold").loc[t]
        for ax, col in zip(axes, ("pcu_mae", "pcu_bias")):
            ax.plot(sw["threshold"], sw[col], color=colour, marker="o", markersize=3, label=names[m])
            ax.plot([t], [best[col]], marker="o", markersize=8, color=colour,
                    markeredgecolor=plotting.SURFACE, markeredgewidth=2, zorder=3)
        axes[0].annotate(f"{t:.2f}: {best['pcu_mae']:.2f} PCU", (t, best["pcu_mae"]), textcoords="offset points",
                         xytext=(6, -14 if i else 8), fontsize=8, color=plotting.INK_2)
    # zoom on the useful range: very low thresholds over-count by an order of magnitude
    lim = 3.5 * max(r["at_threshold"][res["reference_labels"]]["pcu_mae"] for r in res["models"].values())
    off = any(max(abs(x["pcu_mae"]), abs(x["pcu_bias"])) > lim for r in res["models"].values() for x in r["sweep"])
    axes[0].set_ylim(0, lim)
    axes[1].set_ylim(-lim, lim)
    if off:
        for ax in axes:
            ax.text(0.99, 0.02, "lowest thresholds off scale", transform=ax.transAxes, ha="right",
                    fontsize=7, color=plotting.MUTED)
    axes[1].axhline(0, color=plotting.BASELINE, linewidth=1)
    axes[0].set(title="Mean absolute PCU error per image", xlabel="confidence threshold", ylabel="PCU")
    axes[1].set(title="Mean signed PCU error per image (+ = over-count)", xlabel="confidence threshold", ylabel="PCU")
    axes[0].legend(loc="upper center")
    fig.suptitle(f"Val, {res['images']} images, true counts from {res['reference_labels'].upper()} labels "
                 f"(mean {res['true_pcu_per_image_mean']:.1f} PCU per image)", x=0.01, ha="left", fontsize=9,
                 color=plotting.INK_2)
    fig.tight_layout()
    return plotting.save(fig, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    res = run(cfg)
    print(json.dumps({"pcu": res["pcu"]["class_factor"], "true_pcu_per_image_mean": res["true_pcu_per_image_mean"],
                      **{m: {"chosen_threshold": r["chosen_threshold"], "at_threshold": r["at_threshold"]}
                         for m, r in res["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
