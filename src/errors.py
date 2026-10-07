"""Failure mining on test (Phase 6): rank images by PCU error and by missed vehicles, render the worst.

Uses the counting pipeline exactly as evaluated in Phase 5: errors.model's predictions at its frozen
threshold, one box per vehicle, matched to errors.label_set labels at IoU >= 0.5 regardless of class.
Each rendered image shows:
    aqua    predicted box, class correct          orange  predicted box, wrong class "PRED (true)"
    red     labelled vehicle that was missed      violet  prediction with no labelled vehicle
and a header with the PCU error, counts and cues that help tagging (night/IR frame, missed vehicles at
the frame edge or small, how many boxes the other label set has). Tagging the cause of each failure
and naming the dominant pattern are the user's; tags live in paths.tags_file (committed), and tally()
summarises them.

Usage:
    python -m src.errors [--smoke]        # the test split; Phase 6 only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src import plotting
from src.common import add_common_args, load_config, write_result
from src.counts import matched_split, pcu_factors, true_counts

ABBR = {"Hatchback": "HB", "Sedan": "SED", "SUV": "SUV", "MUV": "MUV", "Bus": "BUS", "Truck": "TRK",
        "Three-wheeler": "3W", "Two-wheeler": "2W", "LCV": "LCV", "Mini-bus": "MBUS", "Tempo-traveller": "TT",
        "Bicycle": "CYC", "Van": "VAN", "Others": "OTH"}
BGR = {"ok": (122, 175, 27), "wrong": (52, 104, 235), "missed": (72, 73, 227), "fp": (167, 58, 74)}
EDGE, SMALL_PX = 0.005, 24  # frame-edge margin (normalised); "small" = box height under 24 px at 720 px


def image_table(cfg: dict[str, Any], split: str = "test", allow_test: bool = False) -> tuple[pd.DataFrame, list]:
    """One row per image: PCU and count errors, error types, and tagging cues."""
    e, classes = cfg["errors"], cfg["dataset"]["classes"]
    model, ls = e["model"], e["label_set"]
    other = "mv" if ls == "st" else "st"
    factor = np.array([pcu_factors(cfg)["class_factor"][c] for c in classes])
    ims = matched_split(cfg, model, split, ls, cfg["evaluate"]["conf"][model], allow_test)
    stems = [im["stem"] for im in ims]
    n_other = true_counts(cfg, stems, other).sum(1)
    rows = []
    for im, n_o in zip(ims, n_other):
        g, p, g2p, p2g = im["gt_cls"], im["pred_cls"], im["gt_to_pred"], im["pred_to_gt"]
        hit = g2p >= 0
        wrong = np.zeros(len(g), bool)  # matched, but to a different class (safe when there are no predictions)
        wrong[hit] = p[g2p[hit]] != g[hit]
        miss_h = (im["gt_box"][~hit, 3] - im["gt_box"][~hit, 1]) * 720
        miss_box = im["gt_box"][~hit]
        at_edge = ((miss_box[:, 0] < EDGE) | (miss_box[:, 1] < EDGE) | (miss_box[:, 2] > 1 - EDGE)
                   | (miss_box[:, 3] > 1 - EDGE)).sum() if len(miss_box) else 0
        img = cv2.imread(str(cfg["paths"]["data_root"] / "images" / f"{im['stem']}.jpg"))
        hsv = cv2.cvtColor(cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        rows.append({
            "stem": im["stem"], "pcu_true": float(factor[g].sum()), "pcu_pred": float(factor[p].sum()),
            "vehicles_true": len(g), "vehicles_pred": len(p),
            "missed": int((~hit).sum()), "wrong_class": int(wrong.sum()), "false_pos": int((p2g < 0).sum()),
            "pcu_from_wrong_class": float((factor[p[g2p[wrong]]] - factor[g[wrong]]).sum()) if wrong.any() else 0.0,
            "missed_small": int((miss_h < SMALL_PX).sum()), "missed_at_edge": int(at_edge),
            f"boxes_{other}": int(n_o), "night_ir": bool(hsv[..., 1].mean() < 8), "brightness": float(hsv[..., 2].mean()),
        })
    t = pd.DataFrame(rows)
    t.insert(3, "pcu_err", t["pcu_pred"] - t["pcu_true"])
    t.insert(4, "abs_pcu_err", t["pcu_err"].abs())
    return t.round(3), ims


def select_worst(t: pd.DataFrame, n: int) -> pd.DataFrame:
    """Alternate between the worst by |PCU error| and the worst by missed vehicles, skipping repeats."""
    by_pcu = t.sort_values(["abs_pcu_err", "missed"], ascending=False)["stem"].tolist()
    by_miss = t.sort_values(["missed", "abs_pcu_err"], ascending=False)["stem"].tolist()
    picked: dict[str, str] = {}
    i = 0
    while len(picked) < n and i < len(t):
        for src, lst in (("PCU error", by_pcu), ("missed", by_miss)):
            if i < len(lst) and len(picked) < n:
                s = lst[i]
                picked[s] = picked[s] + " + " + src if s in picked else src
        i += 1
    w = t.set_index("stem").loc[list(picked)].reset_index()
    w.insert(0, "rank", range(1, len(w) + 1))
    w.insert(2, "picked_by", [picked[s] for s in w["stem"]])
    return w


def render(cfg: dict[str, Any], im: dict, row: pd.Series, out: Path) -> Path:
    """The image with predictions and labels coloured by outcome, plus a two-line header."""
    classes = cfg["dataset"]["classes"]
    img = cv2.imread(str(cfg["paths"]["data_root"] / "images" / f"{im['stem']}.jpg"))
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    def box(b, colour, text, thick=2):
        x1, y1, x2, y2 = (b * [w, h, w, h]).round().astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick)
        (tw, th), _ = cv2.getTextSize(text, font, 0.42, 1)
        y = max(y1, th + 4)
        cv2.rectangle(img, (x1, y - th - 4), (x1 + tw + 4, y), colour, -1)
        cv2.putText(img, text, (x1 + 2, y - 3), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    for gi, pi in enumerate(im["gt_to_pred"]):
        if pi < 0:
            box(im["gt_box"][gi], BGR["missed"], ABBR[classes[im["gt_cls"][gi]]])
    for pi, gi in enumerate(im["pred_to_gt"]):
        pc = ABBR[classes[im["pred_cls"][pi]]]
        if gi < 0:
            box(im["pred_box"][pi], BGR["fp"], f"{pc}?")
        elif im["pred_cls"][pi] == im["gt_cls"][gi]:
            box(im["pred_box"][pi], BGR["ok"], pc, 1)
        else:
            box(im["pred_box"][pi], BGR["wrong"], f"{pc} ({ABBR[classes[im['gt_cls'][gi]]]})")
    other = [c for c in row.index if c.startswith("boxes_")][0]
    head = np.zeros((58, w, 3), np.uint8)
    l1 = (f"#{row['rank']}  {row['stem']}  [{row['picked_by']}]   PCU true {row['pcu_true']:.1f}  "
          f"pred {row['pcu_pred']:.1f}  error {row['pcu_err']:+.1f}")
    l2 = (f"vehicles {row['vehicles_true']} -> {row['vehicles_pred']}   missed {row['missed']} "
          f"({row['missed_small']} small, {row['missed_at_edge']} at edge)   wrong class {row['wrong_class']}   "
          f"false pos {row['false_pos']}   {other.split('_')[1].upper()} labels: {row[other]} boxes"
          f"{'   NIGHT/IR' if row['night_ir'] else ''}")
    cv2.putText(head, l1, (8, 22), font, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(head, l2, (8, 47), font, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([head, img]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def run(cfg: dict[str, Any], split: str = "test", allow_test: bool = False) -> dict[str, Any]:
    """Rank, select and render the worst images; write the per-image table and a tag template."""
    e = cfg["errors"]
    out = cfg["paths"]["results_dir"] / "phase6"
    out.mkdir(parents=True, exist_ok=True)
    t, ims = image_table(cfg, split, allow_test)
    t.to_csv(out / f"image_errors_{e['model']}_{split}.csv", index=False)
    worst = select_worst(t, e["n_worst"])
    by_stem = {im["stem"]: im for im in ims}
    paths = [render(cfg, by_stem[r["stem"]], r, out / "worst" / f"{r['rank']:02d}_{r['stem']}.jpg") for _, r in worst.iterrows()]
    worst.to_csv(out / f"worst_{e['model']}_{split}.csv", index=False)
    worst[["rank", "stem"]].assign(cause="", note="").to_csv(out / "tags_template.csv", index=False)
    summary = {"model": e["model"], "label_set": e["label_set"], "threshold": cfg["evaluate"]["conf"][e["model"]],
               "split": split, "images": len(t), "worst_n": len(worst),
               "share_of_total_abs_pcu_error_in_worst": round(float(worst["abs_pcu_err"].sum() / t["abs_pcu_err"].sum()), 3),
               "worst": worst[["rank", "stem", "picked_by", "pcu_err", "missed", "wrong_class", "false_pos"]].to_dict(orient="records")}
    write_result(f"phase6/selection_{e['model']}_{split}", summary, cfg)
    return {"table": t, "worst": worst, "images": paths, "summary": summary}


def tally(cfg: dict[str, Any], split: str = "test") -> dict[str, Any] | None:
    """Count the user's cause tags (paths.tags_file) over the selected images; None if not tagged yet."""
    f, causes = cfg["paths"]["tags_file"], cfg["errors"]["causes"]  # causes: short key -> label
    if not f.exists():
        return None
    tags = pd.read_csv(f, dtype={"stem": str}).fillna("")
    unknown = sorted(set(tags["cause"].str.strip().str.lower()) - set(causes) - {""})
    if unknown:
        raise ValueError(f"unknown cause keys {unknown}; allowed: {list(causes)}")
    worst = pd.read_csv(cfg["paths"]["results_dir"] / "phase6" / f"worst_{cfg['errors']['model']}_{split}.csv", dtype={"stem": str})
    m = worst.merge(tags[["stem", "cause", "note"]], on="stem", how="left")
    m["cause"] = m["cause"].fillna("").str.strip().str.lower().map(lambda c: causes.get(c, "untagged"))
    agg = m.groupby("cause").agg(images=("stem", "size"), abs_pcu_err=("abs_pcu_err", "sum"),
                                 missed=("missed", "sum"), wrong_class=("wrong_class", "sum")).sort_values("images", ascending=False)
    agg["share_of_images"] = (agg["images"] / len(m)).round(3)
    res = {"tagged": int((m["cause"] != "untagged").sum()), "of": len(m), "tally": agg.reset_index().to_dict(orient="records")}
    write_result(f"phase6/tally_{cfg['errors']['model']}_{split}", res, cfg)
    fig, ax = plotting.plt.subplots(figsize=(6.4, 0.45 * len(agg) + 1.2))
    a = agg.iloc[::-1]
    ax.barh(a.index, a["images"], color=plotting.SERIES[0], height=0.6)
    for y, v in enumerate(a["images"]):
        ax.text(v + 0.2, y, str(int(v)), va="center", fontsize=8, color=plotting.INK_2)
    ax.grid(axis="y", visible=False)
    ax.set(title=f"Cause of failure, {res['tagged']} worst test images", xlabel="images")
    plotting.save(fig, cfg["paths"]["results_dir"] / "figures" / f"phase6_tally_{split}.png")
    return {**res, "table": m}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    res = run(cfg, "test", allow_test=True)  # Phase 6 works on the test images by design
    print(res["worst"].drop(columns=["brightness"]).to_string(index=False))


if __name__ == "__main__":
    main()
