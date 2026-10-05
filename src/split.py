"""Pseudo-camera group split with a leakage purge.

1. Candidate split. Groups (splits/groups.csv) are shuffled with a seed and laid end to end; each
   group goes to the split whose share of the image line contains the group's midpoint, which keeps
   image ratios close to split.ratios. Every pseudo-camera lands wholly in one split.
2. Purge. The grouping under-links (Phase 1 grids: rejected boundary pairs were mostly one camera),
   so a looser direct-pair test catches what it misses: any val/test image with a train image at
   CLIP >= split.purge.clip_min AND thumbnail r >= split.purge.r_min is moved to train. One pass,
   flagged against the candidate's own train set, direct pairs only (no chaining). With
   unit: group the flagged image's whole pseudo-camera moves, so groups never straddle splits.
3. Seed search. split.seed_search candidates (seed, seed+1, ...) are purged and scored on class
   balance from label counts only: per class, the total-variation distance between its box share
   in train/val/test and split.ratios, averaged over the 14 classes and both label sets (MV, ST).
   The fixed target also penalises candidates whose purge shrinks val and test. Lower is better.
   No model output is involved, so choosing among seeds cannot leak test results.

The split is frozen once splits/test.sha256 exists: make() refuses to overwrite it without
--force, and check_test_checksum() guards the sealed test set in Phase 5.

Outputs:
    make (--write):  splits_dir/{train,val,test}.txt, test.sha256, split_info.json;
                     results/split.json, split_class_counts.csv, split_seed_search.csv
    preview:         results/phase1_split/ (seed search, class counts, verification grids,
                     leakage summary); writes nothing to splits/

Usage:
    python -m src.split --preview [--smoke]        # plan the split, write nothing to splits/
    python -m src.split --write [--smoke] [--force]  # freeze it
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.common import add_common_args, load_config, write_result

SPLIT_NAMES = ("train", "val", "test")


def assign_groups(groups: pd.DataFrame, ratios: dict[str, float], seed: int) -> pd.Series:
    """Split name per image; whole groups only. Index: stem."""
    sizes = groups.groupby("group").size()
    order = sizes.index.to_numpy()[np.random.default_rng(seed).permutation(len(sizes))]
    n = sizes.loc[order].to_numpy()
    mid = (np.cumsum(n) - n / 2) / n.sum()
    edges = np.cumsum([ratios[s] for s in SPLIT_NAMES])[:2]
    group_split = pd.Series(np.array(SPLIT_NAMES)[np.searchsorted(edges, mid, side="right")], index=order)
    return groups.set_index("stem")["group"].map(group_split).rename("split")


@dataclass
class Purge:
    """Purge of val/test images that have a direct train partner (P[i, j] True).

    passes = 1 flags against the candidate's own train set once. passes = 0 repeats until no
    val/test image has a direct partner in train, which also catches near-duplicates of images
    the purge itself moved into train (it closes over direct pairs, so it can cascade).
    """

    stems: list[str]
    P: np.ndarray        # n x n bool, pairs meeting the purge criterion, diagonal False
    group: np.ndarray    # pseudo-camera per stem, same order
    unit: str = "group"  # "group": move the flagged image's whole pseudo-camera; "image": only it
    passes: int = 1      # 1 = one pass; 0 = repeat until stable

    def flagged(self, s: np.ndarray) -> np.ndarray:
        is_train = s == "train"
        return ~is_train & self.P[:, is_train].any(axis=1)

    def apply(self, split: pd.Series) -> tuple[pd.Series, dict[str, int]]:
        s0 = split.loc[self.stems].to_numpy()
        s, flagged_any, n_pass = s0.copy(), np.zeros(len(s0), bool), 0
        while self.passes == 0 or n_pass < self.passes:
            flagged = self.flagged(s)
            if not flagged.any():
                break
            moved = flagged if self.unit == "image" else (s != "train") & np.isin(self.group, self.group[flagged])
            s, flagged_any, n_pass = np.where(moved, "train", s), flagged_any | flagged, n_pass + 1
        moved = s != s0
        return pd.Series(s, index=pd.Index(self.stems, name="stem"), name="split"), {
            "passes_run": n_pass,
            "flagged_images": int(flagged_any.sum()),
            "moved_images": int(moved.sum()),
            "moved_from_val": int((moved & (s0 == "val")).sum()),
            "moved_from_test": int((moved & (s0 == "test")).sum()),
            # val/test images still having a direct partner in the post-purge train
            "residual_flagged": int(self.flagged(s).sum()),
        }


def class_table(counts: pd.DataFrame, split: pd.Series, classes: list[str]) -> pd.DataFrame:
    """Boxes per class (rows) for each label set and split (columns, e.g. mv/test)."""
    c = counts.assign(split=counts["stem"].map(split))
    t = c.groupby(["label_set", "split"])[classes].sum().T
    return t.reindex(columns=pd.MultiIndex.from_product([["mv", "st"], SPLIT_NAMES]), fill_value=0)


def balance_tv(table: pd.DataFrame, ratios: dict[str, float]) -> pd.Series:
    """Per class: total-variation distance between its share of boxes per split and the target ratios.

    `table` is classes x (train, val, test) for one label set. 0 = perfectly proportional.
    Classes with no boxes in the pool get NaN and drop out of the averages.
    """
    target = np.array([ratios[s] for s in SPLIT_NAMES])
    t = table[list(SPLIT_NAMES)]
    share = t.div(t.sum(axis=1).replace(0, np.nan), axis=0)
    return 0.5 * (share - target).abs().sum(axis=1, min_count=1)


def seed_search(groups: pd.DataFrame, counts: pd.DataFrame, classes: list[str], ratios: dict[str, float],
                base_seed: int, n: int, purge: Purge | None = None) -> pd.DataFrame:
    """Score n seeded group splits (purged first, if given) on class balance. Best first."""
    rows = []
    for seed in range(base_seed, base_seed + n):
        split = assign_groups(groups, ratios, seed)
        moved = {}
        if purge is not None:
            split, moved = purge.apply(split)
        table = class_table(counts, split, classes)
        tv = {ls: balance_tv(table[ls], ratios) for ls in ("mv", "st")}
        rows.append({
            "seed": seed,
            "score": round(float((tv["mv"].mean() + tv["st"].mean()) / 2), 4),
            "score_mv": round(float(tv["mv"].mean()), 4), "score_st": round(float(tv["st"].mean()), 4),
            "worst_class_mv": tv["mv"].idxmax(), "worst_class_tv_mv": round(float(tv["mv"].max()), 3),
            **{f"images_{s}": int((split == s).sum()) for s in SPLIT_NAMES},
            **moved,
            "classes_missing_in_val_or_test_mv": int((table["mv"][["val", "test"]] == 0).any(axis=1).sum()),
            "min_test_boxes_mv": int(table[("mv", "test")].min()),
        })
    return pd.DataFrame(rows).sort_values(["score", "seed"], ignore_index=True)


def purge_rules(p: dict[str, Any]) -> list[dict[str, float | None]]:
    """split.purge as a list of {clip_min, r_min} rules, ORed. Accepts `rules:` or the flat keys."""
    if p.get("rules"):
        return p["rules"]
    rules = [{"clip_min": p["clip_min"], "r_min": p["r_min"]}]
    if p.get("r_only_min") is not None:  # high thumbnail r alone, for pairs CLIP misjudges (IR, glare)
        rules.append({"clip_min": None, "r_min": p["r_only_min"]})
    return rules


def build_purge(cfg: dict[str, Any], groups: pd.DataFrame):
    """The Purge for the pool, plus the cluster Context (similarity matrices) it was built from."""
    from src.cluster import load_context

    ctx = load_context(cfg)
    if ctx.stems != groups["stem"].tolist():
        raise ValueError("splits/groups.csv does not match the pool order")
    p = cfg["split"]["purge"]
    P = np.zeros_like(ctx.sim, dtype=bool)
    for rule in purge_rules(p):  # a pair is purge-linked if it meets ANY rule
        m = np.ones_like(P)
        if rule.get("clip_min") is not None:
            m &= ctx.sim >= rule["clip_min"]
        if rule.get("r_min") is not None:
            m &= ctx.R >= rule["r_min"]
        if rule.get("g_min") is not None:
            m &= ctx.G >= rule["g_min"]
        P |= m
    np.fill_diagonal(P, False)
    return Purge(ctx.stems, P, groups["group"].to_numpy(), p["unit"], p.get("passes", 1)), ctx


def plan(cfg: dict[str, Any]) -> dict[str, Any]:
    """Seed search with the purge inside each candidate; the best candidate, not written anywhere."""
    d, ratios, classes = cfg["paths"]["splits_dir"], cfg["split"]["ratios"], cfg["dataset"]["classes"]
    groups = pd.read_csv(d / "groups.csv", dtype={"stem": str})
    counts = pd.read_csv(cfg["paths"]["data_root"] / "instance_counts.csv", dtype={"stem": str})
    purge, ctx = build_purge(cfg, groups)
    search = seed_search(groups, counts, classes, ratios, cfg["seed"], cfg["split"]["seed_search"], purge)
    seed = int(search.loc[0, "seed"])
    split, stats = purge.apply(assign_groups(groups, ratios, seed))
    if purge.unit == "group":  # whole pseudo-cameras only, before and after the purge
        assert groups.assign(s=split.loc[groups["stem"]].to_numpy()).groupby("group")["s"].nunique().max() == 1
    return {"groups": groups, "counts": counts, "purge": purge, "ctx": ctx, "search": search,
            "seed": seed, "split": split, "purge_stats": stats,
            "table": class_table(counts, split, classes)}


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_test_checksum(cfg: dict[str, Any]) -> str:
    """Raise unless splits/test.txt matches the checksum stored when the split was frozen."""
    d = cfg["paths"]["splits_dir"]
    expected = (d / "test.sha256").read_text().split()[0]
    actual = sha256(d / "test.txt")
    if actual != expected:
        raise RuntimeError(f"splits/test.txt has changed since the freeze ({actual} != {expected})")
    return actual


def make(cfg: dict[str, Any], force: bool = False) -> pd.Series:
    """Plan the split and freeze it: splits/*.txt, test.sha256, split_info.json."""
    d = cfg["paths"]["splits_dir"]
    if (d / "test.sha256").exists() and not force:
        raise FileExistsError(f"{d / 'test.sha256'} exists: the split is frozen. Use --force only before Phase 2.")
    p = plan(cfg)
    n_test = int((p["split"] == "test").sum())
    if n_test < cfg["split"]["min_test_images"]:
        raise RuntimeError(f"post-purge test has {n_test} images (< split.min_test_images); not freezing")
    out = cfg["paths"]["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    p["search"].to_csv(out / "split_seed_search.csv", index=False)

    split = p["split"]
    for s in SPLIT_NAMES:
        (d / f"{s}.txt").write_text("\n".join(sorted(split.index[split == s])) + "\n")
    (d / "test.sha256").write_text(f"{sha256(d / 'test.txt')}  test.txt\n")
    (d / "split_info.json").write_text(json.dumps({
        "seed": p["seed"], "balance_score": float(p["search"].loc[0, "score"]),
        "seeds_tried": len(p["search"]), "link": cfg["cluster"]["link"], "purge": cfg["split"]["purge"],
        "purge_stats": p["purge_stats"], "ratios": cfg["split"]["ratios"]}, indent=2) + "\n")
    return split


def load(cfg: dict[str, Any]) -> pd.Series:
    """The frozen split as a Series: stem -> train / val / test."""
    d = cfg["paths"]["splits_dir"]
    parts = [pd.Series(s, index=(d / f"{s}.txt").read_text().split(), name="split") for s in SPLIT_NAMES]
    return pd.concat(parts).rename_axis("stem")


def summarise(cfg: dict[str, Any], split: pd.Series) -> dict[str, Any]:
    """Sizes, composition and class counts per split; writes results/split.json and the CSV table."""
    d, classes = cfg["paths"]["splits_dir"], cfg["dataset"]["classes"]
    groups = pd.read_csv(d / "groups.csv", dtype={"stem": str})
    counts = pd.read_csv(cfg["paths"]["data_root"] / "instance_counts.csv", dtype={"stem": str})
    table = class_table(counts, split, classes)
    cfg["paths"]["results_dir"].mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg["paths"]["results_dir"] / "split_class_counts.csv")
    floor = cfg["split"]["min_test_instances"]
    official = groups.set_index("stem")["official_split"]
    summary = {
        "split_info": json.loads((d / "split_info.json").read_text()) if (d / "split_info.json").exists() else None,
        "ratios_target": cfg["split"]["ratios"],
        "images": {s: int((split == s).sum()) for s in SPLIT_NAMES},
        "image_share": {s: round(float((split == s).mean()), 4) for s in SPLIT_NAMES},
        "pseudo_cameras": {s: int(groups.loc[groups["stem"].map(split) == s, "group"].nunique()) for s in SPLIT_NAMES},
        "official_split_composition": {s: official[split.index[split == s]].value_counts().to_dict() for s in SPLIT_NAMES},
        "classes_missing": {f"{ls}/{s}": [c for c in classes if table.loc[c, (ls, s)] == 0]
                            for ls in ("mv", "st") for s in ("val", "test")},
        f"classes_under_{floor}_test_instances": {ls: {c: int(table.loc[c, (ls, "test")]) for c in classes
                                                      if table.loc[c, (ls, "test")] < floor} for ls in ("mv", "st")},
        "test_sha256": (d / "test.sha256").read_text().split()[0] if (d / "test.sha256").exists() else None,
        "class_counts": {f"{ls}/{s}": {c: int(table.loc[c, (ls, s)]) for c in classes} for ls in ("mv", "st") for s in SPLIT_NAMES},
    }
    write_result("split", summary, cfg)
    return summary


def nearest_train(ctx, split: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every `side` image: its most similar train image by gradient correlation g, and that g."""
    A, B = np.nonzero(split == side)[0], np.nonzero(split == "train")[0]
    sub = ctx.G[np.ix_(A, B)]
    best = sub.argmax(axis=1)
    return A, B[best], sub[np.arange(len(A)), best]


def audit_pairs(ctx, split: np.ndarray, side: str, k: int, exclude: set[frozenset[str]],
                seed: int | None = None) -> list[tuple[int, int, float]]:
    """(train, side, CLIP) pairs for the residual-leakage audit, skipping hand-labelled pairs.

    seed=None: the k most similar surviving pairs by g (worst case). seed given: k random `side`
    images, each with its most similar train image (an unbiased estimate of residual leakage).
    """
    A, B, g = nearest_train(ctx, split, side)
    fresh = np.array([frozenset((ctx.stems[a], ctx.stems[b])) not in exclude for a, b in zip(A, B)])
    order = np.argsort(-g, kind="stable") if seed is None else np.random.default_rng(seed).permutation(len(A))
    order = [o for o in order if fresh[o]][:k]
    if seed is not None:
        order = sorted(order, key=lambda o: -g[o])
    return [(int(B[o]), int(A[o]), float(ctx.sim[A[o], B[o]])) for o in order]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(float(max(0.0, centre - half)), 3), round(float(min(1.0, centre + half)), 3)


def record_audit(cfg: dict[str, Any], same: int, unclear: int, n: int, top20_same: int, top20_unclear: int,
                 note: str, out_name: str = "phase1_split") -> dict[str, Any]:
    """Store the visual audit counts and the residual-leakage estimate in results/<out_name>/summary.json."""
    path = cfg["paths"]["results_dir"] / out_name / "summary.json"
    summary = json.loads(path.read_text())
    summary["audit"] = {
        "method": f"{n} random fresh test images, each with its most similar train image by gradient correlation g; "
                  "pairs judged by eye (same camera / unclear / different); hand-labelled pairs excluded",
        "random_n": n, "random_same_camera": same, "random_unclear": unclear,
        "residual_leakage_estimate": round(same / n, 3), "residual_leakage_95ci": wilson(same, n),
        "residual_leakage_upper_if_unclear_count": round((same + unclear) / n, 3),
        "top20_worst_case_same_camera": top20_same, "top20_worst_case_unclear": top20_unclear,
        "note": note,
    }
    path.write_text(json.dumps(summary, indent=2, default=str))
    return summary["audit"]


def preview(cfg: dict[str, Any], n_pairs: int = 20, n_audit: int = 40, out_name: str = "phase1_split") -> dict[str, Any]:
    """Plan the split and write the evidence to results/<out_name>/. Nothing goes to splits/."""
    from src.cluster import calibration, leakage, pair_grid

    out = cfg["paths"]["results_dir"] / out_name
    out.mkdir(parents=True, exist_ok=True)
    p = plan(cfg)
    search, split, ctx, purge = p["search"], p["split"], p["ctx"], p["purge"]
    search.to_csv(out / "seed_search.csv", index=False)

    groups = p["groups"]
    official = groups["official_split"].to_numpy()
    is_val = official == "val"
    direct = purge.P[np.ix_(is_val, ~is_val)].any(axis=1)
    lk = {
        "official_val_images_in_pool": int(is_val.sum()),
        "grouping_lower_bound": {"rule": cfg["cluster"]["link"], **leakage(official, groups["group"].to_numpy())},
        "purge_criterion_looser": {
            "criterion": "direct pair meeting any of: " + " | ".join(
                " AND ".join(f"{k.removesuffix('_min')} >= {v}" for k, v in rule.items() if v is not None)
                for rule in purge_rules(cfg["split"]["purge"])),
            "val_images_with_train_partner": int(direct.sum()),
            "leakage_val_share": round(float(direct.mean()), 4),
        },
        "calibration": calibration(len(official), float((~is_val).mean())),
    }
    n_test = int((split == "test").sum())
    summary = {
        "link": cfg["cluster"]["link"], "purge": cfg["split"]["purge"], "ratios_pre_purge": cfg["split"]["ratios"],
        "chosen_seed": p["seed"], "balance_score": float(search.loc[0, "score"]),
        "images_post_purge": {s: int((split == s).sum()) for s in SPLIT_NAMES},
        "purge_stats": p["purge_stats"],
        "test_size_over_seeds": {"min": int(search["images_test"].min()), "median": float(search["images_test"].median()),
                                 "max": int(search["images_test"].max())},
        "stop_test_below_min": n_test < cfg["split"]["min_test_images"],
        "leakage": lk,
    }
    write_result(f"{out_name}/summary", summary, cfg)
    if summary["stop_test_below_min"]:
        return summary  # stop before choosing: no grids, no class table

    p["table"].to_csv(out / "class_counts.csv")
    s = split.loc[ctx.stems].to_numpy()
    labelled = out.parent / "phase1_split" / "labelled_pairs.csv"
    exclude = ({frozenset(x) for x in pd.read_csv(labelled, dtype=str)[["stem_a", "stem_b"]].itertuples(index=False)}
               if labelled.exists() else set())
    args = (cfg, ctx.stems, s, ctx.R)
    pair_grid(*args, audit_pairs(ctx, s, "test", n_pairs, exclude),
              f"WORST CASE: {n_pairs} most similar TEST-TRAIN pairs left after the purge (by g)",
              out / "audit_top20_test_train.jpg", G=ctx.G)
    pair_grid(*args, audit_pairs(ctx, s, "test", n_audit, exclude, seed=cfg["seed"]),
              f"AUDIT: {n_audit} random TEST images, each with its most similar TRAIN image (by g)",
              out / f"audit_random{n_audit}_test_train.jpg", G=ctx.G)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true", help="plan only; results/phase1_split/, nothing in splits/")
    mode.add_argument("--write", action="store_true", help="freeze the split into splits/ (writes test.sha256)")
    ap.add_argument("--force", action="store_true", help="overwrite a frozen split (only before Phase 2)")
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)

    if args.preview:
        print(json.dumps(preview(cfg), indent=2))
        return
    s = summarise(cfg, make(cfg, force=args.force))
    for k, v in s.items():
        if k != "class_counts":
            print(f"{k}: {v}")
    print(pd.read_csv(cfg["paths"]["results_dir"] / "split_class_counts.csv", header=[0, 1], index_col=0).to_string())


if __name__ == "__main__":
    main()
