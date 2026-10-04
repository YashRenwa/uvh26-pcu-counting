"""Pseudo-camera group split: every pseudo-camera goes wholly to train, val or test.

Groups are shuffled with seed + split.seed_offset (the offset exists for the one permitted
re-roll) and laid end to end; each group goes to the split whose share of the image line
contains the group's midpoint, which keeps the image ratios close to split.ratios. Every
class is then checked for instances in val and test, and classes under
split.min_test_instances are flagged as low-confidence.

The split is frozen once splits/test.sha256 exists: this script refuses to overwrite it
without --force, and check_test_checksum() guards the sealed test set in Phase 5.

Outputs:
    splits_dir/{train,val,test}.txt   one stem per line, sorted
    splits_dir/test.sha256            sha256 of test.txt
    results/split.json                sizes, composition, class-count table, flags
    results/split_class_counts.csv    boxes per class per split, MV and ST

Usage:
    python -m src.split [--smoke] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
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


def class_table(counts: pd.DataFrame, split: pd.Series, classes: list[str]) -> pd.DataFrame:
    """Boxes per class (rows) for each label set and split (columns, e.g. mv/test)."""
    c = counts.assign(split=counts["stem"].map(split))
    t = c.groupby(["label_set", "split"])[classes].sum().T
    return t.reindex(columns=pd.MultiIndex.from_product([["mv", "st"], SPLIT_NAMES]))


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
    """Assign pseudo-cameras to splits and write splits/*.txt plus the test checksum."""
    d = cfg["paths"]["splits_dir"]
    if (d / "test.sha256").exists() and not force:
        raise FileExistsError(f"{d / 'test.sha256'} exists: the split is frozen. Use --force only before Phase 2.")
    groups = pd.read_csv(d / "groups.csv", dtype={"stem": str})
    split = assign_groups(groups, cfg["split"]["ratios"], cfg["seed"] + cfg["split"]["seed_offset"])
    assert groups.assign(split=split.loc[groups["stem"]].to_numpy()).groupby("group")["split"].nunique().max() == 1
    for s in SPLIT_NAMES:
        (d / f"{s}.txt").write_text("\n".join(sorted(split.index[split == s])) + "\n")
    (d / "test.sha256").write_text(f"{sha256(d / 'test.txt')}  test.txt\n")
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
        "split_seed": cfg["seed"] + cfg["split"]["seed_offset"],
        "ratios_target": cfg["split"]["ratios"],
        "images": {s: int((split == s).sum()) for s in SPLIT_NAMES},
        "image_share": {s: round(float((split == s).mean()), 4) for s in SPLIT_NAMES},
        "pseudo_cameras": {s: int(groups.loc[groups["stem"].map(split) == s, "group"].nunique()) for s in SPLIT_NAMES},
        "official_split_composition": {s: official[split.index[split == s]].value_counts().to_dict() for s in SPLIT_NAMES},
        "classes_missing": {f"{ls}/{s}": [c for c in classes if table.loc[c, (ls, s)] == 0]
                            for ls in ("mv", "st") for s in ("val", "test")},
        f"classes_under_{floor}_test_instances": {ls: {c: int(table.loc[c, (ls, "test")]) for c in classes
                                                      if table.loc[c, (ls, "test")] < floor} for ls in ("mv", "st")},
        "test_sha256": (d / "test.sha256").read_text().split()[0],
        "class_counts": {f"{ls}/{s}": {c: int(table.loc[c, (ls, s)]) for c in classes} for ls in ("mv", "st") for s in SPLIT_NAMES},
    }
    write_result("split", summary, cfg)
    return summary


def run(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    return summarise(cfg, make(cfg, force=force))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="overwrite a frozen split (only before Phase 2)")
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)

    s = run(cfg, force=args.force)
    for k, v in s.items():
        if k != "class_counts":
            print(f"{k}: {v}")
    print(pd.read_csv(cfg["paths"]["results_dir"] / "split_class_counts.csv", header=[0, 1], index_col=0).to_string())


if __name__ == "__main__":
    main()
