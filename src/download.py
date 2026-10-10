"""Partial download of UVH-26 from the Hugging Face Hub.

The full dataset is about 90 GB of 1080p PNGs, so we fetch the four annotation JSONs plus an
explicit list of images. Image paths come from a file index built with `list_repo_tree`,
because the Hub layout (`data/NNN/`) differs from the dataset card (`images/NNN/`), and each
subfolder holds 5,000 images (17 GB), too coarse to download whole.

The pool is a seeded sample of images that carry both MV and STAPLE labels, stratified by
official split. Sampling is nested: a larger pool.size keeps every image of a smaller one.
The pool list is written once to splits/pool.csv and committed; later runs read it back.

Usage:
    python -m src.download annotations      # the four COCO JSONs
    python -m src.download index            # file_name -> repo path table, from list_repo_tree
    python -m src.download pool [--smoke]   # select the pool and download its PNGs
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tqdm import tqdm

from src.common import add_common_args, load_config

SPLITS = ("train", "val")


def _hub_args(cfg: dict[str, Any]) -> dict[str, str]:
    d = cfg["dataset"]
    return {"repo_id": d["repo_id"], "repo_type": "dataset", "revision": d["revision"]}


def download_annotations(cfg: dict[str, Any]) -> dict[str, dict[str, Path]]:
    """Download the MV and ST COCO JSONs for train and val. Returns {split: {mv|st: path}}."""
    ann = cfg["dataset"]["annotations"]
    paths = {s: {k: cfg["paths"]["raw_root"] / ann[s][k] for k in ("mv", "st")} for s in SPLITS}
    if not all(p.exists() for d in paths.values() for p in d.values()):
        patterns = [ann[s][k] for s in SPLITS for k in ("mv", "st")]
        snapshot_download(**_hub_args(cfg), allow_patterns=patterns, local_dir=cfg["paths"]["raw_root"])
    return paths


def build_file_index(cfg: dict[str, Any], force: bool = False) -> pd.DataFrame:
    """List every image on the Hub at the pinned revision and cache it as file_index.csv."""
    out = cfg["paths"]["raw_root"] / "file_index.csv"
    if out.exists() and not force:
        return pd.read_csv(out, dtype={"subfolder": str})
    api, rows = HfApi(), []
    for split in SPLITS:
        folder = cfg["dataset"]["image_dirs"][split]
        for entry in api.list_repo_tree(**_hub_args(cfg), path_in_repo=folder, recursive=True):
            if not entry.path.endswith(".png"):
                continue
            rows.append({
                "file_name": Path(entry.path).name,
                "official_split": split,
                "subfolder": Path(entry.path).parent.name,
                "repo_path": entry.path,
                "size": entry.size,
            })
    df = pd.DataFrame(rows).sort_values("file_name", ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def annotated_files(cfg: dict[str, Any]) -> dict[str, dict[str, set[str]]]:
    """File names that carry labels, per official split and label set (MV, ST)."""
    paths = download_annotations(cfg)
    files: dict[str, dict[str, set[str]]] = {}
    for split in SPLITS:
        files[split] = {}
        for k, p in paths[split].items():
            with open(p) as f:
                files[split][k] = {im["file_name"] for im in json.load(f)["images"]}
    return files


def eligible_images(cfg: dict[str, Any]) -> pd.DataFrame:
    """File index restricted to images with MV labels (and ST labels if the config requires)."""
    index = build_file_index(cfg)
    files = annotated_files(cfg)
    keep = set()
    for split in SPLITS:
        s = files[split]["mv"] & files[split]["st"] if cfg["pool"]["require_both_labels"] else files[split]["mv"]
        keep |= s
    return index[index["file_name"].isin(keep)].reset_index(drop=True)


def sample_images(cfg: dict[str, Any], n: int, seed: int) -> pd.DataFrame:
    """Seeded, nested sample of n eligible images that keeps the official train/val proportion.

    Each official split gets one fixed permutation and the pool takes a prefix of it, so the
    sample for n is always contained in the sample for any larger n.
    """
    elig = eligible_images(cfg)
    rng = np.random.default_rng(seed)
    n_val = round(n * (elig["official_split"] == "val").mean())
    parts = []
    for split, k in (("train", n - n_val), ("val", n_val)):
        rows = elig[elig["official_split"] == split].sort_values("file_name", ignore_index=True)
        parts.append(rows.iloc[rng.permutation(len(rows))[:k]])
    pool = pd.concat(parts, ignore_index=True)
    pool.insert(0, "stem", pool["file_name"].str.removesuffix(".png"))
    return pool.drop(columns="size")


def select_pool(cfg: dict[str, Any], force: bool = False) -> pd.DataFrame:
    """The pool list: read splits/pool.csv if it exists, else sample it and write it there."""
    out = cfg["paths"]["splits_dir"] / "pool.csv"
    if out.exists() and not force:
        pool = pd.read_csv(out, dtype={"stem": str, "subfolder": str})
        if len(pool) != cfg["pool"]["size"]:
            raise ValueError(f"{out} has {len(pool)} images but pool.size is {cfg['pool']['size']}; "
                             "re-run with --force to resample")
        return pool
    pool = sample_images(cfg, cfg["pool"]["size"], cfg["seed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(out, index=False)
    return pool


def download_images(cfg: dict[str, Any], repo_paths: list[str], workers: int = 16) -> list[Path]:
    """Download the given repo paths into raw_root, skipping files already on disk."""
    raw = cfg["paths"]["raw_root"]
    todo = [p for p in repo_paths if not (raw / p).exists()]

    def fetch(p: str) -> None:
        """Download one file from the pinned Hub revision, retrying with exponential back-off."""
        for attempt in range(5):
            try:
                hf_hub_download(**_hub_args(cfg), filename=p, local_dir=raw)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt * 5)  # back off on Hub rate limits

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(tqdm(ex.map(fetch, todo), total=len(todo), desc="images", mininterval=10))
    return [raw / p for p in repo_paths]


def download_pool(cfg: dict[str, Any], pool: pd.DataFrame, workers: int = 16) -> None:
    """Download PNGs for pool images that are not yet prepared (JPEG already on data_root)."""
    images = cfg["paths"]["data_root"] / "images"
    need = pool[[not (images / f"{s}.jpg").exists() for s in pool["stem"]]]
    print(f"pool {len(pool)} images, {len(pool) - len(need)} already prepared, {len(need)} to download")
    download_images(cfg, need["repo_path"].tolist(), workers=workers)


def main() -> None:
    """CLI: `python -m src.download {annotations,index,pool}`."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["annotations", "index", "pool"])
    ap.add_argument("--force", action="store_true", help="resample the pool even if pool.csv exists")
    ap.add_argument("--workers", type=int, default=16)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)

    if args.what == "annotations":
        for split, d in download_annotations(cfg).items():
            print(split, {k: str(v) for k, v in d.items()})
    elif args.what == "index":
        df = build_file_index(cfg, force=True)
        print(df.groupby(["official_split", "subfolder"]).agg(images=("file_name", "size"), gb=("size", "sum"))
              .assign(gb=lambda x: (x["gb"] / 1e9).round(2)))
    elif args.what == "pool":
        pool = select_pool(cfg, force=args.force)
        print(pool.groupby(["official_split", "subfolder"]).size().rename("images").to_string())
        download_pool(cfg, pool, workers=args.workers)


if __name__ == "__main__":
    main()
