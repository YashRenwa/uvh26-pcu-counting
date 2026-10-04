"""CLIP pseudo-cameras: embed the pool, link near-identical scenes, measure split leakage.

UVH-26 has no camera id. A fixed CCTV camera sees the same background in every frame, so a
CLIP ViT-B/32 image embedding is dominated by the scene. Two images are linked when their
cosine similarity is at least tau; connected components of that graph are pseudo-cameras.
A 64x36 grayscale thumbnail correlation is computed alongside as an independent cross-check.

Leakage = share of official-val pool images whose pseudo-camera also holds an official-train
pool image. The pool is a sample of the dataset, so this undercounts the full split's leakage.

Outputs:
    data_root/clip_embeddings.npz   L2-normalised embeddings (cache)
    splits_dir/groups.csv           stem -> pseudo-camera. Written once, then frozen: later
                                    runs write groups_recomputed.csv and report agreement
    results/cluster.json            group stats, leakage, thumbnail check, tau sweep
    results/figures/                pair grids, leak examples, histograms

Usage:
    python -m src.cluster [--smoke] [--tau T] [--force]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src import plotting
from src.common import add_common_args, load_config, set_seed, write_result
from src.download import select_pool

SWEEP_TAUS = [round(t, 2) for t in np.arange(0.85, 0.985, 0.01)]
SIZE_BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 20), (21, 50), (51, 10**9)]


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_images(cfg: dict[str, Any], stems: list[str]) -> np.ndarray:
    """L2-normalised CLIP image embeddings, (n, 512), cached on data_root."""
    from transformers import CLIPModel, CLIPProcessor

    root, name = cfg["paths"]["data_root"], cfg["cluster"]["clip_model"]
    cache = root / "clip_embeddings.npz"
    if cache.exists():
        z = np.load(cache)
        if str(z["model"]) == name and z["stems"].tolist() == list(stems):
            return z["emb"]

    dev = device()
    model = CLIPModel.from_pretrained(name).eval().to(dev)
    proc = CLIPProcessor.from_pretrained(name)
    bs = cfg["cluster"]["batch_size"]

    def load(stem: str) -> Image.Image:
        return Image.open(root / "images" / f"{stem}.jpg").convert("RGB")

    out = []
    with ThreadPoolExecutor(max_workers=8) as ex, torch.inference_mode():
        for i in tqdm(range(0, len(stems), bs), desc=f"clip ({dev})", mininterval=10):
            px = proc(images=list(ex.map(load, stems[i:i + bs])), return_tensors="pt")["pixel_values"]
            f = model.get_image_features(pixel_values=px.to(dev))
            f = getattr(f, "pooler_output", f)  # transformers 5 returns an output object
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
    emb = np.concatenate(out)
    assert emb.shape[1] == model.config.projection_dim, emb.shape
    np.savez(cache, emb=emb, stems=np.array(stems), model=np.array(name))
    return emb


def thumbnails(cfg: dict[str, Any], stems: list[str]) -> np.ndarray:
    """64x36 grayscale thumbnails, centred and unit-norm, so a dot product is Pearson r."""
    root = cfg["paths"]["data_root"]

    def one(stem: str) -> np.ndarray:
        g = cv2.imread(str(root / "images" / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
        t = cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        t -= t.mean()
        return t / (np.linalg.norm(t) + 1e-8)

    with ThreadPoolExecutor(max_workers=8) as ex:
        return np.stack(list(ex.map(one, stems)))


def components(n: int, ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
    """Connected components of an undirected graph (union-find). Labels 0..G-1 by first image."""
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(ii.tolist(), jj.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)])
    _, first = np.unique(roots, return_index=True)
    order = {r: g for g, r in enumerate(roots[np.sort(first)])}
    return np.array([order[r] for r in roots])


class PairSims:
    """Every unordered image pair (i < j) with its cosine similarity."""

    def __init__(self, sim: np.ndarray):
        self.n = len(sim)
        self.i, self.j = np.triu_indices(self.n, k=1)
        self.s = sim[self.i, self.j]

    def labels(self, tau: float) -> np.ndarray:
        keep = self.s >= tau
        return components(self.n, self.i[keep], self.j[keep])

    def just_above(self, tau: float, k: int) -> list[tuple[int, int, float]]:
        idx = np.nonzero(self.s >= tau)[0]
        idx = idx[np.argsort(self.s[idx], kind="stable")[:k]]
        return [(int(self.i[x]), int(self.j[x]), float(self.s[x])) for x in idx]

    def just_below(self, tau: float, k: int) -> list[tuple[int, int, float]]:
        idx = np.nonzero(self.s < tau)[0]
        idx = idx[np.argsort(-self.s[idx], kind="stable")[:k]]
        return [(int(self.i[x]), int(self.j[x]), float(self.s[x])) for x in idx]


def group_stats(labels: np.ndarray) -> dict[str, Any]:
    sizes = np.bincount(labels)
    return {
        "groups": int(len(sizes)),
        "singletons": int((sizes == 1).sum()),
        "multi_image_groups": int((sizes > 1).sum()),
        "images_in_multi_image_groups": int(sizes[sizes > 1].sum()),
        "largest_group": int(sizes.max()),
        "largest_group_share": round(float(sizes.max() / sizes.sum()), 4),
        "groups_by_size": {(f"{lo}" if lo == hi else f"{lo}+" if hi > 10**8 else f"{lo}-{hi}"):
                           int(((sizes >= lo) & (sizes <= hi)).sum()) for lo, hi in SIZE_BINS},
    }


def leakage(official: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """How many official-val images share a pseudo-camera with an official-train image."""
    is_val = official == "val"
    has_train = np.zeros(labels.max() + 1, bool)
    has_val = np.zeros(labels.max() + 1, bool)
    has_train[labels[~is_val]] = True
    has_val[labels[is_val]] = True
    val_leaked = has_train[labels[is_val]]
    return {
        "val_images": int(is_val.sum()),
        "val_images_sharing_camera_with_train": int(val_leaked.sum()),
        "leakage_val_share": round(float(val_leaked.mean()), 4) if is_val.any() else None,
        "train_images_sharing_camera_with_val": int(has_val[labels[~is_val]].sum()),
        "groups_spanning_train_and_val": int((has_train & has_val).sum()),
    }


def tau_sweep(pairs: PairSims, official: np.ndarray) -> list[dict[str, Any]]:
    """Group stats and leakage at each tau in SWEEP_TAUS: how sensitive is the leakage claim?"""
    rows = []
    for t in SWEEP_TAUS:
        lab = pairs.labels(t)
        stats = {k: v for k, v in group_stats(lab).items() if k != "groups_by_size"}
        rows.append({"tau": t, **stats, "leakage_val_share": leakage(official, lab)["leakage_val_share"]})
    return rows


def weakest_in_group(sim: np.ndarray, labels: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    """For each multi-image group, its least similar member pair; the k lowest overall.

    Connected components can chain different cameras through intermediate frames. If even
    these weakest pairs show one scene, the grouping is not chaining across cameras.
    """
    out = []
    for g in np.nonzero(np.bincount(labels) > 1)[0]:
        m = np.nonzero(labels == g)[0]
        sub = sim[np.ix_(m, m)] + np.eye(len(m)) * 9
        a, b = np.unravel_index(np.argmin(sub), sub.shape)
        out.append((int(m[a]), int(m[b]), float(sub[a, b])))
    return sorted(out, key=lambda t: t[2])[:k]


def cross_split_pairs(sim, official, labels, k, seed) -> list[tuple[int, int, float]]:
    """A random sample of k groups spanning official train and val, most similar pair of each."""
    out = []
    for g in np.unique(labels):
        m = np.nonzero(labels == g)[0]
        tr, va = m[official[m] == "train"], m[official[m] == "val"]
        if len(tr) and len(va):
            sub = sim[np.ix_(tr, va)]
            a, b = np.unravel_index(np.argmax(sub), sub.shape)
            out.append((int(tr[a]), int(va[b]), float(sub[a, b])))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(out), size=min(k, len(out)), replace=False) if out else []
    return [out[p] for p in sorted(pick)]


def pair_grid(cfg, stems, official, R, pairs, title, out: Path) -> Path | None:
    """Render pairs two per row: thumbnails, CLIP cosine, thumbnail r, official splits, stems."""
    if not pairs:
        return None
    root, (w, h), cap, gap = cfg["paths"]["data_root"], (384, 216), 26, 6
    font = cv2.FONT_HERSHEY_SIMPLEX

    def thumb(i: int) -> np.ndarray:
        return cv2.resize(cv2.imread(str(root / "images" / f"{stems[i]}.jpg")), (w, h), interpolation=cv2.INTER_AREA)

    tiles = []
    for i, j, s in pairs:
        tile = np.full((h + cap, 2 * w + gap, 3), 255, np.uint8)
        tile[cap:, :w], tile[cap:, w + gap:] = thumb(i), thumb(j)
        text = f"cos {s:.3f}  r {R[i, j]:.2f}  {official[i]}/{official[j]}  {stems[i]} | {stems[j]}"
        cv2.putText(tile, text, (4, 18), font, 0.5, (11, 11, 11), 1, cv2.LINE_AA)
        tiles.append(tile)
    if len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack([a, np.full((a.shape[0], 3 * gap, 3), 255, np.uint8), b]) for a, b in zip(tiles[::2], tiles[1::2])]
    body = np.vstack([np.vstack([r, np.full((gap, r.shape[1], 3), 255, np.uint8)]) for r in rows])
    head = np.full((40, body.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (4, 28), font, 0.8, (11, 11, 11), 2, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([head, body]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def plot_nn_similarity(nn: np.ndarray, tau: float, out: Path) -> Path:
    fig, ax = plotting.plt.subplots(figsize=(6.4, 3.2))
    lo = min(0.7, float(np.floor(nn.min() * 100) / 100))
    ax.hist(nn, bins=np.arange(lo, 1.0001, 0.005), color=plotting.SERIES[0], edgecolor=plotting.SURFACE, linewidth=0.5)
    ax.axvline(tau, color=plotting.INK, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(tau, ax.get_ylim()[1] * 0.97, f"  τ = {tau:.2f}", va="top", color=plotting.INK, fontsize=9)
    ax.set(title="Each image's similarity to its nearest neighbour in the pool",
           xlabel="CLIP cosine similarity to the most similar other image", ylabel="images")
    ax.yaxis.set_major_locator(plotting.MaxNLocator(integer=True))
    return plotting.save(fig, out)


def plot_group_sizes(labels: np.ndarray, tau: float, out: Path) -> Path:
    sizes = np.bincount(labels)
    names = [f"{lo}" if lo == hi else f"{lo}+" if hi > 10**8 else f"{lo}–{hi}" for lo, hi in SIZE_BINS]
    counts = [int(((sizes >= lo) & (sizes <= hi)).sum()) for lo, hi in SIZE_BINS]
    fig, ax = plotting.plt.subplots(figsize=(6.4, 3.2))
    ax.bar(names, counts, width=0.7, color=plotting.SERIES[0])
    ax.grid(axis="x", visible=False)
    ax.set(title=f"Pseudo-camera sizes at τ = {tau:.2f}: {len(sizes)} groups, largest {sizes.max()} images",
           xlabel="images in the pseudo-camera", ylabel="pseudo-cameras")
    ax.yaxis.set_major_locator(plotting.MaxNLocator(integer=True))
    return plotting.save(fig, out)


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    pairs = len(set(zip(a.tolist(), b.tolist())))
    return pairs == len(set(a.tolist())) == len(set(b.tolist()))


def run(cfg: dict[str, Any], tau: float | None = None, force: bool = False, n_pairs: int = 20) -> dict[str, Any]:
    """Embed, group, measure leakage, write results/cluster.json and the figures.

    If splits/groups.csv is frozen (and not --force), stats, leakage and figures use the frozen
    groups, and the freshly computed grouping is only compared against it. CLIP outputs differ
    slightly across CPU, MPS and CUDA, so a pair right at tau can flip on another machine.
    """
    set_seed(cfg["seed"])
    tau = cfg["cluster"]["tau"] if tau is None else tau
    figs, res = cfg["paths"]["results_dir"] / "figures", {}

    pool = select_pool(cfg)
    stems, official = pool["stem"].tolist(), pool["official_split"].to_numpy()
    emb = embed_images(cfg, stems)
    sim = emb @ emb.T
    T = thumbnails(cfg, stems)
    R = T @ T.T
    pairs = PairSims(sim)
    labels = pairs.labels(tau)

    frozen = cfg["paths"]["splits_dir"] / "groups.csv"
    if frozen.exists() and not force:
        recomputed = labels
        labels = pd.read_csv(frozen, dtype={"stem": str}).set_index("stem").loc[stems, "group"].to_numpy()
        res["frozen_groups_used"] = True
        res["recomputed_groups_identical"] = same_partition(labels, recomputed)
        print(f"Using frozen {frozen}; recomputed grouping identical: {res['recomputed_groups_identical']}")
    else:
        frozen.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"stem": stems, "official_split": official, "group": labels}).to_csv(frozen, index=False)
        res["frozen_groups_used"] = False

    keep = pairs.s >= tau
    nn = np.where(np.eye(len(sim), dtype=bool), -1, sim).max(1)
    res.update({
        "tau": tau, "clip_model": cfg["cluster"]["clip_model"], "device": device(), "images": len(stems),
        "pseudo_cameras": group_stats(labels),
        "leakage": leakage(official, labels),
        "edges": int(keep.sum()),
        "thumbnail_check": {
            "edge_median_r": round(float(np.median(R[pairs.i[keep], pairs.j[keep]])), 3) if keep.any() else None,
            "edges_with_r_below_0.5": int((R[pairs.i[keep], pairs.j[keep]] < 0.5).sum()),
        },
        "nn_similarity_quantiles": {q: round(float(np.quantile(nn, float(q))), 4) for q in ("0.1", "0.25", "0.5", "0.75", "0.9")},
        "tau_sweep": tau_sweep(pairs, official),
    })
    write_result("cluster", res, cfg)

    pair_grid(cfg, stems, official, R, pairs.just_above(tau, n_pairs),
              f"{n_pairs} pairs just ABOVE tau = {tau:.2f}: should be the same camera", figs / "pairs_above_tau.jpg")
    pair_grid(cfg, stems, official, R, pairs.just_below(tau, n_pairs),
              f"{n_pairs} pairs just BELOW tau = {tau:.2f}: should be different cameras", figs / "pairs_below_tau.jpg")
    pair_grid(cfg, stems, official, R, weakest_in_group(sim, labels, n_pairs),
              f"Weakest pair inside each pseudo-camera (tau = {tau:.2f}): chaining check", figs / "pairs_weakest_in_group.jpg")
    pair_grid(cfg, stems, official, R, cross_split_pairs(sim, official, labels, 10, cfg["seed"]),
              "Same pseudo-camera, official TRAIN (left) and official VAL (right)", figs / "leak_examples.jpg")
    plot_nn_similarity(nn, tau, figs / "nn_similarity.png")
    plot_group_sizes(labels, tau, figs / "group_sizes.png")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tau", type=float, default=None, help="override cluster.tau while tuning")
    ap.add_argument("--force", action="store_true", help="overwrite splits/groups.csv (before the freeze only)")
    ap.add_argument("--pairs", type=int, default=20, help="pairs per grid")
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)

    res = run(cfg, tau=args.tau, force=args.force, n_pairs=args.pairs)
    print(json.dumps({k: v for k, v in res.items() if k != "tau_sweep"}, indent=2))
    print(pd.DataFrame(res["tau_sweep"]).to_string(index=False))


if __name__ == "__main__":
    main()
