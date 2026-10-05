"""Pseudo-cameras: link frames from the same fixed camera and measure the official split's leakage.

UVH-26 has no camera id. Two signals are computed for every pair of pool images:
  - CLIP ViT-B/32 cosine similarity of the image embeddings;
  - Pearson r of 64x36 grayscale thumbnails, which tracks scene layout and background.
A linking rule (an optional CLIP floor and an optional r floor) turns pairs into edges, and the
connected components of that graph are pseudo-cameras. The Phase 1 sweep showed that CLIP alone
chains different cameras together, so the rule is chosen from the evidence sweep (--sweep) and
then fixed in configs/base.yaml (cluster.link).

Leakage = share of official-val pool images whose pseudo-camera also holds an official-train pool
image. The pool samples the dataset, so this undercounts the full split's leakage.

Calibration: if the ~2,800 cameras contributed equally, a pool of n frames holds Poisson(n/2800)
frames per camera. That predicts the number of groups, the singleton share, and the leakage to
expect if the official split was random by image.

Outputs:
    data_root/clip_embeddings.npz   CLIP embeddings (cache)
    splits_dir/groups.csv           (run) stem -> pseudo-camera under cluster.link. Written once,
                                    then frozen: later runs only report agreement with it
    results/cluster.json            (run) rule, group stats, leakage, calibration, rule sweep
    results/figures/                (run) pair grids, leak examples, histograms
    results/phase1_sweep/           (--sweep) rule sweep, calibration, scatter, grids, seed search

Usage:
    python -m src.cluster --sweep [--smoke]     # evidence for the linking-rule decision
    python -m src.cluster [--smoke] [--force]   # pseudo-cameras under the decided cluster.link
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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

SIZE_BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 20), (21, 50), (51, 10**9)]
CAMERAS = 2800  # dataset card: "≈ 2,800 Safe City cameras"


@dataclass(frozen=True)
class Rule:
    """Link a pair when CLIP >= clip_min (if set) and thumbnail r >= r_min (if set)."""

    family: str
    clip_min: float | None
    r_min: float | None
    g_min: float | None = None  # gradient-magnitude correlation floor

    @property
    def name(self) -> str:
        if self.family == "G":
            return f"G_g{self.g_min:.2f}"
        if self.family == "C":
            return f"C_r{self.r_min:.2f}" + ("" if self.clip_min is None else f"_clip{self.clip_min:.2f}")
        return f"{self.family}_tau{self.clip_min:.2f}" + ("" if self.r_min is None else f"_r{self.r_min:.1f}")

    @property
    def text(self) -> str:
        conds = [f"CLIP >= {self.clip_min:.2f}"] if self.clip_min is not None else []
        conds += [f"r >= {self.r_min:.2f}"] if self.r_min is not None else []
        conds += [f"g >= {self.g_min:.2f}"] if self.g_min is not None else []
        return " AND ".join(conds)

    @staticmethod
    def from_config(cfg: dict[str, Any]) -> Rule:
        link = cfg["cluster"]["link"]
        clip_min, r_min, g_min = link.get("clip_min"), link.get("r_min"), link.get("g_min")
        if clip_min is None and r_min is None and g_min is None:
            raise ValueError("cluster.link is not decided yet: set clip_min, r_min and/or g_min in configs/base.yaml")
        if g_min is not None and clip_min is None and r_min is None:
            return Rule("G", None, None, g_min)
        family = "A" if r_min is None else "C" if clip_min is None or clip_min < 0.85 else "B"
        return Rule(family, clip_min, r_min, g_min)


# Phase 1 evidence sweep. A: CLIP only. B: CLIP AND r. C: r first, with an optional loose CLIP floor.
SWEEP_RULES = (
    [Rule("A", t, None) for t in (0.90, 0.92, 0.94, 0.96)]
    + [Rule("B", t, r) for t in (0.88, 0.90, 0.92) for r in (0.6, 0.7, 0.8)]
    + [Rule("C", f, r) for r in (0.40, 0.45, 0.50, 0.55, 0.60) for f in (None, 0.80)]
    + [Rule("G", None, None, g) for g in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)]
)


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


def thumbnail_features(cfg: dict[str, Any], stems: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Two 64x36 grayscale thumbnail signals per image, centred and unit-norm (dot product = Pearson r).

    T  (intensity): tracks scene layout and background, but also lighting.
    Tg (Sobel gradient magnitude): a fixed camera keeps its static edges at the same pixels, and
        gradients ignore brightness and colour, so frames from one camera stay correlated across
        lighting, glare and colour-vs-IR changes that move CLIP and intensity correlation.
    Each image is read once; the result is cached on data_root (Drive reads are slow on Colab).
    """
    root = cfg["paths"]["data_root"]
    cache = root / "thumbnails.npz"
    if cache.exists():
        z = np.load(cache)
        if z["stems"].tolist() == list(stems):
            return z["T"], z["Tg"]

    def unit(v: np.ndarray) -> np.ndarray:
        v = v - v.mean()
        return v / (np.linalg.norm(v) + 1e-8)

    def one(stem: str) -> tuple[np.ndarray, np.ndarray]:
        g = cv2.imread(str(root / "images" / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
        t = cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        m = np.hypot(cv2.Sobel(t, cv2.CV_32F, 1, 0), cv2.Sobel(t, cv2.CV_32F, 0, 1))
        return unit(t.ravel()), unit(m.ravel())

    with ThreadPoolExecutor(max_workers=16) as ex:
        T, Tg = map(np.stack, zip(*ex.map(one, stems)))
    np.savez(cache, T=T, Tg=Tg, stems=np.array(stems))
    return T, Tg


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
    """Every unordered image pair (i < j): CLIP cosine s and thumbnail r, over ALL pairs (no gating)."""

    def __init__(self, sim: np.ndarray, R: np.ndarray, G: np.ndarray):
        self.n = len(sim)
        self.i, self.j = np.triu_indices(self.n, k=1)
        self.s = sim[self.i, self.j]
        self.r = R[self.i, self.j]
        self.g = G[self.i, self.j]

    def linked(self, rule: Rule) -> np.ndarray:
        keep = np.ones(len(self.s), bool)
        if rule.clip_min is not None:
            keep &= self.s >= rule.clip_min
        if rule.r_min is not None:
            keep &= self.r >= rule.r_min
        if rule.g_min is not None:
            keep &= self.g >= rule.g_min
        return keep

    def labels(self, rule: Rule) -> np.ndarray:
        keep = self.linked(rule)
        return components(self.n, self.i[keep], self.j[keep])

    def pick(self, mask: np.ndarray, key: np.ndarray, k: int) -> list[tuple[int, int, float]]:
        """The k pairs inside `mask` with the smallest `key`, as (i, j, CLIP cosine)."""
        idx = np.nonzero(mask)[0]
        idx = idx[np.argsort(key[idx], kind="stable")[:k]]
        return [(int(self.i[x]), int(self.j[x]), float(self.s[x])) for x in idx]


@dataclass
class Context:
    stems: list[str]
    official: np.ndarray
    sim: np.ndarray   # CLIP cosine
    R: np.ndarray     # thumbnail intensity correlation
    G: np.ndarray     # thumbnail gradient-magnitude correlation
    pairs: PairSims


def load_context(cfg: dict[str, Any]) -> Context:
    pool = select_pool(cfg)
    stems, official = pool["stem"].tolist(), pool["official_split"].to_numpy()
    emb = embed_images(cfg, stems)
    sim = emb @ emb.T
    T, Tg = thumbnail_features(cfg, stems)
    R = T @ T.T    # full n x n intensity correlation
    G = Tg @ Tg.T  # full n x n gradient-magnitude correlation
    return Context(stems, official, sim, R, G, PairSims(sim, R, G))


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


def calibration(n_images: int, train_share: float, cameras: int = CAMERAS) -> dict[str, float]:
    """Expected pool statistics if every camera contributed equally (Poisson frames per camera).

    With lam = n_images / cameras: groups = cameras (1 - e^-lam); singleton share = e^-lam;
    and if the official split assigned images at random, a val image's camera holds
    Poisson(lam * train_share) other train images in the pool, so leakage = 1 - e^(-lam * train_share).
    """
    lam = n_images / cameras
    return {
        "cameras_assumed": cameras,
        "frames_per_camera_in_pool": round(lam, 3),
        "groups": round(cameras * (1 - np.exp(-lam))),
        "singletons_pct_of_pool": round(100 * np.exp(-lam), 1),
        "leakage_pct_if_official_split_random_by_image": round(100 * (1 - np.exp(-lam * train_share)), 1),
    }


def sweep_table(pairs: PairSims, official: np.ndarray, calib: dict[str, float], rules=SWEEP_RULES) -> pd.DataFrame:
    """Groups, singletons, largest group and leakage per rule, and distance from the calibration.

    Distance = sqrt((dGroups / groups_cal)^2 + (dSingletons / singletons_cal)^2), relative units.
    """
    g0, s0 = calib["groups"], calib["singletons_pct_of_pool"]
    rows = [{"setting": "calibration", "rule": f"Poisson, {calib['cameras_assumed']:,} equal cameras",
             "groups": g0, "singletons_pct_of_pool": s0,
             "leakage_pct": calib["leakage_pct_if_official_split_random_by_image"]}]
    for rule in rules:
        lab = pairs.labels(rule)
        g, lk = group_stats(lab), leakage(official, lab)
        singles = round(100 * g["singletons"] / pairs.n, 1)
        rows.append({
            "setting": rule.name, "rule": rule.text, "edges": int(pairs.linked(rule).sum()),
            "groups": g["groups"], "singletons_pct_of_pool": singles,
            "largest_group": g["largest_group"], "largest_group_pct_of_pool": round(100 * g["largest_group_share"], 1),
            "leakage_pct": round(100 * lk["leakage_val_share"], 1),
            "d_groups": g["groups"] - g0, "d_singletons_pp": round(singles - s0, 1),
            "calib_distance": round(float(np.hypot((g["groups"] - g0) / g0, (singles - s0) / s0)), 3),
        })
    return pd.DataFrame(rows)


def shortlist(table: pd.DataFrame, k: int = 2, max_largest_pct: float = 2.0) -> list[Rule]:
    """The k rules closest to the calibration among those whose largest group is under max_largest_pct."""
    ok = table[table["largest_group_pct_of_pool"] < max_largest_pct].nsmallest(k, "calib_distance")
    by_name = {r.name: r for r in SWEEP_RULES}
    return [by_name[n] for n in ok["setting"]]


def boundary_pairs(pairs: PairSims, rule: Rule, k: int):
    """Pairs on either side of a rule's boundary: (just linked, just rejected).

    One threshold: the k linked pairs with the lowest score, and the k rejected pairs with the
    highest. Two thresholds: linked = k/2 lowest-CLIP + k/2 lowest-r among linked pairs; rejected =
    k/2 that pass CLIP but fail r (highest r) + k/2 that pass r but fail CLIP (highest CLIP).
    """
    ok = pairs.linked(rule)
    if rule.family == "G":
        return pairs.pick(ok, pairs.g, k), pairs.pick(~ok, -pairs.g, k)
    if rule.r_min is None:
        return pairs.pick(ok, pairs.s, k), pairs.pick(~ok, -pairs.s, k)
    if rule.clip_min is None:
        return pairs.pick(ok, pairs.r, k), pairs.pick(~ok, -pairs.r, k)
    h, pass_s, pass_r = k // 2, pairs.s >= rule.clip_min, pairs.r >= rule.r_min
    above = list(dict.fromkeys(pairs.pick(ok, pairs.s, h) + pairs.pick(ok, pairs.r, h)))
    below = pairs.pick(pass_s & ~pass_r, -pairs.r, h) + pairs.pick(pass_r & ~pass_s, -pairs.s, h)
    return above, below


def weakest_in_group(sim: np.ndarray, metric: np.ndarray, labels: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    """For each multi-image group, its least similar member pair by `metric`; the k lowest overall.

    Connected components can chain different cameras through intermediate frames. If even these
    weakest pairs show one scene, the grouping is not chaining across cameras.
    """
    out = []
    for g in np.nonzero(np.bincount(labels) > 1)[0]:
        m = np.nonzero(labels == g)[0]
        sub = metric[np.ix_(m, m)].copy()
        np.fill_diagonal(sub, np.inf)
        a, b = np.unravel_index(np.argmin(sub), sub.shape)
        out.append((float(sub[a, b]), int(m[a]), int(m[b])))
    return [(i, j, float(sim[i, j])) for _, i, j in sorted(out)[:k]]


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


def pair_grid(cfg, stems, official, R, pairs, title, out: Path, G: np.ndarray | None = None) -> Path | None:
    """Render pairs two per row: thumbnails, CLIP cosine, thumbnail r (and g), splits, stems."""
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
        g = f"  g {G[i, j]:.2f}" if G is not None else ""
        text = f"cos {s:.3f}  r {R[i, j]:.2f}{g}  {official[i]}/{official[j]}  {stems[i]} | {stems[j]}"
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


def render_rule_grids(cfg: dict[str, Any], ctx: Context, rule: Rule, out_dir: Path, n_pairs: int = 20) -> np.ndarray:
    """Above-boundary, below-boundary and weakest-in-group grids for one rule. Returns its labels."""
    above, below = boundary_pairs(ctx.pairs, rule, n_pairs)
    labels = ctx.pairs.labels(rule)
    # weakest by the score the rule links on
    metric = ctx.G if rule.family == "G" else ctx.sim if rule.r_min is None else ctx.R
    args = (cfg, ctx.stems, ctx.official, ctx.R)
    pair_grid(*args, above, f"[{rule.text}] linked pairs nearest the boundary: should be ONE camera",
              out_dir / f"{rule.name}_above.jpg", G=ctx.G)
    pair_grid(*args, below, f"[{rule.text}] rejected pairs nearest the boundary: should be DIFFERENT cameras",
              out_dir / f"{rule.name}_below.jpg", G=ctx.G)
    pair_grid(*args, weakest_in_group(ctx.sim, metric, labels, n_pairs),
              f"[{rule.text}] weakest pair inside each group (chaining check)", out_dir / f"{rule.name}_weakest_in_group.jpg",
              G=ctx.G)
    return labels


def plot_nn_hist(values: np.ndarray, xlabel: str, title: str, out: Path, mark: float | None = None) -> Path:
    fig, ax = plotting.plt.subplots(figsize=(6.4, 3.2))
    lo = float(np.floor(min(values.min(), 0.7) * 100) / 100)
    ax.hist(values, bins=np.arange(lo, 1.0001, 0.005 if lo >= 0.6 else 0.01),
            color=plotting.SERIES[0], edgecolor=plotting.SURFACE, linewidth=0.5)
    if mark is not None:
        ax.axvline(mark, color=plotting.INK, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(mark, ax.get_ylim()[1] * 0.97, f"  {mark:.2f}", va="top", color=plotting.INK, fontsize=9)
    ax.set(title=title, xlabel=xlabel, ylabel="images")
    ax.yaxis.set_major_locator(plotting.MaxNLocator(integer=True))
    return plotting.save(fig, out)


def plot_pair_density(pairs: PairSims, out: Path, vlines=(), hlines=(), floor: float | None = None) -> Path:
    """CLIP cosine vs thumbnail r for pairs (all pairs if floor is None), as a log-density hexbin."""
    from matplotlib.colors import LinearSegmentedColormap

    m = np.ones(len(pairs.s), bool) if floor is None else pairs.s >= floor
    x0 = floor if floor is not None else float(np.floor(pairs.s.min() * 20) / 20)
    ramp = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plotting.plt.subplots(figsize=(7.2, 4.6))
    hb = ax.hexbin(pairs.s[m], pairs.r[m], gridsize=(90, 50), bins="log", cmap=ramp, mincnt=1, linewidths=0,
                   extent=(x0, 1.0, -1.0, 1.0))
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("pairs per cell (log scale)", color=plotting.INK_2)
    cb.outline.set_visible(False)
    ax.grid(False)
    for t in vlines:
        ax.axvline(t, color=plotting.MUTED, linewidth=0.7, linestyle=(0, (3, 3)))
        ax.text(t, 1.005, f"{t:.2f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7, color=plotting.INK_2)
    for r in hlines:
        ax.axhline(r, color=plotting.MUTED, linewidth=0.7, linestyle=(0, (3, 3)))
        ax.text(x0 + 0.003, r + 0.008, f"r = {r:.2f}", ha="left", va="bottom", fontsize=7, color=plotting.INK_2)
    scope = f"All {m.sum():,} image pairs" if floor is None else f"All {m.sum():,} image pairs with CLIP cosine ≥ {floor:.2f}"
    ax.set_title(scope, pad=16)
    ax.set(xlabel="CLIP ViT-B/32 cosine similarity", ylabel="64×36 grayscale thumbnail correlation r",
           xlim=(x0, 1.0), ylim=(max(-1.0, float(pairs.r[m].min()) - 0.02), 1.0))
    return plotting.save(fig, out)


def plot_group_sizes(labels: np.ndarray, rule: Rule, out: Path) -> Path:
    sizes = np.bincount(labels)
    names = [f"{lo}" if lo == hi else f"{lo}+" if hi > 10**8 else f"{lo}–{hi}" for lo, hi in SIZE_BINS]
    counts = [int(((sizes >= lo) & (sizes <= hi)).sum()) for lo, hi in SIZE_BINS]
    fig, ax = plotting.plt.subplots(figsize=(6.4, 3.2))
    ax.bar(names, counts, width=0.7, color=plotting.SERIES[0])
    ax.grid(axis="x", visible=False)
    ax.set(title=f"Pseudo-camera sizes [{rule.text}]: {len(sizes)} groups, largest {sizes.max()} images",
           xlabel="images in the pseudo-camera", ylabel="pseudo-cameras")
    ax.yaxis.set_major_locator(plotting.MaxNLocator(integer=True))
    return plotting.save(fig, out)


def nearest_neighbour(M: np.ndarray) -> np.ndarray:
    return np.where(np.eye(len(M), dtype=bool), -np.inf, M).max(1)


def run_sweep(cfg: dict[str, Any], n_pairs: int = 20) -> pd.DataFrame:
    """Evidence for the linking-rule decision. Writes only to results/phase1_sweep/, never splits/."""
    from src.split import assign_groups, class_table, seed_search

    set_seed(cfg["seed"])
    out = cfg["paths"]["results_dir"] / "phase1_sweep"
    out.mkdir(parents=True, exist_ok=True)
    ctx = load_context(cfg)
    calib = calibration(len(ctx.stems), float((ctx.official == "train").mean()))
    table = sweep_table(ctx.pairs, ctx.official, calib)
    table.to_csv(out / "sweep.csv", index=False)
    short = shortlist(table)

    plot_pair_density(ctx.pairs, out / "clip_vs_thumb_r_all_pairs.png", vlines=(0.80,),
                      hlines=sorted({r.r_min for r in SWEEP_RULES if r.family == "C"}))
    plot_nn_hist(nearest_neighbour(ctx.sim), "CLIP cosine similarity to the most similar other image",
                 "Each image's nearest neighbour by CLIP", out / "nn_similarity.png")
    plot_nn_hist(nearest_neighbour(ctx.R), "thumbnail correlation r with the most correlated other image",
                 "Each image's nearest neighbour by thumbnail r", out / "nn_thumb_r.png")

    classes, ratios = cfg["dataset"]["classes"], cfg["split"]["ratios"]
    counts = pd.read_csv(cfg["paths"]["labels_root"] / "instance_counts.csv", dtype={"stem": str})
    searches = {}
    for rule in short:
        labels = render_rule_grids(cfg, ctx, rule, out, n_pairs)
        groups = pd.DataFrame({"stem": ctx.stems, "official_split": ctx.official, "group": labels})
        search = seed_search(groups, counts, classes, ratios, cfg["seed"], cfg["split"]["seed_search"])
        search.to_csv(out / f"split_search_{rule.name}.csv", index=False)
        best = int(search.loc[0, "seed"])
        class_table(counts, assign_groups(groups, ratios, best), classes).to_csv(
            out / f"split_best_{rule.name}_class_counts.csv")
        searches[rule.name] = search.loc[0].to_dict()

    write_result("phase1_sweep/sweep", {
        "images": len(ctx.stems),
        "official_split": {k: int(v) for k, v in zip(*np.unique(ctx.official, return_counts=True))},
        "clip_model": cfg["cluster"]["clip_model"], "device": device(),
        "pairs": int(len(ctx.pairs.s)),
        "calibration": calib,
        "shortlist_rule": "two settings closest to the calibration with largest group < 2% of pool",
        "shortlist": [r.name for r in short],
        "best_split_seed": searches,
        "rules": table.to_dict(orient="records"),
    }, cfg)
    return table


def run(cfg: dict[str, Any], force: bool = False, n_pairs: int = 20) -> dict[str, Any]:
    """Pseudo-cameras under the decided cluster.link rule; results/cluster.json and figures.

    If splits/groups.csv is frozen (and not --force), stats, leakage and figures use the frozen
    groups, and the freshly computed grouping is only compared against it. CLIP outputs differ
    slightly across CPU, MPS and CUDA, so a pair right at a threshold can flip on another machine.
    """
    set_seed(cfg["seed"])
    rule = Rule.from_config(cfg)
    figs, res = cfg["paths"]["results_dir"] / "figures", {"rule": rule.text}
    ctx = load_context(cfg)
    labels = render_rule_grids(cfg, ctx, rule, figs, n_pairs)

    frozen = cfg["paths"]["splits_dir"] / "groups.csv"
    if frozen.exists() and not force:
        recomputed = labels
        labels = pd.read_csv(frozen, dtype={"stem": str}).set_index("stem").loc[ctx.stems, "group"].to_numpy()
        res["frozen_groups_used"] = True
        res["recomputed_groups_identical"] = same_partition(labels, recomputed)
        print(f"Using frozen {frozen}; recomputed grouping identical: {res['recomputed_groups_identical']}")
    else:
        frozen.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"stem": ctx.stems, "official_split": ctx.official, "group": labels}).to_csv(frozen, index=False)
        res["frozen_groups_used"] = False

    calib = calibration(len(ctx.stems), float((ctx.official == "train").mean()))
    res.update({
        "clip_model": cfg["cluster"]["clip_model"], "device": device(), "images": len(ctx.stems),
        "pseudo_cameras": group_stats(labels),
        "leakage": leakage(ctx.official, labels),
        "calibration": calib,
        "rule_sweep": sweep_table(ctx.pairs, ctx.official, calib).to_dict(orient="records"),
    })
    write_result("cluster", res, cfg)

    pair_grid(cfg, ctx.stems, ctx.official, ctx.R, cross_split_pairs(ctx.G, ctx.official, labels, 10, cfg["seed"]),
              "Same pseudo-camera, official TRAIN (left) and official VAL (right)", figs / "leak_examples.jpg", G=ctx.G)
    plot_nn_hist(nearest_neighbour(ctx.G), "gradient correlation g with the most correlated other image",
                 "Each image's nearest neighbour by gradient correlation g", figs / "nn_gradient_g.png", mark=rule.g_min)
    plot_group_sizes(labels, rule, figs / "group_sizes.png")
    return res


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    pairs = len(set(zip(a.tolist(), b.tolist())))
    return pairs == len(set(a.tolist())) == len(set(b.tolist()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="store_true",
                    help="evidence sweep over linking rules into results/phase1_sweep/; writes nothing to splits/")
    ap.add_argument("--force", action="store_true", help="overwrite splits/groups.csv (before the freeze only)")
    ap.add_argument("--pairs", type=int, default=20, help="pairs per grid")
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)

    if args.sweep:
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print(run_sweep(cfg, n_pairs=args.pairs).to_string(index=False))
        return
    res = run(cfg, force=args.force, n_pairs=args.pairs)
    print(json.dumps({k: v for k, v in res.items() if k != "rule_sweep"}, indent=2))


if __name__ == "__main__":
    main()
