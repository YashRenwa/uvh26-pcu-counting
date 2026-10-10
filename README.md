# UVH-26 vehicle counting for signal timing

**Yash Renwa · BITS ID 2025AG05357** · BITS Pilani AIML ZG525 Computer Vision, EC-1 Assignment 1,
Track B (object detection)

Fine-tune a COCO-pretrained YOLOv8 detector to count vehicles by Indian vehicle class on
Bengaluru Safe City CCTV frames, then convert the counts to Passenger Car Units (PCU, IRC:106-1990) so a
traffic engineer can estimate per-junction demand for green-time allocation.

| Submission | File |
|---|---|
| Report (4 pages) | [`report/Yash_Renwa_2025AG05357_EC1_Report.pdf`](report/Yash_Renwa_2025AG05357_EC1_Report.pdf) |
| Notebook, with the outputs of the full Colab run | [`notebooks/uvh26_signal_counting.ipynb`](notebooks/uvh26_signal_counting.ipynb) |

Both files are byte-identical to the versions uploaded to Taxila.

## Results (sealed test set: 477 frames from unseen pseudo-cameras)

| | MV model (baseline) | STAPLE model (ablation) |
|---|---|---|
| mAP@0.50 (MV labels) | 0.571 | 0.555 |
| mAP@0.50:0.95 (MV labels) | 0.489 | 0.475 |
| PCU error per frame, mean absolute (vs STAPLE labels, conf 0.35) | 2.61 (15.8% of 16.5 PCU) | 2.70 (16.4%) |
| PCU bias per frame (negative = under-count) | −0.93 | −1.25 |

- **Ablation:** training on STAPLE labels instead of majority-vote (MV) labels changes mAP@0.50:0.95 by
  −0.007 (paired bootstrap 95% CI −0.023 to +0.008). No detectable effect.
- **Split:** at least 30.9% of official-val frames share a camera with an official-train frame, so the
  split was rebuilt by pseudo-camera. A 40-frame audit of the new test set found residual leakage of about
  17.5% (95% CI 9–32%).
- **Errors:** occlusion is the leading cause among the 30 worst test frames (details in the report, step 7).

## Method in one paragraph

UVH-26 has no camera ids, and a fixed CCTV camera produces many near-identical frames. Frames are linked
into **pseudo-cameras** when the correlation of their 64×36 gradient-magnitude thumbnails is at least 0.48,
then split 60/20/20 by group, with a purge that moves val/test frames still linked to train (g ≥ 0.36).
YOLOv8s (Ultralytics 8.4) is fine-tuned at 960 px for 34 epochs, all layers trainable (the freeze pilot
favoured it). Two models differ **only** in the training label set (MV vs STAPLE). For counting, each
vehicle keeps one box (top class, class-agnostic NMS) at a confidence threshold frozen on val (0.35), and
counts are weighted by IRC:106-1990 PCU factors.

## Running it

### Colab (how the submitted results were produced)

1. Open `notebooks/uvh26_signal_counting.ipynb` in Colab and choose a **T4 GPU** runtime.
2. Add Colab secrets (key icon in the sidebar):
   - `GITHUB_TOKEN`: needed only while this repository is private, so the setup cell can clone it.
   - `HF_TOKEN` (optional): avoids anonymous Hugging Face rate limits during the image download.
3. Set the run switches in the first code cell:
   - `RUN_TRAINING`: `True` trains both models (about 65 minutes each on a T4) and reuses any run already
     finished on Drive; `False` evaluates the weights saved under `DRIVE_ROOT/runs`.
   - `EVALUATE_TEST`: `True` opens the sealed test set (Phase 5 onward).
4. Run all. The setup cell mounts Google Drive, clones this repo at the pinned commit `ee0e619`, installs
   `requirements.txt` (keeping Colab's CUDA torch) and prints the seed and every library version.

Later commits on `main` add only documentation (this README, docstrings), the report and the notebook
with outputs. The executable code, `configs/` and `splits/` are unchanged since `ee0e619`.

### Local (CPU smoke tests)

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt jupyter
.venv/bin/python -m src.download pool      # annotations, file index, pool list, pool PNGs
.venv/bin/python -m src.prepare            # 1280 px JPEGs, YOLO labels for MV and ST
.venv/bin/python -m src.cluster            # pseudo-cameras, leakage, pair grids
.venv/bin/python -m src.split --preview    # plan the purged group split; writes nothing to splits/
.venv/bin/python -m src.split --write      # freeze it: split lists and test.sha256
.venv/bin/python -m src.evaluate coco-context        # COCO-pretrained YOLOv8s as-is on val
.venv/bin/python -m src.train --labels mv --pilot frozen   # freeze pilot arm (also: --pilot full)
.venv/bin/python -m src.train --pilot-summary        # minutes per epoch, epochs within budget
.venv/bin/python -m src.train --labels mv            # baseline (ablation: --labels st)
.venv/bin/python -m src.evaluate detector --run mv --split train   # generalisation gap (and --split val)
.venv/bin/python -m src.counts               # Phase 4: PCU factors, counts, threshold sweep on val
.venv/bin/python -m src.errors               # Phase 6: worst test images, rendered for tagging
```

Add `--smoke` to any of these to run on a 20-image pool with outputs under `data/smoke/`. Every script
prints its options with `--help`, and each module's docstring explains what it does and why.

All tunable numbers live in `configs/base.yaml`. Paths can be redirected with `UVH_RAW_ROOT`,
`UVH_DATA_ROOT`, `UVH_RUNS_ROOT`, `UVH_RESULTS_DIR` and `UVH_WEIGHTS_DIR` (the notebook points them at Drive).

## Data access

The dataset is **public**: no account, request form or licence click-through is needed.

- **Source:** [UVH-26 on the Hugging Face Hub](https://huggingface.co/datasets/iisc-aim/UVH-26)
  (`iisc-aim/UVH-26`), CC BY 4.0, not gated. Every download is pinned to revision
  `59f82c57821e8a54dc40bc1f42e83909dbad0b70`, so the files cannot change underneath the results.
- **What the project uses:** the four COCO annotation files (MV and STAPLE, train and val) and the
  4,000 images listed in [`splits/pool.csv`](splits/pool.csv), about 13.6 GB of 1080p PNGs out of 90 GB.
  `python -m src.download pool` (or the notebook's data cell) fetches exactly these files.
- **Split used for every number:** [`splits/`](splits/) holds the pool, the pseudo-camera groups and the
  train/val/test lists (3,071 / 452 / 477 frames).

Where the Hub differs from the dataset card:
- Images are under `UVH-26-{Train,Val}/data/NNN/`, not `images/NNN/`.
- STAPLE labels cover only 81% of images (21,726 of 26,646). MV covers all. This project uses only images
  with both, so the two models train on identical images.
- Image ids differ between the MV and ST files; join on `file_name`.

## Reproducibility

- One seed (`configs/base.yaml: seed`, 42) is set for `random`, NumPy, PyTorch, CUDA and Ultralytics.
  The split seed 87 was chosen by a 50-seed search for class balance (`results/split_seed_search.csv`).
- The test list is sealed by checksum: `splits/test.sha256` holds the SHA-256 of `splits/test.txt`
  (`f8181d24…`), and the notebook verifies it before Phase 5. `src/evaluate.py` refuses the test split
  unless called with `allow_test=True`.
- Training on CPU is bit-identical across reruns. CUDA runs use `deterministic=True` (with `warn_only`),
  so small differences are possible.

## Layout

```
configs/base.yaml  seed, sizes, epochs, thresholds: the only place numbers are set
src/
  common.py        config loading, path resolution, seeding, results/*.json writer
  download.py      partial Hub download: annotations, file index, pool list, PNGs
  prepare.py       1280 px JPEG, COCO -> YOLO txt, MV and ST label sets
  cluster.py       CLIP and thumbnail similarities, pseudo-cameras, leakage measurement
  split.py         group split with leakage purge, class-balance check, test checksum
  plotting.py      shared matplotlib style for report figures
  train.py         Ultralytics dataset view, YOLOv8 training, freeze pilot, epoch budget
  evaluate.py      COCO context, Ultralytics val, 2x2 label-set table, paired bootstrap
  counts.py        per-image counts, IRC PCU factors, confidence threshold sweep (Phase 4)
  errors.py        failure mining on test, rendered worst images, cause tally (Phase 6)
notebooks/
  uvh26_signal_counting.ipynb       the submission notebook, with outputs
  runs/colab_run_2026-10-07.ipynb   an earlier Colab run, kept as a record
report/
  Yash_Renwa_2025AG05357_EC1_Report.pdf   the submitted report
  report.html, figures/, fonts/           its source (open in Chrome and print to A4 PDF)
  extract_outputs.py                      dumps notebook outputs to report/outputs/ for number checks
splits/            pool.csv, groups.csv, train/val/test.txt, test.sha256 (the frozen split)
results/           small JSON/CSV outputs of Phase 1 (later phases print theirs in the notebook)
analysis/          hand-made inputs: Phase 6 cause tags
CLAUDE.md          working rules given to Claude Code (sealed test set, ablation discipline)
```

Images, prepared data, weights and Ultralytics `runs/` are git-ignored and live on Google Drive.

## AI assistance

Claude Code wrote the code in this repository, working under the rules in `CLAUDE.md`. I made the
framing, design and analysis decisions, ran the notebook on Colab and tagged the failure cases. The
report's AI declaration gives the details.

## Dataset citation

> A. Sharma, C. Mhatre, S. Gawali, R. Bokkasam, B. Kishore, V. Pattanaik, T. Rambha, A. R. Pinjari,
> V. Kovvali, A. Chakraborty, P. Rathore, R. Krishnapuram and Y. Simmhan. *Towards Image Annotations
> and Accurate Vision Models for Indian Traffic, Preliminary Dataset Release, UVH-26-v1.0.* Technical
> Report, Indian Institute of Science, 2025. arXiv:2511.02563.
