# Project rules

Goal: YOLOv8 vehicle detection on UVH-26 (Bengaluru CCTV, 14 classes) to count vehicles
per class and convert to Passenger Car Units for signal timing. Assignment: BITS AIML ZG525 EC-1.

## Hard constraints
- Allowed libraries only: numpy, matplotlib, torch, torchvision, opencv, transformers (CLIP),
  ultralytics (YOLOv8), huggingface_hub for download, pandas/pyyaml/tqdm for plumbing.
  No scikit-learn: write clustering and splitting in numpy.
- Must run on a free Colab T4 or CPU. Cap data via configs/base.yaml, never hard-code sizes.
- One seed (configs/base.yaml: seed) set for random, numpy, torch, CUDA and Ultralytics.
- The test split is sealed. Never evaluate on splits/test.txt unless the task says "Phase 5"
  (sealed test evaluation) or later. Model selection, thresholds and freeze choices use val only.
- Ablation discipline: baseline and ablation runs differ ONLY in the training label set
  (MV vs STAPLE). Same config file, same seed, same epochs, same split.
- Every script writes its key numbers to results/*.json so they can be quoted in the report.

## Working style
- Smoke-test every script on CPU with 20 images before handing it to Colab.
- Keep the notebook thin: it calls src/ functions, shows outputs, and explains each step
  in markdown cells that map to the 8 report steps.
- Ask before changing anything in configs/base.yaml after Phase 1 is frozen.
- Start each session by reading this file and the Decision Log summary the user pastes in.
  The user makes framing, analysis and conclusion calls; Claude proposes options and writes code.

## Dataset facts (confirmed against the Hub during setup, 2026-10-05)
- Repo `iisc-aim/UVH-26` (dataset, not gated, CC BY 4.0), pinned to revision
  `59f82c57821e8a54dc40bc1f42e83909dbad0b70`. Always pass this revision when downloading.
- Images live under `UVH-26-{Train,Val}/data/NNN/<id>.png`, NOT `images/` as the card says.
  Train: data/000-003 hold 5,000 each, data/004 holds 1,349 (21,349). Val: data/000 holds 5,000,
  data/001 holds 297 (5,297). Total 26,646 PNGs, about 3.4 MB each, 89.8 GB.
  Whole subfolders are 17 GB, so pools are built from file lists, not folder globs.
- Filenames are numeric ids, globally unique across train and val.
- Annotations (COCO JSON): `UVH-26-Train/UVH-26-{MV,ST}-Train.json`, `UVH-26-Val/UVH-26-{MV,ST}-Val.json`.
  MV = majority voting, ST = STAPLE. The card says both cover the same images; they do not.
  ST covers a strict subset: train 17,387 of 21,349, val 4,339 of 5,297 (21,726 with both).
  Pools are drawn only from images with both label sets (`pool.require_both_labels`).
- Image `id`s differ between the MV and ST files. Always join MV and ST on `file_name`.
- category_id 1..14 (Hatchback ... Others); YOLO class = category_id - 1. All frames 1920x1080.
- MV has 11.8 boxes/image, ST 13.0; on shared images ST has +0.8 boxes on average and the
  counts agree on only 46% of images. Rare classes: Others (352 MV boxes in total), Mini-bus (873),
  Tempo-traveller (1,680), Van (2,186). Expect them to hit the 30-test-instance floor.

## Environment
- Local: `.venv` (Python 3.12, created with uv). Run scripts as `.venv/bin/python -m src.<module>`.
  `--smoke` runs any script on a 20-image pool nested in the real one, with prepared data,
  splits and results under `data/smoke/`. COCO weights in `weights/`; CLIP in the HF cache.
  Notebook kernel for local execution: `uvh26` (installed inside .venv).
- The split is frozen by files, not by recomputation: `splits/pool.csv`, `splits/groups.csv`,
  `splits/{train,val,test}.txt`, `splits/test.sha256`. CLIP differs slightly across CPU, MPS
  and CUDA, so the notebook recomputes groups only to compare. split.py refuses to overwrite
  a split with a checksum unless `--force`, which is allowed only before Phase 2.
- Colab: the notebook installs requirements.txt minus the lines marked `colab-skip` (keeps Colab's
  CUDA torch and avoids a forced restart) and prints every library version. Large files
  (data/, runs/, weights/) live on Google Drive via the UVH_* env vars and are git-ignored.
- Training (src/train.py): the Ultralytics dataset view is rebuilt under `paths.datasets_root`
  (Colab: /content/datasets, local disk) with per-split label dirs, so MV and ST never share a
  label cache. The data yaml lists train and val only. `ensure_run` trains, resumes (last.pt still
  has an optimizer) or skips (finished). Ultralytics 8.4's `cls_remap` (on, explicit in config)
  seeds the Bus/Truck/Bicycle head rows from COCO; identical for MV and ST. CPU reruns are
  bit-identical; CUDA uses deterministic=True with warn_only, so small differences are possible.
- src/evaluate.py refuses split="test" unless allow_test=True; only a task that says Phase 5 may set it.
- Phase 1 status (2026-10-05): pool and grouping (rule B) decided; the split is NOT frozen. The
  purge left same-camera pairs in test (colour vs IR, glare, low CLIP). On ~90 hand-labelled hard
  pairs, gradient-magnitude correlation (AUC 0.97) and edge overlap (0.96) beat CLIP (0.80) and
  intensity thumbnail r (0.71). Awaiting the user's decision on the linking/purge signal.
- transformers 5.x: `CLIPModel.get_image_features` returns an output object, not a tensor.
  The 512-d projected embedding is `.pooler_output`.
- Ultralytics 8.4 also ships YOLO11/YOLO26; this project uses YOLOv8 (`yolov8*.pt`) per the handout.
