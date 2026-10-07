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
- Colab storage: JPEGs, CLIP/thumbnail caches and runs live on Drive (data_root, runs_root);
  YOLO labels and instance_counts.csv live in labels_root (data/labels inside the clone, local
  disk) because writing thousands of small files to Drive takes most of an hour. Never put
  per-image small files on Drive. The setup cell force-checks-out origin/<COMMIT> each run.
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
- Phase 1 FROZEN (2026-10-05, seed 87): splits/ holds pool.csv, groups.csv, train/val/test.txt
  (3,071 / 452 / 477), test.sha256 and split_info.json. Pseudo-cameras link on gradient-magnitude
  correlation g >= 0.48 (64x36 Sobel thumbnails); purge g >= 0.36 until stable. Random audit of 40
  test images: 7 same-camera (residual leakage ~17.5%, 95% CI 8.7-32%). Official-split leakage:
  >= 30.9% (grouping), ~57% (purge criterion). Evidence and rounds: results/phase1_sweep/, phase1_split/.
  Low-confidence test classes (MV < 30 boxes): Mini-bus 21, Tempo-traveller 26, Others 2.
- Phase 2 (2026-10-05): COCO-as-is on val (coarse classes, MV): mAP50 0.274 / mAP50-95 0.181 over
  all 9, 0.493 / 0.326 over the 5 mappable. Freeze pilot (5 ep, no mosaic): freeze=10 0.405 mAP50-95
  at 1.72 min/ep; all trainable 0.433 at 1.87 min/ep -> freeze null. Epochs 34 (worst case 79 min with
  mosaic costed at 1.35x). Baseline run name "mv", ablation "st"; eval via evaluate.detector_metrics.
- Phase 2 baseline "mv" (T4, 65.4 min, 34 epochs, best = last epoch, still improving): val
  mAP50-95 0.524 / mAP50 0.615 (P 0.692, R 0.554); train 0.824 / 0.914 -> gap 0.300. Largest per-class
  gaps on car subtypes (MUV 0.43, SUV 0.39, Sedan 0.32) and Van (0.51); smallest on Three-/Two-wheeler
  and Bus. Run took 1.92 min/epoch: mosaic did not slow the T4 run (GPU-bound), so the budget was
  conservative; keep 34 epochs for the ablation (fairness over peak accuracy).
- Phase 3 code: train.args_diff compares checkpoint train_args (resumed runs rewrite args.yaml with
  resolved 'auto' values); evaluate.detector_metrics saves per-image stats (runs/eval/<name>/
  per_image.npz) for evaluate.paired_bootstrap. Smoke runs never take an optimizer step (2 batches
  per epoch < nbs/batch = 8), so all smoke models equal the initial weights: test logic, not learning.
- Phase 3 results (2026-10-06): ST run 66.3 min, best.pt = epoch 25 (selected on ST val), MV best =
  epoch 34. args_diff: only data differs. 2x2 val mAP50-95 (trained -> scored): MV->MV 0.5245,
  MV->ST 0.4923, ST->MV 0.5131, ST->ST 0.4948. Paired bootstrap ST-MV: vs ST +0.0025 [-0.013, 0.024],
  vs MV -0.0114 [-0.030, 0.014]: no detectable difference. User decisions: headline = STAPLE column;
  second seed skipped. Scoring labels move mAP more (~0.02-0.03) than training labels.
- Phase 4 code: src/counts.py (one box per vehicle: top class + class-agnostic NMS; IRC:106-1990
  factors interpolated by train ST composition; per-model threshold = min val PCU MAE vs ST labels).
  Decided 2026-10-06: proposed class map, IRC interpolation by train ST composition, ST reference,
  per-model threshold. evaluate.conf FROZEN 2026-10-07: mv 0.35, st 0.35 (val, dedup labels: PCU MAE
  2.736 / 2.708 per image = 14.8% / 14.7% of 18.48 PCU; bias -0.50 / -0.88). Do not change it.
- STAPLE JSONs contain exact duplicate boxes (429 in the pool, 0.8%, mostly Sedan/LCV; MV none).
  Ultralytics drops them on load; prepare.py now drops them too (counts and PCU use the same labels).
- Phase 3 extra (2026-10-06): per-class ST-MV vs ST labels: Hatchback -0.039 [-0.065,-0.009] and
  Two-wheeler -0.012 [-0.017,-0.008] exclude 0; others do not (no multiple-comparison correction).
  Last-epoch weights: ST-MV -0.005 [-0.021, 0.019] vs ST, -0.014 [-0.032, 0.011] vs MV.
- Phase 5 code is in the notebook behind EVALUATE_TEST (default False); it refuses to run until
  evaluate.conf is frozen and verifies test.sha256 first.
- Ultralytics' confusion matrix is unusable here: built from the mAP predictions (conf 0.001,
  multi-label NMS, IoU ties broken arbitrarily), it overstates class confusion. The report uses
  counts.confusion: counting pipeline at the frozen threshold, IoU >= 0.5, class-agnostic matching,
  by class and by IRC category. Counting predictions now also store boxes.
- Phase 6 code: src/errors.py ranks test images by |PCU error| and missed vehicles (alternating),
  renders the 30 worst with outcome colours, writes a tag template; the user's tags go in
  analysis/phase6_tags.csv (rank, stem, cause key, note; keys in config errors.causes) and are committed.
  errors.model (mv) awaits the user's decision.
- Colab results_dir is Drive (UVH_RESULTS_DIR = MyDrive/uvh26/results); results from Colab reach the
  repo only when the user downloads them. Python caches imported src/ modules: after a push, Colab
  needs Runtime > Restart session before re-running cells.
- transformers 5.x: `CLIPModel.get_image_features` returns an output object, not a tensor.
  The 512-d projected embedding is `.pooler_output`.
- Ultralytics 8.4 also ships YOLO11/YOLO26; this project uses YOLOv8 (`yolov8*.pt`) per the handout.
