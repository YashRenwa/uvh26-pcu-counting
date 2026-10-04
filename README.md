# UVH-26 vehicle counting for signal timing

Fine-tune a COCO-pretrained YOLOv8 detector to count vehicles by Indian vehicle class on
Bengaluru Safe City CCTV frames, then convert the counts to Passenger Car Units (PCU) so a
traffic engineer can estimate per-junction demand for green-time allocation.

BITS Pilani AIML ZG525 Computer Vision, EC-1 Assignment 1, Track B (object detection).

Two design choices set this apart from a plain fine-tune:
- **Pseudo-camera split.** The official split likely leaks frames from one camera across train and
  val. We group images into pseudo-cameras with CLIP embeddings, measure that leakage, and split by group.
- **Label-merging ablation.** Train on Majority Voting (MV) labels vs STAPLE (ST) labels, everything else identical.

## Running it

**Colab (the real runs).** Open `notebooks/uvh26_signal_counting.ipynb` on a T4 GPU runtime and
run all. The first cells mount Google Drive, clone this repo at a fixed commit, install
`requirements.txt` and print the seed and library versions. `RUN_TRAINING = False` skips training and
evaluates saved weights. Add `HF_TOKEN` (and `GITHUB_TOKEN` if the repo is private) as Colab
secrets; an `HF_TOKEN` avoids anonymous Hub rate limits during the image download.

**Local (CPU smoke tests).**

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt jupyter
.venv/bin/python -m src.download annotations   # four COCO JSONs, ~150 MB
.venv/bin/python -m src.download index         # file list from the Hub
.venv/bin/python -m src.download smoke         # 20 seeded images for smoke tests
```

All tunable numbers live in `configs/base.yaml`. Paths can be redirected with `UVH_RAW_ROOT`,
`UVH_DATA_ROOT`, `UVH_RUNS_ROOT` and `UVH_WEIGHTS_DIR` (the notebook points them at Drive).

## Layout

```
CLAUDE.md          rules for Claude Code sessions (sealed test set, ablation discipline)
configs/base.yaml  seed, sizes, epochs, thresholds: the only place numbers are set
src/
  common.py        config loading, seeding, results/*.json writer
  download.py      partial Hub download: annotations, file index, image lists
  prepare.py       1280 px JPEG, COCO -> YOLO txt, MV and ST label sets       (Phase 1)
  cluster.py       CLIP embeddings, pseudo-cameras, leakage measurement       (Phase 1)
  split.py         group split, class-balance check, split lists              (Phase 1)
  train.py         thin wrapper over Ultralytics train                        (Phase 2)
  evaluate.py      mAP, PR curve, count and PCU error, threshold selection    (Phases 2-5)
  errors.py        failure mining and image grids                             (Phase 6)
notebooks/         the submission notebook
splits/            train.txt, val.txt, test.txt and test.txt's checksum
results/           small JSON/CSV outputs quoted in the report
```

Images, prepared data, weights and Ultralytics `runs/` are git-ignored and live on Google Drive.

## Dataset

[UVH-26](https://huggingface.co/datasets/iisc-aim/UVH-26) by the AIM group at IISc with the Bengaluru
Traffic Police: 26,646 frames (1920x1080 PNG) from about 2,800 Safe City cameras, 14 vehicle classes,
two consensus label sets (MV and STAPLE). Licence: **CC BY 4.0**. Downloads are pinned to Hub revision
`59f82c57821e8a54dc40bc1f42e83909dbad0b70`.

Where the Hub differs from the dataset card:
- Images are under `UVH-26-{Train,Val}/data/NNN/`, not `images/NNN/`.
- STAPLE labels cover only 81% of images (21,726 of 26,646). MV covers all. This project uses only images with both.
- Image ids differ between the MV and ST files; join on `file_name`.

Citation:

> A. Sharma, C. Mhatre, S. Gawali, R. Bokkasam, B. Kishore, V. Pattanaik, T. Rambha, A. R. Pinjari,
> V. Kovvali, A. Chakraborty, P. Rathore, R. Krishnapuram and Y. Simmhan. *Towards Image Annotations
> and Accurate Vision Models for Indian Traffic, Preliminary Dataset Release, UVH-26-v1.0.* Technical
> Report, Indian Institute of Science, 2025. arXiv:2511.02563.
