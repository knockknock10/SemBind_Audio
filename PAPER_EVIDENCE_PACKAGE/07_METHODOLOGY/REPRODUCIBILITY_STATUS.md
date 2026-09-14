# Reproducibility Status — SemBind-Audio

## Provenance issue (documented, NOT silently corrected)

The canonical E1–E14 result JSON files record:

```
"git_commit": "b33000929396cb592a8dcf8bc8450a21ba6dc998"   # "Initial commit"
```

However, `final_methodology.py` and `scripts/final_run.py` were **only committed later**,
in commit `7f6ba676b74a6cc93ca8e5422dced3f881d8ad77` ("results added"). At the time the
results were generated, those two files were present in the working tree but **untracked**.

**Consequence:** the `git_commit` stored in the result metadata does **not** by itself
recreate the exact code state that produced the results. The historical result metadata
has been left **untouched** (per instructions). To make the package self-contained, the
frozen methodology source and the config are copied into this package
(`07_METHODOLOGY/final_methodology.snapshot.py`,
`07_METHODOLOGY/final_config.snapshot.json`) and hashed below.

## Snapshot integrity

| Item | Value |
|---|---|
| Git HEAD at packaging | `7f6ba676b74a6cc93ca8e5422dced3f881d8ad77` |
| Git branch | `main` |
| Commit recorded in result metadata | `b33000929396cb592a8dcf8bc8450a21ba6dc998` |
| Canonical config hash (verified) | `cf10755c13a0ee3147845724f4f097dccc414c2f2b02eed102d6a2bb21231095` |
| `final_methodology.py` SHA-256 | `efb39479b258d7bea2e0aa2ad372c732f2cb9cbefa355fa58ee2030459d8d58a` |
| `final_config.json` SHA-256 | `b1e80db6af4f60ba1ef4d4f4f39bfbef7e5255ec6efe4396c14ce5c4c2e446e4` |

The config hash was independently recomputed as
`SHA-256(json.dumps(config, sort_keys=True, default=str))` and matches the value stored in
all E1–E14 metadata and in `configs/final_config.json`.

## Environment at verification time

| Component | Version |
|---|---|
| Python | 3.14.6 |
| torch | 2.13.0 |
| transformers | 5.14.1 |
| numpy | 2.4.6 |
| scipy | 1.18.0 |
| pandas | 3.0.3 |
| matplotlib | 3.11.0 |
| soundfile | present |

(`results/final_clean/E1/E1_summary_2620.json` records `torch_version: 2.13.0`.)

## Dataset

| Item | Value |
|---|---|
| Dataset | LibriSpeech test-clean (English) |
| Recordings | 2620 |
| Speakers | 40 |
| Total duration | 5.403467 h |
| Duration (mean / median / min / max) | 7.4246 / 5.79 / 1.285 / 34.955 s |
| Raw files | `data/raw/librispeech/test-clean/` (2620 `.flac`, all present) |
| Evaluation manifest | `results/evaluation_manifest.csv` (2620 rows) |
| Checksum manifest id | `c86c7de6b37bae7fd7e882701daf012bc238cb340301b609d6cac6146efc82a6` |
| Ground truth | `results/final_dataset_ground_truth.json` |
| E2–E10 subset | **First 500 rows of `evaluation_manifest.csv`** (NOT the 100-row dev subset) |

## Model

| Item | Value |
|---|---|
| HuBERT identifier | `facebook/hubert-base-ls960` |
| Feature dim | 768 |
| Embedding cache | `results/final_cache/hubert/<audio_id>.npy` (2620 files) |
| Inference in E1–E14 | Cached-embedding load (HuBERT forward excluded from E14 timing) |

## Frozen methodology parameters

| Parameter | Value |
|---|---|
| Sample rate | 16000 Hz |
| STFT n_fft / hop / win | 1024 / 320 / 1024 |
| STFT window / pad / center | hann / reflect / True |
| alpha (semantic weight) | 0.70 |
| beta (spectral weight) | 0.30 |
| delta_min | 0.002 |
| delta_max | 0.008 |
| fixed delta (E3/E6) | 0.005 |
| Payload size | 256 bits (SHA-256 digest) |
| Redundancy R | 3 |
| Boundary excluded frames | 1 |
| Frequency band | 5%–95% of bins |
| Coordinate allocation | `k = b*R + r`; time frame `frames[(b % fi) + r*fi]`, `fi = max(flen//R,1)` (replicas separated in timeline thirds); freq bin `pool[(k*467) % plen]`; all (freq,time) unique |
| Embedding | bipolar additive on real part: bit=1 → Re+δ, bit=0 → Re−δ |
| Extraction | reference-based `sign(Re(W)−Re(X))` + majority vote over R replicas |
| Verification | 256-bit SHA-256 payload digest compare |

## Reproduction command

```
# E1–E14 were produced by scripts/final_run.py (frozen) — DO NOT rerun for the paper.
# Table/figure/statistics regeneration (this package):
.venv/bin/python PAPER_EVIDENCE_PACKAGE/07_METHODOLOGY/build_tables.py
.venv/bin/python PAPER_EVIDENCE_PACKAGE/07_METHODOLOGY/build_statistics.py
.venv/bin/python PAPER_EVIDENCE_PACKAGE/07_METHODOLOGY/build_figures.py
.venv/bin/python PAPER_EVIDENCE_PACKAGE/07_METHODOLOGY/build_diagnostic_visualization.py
```

## Packaging note

`results/` is git-ignored in the parent repository (`.gitignore` line 20). Only the
explicitly selected evidence files under `PAPER_EVIDENCE_PACKAGE/` are committed;
the entire `results/` directory is **not** force-added.
