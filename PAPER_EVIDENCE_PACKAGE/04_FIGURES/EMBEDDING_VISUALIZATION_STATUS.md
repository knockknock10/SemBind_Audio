# Embedding Visualization Status

**Status: per-coordinate intermediates were NOT persisted during E1–E14.**

## What was checked
The canonical final results under `results/final_clean`, `results/final_ablations`,
`results/final_crypto`, `results/final_robustness`, `results/final_analysis`, and
`results/final_tables` were inspected for per-coordinate or per-frame intermediates
(contribution Δ, semantic suitability S, spectral suitability P, joint suitability J,
selected coordinates).

## Finding
The canonical E1–E14 artifacts persist only **per-recording aggregates**:

- `coords_total`, `coords_unique` (count only)
- `mean_delta`, `min_delta`, `max_delta`
- `snr_db`, `bit_error_rate`, `payload_success`, `direct_accuracy`, `roundtrip_accuracy`

No file records the per-coordinate values `(audio_id, frequency_bin, time_frame, delta,
semantic_contribution, semantic_suitability, spectral_suitability, joint_suitability,
selected)`. Therefore **a real embedding-location heatmap cannot be reconstructed from
the stored aggregate results.**

The files under `results/archive_stale/*_embedding/*_coordinates.csv` exist but are
**legacy/stale** (pre-frozen-methodology; classified as stale in
`results/final_audit/artifact_classification.md`). They are explicitly **not** used.

## Resolution: bounded diagnostic visualization run
Because the canonical aggregates are insufficient, a **diagnostic visualization run**
was performed on **3 real LibriSpeech test-clean recordings** using the **frozen**
`final_methodology.py` unchanged (no methodology edits, no E1–E14 changes). It exports
exactly the per-coordinate fields listed above.

- Input audio IDs (real files from `data/raw/librispeech/test-clean/`):
  `61-70968-0021` (2.69 s), `121-127105-0007` (5.79 s), `8230-279154-0013` (14.56 s)
- Script: `07_METHODOLOGY/build_diagnostic_visualization.py`
- Exported data:
  - `06_RAW_PAPER_DATA/diagnostic_embedding_coordinates.csv` (3 × 768 = 2304 rows)
  - `06_RAW_PAPER_DATA/diagnostic_suitability_frames.csv`
  - `06_RAW_PAPER_DATA/diagnostic_run_metadata.json`
- Figures produced: `FIG2_embedding_locations.*`, `FIG2b_suitability_curves.*`

This run is a **figure-production diagnostic only**; it is not a new experiment and does
not replace or alter any canonical E1–E14 result. Its git commit, config hash, and
methodology constants are recorded in `diagnostic_run_metadata.json`.

## Caveat
Because the diagnostic covers 3 files only, FIG 2 illustrates the mechanism; it is
**not** a corpus-level statistic and must not be cited as one.
