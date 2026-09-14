# SemBind-Audio — Canonical Paper Evidence Package

**This directory is the canonical paper evidence package for SemBind-Audio. It contains
only evidence intended for final manuscript preparation. Legacy, stale, and debugging
results are excluded.**

Source of truth for the manuscript: every number, table, figure, statistical result,
methodology value, and claim below is generated from the **canonical finalized E1–E14
results** (`results/final_clean`, `results/final_ablations`, `results/final_crypto`,
`results/final_robustness`, `results/final_analysis`). The scripts under
`07_METHODOLOGY/` regenerate tables/figures/statistics directly from those canonical
files — no new experiments are run, no numbers are invented, nothing is re-estimated.

## Contents

| Item | Description |
|---|---|
| `01_CANONICAL_EXPERIMENT_REGISTER.csv` | E1–E14: purpose, n, configuration, source file, paper table/figure |
| `02_NUMBER_TRACEABILITY.csv` | Every headline number → experiment → exact source file → caveat |
| `03_TABLES/` | Publication-ready tables (CSV + LaTeX), Tables 1–14 |
| `04_FIGURES/` | Publication figures (PNG 300 dpi + PDF + SVG) + `FIGURE_REGISTER.csv` + `EMBEDDING_VISUALIZATION_STATUS.md` |
| `05_STATISTICS/` | E13 statistical summary (CSV + LaTeX) + interpretation |
| `06_RAW_PAPER_DATA/` | Clean copies of the canonical CSV/JSON artifacts that back every table/figure |
| `07_METHODOLOGY/` | Frozen methodology snapshot, frozen config, environment/provenance record, generation scripts |
| `08_LIMITATIONS/` | `LIMITATIONS.md` — supported negative findings that must appear in the manuscript |
| `09_PAPER_CLAIMS.md` | Claims→evidence map, including explicitly forbidden claims |
| `MANIFEST.csv` | Full inventory of the package |
| `FINAL_VALIDATION.md` | Automated PASS/FAIL validation of the package |

## Canonicity rules

- **Sole evidence source:** this package + the canonical `results/final_*` folders it was
  built from.
- **Excluded:** `results/CURRENT_FINAL/**`, `results/archive_stale/**`,
  `results/integrated_evaluation_*`, `results/*_evaluation_100.*`,
  `results/qim_*`, `results/adaptive_*`, `results/joint_*`, debugging outputs, and any
  manuscript numbers from pre-final experimental states.
- **E2–E10 use the first 500 rows of `evaluation_manifest.csv`, NOT the 100-row dev
  subset.**
- **E9/E10 `snr_db` is pre-attack embedding SNR** — never to be presented as attack
  quality or post-attack SNR.

## Headline canonical results (clean channel)

| Method | n | Roundtrip acc | Payload success | Mean SNR (dB) |
|---|---|---|---|---|
| Joint (final, R=3) | 2620 | 1.0 | 1.0 | **75.3731** |
| Semantic-only | 2620 | 1.0 | 1.0 | 73.3003 |
| Spectral-only | 2620 | 1.0 | 1.0 | 77.1625 |
| Random | 2620 | 0.9999985 | 0.9996183 | 74.2509 |

R redundancy (E7): R=1 99.60% / R=2 99.40% / **R=3 100%** payload success (n=500).
Robustness (E9/E10): payload success **0% under every tested attack** (limitation, L1).

## Regeneration (optional, read-only w.r.t. experiments)

```bash
.venv/bin/python PAPER_EVIDENCE_PACKAGE/07_METHODOLOGY/build_tables.py
.venv/bin/python PAPER_EVIDENCE_PACKAGE/07_METHODOLOGY/build_statistics.py
.venv/bin/python PAPER_EVIDENCE_PACKAGE/07_METHODOLOGY/build_figures.py
.venv/bin/python PAPER_EVIDENCE_PACKAGE/07_METHODOLOGY/build_diagnostic_visualization.py  # FIG2 (3-file diagnostic)
```

**Do not modify E1–E14.** This package is for manuscript preparation only.