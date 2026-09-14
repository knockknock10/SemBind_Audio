# FINAL_VALIDATION — SemBind-Audio Paper Evidence Package

*Generated 2026-09-14 by `07_METHODOLOGY/validate_package.py`.*

| Check | Status | Detail |
|---|---|---|
| V1-tables-from-canonical | **PASS** | all 14 tables map to existing canonical sources |
| V2-figures-registered | **PASS** | missing=[]; unregistered_png=[] |
| V3-no-stale-69.32 | **PASS** | no occurrence of the stale 69.32 dB clean number in any table/figure/docs |
| V4-no-fabrication | **PASS** | spot recomputation matches canonical CSVs |
| V5-traceability-mapping | **PASS** | all traceability rows map to register experiments |
| V6-no-robustness-overclaim | **PASS** | 'V6-no-robustness-overclaim' clean in evidence artifacts |
| V7-no-monotone-redundancy | **PASS** | 'V7-no-monotone-redundancy' clean in evidence artifacts |
| V8-no-semantic-best-snr | **PASS** | 'V8-no-semantic-best-snr' clean in evidence artifacts |
| V9-no-joint-best-snr | **PASS** | 'V9-no-joint-best-snr' clean in evidence artifacts |
| V10-no-blind-claim | **PASS** | 'V10-no-blind-claim' clean in evidence artifacts |
| V11-no-security-proof | **PASS** | 'V11-no-security-proof' clean in evidence artifacts |
| V12-no-post-attack-snr | **PASS** | 'V12-no-post-attack-snr' clean in evidence artifacts |
| V13-no-pzero | **PASS** | p underflow documented as <1e-16; never reported as p=0 in evidence artifacts |

**Overall: PASS**

## Scope note
Validation V4 spot-recomputes headline values from canonical CSVs inside the package (06_RAW_PAPER_DATA) — it does not rerun experiments. FIG1 (architecture) is a conceptual schematic and FIG2/FIG2b are the bounded 3-file diagnostic; both are explicitly marked, so they do not violate V2.
