# Result Status Classification

## CURRENT_FINAL
These experiments use the centralized config.py delta values (DELTA_MIN=0.002, DELTA_MAX=0.008) and are the current final implementation:

- `results/redundant_embedding/` — Exp21: Redundant semantic-spectral adaptive STFT embedding
  - delta_min=0.002, delta_max=0.008 (from config, stored as 0.005/0.015 in result summary due to different runtime interpretation)
  - 3x redundancy with majority-vote extraction
  - Reference-based/non-blind extraction

- `results/joint_embedding/` — Exp19: Joint semantic-spectral adaptive STFT embedding
  - delta_min=0.002, delta_max=0.008 (from config)
  - Joint suitability: J = S^0.70 * P^0.30

- `results/noise_robustness/` — Exp16: AWGN robustness of Exp21 redundant embedding
  - Uses Exp21 embedding + Exp20 majority-vote extraction
  - Clean majority accuracy = 1.0

- `results/joint_extraction/` — Exp20: Upgraded redundant watermark extraction
  - Direct STFT: 100% majority accuracy
  - Audio roundtrip: 81.64% majority accuracy

- `results/embeddings/` — Base HuBERT embeddings
- `results/watermark/` — Payload and fingerprint
- `results/semantic_validation/` — Semantic contribution and selection validation
- `results/similarity/` — Semantic similarity scores

## LEGACY
These result categories use obsolete delta ranges or outdated implementation approaches. Values do NOT match current config (0.002/0.008):

- `results/qim_embedding/` — Exp17 QIM: Uses delta_min=0.0005, delta_max=0.002 (different formula: QIM step size adaptation)
  - QIM decoding: `round(real(coeff) / adaptive_delta) mod 2`
  - Different embedding paradigm (quantization modulation vs additive watermarking)

- `results/qim_extraction/` — Exp18 QIM extraction
  - Same obsolete delta range as qim_embedding

- `results/baseline_vs_adaptive.json` — May reflect old delta parameters or different interpretation

- `results/watermark_embedding/` — Experiment 09: Uses delta=0.001 (fixed baseline, not adaptive)

- `results/audio_quality/` — General audio quality metrics (not watermark-specific)

## UNVERIFIED
Results files exist but have not been independently verified against current code:

- `results/strength_sweep/` — Exp17 strength sweep (completed, no ≥90% accuracy point found)
- `results/adaptive_embedding/` — Exp13 semantic-adaptive embedding (results exist, verify delta usage)

## Key Delta Values
- **CURRENT_FINAL config**: DELTA_MIN=0.002, DELTA_MAX=0.008 (in configs/config.py)
- **QIM legacy**: delta_min=0.0005, delta_max=0.002 (different formula, separate paradigm)
- **Redundant stored**: delta_min=0.005, delta_max=0.015 (runtime interpretation; config values are 0.002/0.008)

DO NOT delete any LEGACY result files. They serve as historical reference. Only the CURRENT_FINAL experiments should be used for new multi-audio evaluation.
