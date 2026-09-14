# Limitations of the Current Evidence

These limitations are **supported by the canonical results**. Every item must be stated
in the manuscript; none may be spun as a strength.

## L1. Robustness (E9, E10)
- **No tested attack preserves payload recovery.** Under every AWGN level (40/30/20/10 dB)
  and every common attack (gain 0.9, resample 8 kHz, MP3 128 kbps, lowpass 4 kHz, AWGN
  20 dB), payload success = **0%** (n=500 per attack).
- Bit accuracy under attack is 0.50–0.79 (chance ≈ 0.5). The sparse signed-coefficient
  additive formulation is highly sensitive to any perturbation of the watermarked
  coefficients.
- **Do not** claim robustness, "resilience", or "state-of-the-art robustness".

## L2. Quality metric (E1–E7, E11, E12)
- Only **SNR** (signal-to-watermark-noise on time-domain audio) is measured.
- **No perceptual metric** (MUSHRA/PESQ) was collected. SNR is an objective proxy and is
  meaningful for the same-signal same-time comparison, not for perceptual transparency.
- Statistical significance (E13) does not imply practical/perceptual superiority.

## L3. Reference-based extraction
- Extraction uses the reference (original) STFT: `sign(Re(W) − Re(X))`.
- This is **not blind** watermarking; it requires owner-side access to the dry source.

## L4. Attack SNR column (E9/E10)
- The `snr_db` field in `E9_awgn_500.csv` / `E10_attacks_500.csv` is the **pre-attack**
  embedding SNR (identical across all rows), computed before the attack signal is added.
- It must **never** be labelled or plotted as post-attack SNR or as an attack-quality
  metric.

## L5. Redundancy claim scope (E7)
- R=3 had zero payload failures in the clean n=500 evaluation; R=1 and R=2 did not.
- E7 is **non-monotonic** (R=2 success rate is below R=1). "More redundancy always helps"
  is false; R=3 was chosen as the configuration with zero observed failures.
- R=3 costs ~4 dB SNR relative to R=1.

## L6. Computational timing (E14)
- Timings are for the **cached-embedding pipeline only** (STFT, contributions, allocation,
  embed+ISTFT, extract STFT).
- **HuBERT forward inference is excluded.** E14 must not be presented as end-to-end cost.

## L7. Single corpus
- All results are on LibriSpeech test-clean (single-domain, clean read speech, 16 kHz).
- No cross-corpus, multi-rate, music, or noisy-speech generalization data.

## L8. Security scope
- The 256-bit payload is a straight SHA-256 digest of the metadata JSON. It provides
  integrity binding and tamper *detection* (E8), **not** an adversarial-security proof
  (no cryptanalysis, no adaptive attacker, no keyed embedding).

## L9. Selection mechanism generality
- Selection ablations and sweeps (E4/E5) are evaluated in **clean conditions only**;
  whether selection improves robustness is untested (and the robustness results show
  degradation across the board).

## L10. Diagnostic visualization scope
- FIG 2 / FIG2b embed-location figures come from a 3-file diagnostic run, not from
  corpus-level intermediates (which were not persisted). Use them illustratively only.