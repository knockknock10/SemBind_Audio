# Paper Claims → Evidence Map

This file is the authoritative map from every intended paper claim to its experiment,
table, figure, source file, and caveat. **If a claim is not listed as SUPPORTED here,
it must not appear in the manuscript.**

All numbers below are from canonical E1–E14 (`results/final_*`). Stale manuscript values
(see correction table at the end) are superseded.

---

## SUPPORTED CLAIMS

### Claim C1 — Clean payload recovery
**Claim:** With the final configuration (joint selection, R=3), all 2620 LibriSpeech
test-clean recordings achieve 100% clean-channel payload recovery after round-trip
(bit accuracy 1.0, BER 0).

- **Experiment:** E1
- **n:** 2620
- **Table:** Table 2 (clean performance)
- **Figure:** —
- **Source:** `results/final_clean/E1/E1_joint_2620.csv`, `E1_summary_2620.json`
- **Caveat:** Random selection under the same protocol had one failure
  (`4446-2275-0028`, BER 0.0039, success 0.99962); report it.

### Claim C2 — Fidelity (SNR) of the final configuration
**Claim:** The final joint configuration attains a mean clean SNR of 75.37 dB
(std 3.77, min 60.14, max 88.24) on all 2620 recordings.

- **Experiment:** E1
- **Table:** Table 2 · **Figure:** —
- **Source:** `results/final_clean/E1/E1_summary_2620.json`
- **Caveat:** SNR is a time-domain distortion proxy; no perceptual metric was measured.

### Claim C3 — Adaptive strength improves clean SNR over fixed strength
**Claim:** Data-adaptive δ (δ = δ_min + (δ_max−δ_min)·J) yields higher mean clean SNR
than a fixed δ=0.005 (74.82 dB vs 74.29 dB) with identical perfect recovery (n=500).

- **Experiment:** E3 · **n:** 500 each
- **Table:** Table 4
- **Source:** `results/final_ablations/E3/E3_fixed_vs_adaptive_500.csv`
- **Caveat:** Both are 100% recovery; the difference is objective distortion only.

### Claim C4 — Feasible semantic-aware selection mechanism
**Claim:** Semantic (HuBERT perturbation-based) and spectral suitability can be combined
multiplicatively into a joint score J that drives coordinate selection and yields a
valid watermark: all sweep settings and ablations recover the 256-bit payload perfectly
in clean conditions.

- **Experiment:** E1 / E4 / E5
- **Table:** Tables 2 / 5 / 6 · **Figure:** FIG1, FIG2, FIG6
- **Source:** `results/final_clean/E1/*`, `results/final_ablations/E4/*`,
  `results/final_ablations/E5/*`
- **Caveat:** "Feasible/valid", not "best" — see C7.

### Claim C5 — R=3 was chosen because it achieved zero clean failures
**Claim:** R=3 achieved zero clean payload failures in the tested N=500 evaluation,
whereas R=1 and R=2 had 2 and 3 failures respectively; therefore R=3 was selected as the
final configuration.

- **Experiment:** E7 · **n:** 500 each
- **Table:** Table 3 · **Figure:** FIG3a/b/c/d
- **Source:** `results/final_ablations/E7/E7_redundancy_500.csv`
- **Caveat:** **Not** "R=3 is universally optimal"; R=3 costs ~4 dB SNR and R=2 is
  non-monotonic vs R=1.

### Claim C6 — Authentication / tamper detection
**Claim:** The 256-bit SHA-256-bound payload verifies on intact watermarked audio
(500/500) and rejects corrupted payload, wrong payload, no-watermark, and truncation
scenarios; truncation of the last half is rejected 500/500.

- **Experiment:** E8 · **n:** 500 scenarios
- **Table:** Table 8
- **Source:** `results/final_crypto/E8/E8_confusion.json`, `E8_detail.csv`
- **Caveat:** TP=500/TN=500 are per-scenario counts over the same 500 files, not a single
  2×2 matrix over 1000 items.

### Claim C7 — Spectral-only selection has the highest clean SNR
**Claim:** Among the ablation strategies, spectral-only selection gives the highest mean
clean SNR (76.63 dB), followed by joint (74.82), random (73.80), and semantic-only
(72.75).

- **Experiment:** E4 (+ E2 joint ref) · **n:** 500 each
- **Table:** Table 5 · **Figure:** FIG6
- **Source:** `results/final_ablations/E4/E4_ablations_500.csv`
- **Caveat:** This **contradicts** any "joint/semantic gives best SNR" claim — see the
  DO-NOT list.

### Claim C8 — Paired differences are statistically significant (large effects)
**Claim:** All six paired method comparisons on per-recording clean SNR are significant
with |t| ≥ 71.9 and |Cohen's d_z| 1.41–4.18 (n=2620).

- **Experiment:** E13 · **n:** 2620
- **Table:** Table 13 · **Figure:** FIG_E13
- **Source:** `results/final_analysis/E13/E13_paired_stats.json`,
  `results/paper_analysis/statistics/E13_effect_sizes.csv`
- **Caveat:** Report p as `< machine precision (< 1e-16)`, NOT "p = 0". Statistical
  significance ≠ practical/perceptual superiority.

### Claim C9 — Duration robustness
**Claim:** Recovery is perfect in every duration bin; mean clean SNR increases with
duration (72.54 dB for <4 s to 80.71 dB for >20 s).

- **Experiment:** E11 · **n:** 2620
- **Table:** Table 11 · **Figure:** FIG7
- **Source:** `results/final_analysis/E11/E11_duration_bins.json`
- **Caveat:** None beyond SNR being an objective proxy.

### Claim C10 — Speaker robustness
**Claim:** All 40 speakers achieve 100% payload recovery; worst-speaker success = 1.0.

- **Experiment:** E12 · **n:** 40 speakers / 2620 recordings
- **Table:** Table 12 · **Figure:** FIG_E12
- **Source:** `results/final_analysis/E12/E12_per_speaker.json`
- **Caveat:** Per-speaker n ranges 32–108.

### Claim C11 — Computational cost (cached-embedding pipeline)
**Claim:** Excluding HuBERT inference, the per-recording pipeline costs ~14.3–15.5 ms
(totals 37.4–40.7 s over 2620 recordings); contribution/allocation/embedding dominate.

- **Experiment:** E14 · **n:** 2620 traces
- **Table:** Table 14
- **Source:** `results/final_analysis/E14/E14_compute_cost.json`
- **Caveat:** **HuBERT forward pass is excluded** (embeddings loaded from cache). Do not
  present as end-to-end runtime.

### Claim C12 — Robustness limitation (honest negative)
**Claim:** The current sparse signed-coefficient formulation is not robust to AWGN or
common signal-processing attacks: payload success is 0 at every tested attack level, with
bit accuracy in 0.50–0.79 (chance ≈ 0.5).

- **Experiment:** E9 / E10 · **n:** 500 per attack
- **Table:** Tables 9 / 10 · **Figure:** FIG8
- **Source:** `results/final_robustness/E9/E9_awgn_500.csv`,
  `results/final_robustness/E10/E10_attacks_500.csv`
- **Caveat:** E9/E10 `snr_db` is **pre-attack** embedding SNR (identical across attacks);
  never label it as attack quality.

---

## DO-NOT MAKE (explicitly forbidden claims)

| Forbidden claim | Why |
|---|---|
| "More redundancy always improves performance" | E7 is non-monotonic (R=2 success 99.40% < R=1 99.60%). |
| "Semantic-only gives the best SNR" | E4: semantic-only has the **lowest** SNR (72.75 dB). |
| "Joint gives the best SNR" | E4: spectral-only is higher (76.63 dB > 74.82 dB). |
| "The method is robust to AWGN / attacks" | E9/E10: payload success = 0 under every attack. |
| "The method is blind" | Extraction uses the reference STFT sign(Re(W)−Re(X)); it is reference-based, not blind. |
| "Cryptographic security is proven" | SHA-256 binding is implemented and verified; no formal security proof is provided. |
| "Perceptually transparent / superior quality" | No MUSHRA/PESQ; SNR is an objective proxy only. |
| "R=3 is universally optimal" | R=3 is the zero-failure choice in the tested N=500 evaluation, not universally optimal. |
| "State-of-the-art / first" | No literature benchmark was run (no related-work comparison). |
| "p = 0" | Underflow; report p < 1e-16. |
| Post-attack SNR improvement | No post-attack SNR was recorded. |

---

## STALE-NUMBER CORRECTION

| Old/stale manuscript value | Canonical value | Source |
|---|---|---|
| Clean SNR ≈ 69.32 dB | Joint mean 75.3731 dB | E1 |
| Replica accuracy ≈ 98.44% | Roundtrip accuracy 100% at R=3 | E1/E7 |
| (any other pre-final clean figure) | Use E1–E14 as the sole source | PAPER_EVIDENCE_PACKAGE |

Canonical clean reference table:

| Method | n | Roundtrip | Payload success | Mean SNR (dB) |
|---|---|---|---|---|
| Joint | 2620 | 1.0 | 1.0 | 75.3731 |
| Semantic | 2620 | 1.0 | 1.0 | 73.3003 |
| Spectral | 2620 | 1.0 | 1.0 | 77.1625 |
| Random | 2620 | 0.9999985 | 0.9996183 | 74.2509 |
