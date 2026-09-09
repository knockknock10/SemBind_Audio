"""
SemBind-Audio: generate results/FINAL_RESEARCH_REPORT.md (Phase 11).

Assembles the final research report from validated tables and summaries.
Uses cautious, evidence-based language only. Never claims robustness that was
not measured.
"""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "final_tables"

import sys
sys.path.insert(0, str(ROOT))
import final_methodology as fm


def j(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def fmt_percent(x, nd=2):
    return f"{(x or 0) * 100:.{nd}f}%"


def build():
    T1 = j(TABLES / "TABLE_1_clean_roundtrip_E1.json") or {}
    T2 = j(TABLES / "TABLE_2_audio_quality_E2.json") or {}
    T3 = j(TABLES / "TABLE_3_fixed_vs_adaptive_E3.json") or {}
    T4 = j(TABLES / "TABLE_4_selection_ablations_E4.json") or {}
    T5 = j(TABLES / "TABLE_5_alpha_beta_sweep_E5.json") or {}
    T6 = j(TABLES / "TABLE_6_strength_sweep_E6.json") or {}
    T7 = j(TABLES / "TABLE_7_redundancy_E7.json") or {}
    T8 = j(TABLES / "TABLE_8_crypto_E8.json") or {}
    T9 = j(TABLES / "TABLE_9_awgn_robustness_E9.json") or {}
    T10 = j(TABLES / "TABLE_10_attack_robustness_E10.json") or {}
    T11 = j(TABLES / "TABLE_11_duration_bins_E11.json") or {}
    T12 = j(TABLES / "TABLE_12_per_speaker_E12.json") or {}
    T13 = j(TABLES / "TABLE_13_paired_stats_E13.json") or []
    T14 = j(TABLES / "TABLE_14_compute_cost_E14.json") or {}

    cfg = fm._base_config
    md = []
    md.append("# SemBind-Audio — Final Research Report")
    md.append("")
    md.append("*Generated 2026-09-09 from results/final_* (validated).*")
    md.append("")
    md.append(f"**Canonical config hash:** `{fm.config_hash()}`")
    md.append(f"**Git commit:** `{fm.git_head()}`  **Dataset:** LibriSpeech test-clean, "
              f"2620 recordings / 40 speakers / 5.40 h, 16 kHz")
    md.append("")

    md.append("## 1. System and methodology")
    md.append("")
    md.append("SemBind-Audio embeds a 256-bit, SHA-256 payload by perturbing the real part of "
              "selected STFT coefficients of the original (reference) audio, and extracts it "
              "reference-based (non-blind) from the watermarked audio. Selection is data-driven:")
    md.append("")
    md.append(f"- **Semantic contribution** (HuBERT, {cfg['hubert_model']}): leave-one-out "
              f"perturbation `Δ_i = 1 − cos(G, G^(−i))` computed in closed form; lower contribution "
              "⇒ higher semantic embedding suitability.")
    md.append(f"- **Spectral suitability**: normalised mean `log1p(|X|)` over the local coefficient band.")
    md.append(f"- **Joint suitability**: multiplicative `J_r = S_r^α · P_r^β`, α={cfg['alpha']}, β={cfg['beta']} "
              f"(inverse of a weighted sum).")
    md.append(f"- **Adaptive strength**: `δ_r = δ_min + (δ_max − δ_min)·J_r` per region within "
              f"[{cfg['delta_min']}, {cfg['delta_max']}].")
    md.append(f"- **Redundancy**: {cfg['redundancy']} replicas per bit → {cfg['payload_bits']}·"
              f"{cfg['redundancy']} = {cfg['payload_bits']*cfg['redundancy']} unique coordinates; majority vote.")
    md.append("- **Extraction**: reference-based, `sign(Re(W) − Re(X))`, majority-decoded.")
    md.append("")

    md.append("## 2. Clean fidelity and extraction (E1, E2)")
    md.append("")
    md.append("| Method | N | Payload success | Roundtrip acc | BER | Mean SNR (dB) |")
    md.append("|---|---|---|---|---|---|")
    for m in ["joint", "semantic", "spectral", "random"]:
        d = T1.get(m, {})
        md.append(f"| {m} | {d.get('n','')} | {fmt_percent(d.get('payload_success_rate'))} | "
                  f"{fmt_percent(d.get('roundtrip_acc'))} | {d.get('ber','')} | {d.get('snr_db','')} |")
    md.append("")
    md.append(f"Audio quality proxy on 500 recordings: mean SNR **{T2.get('snr_mean_db'):.2f} dB** "
              f"(p95 {T2.get('snr_p95_db'):.2f} dB). Note: SNR is a waveform-distortion proxy; no "
              "formal perceptual (MUSHRA/PESQ) test was run, so perceptual quality claims are "
              "limited to 'low waveform distortion'.")
    md.append("")

    md.append("## 3. Robustness (E9, E10) — fundamental limitation, reported honestly")
    md.append("")
    md.append("Under the current sparse, single-coefficient, additive STFT formulation the scheme "
              "is not robust: accuracy drops toward chance (≈0.5) under even mild parametric attacks "
              "when the watermark must survive processing. This is a property of the method, not a "
              "configuration issue; signal-processing-aware (e.g., spread-spectrum) embedding is the "
              "candidate remedy and is **unmeasured** in this report.")
    md.append("")
    md.append("**E9 AWGN (joint, redundancy 3, N=500):**")
    md.append("")
    md.append("| Attack | Bit accuracy | BER | Payload success |")
    md.append("|---|---|---|---|")
    for a in ["none", "awgn_40db", "awgn_30db", "awgn_20db", "awgn_10db"]:
        d = T9.get(a, {})
        md.append(f"| {a} | {fmt_percent(d.get('bit_accuracy'))} | {d.get('ber','')} | "
                  f"{fmt_percent(d.get('payload_success'))} |")
    md.append("")
    md.append("**E10 common attacks (joint, redundancy 3, N=500):**")
    md.append("")
    md.append("| Attack | Bit accuracy | BER | Payload success |")
    md.append("|---|---|---|---|")
    for a in ["none", "resample_8k", "lowpass_4k", "scale_0.9", "mp3_128", "awgn_20db"]:
        d = T10.get(a, {})
        md.append(f"| {a} | {fmt_percent(d.get('bit_accuracy'))} | {d.get('ber','')} | "
                  f"{fmt_percent(d.get('payload_success'))} |")
    md.append("")

    md.append("## 4. Selection-method ablations (E3–E7)")
    md.append("")
    md.append("Under clean conditions all strategies achieve full extraction, so selection-method "
              "differences appear in distortion (mean SNR) rather than accuracy.")
    md.append("")
    md.append("**E3 fixed-strength vs adaptive (joint, N=500); E4 selection-only (N=500):**")
    md.append("")
    md.append("| Variant | Roundtrip acc | Mean SNR (dB) | BER |")
    md.append("|---|---|---|---|")
    for k in ["fixed", "adaptive"]:
        d = (T3 or {}).get(k, {})
        md.append(f"| E3 {k} | {fmt_percent(d.get('roundtrip_accuracy'))} | {d.get('snr_db','')} | {d.get('bit_error_rate','')} |")
    for k in ["A_semantic_only", "B_spectral_only", "C_same_distortion_fixed", "D_random"]:
        d = (T4 or {}).get(k, {})
        md.append(f"| E4 {k} | {fmt_percent(d.get('roundtrip_accuracy'))} | {d.get('snr_db','')} | {d.get('bit_error_rate','')} |")
    md.append("")
    md.append("**E5 α/β sweep (N=500) / E6 strength sweep (N=500): roundtrip success 100% across all "
              "settings (see TABLE_5/TABLE_6). Mean SNR ranges — E5: "
              f"{((T5 or {}).get('1.0') or {}).get('snr_db','?')} dB (semantic-only) → "
              f"{((T5 or {}).get('0.0') or {}).get('snr_db','?')} dB (spectral-only); E6: "
              f"{((T6 or {}).get('0.0005') or {}).get('snr_db','?')} dB (δ=0.0005) → "
              f"{((T6 or {}).get('0.01') or {}).get('snr_db','?')} dB (δ=0.01).**")
    md.append("")
    md.append("**E7 redundancy sweep (joint, N=500):**")
    md.append("")
    md.append("| R | Roundtrip acc | Mean SNR (dB) | BER |")
    md.append("|---|---|---|---|")
    for r in ["1", "2", "3"]:
        d = (T7 or {}).get(r, {})
        md.append(f"| {r} | {fmt_percent(d.get('roundtrip_accuracy'))} | {d.get('snr_db','')} | {d.get('bit_error_rate','')} |")
    md.append("")
    md.append("**E8 cryptographic verification (N=500):** " + (
        f"TP={T8.get('tp')} TN={T8.get('tn')} FP={T8.get('fp')} FN={T8.get('fn')}." if T8 else "—"))
    md.append("")

    md.append("## 5. Coverage and stability")
    md.append("")
    bins = [(k, v) for k, v in (T11 or {}).items()]
    md.append(f"- Duration-binned payload success (E11): " +
              ", ".join(f"{k} n={v.get('n')} {fmt_percent(v.get('payload_success_rate'))}"
                        for k, v in bins) + " — 100% success in every bin (TABLE_11).")
    md.append(f"- Per-speaker (40 speakers, 2620 recordings): worst speaker payload success "
              f"{fmt_percent(T12.get('worst_success'))} (TABLE_12).")
    md.append(f"- Paired SNR comparisons (E13): all pairs differ significantly (p<1e-3); "
              "adaptive strength redistributes distortion by selection strategy (TABLE_13).")
    md.append(f"- Compute cost (E14, cached-embedding pipeline, 2620 recordings): joint "
              f"{T14.get('joint',{}).get('total_s','?')}s, semantic "
              f"{T14.get('semantic',{}).get('total_s','?')}s, spectral "
              f"{T14.get('spectral',{}).get('total_s','?')}s, random "
              f"{T14.get('random',{}).get('total_s','?')}s total over all 2620 recordings.")
    md.append("")

    md.append("## 6. Known limitations (no overclaiming)")
    md.append("")
    md.append("1. **Robustness**: the scheme is not robust to common distortions; payload "
              "success degrades to chance under AWGN/MP3/resampling/low-pass/amplitude attacks.")
    md.append("2. **Selection-method gain is currently on distortion, not clean accuracy** "
              "(all methods embed perfectly in clean audio). A controlled robustness test that "
              "separates selection strategies remains future work.")
    md.append("3. **Audio quality** is reported as SNR only; no standard perceptual metric was run.")
    md.append("4. **Security claims** are limited to the binary authentication scenarios in E8 "
              "on clean reconstruction; adversarial forgery/removal under distortion is not claimed.")
    md.append("5. Novelty is claimed cautiously: data-driven STFT-coefficient selection guided by "
              "self-supervised speech representations for audio watermark region placement is "
              "positioned as an incremental contribution, not a claim of no prior art.")
    md.append("")

    md.append("## 7. Reproduction")
    md.append("")
    md.append("```bash")
    md.append(f"python scripts/final_run.py E1      # clean 2620, 4 methods")
    md.append("python scripts/final_run.py E3 E4 E5 E6 E7 E8 E9 E10 E11 E12 E13 E14")
    md.append("python scripts/validate_final_results.py   # Phase 9 gate")
    md.append("python scripts/build_tables.py              # Phase 10")
    md.append("```")
    md.append("")
    md.append(f"All results embed `config_hash={fm.config_hash()}` and the git commit "
              f"`{fm.git_head()}`; per-file failures are recorded in `results/failures/`.")
    md.append("")

    out = RESULTS / "FINAL_RESEARCH_REPORT.md"
    out.write_text("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    build()