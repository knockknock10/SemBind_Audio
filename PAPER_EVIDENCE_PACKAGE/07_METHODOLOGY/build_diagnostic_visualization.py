"""Diagnostic visualization export: per-coordinate embedding intermediates.

This is a DIAGNOSTIC VISUALIZATION RUN, not a new experiment. It does NOT modify
E1-E14. It re-uses the frozen `final_methodology.py` pipeline unchanged and exports
the per-coordinate intermediates that were not persisted during E1-E14, for exactly
3 real LibriSpeech test-clean recordings.

Outputs:
  06_RAW_PAPER_DATA/diagnostic_embedding_coordinates.csv   (all selected coordinates)
  06_RAW_PAPER_DATA/diagnostic_suitability_frames.csv      (per-STFT-frame curves)
  06_RAW_PAPER_DATA/diagnostic_run_metadata.json
  04_FIGURES/FIG2_embedding_locations.png/.pdf/.svg
"""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import final_methodology as fm  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAW = ROOT / "PAPER_EVIDENCE_PACKAGE" / "06_RAW_PAPER_DATA"
OUT = ROOT / "PAPER_EVIDENCE_PACKAGE" / "04_FIGURES"
AUDIO_DIR = ROOT / "data" / "raw" / "librispeech" / "test-clean"

AUDIO_IDS = ["61-70968-0021", "121-127105-0007", "8230-279154-0013"]


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(ROOT)).stdout.strip()


def path_for(audio_id):
    spk, chap, _ = audio_id.split("-")
    return AUDIO_DIR / spk / chap / f"{audio_id}.flac"


def main():
    payload = fm.load_payload()
    rows = []
    frame_rows = []
    meta_runs = []
    for audio_id in AUDIO_IDS:
        p = path_for(audio_id)
        audio, sr = sf.read(p, dtype="float32")
        stft = fm.compute_stft(audio)
        emb = fm.load_embeddings_cached(audio_id, audio)
        contribution = fm.compute_contribution(emb)
        hubert_frames = int(contribution.numel())
        stft_frames = int(stft.shape[1])

        contrib_map = fm.map_contribution_to_stft(contribution, hubert_frames, stft_frames)
        s_map = fm.map_contribution_to_stft(
            fm.semantic_suitability(contribution), hubert_frames, stft_frames)
        p_map = fm.spectral_suitability(stft)
        j_map = fm.joint_suitability(s_map, p_map)
        delta_map = fm.adaptive_delta(j_map)

        coords = fm.allocate_coordinates(stft_frames, "joint", j_map, audio_id,
                                         redundancy=fm.REDUNDANCY)
        dvals = [float(delta_map[t]) for (f, t) in coords]

        for k, (f, t) in enumerate(coords):
            rows.append({
                "audio_id": audio_id,
                "coordinate_index": k,
                "bit_index": k // fm.REDUNDANCY,
                "replica_index": k % fm.REDUNDANCY,
                "frequency_bin": f,
                "time_frame": t,
                "embedded_delta": round(float(delta_map[t]), 8),
                "semantic_contribution": round(float(contrib_map[t]), 8),
                "semantic_suitability": round(float(s_map[t]), 8),
                "spectral_suitability": round(float(p_map[t]), 8),
                "joint_suitability": round(float(j_map[t]), 8),
                "selected": True,
            })
        for t in range(stft_frames):
            frame_rows.append({
                "audio_id": audio_id,
                "time_frame": t,
                "semantic_contribution": round(float(contrib_map[t]), 8),
                "semantic_suitability": round(float(s_map[t]), 8),
                "spectral_suitability": round(float(p_map[t]), 8),
                "joint_suitability": round(float(j_map[t]), 8),
                "adaptive_delta": round(float(delta_map[t]), 8),
            })
        meta_runs.append({
            "audio_id": audio_id,
            "audio_path": str(p.relative_to(ROOT)),
            "sample_rate": int(sr),
            "duration_seconds": round(len(audio) / sr, 4),
            "stft_frames": stft_frames,
            "hubert_frames": hubert_frames,
            "n_coordinates": len(coords),
            "n_unique": len(set(coords)),
        })
        print("exported", audio_id, stft_frames, hubert_frames, len(coords))

    with open(RAW / "diagnostic_embedding_coordinates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(RAW / "diagnostic_suitability_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
        w.writeheader()
        w.writerows(frame_rows)

    meta = {
        "run_type": "DIAGNOSTIC_VISUALIZATION_ONLY",
        "note": "Not an experiment; does not modify E1-E14. Re-uses frozen methodology unchanged.",
        "audio_ids": AUDIO_IDS,
        "runs": meta_runs,
        "git_commit_at_run": git_head(),
        "config_hash": fm.config_hash(),
        "methodology_module": "final_methodology.py",
        "methodology_constants": {
            "N_FFT": fm.N_FFT, "HOP_LENGTH": fm.HOP_LENGTH, "WIN_LENGTH": fm.WIN_LENGTH,
            "PAYLOAD_BITS": fm.PAYLOAD_BITS, "REDUNDANCY": fm.REDUNDANCY,
            "ALPHA": fm.ALPHA, "BETA": fm.BETA, "DELTA_MIN": fm.DELTA_MIN,
            "DELTA_MAX": fm.DELTA_MAX,
        },
        "payload_hash_sha256": hashlib.sha256(
            "".join(str(int(b)) for b in payload.tolist()).encode()).hexdigest(),
    }
    (RAW / "diagnostic_run_metadata.json").write_text(json.dumps(meta, indent=2))
    print("diagnostic metadata written")

    # ------------------------------------------------------------- figure
    plt.rcParams.update({"font.size": 8, "font.family": "serif",
                         "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 300,
                         "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9))
    for ax, audio_id in zip(axes, AUDIO_IDS):
        p = path_for(audio_id)
        audio, sr = sf.read(p, dtype="float32")
        stft = fm.compute_stft(audio)
        mag = np.log1p(np.abs(stft.numpy()))
        ax.imshow(mag, origin="lower", aspect="auto", cmap="viridis")
        sub = [r for r in rows if r["audio_id"] == audio_id]
        ts = [r["time_frame"] for r in sub]
        fs = [r["frequency_bin"] for r in sub]
        ax.scatter(ts, fs, s=3, c="#ff2d2d", marker=".", linewidths=0)
        ax.set_xlabel("STFT time frame")
        ax.set_title(f"{audio_id}\n{dict((m['audio_id'], m['duration_seconds']) for m in meta_runs)[audio_id]} s",
                     fontsize=7.5)
        if ax is axes[0]:
            ax.set_ylabel("Frequency bin")
    fig.suptitle("FIG 2. Actual embedding locations (red dots) over log-magnitude "
                 "spectrograms — diagnostic run, frozen methodology", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"FIG2_embedding_locations.{ext}")
    plt.close(fig)

    # suitability curve figure per file (compact)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.6))
    for ax, audio_id in zip(axes, AUDIO_IDS):
        fr = [r for r in frame_rows if r["audio_id"] == audio_id]
        fr.sort(key=lambda r: r["time_frame"])
        t = [r["time_frame"] for r in fr]
        ax.plot(t, [r["semantic_suitability"] for r in fr], color="#1f4e79",
                linewidth=1.0, label="$S$ (semantic)")
        ax.plot(t, [r["spectral_suitability"] for r in fr], color="#c00000",
                linewidth=1.0, label="$P$ (spectral)")
        ax.plot(t, [r["joint_suitability"] for r in fr], color="#2e7d32",
                linewidth=1.0, label="$J$ (joint)")
        ax.set_xlabel("STFT time frame")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{audio_id}", fontsize=7.5)
        if ax is axes[0]:
            ax.set_ylabel("Suitability [0-1]")
            ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("FIG 2b. Per-frame suitability curves (diagnostic, frozen methodology)",
                 fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"FIG2b_suitability_curves.{ext}")
    plt.close(fig)
    print("FIG2 written")


if __name__ == "__main__":
    main()