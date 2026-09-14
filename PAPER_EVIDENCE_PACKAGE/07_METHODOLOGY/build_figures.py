"""Build publication figures (PNG 300dpi + PDF + SVG) from canonical E1-E14 data.

All data-driven figures read the canonical aggregate JSON under results/final_*.
Only the architecture schematic (FIG 1) is conceptual (box/arrow diagram drawn in
matplotlib), which is explicitly allowed by the audit spec.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = ROOT / "PAPER_EVIDENCE_PACKAGE" / "04_FIGURES"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "serif",
})

COL = ["#1f4e79", "#c00000", "#2e7d32", "#e08a00", "#6a1b9a", "#00838f"]


def load(rel):
    return json.loads((R / rel).read_text())


def save(fig, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print("wrote", stem)


# ---------------------------------------------------------------- FIG 1 arch
def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.6, 9.2))
    ax.axis("off")
    stages = [
        "Audio (16 kHz)", "STFT (hann, n_fft=1024, hop=320)",
        "HuBERT (facebook/hubert-base-ls960, 768-d)",
        "Temporal semantic representation",
        "Leave-one-out contribution  $\\Delta_i = 1-\\cos(G,G^{(-i)})$",
        "Semantic suitability  $S=\\mathrm{minmax}(1-\\mathrm{minmax}(\\Delta))$",
        "Spectral suitability  $P=\\mathrm{minmax}(\\log(1+|X|))$",
        "Joint suitability  $J=\\max(S,\\epsilon)^\\alpha\\cdot\\max(P,\\epsilon)^\\beta$",
        "Coordinate selection (256 bits $\\times$ 3 replicas, thirds-separated)",
        "Adaptive strength  $\\delta=\\delta_{min}+(\\delta_{max}-\\delta_{min})J$",
        "STFT embedding  ($\\mathrm{Re}\\pm\\delta$, bipolar)",
        "ISTFT reconstruction",
        "Reference-based extraction  $\\mathrm{sign}(\\mathrm{Re}(W)-\\mathrm{Re}(X))$",
        "Majority voting over 3 replicas",
        "SHA-256 payload verification (256-bit)",
    ]
    n = len(stages)
    top, bottom, h = 0.98, 0.02, 0.048
    gap = (top - bottom - n * h) / (n - 1)
    for i, s in enumerate(stages):
        y = top - h - i * (h + gap)
        box = FancyBboxPatch((0.08, y), 0.84, h,
                             boxstyle="round,pad=0.004,rounding_size=0.01",
                             linewidth=0.8, edgecolor="#1f4e79",
                             facecolor="#eaf1f8")
        ax.add_patch(box)
        ax.text(0.5, y + h / 2, s, ha="center", va="center", fontsize=8)
        if i < n - 1:
            y2 = top - h - (i + 1) * (h + gap) + h
            ax.add_patch(FancyArrowPatch((0.5, y), (0.5, y2),
                                         arrowstyle="-|>", mutation_scale=9,
                                         linewidth=0.8, color="#333333"))
    # side annotations
    ax.text(0.955, 0.86, "reference/master\nrequired", fontsize=6.5, color="#c00000",
            ha="left", va="center")
    ax.text(0.955, 0.36, "OWNER only", fontsize=6.5, color="#c00000",
            ha="left", va="center")
    ax.set_xlim(0, 1.12)
    ax.set_ylim(0, 1)
    fig.suptitle("FIG 1. SemBind-Audio pipeline architecture (frozen implementation)",
                 fontsize=9)
    save(fig, "FIG1_architecture")


# ---------------------------------------------------------------- FIG 3 family
def redundancy_figures():
    e7 = load("final_ablations/E7/E7_summary.json")["aggregate"]
    Rs = [1, 2, 3]
    succ = [e7[str(r)]["payload_success"]["mean"] * 100 for r in Rs]
    ber = [e7[str(r)]["bit_error_rate"]["mean"] for r in Rs]
    snr = [e7[str(r)]["snr_db"]["mean"] for r in Rs]
    fails = [2, 3, 0]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(Rs, succ, "o-", color=COL[0], linewidth=1.4, markersize=5)
    for x, y in zip(Rs, succ):
        ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xlabel("Replica redundancy $R$")
    ax.set_ylabel("Payload success (%)")
    ax.set_title("FIG 3a. Payload recovery vs. redundancy (n=500)")
    ax.set_xticks(Rs)
    ax.set_ylim(99.2, 100.15)
    save(fig, "FIG_R1_redundancy_payload_recovery")

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(Rs, ber, "s-", color=COL[1], linewidth=1.4, markersize=5)
    for x, y in zip(Rs, ber):
        ax.annotate(f"{y:.2e}", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6.5)
    ax.set_xlabel("Replica redundancy $R$")
    ax.set_ylabel("Mean bit error rate")
    ax.set_title("FIG 3b. BER vs. redundancy (n=500)")
    ax.set_xticks(Rs)
    save(fig, "FIG_R2_redundancy_BER")

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(Rs, snr, "^-", color=COL[2], linewidth=1.4, markersize=5)
    for x, y in zip(Rs, snr):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xlabel("Replica redundancy $R$")
    ax.set_ylabel("Mean clean SNR (dB)")
    ax.set_title("FIG 3c. SNR vs. redundancy (n=500)")
    ax.set_xticks(Rs)
    save(fig, "FIG_R3_redundancy_SNR_tradeoff")

    # combined trade-off (recovery vs quality)
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    tones = ["#1f4e79", "#c00000", "#2e7d32"]
    for i, r in enumerate(Rs):
        ax.scatter(snr[i], succ[i], s=70, color=tones[i], zorder=3)
        ax.annotate(f"R={r}", (snr[i], succ[i]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean clean SNR (dB)  $\\rightarrow$ higher quality")
    ax.set_ylabel("Payload success (%)  $\\rightarrow$ higher recovery")
    ax.set_title("FIG 3d. Recovery-quality trade-off")
    ax.set_ylim(99.2, 100.15)
    save(fig, "FIG_R3b_redundancy_recovery_quality_tradeoff")


# ---------------------------------------------------------------- FIG 4
def fig_alpha_beta():
    e5 = load("final_ablations/E5/E5_summary.json")["aggregate"]
    mapping = {"1.0": 1.00, "0.75": 0.75, "0.7": 0.70, "0.5": 0.50,
               "0.25": 0.25, "0.0": 0.00}
    alphas = sorted(mapping.values())
    snr = [e5[k]["snr_db"]["mean"] for k, v in sorted(mapping.items(), key=lambda kv: kv[1])]
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    ax.plot(alphas, snr, "o-", color=COL[0], linewidth=1.4, markersize=5)
    ax.axvline(0.70, color="#c00000", linestyle="--", linewidth=0.9)
    ax.text(0.70, min(snr), " final $\\alpha$=0.70", color="#c00000",
            fontsize=7, va="bottom")
    ax.set_xlabel("Semantic weight $\\alpha$  ($\\beta = 1-\\alpha$)")
    ax.set_ylabel("Mean clean SNR (dB)")
    ax.set_title("FIG 4. Suitability-weight sweep (n=500 each)")
    save(fig, "FIG4_alpha_beta_sweep")


# ---------------------------------------------------------------- FIG 5
def fig_strength():
    e6 = load("final_ablations/E6/E6_strength_sweep.json") if False else \
        load("final_ablations/E6/E6_summary.json")["aggregate"]
    deltas = [0.0005, 0.001, 0.002, 0.005, 0.008, 0.01]
    keys = ["0.0005", "0.001", "0.002", "0.005", "0.008", "0.01"]
    snr = [e6[k]["snr_db"]["mean"] for k in keys]
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    ax.plot(deltas, snr, "o-", color=COL[0], linewidth=1.4, markersize=5)
    ax.set_xscale("log")
    ax.set_xlabel("Embedding strength $\\delta$ (log scale)")
    ax.set_ylabel("Mean clean SNR (dB)")
    ax.set_title("FIG 5. Strength sweep (n=500 each; recovery = 100%)")
    save(fig, "FIG5_strength_sweep")


# ---------------------------------------------------------------- FIG 6
def fig_selection_ablation():
    e4 = load("final_ablations/E4/E4_summary.json")["aggregate"]
    e2 = load("final_clean/E2/E2_audio_quality_summary.json")["aggregate"]["joint"]
    labels = ["Semantic-only", "Joint (S,P)", "Random", "Same-distortion fixed",
              "Spectral-only"]
    keys = ["A_semantic_only", None, "D_random", "C_same_distortion_fixed",
            "B_spectral_only"]
    vals = []
    for k in keys:
        vals.append(e2["snr_db"]["mean"] if k is None else e4[k]["snr_db"]["mean"])
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]
    colors = [COL[1], COL[3], COL[3], COL[0], COL[2]]
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    bars = ax.barh(labels, vals, color=colors, height=0.6)
    for b, v in zip(bars, vals):
        ax.text(v + 0.1, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=7)
    ax.set_xlabel("Mean clean SNR (dB)")
    ax.set_title("FIG 6. Selection ablation (n=500 each)")
    ax.set_xlim(70, 78)
    save(fig, "FIG6_selection_ablation")


# ---------------------------------------------------------------- FIG 7
def fig_duration():
    e11 = load("final_analysis/E11/E11_duration_bins.json")["bins"]
    bins = ["<4s", "4-8s", "8-12s", "12-20s", ">20s"]
    ns = [e11[b]["n"] for b in bins]
    snr = [e11[b]["snr_db"] for b in bins]
    labels = [f"{b}\n(n={n})" for b, n in zip(bins, ns)]
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    bars = ax.bar(labels, snr, color=COL[0], width=0.6)
    for b, v in zip(bars, snr):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}",
                ha="center", fontsize=7)
    ax.set_xlabel("Recording duration")
    ax.set_ylabel("Mean clean SNR (dB)")
    ax.set_title("FIG 7. Duration vs. SNR (all 2620; recovery = 100%)")
    ax.set_ylim(70, 82)
    save(fig, "FIG7_duration_vs_snr")


# ---------------------------------------------------------------- FIG 8
def _attack_stats(rel):
    import csv as _csv
    agg = {}
    with open(R / rel) as f:
        for row in _csv.DictReader(f):
            agg.setdefault(row["attack"], []).append(
                (float(row["bit_accuracy"]), float(row["ber"]), float(row["payload_success"])))
    out = {}
    for a, rows in agg.items():
        n = len(rows)
        out[a] = (sum(r[0] for r in rows) / n, sum(r[1] for r in rows) / n,
                  sum(r[2] for r in rows) / n)
    return out


def fig_robustness():
    e9 = _attack_stats("final_robustness/E9/E9_awgn_500.csv")
    e10 = _attack_stats("final_robustness/E10/E10_attacks_500.csv")
    order9 = ["none", "awgn_40db", "awgn_30db", "awgn_20db", "awgn_10db"]
    lab9 = ["None", "AWGN\n40 dB", "AWGN\n30 dB", "AWGN\n20 dB", "AWGN\n10 dB"]
    order10 = ["none", "scale_0.9", "resample_8k", "mp3_128", "lowpass_4k", "awgn_20db"]
    lab10 = ["None", "Gain\n0.9", "Resample\n8 kHz", "MP3\n128", "Lowpass\n4 kHz", "AWGN\n20 dB"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, order, labs, src, title in [
        (axes[0], order9, lab9, e9, "(a) AWGN"),
        (axes[1], order10, lab10, e10, "(b) Common attacks"),
    ]:
        ba = [src[a][0] * 100 for a in order]
        axes_idx = order.index("none")
        colors = ["#2e7d32" if a == "none" else "#c00000" for a in order]
        ax.bar(range(len(order)), ba, color=colors, width=0.65)
        ax.axhline(50, color="#555555", linestyle="--", linewidth=0.9)
        ax.text(len(order) - 0.5, 51.5, "chance", fontsize=6.5, color="#555555",
                ha="right")
        for i, v in enumerate(ba):
            ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=6.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labs)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Bit accuracy (%)")
        ax.set_title(f"FIG 8{title} (payload success = 0 under all attacks)", fontsize=8)
    fig.suptitle("FIG 8. Robustness to signal-processing attacks (n=500 per attack)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "FIG8_robustness_attacks")


# ---------------------------------------------------------------- extra E12
def fig_speakers():
    e12 = load("final_analysis/E12/E12_per_speaker.json")["per_speaker"]
    sp = sorted(e12.keys(), key=lambda k: int(k))
    snr = [e12[k]["snr_db"] for k in sp]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.bar(range(len(sp)), snr, color=COL[0], width=0.75)
    ax.axhline(sum(snr) / len(snr), color="#c00000", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Speaker index (40 LibriSpeech test-clean speakers)")
    ax.set_ylabel("Mean clean SNR (dB)")
    ax.set_xticks(range(0, len(sp), 4))
    ax.set_xticklabels([sp[i] for i in range(0, len(sp), 4)])
    ax.set_title("FIG E12. Per-speaker SNR (all 40 speakers, recovery = 100%)")
    save(fig, "FIG_E12_speaker_snr")


# ---------------------------------------------------------------- extra E13
def fig_effect_sizes():
    import csv as _csv
    rows = list(_csv.DictReader(open(R / "paper_analysis/statistics/E13_effect_sizes.csv")))
    pairs = [r["pair"].replace("_vs_", " vs ") for r in rows]
    dz = [float(r["cohen_dz"]) for r in rows]
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    ax.barh(pairs, dz, color=[COL[0] if v > 0 else COL[1] for v in dz], height=0.6)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Cohen's $d_z$ (paired effect size)")
    ax.set_title("FIG E13. Paired effect sizes for clean SNR (n=2620)")
    save(fig, "FIG_E13_effect_sizes")


if __name__ == "__main__":
    fig_architecture()
    redundancy_figures()
    fig_alpha_beta()
    fig_strength()
    fig_selection_ablation()
    fig_duration()
    fig_robustness()
    fig_speakers()
    fig_effect_sizes()
    print("all figures written")