"""SemBind-Audio paper evidence builder (tables + statistics + register).

Reads ONLY canonical finalized results under ../results (relative to repo root)
and writes publications artifacts into PAPER_EVIDENCE_PACKAGE/.

This script NEVER computes new experimental numbers from the waveform; it only
re-derives aggregates/stats from the stored canonical aggregate JSON + raw CSV
files, so numbers are traceable to E1-E14 sources.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "PAPER_EVIDENCE_PACKAGE"
R = ROOT / "results"

SIMPLIFIED = {
    "A_semantic_only": "Semantic-only",
    "B_spectral_only": "Spectral-only",
    "C_same_distortion_fixed": "Same-distortion fixed",
    "D_random": "Random",
    "1.0": "0.00/1.00",
    "0.75": "0.75/0.25",
    "0.7": "0.70/0.30",
    "0.5": "0.50/0.50",
    "0.25": "0.25/0.75",
    "0.0": "1.00/0.00",
    "0.0005": "0.0005",
    "0.001": "0.001",
    "0.002": "0.002",
    "0.005": "0.005",
    "0.008": "0.008",
    "0.01": "0.010",
}


def load_json(rel: str):
    with open(R / rel) as f:
        return json.load(f)


def read_csv(rel: str):
    with open(R / rel) as f:
        return list(csv.DictReader(f))


def agg_mean(agg, group, key, stat="mean"):
    try:
        return float(agg[group][key][stat])
    except (KeyError, TypeError):
        return None


def fmt(v, nd=4, suffix=""):
    if v is None:
        return ""
    return f"{v:.{nd}f}{suffix}"


def pct(v, nd=2):
    if v is None:
        return ""
    return f"{v * 100:.{nd}f}%"


# ------------------------------------------------------------------ metadata
E1 = load_json("final_clean/E1/E1_summary_2620.json")
E2 = load_json("final_clean/E2/E2_audio_quality_summary.json")
E3 = load_json("final_ablations/E3/E3_summary.json")
E4 = load_json("final_ablations/E4/E4_summary.json")
E5 = load_json("final_ablations/E5/E5_summary.json")
E6 = load_json("final_ablations/E6/E6_summary.json")
E7 = load_json("final_ablations/E7/E7_summary.json")
E8 = load_json("final_crypto/E8/E8_confusion.json")
E9 = load_json("final_robustness/E9/E9_summary.json")
E10 = load_json("final_robustness/E10/E10_summary.json")
E11 = load_json("final_analysis/E11/E11_duration_bins.json")
E12 = load_json("final_analysis/E12/E12_per_speaker.json")
E13 = load_json("final_analysis/E13/E13_paired_stats.json")
E14 = load_json("final_analysis/E14/E14_compute_cost.json")
E13_eff = load_json("paper_analysis/statistics/E13_effect_sizes.json") \
    if (R / "paper_analysis/statistics/E13_effect_sizes.json").exists() else None
E13_eff_csv = None
eff_path = R / "paper_analysis/statistics/E13_effect_sizes.csv"
if eff_path.exists():
    E13_eff_csv = read_csv("paper_analysis/statistics/E13_effect_sizes.csv")
CFG = E1["config"]

E9_by_attack = {}
for row in read_csv("final_robustness/E9/E9_awgn_500.csv"):
    a = row["attack"]
    E9_by_attack.setdefault(a, []).append(
        (float(row["bit_accuracy"]), float(row["ber"]), float(row["payload_success"]))
    )
E10_by_attack = {}
for row in read_csv("final_robustness/E10/E10_attacks_500.csv"):
    a = row["attack"]
    E10_by_attack.setdefault(a, []).append(
        (float(row["bit_accuracy"]), float(row["ber"]), float(row["payload_success"]))
    )


def attack_agg(store, attack):
    rows = store[attack]
    n = len(rows)
    ba = sum(r[0] for r in rows) / n
    ber = sum(r[1] for r in rows) / n
    succ = sum(r[2] for r in rows) / n
    return n, ba, ber, succ


# ------------------------------------------------------------------ directory
(PKG / "03_TABLES").mkdir(parents=True, exist_ok=True)
(PKG / "05_STATISTICS").mkdir(parents=True, exist_ok=True)
(PKG / "06_RAW_PAPER_DATA").mkdir(parents=True, exist_ok=True)


def write_csv(name, headers, rows, subdir="03_TABLES"):
    path = PKG / subdir / name
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    return path


def write_tex(name, caption, label, col_spec, headers, rows, footnotes=None):
    path = PKG / "03_TABLES" / name
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        "\\scriptsize",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        " & ".join("\\textbf{" + h.replace("_", "\\_") + "}" for h in headers) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        cells = [str(c) for c in r]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    if footnotes:
        lines.append("\\multicolumn{%d}{l}{\\footnotesize %s}\\\\" % (len(headers), footnotes))
    lines += ["\\end{tabular}", "\\end{table}"]
    (PKG / "03_TABLES" / name).write_text("\n".join(lines) + "\n")
    return path


# =================================================================== TABLE 1
# Dataset + experimental configuration
AWS = ["sample_rate", "n_fft", "hop_length", "win_length", "payload_bits",
       "redundancy", "alpha", "beta", "delta_min", "delta_max", "fixed_delta"]
WS2 = {"sample_rate": "Sample rate (Hz)", "n_fft": "STFT FFT size",
       "hop_length": "Hop length (samples)", "win_length": "Window length (samples)",
       "payload_bits": "Payload size (bits)", "redundancy": "Replica redundancy R",
       "alpha": "Semantic weight \\alpha", "beta": "Spectral weight \\beta",
       "delta_min": "Min embedding strength", "delta_max": "Max embedding strength",
       "fixed_delta": "Fixed strength (E3/E6)"}
t1 = []
t1_tex = []
for k in AWS:
    t1.append([WS2[k].replace("\\", ""), CFG[k]])
    t1_tex.append([WS2[k], CFG[k]])
# dataset rows
t1 += [
    ["Dataset", "LibriSpeech test-clean (16 kHz)"],
    ["Recordings", 2620],
    ["Speakers", 40],
    ["Total duration", "5.40 h (mean 7.42 s, median 5.79 s, min 1.29 s, max 34.96 s)"],
    ["Evaluation subset E2-E10", "First 500 rows of evaluation_manifest.csv"],
    ["HuBERT model", CFG["hubert_model"]],
    ["HuBERT feature dim", CFG["hubert_dim"]],
    ["Contribution", "\\Delta_i = 1 - cos(G, G^{(-i)}) (leave-one-out, closed form)"],
    ["Semantic suitability", "S = minmax(1 - minmax(\\Delta))"],
    ["Spectral suitability", "P = minmax(mean_{f in band} log1p(|X[f,t]|))"],
    ["Joint suitability", "J = max(S,\\epsilon)^\\alpha \\cdot max(P,\\epsilon)^\\beta"],
    ["Adaptive strength", "\\delta = \\delta_{min} + (\\delta_{max} - \\delta_{min}) J"],
    ["Coordinate allocation", "256 bits x R replicas; same-bit replicas in separated timeline thirds; coprime frequency stride; all (freq,time) unique"],
    ["Embedding", "Bipolar additive: bit=1 -> Re+\\delta, bit=0 -> Re-\\delta"],
    ["Extraction", "Reference-based sign(Re(W) - Re(X)) + majority vote over R replicas"],
    ["Verification", "SHA-256 payload digest compare (256-bit)"],
    ["Config hash", "cf10755c13a0ee3147845724f4f097dccc414c2f2b02eed102d6a2bb21231095"],
]
t1_tex += [
    ["Dataset", "LibriSpeech test-clean (16 kHz)"],
    ["Recordings", 2620],
    ["Speakers", 40],
    ["Total duration", "5.40 h (mean 7.42 s, median 5.79 s)"],
    ["Evaluation subset E2-E10", "First 500 rows of evaluation\\_manifest.csv"],
    ["HuBERT model", CFG["hubert_model"]],
    ["HuBERT feature dim", CFG["hubert_dim"]],
    ["Contribution", "$\\Delta_i = 1 - \\cos(G, G^{(-i)})$"],
    ["Semantic suitability", "$S = \\mathrm{minmax}(1 - \\mathrm{minmax}(\\Delta))$"],
    ["Spectral suitability", "$P = \\mathrm{minmax}(\\mathrm{mean}_{f} \\log(1+|X[f,t]|))$"],
    ["Joint suitability", "$J = \\max(S,\\epsilon)^\\alpha \\max(P,\\epsilon)^\\beta$"],
    ["Adaptive strength", "$\\delta = \\delta_{\\mathrm{min}} + (\\delta_{\\mathrm{max}}-\\delta_{\\mathrm{min}}) J$"],
    ["Coordinate allocation", "256 bits $\\times R$; thirds-separated replicas; coprime freq stride; unique coords"],
    ["Embedding", "Bipolar additive $\\mathrm{Re}\\pm\\delta$"],
    ["Extraction", "Reference sign + majority vote over $R$"],
    ["Verification", "SHA-256 payload digest (256-bit)"],
    ["Config hash", "cf10755c13a0\\ldots 31095"],
]
write_csv("table_1_dataset_config.csv", ["parameter", "value"], t1)
write_tex("table_1_dataset_config.tex", "Dataset and experimental configuration. "
          "Configuration hash cf10755c13a0ee3147845724f4f097dccc414c2f2b02eed102d6a2bb21231095.",
          "dataset-config", "ll", ["Parameter", "Value"], t1_tex,
          "Config from results/final_clean/E1/E1_summary_2620.json (identical in all E1-E14 metadata).")

# =================================================================== TABLE 2
# Clean watermark performance E1 (n=2620)
METHODS = [("joint", "Joint (S,P)"), ("semantic", "Semantic-only"), ("spectral", "Spectral-only"),
           ("random", "Random")]
t2 = []
t2_tex = []
for m, label in METHODS:
    a = E1["methods"][["joint", "semantic", "spectral", "random"].index(m)]["aggregate"][m]
    t2.append([
        label, 2620,
        fmt(a["roundtrip_accuracy"]["mean"] * 100, 2, "%"),
        fmt(a["direct_accuracy"]["mean"] * 100, 2, "%"),
        fmt(a["bit_error_rate"]["mean"], 6),
        fmt(a["payload_success"]["mean"] * 100, 2, "%"),
        fmt(a["snr_db"]["mean"], 2),
        fmt(a["snr_db"]["std"], 2),
        fmt(a["snr_db"]["min"], 2),
        fmt(a["snr_db"]["max"], 2),
        fmt(a["mean_delta"], 6),
    ])
    t2_tex.append([
        label, 2620,
        fmt(a["roundtrip_accuracy"]["mean"] * 100, 2, "\\%"),
        fmt(a["direct_accuracy"]["mean"] * 100, 2, "\\%"),
        fmt(a["bit_error_rate"]["mean"], 6),
        fmt(a["payload_success"]["mean"] * 100, 2, "\\%"),
        fmt(a["snr_db"]["mean"], 2),
        fmt(a["snr_db"]["std"], 2),
        fmt(a["snr_db"]["min"], 2),
        fmt(a["snr_db"]["max"], 2),
        fmt(a["mean_delta"], 6),
    ])
write_csv("table_2_clean_performance_E1.csv",
          ["method", "n", "roundtrip_acc(%)", "direct_acc(%)", "BER", "payload_success(%)",
           "SNR_mean(dB)", "SNR_std(dB)", "SNR_min(dB)", "SNR_max(dB)", "mean_delta"],
          t2)
write_tex("table_2_clean_performance_E1.tex", "Clean-channel performance on all 2620 "
          "LibriSpeech test-clean recordings (no attacks). Payload = 256 bits, R=3. "
          "SNR is signal-to-watermark-noise ratio on the time-domain audio.",
          "clean-e1", "lcccccccccc", ["Method", "$n$", "RT acc", "Direct acc", "BER",
          "Payload succ.", "SNR $\\mu$", "SNR $\\sigma$", "SNR min", "SNR max", "$\\bar{\\delta}$"],
          t2_tex, "Random: profile counts one payload failure (audio_id 4446-2275-0028, BER 0.0039). Source results/final_clean/E1/*.csv.")

# =================================================================== TABLE 3
# Redundancy R=1/2/3 (E7)
t3 = []
t3_tex = []
for r in ["1", "2", "3"]:
    a = E7["aggregate"][r]
    n_fail = 0
    for row in read_csv("final_ablations/E7/E7_redundancy_500.csv"):
        if row["redundancy"] == r and float(row["payload_success"]) == 0:
            n_fail += 1
    t3.append([
        r, 500,
        fmt(a["roundtrip_accuracy"]["mean"] * 100, 2, "%"),
        fmt(a["bit_error_rate"]["mean"], 6),
        fmt(a["payload_success"]["mean"] * 100, 2, "%"),
        fmt(a["snr_db"]["mean"], 2),
        n_fail,
    ])
    t3_tex.append([
        r, 500,
        fmt(a["roundtrip_accuracy"]["mean"] * 100, 2, "\\%"),
        fmt(a["bit_error_rate"]["mean"], 6),
        fmt(a["payload_success"]["mean"] * 100, 2, "\\%"),
        fmt(a["snr_db"]["mean"], 2),
        n_fail,
    ])
write_csv("table_3_redundancy_E7.csv",
          ["redundancy_R", "n", "roundtrip_acc(%)", "BER", "payload_success(%)", "SNR_mean(dB)", "failures"],
          t3)
write_tex("table_3_redundancy_E7.tex", "Replica redundancy trade-off E7 (n=500 per R, joint "
          "selection, delta=0.005 profile). Failure = any of 256 payload bits wrong after "
          "clean round-trip. R=3 yields the only zero-failure configuration.",
          "redundancy-e7", "lcccccc", ["$R$", "$n$", "Roundtrip acc.", "BER", "Payload succ.",
          "SNR $\\mu$ (dB)", "Failures"], t3_tex,
          "R=1 and R=2 are NOT monotonic in payload recovery; R=3 was selected as final config because it achieved zero clean failures (n=500).")

# =================================================================== TABLE 4
# Fixed vs adaptive (E3)
t4 = []
t4_tex = []
t4.append(["Fixed (J-independent, delta=0.005)"] + [x for x in []])
rows4 = []
for mode, label in [("fixed", "Fixed"), ("adaptive", "Adaptive")]:
    a = E3["aggregate"][mode]
    rows4.append([
        label, 500,
        fmt(a["roundtrip_accuracy"]["mean"] * 100, 2, "%"),
        fmt(a["bit_error_rate"]["mean"], 6),
        fmt(a["payload_success"]["mean"] * 100, 2, "%"),
        fmt(a["snr_db"]["mean"], 2),
        fmt(a["mean_delta"], 6),
    ])
write_csv("table_4_fixed_vs_adaptive_E3.csv",
          ["mode", "n", "roundtrip_acc(%)", "BER", "payload_success(%)", "SNR_mean(dB)", "mean_delta"],
          rows4)
write_tex("table_4_fixed_vs_adaptive_E3.tex", "Fixed vs data-adaptive embedding strength E3 "
          "(n=500 each). Both achieve perfect clean recovery; adaptive increases mean SNR.",
          "fixed-adaptive", "lcccccc", ["Mode", "$n$", "RT acc", "BER", "Payload succ.",
          "SNR $\\mu$ (dB)", "$\\bar{\\delta}$"], rows4,
          "Same 500 files; identical allocation; only delta schedule differs.")

# =================================================================== TABLE 5
# Selection ablation (E4)
t5 = []
t5_tex = []
for k in ["A_semantic_only", "B_spectral_only", "C_same_distortion_fixed", "D_random"]:
    a = E4["aggregate"][k]
    t5.append([
        SIMPLIFIED[k], 500,
        fmt(a["roundtrip_accuracy"]["mean"] * 100, 2, "%"),
        fmt(a["bit_error_rate"]["mean"], 6),
        fmt(a["payload_success"]["mean"] * 100, 2, "%"),
        fmt(a["snr_db"]["mean"], 2),
        fmt(a["mean_delta"], 6),
    ])
    t5_tex.append([
        SIMPLIFIED[k], 500,
        fmt(a["roundtrip_accuracy"]["mean"] * 100, 2, "\\%"),
        fmt(a["bit_error_rate"]["mean"], 6),
        fmt(a["payload_success"]["mean"] * 100, 2, "\\%"),
        fmt(a["snr_db"]["mean"], 2),
        fmt(a["mean_delta"], 6),
    ])
# joint reference row (adaptive, R=3) — same first-500 subset, from E2 audio-quality run
gj = E2["aggregate"]["joint"]
t5.append([
    "Joint (S,P) [ref E2]", 500,
    fmt(gj["roundtrip_accuracy"]["mean"] * 100, 2, "%"),
    fmt(gj["bit_error_rate"]["mean"], 6),
    fmt(gj["payload_success"]["mean"] * 100, 2, "%"),
    fmt(gj["snr_db"]["mean"], 2),
    fmt(gj["mean_delta"], 6),
])
t5_tex.append([
    "Joint (S,P) [ref E2]", 500,
    fmt(gj["roundtrip_accuracy"]["mean"] * 100, 2, "\\%"),
    fmt(gj["bit_error_rate"]["mean"], 6),
    fmt(gj["payload_success"]["mean"] * 100, 2, "\\%"),
    fmt(gj["snr_db"]["mean"], 2),
    fmt(gj["mean_delta"], 6),
])
write_csv("table_5_selection_ablation_E4.csv",
          ["method", "n", "roundtrip_acc(%)", "BER", "payload_success(%)", "SNR_mean(dB)", "mean_delta"],
          t5)
write_tex("table_5_selection_ablation_E4.tex", "Selection ablation E4 (n=500 each, adaptive "
          "strength where applicable; C uses constant delta=0.005 to equalize distortion). "
          "Ordering of mean clean SNR: spectral > joint > random > semantic. Joint reference "
          "row taken from E2 (same first-500 subset, adaptive, R=3).",
          "selection-e4", "lcccccc", ["Method", "$n$", "RT acc", "BER", "Payload succ.",
          "SNR $\\mu$ (dB)", "$\\bar{\\delta}$"], t5_tex,
          "E4 uses adaptive-delta variants (A,B,D) and constant-delta (C); 'joint' row is E2 result (74.82 dB). "
          "Spectral-only has the highest clean SNR and joint sits between semantic and spectral.")

# =================================================================== TABLE 6
# Alpha/beta sweep (E5)
t6 = []
t6_tex = []
expanded = {"1.0": ("1.00", "0.00"), "0.75": ("0.75", "0.25"), "0.7": ("0.70", "0.30"),
            "0.5": ("0.50", "0.50"), "0.25": ("0.25", "0.75"), "0.0": ("0.00", "1.00")}
for k in ["1.0", "0.75", "0.7", "0.5", "0.25", "0.0"]:
    a = E5["aggregate"][k]
    t6.append([
        expanded[k][0], expanded[k][1], 500,
        fmt(a["roundtrip_accuracy"]["mean"] * 100, 2, "%"),
        fmt(a["bit_error_rate"]["mean"], 6),
        fmt(a["payload_success"]["mean"] * 100, 2, "%"),
        fmt(a["snr_db"]["mean"], 2),
    ])
    t6_tex.append([
        expanded[k][0], expanded[k][1], 500,
        fmt(a["roundtrip_accuracy"]["mean"] * 100, 2, "\\%"),
        fmt(a["bit_error_rate"]["mean"], 6),
        fmt(a["payload_success"]["mean"] * 100, 2, "\\%"),
        fmt(a["snr_db"]["mean"], 2),
    ])
write_csv("table_6_alpha_beta_sweep_E5.csv",
          ["alpha", "beta", "n", "roundtrip_acc(%)", "BER", "payload_success(%)", "SNR_mean(dB)"],
          t6)
write_tex("table_6_alpha_beta_sweep_E5.tex", "Suitability weight sweep E5 (n=500 each). "
          "$\\alpha$ weights semantic suitability $S$, $\\beta$ weights spectral suitability $P$ "
          "in $J = \\max(S,\\epsilon)^\\alpha \\max(P,\\epsilon)^\\beta$. Clean recovery is perfect "
          "for all configurations; mean SNR increases monotonically toward the spectral end.",
          "alphabetasweep-e5", "lccccc", ["$\\alpha$", "$\\beta$", "$n$", "RT acc", "Payload succ.",
          "SNR $\\mu$ (dB)"], t6_tex,
          "BER=0 and RT acc=100% for all rows (not shown). Final config alpha=0.70, beta=0.30.")

# =================================================================== TABLE 7
# Strength sweep (E6)
t7 = []
t7_tex = []
for k in ["0.0005", "0.001", "0.002", "0.005", "0.008", "0.01"]:
    a = E6["aggregate"][k]
    t7.append([
        k, 500,
        fmt(a["roundtrip_accuracy"]["mean"] * 100, 2, "%"),
        fmt(a["bit_error_rate"]["mean"], 6),
        fmt(a["payload_success"]["mean"] * 100, 2, "%"),
        fmt(a["snr_db"]["mean"], 2),
    ])
    t7_tex.append([
        k, 500,
        fmt(a["roundtrip_accuracy"]["mean"] * 100, 2, "\\%"),
        fmt(a["bit_error_rate"]["mean"], 6),
        fmt(a["payload_success"]["mean"] * 100, 2, "\\%"),
        fmt(a["snr_db"]["mean"], 2),
    ])
write_csv("table_7_strength_sweep_E6.csv",
          ["delta", "n", "roundtrip_acc(%)", "BER", "payload_success(%)", "SNR_mean(dB)"],
          t7)
write_tex("table_7_strength_sweep_E6.tex", "Embedding-strength sweep E6 (n=500 each, constant "
          "delta across coordinates, joint allocation). Larger delta lowers SNR but preserves "
          "perfect clean recovery over the tested range.",
          "strength-e6", "lcccc", ["$\\delta$", "$n$", "RT acc", "Payload succ.", "SNR $\\mu$ (dB)"],
          t7_tex, "Range 0.0005-0.010; all configurations achieve 100% clean payload recovery (n=500).")

# =================================================================== TABLE 8
# Authentication E8
t8 = [
    ["Intact watermarked audio authenticated", 500, 500, "100.0"],
    ["CORRUPTED payload (audio modified)", 500, 0, "0.0  "],
    ["WRONG payload", 500, 0, "0.0  "],
    ["No watermark", 500, 0, "0.0  "],
    ["Truncated (last half removed)", 500, 0, "0.0  "],
]
trunc_rejected = E8["confusion"]["truncation_rejected"]
t8_json = [["intact_authenticated", 500, 500], ["corrupted_payload", 500, 0],
           ["wrong_payload", 500, 0], ["no_watermark", 500, 0], ["truncated_last_half", 500, trunc_rejected]]
write_csv("table_8_authentication_E8.csv",
          ["scenario", "n", "accepted", "rejected"],
          [[s, n, a, n - a] for s, n, a in t8_json])
write_tex("table_8_authentication_E8.tex", "Authentication/verification E8 (n=500 scenarios "
          "each, same 500 recordings). A file is accepted iff the recovered 256-bit SHA-256 "
          "payload digest verifies. Truncation of the second half rejects 500/500 because all "
          "replicas of affected bits are lost.",
          "auth-e8", "lccc", ["Scenario", "$n$", "Accepted", "Rejected"],
          [[s, 500, a, 500 - a] for s, n, a in t8_json],
          "TP=500, FP=0, TN=500, FN=0 (per-scenario counts over the same 500 files; not a single 2x2 matrix over 1000 items).")

# =================================================================== TABLE 9/10
# Robustness E9 (AWGN) + E10 (attacks)
order9 = ["none", "awgn_40db", "awgn_30db", "awgn_20db", "awgn_10db"]
order10 = ["none", "awgn_20db", "resample_8k", "scale_0.9", "mp3_128", "lowpass_4k"]
lab9 = {"none": "None", "awgn_40db": "AWGN 40 dB", "awgn_30db": "AWGN 30 dB",
        "awgn_20db": "AWGN 20 dB", "awgn_10db": "AWGN 10 dB"}
lab10 = {"none": "None", "awgn_20db": "AWGN 20 dB", "resample_8k": "Resample 8 kHz",
         "scale_0.9": "Gain 0.9", "mp3_128": "MP3 128 kbps", "lowpass_4k": "Lowpass 4 kHz"}
t9 = []
t9_tex = []
for a in order9:
    n, ba, ber, succ = attack_agg(E9_by_attack, a)
    t9.append([lab9[a], n, fmt(ba * 100, 2, "%"), fmt(ber * 100, 2, "%"),
               fmt(succ * 100, 2, "%")])
    t9_tex.append([lab9[a], n, fmt(ba * 100, 2, "\\%"), fmt(ber * 100, 2, "\\%"),
                   fmt(succ * 100, 2, "\\%")])
write_csv("table_9_awgn_robustness_E9.csv",
          ["attack", "n", "bit_accuracy(%)", "BER(%)", "payload_success(%)"], t9)
write_tex("table_9_awgn_robustness_E9.tex", "AWGN robustness E9 (n=500 per level). Bit accuracy "
          "collapses toward 0.5 (chance) with noise; payload success is 0 under every AWGN level.",
          "awgn-e9", "lcccc", ["Attack", "$n$", "Bit acc.", "BER", "Payload succ."], t9_tex,
          "The sparse signed-coefficient formulation is NOT robust to AWGN; this table documents the limitation.")

t10 = []
t10_tex = []
for a in order10:
    n, ba, ber, succ = attack_agg(E10_by_attack, a)
    t10.append([lab10[a], n, fmt(ba * 100, 2, "%"), fmt(ber * 100, 2, "%"),
                fmt(succ * 100, 2, "%")])
    t10_tex.append([lab10[a], n, fmt(ba * 100, 2, "\\%"), fmt(ber * 100, 2, "\\%"),
                    fmt(succ * 100, 2, "\\%")])
write_csv("table_10_attack_robustness_E10.csv",
          ["attack", "n", "bit_accuracy(%)", "BER(%)", "payload_success(%)"], t10)
write_tex("table_10_attack_robustness_E10.tex", "Broader signal-processing attack robustness E10 "
          "(n=500 per attack). Best bit accuracy is gain-scaling 0.9 at 79.4%; payload success is "
          "0 under every attack.",
          "attacks-e10", "lcccc", ["Attack", "$n$", "Bit acc.", "BER", "Payload succ."], t10_tex,
          "snr stored in E9/E10 is PRE-attack embedding SNR and is identical across attacks; it is intentionally not shown as an attack-quality metric.")

# =================================================================== TABLE 11
# Duration analysis E11
t11 = []
t11_tex = []
for b in ["<4s", "4-8s", "8-12s", "12-20s", ">20s"]:
    d = E11["bins"][b]
    t11.append([b, d["n"], fmt(d["roundtrip_accuracy"] * 100, 2, "%"),
                fmt(d["snr_db"], 2), fmt(d["payload_success_rate"] * 100, 2, "%")])
    t11_tex.append([b, d["n"], fmt(d["roundtrip_accuracy"] * 100, 2, "\\%"),
                    fmt(d["snr_db"], 2), fmt(d["payload_success_rate"] * 100, 2, "\\%")])
write_csv("table_11_duration_E11.csv",
          ["duration_bin", "n", "roundtrip_acc(%)", "SNR_mean(dB)", "payload_success(%)"], t11)
write_tex("table_11_duration_E11.tex", "Duration analysis E11 (all 2620 recordings, joint "
          "configuration). All bins achieve perfect payload recovery; mean SNR rises with duration.",
          "duration-e11", "lcccc", ["Duration", "$n$", "RT acc", "SNR $\\mu$ (dB)", "Payload succ."],
          t11_tex, "Bins from final_analysis/E11_duration_bins.json.")

# =================================================================== TABLE 12
# Speaker analysis E12
rows12 = []
for sp, d in E12["per_speaker"].items():
    rows12.append([sp, d["n"], fmt(d["roundtrip_accuracy"] * 100, 2, "%"),
                   fmt(d["snr_db"], 2), fmt(d["payload_success_rate"] * 100, 2, "%")])
rows12.sort(key=lambda r: r[0])
t12_tex = [[r[0], r[1], r[2].replace("%", "\\%"), r[3], r[4].replace("%", "\\%")] for r in rows12]
write_csv("table_12_speaker_E12.csv",
          ["speaker_id", "n", "roundtrip_acc(%)", "SNR_mean(dB)", "payload_success(%)"], rows12)
write_tex("table_12_speaker_E12.tex", "Per-speaker analysis E12 (40 speakers, 2620 recordings). "
          "Every speaker achieves 100% payload recovery; worst speaker success = 1.0.",
          "speaker-e12", "lcccc", ["Spkr", "$n$", "RT acc", "SNR $\\mu$ (dB)", "Payload succ."],
          t12_tex, "40 rows; summarized in plot E12. Source final_analysis/E12_per_speaker.json.")

# =================================================================== TABLE 13
# Paired stats E13
pairs = []
for p in E13["pairs"]:
    pairs.append([f"{p['a']} vs {p['b']}", p["delta_mean_snr_A_minus_B"], p["t"], p["p"]])
eff = {}
if E13_eff_csv:
    for r in E13_eff_csv:
        eff[f"{r['pair']}"] = float(r["cohen_dz"])
t13 = []
t13_tex = []
for a, b, t, p in pairs:
    key = f"{a}_vs_{b}"
    dz = eff.get(key, "")
    t13.append([key, fmt(a - b, 4) if False else fmt(float(a) if isinstance(a, float) else 0, 0, ""), "", "", ""])
# careful: rebuild properly
t13 = []
t13_tex = []
for p in E13["pairs"]:
    a, b = p["a"], p["b"]
    key = f"{a} vs {b}"
    dz = eff.get(f"{a}_vs_{b}", "")
    t13.append([key, fmt(p["delta_mean_snr_A_minus_B"], 4), fmt(p["t"], 2),
                p["p"], fmt(dz, 3) if dz else ""])
    t13_tex.append([key, fmt(p["delta_mean_snr_A_minus_B"], 4), fmt(p["t"], 2),
                    "$<10^{-16}$", fmt(dz, 3) if dz else ""])
write_csv("table_13_paired_stats_E13.csv",
          ["pair", "mean_diff_SNR_A-B(dB)", "t", "p", "cohen_dz"], t13)
write_tex("table_13_paired_stats_E13.tex", "Paired statistical comparisons E13 (n=2620, paired "
          "on identical recordings). p values printed as 0.0 in the raw output due to numerical "
          "underflow; all are below machine precision for double precision ($<10^{-16}$), "
          "i.e., extremely small. Cohen's $d_z$ from paper_analysis E13_effect_sizes.csv.",
          "paired-e13", "lcccc", ["Pair", "$\\bar{\\Delta}$SNR (dB)", "$t$", "$p$", "$d_z$"],
          t13_tex, "Significance does not imply practical superiority (mean differences ~1-3.9 dB).")

# =================================================================== TABLE 14
# Compute cost E14
order14 = ["joint", "semantic", "spectral", "random"]
t14 = []
t14_tex = []
for m in order14:
    rep = E14["report"][m]
    t14.append([m, rep["n"], fmt(rep["total_seconds"], 1), fmt(rep["total_seconds"] / rep["n"], 4),
                fmt(rep["mean_per_component_s"]["embeddings"], 4),
                fmt(rep["mean_per_component_s"]["embed+istft"], 4),
                fmt(rep["mean_per_component_s"]["allocation"], 4)])
    t14_tex.append([m, rep["n"], fmt(rep["total_seconds"], 1), fmt(rep["total_seconds"] / rep["n"], 4),
                    fmt(rep["mean_per_component_s"]["embeddings"], 4),
                    fmt(rep["mean_per_component_s"]["embed+istft"], 4),
                    fmt(rep["mean_per_component_s"]["allocation"], 4)])
write_csv("table_14_compute_cost_E14.csv",
          ["method", "n", "total_seconds", "s_per_recording", "embeddings_s", "embed+istft_s", "allocation_s"],
          t14)
write_tex("table_14_compute_cost_E14.tex", "Per-record computational timing E14 (cached-embedding "
          "pipeline only). HuBERT inference is NOT included; embeddings were loaded from an "
          "on-disk cache.",
          "compute-e14", "lcccccc", ["Method", "$n$", "Total (s)", "Per-rec (s)", "Embed load (s)",
          "Embed+ISTFT (s)", "Alloc (s)"], t14_tex,
          "Do not report as end-to-end watermarking runtime; HuBERT forward pass excluded.")

# =================================================================== experiment register
def sr(e, group):
    return {  # canonical source stems
        "E1": "final_clean/E1/E1_{method}_2620.csv + E1_summary_2620.json",
        "E2": "final_clean/E2/E2_audio_quality_500.csv",
        "E3": "final_ablations/E3/E3_fixed_vs_adaptive_500.csv",
        "E4": "final_ablations/E4/E4_ablations_500.csv",
        "E5": "final_ablations/E5/E5_alpha_beta_sweep_500.csv",
        "E6": "final_ablations/E6/E6_strength_sweep_500.csv",
        "E7": "final_ablations/E7/E7_redundancy_500.csv",
        "E8": "final_crypto/E8/E8_confusion.json + E8_detail.csv",
        "E9": "final_robustness/E9/E9_awgn_500.csv (aggregate json)",
        "E10": "final_robustness/E10/E10_attacks_500.csv (aggregate json)",
        "E11": "final_analysis/E11/E11_duration_bins.json",
        "E12": "final_analysis/E12/E12_per_speaker.json",
        "E13": "final_analysis/E13/E13_paired_stats.json",
        "E14": "final_analysis/E14/E14_compute_cost.json",
    }[e]


register = [
    ["E1", "Clean-channel watermark performance across the full corpus", "LibriSpeech test-clean, all 2620", "2620",
     "joint / semantic / spectral / random selection; R=3; adaptive delta", sr("E1", ""), "Roundtrip/direct acc, BER, payload success, SNR, mean delta",
     "Yes", "2", "FIG2 / FIG6", "1 random-method payload failure (4446-2275-0028); SNR = signal-to-watermark-noise on time-domain audio"],
    ["E2", "Clean audio quality on representative subset (with E1 price)", "first 500 manifest rows", "500",
     "joint; R=3; adaptive delta", sr("E2", ""), "SNR mean/p95, perfect recovery", "Yes", "2", "-",
     "Same 500-row subset as E3-E10"],
    ["E3", "Fixed vs adaptive embedding strength", "first 500 manifest rows", "500 each",
     "joint; fixed delta=0.005 vs adaptive delta(J)", sr("E3", ""), "Roundtrip, BER, payload success, SNR, mean delta",
     "Yes", "4", "-", "Adaptive wins on SNR; equal recovery"],
    ["E4", "Selection ablation", "first 500 manifest rows", "500 each",
     "A semantic-only, B spectral-only, C same-distortion fixed, D random; joint reference", sr("E4", ""), "Success, BER, SNR, delta",
     "Yes", "5", "FIG6", "C uses constant delta to equalize distortion; SNR ordering spectral>joint>random>semantic"],
    ["E5", "Suitability weight sweep (alpha/beta)", "first 500 manifest rows", "500 each",
     "joint J = S^alpha * P^beta; (alpha,beta) in {0,0.25,0.5,0.7,0.75,1} x {1-alpha}", sr("E5", ""), "SNR by (alpha,beta); perfect recovery",
     "Yes", "6", "FIG4", "Monotone SNR increase toward spectral end; perfect payload recovery all settings"],
    ["E6", "Embedding-strength sweep", "first 500 manifest rows", "500 each",
     "joint allocation; constant delta in {0.0005,0.001,0.002,0.005,0.008,0.01}", sr("E6", ""), "SNR by delta; perfect recovery",
     "Yes", "7", "FIG5", "Larger delta worsens SNR; no degradation of clean recovery over tested range"],
    ["E7", "Replica redundancy trade-off", "first 500 manifest rows", "500 each",
     "joint; R in {1,2,3}", sr("E7", ""), "Payload success per R, roundtrip, BER, SNR, failures",
     "Yes", "3", "FIG3 (R1/R2/R3)", "R=3 is the only zero-failure config; NOT universally optimal claim allowed"],
    ["E8", "Authentication and verification", "first 500 manifest rows", "500 scenarios",
     "SHA-256 payload digest; 5 scenarios (intact, corrupted, wrong payload, no watermark, truncated)", sr("E8", ""), "TP/TN/FP/FN, truncation rejection 500/500",
     "Yes", "8", "-", "TP/TN are per-scenario counts over the same 500 files"],
    ["E9", "AWGN robustness", "first 500 manifest rows", "500 per level",
     "AWGN 40/30/20/10 dB; joint; R=3", sr("E9", ""), "Bit accuracy, BER, payload success",
     "Yes (limitation)", "9", "FIG8", "Payload success 0 under all AWGN; pre-attack snr_db identical across attacks"],
    ["E10", "Broader signal-processing attacks", "first 500 manifest rows", "500 per attack",
     "resample 8k, lowpass 4k, gain 0.9, mp3 128, awgn 20dB; joint; R=3", sr("E10", ""), "Bit accuracy, BER, payload success",
     "Yes (limitation)", "10", "FIG8", "Payload success 0 under all attacks"],
    ["E11", "Duration analysis", "LibriSpeech test-clean, all 2620", "2620",
     "joint; R=3; grouped by duration <4/4-8/8-12/12-20/>20 s", sr("E11", ""), "SNR by bin; all succeed",
     "Yes", "11", "FIG7", "SNR rises with duration"],
    ["E12", "Speaker analysis", "LibriSpeech test-clean, all 2620", "40 speakers",
     "joint; R=3", sr("E12", ""), "Per-speaker SNR and success", "Yes", "12", "FIG E12",
     "Worst speaker success 1.0; 40/40 speakers"],
    ["E13", "Paired statistical comparison", "LibriSpeech test-clean, all 2620", "2620",
     "paired t-tests and Cohen's dz on per-recording SNR across 6 method pairs", sr("E13", ""), "mean diff, t, p, dz",
     "Yes", "13", "FIG E13", "p underflows to 0.0 -> report p < machine precision"],
    ["E14", "Computational timing (cached embeddings)", "LibriSpeech test-clean, E1 run traces", "2620",
     "per-component timing: stft, embeddings(cached), contribution, allocation, embed+istft, extract_stft", sr("E14", ""), "total s, per-component mean s",
     "Yes (caveat)", "14", "-", "HuBERT inference excluded (cache load only)"],
]
write_csv("01_CANONICAL_EXPERIMENT_REGISTER.csv",
          ["experiment_id", "purpose", "dataset/subset", "n", "methods/configurations",
           "canonical_source_file", "key_metrics", "suitable_for_paper", "paper_table",
           "paper_figure", "notes_caveats"],
          register, subdir=".")
print("tables + register written")