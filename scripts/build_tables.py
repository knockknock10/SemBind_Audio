"""
SemBind-Audio: build final result tables (Phase 10) from validated results.

Reads results/final_*/**/*.json + *.csv and emits:
  results/final_tables/TABLE_1..N.json  (machine-readable)
  results/final_tables/TABLES.md        (markdown rendering)
"""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "final_tables"


def load_summary(path):
    return json.loads(path.read_text()) if path.exists() else None


def csv_agg(path, col, key_desc, value_fields):
    rows = [r for r in csv.DictReader(open(path))]
    out = {}
    for key, _desc in key_desc.items():
        grp = [r for r in rows if r[col] == key] if key != "overall" else rows
        if not grp:
            out[key] = {"n": 0}
            continue
        item = {"n": len(grp)}
        for vf in value_fields:
            vals = [float(r[vf]) for r in grp if r.get(vf) not in (None, "")]
            if vals:
                item[vf] = round(float(np.mean(vals)), 4)
        out[key] = item
    return out


def table1_e1():
    """E1: clean roundtrip per method (n=2620)."""
    t = {}
    for m in ["joint", "semantic", "spectral", "random"]:
        p = RESULTS / "final_clean" / "E1" / f"E1_{m}_2620.csv"
        rows = [r for r in csv.DictReader(open(p))]
        t[m] = {"n": len(rows),
                "payload_success_rate": round(float(np.mean([float(r["payload_success"]) for r in rows])), 4),
                "roundtrip_acc": round(float(np.mean([float(r["roundtrip_accuracy"]) for r in rows])), 4),
                "ber": round(float(np.mean([float(r["bit_error_rate"]) for r in rows])), 6),
                "snr_db": round(float(np.mean([float(r["snr_db"]) for r in rows])), 2)}
    return t


def table2_e2():
    d = load_summary(RESULTS / "final_clean" / "E2" / "E2_audio_quality_summary.json")
    return {"snr_mean_db": d.get("snr_db", {}).get("mean"),
            "snr_p95_db": d.get("snr_db", {}).get("p95")}


def table3_e3():
    return csv_agg(RESULTS / "final_ablations" / "E3" / "E3_fixed_vs_adaptive_500.csv", "mode",
                   {"fixed": "fixed", "adaptive": "adaptive"},
                   ["roundtrip_accuracy", "snr_db", "bit_error_rate"])


def table4_e4():
    return csv_agg(RESULTS / "final_ablations" / "E4" / "E4_ablations_500.csv", "variant",
                   {k: k for k in ["A_semantic_only", "B_spectral_only", "C_same_distortion_fixed", "D_random"]},
                   ["roundtrip_accuracy", "snr_db", "bit_error_rate"])


def table5_e5():
    return csv_agg(RESULTS / "final_ablations" / "E5" / "E5_alpha_beta_sweep_500.csv", "alpha",
                   {x: x for x in ["1.0", "0.75", "0.7", "0.5", "0.25", "0.0"]},
                   ["roundtrip_accuracy", "snr_db", "bit_error_rate"])


def table6_e6():
    return csv_agg(RESULTS / "final_ablations" / "E6" / "E6_strength_sweep_500.csv", "delta_value",
                   {x: x for x in ["0.0005", "0.001", "0.002", "0.005", "0.008", "0.01"]},
                   ["roundtrip_accuracy", "snr_db", "bit_error_rate"])


def table7_e7():
    return csv_agg(RESULTS / "final_ablations" / "E7" / "E7_redundancy_500.csv", "redundancy",
                   {x: x for x in ["1", "2", "3"]},
                   ["roundtrip_accuracy", "snr_db", "bit_error_rate"])


def table8_e8():
    d = load_summary(RESULTS / "final_crypto" / "E8" / "E8_confusion.json")
    return d.get("confusion", {})


def table9_e9():
    return csv_agg(RESULTS / "final_robustness" / "E9" / "E9_awgn_500.csv", "attack",
                   {a: a for a in ["none", "awgn_40db", "awgn_30db", "awgn_20db", "awgn_10db"]},
                   ["bit_accuracy", "ber", "payload_success", "snr_db"])


def table10_e10():
    return csv_agg(RESULTS / "final_robustness" / "E10" / "E10_attacks_500.csv", "attack",
                   {a: a for a in ["none", "resample_8k", "lowpass_4k", "scale_0.9", "mp3_128", "awgn_20db"]},
                   ["bit_accuracy", "ber", "payload_success", "snr_db"])


def table11_e11():
    d = load_summary(RESULTS / "final_analysis" / "E11" / "E11_duration_bins.json")
    return d.get("bins", {})


def table12_e12():
    d = load_summary(RESULTS / "final_analysis" / "E12" / "E12_per_speaker.json")
    per = d.get("per_speaker", {})
    return {"speakers": len(per),
            "worst_success": d.get("overall", {}).get("worst_speaker_success"),
            "mean_snr_db": round(float(np.mean([v["snr_db"] for v in per.values()])), 2),
            "mean_success": round(float(np.mean([v["payload_success_rate"] for v in per.values()])), 4)}


def table13_e13():
    d = load_summary(RESULTS / "final_analysis" / "E13" / "E13_paired_stats.json")
    return d.get("pairs", [])


def table14_e14():
    d = load_summary(RESULTS / "final_analysis" / "E14" / "E14_compute_cost.json")
    return {m: {"total_s": round(v["total_seconds"], 1)} for m, v in d.get("report", {}).items()}


def build():
    TABLES.mkdir(parents=True, exist_ok=True)
    tables = {
        "TABLE_1_clean_roundtrip_E1": table1_e1(),
        "TABLE_2_audio_quality_E2": table2_e2(),
        "TABLE_3_fixed_vs_adaptive_E3": table3_e3(),
        "TABLE_4_selection_ablations_E4": table4_e4(),
        "TABLE_5_alpha_beta_sweep_E5": table5_e5(),
        "TABLE_6_strength_sweep_E6": table6_e6(),
        "TABLE_7_redundancy_E7": table7_e7(),
        "TABLE_8_crypto_E8": table8_e8(),
        "TABLE_9_awgn_robustness_E9": table9_e9(),
        "TABLE_10_attack_robustness_E10": table10_e10(),
        "TABLE_11_duration_bins_E11": table11_e11(),
        "TABLE_12_per_speaker_E12": table12_e12(),
        "TABLE_13_paired_stats_E13": table13_e13(),
        "TABLE_14_compute_cost_E14": table14_e14(),
    }
    for name, data in tables.items():
        (TABLES / f"{name}.json").write_text(json.dumps(data, indent=2, default=str))

    lines = ["# SemBind-Audio — Final Results Tables (2026-09-09)", ""]
    for name, data in tables.items():
        title = name.split("_", 2)[2]
        lines.append(f"## {name} — {title}")
        lines.append("```json")
        lines.append(json.dumps(data, indent=2, default=str))
        lines.append("```")
        lines.append("")
    (TABLES / "TABLES.md").write_text("\n".join(lines))
    print(f"wrote {len(tables)} tables -> {TABLES}")
    return tables


if __name__ == "__main__":
    build()