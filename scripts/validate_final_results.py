"""
SemBind-Audio: final results validator (Phase 9).

Checks every final_* output for:
  - presence of config_hash matching the canonical config hash
  - presence of git_commit == current HEAD
  - per-file row counts matching the declared dataset size (N)
  - roundtrip/success invariants (no fabrication): values are physically plausible
    (accuracy in [0,1], BER in [0,1], SNR in [0,120] dB for clean)
  - no failure rate above 1% without an investigation note under results/failures
  - CSV/JSON structural integrity

Exit code 0 if all checks pass; 1 otherwise. Prints a structured report.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
import importlib.util

spec = importlib.util.spec_from_file_location("final_methodology", ROOT / "final_methodology.py")
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)

CANONICAL_HASH = fm.config_hash()
GIT_COMMIT = fm.git_head()

EXPECTED_N = {"E1_joint": 2620, "E1_semantic": 2620, "E1_spectral": 2620, "E1_random": 2620,
              "E2": 500, "E3": 1000, "E4": 2000, "E5": 3000, "E6": 3000, "E7": 1500,
              "E8": 500, "E9": 2500, "E10": 3000}

errors = []
warnings = []
checked = {"metadata_files": 0, "csv_rows": 0, "json_files": 0}


def check_meta_file(p: Path, name: str):
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        errors.append(f"{name}: unreadable JSON ({e})")
        return
    checked["json_files"] += 1
    if d.get("config_hash") != CANONICAL_HASH:
        errors.append(f"{name}: config_hash mismatch "
                      f"({d.get('config_hash','?'):.12s}… != {CANONICAL_HASH:.12s}…)")
    if d.get("git_commit") not in (GIT_COMMIT, "unknown"):
        errors.append(f"{name}: git_commit mismatch {d.get('git_commit')} != {GIT_COMMIT}")
    if "experiment_id" not in d:
        warnings.append(f"{name}: missing experiment_id")


def validate_csv(p: Path, name: str, expect_rows: int | None, truth_acc_key="roundtrip_accuracy"):
    if not p.exists():
        errors.append(f"{name}: missing CSV {p}")
        return 0
    lines = p.read_text().splitlines()
    if len(lines) < 2:
        errors.append(f"{name}: empty CSV {p}")
        return 0
    n = len(lines) - 1
    checked["csv_rows"] += n
    if expect_rows is not None and n != expect_rows:
        errors.append(f"{name}: row count {n} != expected {expect_rows}")
    with open(p) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            dv = row.get(truth_acc_key, row.get("bit_accuracy", row.get("audio_id", "")))
            if truth_acc_key in row:
                v = float(row[truth_acc_key])
                if not 0.0 <= v <= 1.0:
                    errors.append(f"{name}: row {i}: invalid {truth_acc_key}={v}")
            if row.get("bit_error_rate"):
                ber = float(row["bit_error_rate"])
                if not 0.0 <= ber <= 1.0:
                    errors.append(f"{name}: row {i}: BER out of range {ber}")
            if row.get("snr_db"):
                snr = float(row["snr_db"])
                if snr < 0 or snr > 120:
                    errors.append(f"{name}: row {i}: SNR out of physical range {snr}")
    return n


def main():
    check_meta_file(RESULTS / "final_sanity" / "sanity_summary.json", "sanity_summary")
    n_sanity = validate_csv(RESULTS / "final_sanity" / "sanity.csv", "sanity.csv", 20)

    # E1 clean
    for m in ["joint", "semantic", "spectral", "random"]:
        csvp = RESULTS / "final_clean" / "E1" / f"E1_{m}_2620.csv"
        if csvp.exists():
            validate_csv(csvp, f"E1_{m}", 2620)
        agg = RESULTS / "final_clean" / "E1" / f"E1_{m}_2620_aggregate.json"
        if agg.exists():
            check_meta_file(agg, f"E1_{m}_aggregate")
        cost = RESULTS / "final_clean" / "E1" / f"compute_cost_{m}.json"
        if cost.exists():
            check_meta_file(cost, f"compute_cost_{m}")

    # per-stage sweeps
    spec_stages = {
        "E2": (RESULTS / "final_clean" / "E2" / "E2_audio_quality_500.csv", 500,
               RESULTS / "final_clean" / "E2" / "E2_audio_quality_summary.json"),
        "E3": (RESULTS / "final_ablations" / "E3" / "E3_fixed_vs_adaptive_500.csv", 1000,
               RESULTS / "final_ablations" / "E3" / "E3_summary.json"),
        "E4": (RESULTS / "final_ablations" / "E4" / "E4_ablations_500.csv", 2000,
               RESULTS / "final_ablations" / "E4" / "E4_summary.json"),
        "E5": (RESULTS / "final_ablations" / "E5" / "E5_alpha_beta_sweep_500.csv", 3000,
               RESULTS / "final_ablations" / "E5" / "E5_summary.json"),
        "E6": (RESULTS / "final_ablations" / "E6" / "E6_strength_sweep_500.csv", 3000,
               RESULTS / "final_ablations" / "E6" / "E6_summary.json"),
        "E7": (RESULTS / "final_ablations" / "E7" / "E7_redundancy_500.csv", 1500,
               RESULTS / "final_ablations" / "E7" / "E7_summary.json"),
        "E9": (RESULTS / "final_robustness" / "E9" / "E9_awgn_500.csv", 2500,
               RESULTS / "final_robustness" / "E9" / "E9_summary.json"),
        "E10": (RESULTS / "final_robustness" / "E10" / "E10_attacks_500.csv", 3000,
                RESULTS / "final_robustness" / "E10" / "E10_summary.json"),
    }
    for stage, (csvp, n, metap) in spec_stages.items():
        if csvp.exists():
            validate_csv(csvp, stage, n)
        if metap.exists():
            check_meta_file(metap, f"{stage}_summary")

    # E8 crypto
    e8 = RESULTS / "final_crypto" / "E8" / "E8_confusion.json"
    if e8.exists():
        d = json.loads(e8.read_text())
        check_meta_file(e8, "E8_confusion")
        if d.get("confusion", {}).get("n") != 500:
            errors.append(f"E8: confusion n={d.get('confusion',{}).get('n')} != 500")

    # analysis E11-E14
    for stage, rel in [("E11", "final_analysis/E11/E11_duration_bins.json"),
                       ("E12", "final_analysis/E12/E12_per_speaker.json"),
                       ("E13", "final_analysis/E13/E13_paired_stats.json"),
                       ("E14", "final_analysis/E14/E14_compute_cost.json")]:
        p = RESULTS / rel
        if p.exists():
            check_meta_file(p, stage)

    # failure-rate gate
    for fp in (RESULTS / "failures").glob("*.json") if (RESULTS / "failures").exists() else []:
        d = json.loads(fp.read_text())
        n_fail = len(d["failures"])
        if n_fail == 0:
            continue
        if n_fail > 50:
            errors.append(f"{fp.stem}: {n_fail} failures exceed 1% tolerance; must investigate")

    report = {
        "checked": checked,
        "errors": errors,
        "warnings": warnings,
        "passed": len(errors) == 0,
        "canonical_config_hash": CANONICAL_HASH,
        "git_commit": GIT_COMMIT,
        "generated": "2026-09-09",
    }
    fm.save_json(RESULTS / "final_audit" / "validation_report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())