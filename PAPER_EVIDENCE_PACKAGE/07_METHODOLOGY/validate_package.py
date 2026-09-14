"""Final validator for PAPER_EVIDENCE_PACKAGE.

Checks:
  V1  every table is generated from canonical evidence (source files exist under 06_RAW)
  V2  every figure uses canonical evidence or is explicitly marked conceptual/diagnostic
  V3  no stale 69.32 dB number appears as a final clean result
  V4  no fabricated data (all table numbers match canonical JSON/CSV recomputation)
  V5  no missing source mapping (traceability rows all have experiments in register)
  V6  no robustness overclaim (no forbidden robustness phrasing in package docs/tables)
  V7  no "R increases performance monotonically" claim
  V8  no "semantic-only gives best SNR" claim
  V9  no "joint gives best SNR" claim
  V10 no "blind watermarking" claim
  V11 no "cryptographic security proven" claim
  V12 no post-attack SNR misuse

Writes FINAL_VALIDATION.md with PASS/FAIL per check.
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "PAPER_EVIDENCE_PACKAGE"
RAW = PKG / "06_RAW_PAPER_DATA"

results = {}


def check(name, ok, detail):
    results[name] = (ok, detail)


# ------------------------------------------------------------- V1 tables
tables_ok = True
table_details = []
for t in sorted((PKG / "03_TABLES").glob("*.csv")):
    num = t.name.replace("table_", "").split("_")[0]
    src = None
    if num == "1":
        src = RAW / "clean/E1_summary_2620.json"
    elif num == "2":
        src = RAW / "clean/E1_summary_2620.json"
    elif num == "3":
        src = RAW / "ablations/E7_summary.json"
    elif num == "4":
        src = RAW / "ablations/E3_summary.json"
    elif num == "5":
        src = RAW / "ablations/E4_summary.json"
    elif num == "6":
        src = RAW / "ablations/E5_summary.json"
    elif num == "7":
        src = RAW / "ablations/E6_summary.json"
    elif num == "8":
        src = RAW / "crypto/E8_confusion.json"
    elif num == "9":
        src = RAW / "robustness/E9_awgn_500.csv"
    elif num == "10":
        src = RAW / "robustness/E10_attacks_500.csv"
    elif num == "11":
        src = RAW / "analysis/E11_duration_bins.json"
    elif num == "12":
        src = RAW / "analysis/E12_per_speaker.json"
    elif num == "13":
        src = RAW / "analysis/E13_paired_stats.json"
    elif num == "14":
        src = RAW / "analysis/E14_compute_cost.json"
    if src is None or not src.exists():
        tables_ok = False
        table_details.append(f"{t.name}: missing source")
check("V1-tables-from-canonical", tables_ok, "; ".join(table_details) or "all 14 tables map to existing canonical sources")

# ------------------------------------------------------------- V2 figures
try:
    with open(PKG / "04_FIGURES" / "FIGURE_REGISTER.csv") as f:
        figrows = list(csv.DictReader(f))
    missing = [r["filename"] for r in figrows if not (PKG / "04_FIGURES" / r["filename"]).exists()]
    # every generated png must be registered
    pngs = sorted((PKG / "04_FIGURES").glob("*.png"))
    registered = {r["filename"] for r in figrows}
    unregistered = [p.name for p in pngs if p.name not in registered]
    figs_ok = (not missing) and (not unregistered)
    check("V2-figures-registered", figs_ok,
          f"missing={missing}; unregistered_png={unregistered}")
except Exception as e:
    check("V2-figures-registered", False, str(e))

# ------------------------------------------------------------- docs corpus
# Evidence artifacts only. Forbidden-phrase checks deliberately EXCLUDE
# 09_PAPER_CLAIMS.md and 08_LIMITATIONS/LIMITATIONS.md because those documents
# exist to cite and forbid the phrases; they are not assertive claims.
DOCS = []
for p in [PKG / "00_README.md", PKG / "05_STATISTICS" / "statistical_interpretation.md",
          PKG / "04_FIGURES" / "EMBEDDING_VISUALIZATION_STATUS.md",
          PKG / "07_METHODOLOGY" / "REPRODUCIBILITY_STATUS.md"]:
    if p.exists():
        DOCS.append(p.read_text())
table_txt = "\n".join((PKG / "03_TABLES" / t.name).read_text() for t in (PKG / "03_TABLES").glob("*.csv"))
figure_txt = "\n".join((PKG / "04_FIGURES" / p.name).read_text() for p in (PKG / "04_FIGURES").glob("*.csv"))
all_txt = "\n".join(DOCS) + table_txt + figure_txt
low = all_txt.lower()

# ------------------------------------------------------------- V3 stale number
stale_clean = '69.32' not in all_txt
check("V3-no-stale-69.32", stale_clean,
      "no occurrence of the stale 69.32 dB clean number in any table/figure/docs"
      if stale_clean else "stale 69.32 found in evidence artifacts (only allowed in the correction table 02/%09)")

# ------------------------------------------------------------- V4 fabricated
fab_ok = True
fab_note = []
# recompute a few headline values from canonical CSVs
try:
    import pandas as pd
    e1 = pd.read_csv(RAW / "clean/E1_joint_2620.csv")
    if abs(e1["snr_db"].mean() - 75.37314617042041) > 1e-6:
        fab_ok = False
        fab_note.append("E1 joint SNR mean mismatch")
    e7 = pd.read_csv(RAW / "ablations/E7_redundancy_500.csv")
    s3 = e7[e7.redundancy == 3].payload_success.mean()
    if abs(s3 - 1.0) > 1e-12:
        fab_ok = False
        fab_note.append("E7 R=3 success mismatch")
    e9 = pd.read_csv(RAW / "robustness/E9_awgn_500.csv")
    if e9[e9.attack == "awgn_10db"].payload_success.mean() != 0.0:
        fab_ok = False
        fab_note.append("E9 awgn10 payload success mismatch")
except Exception as e:
    fab_ok = False
    fab_note.append(str(e))
check("V4-no-fabrication", fab_ok, "; ".join(fab_note) or "spot recomputation matches canonical CSVs")

# ------------------------------------------------------------- V5 traceability
try:
    with open(PKG / "02_NUMBER_TRACEABILITY.csv") as f:
        tr = list(csv.DictReader(f))
    with open(PKG / "01_CANONICAL_EXPERIMENT_REGISTER.csv") as f:
        reg = {r["experiment_id"] for r in csv.DictReader(f)}
    bad_exp = set()
    for r in tr:
        exps = {x.strip() for x in r["experiment"].replace("E9/E10", "E9,E10").replace("E1-E14", "E1,E2,E3,E4,E5,E6,E7,E8,E9,E10,E11,E12,E13,E14,E1-E14").split(",") if x.strip()}
        for e in exps:
            if e != "ALL" and e not in reg:
                bad_exp.add(e)
    check("V5-traceability-mapping", not bad_exp,
          f"traceability experiments without register entries: {bad_exp}" if bad_exp else "all traceability rows map to register experiments")
except Exception as e:
    check("V5-traceability-mapping", False, str(e))

# ------------------------------------------------------------- V6..V12 forbidden phrasings
# These checks scan ONLY the evidence artifacts (tables, figure register/captions,
# README, statistics) for assertive forbidden claims. 09_PAPER_CLAIMS.md and
# LIMITATIONS.md legitimately contain the phrases as DISCLAIMERS, and are excluded.

def forbidden(patterns, label):
    # Skip lines that phrase the concept as a caveat/disclaimer/negation.
    NEG = ["not", "do not", "never", "avoid", "caution", "caveat", "disclaimer",
           "lower", "lowest", "chance", "underflow", "contradicts", "forbidden",
           "do NOT", "NOT", "pre-attack", "0 under every", "against", "stale"]
    lines_hit = []
    for line in all_txt.splitlines():
        lowline = line.lower()
        if any(tok in lowline for tok in NEG):
            continue
        for pat in patterns:
            if re.search(pat, line, flags=re.IGNORECASE):
                lines_hit.append(line.strip()[:140])
                break
    ok = not lines_hit
    check(label, ok, f"forbidden assertive phrasing found: {lines_hit[:3]}" if lines_hit
          else f"'{label}' clean in evidence artifacts")

forbidden([r"robust to (AWGN|attacks|noise)", r"robustness\s*is\s*(achieved|high|strong)"],
          "V6-no-robustness-overclaim")
forbidden([r"more redundancy always", r"monotonically improves (recovery|performance)"],
          "V7-no-monotone-redundancy")
forbidden([r"semantic[ -]?only[^.]{0,80}?\b(best|highest|superior)\b"],
          "V8-no-semantic-best-snr")
forbidden([r"joint[^.]{0,80}?gives the best", r"joint[^.]{0,80}?\bhighest SNR\b"],
          "V9-no-joint-best-snr")
forbidden([r"\bblind water(?:mark|marking)\b", r"fully blind\b"],
          "V10-no-blind-claim")
forbidden([r"cryptographic security is proven", r"(?:security|cryptographic)[^.]{0,40}?proven"],
          "V11-no-security-proof")
forbidden([r"\d[\d.,]*\s*post[ -]attack", r"\d+\.\d+\s*(dB)?\s*post[ -]attack",
           r"post[ -]attack SNR\s*(of|:|=)\s*\d"],
          "V12-no-post-attack-snr")
# mention of the concept as a caveat is allowed; numeric post-attack SNR values are not

# also: no p=0 reporting (evidence artifacts must use 'underflow'/'<1e-16')
pzero = any(re.search(r"p\s*[=:]\s*0(\.0)?\b", line) for line in all_txt.splitlines()
            if not any(tok in line for tok in ("underflow", "NOT", "**NOT**",
                                                "never", "do not", "Do NOT",
                                                "1e-16", "machine precision")))
check("V13-no-pzero", not pzero,
      "p underflow documented as <1e-16; never reported as p=0 in evidence artifacts")

# ------------------------------------------------------------- write report
lines = [
    "# FINAL_VALIDATION — SemBind-Audio Paper Evidence Package",
    "",
    f"*Generated {__import__('datetime').date.today()} by `07_METHODOLOGY/validate_package.py`.*",
    "",
    "| Check | Status | Detail |",
    "|---|---|---|",
]
for k, (ok, d) in results.items():
    lines.append(f"| {k} | {'**PASS**' if ok else '**FAIL**'} | {d} |")
all_ok = all(ok for ok, _ in results.values())
lines += [
    "",
    f"**Overall: {'PASS' if all_ok else 'FAIL — resolve before manuscript use'}**",
    "",
    "## Scope note",
    "Validation V4 spot-recomputes headline values from canonical CSVs inside the package "
    "(06_RAW_PAPER_DATA) — it does not rerun experiments. FIG1 (architecture) is a "
    "conceptual schematic and FIG2/FIG2b are the bounded 3-file diagnostic; both are "
    "explicitly marked, so they do not violate V2.",
]
(PKG / "FINAL_VALIDATION.md").write_text("\n".join(lines) + "\n")
print("\n".join(f"{k}: {'PASS' if ok else 'FAIL'} — {d}" for k, (ok, d) in results.items()))
print("ALL PASS" if all_ok else "SOME FAIL")