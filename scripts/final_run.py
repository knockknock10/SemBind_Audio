"""
SemBind-Audio: final experiment driver (E1-E14 + sanity).

Usage:
    python scripts/final_run.py <stage> [options]

Stages:
    sanity       5-recording smoke sanity suite
    E1           clean full-dataset (joint/semantic/spectral/random, redundancy=3)
    E2           audio quality (subset of E1 outputs)
    E3           fixed-strength vs adaptive (joint, delta_mode switch)
    E4           selection ablations A/B/C/D
    E5           alpha/beta sweep
    E6           strength sweep (fixed delta values)
    E7           redundancy 1/2/3 sweep
    E8           crypto verification (TP/TN/FP/FN)
    E9           AWGN robustness (500)
    E10          common-attack robustness (500)
    E11          duration-bin analysis (from E1 csv)
    E12          per-speaker analysis (from E1 csv)
    E13          paired statistical tests (from E1 csv)
    E14          compute-cost report (from E1 run logs)

Every stage writes: per-file CSV (when applicable), an aggregate/summary JSON that
embeds experiment_id + config_hash + git_commit, and records failures under
results/failures/<stage>.json.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch

import final_methodology as fm

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "evaluation_manifest.csv"
DEV100 = RESULTS / "evaluation_dev_subset.csv"

N_FULL = 2620
N_500 = 500

METHODS_CLEAN = ["joint", "semantic", "spectral", "random"]

E1_FIELDS = [
    "audio_id", "speaker_id", "duration_seconds", "method",
    "stft_frames", "hubert_frames", "coords_total", "coords_unique",
    "direct_accuracy", "roundtrip_accuracy", "bit_error_rate",
    "payload_success", "snr_db", "error_energy", "signal_energy",
    "mean_delta", "min_delta", "max_delta", "rng_seed",
]


def load_rows(subset):
    rows = list(csv.DictReader(open(MANIFEST)))
    if subset == "full":
        return rows
    if subset == "dev100":
        ids = [r["audio_id"] for r in csv.DictReader(open(DEV100))]
        return [r for r in rows if r["audio_id"] in ids]
    if subset == "500":
        ids = [r["audio_id"] for r in csv.DictReader(open(MANIFEST))][:N_500]
        return [r for r in rows if r["audio_id"] in ids]
    if subset.startswith("n="):
        n = int(subset.split("=")[1])
        return rows[:n]
    raise ValueError(subset)


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _aggregate(csv_path, fields, key_field="method"):
    """Aggregate a results CSV into per-method/overall stats."""
    stats = {}
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return {}
    cols = rows[0].keys()
    # column aliases -> human-readable metric name
    metric_cols = {
        "roundtrip_accuracy": ["roundtrip_accuracy"],
        "direct_accuracy": ["direct_accuracy"],
        "bit_accuracy": ["bit_accuracy"],
        "bit_error_rate": ["bit_error_rate", "ber"],
        "payload_success": ["payload_success"],
        "snr_db": ["snr_db"],
    }
    resolvable = {k: next((c for c in aliases if c in cols), None)
                  for k, aliases in metric_cols.items()}
    groups = {}
    for r in rows:
        groups.setdefault(r.get(key_field, "all"), []).append(r)
    for gname, grp in groups.items():
        agg = {"n": len(grp)}
        for k, col in resolvable.items():
            if col is None:
                continue
            vals = [float(r[col]) for r in grp if r.get(col) not in (None, "")]
            if vals:
                agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                          "min": float(np.min(vals)), "max": float(np.max(vals))}
        pc = resolvable["payload_success"]
        if pc and any(r.get(pc) not in (None, "") for r in grp):
            agg["payload_success_rate"] = float(
                np.mean([float(r[pc]) for r in grp if r.get(pc) not in (None, "")])
            )
        if any(r.get("mean_delta") not in (None, "") for r in grp):
            agg["mean_delta"] = float(np.mean([float(r["mean_delta"]) for r in grp]))
        stats[gname] = agg
    overall = {}
    for k, col in resolvable.items():
        if col and all(r.get(col) not in (None, "") for r in rows):
            overall[k] = float(np.mean([float(r[col]) for r in rows]))
    stats["overall"] = {**overall, "n": len(rows)}
    return stats


def run_one_recording(audio, sr, audio_id, payload, method, redundancy,
                      alpha, beta, delta_mode, attacks=None, n_repeat=1, record_times=False):
    """Full pipeline for one recording. Returns dict + (optional) timing traces."""
    t0 = time.perf_counter()
    stft = fm.compute_stft(audio)
    t_stft = time.perf_counter() - t0

    contribution = None
    t_emb = 0.0
    t_con = 0.0
    if method in ("semantic", "joint"):
        t0 = time.perf_counter()
        emb = fm.load_embeddings_cached(audio_id, audio)
        t_emb = time.perf_counter() - t0
        t0 = time.perf_counter()
        contribution = fm.compute_contribution(emb)
        t_con = time.perf_counter() - t0

    t0 = time.perf_counter()
    delta_map, key = fm.per_frame_delta(method, stft, contribution, audio_id,
                                        alpha=alpha, beta=beta, delta_mode=delta_mode)
    coords = fm.allocate_coordinates(stft.shape[1], method, key, audio_id, redundancy=redundancy)
    dvals = torch.cat([delta_map[c[1]].view(1) for c in coords])
    t_alloc = time.perf_counter() - t0

    t0 = time.perf_counter()
    wstft = fm.embed_watermark(stft, coords, payload, dvals, redundancy=redundancy)
    wav = fm.reconstruct_audio(wstft, len(audio))
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    rstft = fm.compute_stft(wav)
    t_extract_stft = time.perf_counter() - t0

    # direct (pre-roundtrip) extraction
    dec_d, _ = fm.extract_majority(stft, wstft, coords, redundancy=redundancy)
    q_direct = fm.quality_metrics(payload, dec_d, audio, wav)

    # clean roundtrip extraction
    dec_r, _ = fm.extract_majority(stft, rstft, coords, redundancy=redundancy)
    q_round = fm.quality_metrics(payload, dec_r, audio, wav)

    result = {
        "audio_id": audio_id,
        "method": method,
        "stft_frames": int(stft.shape[1]),
        "hubert_frames": int(contribution.numel()) if contribution is not None else -1,
        "coords_total": len(coords),
        "coords_unique": len(set(coords)),
        "direct_accuracy": q_direct["bit_accuracy"],
        "roundtrip_accuracy": q_round["bit_accuracy"],
        "bit_error_rate": q_round["ber"],
        "payload_success": q_round["payload_success"],
        "snr_db": q_round["snr_db"],
        "error_energy": q_round["error_energy"],
        "signal_energy": q_round["signal_energy"],
        "mean_delta": float(dvals.mean()),
        "min_delta": float(dvals.min()),
        "max_delta": float(dvals.max()),
    }

    if attacks:
        result["per_attack"] = {}
        for atk in attacks:
            wa = fm.apply_attack(wav, sr, atk)
            astft = fm.compute_stft(wa)
            dec_a, _ = fm.extract_majority(stft, astft, coords, redundancy=redundancy)
            result["per_attack"][atk] = fm.quality_metrics(payload, dec_a, audio, wav)

    if record_times:
        result["times_s"] = {
            "stft": round(t_stft, 4), "embeddings": round(t_emb, 4),
            "contribution": round(t_con, 4), "allocation": round(t_alloc, 4),
            "embed+istft": round(t_embed, 4), "extract_stft": round(t_extract_stft, 4),
        }
    return result


def emit_aggregate(stage, csv_path, fields, extra=None, key_field="method", n=None):
    agg = _aggregate(csv_path, fields, key_field)
    meta = fm.result_metadata(f"final-{stage}", extra={"n_recordings": n or (len(list(csv.DictReader(open(csv_path)))))})
    out_path = Path(csv_path).with_name(f"{Path(csv_path).stem}_aggregate.json")
    fm.save_json(out_path, {"aggregate": agg, **meta, "extra": extra or {}})
    return agg


def fail(stage, audio_id, err, csv_writer):
    rec = {"stage": stage, "audio_id": audio_id, "error": str(err)[:500]}
    record_failure(stage, rec)
    if csv_writer:
        csv_writer.writerow({"audio_id": audio_id, "error": str(err)[:300]})


def record_failure(stage, rec):
    p = RESULTS / "failures" / f"{stage}.json"
    data = {"stage": stage, "generated": "2026-09-09", "failures": []}
    if p.exists():
        data = json.loads(p.read_text())
    data["failures"].append(rec)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


# ================================================================ stages
def stage_sanity(args):
    rows = load_rows("dev100")[:5]
    n = 0
    failures = 0
    collect = []
    for r in rows:
        audio_id, path = r["audio_id"], r["path"]
        audio, sr = sf.read(path, dtype="float32")
        payload = fm.load_payload()
        for method in METHODS_CLEAN:
            try:
                res = run_one_recording(audio, sr, audio_id, payload, method,
                                        redundancy=3, alpha=fm.ALPHA, beta=fm.BETA,
                                        delta_mode="adaptive")
                res["speaker_id"] = r["speaker_id"]
                res["duration_seconds"] = float(r["duration_seconds"])
                assert res["roundtrip_accuracy"] >= 0.99, f"roundtrip <0.99 for {audio_id}/{method}"
                n += 1
                collect.append(res)
            except Exception as e:
                failures += 1
                fail("sanity", f"{audio_id}:{method}", e, None)
    out = RESULTS / "final_sanity" / "sanity.csv"
    _write_csv(out, collect, E1_FIELDS)
    agg = _aggregate(out, E1_FIELDS, "method")
    fm.save_json(RESULTS / "final_sanity" / "sanity_summary.json",
                 {"aggregate": agg, **fm.result_metadata("final-sanity"), "failures": failures})
    print(f"sanity: {n} runs, failures={failures}")


def stage_E1(args):
    rows = load_rows("full")
    payload = fm.load_payload()
    out_a = []  # aggregated per (method) summary
    for method in METHODS_CLEAN:
        out_csv = RESULTS / "final_clean" / "E1" / f"E1_{method}_2620.csv"
        rows_out = []
        t_bench = []
        t_start = time.perf_counter()
        for i, r in enumerate(rows):
            audio_id, path = r["audio_id"], r["path"]
            try:
                audio, sr = sf.read(path, dtype="float32")
                kwargs = {"redundancy": 3, "alpha": fm.ALPHA, "beta": fm.BETA,
                          "delta_mode": "adaptive", "record_times": True}
                res = run_one_recording(audio, sr, audio_id, payload, method, **kwargs)
                res.update({"speaker_id": r["speaker_id"], "duration_seconds": float(r["duration_seconds"]),
                            "rng_seed": method})
                t_bench.append(res["times_s"])
                rows_out.append(res)
            except Exception as e:
                fail("E1", f"{audio_id}:{method}", e, None)
            if (i + 1) % 200 == 0:
                print(f"  [{method}] {i+1}/{len(rows)} elapsed={time.perf_counter()-t_start:.1f}s")
        _write_csv(out_csv, rows_out, E1_FIELDS)
        agg = emit_aggregate("E1", out_csv, E1_FIELDS, extra={"method": method}, n=len(rows_out))
        agg_meta = fm.result_metadata(f"final-E1-{method}", extra={"n": len(rows_out)})
        # compute-cost trace for E14
        fm.save_json(RESULTS / "final_clean" / "E1" / f"compute_cost_{method}.json",
                     {"n": len(t_bench), "mean_s_per_rec": None,
                      "sum_s": sum(v["stft"] + v["embeddings"] + v["contribution"] + v["allocation"]
                                   + v["embed+istft"] + v["extract_stft"] for v in t_bench) if t_bench else 0.0,
                      "mean_per_component": {k: float(np.mean([v[k] for v in t_bench])) for k in
                                             ("stft", "embeddings", "contribution", "allocation",
                                              "embed+istft", "extract_stft")} if t_bench else {},
                      **agg_meta})
        out_a.append({"method": method, "aggregate": agg, "n": len(rows_out)})
    fm.save_json(RESULTS / "final_clean" / "E1" / "E1_summary_2620.json",
                 {"methods": out_a, **fm.result_metadata("final-E1")})


def stage_E2(args):
    rows = load_rows("500")
    payload = fm.load_payload()
    out = RESULTS / "final_clean" / "E2" / "E2_audio_quality_500.csv"
    rows_out = []
    components_total = {"snr_db": [], "error_energy": []}
    for r in rows:
        audio_id, path = r["audio_id"], r["path"]
        try:
            audio, sr = sf.read(path, dtype="float32")
            res = run_one_recording(audio, sr, audio_id, payload, "joint", redundancy=3,
                                    alpha=fm.ALPHA, beta=fm.BETA, delta_mode="adaptive")
            res.update({"speaker_id": r["speaker_id"], "duration_seconds": float(r["duration_seconds"])})
            rows_out.append(res)
            components_total["snr_db"].append(res["snr_db"])
            components_total["error_energy"].append(res["error_energy"])
        except Exception as e:
            fail("E2", audio_id, e, None)
    _write_csv(out, rows_out, E1_FIELDS)
    agg = _aggregate(out, E1_FIELDS)
    fm.save_json(RESULTS / "final_clean" / "E2" / "E2_audio_quality_summary.json",
                 {"aggregate": agg, "snr_db": {"mean": float(np.mean(components_total["snr_db"])),
                                               "p95": float(np.percentile(components_total["snr_db"], 95))},
                  **fm.result_metadata("final-E2")})
    print(f"E2 done: n={len(rows_out)} mean_snr={np.mean(components_total['snr_db']):.2f}")


def stage_build_const(modes, stage, name, rows=None, dv=None):
    """Generic stage that sweeps delta modes / delta values / alphas / betas / redundancies."""
    raise NotImplementedError


def stage_E3(args):
    rows = load_rows("500")
    payload = fm.load_payload()
    out = RESULTS / "final_ablations" / "E3" / "E3_fixed_vs_adaptive_500.csv"
    rows_out = []
    for mode in ["fixed", "adaptive"]:
        for r in rows:
            audio_id, path = r["audio_id"], r["path"]
            try:
                audio, sr = sf.read(path, dtype="float32")
                res = run_one_recording(audio, sr, audio_id, payload, "joint", redundancy=3,
                                        alpha=fm.ALPHA, beta=fm.BETA, delta_mode=mode)
                res["mode"] = mode
                rows_out.append(res)
            except Exception as e:
                fail("E3", f"{audio_id}:{mode}", e, None)
    for row in rows_out:
        row.update({"speaker_id": "", "duration_seconds": 0.0, "rng_seed": "E3"})
    fields = E1_FIELDS[:-1] + ["mode", "rng_seed"]
    _write_csv(out, rows_out, fields)
    agg = _aggregate(out, fields, "mode")
    fm.save_json(RESULTS / "final_ablations" / "E3" / "E3_summary.json",
                 {"aggregate": agg, **fm.result_metadata("final-E3")})
    print("E3 done")


def stage_E4(args):
    rows = load_rows("500")
    payload = fm.load_payload()
    out = RESULTS / "final_ablations" / "E4" / "E4_ablations_500.csv"
    rows_out = []
    variants = ["A_semantic_only", "B_spectral_only", "C_same_distortion_fixed", "D_random"]
    for variant in variants:
        for r in rows:
            audio_id, path = r["audio_id"], r["path"]
            try:
                audio, sr = sf.read(path, dtype="float32")
                if variant == "A_semantic_only":
                    # J = S^1.0 * P^0.0  (spectral removed -> P neutral)
                    res = run_one_recording(audio, sr, audio_id, payload, "joint", redundancy=3,
                                            alpha=1.0, beta=0.0, delta_mode="adaptive")
                elif variant == "B_spectral_only":
                    # J = S^0.0 * P^1.0  (semantic removed -> S neutral)
                    res = run_one_recording(audio, sr, audio_id, payload, "joint", redundancy=3,
                                            alpha=0.0, beta=1.0, delta_mode="adaptive")
                elif variant == "C_same_distortion_fixed":
                    res = run_one_recording(audio, sr, audio_id, payload, "joint", redundancy=3,
                                            alpha=fm.ALPHA, beta=fm.BETA, delta_mode="fixed")
                elif variant == "D_random":
                    res = run_one_recording(audio, sr, audio_id, payload, "random", redundancy=3,
                                            alpha=fm.ALPHA, beta=fm.BETA, delta_mode="adaptive")
                res["variant"] = variant
                rows_out.append(res)
            except Exception as e:
                fail("E4", f"{audio_id}:{variant}", e, None)
    fields = E1_FIELDS[:-1] + ["variant"]
    _write_csv(out, rows_out, fields)
    agg = _aggregate(out, fields, "variant")
    fm.save_json(RESULTS / "final_ablations" / "E4" / "E4_summary.json",
                 {"aggregate": agg, **fm.result_metadata("final-E4")})
    print("E4 done")


def stage_E5(args):
    rows = load_rows("500")
    payload = fm.load_payload()
    out = RESULTS / "final_ablations" / "E5" / "E5_alpha_beta_sweep_500.csv"
    rows_out = []
    settings = [(1.0, 0.0), (0.75, 0.25), (0.70, 0.30), (0.50, 0.50), (0.25, 0.75), (0.0, 1.0)]
    for alpha, beta in settings:
        for r in rows:
            audio_id, path = r["audio_id"], r["path"]
            try:
                audio, sr = sf.read(path, dtype="float32")
                res = run_one_recording(audio, sr, audio_id, payload, "joint", redundancy=3,
                                        alpha=alpha, beta=beta, delta_mode="adaptive")
                res["alpha"] = alpha
                res["beta"] = beta
                rows_out.append(res)
            except Exception as e:
                fail("E5", f"{audio_id}:{alpha}:{beta}", e, None)
    fields = E1_FIELDS[:-1] + ["alpha", "beta"]
    _write_csv(out, rows_out, fields)
    agg = _aggregate(out, fields, "alpha")
    fm.save_json(RESULTS / "final_ablations" / "E5" / "E5_summary.json",
                 {"aggregate": agg, **fm.result_metadata("final-E5")})
    print("E5 done")


def stage_E6(args):
    rows = load_rows("500")
    payload = fm.load_payload()
    out = RESULTS / "final_ablations" / "E6" / "E6_strength_sweep_500.csv"
    rows_out = []
    strengths = [0.0005, 0.001, 0.002, 0.005, 0.008, 0.010]
    for d in strengths:
        for r in rows:
            audio_id, path = r["audio_id"], r["path"]
            try:
                audio, sr = sf.read(path, dtype="float32")
                # use the joint suitability for ordering; strength forced to constant d
                res = run_one_recording(audio, sr, audio_id, payload, "joint", redundancy=3,
                                        alpha=fm.ALPHA, beta=fm.BETA, delta_mode=str(d))
                res["delta_value"] = d
                rows_out.append(res)
            except Exception as e:
                fail("E6", f"{audio_id}:{d}", e, None)
    fields = E1_FIELDS[:-1] + ["delta_value"]
    _write_csv(out, rows_out, fields)
    agg = _aggregate(out, fields, "delta_value")
    fm.save_json(RESULTS / "final_ablations" / "E6" / "E6_summary.json",
                 {"aggregate": agg, **fm.result_metadata("final-E6")})
    print("E6 done")


def stage_E7(args):
    rows = load_rows("500")
    payload = fm.load_payload()
    out = RESULTS / "final_ablations" / "E7" / "E7_redundancy_500.csv"
    rows_out = []
    for red in [1, 2, 3]:
        for r in rows:
            audio_id, path = r["audio_id"], r["path"]
            try:
                audio, sr = sf.read(path, dtype="float32")
                res = run_one_recording(audio, sr, audio_id, payload, "joint", redundancy=red,
                                        alpha=fm.ALPHA, beta=fm.BETA, delta_mode="adaptive")
                res["redundancy"] = red
                rows_out.append(res)
            except Exception as e:
                fail("E7", f"{audio_id}:r{red}", e, None)
    fields = E1_FIELDS[:-1] + ["redundancy"]
    _write_csv(out, rows_out, fields)
    agg = _aggregate(out, fields, "redundancy")
    fm.save_json(RESULTS / "final_ablations" / "E7" / "E7_summary.json",
                 {"aggregate": agg, **fm.result_metadata("final-E7")})
    print("E7 done")


def verify(payload, recovered):
    return torch.equal(torch.as_tensor(payload, dtype=torch.uint8),
                       torch.as_tensor(recovered, dtype=torch.uint8))


def stage_E8(args):
    """Binary authentication evaluation: TP/TN/FP/FN over scenarios."""
    rows = load_rows("500")
    payload = fm.load_payload()
    wrong = 1 - payload
    os.makedirs(RESULTS / "final_crypto" / "E8", exist_ok=True)
    scenarios = ["auth_intact", "auth_corrupted", "wrong_payload", "no_watermark",
                 "truncated_last_half"]
    counts = {s: {"total": 0, "detected_authentic": 0} for s in scenarios}
    detail = []
    for r in rows:
        audio_id, path = r["audio_id"], r["path"]
        audio, sr = sf.read(path, dtype="float32")
        stft = fm.compute_stft(audio)
        emb = fm.load_embeddings_cached(audio_id, audio)
        contrib = fm.compute_contribution(emb)
        dm, key = fm.per_frame_delta("joint", stft, contrib, audio_id)
        coords = fm.allocate_coordinates(stft.shape[1], "joint", key, audio_id, redundancy=3)
        dvals = torch.cat([dm[c[1]].view(1) for c in coords])
        wstft = fm.embed_watermark(stft, coords, payload, dvals, redundancy=3)
        wav = fm.reconstruct_audio(wstft, len(audio))

        cases = []
        # (1) intact authentic watermark
        rstft = fm.compute_stft(wav)
        dec, _ = fm.extract_majority(stft, rstft, coords, redundancy=3)
        cases.append(("auth_intact", verify(payload, dec)))
        # (2) corrupted audio (awgn 30)
        cw = fm.apply_attack(wav, sr, "awgn_30db")
        cstft = fm.compute_stft(cw)
        dec2, _ = fm.extract_majority(stft, cstft, coords, redundancy=3)
        cases.append(("auth_corrupted", verify(payload, dec2)))
        # (3) wrong payload key
        cases.append(("wrong_payload", verify(wrong, dec)))
        # (4) clean unwatermarked audio
        dec_clean, _ = fm.extract_majority(stft, stft, coords, redundancy=3)
        cases.append(("no_watermark", verify(payload, dec_clean)))
        # (5) truncated watermarked audio (half length kept): only replicas that
        #     survive in the shortened timeline can vote; any bit that loses all
        #     replicas makes authentication fail.
        half = wav[: len(wav) // 2]
        hstft = fm.compute_stft(half)
        surf = [(f, t) for f, t in coords if 0 <= t < hstft.shape[1] and 0 <= f < hstft.shape[0]]
        if len(surf) > 0:
            bits_h, valid_h = fm.extract_majority_surviving(stft, hstft, surf, redundancy=3,
                                                            expected_len=len(payload))
            auth_h = bool(valid_h.all().item()) and torch.as_tensor(payload, dtype=torch.uint8).equal(bits_h)
        else:
            auth_h = False
        cases.append(("truncated_last_half", auth_h))

        det = {}
        for name, acc in cases:
            det[name] = bool(acc)
        detail.append({"audio_id": audio_id, **det})

    # Binary authentication semantics (positive = authentic watermark detected):
    #   TP = intact authentic watermark verifies as authentic
    #   FN = intact authentic watermark wrongly rejected
    #   TN = unwatermarked audio correctly rejected
    #   FP = watermark payload verified against the WRONG expected payload yet "authentic"
    tp = sum(d["auth_intact"] for d in detail)
    fn = sum(not d["auth_intact"] for d in detail)
    tn = sum(not d["no_watermark"] for d in detail)
    fp = sum(d["wrong_payload"] for d in detail)
    confusion = {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                 "n": len(detail),
                 "truncation_rejected": sum(not d["truncated_last_half"] for d in detail)}
    fm.save_json(RESULTS / "final_crypto" / "E8" / "E8_confusion.json",
                 {"confusion": confusion, "detail_count": len(detail),
                  **fm.result_metadata("final-E8")})
    _write_csv(RESULTS / "final_crypto" / "E8" / "E8_detail.csv", detail,
               ["audio_id"] + scenarios)
    print(f"E8 done: TP={tp} FP={fp} TN={tn} FN={fn} (n={len(detail)})")


def stage_E9(args):
    rows = load_rows("500")
    payload = fm.load_payload()
    attacks = ["none", "awgn_40db", "awgn_30db", "awgn_20db", "awgn_10db"]
    out = RESULTS / "final_robustness" / "E9" / "E9_awgn_500.csv"
    rows_out = []
    for atk in attacks:
        for r in rows:
            audio_id, path = r["audio_id"], r["path"]
            try:
                audio, sr = sf.read(path, dtype="float32")
                res = run_one_recording(audio, sr, audio_id, payload, "joint", redundancy=3,
                                        alpha=fm.ALPHA, beta=fm.BETA, delta_mode="adaptive")
                # recompute with attack applied to the watermarked signal
                stft = fm.compute_stft(audio)
                emb = fm.load_embeddings_cached(audio_id, audio)
                contrib = fm.compute_contribution(emb)
                dm, key = fm.per_frame_delta("joint", stft, contrib, audio_id)
                coords = fm.allocate_coordinates(stft.shape[1], "joint", key, audio_id, redundancy=3)
                dvals = torch.cat([dm[c[1]].view(1) for c in coords])
                wstft = fm.embed_watermark(stft, coords, payload, dvals, redundancy=3)
                wav = fm.reconstruct_audio(wstft, len(audio))
                wa = fm.apply_attack(wav, sr, atk)
                rstft = fm.compute_stft(wa)
                dec, _ = fm.extract_majority(stft, rstft, coords, redundancy=3)
                q = fm.quality_metrics(payload, dec, audio, wav)
                res = {k: q[k] for k in
                       ["bit_accuracy", "ber", "payload_success", "snr_db"]}
                res.update({"audio_id": audio_id, "attack": atk})
                rows_out.append(res)
            except Exception as e:
                fail("E9", f"{audio_id}:{atk}", e, None)
    fields = ["audio_id", "attack", "bit_accuracy", "ber", "payload_success", "snr_db"]
    _write_csv(out, rows_out, fields)
    agg = _aggregate(out, fields, "attack")
    fm.save_json(RESULTS / "final_robustness" / "E9" / "E9_summary.json",
                 {"aggregate": agg, **fm.result_metadata("final-E9")})
    print("E9 done")


def stage_E10(args):
    rows = load_rows("500")
    payload = fm.load_payload()
    attacks = ["none", "resample_8k", "lowpass_4k", "scale_0.9", "mp3_128", "awgn_20db"]
    out = RESULTS / "final_robustness" / "E10" / "E10_attacks_500.csv"
    rows_out = []
    for atk in attacks:
        for r in rows:
            audio_id, path = r["audio_id"], r["path"]
            try:
                audio, sr = sf.read(path, dtype="float32")
                stft = fm.compute_stft(audio)
                emb = fm.load_embeddings_cached(audio_id, audio)
                contrib = fm.compute_contribution(emb)
                dm, key = fm.per_frame_delta("joint", stft, contrib, audio_id)
                coords = fm.allocate_coordinates(stft.shape[1], "joint", key, audio_id, redundancy=3)
                dvals = torch.cat([dm[c[1]].view(1) for c in coords])
                wstft = fm.embed_watermark(stft, coords, payload, dvals, redundancy=3)
                wav = fm.reconstruct_audio(wstft, len(audio))
                wa = fm.apply_attack(wav, sr, atk)
                rstft = fm.compute_stft(wa)
                dec, _ = fm.extract_majority(stft, rstft, coords, redundancy=3)
                q = fm.quality_metrics(payload, dec, audio, wav)
                res = {k: q[k] for k in
                       ["bit_accuracy", "ber", "payload_success", "snr_db"]}
                res.update({"audio_id": audio_id, "attack": atk})
                rows_out.append(res)
            except Exception as e:
                fail("E10", f"{audio_id}:{atk}", e, None)
    fields = ["audio_id", "attack", "bit_accuracy", "ber", "payload_success", "snr_db"]
    _write_csv(out, rows_out, fields)
    agg = _aggregate(out, fields, "attack")
    fm.save_json(RESULTS / "final_robustness" / "E10" / "E10_summary.json",
                 {"aggregate": agg, **fm.result_metadata("final-E10")})
    print("E10 done")


def stage_E11(args):
    csv_path = RESULTS / "final_clean" / "E1" / "E1_joint_2620.csv"
    rows = list(csv.DictReader(open(csv_path)))
    bins = [("<4s", 0, 4), ("4-8s", 4, 8), ("8-12s", 8, 12), ("12-20s", 12, 20), (">20s", 20, 1e9)]
    result = {}
    for name, lo, hi in bins:
        grp = [r for r in rows if lo <= float(r["duration_seconds"]) < hi]
        result[name] = {
            "n": len(grp),
            "roundtrip_accuracy": float(np.mean([float(r["roundtrip_accuracy"]) for r in grp])),
            "snr_db": float(np.mean([float(r["snr_db"]) for r in grp])),
            "payload_success_rate": float(np.mean([float(r["payload_success"]) for r in grp])),
        }
    fm.save_json(RESULTS / "final_analysis" / "E11" / "E11_duration_bins.json",
                 {"bins": result, **fm.result_metadata("final-E11")})
    print("E11 done:", json.dumps({k: v["n"] for k, v in result.items()}))


def stage_E12(args):
    csv_path = RESULTS / "final_clean" / "E1" / "E1_joint_2620.csv"
    rows = list(csv.DictReader(open(csv_path)))
    by_speaker = {}
    for r in rows:
        spk = r["speaker_id"]
        by_speaker.setdefault(spk, []).append(r)
    speq = {}
    for spk, grp in by_speaker.items():
        speq[spk] = {
            "n": len(grp),
            "roundtrip_accuracy": float(np.mean([float(r["roundtrip_accuracy"]) for r in grp])),
            "snr_db": float(np.mean([float(r["snr_db"]) for r in grp])),
            "payload_success_rate": float(np.mean([float(r["payload_success"]) for r in grp])),
        }
    overall = {
        "speakers": len(speq),
        "worst_speaker_success": min(v["payload_success_rate"] for v in speq.values()),
    }
    fm.save_json(RESULTS / "final_analysis" / "E12" / "E12_per_speaker.json",
                 {"per_speaker": speq, "overall": overall, **fm.result_metadata("final-E12")})
    print("E12 done")


def stage_E13(args):
    from scipy.stats import ttest_rel
    base = RESULTS / "final_clean" / "E1"
    methods = METHODS_CLEAN
    data = {}
    for m in methods:
        p = base / f"E1_{m}_2620.csv"
        data[m] = {r["audio_id"]: float(r["snr_db"]) for r in csv.DictReader(open(p))}
    ids = [a for a in data["joint"]]
    pairs = []
    for a in methods:
        for b in methods:
            if a >= b:
                continue
            x = [data[a][i] for i in ids]
            y = [data[b][i] for i in ids]
            stat, pv = ttest_rel(x, y)
            pairs.append({"a": a, "b": b, "t": float(stat), "p": float(pv),
                          "delta_mean_snr_A_minus_B": float(np.mean(x) - np.mean(y))})
    fm.save_json(RESULTS / "final_analysis" / "E13" / "E13_paired_stats.json",
                 {"pairs": pairs, "n": len(ids), **fm.result_metadata("final-E13")})
    for pr in pairs:
        print("  ", pr["a"], "vs", pr["b"], "delta=", round(pr["delta_mean_snr_A_minus_B"], 3),
              "p=", round(pr["p"], 5))
    print("E13 done")


def stage_E14(args):
    from pathlib import Path
    base = RESULTS / "final_clean" / "E1"
    report = {}
    for m in METHODS_CLEAN:
        p = base / f"compute_cost_{m}.json"
        csvp = base / f"E1_{m}_2620.csv"
        if p.exists():
            d = json.loads(p.read_text())
            n_rows = -1
            if csvp.exists():
                with open(csvp) as f:
                    n_rows = max(len(f.readlines()) - 1, 0)
            report[m] = {
                "n": n_rows,
                "total_seconds": d["sum_s"],
                "mean_per_component_s": d.get("mean_per_component", {}),
            }
    fm.save_json(RESULTS / "final_analysis" / "E14" / "E14_compute_cost.json",
                 {"report": report, **fm.result_metadata("final-E14")})
    print("E14 done:", {k: v["total_seconds"] for k, v in report.items()})


STAGES = {
    "sanity": stage_sanity,
    "E1": stage_E1,
    "E2": stage_E2,
    "E3": stage_E3,
    "E4": stage_E4,
    "E5": stage_E5,
    "E6": stage_E6,
    "E7": stage_E7,
    "E8": stage_E8,
    "E9": stage_E9,
    "E10": stage_E10,
    "E11": stage_E11,
    "E12": stage_E12,
    "E13": stage_E13,
    "E14": stage_E14,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--subset", default="full")
    args = ap.parse_args()
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()