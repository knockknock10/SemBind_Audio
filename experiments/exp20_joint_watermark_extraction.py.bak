from __future__ import annotations

"""SemBind-Audio Exp20: upgraded 3x redundant watermark extraction.

This replaces the earlier Exp20 logic. It validates Exp21's 3 replicas/bit,
extracts each replica from the direct saved STFT and from the WAV round trip,
then performs payload-level majority voting. The current embedding rule is
reference-based/non-blind, so the clean reference STFT is required.
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SAMPLE_RATE = 16000
N_FFT, HOP_LENGTH, WIN_LENGTH = 1024, 320, 1024
WINDOW = torch.hann_window(WIN_LENGTH)
REDUNDANCY = 3
EPS = 1e-12

PAYLOAD = RESULTS / "watermark" / "payload_bits.pt"
REF_STFT = RESULTS / "redundant_embedding" / "original_stft.pt"
DIRECT_STFT = RESULTS / "redundant_embedding" / "redundant_watermarked_stft.pt"
COORDS = RESULTS / "redundant_embedding" / "redundant_embedding_coordinates.csv"
WM_AUDIO = RESULTS / "redundant_embedding" / "redundant_watermarked_audio.wav"
OUT = RESULTS / "joint_extraction"


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} not found:\n{path}")


def load_bits(path: Path) -> torch.Tensor:
    require(path, "payload")
    x = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    x = (x.detach().cpu().flatten().to(torch.int64) != 0).to(torch.int64)
    if x.numel() == 0:
        raise ValueError("Payload is empty.")
    return x


def load_stft(path: Path) -> torch.Tensor:
    require(path, "STFT")
    x = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not torch.is_complex(x):
        raise TypeError(f"Expected complex STFT in {path}")
    return x.detach().cpu()


def load_audio(path: Path) -> Tuple[torch.Tensor, int]:
    require(path, "watermarked WAV")
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        a = a.mean(axis=1)
    return torch.from_numpy(a).flatten().float(), int(sr)


def stft(audio: torch.Tensor) -> torch.Tensor:
    return torch.stft(audio.flatten().float(), n_fft=N_FFT,
                      hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                      window=WINDOW, center=True, return_complex=True)


def load_coordinates() -> List[Dict]:
    require(COORDS, "Exp21 coordinate CSV")
    with COORDS.open("r", newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise ValueError("Coordinate CSV is empty.")
    required = {
        "bit_index", "replica_index", "payload_bit", "region_id",
        "time_frame", "frequency_bin", "semantic_score",
        "semantic_suitability", "spectral_suitability",
        "joint_suitability", "adaptive_strength", "original_real",
        "watermarked_real", "delta_change",
    }
    missing = sorted(required - set(raw[0]))
    if missing:
        raise ValueError(f"Missing coordinate columns: {missing}\nAvailable: {list(raw[0])}")
    rows = []
    for r in raw:
        rows.append({
            "bit_index": int(float(r["bit_index"])),
            "replica_index": int(float(r["replica_index"])),
            "payload_bit": int(float(r["payload_bit"])),
            "region_id": int(float(r["region_id"])),
            "time_frame": int(float(r["time_frame"])),
            "frequency_bin": int(float(r["frequency_bin"])),
            "semantic_score": float(r["semantic_score"]),
            "semantic_suitability": float(r["semantic_suitability"]),
            "spectral_suitability": float(r["spectral_suitability"]),
            "joint_suitability": float(r["joint_suitability"]),
            "adaptive_strength": float(r["adaptive_strength"]),
            "original_real": float(r["original_real"]),
            "watermarked_real": float(r["watermarked_real"]),
            "delta_change": float(r["delta_change"]),
        })
    rows.sort(key=lambda x: (x["bit_index"], x["replica_index"]))
    return rows


def validate_mapping(payload: torch.Tensor, rows: List[Dict]) -> None:
    expected = int(payload.numel()) * REDUNDANCY
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} coordinates, found {len(rows)}.")
    coords = [(r["time_frame"], r["frequency_bin"]) for r in rows]
    if len(set(coords)) != len(coords):
        raise ValueError("Duplicate STFT coordinates exist in the mapping.")
    groups: Dict[int, List[Dict]] = {}
    for r in rows:
        groups.setdefault(r["bit_index"], []).append(r)
    if sorted(groups) != list(range(payload.numel())):
        raise ValueError("Bit-index coverage is not exactly 0..payload_length-1.")
    for i in range(payload.numel()):
        g = sorted(groups[i], key=lambda r: r["replica_index"])
        if [r["replica_index"] for r in g] != list(range(REDUNDANCY)):
            raise ValueError(f"Bit {i} does not contain replicas 0..{REDUNDANCY-1}.")
        expected_bit = int(payload[i])
        if any(r["payload_bit"] != expected_bit for r in g):
            raise ValueError(f"Payload mismatch in bit {i}.")


def extract_rows(reference: torch.Tensor, observed: torch.Tensor, rows: List[Dict]) -> List[Dict]:
    out = []
    for r in rows:
        f, t = r["frequency_bin"], r["time_frame"]
        if not (0 <= f < reference.shape[0] and 0 <= t < reference.shape[1]):
            raise IndexError(f"Coordinate ({f},{t}) outside STFT {tuple(reference.shape)}")
        ref = float(reference[f, t].real)
        obs = float(observed[f, t].real)
        delta = float(r["adaptive_strength"])
        if delta <= 0:
            raise ValueError(f"Non-positive delta at bit {r['bit_index']} replica {r['replica_index']}")
        change = obs - ref
        # Exact inverse of Exp21: positive change is bit 1, negative change is bit 0.
        bit = int(change >= 0.0)
        out.append({**r,
                    "reference_stft_real": ref,
                    "observed_stft_real": obs,
                    "observed_change": change,
                    "normalized_displacement": change / delta,
                    "extracted_bit": bit,
                    "correct": int(bit == r["payload_bit"]),
                    "reference_csv_error": abs(ref - r["original_real"])})
    return out


def majority(group: List[Dict]) -> Tuple[int, float, str]:
    g = sorted(group, key=lambda r: r["replica_index"])
    bits = [int(r["extracted_bit"]) for r in g]
    ones = sum(bits)
    zeros = len(bits) - ones
    return (1 if ones > zeros else 0,
            max(ones, zeros) / len(bits),
            "".join(map(str, bits)))


def weighted_vote(group: List[Dict]) -> Tuple[int, float]:
    # Diagnostic only. The paper's primary redundancy result is unweighted majority vote.
    score = 0.0
    weight_sum = 0.0
    for r in group:
        w = max(float(r["joint_suitability"]), EPS)
        score += w * math.tanh(float(r["normalized_displacement"]))
        weight_sum += w
    s = score / max(weight_sum, EPS)
    return int(s >= 0), min(1.0, abs(s))


def vote_payload(rows: List[Dict], payload: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
    groups: Dict[int, List[Dict]] = {}
    for r in rows:
        groups.setdefault(r["bit_index"], []).append(r)
    recovered, details = [], []
    for i in range(payload.numel()):
        g = sorted(groups[i], key=lambda r: r["replica_index"])
        bit, conf, pattern = majority(g)
        wbit, wconf = weighted_vote(g)
        correct_replicas = sum(int(r["correct"]) for r in g)
        expected = int(payload[i])
        details.append({
            "bit_index": i,
            "expected_bit": expected,
            "replica_0": int(g[0]["extracted_bit"]),
            "replica_1": int(g[1]["extracted_bit"]),
            "replica_2": int(g[2]["extracted_bit"]),
            "vote_pattern": pattern,
            "correct_replicas": correct_replicas,
            "majority_bit": bit,
            "majority_confidence": conf,
            "majority_correct": int(bit == expected),
            "weighted_bit": wbit,
            "weighted_confidence": wconf,
            "weighted_correct": int(wbit == expected),
        })
        recovered.append(bit)
    return torch.tensor(recovered, dtype=torch.int64), details


def metrics(expected: torch.Tensor, got: torch.Tensor) -> Dict:
    total = int(expected.numel())
    correct = int((expected == got).sum())
    return {"total_bits": total, "correct_bits": correct,
            "incorrect_bits": total - correct,
            "accuracy": correct / total if total else 0.0,
            "ber": (total - correct) / total if total else 0.0}


def replica_metrics(rows: List[Dict]) -> Dict:
    n = len(rows)
    c = sum(int(r["correct"]) for r in rows)
    return {"total_replicas": n, "correct_replicas": c,
            "incorrect_replicas": n-c,
            "accuracy": c/n if n else 0.0,
            "ber": (n-c)/n if n else 0.0}


def vote_metrics(details: List[Dict]) -> Dict:
    n = len(details)
    c = sum(int(r["majority_correct"]) for r in details)
    wc = sum(int(r["weighted_correct"]) for r in details)
    two = sum(int(r["correct_replicas"] >= 2) for r in details)
    three = sum(int(r["correct_replicas"] == 3) for r in details)
    return {"payload_bits": n, "majority_correct_bits": c,
            "majority_incorrect_bits": n-c,
            "majority_accuracy": c/n if n else 0.0,
            "majority_ber": (n-c)/n if n else 0.0,
            "weighted_accuracy": wc/n if n else 0.0,
            "weighted_ber": (n-wc)/n if n else 0.0,
            "bits_at_least_2_of_3": two,
            "bits_all_3_correct": three}


def save_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def save_bits(path: Path, bits: torch.Tensor) -> None:
    torch.save(bits.cpu(), path)
    path.with_suffix(".txt").write_text("".join(map(str, bits.tolist())) + "\n", encoding="utf-8")


def run_case(name: str, reference: torch.Tensor, observed: torch.Tensor,
             payload: torch.Tensor, rows: List[Dict]) -> Tuple[Dict, List[Dict], List[Dict]]:
    replica_rows = extract_rows(reference, observed, rows)
    recovered, details = vote_payload(replica_rows, payload)
    return {
        "name": name,
        "replica": replica_metrics(replica_rows),
        "payload": vote_metrics(details),
        "exact_verification": bool(torch.equal(payload, recovered)),
    }, replica_rows, details


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("SemBind-Audio : Experiment 20")
    print("Upgraded Redundant Semantic-Spectral Watermark Extraction")
    print("=" * 72)

    payload = load_bits(PAYLOAD)
    rows = load_coordinates()
    validate_mapping(payload, rows)

    reference = load_stft(REF_STFT)
    direct = load_stft(DIRECT_STFT)
    if reference.shape != direct.shape:
        raise ValueError(f"Reference/direct STFT shape mismatch: {reference.shape} vs {direct.shape}")

    print("\nPayload / Mapping Validation")
    print("-" * 72)
    print(f"Payload Length        : {payload.numel()} bits")
    print(f"Redundancy            : {REDUNDANCY}x")
    print(f"Coordinate Rows       : {len(rows)}")
    print("Payload Mapping       : PASSED")
    print(f"Reference STFT Shape  : {tuple(reference.shape)}")

    direct_summary, direct_rows, direct_details = run_case("direct_stft", reference, direct, payload, rows)
    print("\nDIRECT STFT EXTRACTION")
    print("-" * 72)
    print(f"Replica Accuracy      : {direct_summary['replica']['accuracy']:.6f}")
    print(f"Majority Accuracy     : {direct_summary['payload']['majority_accuracy']:.6f}")
    print(f"Majority BER          : {direct_summary['payload']['majority_ber']:.6f}")
    print(f"Verification          : {'PASSED' if direct_summary['exact_verification'] else 'FAILED'}")

    audio, sr = load_audio(WM_AUDIO)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz")
    roundtrip = stft(audio)
    if roundtrip.shape != reference.shape:
        raise ValueError(f"Round-trip STFT shape mismatch: {roundtrip.shape} vs {reference.shape}")

    rt_summary, rt_rows, rt_details = run_case("audio_roundtrip", reference, roundtrip, payload, rows)
    print("\nAUDIO ROUND-TRIP EXTRACTION")
    print("-" * 72)
    print(f"Watermarked Samples   : {audio.numel()}")
    print(f"Round-trip STFT Shape  : {tuple(roundtrip.shape)}")
    print(f"Replica Accuracy      : {rt_summary['replica']['accuracy']:.6f}")
    print(f"Replica BER           : {rt_summary['replica']['ber']:.6f}")
    print(f"Majority Accuracy     : {rt_summary['payload']['majority_accuracy']:.6f}")
    print(f"Majority BER          : {rt_summary['payload']['majority_ber']:.6f}")
    print(f"Weighted Accuracy     : {rt_summary['payload']['weighted_accuracy']:.6f}")
    print(f"Bits >=2/3 Correct    : {rt_summary['payload']['bits_at_least_2_of_3']}")
    print(f"Bits 3/3 Correct      : {rt_summary['payload']['bits_all_3_correct']}")
    print(f"Verification          : {'PASSED' if rt_summary['exact_verification'] else 'FAILED'}")

    direct_acc = direct_summary["payload"]["majority_accuracy"]
    rt_acc = rt_summary["payload"]["majority_accuracy"]
    replica_drop = direct_summary["replica"]["accuracy"] - rt_summary["replica"]["accuracy"]
    payload_drop = direct_acc - rt_acc
    ref_err = [r["reference_csv_error"] for r in rt_rows]
    disp = [r["normalized_displacement"] for r in rt_rows]

    print("\nFINAL REDUNDANCY RESULT")
    print("-" * 72)
    print(f"Direct Majority Accuracy     : {direct_acc:.6f}")
    print(f"Round-trip Majority Accuracy : {rt_acc:.6f}")
    print(f"Majority Accuracy Drop       : {payload_drop:.6f}")
    print(f"Round-trip Replica Accuracy  : {rt_summary['replica']['accuracy']:.6f}")
    print(f"Replica Accuracy Drop        : {replica_drop:.6f}")
    print(f"Mean Reference CSV Error     : {np.mean(ref_err):.12e}")
    print(f"Max Reference CSV Error      : {np.max(ref_err):.12e}")

    save_bits(OUT / "redundant_direct_extraction_bits.pt", torch.tensor([d["majority_bit"] for d in direct_details], dtype=torch.int64))
    save_bits(OUT / "redundant_roundtrip_extraction_bits.pt", torch.tensor([d["majority_bit"] for d in rt_details], dtype=torch.int64))
    save_csv(OUT / "redundant_direct_extraction_details.csv", direct_rows)
    save_csv(OUT / "redundant_roundtrip_replica_details.csv", rt_rows)
    save_csv(OUT / "redundant_payload_vote_details.csv", rt_details)

    summary = {
        "experiment": 20,
        "method": "Redundant Semantic-Spectral Watermark Extraction",
        "source_embedding_experiment": 21,
        "extraction_type": "reference-based / non-blind",
        "payload_bits": int(payload.numel()),
        "redundancy": REDUNDANCY,
        "stft_configuration": {"n_fft": N_FFT, "hop_length": HOP_LENGTH, "win_length": WIN_LENGTH},
        "direct_stft": direct_summary,
        "audio_roundtrip": rt_summary,
        "comparison": {"majority_accuracy_drop": payload_drop, "replica_accuracy_drop": replica_drop},
        "roundtrip_normalized_displacement": {
            "mean": float(np.mean(disp)), "std": float(np.std(disp)),
            "min": float(np.min(disp)), "max": float(np.max(disp))
        },
        "outputs": [
            str(OUT / "redundant_direct_extraction_bits.pt"),
            str(OUT / "redundant_roundtrip_extraction_bits.pt"),
            str(OUT / "redundant_direct_extraction_details.csv"),
            str(OUT / "redundant_roundtrip_replica_details.csv"),
            str(OUT / "redundant_payload_vote_details.csv"),
        ],
    }
    (OUT / "redundant_extraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nResults Saved")
    print(OUT)
    print("\n" + "=" * 72)
    print("Experiment 20 Upgrade Complete")
    print("=" * 72)


if __name__ == "__main__":
    main()
