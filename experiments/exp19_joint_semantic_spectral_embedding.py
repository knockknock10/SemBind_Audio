"""
SemBind-Audio : Experiment 19
Joint Semantic-Spectral Adaptive STFT Embedding

Purpose
-------
This experiment implements a clean, reproducible version of the proposed
joint selection mechanism:

    HuBERT semantic contribution
              +
    local spectral suitability
              |
              v
      joint region suitability
              |
              v
     adaptive embedding strength
              |
              v
        STFT watermarking

Important methodological rules
------------------------------
1. The experiment always starts from results/original_audio.wav.
2. It does NOT reuse a previously reconstructed/watermarked waveform.
3. Semantic regions are reconstructed directly from the validated
   semantic-selection CSV rather than assuming that the payload mapping
   CSV contains semantic metadata.
4. Spectral suitability is calculated independently from the original STFT.
5. The joint score is normalized and used to control the embedding strength.
6. Audio is written with soundfile, avoiding the TorchCodec dependency of
   newer torchaudio versions.
7. Embedding statistics are calculated only from the coordinates that were
   actually modified.
8. The original and watermarked STFTs are saved for later extraction and
   ablation experiments.

Joint suitability
-----------------
Let S_r be normalized semantic suitability and P_r be normalized spectral
suitability for region r.

    J_r = S_r^alpha * P_r^beta

The adaptive perturbation strength is

    delta_r = delta_min + (delta_max - delta_min) * J_r

This retains the central SemBind-Audio idea: semantic importance controls
WHERE embedding is allowed, while spectral suitability provides an
additional local signal-domain constraint for HOW STRONGLY it is embedded.

This is an experimental research implementation, not a claim that
mean spectral energy is a complete psychoacoustic masking model.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch


# ============================================================================
# PROJECT / CONFIG
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

from configs.config import (
    ALPHA,
    BETA,
    DELTA_MIN,
    DELTA_MAX,
)

SAMPLE_RATE = 16000

N_FFT = 1024
HOP_LENGTH = 320
WIN_LENGTH = 1024

WINDOW = torch.hann_window(WIN_LENGTH)

PAYLOAD_PATH = RESULTS_DIR / "watermark" / "payload_bits.pt"
ORIGINAL_AUDIO_PATH = RESULTS_DIR / "original_audio.wav"

SEMANTIC_SCORE_PATH = (
    RESULTS_DIR
    / "semantic_validation"
    / "semantic_selection_validation.csv"
)

OUTPUT_DIR = RESULTS_DIR / "joint_embedding"

ORIGINAL_STFT_PATH = OUTPUT_DIR / "original_stft.pt"
WATERMARKED_STFT_PATH = OUTPUT_DIR / "joint_watermarked_stft.pt"
EMBEDDED_BITS_PATH = OUTPUT_DIR / "embedded_bits.pt"
COORDINATES_PATH = OUTPUT_DIR / "joint_embedding_coordinates.csv"
REGION_PARAMETERS_PATH = OUTPUT_DIR / "joint_region_parameters.csv"
WATERMARKED_AUDIO_PATH = OUTPUT_DIR / "joint_watermarked_audio.wav"
DIFFERENCE_AUDIO_PATH = OUTPUT_DIR / "joint_difference.wav"
SUMMARY_PATH = OUTPUT_DIR / "joint_embedding_summary.json"


# ============================================================================
# RESEARCH PARAMETERS
# ============================================================================

# Semantic threshold is already encoded by the validated CSV's "selected"
# column. We therefore do not invent another semantic threshold here.

# Candidate region filtering.
MIN_REGION_FRAMES = 2

# Use a conservative frequency band rather than DC / Nyquist bins.
FREQ_START_RATIO = 0.05
FREQ_END_RATIO = 0.95

# Joint score exponents.
ALPHA = 0.70
BETA = 0.30

# Adaptive embedding range.
# Values from central configuration:
#     DELTA_MIN = 0.002, DELTA_MAX = 0.008
# (Local assignments shadow the config import; using config values directly.)

# Small numerical constant.
EPS = 1e-12


# ============================================================================
# UTILITIES
# ============================================================================

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_payload() -> torch.Tensor:
    if not PAYLOAD_PATH.exists():
        raise FileNotFoundError(
            f"Payload file not found:\n{PAYLOAD_PATH}"
        )

    payload = torch.load(
        PAYLOAD_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, torch.Tensor):
        payload = torch.tensor(payload)

    payload = payload.detach().cpu().flatten().to(torch.int64)

    if payload.numel() == 0:
        raise ValueError("Payload contains zero bits.")

    payload = (payload != 0).to(torch.int64)

    return payload


def load_original_audio() -> Tuple[torch.Tensor, int]:
    if not ORIGINAL_AUDIO_PATH.exists():
        raise FileNotFoundError(
            "Clean original audio was not found:\n"
            f"{ORIGINAL_AUDIO_PATH}\n\n"
            "Exp19 must start from results/original_audio.wav."
        )

    waveform_np, sample_rate = sf.read(
        str(ORIGINAL_AUDIO_PATH),
        dtype="float32",
        always_2d=False,
    )

    waveform = torch.from_numpy(
        np.asarray(waveform_np, dtype=np.float32)
    )

    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)

    waveform = waveform.contiguous().float()

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected sample rate {SAMPLE_RATE}, "
            f"got {sample_rate}."
        )

    return waveform, int(sample_rate)


def compute_stft(waveform: torch.Tensor) -> torch.Tensor:
    waveform = waveform.float().flatten()

    return torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        center=True,
        return_complex=True,
    )


def reconstruct_audio(stft: torch.Tensor, length: int) -> torch.Tensor:
    waveform = torch.istft(
        stft,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        center=True,
        length=length,
    )

    return waveform.float()


# ============================================================================
# SEMANTIC SCORE LOADING
# ============================================================================

def load_semantic_scores() -> List[Dict]:
    if not SEMANTIC_SCORE_PATH.exists():
        raise FileNotFoundError(
            "Semantic validation file not found:\n"
            f"{SEMANTIC_SCORE_PATH}"
        )

    with SEMANTIC_SCORE_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Semantic score CSV is empty.")

    required = {"frame", "normalized_score", "selected"}

    missing = required.difference(rows[0].keys())

    if missing:
        raise ValueError(
            "Semantic CSV is missing required columns:\n"
            f"{sorted(missing)}\n\n"
            f"Available columns:\n{list(rows[0].keys())}"
        )

    parsed = []

    for row in rows:
        try:
            frame = int(float(row["frame"]))
            score = float(row["normalized_score"])

            selected_raw = str(row["selected"]).strip().lower()

            selected = selected_raw in {
                "1",
                "true",
                "yes",
                "selected",
            }

        except Exception as exc:
            raise ValueError(
                f"Invalid semantic row:\n{row}"
            ) from exc

        parsed.append(
            {
                "frame": frame,
                "normalized_score": score,
                "selected": selected,
            }
        )

    parsed.sort(key=lambda x: x["frame"])

    return parsed


# ============================================================================
# SEMANTIC REGION CONSTRUCTION
# ============================================================================

def merge_selected_runs(
    semantic_rows: List[Dict],
) -> List[Dict]:
    """
    Convert selected semantic frames into contiguous semantic regions.

    The CSV frame index is the native HuBERT-frame coordinate system.
    Regions are initially constructed there and only later mapped to STFT.
    """

    selected_frames = [
        row["frame"]
        for row in semantic_rows
        if row["selected"]
    ]

    if not selected_frames:
        raise ValueError(
            "No selected semantic frames were found."
        )

    selected_frames = sorted(set(selected_frames))

    runs = []

    start = selected_frames[0]
    previous = selected_frames[0]

    for frame in selected_frames[1:]:
        if frame == previous + 1:
            previous = frame
            continue

        runs.append((start, previous))
        start = frame
        previous = frame

    runs.append((start, previous))

    regions = []

    score_lookup = {
        row["frame"]: row["normalized_score"]
        for row in semantic_rows
    }

    region_id = 1

    for start, end in runs:
        length = end - start + 1

        if length < MIN_REGION_FRAMES:
            continue

        values = [
            score_lookup[f]
            for f in range(start, end + 1)
            if f in score_lookup
        ]

        if not values:
            continue

        regions.append(
            {
                "region_id": region_id,
                "semantic_start": start,
                "semantic_end": end,
                "semantic_score": float(np.mean(values)),
                "semantic_min": float(np.min(values)),
                "semantic_max": float(np.max(values)),
                "semantic_frames": length,
            }
        )

        region_id += 1

    if not regions:
        raise ValueError(
            "Semantic selection produced no usable regions."
        )

    return regions


def map_semantic_to_stft(
    semantic_frame: float,
    semantic_count: int,
    stft_count: int,
) -> int:
    """
    Map a HuBERT-frame coordinate to an STFT-frame coordinate using
    normalized temporal position.

    This is equivalent to the project's validated linear temporal alignment.
    """

    if semantic_count <= 1:
        return 0

    if stft_count <= 1:
        return 0

    position = semantic_frame / float(semantic_count - 1)

    mapped = round(
        position * float(stft_count - 1)
    )

    return int(
        max(
            0,
            min(stft_count - 1, mapped),
        )
    )


def map_regions_to_stft(
    regions: List[Dict],
    semantic_count: int,
    stft_count: int,
) -> List[Dict]:

    mapped = []

    for region in regions:
        start = map_semantic_to_stft(
            region["semantic_start"],
            semantic_count,
            stft_count,
        )

        end = map_semantic_to_stft(
            region["semantic_end"],
            semantic_count,
            stft_count,
        )

        if end < start:
            start, end = end, start

        item = dict(region)

        item["stft_start"] = start
        item["stft_end"] = end

        mapped.append(item)

    return mapped


# ============================================================================
# SPECTRAL SUITABILITY
# ============================================================================

def spectral_statistics(
    stft: torch.Tensor,
    region: Dict,
) -> Tuple[float, float]:
    """
    Compute local spectral suitability.

    The proxy is intentionally simple and explicit:

        log1p(|X(f,t)|)

    averaged over the eligible local region.

    Higher local magnitude gives greater spectral suitability because
    embedding in an already active spectral area generally offers more
    signal-domain headroom than embedding in a near-silent area.

    This is NOT described as a full psychoacoustic model.
    """

    freq_bins = stft.shape[0]
    time_frames = stft.shape[1]

    freq_start = max(
        1,
        int(freq_bins * FREQ_START_RATIO),
    )

    freq_end = min(
        freq_bins - 2,
        int(freq_bins * FREQ_END_RATIO),
    )

    time_start = max(
        0,
        int(region["stft_start"]),
    )

    time_end = min(
        time_frames - 1,
        int(region["stft_end"]),
    )

    block = stft[
        freq_start:freq_end + 1,
        time_start:time_end + 1,
    ]

    magnitude = torch.abs(block)

    log_magnitude = torch.log1p(magnitude)

    return (
        float(log_magnitude.mean().item()),
        float(magnitude.mean().item()),
    )


def minmax(values: List[float]) -> List[float]:
    if not values:
        return []

    lo = min(values)
    hi = max(values)

    if abs(hi - lo) < EPS:
        return [1.0 for _ in values]

    return [
        (x - lo) / (hi - lo)
        for x in values
    ]


def calculate_region_parameters(
    stft: torch.Tensor,
    regions: List[Dict],
) -> List[Dict]:

    semantic_scores = [
        region["semantic_score"]
        for region in regions
    ]

    # Lower semantic importance = greater embedding suitability.
    semantic_suitability = [
        1.0 - x
        for x in semantic_scores
    ]

    spectral_values = []

    temporary = []

    for region in regions:
        log_mean, magnitude_mean = spectral_statistics(
            stft,
            region,
        )

        spectral_values.append(log_mean)

        temporary.append(
            (
                log_mean,
                magnitude_mean,
            )
        )

    normalized_semantic = minmax(
        semantic_suitability
    )

    normalized_spectral = minmax(
        spectral_values
    )

    results = []

    for i, region in enumerate(regions):

        s = max(
            normalized_semantic[i],
            EPS,
        )

        p = max(
            normalized_spectral[i],
            EPS,
        )

        joint = (
            (s ** ALPHA)
            *
            (p ** BETA)
        )

        delta = (
            DELTA_MIN
            +
            (DELTA_MAX - DELTA_MIN)
            * joint
        )

        log_mean, magnitude_mean = temporary[i]

        item = dict(region)

        item.update(
            {
                "semantic_suitability": float(
                    normalized_semantic[i]
                ),
                "spectral_log_mean": float(
                    log_mean
                ),
                "spectral_mean": float(
                    magnitude_mean
                ),
                "spectral_suitability": float(
                    normalized_spectral[i]
                ),
                "joint_suitability": float(joint),
                "adaptive_strength": float(delta),
                "freq_start": int(
                    max(
                        1,
                        int(
                            stft.shape[0]
                            * FREQ_START_RATIO
                        ),
                    )
                ),
                "freq_end": int(
                    min(
                        stft.shape[0] - 2,
                        int(
                            stft.shape[0]
                            * FREQ_END_RATIO
                        ),
                    )
                ),
            }
        )

        results.append(item)

    return results


# ============================================================================
# CAPACITY / COORDINATE ALLOCATION
# ============================================================================

def calculate_capacity(
    regions: List[Dict],
) -> int:

    total = 0

    for region in regions:
        time_count = (
            region["stft_end"]
            - region["stft_start"]
            + 1
        )

        freq_count = (
            region["freq_end"]
            - region["freq_start"]
            + 1
        )

        total += time_count * freq_count

    return int(total)


def allocate_coordinates(
    payload: torch.Tensor,
    regions: List[Dict],
) -> List[Dict]:

    coordinates = []

    bit_index = 0

    for region in regions:

        delta = float(
            region["adaptive_strength"]
        )

        for t in range(
            region["stft_start"],
            region["stft_end"] + 1,
        ):

            for f in range(
                region["freq_start"],
                region["freq_end"] + 1,
            ):

                if bit_index >= payload.numel():
                    return coordinates

                bit = int(
                    payload[bit_index].item()
                )

                coordinates.append(
                    {
                        "bit_index": bit_index,
                        "payload_bit": bit,
                        "region_id": region["region_id"],
                        "time_frame": t,
                        "frequency_bin": f,
                        "semantic_score": region[
                            "semantic_score"
                        ],
                        "semantic_suitability": region[
                            "semantic_suitability"
                        ],
                        "spectral_suitability": region[
                            "spectral_suitability"
                        ],
                        "joint_suitability": region[
                            "joint_suitability"
                        ],
                        "adaptive_strength": delta,
                    }
                )

                bit_index += 1

    return coordinates


# ============================================================================
# EMBEDDING
# ============================================================================

def embed_payload(
    original_stft: torch.Tensor,
    payload: torch.Tensor,
    coordinates: List[Dict],
) -> Tuple[torch.Tensor, List[Dict]]:

    watermarked = original_stft.clone()

    records = []

    for coordinate in coordinates:

        bit = int(
            coordinate["payload_bit"]
        )

        f = int(
            coordinate["frequency_bin"]
        )

        t = int(
            coordinate["time_frame"]
        )

        strength = float(
            coordinate["adaptive_strength"]
        )

        original_value = watermarked[f, t]

        original_real = float(
            original_value.real.item()
        )

        # Binary bipolar embedding:
        #
        # bit 1 -> increase real component
        # bit 0 -> decrease real component
        #
        # The imaginary component remains unchanged.
        if bit == 1:
            new_real = original_real + strength
        else:
            new_real = original_real - strength

        new_value = torch.complex(
            torch.tensor(
                new_real,
                dtype=original_value.real.dtype,
            ),
            original_value.imag,
        )

        watermarked[f, t] = new_value

        actual_change = abs(
            new_real - original_real
        )

        record = dict(coordinate)

        record.update(
            {
                "original_real": original_real,
                "watermarked_real": new_real,
                "delta_change": actual_change,
            }
        )

        records.append(record)

    return watermarked, records


# ============================================================================
# QUALITY METRICS
# ============================================================================

def quality_metrics(
    original: torch.Tensor,
    watermarked: torch.Tensor,
) -> Dict[str, float]:

    original = original.float().flatten()
    watermarked = watermarked.float().flatten()

    if original.numel() != watermarked.numel():
        n = min(
            original.numel(),
            watermarked.numel(),
        )

        original = original[:n]
        watermarked = watermarked[:n]

    difference = watermarked - original

    mse = float(
        torch.mean(
            difference ** 2
        ).item()
    )

    rmse = math.sqrt(mse)

    signal_power = float(
        torch.mean(
            original ** 2
        ).item()
    )

    if mse > 0:
        snr = 10.0 * math.log10(
            signal_power / mse
        )
    else:
        snr = float("inf")

    peak_error = float(
        torch.max(
            torch.abs(difference)
        ).item()
    )

    original_rms = math.sqrt(
        signal_power
    )

    watermarked_rms = math.sqrt(
        float(
            torch.mean(
                watermarked ** 2
            ).item()
        )
    )

    numerator = torch.sum(
        (original - original.mean())
        *
        (watermarked - watermarked.mean())
    )

    denominator = torch.sqrt(
        torch.sum(
            (original - original.mean()) ** 2
        )
        *
        torch.sum(
            (watermarked - watermarked.mean()) ** 2
        )
    )

    if denominator.item() > 0:
        correlation = float(
            (numerator / denominator).item()
        )
    else:
        correlation = 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "snr_db": snr,
        "peak_absolute_error": peak_error,
        "original_rms": original_rms,
        "watermarked_rms": watermarked_rms,
        "waveform_correlation": correlation,
    }


# ============================================================================
# SAVE FUNCTIONS
# ============================================================================

def save_csv(
    path: Path,
    rows: List[Dict],
) -> None:

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def save_audio(
    path: Path,
    waveform: torch.Tensor,
    sample_rate: int,
) -> None:

    audio = (
        waveform.detach()
        .cpu()
        .flatten()
        .numpy()
    )

    sf.write(
        str(path),
        audio,
        sample_rate,
        subtype="PCM_16",
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    ensure_output_dir()

    print("=" * 64)
    print("SemBind-Audio : Experiment 19")
    print("Joint Semantic-Spectral Adaptive STFT Embedding")
    print("=" * 64)

    # ------------------------------------------------------------------
    # Load audio
    # ------------------------------------------------------------------

    original_audio, sample_rate = (
        load_original_audio()
    )

    print()
    print("Original Audio")
    print("-" * 64)
    print(
        f"Audio Path      : {ORIGINAL_AUDIO_PATH}"
    )
    print(
        f"Sample Rate     : {sample_rate}"
    )
    print(
        f"Samples         : {original_audio.numel()}"
    )
    print("Channels        : 1")
    print(
        f"Duration        : "
        f"{original_audio.numel() / sample_rate:.4f} s"
    )
    print(
        f"Original RMS     : "
        f"{torch.sqrt(torch.mean(original_audio ** 2)).item():.10f}"
    )
    print(
        f"Original Peak    : "
        f"{torch.max(torch.abs(original_audio)).item():.10f}"
    )

    # ------------------------------------------------------------------
    # STFT
    # ------------------------------------------------------------------

    original_stft = compute_stft(
        original_audio
    )

    print()
    print(
        f"STFT Shape            : "
        f"{tuple(original_stft.shape)}"
    )

    # ------------------------------------------------------------------
    # Payload
    # ------------------------------------------------------------------

    payload = load_payload()

    print(
        f"Payload File          : "
        f"{PAYLOAD_PATH}"
    )

    print(
        f"Payload Length        : "
        f"{payload.numel()} bits"
    )

    # ------------------------------------------------------------------
    # Semantic scores
    # ------------------------------------------------------------------

    semantic_rows = load_semantic_scores()

    semantic_count = (
        max(
            row["frame"]
            for row in semantic_rows
        )
        + 1
    )

    print(
        f"Semantic Score File   : "
        f"{SEMANTIC_SCORE_PATH}"
    )

    print(
        f"Semantic Score Rows   : "
        f"{len(semantic_rows)}"
    )

    selected_count = sum(
        1
        for row in semantic_rows
        if row["selected"]
    )

    print(
        f"Selected Semantic Frames : "
        f"{selected_count}"
    )

    # ------------------------------------------------------------------
    # Build semantic regions
    # ------------------------------------------------------------------

    semantic_regions = merge_selected_runs(
        semantic_rows
    )

    regions = map_regions_to_stft(
        semantic_regions,
        semantic_count=semantic_count,
        stft_count=original_stft.shape[1],
    )

    print(
        f"Candidate Regions    : "
        f"{len(regions)}"
    )

    # ------------------------------------------------------------------
    # Joint semantic + spectral parameters
    # ------------------------------------------------------------------

    regions = calculate_region_parameters(
        original_stft,
        regions,
    )

    print()
    print("=" * 64)
    print("Joint Semantic-Spectral Region Parameters")
    print("=" * 64)

    for region in regions:

        print(
            f"Region {region['region_id']:02d} | "
            f"Semantic: "
            f"{region['semantic_score']:.6f} | "
            f"Semantic Suit.: "
            f"{region['semantic_suitability']:.6f} | "
            f"Spectral Suit.: "
            f"{region['spectral_suitability']:.6f} | "
            f"Joint Suit.: "
            f"{region['joint_suitability']:.6f} | "
            f"Delta: "
            f"{region['adaptive_strength']:.8f}"
        )

    # ------------------------------------------------------------------
    # Capacity
    # ------------------------------------------------------------------

    capacity = calculate_capacity(
        regions
    )

    print()
    print(
        f"Available Capacity    : "
        f"{capacity} bits"
    )

    if capacity < payload.numel():
        raise ValueError(
            "\nInsufficient embedding capacity.\n"
            f"Required : {payload.numel()} bits\n"
            f"Available: {capacity} bits\n"
        )

    print(
        "Capacity Check        : SUFFICIENT"
    )

    # ------------------------------------------------------------------
    # Coordinate allocation
    # ------------------------------------------------------------------

    coordinates = allocate_coordinates(
        payload,
        regions,
    )

    if len(coordinates) != payload.numel():
        raise RuntimeError(
            "Coordinate allocation failed.\n"
            f"Expected : {payload.numel()}\n"
            f"Allocated: {len(coordinates)}"
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    watermarked_stft, records = embed_payload(
        original_stft,
        payload,
        coordinates,
    )

    print()
    print("Embedding Complete")
    print()
    print(
        f"Embedded Bits         : "
        f"{len(records)}"
    )
    print(
        f"Embedding Coordinates : "
        f"{len(records)}"
    )

    actual_changes = torch.tensor(
        [
            row["delta_change"]
            for row in records
        ],
        dtype=torch.float64,
    )

    if actual_changes.numel() > 0:
        print(
            f"Mean Embedded Change  : "
            f"{actual_changes.mean().item():.10f}"
        )
        print(
            f"Max Embedded Change   : "
            f"{actual_changes.max().item():.10f}"
        )

    # Global STFT magnitude difference.
    magnitude_change = (
        torch.abs(watermarked_stft)
        -
        torch.abs(original_stft)
    ).abs()

    print(
        f"Mean Magnitude Change : "
        f"{magnitude_change.mean().item():.10f}"
    )
    print(
        f"Maximum Magnitude Change: "
        f"{magnitude_change.max().item():.10f}"
    )

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    watermarked_audio = reconstruct_audio(
        watermarked_stft,
        length=original_audio.numel(),
    )

    difference_audio = (
        watermarked_audio
        -
        original_audio
    )

    # ------------------------------------------------------------------
    # Audio quality
    # ------------------------------------------------------------------

    metrics = quality_metrics(
        original_audio,
        watermarked_audio,
    )

    print()
    print("=" * 64)
    print("Audio Quality")
    print("=" * 64)

    print(
        f"MSE                  : "
        f"{metrics['mse']:.12e}"
    )
    print(
        f"RMSE                 : "
        f"{metrics['rmse']:.12e}"
    )
    print(
        f"SNR                  : "
        f"{metrics['snr_db']:.6f} dB"
    )
    print(
        f"Peak Absolute Error  : "
        f"{metrics['peak_absolute_error']:.12e}"
    )
    print(
        f"Original RMS         : "
        f"{metrics['original_rms']:.12e}"
    )
    print(
        f"Watermarked RMS      : "
        f"{metrics['watermarked_rms']:.12e}"
    )
    print(
        f"Waveform Correlation : "
        f"{metrics['waveform_correlation']:.12f}"
    )

    # ------------------------------------------------------------------
    # Save tensors
    # ------------------------------------------------------------------

    torch.save(
        original_stft.cpu(),
        ORIGINAL_STFT_PATH,
    )

    torch.save(
        watermarked_stft.cpu(),
        WATERMARKED_STFT_PATH,
    )

    torch.save(
        payload.cpu(),
        EMBEDDED_BITS_PATH,
    )

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------

    save_csv(
        COORDINATES_PATH,
        records,
    )

    save_csv(
        REGION_PARAMETERS_PATH,
        regions,
    )

    # ------------------------------------------------------------------
    # Save audio
    # ------------------------------------------------------------------

    save_audio(
        WATERMARKED_AUDIO_PATH,
        watermarked_audio,
        sample_rate,
    )

    save_audio(
        DIFFERENCE_AUDIO_PATH,
        difference_audio,
        sample_rate,
    )

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------

    summary = {
        "experiment": "exp19_joint_semantic_spectral_embedding",
        "description": (
            "Joint semantic-spectral adaptive STFT "
            "watermark embedding"
        ),
        "audio_path": str(
            ORIGINAL_AUDIO_PATH
        ),
        "sample_rate": sample_rate,
        "audio_samples": int(
            original_audio.numel()
        ),
        "stft_shape": [
            int(original_stft.shape[0]),
            int(original_stft.shape[1]),
        ],
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "win_length": WIN_LENGTH,
        "payload_bits": int(
            payload.numel()
        ),
        "semantic_score_rows": len(
            semantic_rows
        ),
        "selected_semantic_frames": selected_count,
        "candidate_regions": len(
            regions
        ),
        "available_capacity_bits": capacity,
        "embedded_bits": len(records),
        "alpha": ALPHA,
        "beta": BETA,
        "delta_min": DELTA_MIN,
        "delta_max": DELTA_MAX,
        "frequency_start_ratio": FREQ_START_RATIO,
        "frequency_end_ratio": FREQ_END_RATIO,
        "metrics": metrics,
        "mean_embedded_change": float(
            actual_changes.mean().item()
        ),
        "max_embedded_change": float(
            actual_changes.max().item()
        ),
        "mean_global_magnitude_change": float(
            magnitude_change.mean().item()
        ),
        "max_global_magnitude_change": float(
            magnitude_change.max().item()
        ),
        "output_files": {
            "original_stft": str(
                ORIGINAL_STFT_PATH
            ),
            "watermarked_stft": str(
                WATERMARKED_STFT_PATH
            ),
            "embedded_bits": str(
                EMBEDDED_BITS_PATH
            ),
            "coordinates": str(
                COORDINATES_PATH
            ),
            "region_parameters": str(
                REGION_PARAMETERS_PATH
            ),
            "watermarked_audio": str(
                WATERMARKED_AUDIO_PATH
            ),
            "difference_audio": str(
                DIFFERENCE_AUDIO_PATH
            ),
        },
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("Results Saved")
    print(ORIGINAL_STFT_PATH)
    print(WATERMARKED_STFT_PATH)
    print(EMBEDDED_BITS_PATH)
    print(COORDINATES_PATH)
    print(REGION_PARAMETERS_PATH)
    print(WATERMARKED_AUDIO_PATH)
    print(DIFFERENCE_AUDIO_PATH)
    print(SUMMARY_PATH)

    print()
    print("=" * 64)
    print("Experiment 19 Complete")
    print("=" * 64)


if __name__ == "__main__":
    main()