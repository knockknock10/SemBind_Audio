"""
SemBind-Audio
Experiment 17: Semantic-Adaptive QIM Watermark Embedding

Purpose
-------
Replace the fragile additive watermarking rule

    Xw = X +/- alpha

with a Quantization Index Modulation (QIM) scheme whose
quantization step is adapted according to semantic importance.

Core idea
---------
Low semantic importance
        -> higher suitability
        -> larger QIM step
        -> stronger / more robust watermark

Higher semantic importance
        -> lower suitability
        -> smaller QIM step
        -> lower perceptual impact

This experiment:
    1. Loads the original STFT.
    2. Loads the 256-bit cryptographic payload.
    3. Loads the semantic STFT regions.
    4. Loads semantic scores.
    5. Computes a semantic-adaptive QIM step for every region.
    6. Embeds all 256 payload bits.
    7. Reconstructs the watermarked audio using iSTFT.
    8. Computes audio-quality metrics.
    9. Saves all artifacts without modifying previous experiments.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import librosa


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIG
# ============================================================

from configs.config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
)


# ============================================================
# INPUT FILES
# ============================================================

STFT_FILE = (
    PROJECT_ROOT
    / "results"
    / "spectrograms"
    / "stft.pt"
)

PAYLOAD_BITS_FILE = (
    PROJECT_ROOT
    / "results"
    / "watermark"
    / "payload_bits.pt"
)

REGION_MAPPING_FILE = (
    PROJECT_ROOT
    / "results"
    / "stft_mapping"
    / "stft_region_mapping.csv"
)

SEMANTIC_REGION_FILE = (
    PROJECT_ROOT
    / "results"
    / "spectrogram_mapping"
    / "mapped_regions.csv"
)

SEMANTIC_SCORE_FILE = (
    PROJECT_ROOT
    / "results"
    / "region_selection"
    / "semantic_scores.csv"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "qim_embedding"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# QIM PARAMETERS
# ============================================================

from configs.config import (
    DELTA_MIN,
    DELTA_MAX,
)

# Frequency bins previously considered usable.
# We preserve the same exclusion used by Exp08.
FREQUENCY_MARGIN = 4


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def find_column(
    fieldnames: List[str],
    candidates: List[str],
) -> str:
    """
    Find a column using several possible names.

    This makes the experiment robust to small naming
    differences between previous experiment outputs.
    """

    normalized = {
        name.strip().lower(): name
        for name in fieldnames
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    raise ValueError(
        "Could not find required column.\n"
        f"Available columns: {fieldnames}\n"
        f"Required candidates: {candidates}"
    )


def load_csv(path: Path) -> List[Dict[str, str]]:
    """Load CSV into a list of dictionaries."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

    if not rows:
        raise ValueError(
            f"CSV file contains no rows:\n{path}"
        )

    return rows


def load_payload(path: Path) -> torch.Tensor:
    """
    Load payload bits and normalize to a 1-D uint8 tensor.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Payload file not found:\n{path}"
        )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(payload, torch.Tensor):
        payload = torch.tensor(payload)

    payload = payload.detach().cpu().flatten()

    payload = payload.to(torch.uint8)

    unique_values = set(
        int(x)
        for x in payload.tolist()
    )

    if not unique_values.issubset({0, 1}):
        raise ValueError(
            "Payload contains values other than 0 and 1:\n"
            f"{unique_values}"
        )

    if payload.numel() == 0:
        raise ValueError(
            "Payload contains zero bits."
        )

    return payload


def normalize_scores(
    scores: np.ndarray,
) -> np.ndarray:
    """
    Normalize semantic scores to [0, 1].

    Lower score means lower semantic importance.
    """

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    minimum = float(scores.min())
    maximum = float(scores.max())

    if maximum - minimum < 1e-12:
        return np.zeros_like(scores)

    return (
        (scores - minimum)
        / (maximum - minimum)
    )


def semantic_delta(
    normalized_score: float,
) -> float:
    """
    Compute semantic-adaptive QIM step.

    Delta = Delta_min +
            (Delta_max - Delta_min) * (1 - semantic_score)

    Therefore:

        score -> 0
        Delta -> Delta_max

        score -> 1
        Delta -> Delta_min
    """

    score = float(
        np.clip(
            normalized_score,
            0.0,
            1.0,
        )
    )

    delta = (
        DELTA_MIN
        + (
            DELTA_MAX
            - DELTA_MIN
        )
        * (
            1.0
            - score
        )
    )

    return float(delta)


def qim_embed(
    value: float,
    bit: int,
    delta: float,
) -> float:
    """
    QIM embedding.

    Bit 0:
        integer quantization state

    Bit 1:
        half-step shifted quantization state
    """

    if delta <= 0:
        raise ValueError(
            "QIM delta must be positive."
        )

    q = round(
        value / delta
    )

    if bit == 0:

        return (
            delta
            * q
        )

    if bit == 1:

        return (
            delta
            * (
                q
                + 0.5
            )
        )

    raise ValueError(
        f"Invalid watermark bit: {bit}"
    )


def qim_extract(
    value: float,
    delta: float,
) -> int:
    """
    QIM extraction rule.

    Determine whether the received coefficient is
    closer to the 0-state or the 1-state.
    """

    if delta <= 0:
        raise ValueError(
            "QIM delta must be positive."
        )

    normalized = (
        value / delta
    )

    fractional = (
        normalized
        - math.floor(normalized)
    )

    distance_to_zero = min(
        fractional,
        1.0 - fractional,
    )

    distance_to_one = abs(
        fractional - 0.5
    )

    if distance_to_one < distance_to_zero:
        return 1

    return 0


def select_coordinates(
    regions: List[Dict[str, int]],
    payload_length: int,
) -> List[Tuple[int, int, int]]:
    """
    Select deterministic STFT coordinates.

    Returns:
        (region_id, time_frame, frequency_bin)
    """

    coordinates = []

    for region in regions:

        region_id = int(
            region["region_id"]
        )

        time_start = int(
            region["time_start_frame"]
        )

        time_end = int(
            region["time_end_frame"]
        )

        frequency_start = int(
            region["frequency_start_bin"]
        )

        frequency_end = int(
            region["frequency_end_bin"]
        )

        for time_frame in range(
            time_start,
            time_end + 1,
        ):

            for frequency_bin in range(
                frequency_start,
                frequency_end + 1,
            ):

                coordinates.append(
                    (
                        region_id,
                        time_frame,
                        frequency_bin,
                    )
                )

                if len(coordinates) >= payload_length:
                    return coordinates

    return coordinates


def audio_metrics(
    original: np.ndarray,
    watermarked: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate waveform quality metrics.
    """

    length = min(
        len(original),
        len(watermarked),
    )

    original = original[:length]
    watermarked = watermarked[:length]

    error = (
        watermarked
        - original
    )

    mse = float(
        np.mean(
            error ** 2
        )
    )

    rmse = float(
        np.sqrt(mse)
    )

    signal_power = float(
        np.mean(
            original ** 2
        )
    )

    if mse > 0:
        snr = float(
            10.0
            * np.log10(
                signal_power
                / mse
            )
        )
    else:
        snr = float("inf")

    peak_error = float(
        np.max(
            np.abs(error)
        )
    )

    original_rms = float(
        np.sqrt(
            np.mean(
                original ** 2
            )
        )
    )

    watermarked_rms = float(
        np.sqrt(
            np.mean(
                watermarked ** 2
            )
        )
    )

    if (
        np.std(original) > 0
        and np.std(watermarked) > 0
    ):

        correlation = float(
            np.corrcoef(
                original,
                watermarked,
            )[0, 1]
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 64)
    print(
        "SemBind-Audio : Experiment 17"
    )
    print(
        "Semantic-Adaptive QIM Watermark Embedding"
    )
    print("=" * 64)

    # --------------------------------------------------------
    # Load STFT
    # --------------------------------------------------------

    if not STFT_FILE.exists():
        raise FileNotFoundError(
            f"STFT file not found:\n{STFT_FILE}"
        )

    original_stft = torch.load(
        STFT_FILE,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(
        original_stft,
        torch.Tensor,
    ):
        original_stft = torch.tensor(
            original_stft
        )

    original_stft = (
        original_stft
        .detach()
        .cpu()
    )

    if not torch.is_complex(
        original_stft
    ):
        raise ValueError(
            "Expected complex STFT tensor."
        )

    print()
    print(
        f"STFT Shape          : "
        f"{tuple(original_stft.shape)}"
    )

    frequency_bins = (
        original_stft.shape[0]
    )

    time_frames = (
        original_stft.shape[1]
    )

    print(
        f"Frequency Bins      : "
        f"{frequency_bins}"
    )

    print(
        f"Time Frames         : "
        f"{time_frames}"
    )

    # --------------------------------------------------------
    # Load payload
    # --------------------------------------------------------

    payload = load_payload(
        PAYLOAD_BITS_FILE
    )

    payload_length = int(
        payload.numel()
    )

    print(
        f"Payload Length      : "
        f"{payload_length} bits"
    )

    # --------------------------------------------------------
    # Load semantic score file
    # --------------------------------------------------------

    semantic_rows = load_csv(
        SEMANTIC_SCORE_FILE
    )

    fieldnames = list(
        semantic_rows[0].keys()
    )

    frame_column = find_column(
        fieldnames,
        [
            "frame",
            "frame_id",
        ],
    )

    score_column = find_column(
        fieldnames,
        [
            "normalized_score",
            "semantic_score",
            "score",
        ],
    )

    semantic_score_by_frame = {}

    for row in semantic_rows:

        frame = int(
            float(
                row[frame_column]
            )
        )

        score = float(
            row[score_column]
        )

        semantic_score_by_frame[
            frame
        ] = score

    print(
        f"Semantic Score Rows : "
        f"{len(semantic_score_by_frame)}"
    )

    # --------------------------------------------------------
    # Load STFT region mapping
    # --------------------------------------------------------

    region_rows = load_csv(
        REGION_MAPPING_FILE
    )

    region_fields = list(
        region_rows[0].keys()
    )

    region_id_column = find_column(
        region_fields,
        ["region_id"],
    )

    time_start_column = find_column(
        region_fields,
        [
            "time_start_frame",
            "start_frame",
        ],
    )

    time_end_column = find_column(
        region_fields,
        [
            "time_end_frame",
            "end_frame",
        ],
    )

    frequency_start_column = find_column(
        region_fields,
        [
            "frequency_start_bin",
            "freq_start_bin",
        ],
    )

    frequency_end_column = find_column(
        region_fields,
        [
            "frequency_end_bin",
            "freq_end_bin",
        ],
    )

    regions = []

    for row in region_rows:

        regions.append(
            {
                "region_id": int(
                    float(
                        row[
                            region_id_column
                        ]
                    )
                ),

                "time_start_frame": int(
                    float(
                        row[
                            time_start_column
                        ]
                    )
                ),

                "time_end_frame": int(
                    float(
                        row[
                            time_end_column
                        ]
                    )
                ),

                "frequency_start_bin": int(
                    float(
                        row[
                            frequency_start_column
                        ]
                    )
                ),

                "frequency_end_bin": int(
                    float(
                        row[
                            frequency_end_column
                        ]
                    )
                ),
            }
        )

    print(
        f"Candidate Regions   : "
        f"{len(regions)}"
    )

    # --------------------------------------------------------
    # Calculate semantic score for each region
    # --------------------------------------------------------

    region_scores = []

    for region in regions:

        frames = range(
            region[
                "time_start_frame"
            ],
            region[
                "time_end_frame"
            ] + 1,
        )

        scores = []

        for frame in frames:

            if frame in semantic_score_by_frame:

                scores.append(
                    semantic_score_by_frame[
                        frame
                    ]
                )

        if not scores:
            raise ValueError(
                "No semantic score found "
                f"for region {region['region_id']}."
            )

        mean_score = float(
            np.mean(scores)
        )

        region_scores.append(
            {
                "region_id":
                    region["region_id"],

                "semantic_score":
                    mean_score,
            }
        )

    # --------------------------------------------------------
    # Normalize region scores
    # --------------------------------------------------------

    raw_scores = np.array(
        [
            row["semantic_score"]
            for row in region_scores
        ],
        dtype=np.float64,
    )

    normalized_region_scores = (
        normalize_scores(
            raw_scores
        )
    )

    # --------------------------------------------------------
    # Calculate adaptive QIM steps
    # --------------------------------------------------------

    region_delta_lookup = {}

    print()
    print(
        "Semantic-Adaptive QIM Parameters"
    )
    print("-" * 64)

    for index, row in enumerate(
        region_scores
    ):

        region_id = row[
            "region_id"
        ]

        semantic_score = row[
            "semantic_score"
        ]

        normalized_score = float(
            normalized_region_scores[
                index
            ]
        )

        suitability = (
            1.0
            - normalized_score
        )

        delta = semantic_delta(
            normalized_score
        )

        region_delta_lookup[
            region_id
        ] = delta

        print(
            f"Region {region_id:02d} | "
            f"Semantic Score: "
            f"{semantic_score:.6f} | "
            f"Suitability: "
            f"{suitability:.6f} | "
            f"QIM Delta: "
            f"{delta:.8f}"
        )

    # --------------------------------------------------------
    # Determine usable frequency range
    # --------------------------------------------------------

    frequency_start = (
        FREQUENCY_MARGIN
    )

    frequency_end = (
        frequency_bins
        - FREQUENCY_MARGIN
        - 1
    )

    # Update regions so coordinates remain valid.
    valid_regions = []

    for region in regions:

        time_start = max(
            0,
            region[
                "time_start_frame"
            ],
        )

        time_end = min(
            time_frames - 1,
            region[
                "time_end_frame"
            ],
        )

        freq_start = max(
            frequency_start,
            region[
                "frequency_start_bin"
            ],
        )

        freq_end = min(
            frequency_end,
            region[
                "frequency_end_bin"
            ],
        )

        if (
            time_start > time_end
            or freq_start > freq_end
        ):
            continue

        valid_region = dict(
            region
        )

        valid_region[
            "time_start_frame"
        ] = time_start

        valid_region[
            "time_end_frame"
        ] = time_end

        valid_region[
            "frequency_start_bin"
        ] = freq_start

        valid_region[
            "frequency_end_bin"
        ] = freq_end

        valid_regions.append(
            valid_region
        )

    coordinates = select_coordinates(
        valid_regions,
        payload_length,
    )

    available_capacity = sum(
        (
            region[
                "time_end_frame"
            ]
            - region[
                "time_start_frame"
            ]
            + 1
        )
        *
        (
            region[
                "frequency_end_bin"
            ]
            - region[
                "frequency_start_bin"
            ]
            + 1
        )
        for region in valid_regions
    )

    print()
    print(
        f"Available Capacity  : "
        f"{available_capacity} bits"
    )

    print(
        "Capacity Check      : "
        + (
            "SUFFICIENT"
            if available_capacity
            >= payload_length
            else "INSUFFICIENT"
        )
    )

    if available_capacity < payload_length:

        raise ValueError(
            "Insufficient STFT embedding capacity."
        )

    # --------------------------------------------------------
    # Embed watermark
    # --------------------------------------------------------

    watermarked_stft = (
        original_stft.clone()
    )

    embedded_records = []

    magnitude_changes = []

    embedded_changes = []

    for bit_index, (
        region_id,
        time_frame,
        frequency_bin,
    ) in enumerate(
        coordinates
    ):

        bit = int(
            payload[
                bit_index
            ].item()
        )

        original_value = (
            watermarked_stft[
                frequency_bin,
                time_frame,
            ]
        )

        real_value = float(
            original_value.real.item()
        )

        delta = (
            region_delta_lookup[
                region_id
            ]
        )

        watermarked_real = qim_embed(
            real_value,
            bit,
            delta,
        )

        new_value = torch.complex(
            torch.tensor(
                watermarked_real,
                dtype=original_value.real.dtype,
            ),
            original_value.imag,
        )

        watermarked_stft[
            frequency_bin,
            time_frame,
        ] = new_value

        change = abs(
            watermarked_real
            - real_value
        )

        magnitude_changes.append(
            abs(
                new_value.item()
                - original_value.item()
            )
        )

        embedded_changes.append(
            change
        )

        embedded_records.append(
            {
                "bit_index":
                    bit_index,

                "payload_bit":
                    bit,

                "region_id":
                    region_id,

                "time_frame":
                    time_frame,

                "frequency_bin":
                    frequency_bin,

                "semantic_score":
                    region_scores[
                        region_id - 1
                    ][
                        "semantic_score"
                    ],

                "adaptive_delta":
                    delta,

                "original_real":
                    real_value,

                "watermarked_real":
                    watermarked_real,

                "delta_change":
                    change,
            }
        )

    # --------------------------------------------------------
    # Reconstruct audio
    # --------------------------------------------------------

    watermarked_audio = (
        librosa.istft(
            watermarked_stft.numpy(),
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_fft=N_FFT,
            length=None,
        )
    )

    # --------------------------------------------------------
    # Load original audio
    # --------------------------------------------------------

    audio_candidates = [
        PROJECT_ROOT
        / "results"
        / "audio_quality"
        / "stft_reconstructed_audio.wav",

        PROJECT_ROOT
        / "results"
        / "watermark_embedding"
        / "watermarked_audio.wav",
    ]

    original_audio = None

    for candidate in audio_candidates:

        if candidate.exists():

            loaded, sr = sf.read(
                candidate,
                dtype="float32",
            )

            if loaded.ndim > 1:
                loaded = np.mean(
                    loaded,
                    axis=1,
                )

            if sr != SAMPLE_RATE:

                loaded = librosa.resample(
                    loaded.astype(
                        np.float32
                    ),
                    orig_sr=sr,
                    target_sr=SAMPLE_RATE,
                )

            original_audio = loaded

            break

    if original_audio is None:

        raise FileNotFoundError(
            "Could not locate a reference audio file."
        )

    # --------------------------------------------------------
    # Align lengths
    # --------------------------------------------------------

    common_length = min(
        len(original_audio),
        len(watermarked_audio),
    )

    original_audio = (
        original_audio[
            :common_length
        ]
    )

    watermarked_audio = (
        watermarked_audio[
            :common_length
        ]
    )

    # --------------------------------------------------------
    # Audio quality
    # --------------------------------------------------------

    metrics = audio_metrics(
        original_audio,
        watermarked_audio,
    )

    # --------------------------------------------------------
    # Save STFT
    # --------------------------------------------------------

    torch.save(
        original_stft,
        OUTPUT_DIR
        / "original_stft.pt",
    )

    torch.save(
        watermarked_stft,
        OUTPUT_DIR
        / "qim_watermarked_stft.pt",
    )

    torch.save(
        payload,
        OUTPUT_DIR
        / "embedded_bits.pt",
    )

    # --------------------------------------------------------
    # Save coordinates
    # --------------------------------------------------------

    coordinates_file = (
        OUTPUT_DIR
        / "qim_embedding_coordinates.csv"
    )

    coordinate_fields = [
        "bit_index",
        "payload_bit",
        "region_id",
        "time_frame",
        "frequency_bin",
        "semantic_score",
        "adaptive_delta",
        "original_real",
        "watermarked_real",
        "delta_change",
    ]

    with coordinates_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=coordinate_fields,
        )

        writer.writeheader()

        writer.writerows(
            embedded_records
        )

    # --------------------------------------------------------
    # Save region weights
    # --------------------------------------------------------

    region_file = (
        OUTPUT_DIR
        / "semantic_qim_region_parameters.csv"
    )

    region_fields = [
        "region_id",
        "semantic_score",
        "normalized_semantic_score",
        "semantic_suitability",
        "adaptive_delta",
    ]

    with region_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=region_fields,
        )

        writer.writeheader()

        for index, row in enumerate(
            region_scores
        ):

            normalized = float(
                normalized_region_scores[
                    index
                ]
            )

            writer.writerow(
                {
                    "region_id":
                        row["region_id"],

                    "semantic_score":
                        row[
                            "semantic_score"
                        ],

                    "normalized_semantic_score":
                        normalized,

                    "semantic_suitability":
                        1.0 - normalized,

                    "adaptive_delta":
                        region_delta_lookup[
                            row["region_id"]
                        ],
                }
            )

    # --------------------------------------------------------
    # Save watermarked audio
    # --------------------------------------------------------

    watermarked_audio_path = (
        OUTPUT_DIR
        / "qim_watermarked_audio.wav"
    )

    sf.write(
        watermarked_audio_path,
        watermarked_audio,
        SAMPLE_RATE,
        subtype="PCM_32",
    )

    # --------------------------------------------------------
    # Save difference audio
    # --------------------------------------------------------

    difference_audio = (
        watermarked_audio
        - original_audio
    )

    difference_path = (
        OUTPUT_DIR
        / "qim_difference.wav"
    )

    sf.write(
        difference_path,
        difference_audio,
        SAMPLE_RATE,
        subtype="PCM_32",
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    magnitude_changes_np = np.asarray(
        magnitude_changes,
        dtype=np.float64,
    )

    embedded_changes_np = np.asarray(
        embedded_changes,
        dtype=np.float64,
    )

    summary = {

        "experiment":
            "Experiment 17",

        "method":
            "Semantic-Adaptive QIM",

        "sample_rate":
            SAMPLE_RATE,

        "n_fft":
            N_FFT,

        "hop_length":
            HOP_LENGTH,

        "win_length":
            WIN_LENGTH,

        "stft_shape":
            list(
                original_stft.shape
            ),

        "payload_length_bits":
            payload_length,

        "candidate_regions":
            len(regions),

        "available_capacity_bits":
            available_capacity,

        "delta_min":
            DELTA_MIN,

        "delta_max":
            DELTA_MAX,

        "embedded_bits":
            len(
                embedded_records
            ),

        "embedding_coordinates":
            len(
                embedded_records
            ),

        "mean_magnitude_change":
            float(
                magnitude_changes_np.mean()
            ),

        "maximum_magnitude_change":
            float(
                magnitude_changes_np.max()
            ),

        "mean_embedded_change":
            float(
                embedded_changes_np.mean()
            ),

        "maximum_embedded_change":
            float(
                embedded_changes_np.max()
            ),

        "audio_quality":
            metrics,

        "formula":
            (
                "Delta_i = Delta_min + "
                "(Delta_max - Delta_min) "
                "* (1 - normalized_semantic_score)"
            ),

        "qim_bit_zero":
            "Delta * round(x / Delta)",

        "qim_bit_one":
            "Delta * (round(x / Delta) + 0.5)",
    }

    summary_file = (
        OUTPUT_DIR
        / "qim_embedding_summary.json"
    )

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print(
        "Embedding Complete"
    )

    print()
    print(
        f"Embedded Bits           : "
        f"{len(embedded_records)}"
    )

    print(
        f"Embedding Coordinates   : "
        f"{len(embedded_records)}"
    )

    print(
        f"Mean Magnitude Change  : "
        f"{magnitude_changes_np.mean():.10f}"
    )

    print(
        f"Maximum Magnitude Change: "
        f"{magnitude_changes_np.max():.10f}"
    )

    print(
        f"Mean Embedded Change   : "
        f"{embedded_changes_np.mean():.10f}"
    )

    print(
        f"Max Embedded Change    : "
        f"{embedded_changes_np.max():.10f}"
    )

    print()
    print(
        "## Audio Quality"
    )

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

    print()
    print(
        "Results Saved"
    )

    print(
        OUTPUT_DIR
        / "original_stft.pt"
    )

    print(
        OUTPUT_DIR
        / "qim_watermarked_stft.pt"
    )

    print(
        OUTPUT_DIR
        / "embedded_bits.pt"
    )

    print(
        coordinates_file
    )

    print(
        region_file
    )

    print(
        watermarked_audio_path
    )

    print(
        difference_path
    )

    print(
        summary_file
    )


if __name__ == "__main__":
    main()