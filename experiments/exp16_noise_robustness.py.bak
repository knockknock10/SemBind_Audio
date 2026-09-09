from __future__ import annotations

"""
SemBind-Audio
Experiment 16: Additive Noise Robustness
=========================================

UPGRADED VERSION

This experiment evaluates the CURRENT SemBind-Audio watermark:

    Exp21:
        Joint semantic-spectral adaptive embedding
        + 3x redundant coordinates

    Exp20:
        Reference-based extraction
        + replica-level extraction
        + majority voting

Attack:
    Additive White Gaussian Noise (AWGN)

Attack levels:
    40 dB
    30 dB
    20 dB
    10 dB

For every payload bit:

    replica 0
    replica 1
    replica 2
          |
          v
    majority vote
          |
          v
    recovered payload bit

The primary robustness metric is therefore:

    MAJORITY ACCURACY
    MAJORITY BER

Replica accuracy is reported as a secondary diagnostic.

IMPORTANT
---------
This is a non-blind/reference-based extraction experiment.

The original clean STFT is required during extraction.
No new semantic regions are selected after the attack.
The exact Exp21 coordinates are reused.
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch


# ============================================================================
# PROJECT
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent

RESULTS = ROOT / "results"

SAMPLE_RATE = 16000

N_FFT = 1024
HOP_LENGTH = 320
WIN_LENGTH = 1024

WINDOW = torch.hann_window(
    WIN_LENGTH
)

REDUNDANCY = 3

RANDOM_SEED = 42

NOISE_SNR_LEVELS = [
    40.0,
    30.0,
    20.0,
    10.0,
]


# ============================================================================
# CURRENT SEMBIND-AUDIO FILES
# ============================================================================

PAYLOAD_FILE = (
    RESULTS
    / "watermark"
    / "payload_bits.pt"
)

REFERENCE_STFT_FILE = (
    RESULTS
    / "redundant_embedding"
    / "original_stft.pt"
)

WATERMARKED_STFT_FILE = (
    RESULTS
    / "redundant_embedding"
    / "redundant_watermarked_stft.pt"
)

COORDINATES_FILE = (
    RESULTS
    / "redundant_embedding"
    / "redundant_embedding_coordinates.csv"
)

WATERMARKED_AUDIO_FILE = (
    RESULTS
    / "redundant_embedding"
    / "redundant_watermarked_audio.wav"
)

OUTPUT_DIR = (
    RESULTS
    / "noise_robustness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# UTILITIES
# ============================================================================

def require(
    path: Path,
    label: str,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired {label} not found:\n"
            f"{path}\n"
        )


# ============================================================================
# LOAD PAYLOAD
# ============================================================================

def load_payload() -> np.ndarray:

    require(
        PAYLOAD_FILE,
        "payload file",
    )

    payload = torch.load(
        PAYLOAD_FILE,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        payload,
        torch.Tensor,
    ):

        payload = torch.tensor(
            payload
        )

    payload = (
        payload
        .detach()
        .cpu()
        .flatten()
        .to(torch.int64)
        .numpy()
    )

    payload = (
        payload != 0
    ).astype(
        np.int64
    )

    if len(payload) == 0:

        raise ValueError(
            "Payload is empty."
        )

    return payload


# ============================================================================
# LOAD STFT
# ============================================================================

def load_stft(
    path: Path,
    label: str,
) -> torch.Tensor:

    require(
        path,
        label,
    )

    stft = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        stft,
        torch.Tensor,
    ):

        stft = torch.tensor(
            stft
        )

    if not torch.is_complex(
        stft
    ):

        raise TypeError(
            f"{label} must be a complex STFT.\n"
            f"Path: {path}"
        )

    return (
        stft
        .detach()
        .cpu()
    )


# ============================================================================
# LOAD AUDIO
# ============================================================================

def load_audio(
    path: Path,
) -> Tuple[np.ndarray, int]:

    require(
        path,
        "watermarked audio",
    )

    audio, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.ndim == 2:

        audio = np.mean(
            audio,
            axis=1,
        )

    audio = audio.flatten()

    return (
        audio,
        int(sr),
    )


# ============================================================================
# STFT
# ============================================================================

def generate_stft(
    audio: np.ndarray,
) -> torch.Tensor:

    waveform = torch.from_numpy(
        np.asarray(
            audio,
            dtype=np.float32,
        )
    ).float()

    return torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        center=True,
        return_complex=True,
    )


# ============================================================================
# LOAD COORDINATES
# ============================================================================

def load_coordinates() -> List[Dict]:

    require(
        COORDINATES_FILE,
        "Exp21 redundant coordinate CSV",
    )

    with COORDINATES_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        raw = list(
            csv.DictReader(f)
        )

    if not raw:

        raise ValueError(
            "Coordinate CSV is empty."
        )

    required_columns = {
        "bit_index",
        "replica_index",
        "payload_bit",
        "region_id",
        "time_frame",
        "frequency_bin",
        "magnitude",
        "normalized_magnitude",
        "semantic_score",
        "semantic_suitability",
        "spectral_suitability",
        "joint_suitability",
        "adaptive_strength",
        "original_real",
        "watermarked_real",
        "delta_change",
    }

    missing = sorted(
        required_columns
        -
        set(raw[0].keys())
    )

    if missing:

        raise ValueError(
            "Coordinate CSV is missing "
            f"columns:\n{missing}"
        )

    rows = []

    for row in raw:

        rows.append(
            {
                "bit_index":
                    int(
                        float(
                            row[
                                "bit_index"
                            ]
                        )
                    ),

                "replica_index":
                    int(
                        float(
                            row[
                                "replica_index"
                            ]
                        )
                    ),

                "payload_bit":
                    int(
                        float(
                            row[
                                "payload_bit"
                            ]
                        )
                    ),

                "region_id":
                    int(
                        float(
                            row[
                                "region_id"
                            ]
                        )
                    ),

                "time_frame":
                    int(
                        float(
                            row[
                                "time_frame"
                            ]
                        )
                    ),

                "frequency_bin":
                    int(
                        float(
                            row[
                                "frequency_bin"
                            ]
                        )
                    ),

                "magnitude":
                    float(
                        row[
                            "magnitude"
                        ]
                    ),

                "normalized_magnitude":
                    float(
                        row[
                            "normalized_magnitude"
                        ]
                    ),

                "semantic_score":
                    float(
                        row[
                            "semantic_score"
                        ]
                    ),

                "semantic_suitability":
                    float(
                        row[
                            "semantic_suitability"
                        ]
                    ),

                "spectral_suitability":
                    float(
                        row[
                            "spectral_suitability"
                        ]
                    ),

                "joint_suitability":
                    float(
                        row[
                            "joint_suitability"
                        ]
                    ),

                "adaptive_strength":
                    float(
                        row[
                            "adaptive_strength"
                        ]
                    ),

                "original_real":
                    float(
                        row[
                            "original_real"
                        ]
                    ),

                "watermarked_real":
                    float(
                        row[
                            "watermarked_real"
                        ]
                    ),

                "delta_change":
                    float(
                        row[
                            "delta_change"
                        ]
                    ),
            }
        )

    rows.sort(
        key=lambda x: (
            x["bit_index"],
            x["replica_index"],
        )
    )

    return rows


# ============================================================================
# VALIDATE MAPPING
# ============================================================================

def validate_mapping(
    payload: np.ndarray,
    rows: List[Dict],
) -> None:

    expected_coordinates = (
        len(payload)
        *
        REDUNDANCY
    )

    print()
    print(
        "Payload / Mapping Validation"
    )
    print("-" * 72)

    print(
        f"Payload Length        : "
        f"{len(payload)} bits"
    )

    print(
        f"Redundancy            : "
        f"{REDUNDANCY}x"
    )

    print(
        f"Expected Coordinates  : "
        f"{expected_coordinates}"
    )

    print(
        f"Coordinate Rows       : "
        f"{len(rows)}"
    )

    if len(rows) != expected_coordinates:

        raise ValueError(
            "\nExp21 coordinate count "
            "does not match payload redundancy.\n"
            f"Expected : {expected_coordinates}\n"
            f"Found    : {len(rows)}"
        )

    coordinates = [
        (
            row["time_frame"],
            row["frequency_bin"],
        )
        for row in rows
    ]

    if len(
        set(coordinates)
    ) != len(coordinates):

        raise ValueError(
            "Duplicate STFT coordinates "
            "found in Exp21 mapping."
        )

    groups = {}

    for row in rows:

        groups.setdefault(
            row["bit_index"],
            [],
        ).append(
            row
        )

    expected_indices = list(
        range(
            len(payload)
        )
    )

    if sorted(
        groups.keys()
    ) != expected_indices:

        raise ValueError(
            "Bit-index coverage is not "
            "exactly 0..payload_length-1."
        )

    for bit_index in expected_indices:

        group = sorted(
            groups[bit_index],
            key=lambda x:
                x["replica_index"],
        )

        replica_indices = [
            row["replica_index"]
            for row in group
        ]

        if replica_indices != [
            0,
            1,
            2,
        ]:

            raise ValueError(
                f"Bit {bit_index} does not "
                "contain replicas 0,1,2."
            )

        expected_bit = int(
            payload[
                bit_index
            ]
        )

        for row in group:

            if (
                row["payload_bit"]
                !=
                expected_bit
            ):

                raise ValueError(
                    f"Payload mismatch "
                    f"at bit {bit_index}."
                )

    print(
        "Payload Mapping       : PASSED"
    )


# ============================================================================
# EXTRACT REPLICAS
# ============================================================================

def extract_replicas(
    reference_stft: torch.Tensor,
    observed_stft: torch.Tensor,
    rows: List[Dict],
) -> List[Dict]:

    if (
        reference_stft.shape
        !=
        observed_stft.shape
    ):

        raise ValueError(
            "Reference and observed "
            "STFT shapes differ.\n"
            f"Reference: "
            f"{tuple(reference_stft.shape)}\n"
            f"Observed : "
            f"{tuple(observed_stft.shape)}"
        )

    extracted = []

    for row in rows:

        f = row[
            "frequency_bin"
        ]

        t = row[
            "time_frame"
        ]

        if not (
            0 <= f
            <
            reference_stft.shape[0]
        ):

            raise IndexError(
                f"Frequency bin {f} "
                "outside STFT."
            )

        if not (
            0 <= t
            <
            reference_stft.shape[1]
        ):

            raise IndexError(
                f"Time frame {t} "
                "outside STFT."
            )

        reference_real = float(
            reference_stft[
                f,
                t
            ].real.item()
        )

        observed_real = float(
            observed_stft[
                f,
                t
            ].real.item()
        )

        change = (
            observed_real
            -
            reference_real
        )

        # Exp21 encoding:
        #
        # bit 1 -> positive displacement
        # bit 0 -> negative displacement
        #
        # Zero is technically ambiguous.
        # We count it as 0 for deterministic
        # extraction.

        extracted_bit = int(
            change >= 0.0
        )

        delta = float(
            row[
                "adaptive_strength"
            ]
        )

        normalized_displacement = (
            change
            /
            delta
            if abs(delta) > 1e-12
            else 0.0
        )

        extracted.append(
            {
                **row,

                "reference_stft_real":
                    reference_real,

                "observed_stft_real":
                    observed_real,

                "observed_change":
                    change,

                "normalized_displacement":
                    normalized_displacement,

                "extracted_bit":
                    extracted_bit,

                "correct":
                    int(
                        extracted_bit
                        ==
                        row[
                            "payload_bit"
                        ]
                    ),

                "reference_csv_error":
                    abs(
                        reference_real
                        -
                        row[
                            "original_real"
                        ]
                    ),
            }
        )

    return extracted


# ============================================================================
# MAJORITY VOTE
# ============================================================================

def majority_vote(
    group: List[Dict],
) -> Tuple[int, float, str]:

    group = sorted(
        group,
        key=lambda x:
            x["replica_index"],
    )

    bits = [
        int(
            row[
                "extracted_bit"
            ]
        )
        for row in group
    ]

    ones = sum(bits)

    zeros = (
        len(bits)
        -
        ones
    )

    majority_bit = (
        1
        if ones > zeros
        else 0
    )

    confidence = (
        max(
            ones,
            zeros,
        )
        /
        len(bits)
    )

    pattern = "".join(
        str(bit)
        for bit in bits
    )

    return (
        majority_bit,
        confidence,
        pattern,
    )


# ============================================================================
# WEIGHTED DIAGNOSTIC
# ============================================================================

def weighted_vote(
    group: List[Dict],
) -> Tuple[int, float]:

    score = 0.0

    weight_sum = 0.0

    for row in group:

        weight = max(
            float(
                row[
                    "joint_suitability"
                ]
            ),
            1e-12,
        )

        displacement = float(
            row[
                "normalized_displacement"
            ]
        )

        score += (
            weight
            *
            math.tanh(
                displacement
            )
        )

        weight_sum += weight

    score /= max(
        weight_sum,
        1e-12,
    )

    bit = int(
        score >= 0.0
    )

    return (
        bit,
        float(score),
    )


# ============================================================================
# PAYLOAD-LEVEL METRICS
# ============================================================================

def calculate_metrics(
    payload: np.ndarray,
    extracted: List[Dict],
) -> Dict:

    groups = {}

    for row in extracted:

        groups.setdefault(
            row["bit_index"],
            [],
        ).append(
            row
        )

    majority_bits = []

    weighted_bits = []

    confidence_values = []

    three_of_three = 0

    two_of_three = 0

    one_of_three = 0

    zero_of_three = 0

    replica_correct = 0

    total_replicas = 0

    majority_details = []

    for bit_index in range(
        len(payload)
    ):

        group = groups[
            bit_index
        ]

        majority_bit, confidence, pattern = (
            majority_vote(
                group
            )
        )

        weighted_bit, weighted_score = (
            weighted_vote(
                group
            )
        )

        expected = int(
            payload[
                bit_index
            ]
        )

        correct_count = sum(
            int(
                row[
                    "extracted_bit"
                ]
                ==
                expected
            )
            for row in group
        )

        replica_correct += (
            correct_count
        )

        total_replicas += (
            len(group)
        )

        if correct_count == 3:

            three_of_three += 1

        elif correct_count == 2:

            two_of_three += 1

        elif correct_count == 1:

            one_of_three += 1

        else:

            zero_of_three += 1

        majority_bits.append(
            majority_bit
        )

        weighted_bits.append(
            weighted_bit
        )

        confidence_values.append(
            confidence
        )

        majority_details.append(
            {
                "bit_index":
                    bit_index,

                "expected_bit":
                    expected,

                "replica_pattern":
                    pattern,

                "replica_correct":
                    correct_count,

                "majority_bit":
                    majority_bit,

                "majority_correct":
                    int(
                        majority_bit
                        ==
                        expected
                    ),

                "majority_confidence":
                    confidence,

                "weighted_bit":
                    weighted_bit,

                "weighted_score":
                    weighted_score,
            }
        )

    majority_bits = np.asarray(
        majority_bits,
        dtype=np.int64,
    )

    weighted_bits = np.asarray(
        weighted_bits,
        dtype=np.int64,
    )

    payload = np.asarray(
        payload,
        dtype=np.int64,
    )

    majority_correct = int(
        np.sum(
            majority_bits
            ==
            payload
        )
    )

    weighted_correct = int(
        np.sum(
            weighted_bits
            ==
            payload
        )
    )

    total_bits = len(
        payload
    )

    majority_accuracy = (
        majority_correct
        /
        total_bits
    )

    weighted_accuracy = (
        weighted_correct
        /
        total_bits
    )

    replica_accuracy = (
        replica_correct
        /
        total_replicas
    )

    return {
        "total_bits":
            int(total_bits),

        "replica_total":
            int(total_replicas),

        "replica_correct":
            int(replica_correct),

        "replica_incorrect":
            int(
                total_replicas
                -
                replica_correct
            ),

        "replica_accuracy":
            float(
                replica_accuracy
            ),

        "replica_ber":
            float(
                1.0
                -
                replica_accuracy
            ),

        "majority_correct":
            int(
                majority_correct
            ),

        "majority_incorrect":
            int(
                total_bits
                -
                majority_correct
            ),

        "majority_accuracy":
            float(
                majority_accuracy
            ),

        "majority_ber":
            float(
                1.0
                -
                majority_accuracy
            ),

        "weighted_correct":
            int(
                weighted_correct
            ),

        "weighted_accuracy":
            float(
                weighted_accuracy
            ),

        "bits_3_of_3":
            int(
                three_of_three
            ),

        "bits_2_of_3":
            int(
                two_of_three
            ),

        "bits_1_of_3":
            int(
                one_of_three
            ),

        "bits_0_of_3":
            int(
                zero_of_three
            ),

        "mean_majority_confidence":
            float(
                np.mean(
                    confidence_values
                )
            ),

        "majority_details":
            majority_details,
    }


# ============================================================================
# AWGN
# ============================================================================

def add_awgn(
    audio: np.ndarray,
    target_snr_db: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:

    audio = np.asarray(
        audio,
        dtype=np.float64,
    )

    signal_power = float(
        np.mean(
            audio ** 2
        )
    )

    if signal_power <= 0.0:

        raise ValueError(
            "Audio signal power is zero."
        )

    snr_linear = (
        10.0
        **
        (
            target_snr_db
            /
            10.0
        )
    )

    noise_power = (
        signal_power
        /
        snr_linear
    )

    noise_std = math.sqrt(
        noise_power
    )

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=audio.shape,
    )

    attacked = (
        audio
        +
        noise
    )

    # Keep WAV writing safe.
    attacked = np.clip(
        attacked,
        -1.0,
        1.0,
    )

    return (
        attacked.astype(
            np.float32
        ),
        noise.astype(
            np.float64
        ),
    )


# ============================================================================
# ACTUAL ATTACK SNR
# ============================================================================

def calculate_snr(
    signal: np.ndarray,
    noise: np.ndarray,
) -> float:

    signal_power = float(
        np.mean(
            np.asarray(
                signal,
                dtype=np.float64,
            )
            ** 2
        )
    )

    noise_power = float(
        np.mean(
            np.asarray(
                noise,
                dtype=np.float64,
            )
            ** 2
        )
    )

    if noise_power <= 0.0:

        return float(
            "inf"
        )

    return float(
        10.0
        *
        np.log10(
            signal_power
            /
            noise_power
        )
    )


# ============================================================================
# AUDIO METRICS
# ============================================================================

def calculate_audio_metrics(
    clean: np.ndarray,
    attacked: np.ndarray,
) -> Dict:

    clean = np.asarray(
        clean,
        dtype=np.float64,
    )

    attacked = np.asarray(
        attacked,
        dtype=np.float64,
    )

    n = min(
        len(clean),
        len(attacked),
    )

    clean = clean[:n]

    attacked = attacked[:n]

    error = (
        attacked
        -
        clean
    )

    mse = float(
        np.mean(
            error ** 2
        )
    )

    rmse = float(
        np.sqrt(
            mse
        )
    )

    signal_power = float(
        np.mean(
            clean ** 2
        )
    )

    noise_power = float(
        np.mean(
            error ** 2
        )
    )

    if noise_power > 0.0:

        snr = float(
            10.0
            *
            np.log10(
                signal_power
                /
                noise_power
            )
        )

    else:

        snr = float(
            "inf"
        )

    peak_error = float(
        np.max(
            np.abs(
                error
            )
        )
    )

    if (
        np.std(clean) > 0
        and
        np.std(attacked) > 0
    ):

        correlation = float(
            np.corrcoef(
                clean,
                attacked,
            )[0, 1]
        )

    else:

        correlation = 0.0

    return {
        "mse":
            mse,

        "rmse":
            rmse,

        "snr_db":
            snr,

        "peak_absolute_error":
            peak_error,

        "waveform_correlation":
            correlation,
    }


# ============================================================================
# SAVE EXTRACTION DETAILS
# ============================================================================

def save_extraction_details(
    extracted: List[Dict],
    metrics: Dict,
    target_snr: float,
) -> Path:

    rows = []

    majority_map = {
        item[
            "bit_index"
        ]:
        item
        for item in metrics[
            "majority_details"
        ]
    }

    for row in extracted:

        bit_index = row[
            "bit_index"
        ]

        majority_info = (
            majority_map[
                bit_index
            ]
        )

        rows.append(
            {
                "bit_index":
                    bit_index,

                "replica_index":
                    row[
                        "replica_index"
                    ],

                "payload_bit":
                    row[
                        "payload_bit"
                    ],

                "region_id":
                    row[
                        "region_id"
                    ],

                "time_frame":
                    row[
                        "time_frame"
                    ],

                "frequency_bin":
                    row[
                        "frequency_bin"
                    ],

                "joint_suitability":
                    row[
                        "joint_suitability"
                    ],

                "adaptive_strength":
                    row[
                        "adaptive_strength"
                    ],

                "reference_stft_real":
                    row[
                        "reference_stft_real"
                    ],

                "observed_stft_real":
                    row[
                        "observed_stft_real"
                    ],

                "observed_change":
                    row[
                        "observed_change"
                    ],

                "normalized_displacement":
                    row[
                        "normalized_displacement"
                    ],

                "extracted_bit":
                    row[
                        "extracted_bit"
                    ],

                "replica_correct":
                    row[
                        "correct"
                    ],

                "replica_pattern":
                    majority_info[
                        "replica_pattern"
                    ],

                "replica_correct_count":
                    majority_info[
                        "replica_correct"
                    ],

                "majority_bit":
                    majority_info[
                        "majority_bit"
                    ],

                "majority_correct":
                    majority_info[
                        "majority_correct"
                    ],

                "majority_confidence":
                    majority_info[
                        "majority_confidence"
                    ],

                "weighted_bit":
                    majority_info[
                        "weighted_bit"
                    ],

                "weighted_score":
                    majority_info[
                        "weighted_score"
                    ],

                "reference_csv_error":
                    row[
                        "reference_csv_error"
                    ],
            }
        )

    output = (
        OUTPUT_DIR
        /
        f"extraction_{int(target_snr)}db.csv"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        output,
        index=False,
    )

    return output


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print(
        "=" * 72
    )

    print(
        "SemBind-Audio : Experiment 16"
    )

    print(
        "Upgraded Redundant AWGN Robustness"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------------
    # Load everything
    # ------------------------------------------------------------------------

    payload = load_payload()

    reference_stft = load_stft(
        REFERENCE_STFT_FILE,
        "reference STFT",
    )

    watermarked_stft = load_stft(
        WATERMARKED_STFT_FILE,
        "watermarked STFT",
    )

    coordinates = load_coordinates()

    watermarked_audio, sample_rate = (
        load_audio(
            WATERMARKED_AUDIO_FILE
        )
    )

    # ------------------------------------------------------------------------
    # Basic information
    # ------------------------------------------------------------------------

    print()

    print(
        "Current Watermark"
    )

    print(
        "-" * 72
    )

    print(
        f"Payload Length        : "
        f"{len(payload)} bits"
    )

    print(
        f"Redundancy            : "
        f"{REDUNDANCY}x"
    )

    print(
        f"Coordinates           : "
        f"{len(coordinates)}"
    )

    print(
        f"Sample Rate           : "
        f"{sample_rate}"
    )

    print(
        f"Reference STFT Shape  : "
        f"{tuple(reference_stft.shape)}"
    )

    print(
        f"Watermarked STFT Shape: "
        f"{tuple(watermarked_stft.shape)}"
    )

    # ------------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------------

    validate_mapping(
        payload,
        coordinates,
    )

    if (
        reference_stft.shape
        !=
        watermarked_stft.shape
    ):

        raise ValueError(
            "Reference and watermarked "
            "STFT shapes differ."
        )

    # ------------------------------------------------------------------------
    # Validate clean watermarked STFT
    # ------------------------------------------------------------------------

    clean_extracted = extract_replicas(
        reference_stft,
        watermarked_stft,
        coordinates,
    )

    clean_metrics = calculate_metrics(
        payload,
        clean_extracted,
    )

    print()

    print(
        "CLEAN WATERMARK VALIDATION"
    )

    print(
        "-" * 72
    )

    print(
        f"Replica Accuracy      : "
        f"{clean_metrics['replica_accuracy']:.6f}"
    )

    print(
        f"Majority Accuracy     : "
        f"{clean_metrics['majority_accuracy']:.6f}"
    )

    print(
        f"Majority BER          : "
        f"{clean_metrics['majority_ber']:.6f}"
    )

    if (
        clean_metrics[
            "majority_accuracy"
        ]
        <
        1.0
    ):

        raise ValueError(
            "Clean redundant watermark "
            "does not recover perfectly. "
            "Do not proceed with noise "
            "robustness until this is fixed."
        )

    print(
        "Clean Verification    : PASSED"
    )

    # ------------------------------------------------------------------------
    # Random generator
    # ------------------------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    all_results = []

    # ------------------------------------------------------------------------
    # Noise attacks
    # ------------------------------------------------------------------------

    for target_snr in (
        NOISE_SNR_LEVELS
    ):

        print()

        print(
            "=" * 72
        )

        print(
            f"NOISE ATTACK : "
            f"{target_snr:.0f} dB"
        )

        print(
            "=" * 72
        )

        # --------------------------------------------------------------------
        # Generate attack
        # --------------------------------------------------------------------

        attacked_audio, noise = (
            add_awgn(
                watermarked_audio,
                target_snr,
                rng,
            )
        )

        actual_snr = calculate_snr(
            watermarked_audio,
            noise,
        )

        # --------------------------------------------------------------------
        # Save attack
        # --------------------------------------------------------------------

        attacked_path = (
            OUTPUT_DIR
            /
            f"noisy_{int(target_snr)}db.wav"
        )

        sf.write(
            str(attacked_path),
            attacked_audio,
            sample_rate,
        )

        # --------------------------------------------------------------------
        # STFT
        # --------------------------------------------------------------------

        attacked_stft = generate_stft(
            attacked_audio
        )

        # --------------------------------------------------------------------
        # Validate STFT
        # --------------------------------------------------------------------

        if (
            attacked_stft.shape
            !=
            reference_stft.shape
        ):

            raise ValueError(
                f"STFT shape changed "
                f"under {target_snr} dB attack.\n"
                f"Expected: "
                f"{tuple(reference_stft.shape)}\n"
                f"Observed: "
                f"{tuple(attacked_stft.shape)}"
            )

        # --------------------------------------------------------------------
        # Extract 3 replicas
        # --------------------------------------------------------------------

        extracted = extract_replicas(
            reference_stft,
            attacked_stft,
            coordinates,
        )

        metrics = calculate_metrics(
            payload,
            extracted,
        )

        # --------------------------------------------------------------------
        # Audio quality
        # --------------------------------------------------------------------

        audio_metrics = (
            calculate_audio_metrics(
                watermarked_audio,
                attacked_audio,
            )
        )

        # --------------------------------------------------------------------
        # Save detailed CSV
        # --------------------------------------------------------------------

        extraction_file = (
            save_extraction_details(
                extracted,
                metrics,
                target_snr,
            )
        )

        # --------------------------------------------------------------------
        # Print
        # --------------------------------------------------------------------

        print()

        print(
            f"Target Attack SNR     : "
            f"{target_snr:.2f} dB"
        )

        print(
            f"Actual Attack SNR     : "
            f"{actual_snr:.6f} dB"
        )

        print()

        print(
            f"Replica Accuracy      : "
            f"{metrics['replica_accuracy']:.6f}"
        )

        print(
            f"Replica BER           : "
            f"{metrics['replica_ber']:.6f}"
        )

        print()

        print(
            f"Majority Accuracy     : "
            f"{metrics['majority_accuracy']:.6f}"
        )

        print(
            f"Majority BER          : "
            f"{metrics['majority_ber']:.6f}"
        )

        print()

        print(
            f"Weighted Accuracy     : "
            f"{metrics['weighted_accuracy']:.6f}"
        )

        print()

        print(
            f"Bits 3/3 Correct      : "
            f"{metrics['bits_3_of_3']}"
        )

        print(
            f"Bits 2/3 Correct      : "
            f"{metrics['bits_2_of_3']}"
        )

        print(
            f"Bits 1/3 Correct      : "
            f"{metrics['bits_1_of_3']}"
        )

        print(
            f"Bits 0/3 Correct      : "
            f"{metrics['bits_0_of_3']}"
        )

        print()

        print(
            f"Attack MSE            : "
            f"{audio_metrics['mse']:.12e}"
        )

        print(
            f"Attack RMSE           : "
            f"{audio_metrics['rmse']:.12e}"
        )

        print(
            f"Attack SNR            : "
            f"{audio_metrics['snr_db']:.6f} dB"
        )

        print(
            f"Peak Absolute Error   : "
            f"{audio_metrics['peak_absolute_error']:.12e}"
        )

        print(
            f"Waveform Correlation  : "
            f"{audio_metrics['waveform_correlation']:.12f}"
        )

        print()

        if (
            metrics[
                "majority_accuracy"
            ]
            >=
            0.90
        ):

            verification = (
                "PASSED"
            )

        else:

            verification = (
                "DEGRADED"
            )

        print(
            f"Majority Verification : "
            f"{verification}"
        )

        # --------------------------------------------------------------------
        # Reference consistency
        # --------------------------------------------------------------------

        reference_errors = [
            row[
                "reference_csv_error"
            ]
            for row in extracted
        ]

        mean_reference_error = float(
            np.mean(
                reference_errors
            )
        )

        max_reference_error = float(
            np.max(
                reference_errors
            )
        )

        # --------------------------------------------------------------------
        # Store result
        # --------------------------------------------------------------------

        result = {
            "target_attack_snr_db":
                float(
                    target_snr
                ),

            "actual_attack_snr_db":
                float(
                    actual_snr
                ),

            "audio_file":
                str(
                    attacked_path
                ),

            "payload_length":
                int(
                    len(payload)
                ),

            "redundancy":
                int(
                    REDUNDANCY
                ),

            "coordinate_count":
                int(
                    len(coordinates)
                ),

            "total_replicas":
                int(
                    metrics[
                        "replica_total"
                    ]
                ),

            "replica_correct":
                int(
                    metrics[
                        "replica_correct"
                    ]
                ),

            "replica_incorrect":
                int(
                    metrics[
                        "replica_incorrect"
                    ]
                ),

            "replica_accuracy":
                float(
                    metrics[
                        "replica_accuracy"
                    ]
                ),

            "replica_ber":
                float(
                    metrics[
                        "replica_ber"
                    ]
                ),

            "majority_correct":
                int(
                    metrics[
                        "majority_correct"
                    ]
                ),

            "majority_incorrect":
                int(
                    metrics[
                        "majority_incorrect"
                    ]
                ),

            "majority_accuracy":
                float(
                    metrics[
                        "majority_accuracy"
                    ]
                ),

            "majority_ber":
                float(
                    metrics[
                        "majority_ber"
                    ]
                ),

            "weighted_accuracy":
                float(
                    metrics[
                        "weighted_accuracy"
                    ]
                ),

            "bits_3_of_3":
                int(
                    metrics[
                        "bits_3_of_3"
                    ]
                ),

            "bits_2_of_3":
                int(
                    metrics[
                        "bits_2_of_3"
                    ]
                ),

            "bits_1_of_3":
                int(
                    metrics[
                        "bits_1_of_3"
                    ]
                ),

            "bits_0_of_3":
                int(
                    metrics[
                        "bits_0_of_3"
                    ]
                ),

            "mean_majority_confidence":
                float(
                    metrics[
                        "mean_majority_confidence"
                    ]
                ),

            "mse":
                float(
                    audio_metrics[
                        "mse"
                    ]
                ),

            "rmse":
                float(
                    audio_metrics[
                        "rmse"
                    ]
                ),

            "snr_db":
                float(
                    audio_metrics[
                        "snr_db"
                    ]
                ),

            "peak_absolute_error":
                float(
                    audio_metrics[
                        "peak_absolute_error"
                    ]
                ),

            "waveform_correlation":
                float(
                    audio_metrics[
                        "waveform_correlation"
                    ]
                ),

            "mean_reference_csv_error":
                mean_reference_error,

            "max_reference_csv_error":
                max_reference_error,

            "extraction_details":
                str(
                    extraction_file
                ),
        }

        all_results.append(
            result
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()

    print(
        "=" * 72
    )

    print(
        "NOISE ROBUSTNESS SUMMARY"
    )

    print(
        "=" * 72
    )

    summary_columns = [
        "target_attack_snr_db",
        "actual_attack_snr_db",
        "replica_accuracy",
        "replica_ber",
        "majority_accuracy",
        "majority_ber",
        "weighted_accuracy",
        "bits_3_of_3",
        "bits_2_of_3",
        "bits_1_of_3",
        "bits_0_of_3",
    ]

    summary_df = pd.DataFrame(
        all_results
    )

    print(
        summary_df[
            summary_columns
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        /
        "noise_robustness_results.csv"
    )

    summary_df.to_csv(
        csv_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------------------

    json_path = (
        OUTPUT_DIR
        /
        "noise_robustness_summary.json"
    )

    summary = {
        "experiment":
            16,

        "description":
            (
                "Additive Gaussian noise "
                "robustness of the upgraded "
                "3x redundant semantic-spectral "
                "watermark."
            ),

        "method":
            (
                "Exp21 redundant embedding "
                "+ Exp20 majority-vote extraction"
            ),

        "sample_rate":
            SAMPLE_RATE,

        "n_fft":
            N_FFT,

        "hop_length":
            HOP_LENGTH,

        "win_length":
            WIN_LENGTH,

        "payload_length":
            int(
                len(payload)
            ),

        "redundancy":
            REDUNDANCY,

        "coordinate_count":
            int(
                len(coordinates)
            ),

        "random_seed":
            RANDOM_SEED,

        "clean_replica_accuracy":
            float(
                clean_metrics[
                    "replica_accuracy"
                ]
            ),

        "clean_majority_accuracy":
            float(
                clean_metrics[
                    "majority_accuracy"
                ]
            ),

        "clean_majority_ber":
            float(
                clean_metrics[
                    "majority_ber"
                ]
            ),

        "results":
            all_results,
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    # =========================================================================
    # FINAL
    # =========================================================================

    print()

    print(
        "Results Saved"
    )

    print(
        csv_path
    )

    print(
        json_path
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "=" * 72
    )

    print(
        "Experiment 16 Upgrade Complete"
    )

    print(
        "=" * 72
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()