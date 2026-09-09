"""
SemBind-Audio : Experiment 17
Embedding Strength Sweep and Noise-Robustness Trade-off

Purpose
-------
Determine how the baseline embedding strength affects:

1. Clean watermark extraction
2. Additive-noise robustness
3. Audio imperceptibility

The semantic-adaptive weighting from Experiment 13 is preserved.

For region r:

    alpha_r = alpha_0 * suitability_factor_r

where:

    suitability_factor_r =
        adaptive_strength_r / original_baseline_strength

This experiment changes only the global baseline strength.
"""

from pathlib import Path
import sys
import json
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import librosa


from configs.config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
)


# ================================================================
# PATHS
# ================================================================

RESULTS_ROOT = PROJECT_ROOT / "results"

STFT_FILE = (
    RESULTS_ROOT
    / "adaptive_embedding"
    / "original_stft.pt"
)

COORDINATES_FILE = (
    RESULTS_ROOT
    / "adaptive_embedding"
    / "adaptive_embedding_coordinates.csv"
)

PAYLOAD_FILE = (
    RESULTS_ROOT
    / "watermark"
    / "payload_bits.pt"
)

OUTPUT_DIR = (
    RESULTS_ROOT
    / "strength_sweep"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# EXPERIMENT PARAMETERS
# ================================================================

ORIGINAL_BASELINE = 0.001

STRENGTHS = [
    0.00025,
    0.00050,
    0.00075,
    0.00100,
    0.00150,
    0.00200,
    0.00300,
    0.00500,
    0.01000,
]

ATTACK_SNRS = [
    40.0,
    30.0,
    20.0,
]

RANDOM_SEED = 42


# ================================================================
# LOAD HELPERS
# ================================================================

def load_tensor(path):

    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def load_payload():

    payload = load_tensor(
        PAYLOAD_FILE
    )

    payload = torch.as_tensor(
        payload,
        dtype=torch.int64,
    ).flatten()

    if payload.numel() == 0:
        raise ValueError(
            "Payload is empty."
        )

    if not torch.all(
        (payload == 0)
        |
        (payload == 1)
    ):
        raise ValueError(
            "Payload must contain only 0 and 1."
        )

    return payload


# ================================================================
# VALIDATION
# ================================================================

def validate_inputs(
    stft,
    coordinates,
    payload,
):

    required_columns = {
        "bit_index",
        "payload_bit",
        "time_frame",
        "frequency_bin",
        "adaptive_strength",
    }

    missing = (
        required_columns
        -
        set(coordinates.columns)
    )

    if missing:
        raise ValueError(
            "Missing required coordinate columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    if len(coordinates) != payload.numel():

        raise ValueError(
            f"Coordinate rows = {len(coordinates)} "
            f"but payload length = {payload.numel()}."
        )

    frequency_bins = stft.shape[0]
    time_frames = stft.shape[1]

    for _, row in coordinates.iterrows():

        f = int(
            row["frequency_bin"]
        )

        t = int(
            row["time_frame"]
        )

        if f < 0 or f >= frequency_bins:

            raise ValueError(
                f"Invalid frequency bin: {f}"
            )

        if t < 0 or t >= time_frames:

            raise ValueError(
                f"Invalid time frame: {t}"
            )


# ================================================================
# SEMANTIC ADAPTIVE FACTOR
# ================================================================

def calculate_factors(
    coordinates
):

    factors = (
        coordinates[
            "adaptive_strength"
        ].astype(float)
        /
        ORIGINAL_BASELINE
    )

    return factors.to_numpy(
        dtype=np.float64
    )


# ================================================================
# WATERMARK EMBEDDING
# ================================================================

def embed_watermark(
    original_stft,
    coordinates,
    payload,
    baseline_strength,
    factors,
):

    watermarked_stft = (
        original_stft.clone()
    )

    actual_strengths = []
    deltas = []

    for i, row in coordinates.iterrows():

        bit = int(
            payload[i].item()
        )

        frequency_bin = int(
            row["frequency_bin"]
        )

        time_frame = int(
            row["time_frame"]
        )

        adaptive_strength = (
            baseline_strength
            *
            factors[i]
        )

        if bit == 1:
            delta = adaptive_strength
        else:
            delta = -adaptive_strength

        original_value = (
            watermarked_stft[
                frequency_bin,
                time_frame,
            ]
        )

        watermarked_stft[
            frequency_bin,
            time_frame
        ] = (
            original_value
            +
            torch.tensor(
                delta,
                dtype=original_value.real.dtype,
            )
        )

        actual_strengths.append(
            adaptive_strength
        )

        deltas.append(
            delta
        )

    return (
        watermarked_stft,
        np.asarray(
            actual_strengths
        ),
        np.asarray(
            deltas
        ),
    )


# ================================================================
# STFT → AUDIO
# ================================================================

def reconstruct_audio(
    stft
):

    stft_np = (
        stft
        .detach()
        .cpu()
        .numpy()
    )

    audio = librosa.istft(
        stft_np,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
    )

    return audio.astype(
        np.float32
    )


# ================================================================
# METRICS
# ================================================================

def calculate_mse(
    original,
    processed,
):

    n = min(
        len(original),
        len(processed),
    )

    original = original[:n]
    processed = processed[:n]

    return float(
        np.mean(
            (original - processed) ** 2
        )
    )


def calculate_rmse(
    original,
    processed,
):

    return math.sqrt(
        calculate_mse(
            original,
            processed,
        )
    )


def calculate_snr(
    original,
    processed,
):

    n = min(
        len(original),
        len(processed),
    )

    original = original[:n]
    processed = processed[:n]

    signal_power = np.mean(
        original ** 2
    )

    noise_power = np.mean(
        (processed - original) ** 2
    )

    if noise_power == 0:
        return float("inf")

    return float(
        10
        *
        np.log10(
            signal_power
            /
            noise_power
        )
    )


def calculate_correlation(
    original,
    processed,
):

    n = min(
        len(original),
        len(processed),
    )

    original = original[:n]
    processed = processed[:n]

    if (
        np.std(original) == 0
        or
        np.std(processed) == 0
    ):
        return 0.0

    return float(
        np.corrcoef(
            original,
            processed,
        )[0, 1]
    )


# ================================================================
# NOISE
# ================================================================

def add_noise(
    audio,
    target_snr_db,
    rng,
):

    signal_power = np.mean(
        audio.astype(
            np.float64
        ) ** 2
    )

    noise_power = (
        signal_power
        /
        (
            10
            **
            (
                target_snr_db
                /
                10.0
            )
        )
    )

    noise = rng.normal(
        0.0,
        np.sqrt(noise_power),
        size=audio.shape,
    )

    attacked_audio = (
        audio
        +
        noise.astype(
            np.float32
        )
    )

    actual_snr = calculate_snr(
        audio,
        attacked_audio,
    )

    return (
        attacked_audio.astype(
            np.float32
        ),
        actual_snr,
    )


# ================================================================
# EXTRACTION
# ================================================================

def extract_bits(
    original_stft,
    attacked_stft,
    coordinates,
):

    extracted = []

    for _, row in coordinates.iterrows():

        frequency_bin = int(
            row["frequency_bin"]
        )

        time_frame = int(
            row["time_frame"]
        )

        original_value = (
            original_stft[
                frequency_bin,
                time_frame,
            ]
        )

        attacked_value = (
            attacked_stft[
                frequency_bin,
                time_frame,
            ]
        )

        change = (
            attacked_value.real
            -
            original_value.real
        )

        if float(change) >= 0:

            extracted.append(1)

        else:

            extracted.append(0)

    return torch.tensor(
        extracted,
        dtype=torch.int64,
    )


# ================================================================
# BIT METRICS
# ================================================================

def calculate_bit_metrics(
    extracted,
    payload,
):

    extracted = torch.as_tensor(
        extracted,
        dtype=torch.int64,
    )

    payload = torch.as_tensor(
        payload,
        dtype=torch.int64,
    )

    correct = int(
        (
            extracted
            ==
            payload
        ).sum().item()
    )

    total = int(
        payload.numel()
    )

    incorrect = (
        total
        -
        correct
    )

    accuracy = (
        correct
        /
        total
    )

    ber = (
        incorrect
        /
        total
    )

    return {
        "total_bits": total,
        "correct_bits": correct,
        "incorrect_bits": incorrect,
        "bit_accuracy": accuracy,
        "bit_error_rate": ber,
    }


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 64)
    print(
        "SemBind-Audio : Experiment 17"
    )
    print(
        "Embedding Strength Sweep"
    )
    print("=" * 64)

    print()

    original_stft = load_tensor(
        STFT_FILE
    )

    if not torch.is_complex(
        original_stft
    ):
        original_stft = (
            original_stft
            .to(torch.complex64)
        )

    coordinates = pd.read_csv(
        COORDINATES_FILE
    )

    payload = load_payload()

    validate_inputs(
        original_stft,
        coordinates,
        payload,
    )

    factors = calculate_factors(
        coordinates
    )

    print(
        f"STFT Shape       : "
        f"{tuple(original_stft.shape)}"
    )

    print(
        f"Payload Length   : "
        f"{payload.numel()} bits"
    )

    print(
        f"Coordinates      : "
        f"{len(coordinates)}"
    )

    print(
        f"Strength Values  : "
        f"{len(STRENGTHS)}"
    )

    print(
        f"Noise Attacks    : "
        f"{ATTACK_SNRS}"
    )

    print()

    original_audio = reconstruct_audio(
        original_stft
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    all_results = []

    # ============================================================
    # SWEEP
    # ============================================================

    for baseline_strength in STRENGTHS:

        print("=" * 64)
        print(
            f"Baseline Strength : "
            f"{baseline_strength:.8f}"
        )
        print("=" * 64)

        (
            watermarked_stft,
            actual_strengths,
            deltas,
        ) = embed_watermark(
            original_stft,
            coordinates,
            payload,
            baseline_strength,
            factors,
        )

        watermarked_audio = (
            reconstruct_audio(
                watermarked_stft
            )
        )

        # --------------------------------------------------------
        # CLEAN QUALITY
        # --------------------------------------------------------

        clean_mse = calculate_mse(
            original_audio,
            watermarked_audio,
        )

        clean_rmse = calculate_rmse(
            original_audio,
            watermarked_audio,
        )

        clean_snr = calculate_snr(
            original_audio,
            watermarked_audio,
        )

        clean_corr = calculate_correlation(
            original_audio,
            watermarked_audio,
        )

        # --------------------------------------------------------
        # CLEAN EXTRACTION
        # --------------------------------------------------------

        extracted_clean = extract_bits(
            original_stft,
            watermarked_stft,
            coordinates,
        )

        clean_bits = calculate_bit_metrics(
            extracted_clean,
            payload,
        )

        mean_embedded_change = float(
            np.mean(
                np.abs(deltas)
            )
        )

        max_embedded_change = float(
            np.max(
                np.abs(deltas)
            )
        )

        print()
        print(
            "Clean Condition"
        )

        print(
            f"Bit Accuracy       : "
            f"{clean_bits['bit_accuracy']:.6f}"
        )

        print(
            f"BER                : "
            f"{clean_bits['bit_error_rate']:.6f}"
        )

        print(
            f"MSE                : "
            f"{clean_mse:.12e}"
        )

        print(
            f"RMSE               : "
            f"{clean_rmse:.12e}"
        )

        print(
            f"SNR                : "
            f"{clean_snr:.6f} dB"
        )

        print(
            f"Correlation        : "
            f"{clean_corr:.12f}"
        )

        print(
            f"Mean Embedded Δ    : "
            f"{mean_embedded_change:.10f}"
        )

        print(
            f"Max Embedded Δ     : "
            f"{max_embedded_change:.10f}"
        )

        all_results.append(
            {
                "baseline_strength":
                    baseline_strength,
                "condition":
                    "clean",
                "attack_snr_db":
                    None,
                "actual_attack_snr_db":
                    None,
                "mse":
                    clean_mse,
                "rmse":
                    clean_rmse,
                "snr_db":
                    clean_snr,
                "waveform_correlation":
                    clean_corr,
                "bit_accuracy":
                    clean_bits[
                        "bit_accuracy"
                    ],
                "bit_error_rate":
                    clean_bits[
                        "bit_error_rate"
                    ],
                "correct_bits":
                    clean_bits[
                        "correct_bits"
                    ],
                "incorrect_bits":
                    clean_bits[
                        "incorrect_bits"
                    ],
                "mean_embedded_change":
                    mean_embedded_change,
                "max_embedded_change":
                    max_embedded_change,
            }
        )

        # --------------------------------------------------------
        # NOISE ATTACKS
        # --------------------------------------------------------

        for attack_snr in ATTACK_SNRS:

            (
                attacked_audio,
                actual_attack_snr,
            ) = add_noise(
                watermarked_audio,
                attack_snr,
                rng,
            )

            attacked_stft_np = librosa.stft(
                attacked_audio,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=WIN_LENGTH,
            )

            attacked_stft = torch.from_numpy(
                attacked_stft_np
            ).to(torch.complex64)

            # Ensure dimensions match.
            min_frequency = min(
                original_stft.shape[0],
                attacked_stft.shape[0],
            )

            min_time = min(
                original_stft.shape[1],
                attacked_stft.shape[1],
            )

            reference_stft = (
                original_stft[
                    :min_frequency,
                    :min_time,
                ]
            )

            attacked_stft = (
                attacked_stft[
                    :min_frequency,
                    :min_time,
                ]
            )

            valid_mask = (
                coordinates[
                    "frequency_bin"
                ].astype(int)
                <
                min_frequency
            ) & (
                coordinates[
                    "time_frame"
                ].astype(int)
                <
                min_time
            )

            valid_coordinates = (
                coordinates[
                    valid_mask
                ].copy()
            )

            valid_indices = (
                valid_coordinates[
                    "bit_index"
                ].astype(int)
                .to_numpy()
            )

            attack_payload = payload[
                valid_indices
            ]

            extracted_attack = extract_bits(
                reference_stft,
                attacked_stft,
                valid_coordinates,
            )

            attack_bits = calculate_bit_metrics(
                extracted_attack,
                attack_payload,
            )

            attack_mse = calculate_mse(
                watermarked_audio,
                attacked_audio,
            )

            attack_rmse = calculate_rmse(
                watermarked_audio,
                attacked_audio,
            )

            attack_corr = calculate_correlation(
                watermarked_audio,
                attacked_audio,
            )

            print()
            print(
                f"Noise Attack : "
                f"{attack_snr:.0f} dB"
            )

            print(
                f"Actual SNR   : "
                f"{actual_attack_snr:.6f} dB"
            )

            print(
                f"Bit Accuracy : "
                f"{attack_bits['bit_accuracy']:.6f}"
            )

            print(
                f"BER          : "
                f"{attack_bits['bit_error_rate']:.6f}"
            )

            print(
                f"Correct Bits : "
                f"{attack_bits['correct_bits']}"
            )

            print(
                f"Attack MSE   : "
                f"{attack_mse:.12e}"
            )

            print(
                f"Correlation  : "
                f"{attack_corr:.12f}"
            )

            all_results.append(
                {
                    "baseline_strength":
                        baseline_strength,
                    "condition":
                        f"noise_{attack_snr:.0f}dB",
                    "attack_snr_db":
                        attack_snr,
                    "actual_attack_snr_db":
                        actual_attack_snr,
                    "mse":
                        attack_mse,
                    "rmse":
                        attack_rmse,
                    "snr_db":
                        attack_snr,
                    "waveform_correlation":
                        attack_corr,
                    "bit_accuracy":
                        attack_bits[
                            "bit_accuracy"
                        ],
                    "bit_error_rate":
                        attack_bits[
                            "bit_error_rate"
                        ],
                    "correct_bits":
                        attack_bits[
                            "correct_bits"
                        ],
                    "incorrect_bits":
                        attack_bits[
                            "incorrect_bits"
                        ],
                    "mean_embedded_change":
                        mean_embedded_change,
                    "max_embedded_change":
                        max_embedded_change,
                }
            )

    # ============================================================
    # SAVE
    # ============================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_file = (
        OUTPUT_DIR
        /
        "strength_sweep_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    clean_df = results_df[
        results_df["condition"]
        ==
        "clean"
    ]

    clean_df.to_csv(
        OUTPUT_DIR
        /
        "clean_strength_tradeoff.csv",
        index=False,
    )

    noise_df = results_df[
        results_df["condition"]
        !=
        "clean"
    ]

    noise_df.to_csv(
        OUTPUT_DIR
        /
        "noise_strength_tradeoff.csv",
        index=False,
    )

    # ============================================================
    # AUTOMATIC CANDIDATE SEARCH
    # ============================================================

    robust_40 = results_df[
        (
            results_df["condition"]
            ==
            "noise_40dB"
        )
        &
        (
            results_df["bit_accuracy"]
            >=
            0.90
        )
    ]

    robust_30 = results_df[
        (
            results_df["condition"]
            ==
            "noise_30dB"
        )
        &
        (
            results_df["bit_accuracy"]
            >=
            0.90
        )
    ]

    candidate_40 = None

    if not robust_40.empty:

        candidate_40 = float(
            robust_40[
                "baseline_strength"
            ].iloc[0]
        )

    candidate_30 = None

    if not robust_30.empty:

        candidate_30 = float(
            robust_30[
                "baseline_strength"
            ].iloc[0]
        )

    summary = {

        "experiment":
            17,

        "description":
            (
                "Controlled sweep of global "
                "semantic-adaptive embedding "
                "strength."
            ),

        "original_baseline":
            ORIGINAL_BASELINE,

        "strength_values":
            STRENGTHS,

        "attack_snr_values_db":
            ATTACK_SNRS,

        "payload_length_bits":
            int(
                payload.numel()
            ),

        "stft_shape":
            list(
                original_stft.shape
            ),

        "candidate_strength_40db_90pct":
            candidate_40,

        "candidate_strength_30db_90pct":
            candidate_30,

        "interpretation":
            (
                "Use this experiment to determine "
                "the trade-off between imperceptibility "
                "and robustness. The final embedding "
                "strength should be selected from "
                "experimental evidence rather than "
                "being manually fixed."
            ),
    }

    with open(
        OUTPUT_DIR
        /
        "strength_sweep_summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    # ============================================================
    # FINAL
    # ============================================================

    print()
    print("=" * 64)
    print(
        "Experiment 17 Complete"
    )
    print("=" * 64)

    print()
    print(
        "Results Saved"
    )

    print(
        results_file
    )

    print(
        OUTPUT_DIR
        /
        "clean_strength_tradeoff.csv"
    )

    print(
        OUTPUT_DIR
        /
        "noise_strength_tradeoff.csv"
    )

    print(
        OUTPUT_DIR
        /
        "strength_sweep_summary.json"
    )


if __name__ == "__main__":
    main()

