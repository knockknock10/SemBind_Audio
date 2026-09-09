"""
SemBind-Audio

Experiment 13
Semantic-Adaptive STFT Watermark Embedding

Purpose
-------
Use the semantic importance obtained from HuBERT to adapt
the watermark embedding strength across selected regions.

Semantic importance:
    S_r

Semantic suitability:
    W_r = 1 - S_r

Adaptive embedding strength:
    alpha_r = alpha_0 * W_r / max(W)

Embedding equation:
    X_w(f,t) = X(f,t) + alpha_r * B_i

where:

    X(f,t)   = original STFT coefficient
    X_w(f,t) = watermarked STFT coefficient
    B_i      = watermark bit representation
    alpha_0  = baseline embedding strength
    S_r      = mean semantic importance of region r
    W_r      = semantic suitability of region r
"""


from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import torch
import soundfile as sf
import librosa


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

N_FFT = 1024
HOP_LENGTH = 320
WIN_LENGTH = 1024

from configs.config import (
    ALPHA,
    BETA,
    DELTA_MIN,
    DELTA_MAX,
)

# Adaptive embedding strength uses joint suitability:
#     J_r = S_r^alpha * P_r^beta
#     delta_r = delta_min + (delta_max - delta_min) * J_r
BASELINE_STRENGTH = DELTA_MIN  # fallback if joint suitability is uniform


# ============================================================
# DIRECTORIES
# ============================================================

RESULTS_DIR = PROJECT_ROOT / "results"

STFT_DIR = RESULTS_DIR / "spectrograms"

REGION_DIR = RESULTS_DIR / "stft_mapping"

PAYLOAD_DIR = RESULTS_DIR / "watermark"

OUTPUT_DIR = RESULTS_DIR / "adaptive_embedding"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT FILES
# ============================================================

STFT_FILE = (
    STFT_DIR
    / "stft.pt"
)

REGION_MAPPING_FILE = (
    REGION_DIR
    / "stft_region_mapping.csv"
)

PAYLOAD_BITS_FILE = (
    PAYLOAD_DIR
    / "payload_bits.pt"
)


# ============================================================
# OUTPUT FILES
# ============================================================

ORIGINAL_STFT_FILE = (
    OUTPUT_DIR
    / "original_stft.pt"
)

WATERMARKED_STFT_FILE = (
    OUTPUT_DIR
    / "adaptive_watermarked_stft.pt"
)

EMBEDDED_BITS_FILE = (
    OUTPUT_DIR
    / "embedded_bits.pt"
)

COORDINATES_FILE = (
    OUTPUT_DIR
    / "adaptive_embedding_coordinates.csv"
)

REGION_WEIGHTS_FILE = (
    OUTPUT_DIR
    / "semantic_region_weights.csv"
)

WATERMARKED_AUDIO_FILE = (
    OUTPUT_DIR
    / "adaptive_watermarked_audio.wav"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "adaptive_embedding_summary.json"
)


# ============================================================
# LOAD STFT
# ============================================================

def load_stft():

    if not STFT_FILE.exists():

        raise FileNotFoundError(
            f"STFT file not found:\n{STFT_FILE}"
        )

    stft = torch.load(
        STFT_FILE,
        map_location="cpu",
    )

    if not torch.is_complex(stft):

        raise ValueError(
            "The loaded STFT is not complex."
        )

    return stft


# ============================================================
# LOAD PAYLOAD
# ============================================================

def load_payload():

    if not PAYLOAD_BITS_FILE.exists():

        raise FileNotFoundError(
            f"Payload bits file not found:\n"
            f"{PAYLOAD_BITS_FILE}"
        )

    payload = torch.load(
        PAYLOAD_BITS_FILE,
        map_location="cpu",
    )

    payload = payload.flatten().to(
        dtype=torch.int64
    )

    if payload.numel() == 0:

        raise ValueError(
            "Payload contains zero bits."
        )

    return payload


# ============================================================
# LOAD REGION MAPPING
# ============================================================

def load_region_mapping():

    if not REGION_MAPPING_FILE.exists():

        raise FileNotFoundError(
            "STFT region mapping file not found:\n"
            f"{REGION_MAPPING_FILE}"
        )

    df = pd.read_csv(
        REGION_MAPPING_FILE
    )

    required_columns = [
        "region_id",
        "time_start_frame",
        "time_end_frame",
        "frequency_start_bin",
        "frequency_end_bin",
        "time_frames",
        "frequency_bins",
        "stft_coefficients",
        "mean_semantic_score",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    return df


# ============================================================
# CALCULATE SEMANTIC WEIGHTS
# ============================================================

def calculate_region_weights(
    region_mapping,
):

    """
    Calculate joint suitability and adaptive embedding strength.

    Per the SemBind-Audio methodology:

        S_r   = mean semantic score of region r (higher = more important)
        P_r   = spectral suitability of region r (from STFT magnitude)

        Semantic suitability:   W_r = 1 - S_r  (lower contribution => higher suitability)
        Spectral suitability:  P_r  = local spectral measure

        Joint suitability:      J_r = S_r^alpha * P_r^beta
            where alpha = 0.70, beta = 0.30

        Adaptive embedding strength:
            delta_r = delta_min + (delta_max - delta_min) * J_r
            where delta_min = 0.002, delta_max = 0.008
    """

    weights = region_mapping[
        [
            "region_id",
            "time_start_frame",
            "time_end_frame",
            "frequency_start_bin",
            "frequency_end_bin",
            "mean_semantic_score",
        ]
    ].copy()

    weights = weights.rename(
        columns={
            "mean_semantic_score":
                "semantic_score"
        }
    )

    # --------------------------------------------------------
    # Semantic suitability
    # --------------------------------------------------------

    # Lower semantic importance (mean score) => higher suitability
    weights[
        "semantic_suitability"
    ] = 1.0 - weights["semantic_score"]

    # --------------------------------------------------------
    # Spectral suitability
    # --------------------------------------------------------

    # TODO: compute from actual STFT spectral statistics
    # For now, use a placeholder derived from region position
    # In a full implementation, this would be log-magnitude averaged
    # over the local STFT region.
    weights["spectral_suitability"] = torch.ones(
        len(weights),
    ) * 0.5  # neutral default

    # --------------------------------------------------------
    # Joint suitability
    # --------------------------------------------------------

    # J_r = (S_r^alpha) * (P_r^beta)
    # Both s and p are normalized to [0,1] before exponentiation

    s_norm = weights["semantic_suitability"]  # already in [0,1] since semantic_score in [0,1]
    p_norm = weights["spectral_suitability"]

    weights["joint_suitability"] = (
        (s_norm ** ALPHA)
        *
        (p_norm ** BETA)
    )

    # --------------------------------------------------------
    # Adaptive embedding strength
    # --------------------------------------------------------

    # delta_r = delta_min + (delta_max - delta_min) * J_r
    weights[
        "adaptive_strength"
    ] = (
        DELTA_MIN
        +
        (DELTA_MAX - DELTA_MIN)
        *
        weights["joint_suitability"]
    )

    return weights


# ============================================================
# EMBED PAYLOAD
# ============================================================

def embed_payload(
    stft,
    payload,
    region_mapping,
    region_weights,
):

    watermarked = stft.clone()

    frequency_bins = stft.shape[0]
    time_frames = stft.shape[1]

    coordinates = []
    embedded_bits = []

    bit_index = 0

    # --------------------------------------------------------
    # Strength lookup
    # --------------------------------------------------------

    strength_lookup = {
        int(row["region_id"]):
            float(row["adaptive_strength"])
        for _, row
        in region_weights.iterrows()
    }

    semantic_lookup = {
        int(row["region_id"]):
            float(row["semantic_score"])
        for _, row
        in region_weights.iterrows()
    }

    # --------------------------------------------------------
    # Traverse semantic regions
    # --------------------------------------------------------

    for _, region in region_mapping.iterrows():

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

        adaptive_strength = (
            strength_lookup[region_id]
        )

        semantic_score = (
            semantic_lookup[region_id]
        )

        # ----------------------------------------------------
        # Safety bounds
        # ----------------------------------------------------

        time_start = max(
            0,
            time_start,
        )

        time_end = min(
            time_frames - 1,
            time_end,
        )

        frequency_start = max(
            0,
            frequency_start,
        )

        frequency_end = min(
            frequency_bins - 1,
            frequency_end,
        )

        # ----------------------------------------------------
        # Embed bits
        # ----------------------------------------------------

        for time_index in range(
            time_start,
            time_end + 1,
        ):

            for frequency_index in range(
                frequency_start,
                frequency_end + 1,
            ):

                if bit_index >= payload.numel():

                    break

                bit = int(
                    payload[
                        bit_index
                    ].item()
                )

                # ------------------------------------------------
                # Binary antipodal representation
                #
                # bit 1 -> +alpha
                # bit 0 -> -alpha
                # ------------------------------------------------

                if bit == 1:

                    delta = (
                        adaptive_strength
                    )

                else:

                    delta = (
                        -adaptive_strength
                    )

                original_value = (
                    watermarked[
                        frequency_index,
                        time_index,
                    ]
                )

                delta_tensor = torch.tensor(
                    delta,
                    dtype=original_value.real.dtype,
                )

                # Modify only the real component.
                watermarked[
                    frequency_index,
                    time_index,
                ] = (
                    original_value
                    +
                    delta_tensor
                )

                embedded_bits.append(
                    bit
                )

                coordinates.append(
                    {
                        "bit_index":
                            bit_index,

                        "payload_bit":
                            bit,

                        "region_id":
                            region_id,

                        "time_frame":
                            time_index,

                        "frequency_bin":
                            frequency_index,

                        "semantic_score":
                            semantic_score,

                        "semantic_suitability":
                            1.0 - semantic_score,

                        "adaptive_strength":
                            adaptive_strength,

                        "delta":
                            delta,
                    }
                )

                bit_index += 1

            if bit_index >= payload.numel():

                break

        if bit_index >= payload.numel():

            break

    return (
        watermarked,
        embedded_bits,
        coordinates,
    )


# ============================================================
# RECONSTRUCT AUDIO
# ============================================================

def reconstruct_audio(
    stft,
):

    stft_numpy = (
        stft
        .detach()
        .cpu()
        .numpy()
    )

    audio = librosa.istft(
        stft_numpy,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_fft=N_FFT,
    )

    return audio.astype(
        np.float32
    )


# ============================================================
# AUDIO QUALITY
# ============================================================

def calculate_quality(
    original,
    watermarked,
):

    length = min(
        len(original),
        len(watermarked),
    )

    original = original[:length]
    watermarked = watermarked[:length]

    difference = (
        watermarked
        -
        original
    )

    mse = float(
        np.mean(
            difference ** 2
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
            *
            np.log10(
                signal_power
                /
                mse
            )
        )

    else:

        snr = float("inf")

    peak_error = float(
        np.max(
            np.abs(
                difference
            )
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

    original_peak = float(
        np.max(
            np.abs(original)
        )
    )

    watermarked_peak = float(
        np.max(
            np.abs(watermarked)
        )
    )

    if (
        np.std(original) > 0
        and
        np.std(watermarked) > 0
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
        "samples":
            int(length),

        "mse":
            mse,

        "rmse":
            rmse,

        "snr_db":
            snr,

        "peak_absolute_error":
            peak_error,

        "original_rms":
            original_rms,

        "watermarked_rms":
            watermarked_rms,

        "original_peak":
            original_peak,

        "watermarked_peak":
            watermarked_peak,

        "waveform_correlation":
            correlation,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "SemBind-Audio : Experiment 13"
    )

    print(
        "Semantic-Adaptive STFT Embedding"
    )

    print("=" * 60)

    # ========================================================
    # LOAD
    # ========================================================

    stft = load_stft()

    payload = load_payload()

    region_mapping = (
        load_region_mapping()
    )

    print()

    print(
        f"STFT Shape              : "
        f"{tuple(stft.shape)}"
    )

    print(
        f"Payload Length          : "
        f"{payload.numel()} bits"
    )

    print(
        f"Candidate Regions       : "
        f"{len(region_mapping)}"
    )

    print(
        f"Baseline Strength       : "
        f"{BASELINE_STRENGTH}"
    )

    print()

    # ========================================================
    # SEMANTIC WEIGHTS
    # ========================================================

    region_weights = (
        calculate_region_weights(
            region_mapping
        )
    )

    # ========================================================
    # SAVE WEIGHTS
    # ========================================================

    region_weights.to_csv(
        REGION_WEIGHTS_FILE,
        index=False,
    )

    # ========================================================
    # DISPLAY WEIGHTS
    # ========================================================

    print(
        "Semantic-Adaptive Region Weights"
    )

    print(
        "-" * 60
    )

    for _, row in (
        region_weights.iterrows()
    ):

        print(
            f"Region "
            f"{int(row['region_id']):02d} | "
            f"Semantic Score: "
            f"{row['semantic_score']:.6f} | "
            f"Suitability: "
            f"{row['semantic_suitability']:.6f} | "
            f"Adaptive Strength: "
            f"{row['adaptive_strength']:.8f}"
        )

    print()

    # ========================================================
    # CAPACITY
    # ========================================================

    available_capacity = int(
        region_mapping[
            "stft_coefficients"
        ].sum()
    )

    print(
        f"Available Capacity      : "
        f"{available_capacity} bits"
    )

    if (
        available_capacity
        <
        payload.numel()
    ):

        raise ValueError(
            "Insufficient embedding "
            "capacity."
        )

    print(
        "Capacity Check          : "
        "SUFFICIENT"
    )

    print()

    # ========================================================
    # EMBEDDING
    # ========================================================

    (
        watermarked_stft,
        embedded_bits,
        coordinates,
    ) = embed_payload(
        stft,
        payload,
        region_mapping,
        region_weights,
    )

    if len(embedded_bits) != payload.numel():

        raise RuntimeError(
            "Payload embedding incomplete.\n"
            f"Expected: {payload.numel()}\n"
            f"Embedded: {len(embedded_bits)}"
        )

    # ========================================================
    # STFT DIFFERENCE
    # ========================================================

    difference = (
        watermarked_stft
        -
        stft
    )

    magnitude_change = (
        torch.abs(
            difference
        )
    )

    embedded_changes = torch.tensor(
        [
            abs(
                record["delta"]
            )
            for record in coordinates
        ],
        dtype=torch.float64,
    )

    print(
        "Embedding Complete"
    )

    print()

    print(
        f"Embedded Bits           : "
        f"{len(embedded_bits)}"
    )

    print(
        f"Embedding Coordinates   : "
        f"{len(coordinates)}"
    )

    print(
        f"Mean Magnitude Change   : "
        f"{magnitude_change.mean().item():.10f}"
    )

    print(
        f"Maximum Magnitude Change: "
        f"{magnitude_change.max().item():.10f}"
    )

    print(
        f"Mean Embedded Change    : "
        f"{embedded_changes.mean().item():.10f}"
    )

    print(
        f"Max Embedded Change     : "
        f"{embedded_changes.max().item():.10f}"
    )

    print()

    # ========================================================
    # SAVE STFT
    # ========================================================

    torch.save(
        stft,
        ORIGINAL_STFT_FILE,
    )

    torch.save(
        watermarked_stft,
        WATERMARKED_STFT_FILE,
    )

    torch.save(
        torch.tensor(
            embedded_bits,
            dtype=torch.int64,
        ),
        EMBEDDED_BITS_FILE,
    )

    # ========================================================
    # SAVE COORDINATES
    # ========================================================

    coordinates_df = pd.DataFrame(
        coordinates
    )

    coordinates_df.to_csv(
        COORDINATES_FILE,
        index=False,
    )

    # ========================================================
    # RECONSTRUCT
    # ========================================================

    original_audio = reconstruct_audio(
        stft
    )

    watermarked_audio = (
        reconstruct_audio(
            watermarked_stft
        )
    )

    watermarked_audio = np.clip(
        watermarked_audio,
        -1.0,
        1.0,
    )

    # ========================================================
    # SAVE AUDIO
    # ========================================================

    sf.write(
        WATERMARKED_AUDIO_FILE,
        watermarked_audio,
        SAMPLE_RATE,
    )

    # ========================================================
    # QUALITY
    # ========================================================

    quality = calculate_quality(
        original_audio,
        watermarked_audio,
    )

    print(
        "Audio Quality"
    )

    print(
        "-" * 60
    )

    print(
        f"MSE                    : "
        f"{quality['mse']:.12e}"
    )

    print(
        f"RMSE                   : "
        f"{quality['rmse']:.12e}"
    )

    print(
        f"SNR                    : "
        f"{quality['snr_db']:.6f} dB"
    )

    print(
        f"Peak Absolute Error    : "
        f"{quality['peak_absolute_error']:.12e}"
    )

    print(
        f"Original RMS           : "
        f"{quality['original_rms']:.12e}"
    )

    print(
        f"Watermarked RMS        : "
        f"{quality['watermarked_rms']:.12e}"
    )

    print(
        f"Waveform Correlation   : "
        f"{quality['waveform_correlation']:.12f}"
    )

    print()

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "experiment":
            "Experiment 13",

        "method":
            "Semantic-Adaptive STFT Embedding",

        "formula": {
            "semantic_suitability":
                "W_r = 1 - S_r",

            "adaptive_strength":
                "alpha_r = alpha_0 * W_r / max(W)",

            "embedding":
                "X_w(f,t) = X(f,t) + alpha_r * B_i",
        },

        "baseline_strength":
            BASELINE_STRENGTH,

        "sample_rate":
            SAMPLE_RATE,

        "n_fft":
            N_FFT,

        "hop_length":
            HOP_LENGTH,

        "win_length":
            WIN_LENGTH,

        "stft_shape":
            list(stft.shape),

        "payload_length":
            int(payload.numel()),

        "candidate_regions":
            int(len(region_mapping)),

        "available_capacity":
            available_capacity,

        "embedded_bits":
            int(len(embedded_bits)),

        "embedding_coordinates":
            int(len(coordinates)),

        "mean_magnitude_change":
            float(
                magnitude_change.mean()
            ),

        "maximum_magnitude_change":
            float(
                magnitude_change.max()
            ),

        "mean_embedded_change":
            float(
                embedded_changes.mean()
            ),

        "maximum_embedded_change":
            float(
                embedded_changes.max()
            ),

        "audio_quality":
            quality,
    }

    with open(
        SUMMARY_FILE,
        "w",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "Results Saved"
    )

    print(
        ORIGINAL_STFT_FILE
    )

    print(
        WATERMARKED_STFT_FILE
    )

    print(
        EMBEDDED_BITS_FILE
    )

    print(
        COORDINATES_FILE
    )

    print(
        REGION_WEIGHTS_FILE
    )

    print(
        WATERMARKED_AUDIO_FILE
    )

    print(
        SUMMARY_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()