"""
SemBind-Audio
Experiment 09

Semantic-Guided STFT Watermark Embedding

This experiment:
1. Loads the original project STFT.
2. Loads the 256-bit watermark payload.
3. Loads semantically selected STFT regions.
4. Embeds the payload into the selected regions.
5. Preserves the phase of each STFT coefficient.
6. Reconstructs the watermarked waveform using the
   exact STFT configuration used by SemBind-Audio.
7. Saves all intermediate and final results.
"""

from pathlib import Path
import sys
import json


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# LIBRARIES
# =========================================================

import numpy as np
import pandas as pd
import torch
import librosa
import soundfile as sf


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

from configs.config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
)


# =========================================================
# PATHS
# =========================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

WATERMARK_DIR = (
    RESULTS_DIR
    / "watermark"
)

SPECTROGRAM_DIR = (
    RESULTS_DIR
    / "spectrograms"
)

STFT_MAPPING_DIR = (
    RESULTS_DIR
    / "stft_mapping"
)

EMBEDDING_DIR = (
    RESULTS_DIR
    / "watermark_embedding"
)


PAYLOAD_FILE = (
    WATERMARK_DIR
    / "payload_bits.pt"
)

STFT_FILE = (
    SPECTROGRAM_DIR
    / "stft.pt"
)

REGION_FILE = (
    STFT_MAPPING_DIR
    / "stft_region_mapping.csv"
)

ORIGINAL_AUDIO_FILE = (
    RESULTS_DIR
    / "original_audio.wav"
)


# =========================================================
# WATERMARK CONFIGURATION
# =========================================================

EMBEDDING_STRENGTH = 0.001


# =========================================================
# BIT EMBEDDING
# =========================================================

def embed_bit(
    coefficient: torch.Tensor,
    bit: int,
    strength: float,
) -> torch.Tensor:
    """
    Embed one binary bit into the magnitude of a
    complex STFT coefficient.

    Phase is preserved.

    bit = 0:
        magnitude * (1 - strength)

    bit = 1:
        magnitude * (1 + strength)
    """

    magnitude = torch.abs(
        coefficient
    )

    phase = torch.angle(
        coefficient
    )

    if magnitude.item() == 0.0:
        return coefficient

    if bit == 0:

        new_magnitude = (
            magnitude
            * (1.0 - strength)
        )

    else:

        new_magnitude = (
            magnitude
            * (1.0 + strength)
        )

    return torch.polar(
        new_magnitude,
        phase,
    )


# =========================================================
# LOAD PAYLOAD
# =========================================================

def load_payload():
    """
    Load the 256-bit watermark payload.
    """

    if not PAYLOAD_FILE.exists():

        raise FileNotFoundError(
            "Payload file not found:\n"
            f"{PAYLOAD_FILE}"
        )

    payload = torch.load(
        PAYLOAD_FILE,
        map_location="cpu",
    )

    payload = (
        payload
        .flatten()
        .to(torch.uint8)
    )

    if payload.numel() == 0:

        raise ValueError(
            "Watermark payload is empty."
        )

    invalid_bits = payload[
        (payload != 0)
        & (payload != 1)
    ]

    if invalid_bits.numel() > 0:

        raise ValueError(
            "Payload contains values other "
            "than binary 0/1."
        )

    return payload


# =========================================================
# LOAD STFT
# =========================================================

def load_stft():
    """
    Load the original complex STFT.
    """

    if not STFT_FILE.exists():

        raise FileNotFoundError(
            "STFT file not found:\n"
            f"{STFT_FILE}"
        )

    stft = torch.load(
        STFT_FILE,
        map_location="cpu",
    )

    stft = (
        stft
        .detach()
        .cpu()
    )

    if not torch.is_complex(stft):

        raise TypeError(
            "The saved STFT must be "
            "complex-valued."
        )

    if stft.ndim != 2:

        raise ValueError(
            "Expected a 2D STFT tensor "
            "(frequency, time).\n"
            f"Received: {tuple(stft.shape)}"
        )

    return stft


# =========================================================
# LOAD ORIGINAL AUDIO
# =========================================================

def load_original_audio():
    """
    Load the exact preprocessed audio used for
    STFT generation.
    """

    if not ORIGINAL_AUDIO_FILE.exists():

        raise FileNotFoundError(
            "Original audio not found:\n"
            f"{ORIGINAL_AUDIO_FILE}"
        )

    audio, sample_rate = sf.read(
        ORIGINAL_AUDIO_FILE,
        always_2d=False,
    )

    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1,
        )

    audio = audio.astype(
        np.float32
    )

    if sample_rate != SAMPLE_RATE:

        raise ValueError(
            "Sample-rate mismatch.\n"
            f"Expected: {SAMPLE_RATE}\n"
            f"Found: {sample_rate}"
        )

    return audio, sample_rate


# =========================================================
# LOAD REGION MAPPING
# =========================================================

def load_regions():
    """
    Load semantic-guided STFT embedding regions.
    """

    if not REGION_FILE.exists():

        raise FileNotFoundError(
            "STFT region mapping not found:\n"
            f"{REGION_FILE}"
        )

    regions = pd.read_csv(
        REGION_FILE
    )

    if regions.empty:

        raise ValueError(
            "No STFT embedding regions found."
        )

    required_columns = [
        "region_id",
        "time_start_frame",
        "time_end_frame",
        "frequency_start_bin",
        "frequency_end_bin",
    ]

    missing = [
        column
        for column in required_columns
        if column not in regions.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns "
            f"in STFT mapping: {missing}"
        )

    return regions


# =========================================================
# GENERATE EMBEDDING COORDINATES
# =========================================================

def generate_embedding_coordinates(
    regions,
    frequency_bins,
    time_frames,
):
    """
    Convert candidate STFT regions into individual
    coefficient coordinates.

    Each coefficient provides capacity for one bit.
    """

    coordinates = []

    for _, region in regions.iterrows():

        region_id = int(
            region[
                "region_id"
            ]
        )

        start_time = int(
            region[
                "time_start_frame"
            ]
        )

        end_time = int(
            region[
                "time_end_frame"
            ]
        )

        start_frequency = int(
            region[
                "frequency_start_bin"
            ]
        )

        end_frequency = int(
            region[
                "frequency_end_bin"
            ]
        )

        for time_frame in range(
            start_time,
            end_time + 1,
        ):

            # Region mapping uses one-based frame
            # numbering.
            time_index = (
                time_frame - 1
            )

            if (
                time_index < 0
                or time_index >= time_frames
            ):
                continue

            for frequency_bin in range(
                start_frequency,
                end_frequency + 1,
            ):

                frequency_index = (
                    frequency_bin
                )

                if (
                    frequency_index < 0
                    or frequency_index >= frequency_bins
                ):
                    continue

                coordinates.append(
                    {
                        "region_id":
                            region_id,

                        "frequency_bin":
                            frequency_index,

                        "time_frame":
                            time_frame,

                        "time_index":
                            time_index,
                    }
                )

    return coordinates


# =========================================================
# EMBED PAYLOAD
# =========================================================

def embed_payload(
    original_stft,
    payload,
    coordinates,
):
    """
    Embed the complete watermark payload.
    """

    watermarked_stft = (
        original_stft.clone()
    )

    embedding_records = []

    for bit_index, coordinate in enumerate(
        coordinates[
            :payload.numel()
        ]
    ):

        bit = int(
            payload[
                bit_index
            ].item()
        )

        frequency_bin = int(
            coordinate[
                "frequency_bin"
            ]
        )

        time_index = int(
            coordinate[
                "time_index"
            ]
        )

        region_id = int(
            coordinate[
                "region_id"
            ]
        )

        time_frame = int(
            coordinate[
                "time_frame"
            ]
        )

        original_value = (
            watermarked_stft[
                frequency_bin,
                time_index,
            ]
        )

        modified_value = embed_bit(
            original_value,
            bit,
            EMBEDDING_STRENGTH,
        )

        watermarked_stft[
            frequency_bin,
            time_index,
        ] = modified_value

        embedding_records.append(
            {
                "bit_index":
                    bit_index + 1,

                "bit":
                    bit,

                "region_id":
                    region_id,

                "frequency_bin":
                    frequency_bin,

                "time_frame":
                    time_frame,

                "time_index":
                    time_index,

                "original_magnitude":
                    float(
                        torch.abs(
                            original_value
                        )
                    ),

                "modified_magnitude":
                    float(
                        torch.abs(
                            modified_value
                        )
                    ),

                "magnitude_change":
                    float(
                        torch.abs(
                            torch.abs(
                                modified_value
                            )
                            -
                            torch.abs(
                                original_value
                            )
                        )
                    ),
            }
        )

    return (
        watermarked_stft,
        embedding_records,
    )


# =========================================================
# RECONSTRUCT AUDIO
# =========================================================

def reconstruct_audio(
    watermarked_stft,
    target_length,
):
    """
    Reconstruct waveform using the exact configuration
    used by the SemBind-Audio STFT generator.
    """

    stft_numpy = (
        watermarked_stft
        .detach()
        .cpu()
        .numpy()
    )

    waveform = librosa.istft(
        stft_numpy,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH,

        win_length=WIN_LENGTH,

        window="hann",

        center=True,

        length=target_length,
    )

    waveform = waveform.astype(
        np.float32
    )

    if not np.all(
        np.isfinite(waveform)
    ):

        raise RuntimeError(
            "Reconstructed waveform contains "
            "NaN or Inf values."
        )

    return waveform


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)

    print(
        "SemBind-Audio : Experiment 09"
    )

    print(
        "Semantic-Guided STFT Watermark Embedding"
    )

    print("=" * 60)

    # -----------------------------------------------------
    # Create output directory
    # -----------------------------------------------------

    EMBEDDING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    payload = load_payload()

    original_stft = load_stft()

    regions = load_regions()

    original_audio, sample_rate = (
        load_original_audio()
    )

    # -----------------------------------------------------
    # STFT information
    # -----------------------------------------------------

    frequency_bins = int(
        original_stft.shape[0]
    )

    time_frames = int(
        original_stft.shape[1]
    )

    payload_length = int(
        payload.numel()
    )

    # -----------------------------------------------------
    # Print experiment configuration
    # -----------------------------------------------------

    print()

    print(
        f"STFT Shape          : "
        f"{tuple(original_stft.shape)}"
    )

    print(
        f"Payload Length      : "
        f"{payload_length} bits"
    )

    print(
        f"Candidate Regions   : "
        f"{len(regions)}"
    )

    print(
        f"Embedding Strength  : "
        f"{EMBEDDING_STRENGTH}"
    )

    print(
        f"Sample Rate         : "
        f"{sample_rate}"
    )

    print()

    print(
        "STFT Configuration"
    )

    print(
        f"N_FFT               : "
        f"{N_FFT}"
    )

    print(
        f"HOP_LENGTH          : "
        f"{HOP_LENGTH}"
    )

    print(
        f"WIN_LENGTH          : "
        f"{WIN_LENGTH}"
    )

    print()

    # -----------------------------------------------------
    # Generate available coefficient coordinates
    # -----------------------------------------------------

    coordinates = (
        generate_embedding_coordinates(
            regions,
            frequency_bins,
            time_frames,
        )
    )

    total_capacity = len(
        coordinates
    )

    print(
        f"Available Capacity  : "
        f"{total_capacity} bits"
    )

    if total_capacity < payload_length:

        raise RuntimeError(
            "Insufficient embedding capacity.\n"
            f"Required: {payload_length}\n"
            f"Available: {total_capacity}"
        )

    print(
        "Capacity Check      : SUFFICIENT"
    )

    print()

    # -----------------------------------------------------
    # Embed watermark
    # -----------------------------------------------------

    (
        watermarked_stft,
        embedding_records,
    ) = embed_payload(
        original_stft,
        payload,
        coordinates,
    )

    embedded_bits = len(
        embedding_records
    )

    if embedded_bits != payload_length:

        raise RuntimeError(
            "Payload was not completely embedded.\n"
            f"Required: {payload_length}\n"
            f"Embedded: {embedded_bits}"
        )

    # -----------------------------------------------------
    # Save STFTs
    # -----------------------------------------------------

    torch.save(
        original_stft,
        EMBEDDING_DIR
        / "original_stft.pt",
    )

    torch.save(
        watermarked_stft,
        EMBEDDING_DIR
        / "watermarked_stft.pt",
    )

    # -----------------------------------------------------
    # Save payload
    # -----------------------------------------------------

    torch.save(
        payload,
        EMBEDDING_DIR
        / "embedded_bits.pt",
    )

    # -----------------------------------------------------
    # Save coordinates
    # -----------------------------------------------------

    embedding_df = pd.DataFrame(
        embedding_records
    )

    embedding_df.to_csv(
        EMBEDDING_DIR
        / "embedding_coordinates.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Reconstruct watermarked audio
    # -----------------------------------------------------

    watermarked_audio = (
        reconstruct_audio(
            watermarked_stft,
            len(original_audio),
        )
    )

    # -----------------------------------------------------
    # Check waveform peak
    # -----------------------------------------------------

    watermarked_peak = float(
        np.max(
            np.abs(
                watermarked_audio
            )
        )
    )

    clipping_scaling_applied = False

    if watermarked_peak > 1.0:

        watermarked_audio = (
            watermarked_audio
            / watermarked_peak
        )

        clipping_scaling_applied = True

    # -----------------------------------------------------
    # Save watermarked audio
    # -----------------------------------------------------

    watermarked_audio_file = (
        EMBEDDING_DIR
        / "watermarked_audio.wav"
    )

    sf.write(
        watermarked_audio_file,
        watermarked_audio,
        SAMPLE_RATE,
    )

    # -----------------------------------------------------
    # Calculate STFT magnitude changes
    # -----------------------------------------------------

    original_magnitude = torch.abs(
        original_stft
    )

    watermarked_magnitude = torch.abs(
        watermarked_stft
    )

    magnitude_difference = (
        watermarked_magnitude
        -
        original_magnitude
    )

    mean_magnitude_change = float(
        torch.mean(
            torch.abs(
                magnitude_difference
            )
        )
    )

    maximum_magnitude_change = float(
        torch.max(
            torch.abs(
                magnitude_difference
            )
        )
    )

    # -----------------------------------------------------
    # Calculate changes only at embedded positions
    # -----------------------------------------------------

    embedded_changes = []

    for record in embedding_records:

        frequency_bin = int(
            record[
                "frequency_bin"
            ]
        )

        time_index = int(
            record[
                "time_index"
            ]
        )

        original_value = (
            original_stft[
                frequency_bin,
                time_index,
            ]
        )

        modified_value = (
            watermarked_stft[
                frequency_bin,
                time_index,
            ]
        )

        change = float(
            torch.abs(
                torch.abs(
                    modified_value
                )
                -
                torch.abs(
                    original_value
                )
            )
        )

        embedded_changes.append(
            change
        )

    mean_embedded_change = float(
        np.mean(
            embedded_changes
        )
    )

    maximum_embedded_change = float(
        np.max(
            embedded_changes
        )
    )

    # -----------------------------------------------------
    # Save summary
    # -----------------------------------------------------

    summary = {

        "experiment":
            "Experiment 09",

        "description":
            "Semantic-guided STFT watermark embedding",

        "status":
            "success",

        "payload_length_bits":
            payload_length,

        "embedded_bits":
            embedded_bits,

        "candidate_regions":
            int(
                len(regions)
            ),

        "available_capacity_bits":
            total_capacity,

        "embedding_coordinates":
            embedded_bits,

        "embedding_strength":
            EMBEDDING_STRENGTH,

        "stft_shape":
            list(
                original_stft.shape
            ),

        "frequency_bins":
            frequency_bins,

        "time_frames":
            time_frames,

        "sample_rate":
            SAMPLE_RATE,

        "n_fft":
            N_FFT,

        "hop_length":
            HOP_LENGTH,

        "win_length":
            WIN_LENGTH,

        "mean_stft_magnitude_change":
            mean_magnitude_change,

        "maximum_stft_magnitude_change":
            maximum_magnitude_change,

        "mean_embedded_magnitude_change":
            mean_embedded_change,

        "maximum_embedded_magnitude_change":
            maximum_embedded_change,

        "original_audio_samples":
            int(
                len(original_audio)
            ),

        "watermarked_audio_samples":
            int(
                len(watermarked_audio)
            ),

        "watermarked_audio_peak_before_scaling":
            watermarked_peak,

        "clipping_scaling_applied":
            clipping_scaling_applied,

        "reconstruction":
            "librosa.istft",

        "stft_representation":
            "one-sided complex STFT",
    }

    with open(
        EMBEDDING_DIR
        / "embedding_summary.json",
        "w",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print()

    print(
        "Embedding Complete"
    )

    print()

    print(
        f"Embedded Bits           : "
        f"{embedded_bits}"
    )

    print(
        f"Embedding Coordinates   : "
        f"{embedded_bits}"
    )

    print(
        f"Mean Magnitude Change   : "
        f"{mean_magnitude_change:.10f}"
    )

    print(
        f"Maximum Magnitude Change: "
        f"{maximum_magnitude_change:.10f}"
    )

    print(
        f"Mean Embedded Change    : "
        f"{mean_embedded_change:.10f}"
    )

    print(
        f"Max Embedded Change     : "
        f"{maximum_embedded_change:.10f}"
    )

    print()

    print(
        "Results Saved"
    )

    print(
        EMBEDDING_DIR
        / "original_stft.pt"
    )

    print(
        EMBEDDING_DIR
        / "watermarked_stft.pt"
    )

    print(
        EMBEDDING_DIR
        / "embedded_bits.pt"
    )

    print(
        EMBEDDING_DIR
        / "embedding_coordinates.csv"
    )

    print(
        watermarked_audio_file
    )

    print(
        EMBEDDING_DIR
        / "embedding_summary.json"
    )


if __name__ == "__main__":
    main()