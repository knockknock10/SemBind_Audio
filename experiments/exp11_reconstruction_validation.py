"""
SemBind-Audio

Experiment 11
STFT Reconstruction Validation

Purpose:
Validate that the original STFT can be converted back
to the original waveform using the exact same STFT
configuration used during spectrogram generation.

This experiment contains NO watermark modification.

Therefore, any reconstruction error observed here belongs
to the STFT/ISTFT pipeline rather than watermark embedding.
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
import torch
import soundfile as sf
import librosa


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

SPECTROGRAM_DIR = (
    RESULTS_DIR
    / "spectrograms"
)

QUALITY_DIR = (
    RESULTS_DIR
    / "audio_quality"
)

ORIGINAL_AUDIO_FILE = (
    RESULTS_DIR
    / "original_audio.wav"
)

STFT_FILE = (
    SPECTROGRAM_DIR
    / "stft.pt"
)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)

    print(
        "SemBind-Audio : "
        "STFT Reconstruction Validation"
    )

    print("=" * 60)

    # -----------------------------------------------------
    # Create output directory
    # -----------------------------------------------------

    QUALITY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Validate files
    # -----------------------------------------------------

    if not ORIGINAL_AUDIO_FILE.exists():

        raise FileNotFoundError(
            "Original audio not found:\n"
            f"{ORIGINAL_AUDIO_FILE}"
        )

    if not STFT_FILE.exists():

        raise FileNotFoundError(
            "STFT file not found:\n"
            f"{STFT_FILE}"
        )

    # -----------------------------------------------------
    # Load original audio
    # -----------------------------------------------------

    original_audio, original_sample_rate = (
        sf.read(
            ORIGINAL_AUDIO_FILE,
            always_2d=False,
        )
    )

    if original_audio.ndim > 1:

        original_audio = np.mean(
            original_audio,
            axis=1,
        )

    original_audio = original_audio.astype(
        np.float64
    )

    if original_sample_rate != SAMPLE_RATE:

        raise ValueError(
            "Sample-rate mismatch.\n"
            f"Expected: {SAMPLE_RATE}\n"
            f"Found: {original_sample_rate}"
        )

    # -----------------------------------------------------
    # Load STFT
    # -----------------------------------------------------

    stft = torch.load(
        STFT_FILE,
        map_location="cpu",
    )

    stft = (
        stft
        .detach()
        .cpu()
        .numpy()
    )

    if not np.iscomplexobj(stft):

        raise TypeError(
            "The saved STFT must be complex-valued."
        )

    # -----------------------------------------------------
    # Print configuration
    # -----------------------------------------------------

    print()

    print(
        f"Original samples       : "
        f"{len(original_audio)}"
    )

    print(
        f"STFT Shape             : "
        f"{stft.shape}"
    )

    print()

    print(
        "STFT Configuration"
    )

    print(
        f"N_FFT                  : "
        f"{N_FFT}"
    )

    print(
        f"HOP_LENGTH             : "
        f"{HOP_LENGTH}"
    )

    print(
        f"WIN_LENGTH             : "
        f"{WIN_LENGTH}"
    )

    print(
        f"Sample Rate             : "
        f"{SAMPLE_RATE}"
    )

    # -----------------------------------------------------
    # Reconstruct waveform
    # -----------------------------------------------------

    reconstructed_audio = librosa.istft(
        stft,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH,

        win_length=WIN_LENGTH,

        window="hann",

        center=True,

        length=len(original_audio),
    )

    reconstructed_audio = (
        reconstructed_audio
        .astype(np.float64)
    )

    # -----------------------------------------------------
    # Validate waveform
    # -----------------------------------------------------

    if not np.all(
        np.isfinite(
            reconstructed_audio
        )
    ):

        raise RuntimeError(
            "Reconstructed waveform contains "
            "NaN or Inf values."
        )

    # -----------------------------------------------------
    # Calculate difference
    # -----------------------------------------------------

    difference = (
        reconstructed_audio
        -
        original_audio
    )

    # -----------------------------------------------------
    # MSE
    # -----------------------------------------------------

    mse = float(
        np.mean(
            difference ** 2
        )
    )

    # -----------------------------------------------------
    # RMSE
    # -----------------------------------------------------

    rmse = float(
        np.sqrt(
            mse
        )
    )

    # -----------------------------------------------------
    # Signal power
    # -----------------------------------------------------

    signal_power = float(
        np.mean(
            original_audio ** 2
        )
    )

    # -----------------------------------------------------
    # Noise power
    # -----------------------------------------------------

    noise_power = float(
        np.mean(
            difference ** 2
        )
    )

    # -----------------------------------------------------
    # SNR
    # -----------------------------------------------------

    if noise_power == 0:

        snr = float("inf")

    else:

        snr = float(
            10.0
            * np.log10(
                signal_power
                /
                noise_power
            )
        )

    # -----------------------------------------------------
    # Peak error
    # -----------------------------------------------------

    peak_absolute_error = float(
        np.max(
            np.abs(
                difference
            )
        )
    )

    # -----------------------------------------------------
    # Correlation
    # -----------------------------------------------------

    if (
        np.std(original_audio) == 0
        or
        np.std(reconstructed_audio) == 0
    ):

        correlation = 0.0

    else:

        correlation = float(
            np.corrcoef(
                original_audio,
                reconstructed_audio,
            )[0, 1]
        )

    # -----------------------------------------------------
    # RMS
    # -----------------------------------------------------

    original_rms = float(
        np.sqrt(
            np.mean(
                original_audio ** 2
            )
        )
    )

    reconstructed_rms = float(
        np.sqrt(
            np.mean(
                reconstructed_audio ** 2
            )
        )
    )

    # -----------------------------------------------------
    # Peak values
    # -----------------------------------------------------

    original_peak = float(
        np.max(
            np.abs(
                original_audio
            )
        )
    )

    reconstructed_peak = float(
        np.max(
            np.abs(
                reconstructed_audio
            )
        )
    )

    # -----------------------------------------------------
    # Save reconstructed waveform
    # -----------------------------------------------------

    reconstructed_audio_file = (
        QUALITY_DIR
        / "stft_reconstructed_audio.wav"
    )

    sf.write(
        reconstructed_audio_file,
        reconstructed_audio.astype(
            np.float32
        ),
        SAMPLE_RATE,
    )

    # -----------------------------------------------------
    # Save difference waveform
    # -----------------------------------------------------

    difference_file = (
        QUALITY_DIR
        / "stft_reconstruction_difference.wav"
    )

    sf.write(
        difference_file,
        difference.astype(
            np.float32
        ),
        SAMPLE_RATE,
    )

    # -----------------------------------------------------
    # Save summary
    # -----------------------------------------------------

    summary = {

        "experiment":
            "STFT Reconstruction Validation",

        "purpose":
            "Validate STFT-to-waveform reconstruction "
            "before evaluating watermark distortion.",

        "original_samples":
            int(
                len(original_audio)
            ),

        "reconstructed_samples":
            int(
                len(reconstructed_audio)
            ),

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
                stft.shape
            ),

        "mse":
            mse,

        "rmse":
            rmse,

        "snr_db":
            snr,

        "peak_absolute_error":
            peak_absolute_error,

        "correlation":
            correlation,

        "original_rms":
            original_rms,

        "reconstructed_rms":
            reconstructed_rms,

        "original_peak":
            original_peak,

        "reconstructed_peak":
            reconstructed_peak,

        "reconstruction_method":
            "librosa.istft",

        "window":
            "hann",

        "center":
            True,
    }

    summary_file = (
        QUALITY_DIR
        / "stft_reconstruction_validation.json"
    )

    with open(
        summary_file,
        "w",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    print()

    print(
        "STFT Reconstruction Quality"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"MSE                  : "
        f"{mse:.12e}"
    )

    print(
        f"RMSE                 : "
        f"{rmse:.12e}"
    )

    print(
        f"SNR                  : "
        f"{snr:.6f} dB"
    )

    print(
        f"Peak Absolute Error  : "
        f"{peak_absolute_error:.12e}"
    )

    print(
        f"Correlation          : "
        f"{correlation:.12f}"
    )

    print(
        f"Original RMS         : "
        f"{original_rms:.12e}"
    )

    print(
        f"Reconstructed RMS    : "
        f"{reconstructed_rms:.12e}"
    )

    print(
        f"Original Peak        : "
        f"{original_peak:.12e}"
    )

    print(
        f"Reconstructed Peak   : "
        f"{reconstructed_peak:.12e}"
    )

    print()

    print(
        "Results Saved"
    )

    print(
        reconstructed_audio_file
    )

    print(
        difference_file
    )

    print(
        summary_file
    )


if __name__ == "__main__":
    main()