"""
SemBind-Audio

Audio preprocessing utilities.

Parameters are centralized in configs.config.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from configs.config import SAMPLE_RATE


def to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono.

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform. Shape: (samples,) or (channels, samples).

    Returns
    -------
    np.ndarray
        Mono audio. Shape: (samples,).
    """
    if audio.ndim == 2:
        return np.mean(audio, axis=1)
    return audio


def normalize(
    audio: np.ndarray,
    target_peak: float = 1.0,
) -> np.ndarray:
    """
    Normalize audio to the range [-target_peak, target_peak].

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform.
    target_peak : float
        Target peak amplitude. Default 1.0.

    Returns
    -------
    np.ndarray
        Normalized audio.
    """
    peak = float(np.max(np.abs(audio)))

    if peak == 0:
        return audio

    return audio * (target_peak / peak)


def validate_sample_rate(sample_rate: int):
    """
    Ensure the sample rate matches the project configuration.

    Parameters
    ----------
    sample_rate : int
        Sample rate to validate.

    Raises
    ------
    ValueError
        If sample rate does not match SAMPLE_RATE config.
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected sample rate {SAMPLE_RATE}, got {sample_rate}"
        )


def preprocess(
    audio: np.ndarray,
    sample_rate: int,
    *,
    apply_normalization: bool = True,
) -> np.ndarray:
    """
    Complete preprocessing pipeline.

    Parameters
    ----------
    audio : np.ndarray
        Raw audio waveform.
    sample_rate : int
        Audio sample rate.
    apply_normalization : bool
        If True, normalize audio to [-1, 1] peak range.
        If False, only convert to mono and resample (if needed).
        This parameter is central in config to avoid accidental
        normalization that changes experimental interpretation.

    Returns
    -------
    np.ndarray
        Preprocessed audio waveform, mono, resampled if needed,
        optionally normalized.
    """
    validate_sample_rate(sample_rate)

    audio = to_mono(audio)

    if apply_normalization:
        audio = normalize(audio, target_peak=1.0)

    return audio