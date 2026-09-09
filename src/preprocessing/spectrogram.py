"""
SemBind-Audio

Spectrogram generation utilities.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from configs.config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
    N_MELS,
)


class SpectrogramGenerator:
    """
    Generate different spectrogram representations.
    """

    @staticmethod
    def stft(audio: np.ndarray):

        return librosa.stft(
            audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
        )

    @staticmethod
    def mel(audio: np.ndarray):

        return librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
        )

    @staticmethod
    def log_mel(audio: np.ndarray):

        mel = SpectrogramGenerator.mel(audio)

        return librosa.power_to_db(
            mel,
            ref=np.max,
        )

    @staticmethod
    def save(
        spectrogram,
        output_path,
        title="Spectrogram",
        y_axis="linear",
    ):

        plt.figure(figsize=(10, 4))

        librosa.display.specshow(
            spectrogram,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis=y_axis,
        )

        plt.colorbar(format="%+2.0f dB")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()