import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import soundfile as sf

from src.dataset.audio_dataset import AudioDataset
from src.preprocessing.audio_loader import preprocess
from src.preprocessing.spectrogram import SpectrogramGenerator


dataset = AudioDataset()

sample = dataset.load(0)

audio = preprocess(
    sample["audio"],
    sample["sample_rate"],
)

stft = SpectrogramGenerator.stft(audio)
mel = SpectrogramGenerator.mel(audio)
logmel = SpectrogramGenerator.log_mel(audio)

print("STFT Shape    :", stft.shape)
print("Mel Shape     :", mel.shape)
print("Log-Mel Shape :", logmel.shape)

RESULT_DIR = PROJECT_ROOT / "results"
SPECTROGRAM_DIR = RESULT_DIR / "spectrograms"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SPECTROGRAM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ---------------------------------------------------------
# Save the exact preprocessed audio used for the
# spectrogram and watermarking experiments.
# ---------------------------------------------------------

original_audio_path = (
    RESULT_DIR / "original_audio.wav"
)

sf.write(
    original_audio_path,
    audio,
    sample["sample_rate"],
)

# ---------------------------------------------------------
# Save STFT tensor
# ---------------------------------------------------------

torch.save(
    torch.tensor(
        stft,
        dtype=torch.complex64,
    ),
    SPECTROGRAM_DIR / "stft.pt",
)

# ---------------------------------------------------------
# Save Log-Mel tensor
# ---------------------------------------------------------

torch.save(
    torch.tensor(
        logmel,
        dtype=torch.float32,
    ),
    SPECTROGRAM_DIR
    / "log_mel_spectrogram.pt",
)

# ---------------------------------------------------------
# Save visualization
# ---------------------------------------------------------

SpectrogramGenerator.save(
    logmel,
    RESULT_DIR / "logmel.png",
    title="Log-Mel Spectrogram",
    y_axis="mel",
)

print()
print(
    "Saved:",
    original_audio_path,
)

print(
    "Saved:",
    SPECTROGRAM_DIR / "stft.pt",
)

print(
    "Saved:",
    SPECTROGRAM_DIR
    / "log_mel_spectrogram.pt",
)

print(
    "Saved:",
    RESULT_DIR / "logmel.png",
)