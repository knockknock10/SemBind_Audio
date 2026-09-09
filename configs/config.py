"""
Global configuration for SemBind-Audio.
"""

from pathlib import Path

# Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Audio

SAMPLE_RATE = 16000

# Spectrogram

N_FFT = 1024
HOP_LENGTH = 320
WIN_LENGTH = 1024
N_MELS = 128

# Models

HUBERT_MODEL = "facebook/hubert-base-ls960"

# STFT embedding configuration

# Joint suitability exponents
ALPHA = 0.70
BETA = 0.30

# Adaptive embedding strength range
DELTA_MIN = 0.002
DELTA_MAX = 0.008

# Payload configuration
PAYLOAD_BITS = 256
REDUNDANCY = 3

# HuBERT configuration
HUBERT_EMBEDDING_DIM = 768