import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.audio_dataset import AudioDataset

dataset = AudioDataset()

dataset.info()

sample = dataset.load(0)

print()

for key, value in sample.items():

    if key == "audio":
        print(f"{key:15}: shape={value.shape}")

    else:
        print(f"{key:15}: {value}")