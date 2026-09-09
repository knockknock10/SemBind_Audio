import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.dataset.audio_dataset import AudioDataset
from src.preprocessing.audio_loader import preprocess

dataset = AudioDataset()

sample = dataset.load(0)

audio = preprocess(
    sample["audio"],
    sample["sample_rate"]
)

print("Original Length :", len(sample["audio"]))
print("Processed Length:", len(audio))
print("Maximum Value   :", audio.max())
print("Minimum Value   :", audio.min())