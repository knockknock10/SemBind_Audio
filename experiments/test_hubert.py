import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.audio_dataset import AudioDataset
from src.preprocessing.audio_loader import preprocess
from src.embedding.hubert_embedding import HuBERTEmbedding

dataset = AudioDataset()

sample = dataset.load(0)

audio = preprocess(
    sample["audio"],
    sample["sample_rate"],
)

extractor = HuBERTEmbedding()

embedding = extractor.extract(audio)

print()

print("Embedding Shape :", embedding.shape)
print("Feature Dimension :", embedding.shape[1])
print("Frames            :", embedding.shape[0])