"""
Experiment 01

HuBERT Embedding Extraction
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch

from src.dataset.audio_dataset import AudioDataset
from src.preprocessing.audio_loader import preprocess
from src.embedding.hubert_embedding import HuBERTEmbedding


RESULT_DIR = PROJECT_ROOT / "results" / "embeddings"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 60)
    print("SemBind-Audio : Experiment 01")
    print("HuBERT Embedding Extraction")
    print("=" * 60)

    dataset = AudioDataset()

    sample = dataset.load(0)

    audio = preprocess(
        sample["audio"],
        sample["sample_rate"],
    )

    extractor = HuBERTEmbedding()

    embeddings = extractor.extract(audio)

    print(f"Embedding Shape : {embeddings.shape}")
    print(f"Frames          : {embeddings.shape[0]}")
    print(f"Dimension       : {embeddings.shape[1]}")

    torch.save(
        embeddings.cpu(),
        RESULT_DIR / "hubert_embeddings.pt",
    )

    summary = {
        "frames": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "device": str(embeddings.device),
        "dtype": str(embeddings.dtype),
    }

    with open(
        RESULT_DIR / "embedding_summary.json",
        "w",
    ) as f:
        json.dump(summary, f, indent=4)

    print()
    print("Saved:")
    print(RESULT_DIR / "hubert_embeddings.pt")
    print(RESULT_DIR / "embedding_summary.json")


if __name__ == "__main__":
    main()