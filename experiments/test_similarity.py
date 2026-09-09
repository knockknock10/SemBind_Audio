"""
SemBind-Audio

Test Semantic Similarity Module
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import torch

from src.dataset.audio_dataset import AudioDataset
from src.preprocessing.audio_loader import preprocess
from src.embedding.hubert_embedding import HuBERTEmbedding
from src.semantic.similarity import SemanticSimilarity


def main():

    print("=" * 60)
    print("SemBind-Audio : Semantic Similarity Test")
    print("=" * 60)

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    dataset = AudioDataset()

    sample = dataset.load(0)

    # --------------------------------------------------
    # Preprocess Audio
    # --------------------------------------------------

    audio = preprocess(
        sample["audio"],
        sample["sample_rate"],
    )

    # --------------------------------------------------
    # Generate HuBERT Embeddings
    # --------------------------------------------------

    extractor = HuBERTEmbedding()

    embeddings = extractor.extract(audio)

    # --------------------------------------------------
    # Compute Semantic Similarity
    # --------------------------------------------------

    similarity = SemanticSimilarity(embeddings)

    results = similarity.compute()

    global_embedding = results["global_embedding"]
    raw_scores = results["raw_scores"]
    normalized_scores = results["normalized_scores"]

    # --------------------------------------------------
    # Print Results
    # --------------------------------------------------

    print("\nEmbedding Information")
    print("-" * 60)

    print(f"Embedding Shape      : {embeddings.shape}")
    print(f"Frames               : {embeddings.shape[0]}")
    print(f"Embedding Dimension  : {embeddings.shape[1]}")

    print("\nGlobal Embedding")
    print("-" * 60)

    print(f"Shape                : {global_embedding.shape}")

    print("\nRaw Similarity Scores")
    print("-" * 60)

    print(f"Shape                : {raw_scores.shape}")
    print(f"Minimum              : {torch.min(raw_scores):.4f}")
    print(f"Maximum              : {torch.max(raw_scores):.4f}")
    print(f"Mean                 : {torch.mean(raw_scores):.4f}")

    print("\nNormalized Similarity Scores")
    print("-" * 60)

    print(f"Shape                : {normalized_scores.shape}")
    print(f"Minimum              : {torch.min(normalized_scores):.4f}")
    print(f"Maximum              : {torch.max(normalized_scores):.4f}")
    print(f"Mean                 : {torch.mean(normalized_scores):.4f}")

    print("\nFirst 10 Semantic Scores")
    print("-" * 60)

    for i in range(10):
        print(
            f"Frame {i + 1:03d} : "
            f"{normalized_scores[i].item():.4f}"
        )

    print("\nTop 10 Most Important Frames")
    print("-" * 60)

    values, indices = torch.topk(normalized_scores, k=10)

    for rank, (idx, value) in enumerate(zip(indices, values), start=1):
        print(
            f"{rank:02d}. "
            f"Frame {idx.item()+1:03d} "
            f"Score = {value.item():.4f}"
        )

    print("\nLeast 10 Important Frames")
    print("-" * 60)

    values, indices = torch.topk(
        normalized_scores,
        k=10,
        largest=False
    )

    for rank, (idx, value) in enumerate(zip(indices, values), start=1):
        print(
            f"{rank:02d}. "
            f"Frame {idx.item()+1:03d} "
            f"Score = {value.item():.4f}"
        )


if __name__ == "__main__":
    main()