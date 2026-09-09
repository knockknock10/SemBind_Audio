from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd

from src.semantic.similarity import SemanticSimilarity


RESULT_DIR = PROJECT_ROOT / "results" / "similarity"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_FILE = (
    PROJECT_ROOT
    / "results"
    / "embeddings"
    / "hubert_embeddings.pt"
)


def main():

    print("=" * 60)
    print("SemBind-Audio : Experiment 02")
    print("Semantic Similarity")
    print("=" * 60)

    embeddings = torch.load(
        EMBEDDING_FILE,
        map_location="cpu",
    )

    similarity = SemanticSimilarity(
        embeddings
    )

    results = similarity.compute()

    global_embedding = results["global_embedding"]
    raw_scores = results["raw_scores"]
    normalized_scores = results["normalized_scores"]

    torch.save(
        global_embedding,
        RESULT_DIR / "global_embedding.pt",
    )

    torch.save(
        raw_scores,
        RESULT_DIR / "raw_scores.pt",
    )

    torch.save(
        normalized_scores,
        RESULT_DIR / "normalized_scores.pt",
    )

    pd.DataFrame(
        global_embedding.numpy(),
        columns=["value"],
    ).to_csv(
        RESULT_DIR / "global_embedding.csv",
        index=False,
    )

    pd.DataFrame(
        raw_scores.numpy(),
        columns=["raw_similarity"],
    ).to_csv(
        RESULT_DIR / "raw_scores.csv",
        index=False,
    )

    pd.DataFrame(
        normalized_scores.numpy(),
        columns=["normalized_similarity"],
    ).to_csv(
        RESULT_DIR / "normalized_scores.csv",
        index=False,
    )

    summary = {
        "frames": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "raw_min": float(torch.min(raw_scores)),
        "raw_max": float(torch.max(raw_scores)),
        "raw_mean": float(torch.mean(raw_scores)),
        "normalized_min": float(torch.min(normalized_scores)),
        "normalized_max": float(torch.max(normalized_scores)),
        "normalized_mean": float(torch.mean(normalized_scores)),
    }

    with open(
        RESULT_DIR / "semantic_similarity.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    print()

    print(f"Embedding Shape : {embeddings.shape}")
    print(f"Global Shape    : {global_embedding.shape}")

    print()

    print("Raw Similarity")
    print(f"Minimum : {torch.min(raw_scores):.4f}")
    print(f"Maximum : {torch.max(raw_scores):.4f}")
    print(f"Mean    : {torch.mean(raw_scores):.4f}")

    print()

    print("Normalized Similarity")
    print(f"Minimum : {torch.min(normalized_scores):.4f}")
    print(f"Maximum : {torch.max(normalized_scores):.4f}")
    print(f"Mean    : {torch.mean(normalized_scores):.4f}")

    print()

    print("First 10 Semantic Scores")

    for i in range(10):

        print(
            f"Frame {i+1:03d} : "
            f"{normalized_scores[i].item():.4f}"
        )

    print()

    print("Top 10 Frames")

    values, indices = torch.topk(
        normalized_scores,
        k=10,
    )

    top_frames = []

    for rank, (idx, value) in enumerate(
        zip(indices, values),
        start=1,
    ):

        top_frames.append(
            {
                "rank": rank,
                "frame": idx.item() + 1,
                "score": float(value),
            }
        )

        print(
            f"{rank:02d}. "
            f"Frame {idx.item()+1:03d} "
            f"Score = {value.item():.4f}"
        )

    print()

    print("Least 10 Frames")

    values, indices = torch.topk(
        normalized_scores,
        k=10,
        largest=False,
    )

    least_frames = []

    for rank, (idx, value) in enumerate(
        zip(indices, values),
        start=1,
    ):

        least_frames.append(
            {
                "rank": rank,
                "frame": idx.item() + 1,
                "score": float(value),
            }
        )

        print(
            f"{rank:02d}. "
            f"Frame {idx.item()+1:03d} "
            f"Score = {value.item():.4f}"
        )

    with open(
        RESULT_DIR / "top_frames.json",
        "w",
    ) as f:

        json.dump(
            top_frames,
            f,
            indent=4,
        )

    with open(
        RESULT_DIR / "least_frames.json",
        "w",
    ) as f:

        json.dump(
            least_frames,
            f,
            indent=4,
        )

    print()

    print("Results Saved")

    print(RESULT_DIR / "global_embedding.pt")
    print(RESULT_DIR / "global_embedding.csv")

    print(RESULT_DIR / "raw_scores.pt")
    print(RESULT_DIR / "raw_scores.csv")

    print(RESULT_DIR / "normalized_scores.pt")
    print(RESULT_DIR / "normalized_scores.csv")

    print(RESULT_DIR / "semantic_similarity.json")

    print(RESULT_DIR / "top_frames.json")
    print(RESULT_DIR / "least_frames.json")


if __name__ == "__main__":
    main()