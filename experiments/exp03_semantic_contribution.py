from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd

from src.semantic.contribution import SemanticContribution


RESULT_DIR = PROJECT_ROOT / "results" / "contribution"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_FILE = (
    PROJECT_ROOT
    / "results"
    / "embeddings"
    / "hubert_embeddings.pt"
)


def main():

    print("=" * 60)
    print("SemBind-Audio : Experiment 03")
    print("Semantic Contribution")
    print("=" * 60)

    embeddings = torch.load(
        EMBEDDING_FILE,
        map_location="cpu",
    )

    contribution = SemanticContribution(
        embeddings
    )

    results = contribution.compute()

    original = results["original_global_embedding"]
    perturbed = results["perturbed_global_embeddings"]
    raw_scores = results["raw_scores"]
    normalized_scores = results["normalized_scores"]
    frame_indices = results["frame_indices"]

    torch.save(
        original,
        RESULT_DIR / "original_global_embedding.pt",
    )

    torch.save(
        perturbed,
        RESULT_DIR / "perturbed_global_embeddings.pt",
    )

    torch.save(
        raw_scores,
        RESULT_DIR / "semantic_shift.pt",
    )

    torch.save(
        normalized_scores,
        RESULT_DIR / "normalized_scores.pt",
    )

    pd.DataFrame(
        original.numpy(),
        columns=["value"],
    ).to_csv(
        RESULT_DIR / "original_global_embedding.csv",
        index=False,
    )

    pd.DataFrame(
        perturbed.numpy(),
    ).to_csv(
        RESULT_DIR / "perturbed_global_embeddings.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "frame": frame_indices.numpy() + 1,
            "semantic_shift": raw_scores.numpy(),
            "normalized_score": normalized_scores.numpy(),
        }
    ).to_csv(
        RESULT_DIR / "semantic_contribution.csv",
        index=False,
    )

    summary = {
        "frames": int(frame_indices.shape[0]),
        "minimum": float(raw_scores.min()),
        "maximum": float(raw_scores.max()),
        "mean": float(raw_scores.mean()),
    }

    with open(
        RESULT_DIR / "semantic_contribution.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    print()
    print(f"Embedding Shape : {embeddings.shape}")
    print(f"Frames          : {frame_indices.shape[0]}")

    print()
    print("Semantic Contribution Statistics")
    print(f"Minimum : {raw_scores.min():.8f}")
    print(f"Maximum : {raw_scores.max():.8f}")
    print(f"Mean    : {raw_scores.mean():.8f}")

    print()

    values, indices = torch.topk(
        normalized_scores,
        k=10,
    )

    top_contributors = []

    print("Top 10 Contributors")

    for rank, (idx, value) in enumerate(
        zip(indices, values),
        start=1,
    ):

        record = {
            "rank": rank,
            "frame": idx.item() + 1,
            "score": float(value),
        }

        top_contributors.append(record)

        print(
            f"{rank:02d}. "
            f"Frame {idx.item()+1:03d} "
            f"{value.item():.6f}"
        )

    print()

    values, indices = torch.topk(
        normalized_scores,
        k=10,
        largest=False,
    )

    least_contributors = []

    print("Least 10 Contributors")

    for rank, (idx, value) in enumerate(
        zip(indices, values),
        start=1,
    ):

        record = {
            "rank": rank,
            "frame": idx.item() + 1,
            "score": float(value),
        }

        least_contributors.append(record)

        print(
            f"{rank:02d}. "
            f"Frame {idx.item()+1:03d} "
            f"{value.item():.6f}"
        )

    with open(
        RESULT_DIR / "top_contributors.json",
        "w",
    ) as f:
        json.dump(
            top_contributors,
            f,
            indent=4,
        )

    with open(
        RESULT_DIR / "least_contributors.json",
        "w",
    ) as f:
        json.dump(
            least_contributors,
            f,
            indent=4,
        )

    print()
    print("Results Saved")

    print(RESULT_DIR / "original_global_embedding.pt")
    print(RESULT_DIR / "original_global_embedding.csv")

    print(RESULT_DIR / "perturbed_global_embeddings.pt")
    print(RESULT_DIR / "perturbed_global_embeddings.csv")

    print(RESULT_DIR / "semantic_shift.pt")
    print(RESULT_DIR / "normalized_scores.pt")

    print(RESULT_DIR / "semantic_contribution.csv")
    print(RESULT_DIR / "semantic_contribution.json")

    print(RESULT_DIR / "top_contributors.json")
    print(RESULT_DIR / "least_contributors.json")


if __name__ == "__main__":
    main()