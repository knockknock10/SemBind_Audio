from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
import numpy as np

CONTRIBUTION_DIR = PROJECT_ROOT / "results" / "contribution"
RESULT_DIR = PROJECT_ROOT / "results" / "region_selection"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = CONTRIBUTION_DIR / "semantic_contribution.csv"

PERCENTILE = 25
MIN_REGION_LENGTH = 3


def merge_consecutive_frames(frames):
    if not frames:
        return []

    regions = []

    start = frames[0]
    previous = frames[0]

    for frame in frames[1:]:
        if frame == previous + 1:
            previous = frame
        else:
            regions.append((start, previous))
            start = frame
            previous = frame

    regions.append((start, previous))

    return regions


def main():
    print("=" * 60)
    print("SemBind-Audio : Experiment 04")
    print("Semantic Region Selection")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    frames = df["frame"].to_numpy()
    scores = df["normalized_score"].to_numpy()

    threshold = float(np.percentile(scores, PERCENTILE))

    selected_mask = scores <= threshold

    selected_frames = frames[selected_mask]

    regions = merge_consecutive_frames(
        selected_frames.tolist()
    )

    filtered_regions = []

    for start, end in regions:
        length = end - start + 1

        if length >= MIN_REGION_LENGTH:
            filtered_regions.append(
                {
                    "region_id": len(filtered_regions) + 1,
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "length": int(length),
                    "mean_score": float(
                        df[
                            (df["frame"] >= start)
                            & (df["frame"] <= end)
                        ]["normalized_score"].mean()
                    ),
                    "minimum_score": float(
                        df[
                            (df["frame"] >= start)
                            & (df["frame"] <= end)
                        ]["normalized_score"].min()
                    ),
                    "maximum_score": float(
                        df[
                            (df["frame"] >= start)
                            & (df["frame"] <= end)
                        ]["normalized_score"].max()
                    ),
                }
            )

    selected_df = df[selected_mask].copy()

    selected_df.to_csv(
        RESULT_DIR / "selected_frames.csv",
        index=False,
    )

    region_df = pd.DataFrame(filtered_regions)

    region_df.to_csv(
        RESULT_DIR / "candidate_regions.csv",
        index=False,
    )

    mask_tensor = torch.tensor(
        selected_mask,
        dtype=torch.bool,
    )

    torch.save(
        mask_tensor,
        RESULT_DIR / "candidate_region_mask.pt",
    )

    summary = {
        "total_frames": int(len(frames)),
        "selected_frames": int(len(selected_frames)),
        "selection_percent": float(
            len(selected_frames) / len(frames) * 100
        ),
        "percentile": PERCENTILE,
        "threshold": threshold,
        "minimum_region_length": MIN_REGION_LENGTH,
        "candidate_regions": int(len(filtered_regions)),
    }

    with open(
        RESULT_DIR / "region_selection_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    semantic_scores = pd.DataFrame(
        {
            "frame": frames,
            "normalized_score": scores,
            "selected": selected_mask,
        }
    )

    semantic_scores.to_csv(
        RESULT_DIR / "semantic_scores.csv",
        index=False,
    )

    print()
    print(f"Total Frames              : {len(frames)}")
    print(f"Selection Percentile      : {PERCENTILE}%")
    print(f"Semantic Threshold        : {threshold:.6f}")
    print(f"Selected Frames           : {len(selected_frames)}")
    print(
        f"Selected Percentage      : "
        f"{len(selected_frames) / len(frames) * 100:.2f}%"
    )
    print(
        f"Candidate Regions         : "
        f"{len(filtered_regions)}"
    )

    print()

    if filtered_regions:
        print("Candidate Embedding Regions")

        for region in filtered_regions:
            print(
                f"{region['region_id']:02d}. "
                f"Frame {region['start_frame']:03d} - "
                f"{region['end_frame']:03d} | "
                f"Length: {region['length']:03d} | "
                f"Mean Score: "
                f"{region['mean_score']:.6f}"
            )
    else:
        print("No candidate regions found.")

    print()
    print("Results Saved")
    print(RESULT_DIR / "semantic_scores.csv")
    print(RESULT_DIR / "selected_frames.csv")
    print(RESULT_DIR / "candidate_regions.csv")
    print(RESULT_DIR / "candidate_region_mask.pt")
    print(RESULT_DIR / "region_selection_summary.json")


if __name__ == "__main__":
    main()