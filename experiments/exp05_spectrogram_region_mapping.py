from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
import numpy as np

REGION_DIR = PROJECT_ROOT / "results" / "region_selection"
RESULT_DIR = PROJECT_ROOT / "results" / "spectrogram_mapping"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

REGION_FILE = REGION_DIR / "candidate_regions.csv"

HUBERT_FRAMES = 521
SPECTROGRAM_FRAMES = 322


def map_frame(frame, source_frames, target_frames):
    normalized = (frame - 1) / (source_frames - 1)
    mapped = normalized * (target_frames - 1)
    return int(round(mapped)) + 1


def main():
    print("=" * 60)
    print("SemBind-Audio : Experiment 05")
    print("Temporal to Spectrogram Region Mapping")
    print("=" * 60)

    regions = pd.read_csv(REGION_FILE)

    mapped_regions = []

    for _, region in regions.iterrows():

        start_frame = int(region["start_frame"])
        end_frame = int(region["end_frame"])

        mapped_start = map_frame(
            start_frame,
            HUBERT_FRAMES,
            SPECTROGRAM_FRAMES,
        )

        mapped_end = map_frame(
            end_frame,
            HUBERT_FRAMES,
            SPECTROGRAM_FRAMES,
        )

        if mapped_end < mapped_start:
            mapped_start, mapped_end = (
                mapped_end,
                mapped_start,
            )

        mapped_length = mapped_end - mapped_start + 1

        mapped_regions.append(
            {
                "region_id": int(region["region_id"]),
                "hubert_start_frame": start_frame,
                "hubert_end_frame": end_frame,
                "hubert_length": int(region["length"]),
                "spectrogram_start_frame": mapped_start,
                "spectrogram_end_frame": mapped_end,
                "spectrogram_length": mapped_length,
                "mean_semantic_score": float(
                    region["mean_score"]
                ),
                "minimum_semantic_score": float(
                    region["minimum_score"]
                ),
                "maximum_semantic_score": float(
                    region["maximum_score"]
                ),
            }
        )

    mapped_df = pd.DataFrame(mapped_regions)

    mapped_df.to_csv(
        RESULT_DIR / "mapped_regions.csv",
        index=False,
    )

    mapping_tensor = torch.tensor(
        mapped_df[
            [
                "spectrogram_start_frame",
                "spectrogram_end_frame",
            ]
        ].values,
        dtype=torch.long,
    )

    torch.save(
        mapping_tensor,
        RESULT_DIR / "spectrogram_region_mapping.pt",
    )

    summary = {
        "hubert_frames": HUBERT_FRAMES,
        "spectrogram_frames": SPECTROGRAM_FRAMES,
        "input_regions": int(len(regions)),
        "mapped_regions": int(len(mapped_df)),
        "mapping_method": "normalized temporal interpolation",
    }

    with open(
        RESULT_DIR / "spectrogram_mapping_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    print()
    print(f"HuBERT Frames              : {HUBERT_FRAMES}")
    print(f"Spectrogram Frames        : {SPECTROGRAM_FRAMES}")
    print(f"Candidate Regions         : {len(regions)}")
    print()

    print("Mapped Spectrogram Regions")

    for _, row in mapped_df.iterrows():

        print(
            f"{int(row['region_id']):02d}. "
            f"HuBERT "
            f"{int(row['hubert_start_frame']):03d}-"
            f"{int(row['hubert_end_frame']):03d}"
            f"  ->  "
            f"Spectrogram "
            f"{int(row['spectrogram_start_frame']):03d}-"
            f"{int(row['spectrogram_end_frame']):03d}"
            f"  | Length: "
            f"{int(row['spectrogram_length']):03d}"
        )

    print()
    print("Results Saved")
    print(RESULT_DIR / "mapped_regions.csv")
    print(RESULT_DIR / "spectrogram_region_mapping.pt")
    print(RESULT_DIR / "spectrogram_mapping_summary.json")


if __name__ == "__main__":
    main()