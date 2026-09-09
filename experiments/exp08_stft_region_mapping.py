from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd


MAPPING_DIR = (
    PROJECT_ROOT
    / "results"
    / "spectrogram_mapping"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "stft_mapping"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REGION_FILE = (
    MAPPING_DIR
    / "mapped_regions.csv"
)

STFT_FILE = (
    PROJECT_ROOT
    / "results"
    / "spectrograms"
    / "stft.pt"
)

# Your actual STFT has 522 time frames.
EXPECTED_TIME_FRAMES = 522

# We will reserve the extreme frequency bins.
FREQUENCY_MARGIN = 4


def main():

    print("=" * 60)
    print("SemBind-Audio : Experiment 08")
    print("Semantic Region -> STFT Mapping")
    print("=" * 60)

    # --------------------------------------------------
    # Load STFT
    # --------------------------------------------------

    stft = torch.load(
        STFT_FILE,
        map_location="cpu",
    )

    if not torch.is_tensor(stft):
        raise TypeError(
            "STFT must be a torch.Tensor."
        )

    stft = stft.detach().cpu()

    original_shape = tuple(
        stft.shape
    )

    if stft.ndim != 2:
        raise ValueError(
            "Expected a 2D STFT tensor "
            f"(frequency, time). "
            f"Got {original_shape}"
        )

    frequency_bins = int(
        stft.shape[0]
    )

    time_frames = int(
        stft.shape[1]
    )

    # --------------------------------------------------
    # Load semantic regions
    # --------------------------------------------------

    regions = pd.read_csv(
        REGION_FILE
    )

    print()
    print(
        f"STFT Shape        : "
        f"{original_shape}"
    )

    print(
        f"Frequency Bins    : "
        f"{frequency_bins}"
    )

    print(
        f"Time Frames       : "
        f"{time_frames}"
    )

    print(
        f"Semantic Regions  : "
        f"{len(regions)}"
    )

    # --------------------------------------------------
    # Frequency range
    # --------------------------------------------------

    frequency_start = (
        FREQUENCY_MARGIN
    )

    frequency_end = (
        frequency_bins
        - FREQUENCY_MARGIN
    )

    usable_frequency_bins = (
        frequency_end
        - frequency_start
    )

    print(
        f"Usable Frequency Bins : "
        f"{usable_frequency_bins}"
    )

    # --------------------------------------------------
    # Map regions
    # --------------------------------------------------

    mapped_regions = []

    total_coefficients = 0

    for _, region in regions.iterrows():

        region_id = int(
            region["region_id"]
        )

        start_frame = int(
            region[
                "spectrogram_start_frame"
            ]
        )

        end_frame = int(
            region[
                "spectrogram_end_frame"
            ]
        )

        # Convert 1-based frame numbers
        # into 0-based tensor indices.

        start_index = max(
            0,
            start_frame - 1,
        )

        end_index = min(
            time_frames - 1,
            end_frame - 1,
        )

        if end_index < start_index:
            continue

        selected_time_frames = (
            end_index
            - start_index
            + 1
        )

        coefficient_count = (
            selected_time_frames
            * usable_frequency_bins
        )

        total_coefficients += (
            coefficient_count
        )

        mapped_regions.append(
            {
                "region_id":
                    region_id,

                "time_start_frame":
                    start_frame,

                "time_end_frame":
                    end_frame,

                "frequency_start_bin":
                    frequency_start,

                "frequency_end_bin":
                    frequency_end - 1,

                "time_frames":
                    selected_time_frames,

                "frequency_bins":
                    usable_frequency_bins,

                "stft_coefficients":
                    coefficient_count,

                "mean_semantic_score":
                    float(
                        region[
                            "mean_semantic_score"
                        ]
                    ),
            }
        )

    mapped_df = pd.DataFrame(
        mapped_regions
    )

    # --------------------------------------------------
    # Save mapping
    # --------------------------------------------------

    mapped_df.to_csv(
        RESULT_DIR
        / "stft_region_mapping.csv",
        index=False,
    )

    # --------------------------------------------------
    # Save summary
    # --------------------------------------------------

    summary = {
        "stft_shape":
            list(
                original_shape
            ),

        "frequency_bins":
            frequency_bins,

        "time_frames":
            time_frames,

        "semantic_regions":
            int(
                len(regions)
            ),

        "usable_frequency_bins":
            usable_frequency_bins,

        "total_stft_coefficients":
            int(
                total_coefficients
            ),

        "frequency_margin":
            FREQUENCY_MARGIN,

        "mapping_domain":
            "STFT",
    }

    with open(
        RESULT_DIR
        / "stft_mapping_summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print()
    print(
        "STFT Embedding Regions"
    )

    print("-" * 60)

    for _, row in mapped_df.iterrows():

        print(
            f"Region "
            f"{int(row['region_id']):02d} | "
            f"Time "
            f"{int(row['time_start_frame']):03d}-"
            f"{int(row['time_end_frame']):03d} | "
            f"Frequency "
            f"{int(row['frequency_start_bin']):03d}-"
            f"{int(row['frequency_end_bin']):03d} | "
            f"Coefficients: "
            f"{int(row['stft_coefficients'])}"
        )

    print()
    print(
        f"Total STFT Coefficients : "
        f"{total_coefficients}"
    )

    print()
    print("Results Saved")

    print(
        RESULT_DIR
        / "stft_region_mapping.csv"
    )

    print(
        RESULT_DIR
        / "stft_mapping_summary.json"
    )


if __name__ == "__main__":
    main()