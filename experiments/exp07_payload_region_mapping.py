from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd


WATERMARK_DIR = (
    PROJECT_ROOT
    / "results"
    / "watermark"
)

MAPPING_DIR = (
    PROJECT_ROOT
    / "results"
    / "spectrogram_mapping"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "payload_mapping"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PAYLOAD_FILE = (
    WATERMARK_DIR
    / "payload_bits.pt"
)

REGION_FILE = (
    MAPPING_DIR
    / "mapped_regions.csv"
)

SPECTROGRAM_FILE = (
    PROJECT_ROOT
    / "results"
    / "spectrograms"
    / "log_mel_spectrogram.pt"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

HUBERT_FRAMES = 521

# We reserve the extreme Mel bins and use the remaining
# bins as candidate embedding coefficients.
FREQUENCY_MARGIN = 2

# Planning assumption:
# one watermark bit per selected coefficient.
BITS_PER_COEFFICIENT = 1


# ---------------------------------------------------------
# Temporal mapping
# ---------------------------------------------------------

def map_hubert_frame_to_spectrogram(
    frame,
    hubert_frames,
    spectrogram_frames,
):
    """
    Map a 1-based HuBERT frame index to a
    1-based spectrogram frame index using
    normalized temporal position.
    """

    normalized_position = (
        (frame - 1)
        / (hubert_frames - 1)
    )

    mapped_position = (
        normalized_position
        * (spectrogram_frames - 1)
    )

    return int(
        round(mapped_position)
    ) + 1


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("SemBind-Audio : Experiment 07")
    print("Spectrogram Capacity & Payload Mapping")
    print("=" * 60)

    # -----------------------------------------------------
    # Load payload
    # -----------------------------------------------------

    payload_bits = torch.load(
        PAYLOAD_FILE,
        map_location="cpu",
    )

    payload_bits = payload_bits.flatten()

    payload_length = int(
        payload_bits.numel()
    )

    # -----------------------------------------------------
    # Load spectrogram
    # -----------------------------------------------------

    spectrogram = torch.load(
        SPECTROGRAM_FILE,
        map_location="cpu",
    )

    if not torch.is_tensor(
        spectrogram
    ):
        raise TypeError(
            "Log-Mel spectrogram must "
            "be a torch.Tensor."
        )

    spectrogram = (
        spectrogram
        .detach()
        .cpu()
    )

    original_shape = tuple(
        spectrogram.shape
    )

    # We expect:
    #
    # frequency × time
    #
    if spectrogram.ndim != 2:

        raise ValueError(
            "Expected a 2D Log-Mel "
            f"spectrogram. Got: "
            f"{original_shape}"
        )

    frequency_bins = int(
        spectrogram.shape[0]
    )

    spectrogram_frames = int(
        spectrogram.shape[1]
    )

    # -----------------------------------------------------
    # Determine usable frequency bins
    # -----------------------------------------------------

    if frequency_bins <= (
        2 * FREQUENCY_MARGIN
    ):

        raise ValueError(
            "Not enough frequency bins "
            "for configured margin."
        )

    eligible_frequency_start = (
        FREQUENCY_MARGIN
    )

    eligible_frequency_end = (
        frequency_bins
        - FREQUENCY_MARGIN
    )

    eligible_frequency_bins = (
        eligible_frequency_end
        - eligible_frequency_start
    )

    # -----------------------------------------------------
    # Load semantic regions
    # -----------------------------------------------------

    regions = pd.read_csv(
        REGION_FILE
    )

    print()

    print(
        f"HuBERT Frames          : "
        f"{HUBERT_FRAMES}"
    )

    print(
        f"Log-Mel Shape          : "
        f"{original_shape}"
    )

    print(
        f"Frequency Bins         : "
        f"{frequency_bins}"
    )

    print(
        f"Time Frames            : "
        f"{spectrogram_frames}"
    )

    print(
        f"Eligible Frequency Bins: "
        f"{eligible_frequency_bins}"
    )

    print(
        f"Payload Length         : "
        f"{payload_length} bits"
    )

    print()

    # -----------------------------------------------------
    # Map the ORIGINAL HuBERT regions directly
    # onto the actual 522-frame Log-Mel spectrogram.
    #
    # This avoids using the earlier 322-frame
    # intermediate mapping.
    # -----------------------------------------------------

    region_records = []

    total_capacity = 0

    for _, region in regions.iterrows():

        region_id = int(
            region["region_id"]
        )

        hubert_start = int(
            region[
                "hubert_start_frame"
            ]
        )

        hubert_end = int(
            region[
                "hubert_end_frame"
            ]
        )

        spectrogram_start = (
            map_hubert_frame_to_spectrogram(
                hubert_start,
                HUBERT_FRAMES,
                spectrogram_frames,
            )
        )

        spectrogram_end = (
            map_hubert_frame_to_spectrogram(
                hubert_end,
                HUBERT_FRAMES,
                spectrogram_frames,
            )
        )

        if (
            spectrogram_end
            < spectrogram_start
        ):

            (
                spectrogram_start,
                spectrogram_end,
            ) = (
                spectrogram_end,
                spectrogram_start,
            )

        time_length = (
            spectrogram_end
            - spectrogram_start
            + 1
        )

        capacity = (
            time_length
            * eligible_frequency_bins
            * BITS_PER_COEFFICIENT
        )

        total_capacity += capacity

        region_records.append(
            {
                "region_id": region_id,

                "hubert_start_frame":
                    hubert_start,

                "hubert_end_frame":
                    hubert_end,

                "spectrogram_start_frame":
                    spectrogram_start,

                "spectrogram_end_frame":
                    spectrogram_end,

                "time_frames":
                    time_length,

                "frequency_bins":
                    frequency_bins,

                "eligible_frequency_start":
                    eligible_frequency_start,

                "eligible_frequency_end":
                    eligible_frequency_end - 1,

                "eligible_frequency_bins":
                    eligible_frequency_bins,

                "capacity_bits":
                    int(capacity),

                "mean_semantic_score":
                    float(
                        region[
                            "mean_semantic_score"
                        ]
                    ),
            }
        )

    capacity_df = pd.DataFrame(
        region_records
    )

    # -----------------------------------------------------
    # Save capacity analysis
    # -----------------------------------------------------

    capacity_df.to_csv(
        RESULT_DIR
        / "region_capacity.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Capacity check
    # -----------------------------------------------------

    capacity_sufficient = (
        total_capacity
        >= payload_length
    )

    print(
        "Region Capacity Analysis"
    )

    print("-" * 60)

    for _, row in capacity_df.iterrows():

        print(
            f"Region "
            f"{int(row['region_id']):02d} | "
            f"HuBERT "
            f"{int(row['hubert_start_frame']):03d}-"
            f"{int(row['hubert_end_frame']):03d} "
            f"-> "
            f"Log-Mel "
            f"{int(row['spectrogram_start_frame']):03d}-"
            f"{int(row['spectrogram_end_frame']):03d} "
            f"| Time: "
            f"{int(row['time_frames']):03d} "
            f"| Capacity: "
            f"{int(row['capacity_bits']):04d} bits"
        )

    print()

    print(
        f"Total Available Capacity : "
        f"{total_capacity} bits"
    )

    print(
        f"Required Payload          : "
        f"{payload_length} bits"
    )

    print()

    if capacity_sufficient:

        print(
            "Capacity Check            : "
            "SUFFICIENT"
        )

    else:

        print(
            "Capacity Check            : "
            "INSUFFICIENT"
        )

    # -----------------------------------------------------
    # Payload-to-region mapping
    # -----------------------------------------------------

    mappings = []

    bit_pointer = 0

    if capacity_sufficient:

        for _, row in capacity_df.iterrows():

            if (
                bit_pointer
                >= payload_length
            ):
                break

            region_capacity = int(
                row[
                    "capacity_bits"
                ]
            )

            remaining_bits = (
                payload_length
                - bit_pointer
            )

            bits_assigned = min(
                region_capacity,
                remaining_bits,
            )

            bit_start = (
                bit_pointer + 1
            )

            bit_end = (
                bit_pointer
                + bits_assigned
            )

            mappings.append(
                {
                    "region_id":
                        int(
                            row[
                                "region_id"
                            ]
                        ),

                    "spectrogram_start_frame":
                        int(
                            row[
                                "spectrogram_start_frame"
                            ]
                        ),

                    "spectrogram_end_frame":
                        int(
                            row[
                                "spectrogram_end_frame"
                            ]
                        ),

                    "eligible_frequency_start":
                        int(
                            row[
                                "eligible_frequency_start"
                            ]
                        ),

                    "eligible_frequency_end":
                        int(
                            row[
                                "eligible_frequency_end"
                            ]
                        ),

                    "capacity_bits":
                        region_capacity,

                    "payload_bit_start":
                        bit_start,

                    "payload_bit_end":
                        bit_end,

                    "bits_assigned":
                        int(
                            bits_assigned
                        ),
                }
            )

            bit_pointer += (
                bits_assigned
            )

    mapping_df = pd.DataFrame(
        mappings
    )

    mapping_df.to_csv(
        RESULT_DIR
        / "payload_region_mapping.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = {

        "payload_length_bits":
            payload_length,

        "hubert_frames":
            HUBERT_FRAMES,

        "spectrogram_shape":
            list(
                original_shape
            ),

        "frequency_bins":
            frequency_bins,

        "spectrogram_frames":
            spectrogram_frames,

        "frequency_margin":
            FREQUENCY_MARGIN,

        "eligible_frequency_bins":
            eligible_frequency_bins,

        "candidate_regions":
            int(
                len(regions)
            ),

        "total_available_capacity_bits":
            int(
                total_capacity
            ),

        "capacity_sufficient":
            bool(
                capacity_sufficient
            ),

        "bits_per_coefficient":
            BITS_PER_COEFFICIENT,

        "regions_used":
            int(
                len(mapping_df)
            ),

        "payload_bits_mapped":
            int(
                bit_pointer
            ),

        "unused_capacity_bits":
            int(
                max(
                    0,
                    total_capacity
                    - payload_length,
                )
            ),
    }

    with open(
        RESULT_DIR
        / "payload_mapping_summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    # -----------------------------------------------------
    # Print mapping
    # -----------------------------------------------------

    if len(mapping_df) > 0:

        print()

        print(
            "Payload Region Mapping"
        )

        print("-" * 60)

        for _, row in (
            mapping_df.iterrows()
        ):

            print(
                f"Region "
                f"{int(row['region_id']):02d} | "
                f"Log-Mel "
                f"{int(row['spectrogram_start_frame']):03d}-"
                f"{int(row['spectrogram_end_frame']):03d} "
                f"| Bits "
                f"{int(row['payload_bit_start']):03d}-"
                f"{int(row['payload_bit_end']):03d} "
                f"| Assigned: "
                f"{int(row['bits_assigned'])}"
            )

    print()

    print(
        "Results Saved"
    )

    print(
        RESULT_DIR
        / "region_capacity.csv"
    )

    print(
        RESULT_DIR
        / "payload_region_mapping.csv"
    )

    print(
        RESULT_DIR
        / "payload_mapping_summary.json"
    )


if __name__ == "__main__":
    main()
    