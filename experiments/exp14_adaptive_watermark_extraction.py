"""
SemBind-Audio

Experiment 14
Adaptive Watermark Extraction

This experiment extracts the watermark embedded by
Experiment 13: Semantic-Adaptive STFT Embedding.

Experiment 13 embedding rule
----------------------------

For every selected STFT coefficient:

    bit = 1:
        real(X_w) = real(X) + alpha

    bit = 0:
        real(X_w) = real(X) - alpha

Only the REAL component of the STFT coefficient
is modified.

Therefore extraction is performed using:

    delta = real(X_w) - real(X)

    delta > 0 -> bit 1
    delta < 0 -> bit 0

Inputs
------

results/adaptive_embedding/original_stft.pt

results/adaptive_embedding/adaptive_watermarked_stft.pt

results/adaptive_embedding/adaptive_embedding_coordinates.csv

results/watermark/payload_bits.pt


Outputs
-------

results/adaptive_extraction/extracted_bits.pt

results/adaptive_extraction/extracted_bits.txt

results/adaptive_extraction/extraction_details.csv

results/adaptive_extraction/extraction_summary.json
"""

from pathlib import Path
import sys
import json

import torch
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# RESULT DIRECTORY
# ============================================================

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "adaptive_extraction"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT FILES
# ============================================================

ORIGINAL_STFT_FILE = (
    PROJECT_ROOT
    / "results"
    / "adaptive_embedding"
    / "original_stft.pt"
)

WATERMARKED_STFT_FILE = (
    PROJECT_ROOT
    / "results"
    / "adaptive_embedding"
    / "adaptive_watermarked_stft.pt"
)

COORDINATES_FILE = (
    PROJECT_ROOT
    / "results"
    / "adaptive_embedding"
    / "adaptive_embedding_coordinates.csv"
)

PAYLOAD_BITS_FILE = (
    PROJECT_ROOT
    / "results"
    / "watermark"
    / "payload_bits.pt"
)


# ============================================================
# OUTPUT FILES
# ============================================================

EXTRACTED_BITS_FILE = (
    RESULT_DIR
    / "extracted_bits.pt"
)

EXTRACTED_BITS_TXT = (
    RESULT_DIR
    / "extracted_bits.txt"
)

DETAILS_FILE = (
    RESULT_DIR
    / "extraction_details.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "extraction_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def load_tensor(path: Path) -> torch.Tensor:
    """
    Load a tensor using CPU.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    tensor = torch.load(
        path,
        map_location="cpu",
    )

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    return tensor.detach().cpu()


def load_payload(path: Path) -> torch.Tensor:
    """
    Load the original watermark payload.

    The payload is converted into a flat binary uint8 tensor.
    """

    payload = load_tensor(path)

    payload = payload.flatten()

    payload = (
        payload > 0
    ).to(torch.uint8)

    return payload


def find_column(
    dataframe: pd.DataFrame,
    candidates,
    description: str,
):
    """
    Find the first matching column from a list of candidates.
    """

    for candidate in candidates:

        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        "\nCould not find required column "
        f"for {description}.\n"
        f"Available columns: "
        f"{list(dataframe.columns)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SemBind-Audio : Experiment 14")
    print("Adaptive Watermark Extraction")
    print("=" * 60)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required_files = [
        ORIGINAL_STFT_FILE,
        WATERMARKED_STFT_FILE,
        COORDINATES_FILE,
        PAYLOAD_BITS_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired input file does not exist:\n{path}"
            )

    # --------------------------------------------------------
    # Load STFTs
    # --------------------------------------------------------

    original_stft = load_tensor(
        ORIGINAL_STFT_FILE
    )

    watermarked_stft = load_tensor(
        WATERMARKED_STFT_FILE
    )

    # --------------------------------------------------------
    # Validate STFTs
    # --------------------------------------------------------

    if original_stft.ndim != 2:

        raise ValueError(
            "Original STFT must be a 2-D tensor."
        )

    if watermarked_stft.ndim != 2:

        raise ValueError(
            "Watermarked STFT must be a 2-D tensor."
        )

    if original_stft.shape != watermarked_stft.shape:

        raise ValueError(
            "\nOriginal and watermarked STFT shapes "
            "do not match.\n"
            f"Original   : {original_stft.shape}\n"
            f"Watermarked: {watermarked_stft.shape}"
        )

    # --------------------------------------------------------
    # Load payload
    # --------------------------------------------------------

    payload = load_payload(
        PAYLOAD_BITS_FILE
    )

    # --------------------------------------------------------
    # Load embedding coordinates
    # --------------------------------------------------------

    coordinates = pd.read_csv(
        COORDINATES_FILE
    )

    print()

    print(
        f"Original STFT Shape    : "
        f"{tuple(original_stft.shape)}"
    )

    print(
        f"Watermarked STFT Shape : "
        f"{tuple(watermarked_stft.shape)}"
    )

    print(
        f"Payload Length         : "
        f"{payload.numel()} bits"
    )

    print(
        f"Coordinate Rows        : "
        f"{len(coordinates)}"
    )

    print(
        f"Coordinate Columns     : "
        f"{list(coordinates.columns)}"
    )

    # --------------------------------------------------------
    # Locate coordinate columns
    # --------------------------------------------------------

    time_column = find_column(
        coordinates,
        [
            "time_frame",
            "time",
            "frame",
            "spectrogram_frame",
            "stft_time_frame",
        ],
        "time frame",
    )

    frequency_column = find_column(
        coordinates,
        [
            "frequency_bin",
            "frequency",
            "freq",
            "frequency_index",
            "stft_frequency_bin",
        ],
        "frequency bin",
    )

    # --------------------------------------------------------
    # Check coordinate count
    # --------------------------------------------------------

    if len(coordinates) < payload.numel():

        raise ValueError(
            "\nNot enough embedding coordinates.\n"
            f"Payload bits : {payload.numel()}\n"
            f"Coordinates  : {len(coordinates)}"
        )

    if len(coordinates) > payload.numel():

        print()
        print(
            "Warning: More coordinates than payload bits."
        )

        print(
            f"Using first {payload.numel()} coordinates."
        )

        coordinates = coordinates.iloc[
            :payload.numel()
        ].copy()

    # --------------------------------------------------------
    # STFT dimensions
    # --------------------------------------------------------

    frequency_bins = original_stft.shape[0]
    time_frames = original_stft.shape[1]

    # --------------------------------------------------------
    # Extraction
    # --------------------------------------------------------

    extracted_bits = []

    extraction_records = []

    positive_changes = 0
    negative_changes = 0
    zero_changes = 0

    for bit_index, (_, row) in enumerate(
        coordinates.iterrows()
    ):

        frequency_bin = int(
            row[frequency_column]
        )

        time_frame = int(
            row[time_column]
        )

        # ----------------------------------------------------
        # Coordinate validation
        # ----------------------------------------------------

        if not (
            0
            <= frequency_bin
            < frequency_bins
        ):

            raise ValueError(
                "\nFrequency coordinate out of bounds.\n"
                f"Bit index     : {bit_index}\n"
                f"Frequency bin : {frequency_bin}\n"
                f"Valid range  : "
                f"0-{frequency_bins - 1}"
            )

        if not (
            0
            <= time_frame
            < time_frames
        ):

            raise ValueError(
                "\nTime coordinate out of bounds.\n"
                f"Bit index  : {bit_index}\n"
                f"Time frame : {time_frame}\n"
                f"Valid range: "
                f"0-{time_frames - 1}"
            )

        # ----------------------------------------------------
        # Read original coefficient
        # ----------------------------------------------------

        original_value = original_stft[
            frequency_bin,
            time_frame,
        ]

        # ----------------------------------------------------
        # Read watermarked coefficient
        # ----------------------------------------------------

        watermarked_value = watermarked_stft[
            frequency_bin,
            time_frame,
        ]

        # ----------------------------------------------------
        # Experiment 13 modifies ONLY real component
        #
        # bit 1:
        #
        #     real(Xw) = real(X) + alpha
        #
        # bit 0:
        #
        #     real(Xw) = real(X) - alpha
        #
        # Therefore:
        #
        #     delta = real(Xw) - real(X)
        #
        #     delta > 0 -> 1
        #     delta < 0 -> 0
        # ----------------------------------------------------

        original_real = (
            original_value.real
        )

        watermarked_real = (
            watermarked_value.real
        )

        delta = (
            watermarked_real
            - original_real
        )

        delta_value = float(
            delta.item()
        )

        # ----------------------------------------------------
        # Recover bit
        # ----------------------------------------------------

        if delta_value > 0:

            extracted_bit = 1
            positive_changes += 1

        elif delta_value < 0:

            extracted_bit = 0
            negative_changes += 1

        else:

            # No modification means extraction failure.
            extracted_bit = 0
            zero_changes += 1

        extracted_bits.append(
            extracted_bit
        )

        # ----------------------------------------------------
        # Store extraction details
        # ----------------------------------------------------

        original_bit = int(
            payload[bit_index].item()
        )

        correct = (
            extracted_bit
            == original_bit
        )

        extraction_records.append(
            {
                "bit_index": bit_index,
                "time_frame": time_frame,
                "frequency_bin": frequency_bin,
                "original_bit": original_bit,
                "extracted_bit": extracted_bit,
                "correct": correct,
                "original_real": float(
                    original_real.item()
                ),
                "watermarked_real": float(
                    watermarked_real.item()
                ),
                "real_change": delta_value,
                "original_magnitude": float(
                    torch.abs(
                        original_value
                    ).item()
                ),
                "watermarked_magnitude": float(
                    torch.abs(
                        watermarked_value
                    ).item()
                ),
            }
        )

    # --------------------------------------------------------
    # Convert extracted bits
    # --------------------------------------------------------

    extracted_bits = torch.tensor(
        extracted_bits,
        dtype=torch.uint8,
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    matches = (
        extracted_bits
        == payload
    )

    total_bits = int(
        payload.numel()
    )

    correct_bits = int(
        matches.sum().item()
    )

    incorrect_bits = (
        total_bits
        - correct_bits
    )

    bit_accuracy = (
        correct_bits / total_bits
    )

    bit_error_rate = (
        incorrect_bits / total_bits
    )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    details_dataframe = pd.DataFrame(
        extraction_records
    )

    # --------------------------------------------------------
    # Print extraction statistics
    # --------------------------------------------------------

    print()
    print(
        "## Extraction Statistics"
    )

    print()

    print(
        f"Positive Changes : "
        f"{positive_changes}"
    )

    print(
        f"Negative Changes : "
        f"{negative_changes}"
    )

    print(
        f"Zero Changes     : "
        f"{zero_changes}"
    )

    print()

    print(
        f"Total Bits       : "
        f"{total_bits}"
    )

    print(
        f"Correct Bits     : "
        f"{correct_bits}"
    )

    print(
        f"Incorrect Bits   : "
        f"{incorrect_bits}"
    )

    print(
        f"Bit Accuracy     : "
        f"{bit_accuracy:.6f}"
    )

    print(
        f"Bit Error Rate   : "
        f"{bit_error_rate:.6f}"
    )

    # --------------------------------------------------------
    # Verify extraction
    # --------------------------------------------------------

    if incorrect_bits == 0:

        verification = "PASSED"

    else:

        verification = "FAILED"

    print()

    print(
        f"Extraction Verification : "
        f"{verification}"
    )

    # --------------------------------------------------------
    # Save extracted tensor
    # --------------------------------------------------------

    torch.save(
        extracted_bits,
        EXTRACTED_BITS_FILE,
    )

    # --------------------------------------------------------
    # Save bit string
    # --------------------------------------------------------

    bit_string = "".join(
        str(int(bit.item()))
        for bit in extracted_bits
    )

    with open(
        EXTRACTED_BITS_TXT,
        "w",
    ) as file:

        file.write(
            bit_string
        )

    # --------------------------------------------------------
    # Save detailed CSV
    # --------------------------------------------------------

    details_dataframe.to_csv(
        DETAILS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "experiment": (
            "Experiment 14 - "
            "Adaptive Watermark Extraction"
        ),

        "embedding_method": (
            "Semantic-Adaptive STFT "
            "Real-Component Embedding"
        ),

        "stft_shape": [
            int(value)
            for value in original_stft.shape
        ],

        "payload_length_bits": total_bits,

        "embedding_coordinates": len(
            coordinates
        ),

        "positive_changes": (
            positive_changes
        ),

        "negative_changes": (
            negative_changes
        ),

        "zero_changes": (
            zero_changes
        ),

        "correct_bits": correct_bits,

        "incorrect_bits": incorrect_bits,

        "bit_accuracy": bit_accuracy,

        "bit_error_rate": bit_error_rate,

        "verification": verification,
    }

    with open(
        SUMMARY_FILE,
        "w",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print(
        "Results Saved"
    )

    print(
        EXTRACTED_BITS_FILE
    )

    print(
        EXTRACTED_BITS_TXT
    )

    print(
        DETAILS_FILE
    )

    print(
        SUMMARY_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()