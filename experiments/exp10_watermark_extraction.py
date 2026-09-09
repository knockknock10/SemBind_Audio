from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd


EMBEDDING_DIR = (
    PROJECT_ROOT
    / "results"
    / "watermark_embedding"
)

ORIGINAL_STFT_FILE = (
    EMBEDDING_DIR
    / "original_stft.pt"
)

WATERMARKED_STFT_FILE = (
    EMBEDDING_DIR
    / "watermarked_stft.pt"
)

COORDINATES_FILE = (
    EMBEDDING_DIR
    / "embedding_coordinates.csv"
)

EMBEDDED_BITS_FILE = (
    EMBEDDING_DIR
    / "embedded_bits.pt"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "watermark_extraction"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EPSILON = 1e-12


def main():

    print("=" * 60)
    print("SemBind-Audio : Experiment 10")
    print("Watermark Extraction")
    print("=" * 60)

    original_stft = torch.load(
        ORIGINAL_STFT_FILE,
        map_location="cpu",
    ).detach().cpu()

    watermarked_stft = torch.load(
        WATERMARKED_STFT_FILE,
        map_location="cpu",
    ).detach().cpu()

    coordinates = pd.read_csv(
        COORDINATES_FILE
    )

    embedded_bits = torch.load(
        EMBEDDED_BITS_FILE,
        map_location="cpu",
    ).flatten().to(torch.uint8)

    if (
        original_stft.shape
        != watermarked_stft.shape
    ):
        raise ValueError(
            "Original and watermarked "
            "STFT shapes do not match."
        )

    extracted_bits = []

    extraction_records = []

    for _, row in coordinates.iterrows():

        bit_index = int(
            row["bit_index"]
        )

        frequency_bin = int(
            row["frequency_bin"]
        )

        time_index = int(
            row["time_index"]
        )

        original_value = (
            original_stft[
                frequency_bin,
                time_index,
            ]
        )

        watermarked_value = (
            watermarked_stft[
                frequency_bin,
                time_index,
            ]
        )

        original_magnitude = float(
            torch.abs(
                original_value
            )
        )

        watermarked_magnitude = float(
            torch.abs(
                watermarked_value
            )
        )

        if (
            original_magnitude
            < EPSILON
        ):

            relative_change = 0.0

        else:

            relative_change = (
                watermarked_magnitude
                - original_magnitude
            ) / original_magnitude

        if relative_change > 0:
            extracted_bit = 1

        elif relative_change < 0:
            extracted_bit = 0

        else:
            extracted_bit = 0

        expected_bit = int(
            embedded_bits[
                bit_index - 1
            ].item()
        )

        correct = (
            extracted_bit
            == expected_bit
        )

        extracted_bits.append(
            extracted_bit
        )

        extraction_records.append(
            {
                "bit_index":
                    bit_index,

                "frequency_bin":
                    frequency_bin,

                "time_index":
                    time_index,

                "original_magnitude":
                    original_magnitude,

                "watermarked_magnitude":
                    watermarked_magnitude,

                "relative_change":
                    relative_change,

                "expected_bit":
                    expected_bit,

                "extracted_bit":
                    extracted_bit,

                "correct":
                    correct,
            }
        )

    extracted_bits = torch.tensor(
        extracted_bits,
        dtype=torch.uint8,
    )

    records_df = pd.DataFrame(
        extraction_records
    )

    records_df.to_csv(
        RESULT_DIR
        / "extraction_details.csv",
        index=False,
    )

    torch.save(
        extracted_bits,
        RESULT_DIR
        / "extracted_bits.pt",
    )

    # --------------------------------------------------
    # Accuracy
    # --------------------------------------------------

    total_bits = len(
        extracted_bits
    )

    correct_bits = int(
        (
            extracted_bits
            == embedded_bits
        ).sum().item()
    )

    incorrect_bits = (
        total_bits
        - correct_bits
    )

    accuracy = (
        correct_bits
        / total_bits
        if total_bits > 0
        else 0.0
    )

    bit_error_rate = (
        incorrect_bits
        / total_bits
        if total_bits > 0
        else 0.0
    )

    extracted_string = "".join(
        str(int(bit))
        for bit in extracted_bits
    )

    with open(
        RESULT_DIR
        / "extracted_bits.txt",
        "w",
    ) as f:

        f.write(
            extracted_string
        )

    summary = {

        "total_bits":
            total_bits,

        "correct_bits":
            correct_bits,

        "incorrect_bits":
            incorrect_bits,

        "bit_accuracy":
            accuracy,

        "bit_error_rate":
            bit_error_rate,

        "status":
            "completed",
    }

    with open(
        RESULT_DIR
        / "extraction_summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
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
        f"{accuracy:.6f}"
    )

    print(
        f"Bit Error Rate   : "
        f"{bit_error_rate:.6f}"
    )

    print()

    print(
        "Results Saved"
    )

    print(
        RESULT_DIR
        / "extraction_details.csv"
    )

    print(
        RESULT_DIR
        / "extracted_bits.pt"
    )

    print(
        RESULT_DIR
        / "extracted_bits.txt"
    )

    print(
        RESULT_DIR
        / "extraction_summary.json"
    )


if __name__ == "__main__":
    main()