"""
SemBind-Audio
Experiment 12

Semantic Selection Validation

Purpose
-------
Validate whether the regions selected for watermark embedding
actually correspond to lower semantic importance according to
the HuBERT-derived semantic contribution scores.

Core formulation
----------------

D_t = 1 - cosine(g, g_-t)

S_t = (D_t - min(D)) / (max(D) - min(D))

Selected regions:
    S_t <= Q_25(S)

The experiment compares:

1. Selected semantic frames
2. Non-selected semantic frames

and reports:

    mean semantic score
    median semantic score
    minimum
    maximum
    standard deviation
    semantic separation
    selected/excluded ratio
"""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import torch


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# PATHS
# =========================================================

RESULTS_DIR = PROJECT_ROOT / "results"

CONTRIBUTION_DIR = (
    RESULTS_DIR / "contribution"
)

REGION_DIR = (
    RESULTS_DIR / "region_selection"
)

OUTPUT_DIR = (
    RESULTS_DIR / "semantic_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# INPUT FILES
# =========================================================

SEMANTIC_SCORE_FILE = (
    REGION_DIR
    / "semantic_scores.csv"
)

SELECTED_FRAMES_FILE = (
    REGION_DIR
    / "selected_frames.csv"
)


# =========================================================
# OUTPUT FILES
# =========================================================

VALIDATION_CSV = (
    OUTPUT_DIR
    / "semantic_selection_validation.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "semantic_selection_validation.json"
)


# =========================================================
# LOAD SEMANTIC SCORES
# =========================================================

def load_semantic_scores():

    if not SEMANTIC_SCORE_FILE.exists():

        raise FileNotFoundError(
            "Semantic score file not found:\n"
            f"{SEMANTIC_SCORE_FILE}"
        )

    scores = pd.read_csv(
        SEMANTIC_SCORE_FILE
    )

    if scores.empty:

        raise ValueError(
            "Semantic score file is empty."
        )

    print(
        f"Loaded semantic score rows : "
        f"{len(scores)}"
    )

    print(
        f"Columns                    : "
        f"{list(scores.columns)}"
    )

    return scores


# =========================================================
# IDENTIFY SCORE COLUMN
# =========================================================

def find_score_column(scores):

    possible_columns = [

        "normalized_score",

        "semantic_score",

        "score",

        "semantic_shift",
    ]

    for column in possible_columns:

        if column in scores.columns:

            return column

    raise ValueError(
        "Could not identify semantic score column.\n"
        f"Available columns: {list(scores.columns)}"
    )


# =========================================================
# IDENTIFY FRAME COLUMN
# =========================================================

def find_frame_column(scores):

    possible_columns = [

        "frame",

        "frame_index",

        "frame_id",
    ]

    for column in possible_columns:

        if column in scores.columns:

            return column

    raise ValueError(
        "Could not identify frame column.\n"
        f"Available columns: {list(scores.columns)}"
    )


# =========================================================
# LOAD SELECTED FRAMES
# =========================================================

def load_selected_frames():

    if not SELECTED_FRAMES_FILE.exists():

        raise FileNotFoundError(
            "Selected frame file not found:\n"
            f"{SELECTED_FRAMES_FILE}"
        )

    selected = pd.read_csv(
        SELECTED_FRAMES_FILE
    )

    if selected.empty:

        raise ValueError(
            "Selected frame file is empty."
        )

    print(
        f"Selected frame rows        : "
        f"{len(selected)}"
    )

    print(
        f"Columns                    : "
        f"{list(selected.columns)}"
    )

    return selected


# =========================================================
# EXTRACT FRAME IDS
# =========================================================

def extract_selected_frame_ids(
    selected,
):

    frame_column = find_frame_column(
        selected
    )

    frame_ids = set(
        selected[
            frame_column
        ]
        .astype(int)
        .tolist()
    )

    return frame_ids


# =========================================================
# STATISTICS
# =========================================================

def statistics(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {

        "count":
            int(len(values)),

        "minimum":
            float(np.min(values)),

        "maximum":
            float(np.max(values)),

        "mean":
            float(np.mean(values)),

        "median":
            float(np.median(values)),

        "std":
            float(np.std(values)),

    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)

    print(
        "SemBind-Audio : Experiment 12"
    )

    print(
        "Semantic Selection Validation"
    )

    print("=" * 60)

    # -----------------------------------------------------
    # Load semantic contribution scores
    # -----------------------------------------------------

    scores = load_semantic_scores()

    score_column = find_score_column(
        scores
    )

    frame_column = find_frame_column(
        scores
    )

    # -----------------------------------------------------
    # Load selected semantic frames
    # -----------------------------------------------------

    selected = load_selected_frames()

    selected_frame_ids = (
        extract_selected_frame_ids(
            selected
        )
    )

    # -----------------------------------------------------
    # Normalize frame indexing
    # -----------------------------------------------------

    scores[frame_column] = (
        scores[frame_column]
        .astype(int)
    )

    scores[score_column] = (
        scores[score_column]
        .astype(float)
    )

    # -----------------------------------------------------
    # Determine selected / excluded frames
    # -----------------------------------------------------

    scores["selected"] = (
        scores[frame_column]
        .isin(selected_frame_ids)
    )

    selected_scores = (
        scores.loc[
            scores["selected"],
            score_column,
        ]
        .to_numpy()
    )

    excluded_scores = (
        scores.loc[
            ~scores["selected"],
            score_column,
        ]
        .to_numpy()
    )

    # -----------------------------------------------------
    # Safety checks
    # -----------------------------------------------------

    if len(selected_scores) == 0:

        raise RuntimeError(
            "No selected semantic frames "
            "were matched."
        )

    if len(excluded_scores) == 0:

        raise RuntimeError(
            "No excluded semantic frames "
            "were available."
        )

    # -----------------------------------------------------
    # Calculate statistics
    # -----------------------------------------------------

    selected_stats = statistics(
        selected_scores
    )

    excluded_stats = statistics(
        excluded_scores
    )

    # -----------------------------------------------------
    # Semantic separation
    # -----------------------------------------------------

    semantic_separation = (
        excluded_stats["mean"]
        -
        selected_stats["mean"]
    )

    # -----------------------------------------------------
    # Selected / excluded ratio
    # -----------------------------------------------------

    if excluded_stats["mean"] != 0:

        semantic_ratio = (
            selected_stats["mean"]
            /
            excluded_stats["mean"]
        )

    else:

        semantic_ratio = float("inf")

    # -----------------------------------------------------
    # Percentile threshold
    # -----------------------------------------------------

    threshold = float(
        np.percentile(
            scores[score_column].to_numpy(),
            25,
        )
    )

    # -----------------------------------------------------
    # Verify selection condition
    # -----------------------------------------------------

    threshold_violations = int(
        np.sum(
            selected_scores
            >
            threshold
        )
    )

    # -----------------------------------------------------
    # Create frame-level validation table
    # -----------------------------------------------------

    validation = scores[
        [
            frame_column,
            score_column,
            "selected",
        ]
    ].copy()

    validation["group"] = np.where(
        validation["selected"],
        "selected",
        "excluded",
    )

    validation.to_csv(
        VALIDATION_CSV,
        index=False,
    )

    # -----------------------------------------------------
    # Build summary
    # -----------------------------------------------------

    summary = {

        "experiment":
            "Experiment 12",

        "purpose":
            "Validate that semantic-guided "
            "selected regions have lower "
            "HuBERT-derived semantic importance.",

        "semantic_score_formula":
            "D_t = 1 - cosine(g, g_-t)",

        "normalization_formula":
            "S_t = (D_t - min(D)) / "
            "(max(D) - min(D))",

        "selection_formula":
            "S_t <= Q_25(S)",

        "score_column":
            score_column,

        "frame_column":
            frame_column,

        "total_frames":
            int(len(scores)),

        "selected_frames":
            int(len(selected_scores)),

        "excluded_frames":
            int(len(excluded_scores)),

        "selection_percentage":
            float(
                100
                * len(selected_scores)
                /
                len(scores)
            ),

        "semantic_threshold":
            threshold,

        "selected_statistics":
            selected_stats,

        "excluded_statistics":
            excluded_stats,

        "semantic_separation":
            float(
                semantic_separation
            ),

        "selected_excluded_ratio":
            float(
                semantic_ratio
            ),

        "threshold_violations":
            threshold_violations,

        "hypothesis":
            "Selected regions should have "
            "lower semantic importance than "
            "excluded regions.",

        "hypothesis_supported":
            bool(
                selected_stats["mean"]
                <
                excluded_stats["mean"]
            ),
    }

    # -----------------------------------------------------
    # Save summary
    # -----------------------------------------------------

    with open(
        SUMMARY_JSON,
        "w",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print()

    print(
        "Semantic Selection Statistics"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Total Frames           : "
        f"{len(scores)}"
    )

    print(
        f"Selected Frames        : "
        f"{len(selected_scores)}"
    )

    print(
        f"Excluded Frames        : "
        f"{len(excluded_scores)}"
    )

    print()

    print(
        f"25th Percentile        : "
        f"{threshold:.8f}"
    )

    print()

    print(
        "Selected Regions"
    )

    print(
        f"Mean                   : "
        f"{selected_stats['mean']:.8f}"
    )

    print(
        f"Median                 : "
        f"{selected_stats['median']:.8f}"
    )

    print(
        f"Minimum                : "
        f"{selected_stats['minimum']:.8f}"
    )

    print(
        f"Maximum                : "
        f"{selected_stats['maximum']:.8f}"
    )

    print(
        f"Std                    : "
        f"{selected_stats['std']:.8f}"
    )

    print()

    print(
        "Excluded Regions"
    )

    print(
        f"Mean                   : "
        f"{excluded_stats['mean']:.8f}"
    )

    print(
        f"Median                 : "
        f"{excluded_stats['median']:.8f}"
    )

    print(
        f"Minimum                : "
        f"{excluded_stats['minimum']:.8f}"
    )

    print(
        f"Maximum                : "
        f"{excluded_stats['maximum']:.8f}"
    )

    print(
        f"Std                    : "
        f"{excluded_stats['std']:.8f}"
    )

    print()

    print(
        "Semantic Separation"
    )

    print(
        f"Excluded Mean - "
        f"Selected Mean : "
        f"{semantic_separation:.8f}"
    )

    print(
        f"Selected / Excluded Ratio : "
        f"{semantic_ratio:.8f}"
    )

    print()

    if summary[
        "hypothesis_supported"
    ]:

        print(
            "Hypothesis Result : SUPPORTED"
        )

        print(
            "Selected regions have lower "
            "semantic importance."
        )

    else:

        print(
            "Hypothesis Result : NOT SUPPORTED"
        )

        print(
            "Selected regions are not "
            "semantically lower on average."
        )

    print()

    print(
        f"Threshold Violations : "
        f"{threshold_violations}"
    )

    print()

    print(
        "Results Saved"
    )

    print(
        VALIDATION_CSV
    )

    print(
        SUMMARY_JSON
    )


if __name__ == "__main__":
    main()