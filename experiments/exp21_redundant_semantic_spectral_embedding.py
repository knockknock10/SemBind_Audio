"""
SemBind-Audio : Experiment 21
Redundant Semantic-Spectral Adaptive STFT Watermark Embedding

Research purpose
----------------
Exp19 established the joint semantic-spectral embedding mechanism.

Exp21 improves robustness by introducing spatial redundancy:

    1 payload bit -> REDUNDANCY independent STFT coefficients

The same bit is embedded into multiple coefficients and can later be
recovered using majority voting.

Core SemBind-Audio principle
----------------------------
HuBERT semantic importance determines WHERE embedding is permitted.

Local spectral suitability determines which coefficients are preferable.

Joint suitability:

    J_r = S_r^alpha * P_r^beta

Adaptive embedding strength:

    delta_r =
        delta_min +
        (delta_max - delta_min) * J_r

Embedding:

    bit = 1 -> real(X') = real(X) + delta
    bit = 0 -> real(X') = real(X) - delta

Important methodological rule
-----------------------------
This experiment ALWAYS starts from:

    results/original_audio.wav

It never reuses a previously watermarked waveform.

Robustness improvement
-----------------------
The three replicas of a bit are selected from different coefficients.

The allocator:

1. Builds all valid semantic-spectral candidates.
2. Removes duplicate STFT coordinates.
3. Rejects extremely weak coefficients.
4. Ranks candidates by joint suitability and local magnitude.
5. Attempts spatial separation.
6. Gradually relaxes separation if necessary.
7. Ensures every payload bit receives exactly REDUNDANCY replicas.
8. Stores the exact coordinates needed by the later extraction experiment.

This is an experimental research implementation.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

from configs.config import (
    ALPHA,
    BETA,
    DELTA_MIN,
    DELTA_MAX,
)

SAMPLE_RATE = 16000

N_FFT = 1024
HOP_LENGTH = 320
WIN_LENGTH = 1024

WINDOW = torch.hann_window(WIN_LENGTH)


# ============================================================================
# INPUT FILES
# ============================================================================

PAYLOAD_PATH = (
    RESULTS_DIR
    / "watermark"
    / "payload_bits.pt"
)

ORIGINAL_AUDIO_PATH = (
    RESULTS_DIR
    / "original_audio.wav"
)

SEMANTIC_SCORE_PATH = (
    RESULTS_DIR
    / "semantic_validation"
    / "semantic_selection_validation.csv"
)


# ============================================================================
# OUTPUT FILES
# ============================================================================

OUTPUT_DIR = (
    RESULTS_DIR
    / "redundant_embedding"
)

ORIGINAL_STFT_PATH = (
    OUTPUT_DIR
    / "original_stft.pt"
)

WATERMARKED_STFT_PATH = (
    OUTPUT_DIR
    / "redundant_watermarked_stft.pt"
)

EMBEDDED_BITS_PATH = (
    OUTPUT_DIR
    / "embedded_bits.pt"
)

COORDINATES_PATH = (
    OUTPUT_DIR
    / "redundant_embedding_coordinates.csv"
)

REGION_PARAMETERS_PATH = (
    OUTPUT_DIR
    / "redundant_region_parameters.csv"
)

WATERMARKED_AUDIO_PATH = (
    OUTPUT_DIR
    / "redundant_watermarked_audio.wav"
)

DIFFERENCE_AUDIO_PATH = (
    OUTPUT_DIR
    / "redundant_difference.wav"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "redundant_embedding_summary.json"
)


# ============================================================================
# RESEARCH PARAMETERS
# ============================================================================

# ---------------------------------------------------------------------------
# Redundancy
# ---------------------------------------------------------------------------

REDUNDANCY = 3


# ---------------------------------------------------------------------------
# Semantic-spectral weighting
# ---------------------------------------------------------------------------

ALPHA = 0.70
BETA = 0.30


# ---------------------------------------------------------------------------
# Adaptive embedding strength
# ---------------------------------------------------------------------------
# Values from central configuration:
#     DELTA_MIN = 0.002, DELTA_MAX = 0.008
# (Local assignments removed; using config values directly.)


# ---------------------------------------------------------------------------
# Frequency range
#
# Avoid DC and Nyquist edge.
# ---------------------------------------------------------------------------

FREQ_START_RATIO = 0.05
FREQ_END_RATIO = 0.95


# ---------------------------------------------------------------------------
# Semantic region filtering
# ---------------------------------------------------------------------------

MIN_REGION_FRAMES = 2


# ---------------------------------------------------------------------------
# Spectral candidate filtering
#
# IMPORTANT:
# This is deliberately less aggressive than the previous implementation.
# The old 0.10 global min-max threshold could discard too many coefficients
# after regional selection.
# ---------------------------------------------------------------------------

MIN_NORMALIZED_MAGNITUDE = 0.02


# ---------------------------------------------------------------------------
# Preferred spatial separation
#
# A replica should preferably not be immediately adjacent to another
# replica in both dimensions.
# ---------------------------------------------------------------------------

PREFERRED_TIME_SEPARATION = 3
PREFERRED_FREQ_SEPARATION = 8


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

EPS = 1e-12


# ============================================================================
# FILE UTILITIES
# ============================================================================

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def require_file(
    path: Path,
    description: str,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired {description} not found:\n"
            f"{path}\n"
        )


# ============================================================================
# PAYLOAD
# ============================================================================

def load_payload() -> torch.Tensor:

    require_file(
        PAYLOAD_PATH,
        "payload",
    )

    payload = torch.load(
        PAYLOAD_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        payload,
        torch.Tensor,
    ):
        payload = torch.tensor(payload)

    payload = (
        payload
        .detach()
        .cpu()
        .flatten()
        .to(torch.int64)
    )

    if payload.numel() == 0:

        raise ValueError(
            "Payload contains zero bits."
        )

    payload = (
        payload != 0
    ).to(torch.int64)

    return payload


# ============================================================================
# AUDIO
# ============================================================================

def load_original_audio() -> Tuple[torch.Tensor, int]:

    require_file(
        ORIGINAL_AUDIO_PATH,
        "original audio",
    )

    audio_np, sample_rate = sf.read(
        str(ORIGINAL_AUDIO_PATH),
        dtype="float32",
        always_2d=False,
    )

    audio_np = np.asarray(
        audio_np,
        dtype=np.float32,
    )

    audio = torch.from_numpy(
        audio_np
    )

    # Convert stereo/multichannel to mono.
    if audio.ndim == 2:

        audio = audio.mean(
            dim=1
        )

    audio = (
        audio
        .contiguous()
        .float()
        .flatten()
    )

    if sample_rate != SAMPLE_RATE:

        raise ValueError(
            f"Expected sample rate "
            f"{SAMPLE_RATE}, got "
            f"{sample_rate}."
        )

    return (
        audio,
        int(sample_rate),
    )


def compute_stft(
    audio: torch.Tensor,
) -> torch.Tensor:

    audio = (
        audio
        .float()
        .flatten()
    )

    return torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        center=True,
        return_complex=True,
    )


def reconstruct_audio(
    stft: torch.Tensor,
    length: int,
) -> torch.Tensor:

    return torch.istft(
        stft,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        center=True,
        length=length,
    ).float()


# ============================================================================
# SEMANTIC SCORE LOADING
# ============================================================================

def load_semantic_scores() -> List[Dict]:

    require_file(
        SEMANTIC_SCORE_PATH,
        "semantic validation CSV",
    )

    with SEMANTIC_SCORE_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

        fields = (
            reader.fieldnames
            or []
        )

    required = {
        "frame",
        "normalized_score",
        "selected",
    }

    missing = (
        required
        - set(fields)
    )

    if missing:

        raise ValueError(
            "\nSemantic CSV is missing required columns:\n"
            f"{sorted(missing)}\n\n"
            f"Available columns:\n"
            f"{fields}\n"
        )

    parsed = []

    for row in rows:

        frame = int(
            float(
                row["frame"]
            )
        )

        score = float(
            row["normalized_score"]
        )

        if not math.isfinite(score):

            raise ValueError(
                f"Invalid semantic score "
                f"at frame {frame}: {score}"
            )

        selected_raw = (
            str(
                row["selected"]
            )
            .strip()
            .lower()
        )

        selected = (
            selected_raw
            in {
                "1",
                "true",
                "yes",
                "selected",
            }
        )

        parsed.append(
            {
                "frame": frame,
                "normalized_score": score,
                "selected": selected,
            }
        )

    parsed.sort(
        key=lambda x:
            x["frame"]
    )

    if not parsed:

        raise ValueError(
            "Semantic CSV contains no rows."
        )

    return parsed


# ============================================================================
# SEMANTIC REGION CONSTRUCTION
# ============================================================================

def merge_selected_regions(
    semantic_rows: List[Dict],
) -> List[Dict]:

    selected_frames = sorted(
        {
            row["frame"]
            for row in semantic_rows
            if row["selected"]
        }
    )

    if not selected_frames:

        raise ValueError(
            "No selected semantic frames "
            "were found."
        )

    score_lookup = {
        row["frame"]:
            row["normalized_score"]
        for row in semantic_rows
    }

    runs = []

    start = selected_frames[0]
    previous = selected_frames[0]

    for frame in selected_frames[1:]:

        if frame == previous + 1:

            previous = frame

        else:

            runs.append(
                (
                    start,
                    previous,
                )
            )

            start = frame
            previous = frame

    runs.append(
        (
            start,
            previous,
        )
    )

    regions = []

    region_id = 1

    for start, end in runs:

        length = (
            end
            -
            start
            +
            1
        )

        if length < MIN_REGION_FRAMES:

            continue

        scores = [
            score_lookup[frame]
            for frame in range(
                start,
                end + 1,
            )
            if frame in score_lookup
        ]

        if not scores:

            continue

        regions.append(
            {
                "region_id":
                    region_id,

                "semantic_start":
                    start,

                "semantic_end":
                    end,

                "semantic_frames":
                    length,

                "semantic_score":
                    float(
                        np.mean(
                            scores
                        )
                    ),
            }
        )

        region_id += 1

    if not regions:

        raise ValueError(
            "No usable semantic regions."
        )

    return regions


# ============================================================================
# TEMPORAL MAPPING
# ============================================================================

def map_frame(
    frame: int,
    semantic_count: int,
    stft_count: int,
) -> int:

    if semantic_count <= 1:
        return 0

    if stft_count <= 1:
        return 0

    position = (
        frame
        /
        float(
            semantic_count - 1
        )
    )

    mapped = round(
        position
        *
        float(
            stft_count - 1
        )
    )

    return max(
        0,
        min(
            stft_count - 1,
            mapped,
        ),
    )


def map_regions(
    regions: List[Dict],
    semantic_count: int,
    stft_count: int,
) -> List[Dict]:

    result = []

    for region in regions:

        start = map_frame(
            region["semantic_start"],
            semantic_count,
            stft_count,
        )

        end = map_frame(
            region["semantic_end"],
            semantic_count,
            stft_count,
        )

        if end < start:

            start, end = (
                end,
                start,
            )

        item = dict(
            region
        )

        item["stft_start"] = start
        item["stft_end"] = end

        result.append(
            item
        )

    return result


# ============================================================================
# SPECTRAL STATISTICS
# ============================================================================

def spectral_statistics(
    stft: torch.Tensor,
    region: Dict,
) -> Tuple[float, float]:

    freq_bins = stft.shape[0]
    time_bins = stft.shape[1]

    freq_start = max(
        1,
        int(
            freq_bins
            *
            FREQ_START_RATIO
        ),
    )

    freq_end = min(
        freq_bins - 2,
        int(
            freq_bins
            *
            FREQ_END_RATIO
        ),
    )

    time_start = max(
        0,
        int(
            region["stft_start"]
        ),
    )

    time_end = min(
        time_bins - 1,
        int(
            region["stft_end"]
        ),
    )

    block = stft[
        freq_start:
        freq_end + 1,
        time_start:
        time_end + 1,
    ]

    magnitude = torch.abs(
        block
    )

    log_magnitude = torch.log1p(
        magnitude
    )

    return (
        float(
            log_magnitude
            .mean()
            .item()
        ),
        float(
            magnitude
            .mean()
            .item()
        ),
    )


def minmax(
    values: List[float],
) -> List[float]:

    if not values:

        return []

    low = min(values)
    high = max(values)

    if abs(
        high - low
    ) < EPS:

        return [
            1.0
            for _ in values
        ]

    return [
        (
            value - low
        )
        /
        (
            high - low
        )
        for value in values
    ]


# ============================================================================
# REGION PARAMETERS
# ============================================================================

def calculate_region_parameters(
    stft: torch.Tensor,
    regions: List[Dict],
) -> List[Dict]:

    semantic_suitability = [
        1.0
        -
        float(
            region[
                "semantic_score"
            ]
        )
        for region in regions
    ]

    spectral_values = []

    spectral_raw = []

    for region in regions:

        log_mean, magnitude_mean = (
            spectral_statistics(
                stft,
                region,
            )
        )

        spectral_values.append(
            log_mean
        )

        spectral_raw.append(
            (
                log_mean,
                magnitude_mean,
            )
        )

    semantic_norm = minmax(
        semantic_suitability
    )

    spectral_norm = minmax(
        spectral_values
    )

    results = []

    freq_start = max(
        1,
        int(
            stft.shape[0]
            *
            FREQ_START_RATIO
        ),
    )

    freq_end = min(
        stft.shape[0] - 2,
        int(
            stft.shape[0]
            *
            FREQ_END_RATIO
        ),
    )

    for index, region in enumerate(
        regions
    ):

        s = max(
            semantic_norm[index],
            EPS,
        )

        p = max(
            spectral_norm[index],
            EPS,
        )

        joint = (
            (s ** ALPHA)
            *
            (p ** BETA)
        )

        delta = (
            DELTA_MIN
            +
            (
                DELTA_MAX
                -
                DELTA_MIN
            )
            *
            joint
        )

        item = dict(
            region
        )

        item.update(
            {
                "semantic_suitability":
                    float(
                        semantic_norm[
                            index
                        ]
                    ),

                "spectral_log_mean":
                    float(
                        spectral_raw[
                            index
                        ][0]
                    ),

                "spectral_mean":
                    float(
                        spectral_raw[
                            index
                        ][1]
                    ),

                "spectral_suitability":
                    float(
                        spectral_norm[
                            index
                        ]
                    ),

                "joint_suitability":
                    float(
                        joint
                    ),

                "adaptive_strength":
                    float(
                        delta
                    ),

                "freq_start":
                    freq_start,

                "freq_end":
                    freq_end,
            }
        )

        results.append(
            item
        )

    return results


# ============================================================================
# CAPACITY
# ============================================================================

def calculate_capacity(
    regions: List[Dict],
) -> int:

    total = 0

    for region in regions:

        time_count = (
            region["stft_end"]
            -
            region["stft_start"]
            +
            1
        )

        freq_count = (
            region["freq_end"]
            -
            region["freq_start"]
            +
            1
        )

        total += (
            time_count
            *
            freq_count
        )

    return int(
        total
    )


# ============================================================================
# CANDIDATE GENERATION
# ============================================================================

def build_candidates(
    stft: torch.Tensor,
    regions: List[Dict],
) -> List[Dict]:

    candidates = []

    seen = set()

    for region in regions:

        for t in range(
            region["stft_start"],
            region["stft_end"] + 1,
        ):

            for f in range(
                region["freq_start"],
                region["freq_end"] + 1,
            ):

                # --------------------------------------------------------
                # Protect against duplicate coordinates when semantic
                # regions touch/overlap after temporal mapping.
                # --------------------------------------------------------

                key = (
                    t,
                    f,
                )

                if key in seen:
                    continue

                seen.add(key)

                coefficient = stft[
                    f,
                    t
                ]

                magnitude = float(
                    torch.abs(
                        coefficient
                    ).item()
                )

                normalized_placeholder = 0.0

                candidates.append(
                    {
                        "region":
                            region,

                        "time_frame":
                            int(t),

                        "frequency_bin":
                            int(f),

                        "magnitude":
                            magnitude,

                        "normalized_magnitude":
                            normalized_placeholder,
                    }
                )

    if not candidates:

        raise ValueError(
            "No STFT candidates "
            "were generated."
        )

    # -----------------------------------------------------------------------
    # Normalize magnitude globally over valid candidates.
    # -----------------------------------------------------------------------

    magnitudes = [
        item["magnitude"]
        for item in candidates
    ]

    low = min(
        magnitudes
    )

    high = max(
        magnitudes
    )

    if (
        high - low
    ) < EPS:

        for item in candidates:

            item[
                "normalized_magnitude"
            ] = 1.0

    else:

        for item in candidates:

            item[
                "normalized_magnitude"
            ] = (
                item["magnitude"]
                -
                low
            ) / (
                high
                -
                low
            )

    # -----------------------------------------------------------------------
    # Weak coefficient filtering.
    #
    # If filtering leaves too few candidates, progressively relax it.
    # -----------------------------------------------------------------------

    thresholds = [
        MIN_NORMALIZED_MAGNITUDE,
        0.01,
        0.0,
    ]

    required_candidates = None

    # required_candidates is intentionally determined later by allocator.
    # Here we simply select the strongest valid threshold.
    for threshold in thresholds:

        filtered = [
            item
            for item in candidates
            if (
                item[
                    "normalized_magnitude"
                ]
                >= threshold
            )
        ]

        if filtered:

            candidates = filtered
            break

    # -----------------------------------------------------------------------
    # Rank candidates.
    #
    # Region suitability is the primary criterion.
    # Local magnitude is the secondary criterion.
    # -----------------------------------------------------------------------

    candidates.sort(
        key=lambda item: (
            -float(
                item[
                    "region"
                ][
                    "joint_suitability"
                ]
            ),
            -float(
                item[
                    "normalized_magnitude"
                ]
            ),
            item[
                "time_frame"
            ],
            item[
                "frequency_bin"
            ],
        )
    )

    return candidates


# ============================================================================
# SPATIAL SEPARATION
# ============================================================================

def spatially_separated(
    candidate: Dict,
    selected: List[Dict],
    time_distance: int,
    freq_distance: int,
) -> bool:

    t = candidate[
        "time_frame"
    ]

    f = candidate[
        "frequency_bin"
    ]

    for other in selected:

        ot = other[
            "time_frame"
        ]

        of = other[
            "frequency_bin"
        ]

        dt = abs(
            t - ot
        )

        df = abs(
            f - of
        )

        # Reject only when the candidate is close in BOTH dimensions.
        if (
            dt < time_distance
            and
            df < freq_distance
        ):

            return False

    return True


# ============================================================================
# REDUNDANT ALLOCATION
# ============================================================================

def allocate_redundant_coordinates(
    payload: torch.Tensor,
    candidates: List[Dict],
) -> List[Dict]:

    payload_length = int(
        payload.numel()
    )

    required = (
        payload_length
        *
        REDUNDANCY
    )

    if len(candidates) < required:

        raise ValueError(
            "\nInsufficient candidate "
            "coefficients.\n"
            f"Required : {required}\n"
            f"Available: {len(candidates)}\n"
        )

    # -----------------------------------------------------------------------
    # We allocate replicas bit-by-bit.
    #
    # For each payload bit we first try to place replicas far apart.
    # If that is impossible, we relax the spatial constraint.
    #
    # This prevents the previous failure where 768 coordinates existed
    # conceptually but the allocator could not find enough candidates.
    # -----------------------------------------------------------------------

    separation_levels = [
        (
            PREFERRED_TIME_SEPARATION,
            PREFERRED_FREQ_SEPARATION,
        ),
        (
            2,
            4,
        ),
        (
            1,
            2,
        ),
        (
            0,
            0,
        ),
    ]

    unused = list(
        candidates
    )

    selected_global = []

    allocation = {
        bit_index: []
        for bit_index in range(
            payload_length
        )
    }

    # -----------------------------------------------------------------------
    # We perform allocation in rounds.
    #
    # Each round attempts to give every payload bit one replica.
    # This guarantees that redundancy is balanced rather than giving the
    # first few bits all the strongest coefficients.
    # -----------------------------------------------------------------------

    for replica_index in range(
        REDUNDANCY
    ):

        for bit_index in range(
            payload_length
        ):

            if not unused:

                break

            # ---------------------------------------------------------------
            # Prefer candidates spatially separated from:
            #   1. previous replicas of the same bit
            #   2. already selected coefficients
            # ---------------------------------------------------------------

            previous_replicas = allocation[
                bit_index
            ]

            selected_candidate = None
            selected_position = None

            for (
                time_sep,
                freq_sep,
            ) in separation_levels:

                for position, candidate in enumerate(
                    unused
                ):

                    # Same coordinate can never be reused.
                    if candidate in selected_global:
                        continue

                    # Prefer separation from previous replicas of this bit.
                    if not spatially_separated(
                        candidate,
                        previous_replicas,
                        time_sep,
                        freq_sep,
                    ):
                        continue

                    # Also avoid excessive clustering with global
                    # selections whenever possible.
                    if not spatially_separated(
                        candidate,
                        selected_global,
                        max(
                            1,
                            time_sep // 2,
                        ),
                        max(
                            1,
                            freq_sep // 2,
                        ),
                    ):
                        continue

                    selected_candidate = candidate
                    selected_position = position
                    break

                if selected_candidate is not None:
                    break

            # ---------------------------------------------------------------
            # If strict separation failed, choose the best remaining
            # candidate. Capacity is more important than an artificial
            # separation constraint.
            # ---------------------------------------------------------------

            if selected_candidate is None:

                if not unused:
                    break

                selected_candidate = unused[0]
                selected_position = 0

            # ---------------------------------------------------------------
            # Commit candidate.
            # ---------------------------------------------------------------

            unused.pop(
                selected_position
            )

            selected_global.append(
                selected_candidate
            )

            allocation[
                bit_index
            ].append(
                selected_candidate
            )

    # -----------------------------------------------------------------------
    # Validate allocation.
    # -----------------------------------------------------------------------

    incomplete = [
        bit_index
        for bit_index, group
        in allocation.items()
        if len(group) != REDUNDANCY
    ]

    if incomplete:

        raise RuntimeError(
            "\nRedundant allocation failed.\n"
            f"Required replicas per bit : "
            f"{REDUNDANCY}\n"
            f"Payload bits             : "
            f"{payload_length}\n"
            f"Incomplete bits          : "
            f"{incomplete[:20]}\n"
            f"Allocated coordinates    : "
            f"{len(selected_global)}\n"
            f"Required coordinates     : "
            f"{required}\n"
        )

    # -----------------------------------------------------------------------
    # Convert allocation into experiment records.
    # -----------------------------------------------------------------------

    coordinates = []

    for bit_index in range(
        payload_length
    ):

        bit = int(
            payload[
                bit_index
            ].item()
        )

        group = allocation[
            bit_index
        ]

        for replica_index, candidate in enumerate(
            group
        ):

            region = candidate[
                "region"
            ]

            coordinates.append(
                {
                    "bit_index":
                        bit_index,

                    "replica_index":
                        replica_index,

                    "payload_bit":
                        bit,

                    "region_id":
                        region[
                            "region_id"
                        ],

                    "time_frame":
                        candidate[
                            "time_frame"
                        ],

                    "frequency_bin":
                        candidate[
                            "frequency_bin"
                        ],

                    "magnitude":
                        candidate[
                            "magnitude"
                        ],

                    "normalized_magnitude":
                        candidate[
                            "normalized_magnitude"
                        ],

                    "semantic_score":
                        region[
                            "semantic_score"
                        ],

                    "semantic_suitability":
                        region[
                            "semantic_suitability"
                        ],

                    "spectral_suitability":
                        region[
                            "spectral_suitability"
                        ],

                    "joint_suitability":
                        region[
                            "joint_suitability"
                        ],

                    "adaptive_strength":
                        region[
                            "adaptive_strength"
                        ],
                }
            )

    if len(coordinates) != required:

        raise RuntimeError(
            "\nFinal coordinate count mismatch.\n"
            f"Expected : {required}\n"
            f"Actual   : {len(coordinates)}\n"
        )

    return coordinates


# ============================================================================
# EMBEDDING
# ============================================================================

def embed_payload(
    original_stft: torch.Tensor,
    coordinates: List[Dict],
) -> Tuple[
    torch.Tensor,
    List[Dict],
]:

    watermarked = (
        original_stft.clone()
    )

    records = []

    for coordinate in coordinates:

        bit = int(
            coordinate[
                "payload_bit"
            ]
        )

        f = int(
            coordinate[
                "frequency_bin"
            ]
        )

        t = int(
            coordinate[
                "time_frame"
            ]
        )

        strength = float(
            coordinate[
                "adaptive_strength"
            ]
        )

        original_value = (
            watermarked[
                f,
                t
            ]
        )

        original_real = float(
            original_value.real.item()
        )

        # ---------------------------------------------------------------
        # Signed embedding.
        # ---------------------------------------------------------------

        if bit == 1:

            new_real = (
                original_real
                +
                strength
            )

        else:

            new_real = (
                original_real
                -
                strength
            )

        new_value = torch.complex(
            torch.tensor(
                new_real,
                dtype=original_value.real.dtype,
            ),
            original_value.imag,
        )

        watermarked[
            f,
            t
        ] = new_value

        record = dict(
            coordinate
        )

        record.update(
            {
                "original_real":
                    original_real,

                "watermarked_real":
                    new_real,

                "delta_change":
                    abs(
                        new_real
                        -
                        original_real
                    ),
            }
        )

        records.append(
            record
        )

    return (
        watermarked,
        records,
    )


# ============================================================================
# QUALITY METRICS
# ============================================================================

def quality_metrics(
    original: torch.Tensor,
    watermarked: torch.Tensor,
) -> Dict[str, float]:

    original = (
        original
        .float()
        .flatten()
    )

    watermarked = (
        watermarked
        .float()
        .flatten()
    )

    n = min(
        original.numel(),
        watermarked.numel(),
    )

    original = original[
        :n
    ]

    watermarked = watermarked[
        :n
    ]

    difference = (
        watermarked
        -
        original
    )

    mse = float(
        torch.mean(
            difference ** 2
        ).item()
    )

    rmse = math.sqrt(
        mse
    )

    signal_power = float(
        torch.mean(
            original ** 2
        ).item()
    )

    if mse > EPS:

        snr = (
            10.0
            *
            math.log10(
                signal_power
                /
                mse
            )
        )

    else:

        snr = float(
            "inf"
        )

    peak_error = float(
        torch.max(
            torch.abs(
                difference
            )
        ).item()
    )

    original_rms = math.sqrt(
        signal_power
    )

    watermarked_rms = math.sqrt(
        float(
            torch.mean(
                watermarked ** 2
            ).item()
        )
    )

    original_centered = (
        original
        -
        original.mean()
    )

    watermarked_centered = (
        watermarked
        -
        watermarked.mean()
    )

    numerator = torch.sum(
        original_centered
        *
        watermarked_centered
    )

    denominator = torch.sqrt(
        torch.sum(
            original_centered ** 2
        )
        *
        torch.sum(
            watermarked_centered ** 2
        )
    )

    if denominator.item() > EPS:

        correlation = float(
            (
                numerator
                /
                denominator
            ).item()
        )

    else:

        correlation = 0.0

    return {
        "mse":
            mse,

        "rmse":
            rmse,

        "snr_db":
            snr,

        "peak_absolute_error":
            peak_error,

        "original_rms":
            original_rms,

        "watermarked_rms":
            watermarked_rms,

        "waveform_correlation":
            correlation,
    }


# ============================================================================
# CSV / AUDIO SAVING
# ============================================================================

def save_csv(
    path: Path,
    rows: List[Dict],
) -> None:

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def save_audio(
    path: Path,
    waveform: torch.Tensor,
    sample_rate: int,
) -> None:

    audio = (
        waveform
        .detach()
        .cpu()
        .flatten()
        .numpy()
    )

    sf.write(
        str(path),
        audio,
        sample_rate,
        subtype="PCM_16",
    )


# ============================================================================
# ALLOCATION DIAGNOSTICS
# ============================================================================

def allocation_statistics(
    coordinates: List[Dict],
) -> Dict[str, float]:

    if not coordinates:

        return {
            "coordinates":
                0,

            "unique_coordinates":
                0,

            "mean_joint_suitability":
                0.0,

            "mean_normalized_magnitude":
                0.0,
        }

    unique = {
        (
            int(row["time_frame"]),
            int(row["frequency_bin"]),
        )
        for row in coordinates
    }

    return {
        "coordinates":
            len(coordinates),

        "unique_coordinates":
            len(unique),

        "mean_joint_suitability":
            float(
                np.mean(
                    [
                        float(
                            row[
                                "joint_suitability"
                            ]
                        )
                        for row in coordinates
                    ]
                )
            ),

        "mean_normalized_magnitude":
            float(
                np.mean(
                    [
                        float(
                            row[
                                "normalized_magnitude"
                            ]
                        )
                        for row in coordinates
                    ]
                )
            ),
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    ensure_output_dir()

    print(
        "=" * 68
    )

    print(
        "SemBind-Audio : Experiment 21"
    )

    print(
        "Redundant Semantic-Spectral "
        "Adaptive STFT Embedding"
    )

    print(
        "=" * 68
    )

    # -----------------------------------------------------------------------
    # Load original audio
    # -----------------------------------------------------------------------

    original_audio, sample_rate = (
        load_original_audio()
    )

    print()
    print(
        "Original Audio"
    )

    print(
        "-" * 68
    )

    print(
        f"Audio Path       : "
        f"{ORIGINAL_AUDIO_PATH}"
    )

    print(
        f"Sample Rate      : "
        f"{sample_rate}"
    )

    print(
        f"Samples          : "
        f"{original_audio.numel()}"
    )

    print(
        f"Duration         : "
        f"{original_audio.numel() / sample_rate:.4f} s"
    )

    print(
        f"Original RMS     : "
        f"{torch.sqrt(torch.mean(original_audio ** 2)).item():.10f}"
    )

    print(
        f"Original Peak    : "
        f"{torch.max(torch.abs(original_audio)).item():.10f}"
    )

    # -----------------------------------------------------------------------
    # STFT
    # -----------------------------------------------------------------------

    original_stft = (
        compute_stft(
            original_audio
        )
    )

    print()
    print(
        f"STFT Shape       : "
        f"{tuple(original_stft.shape)}"
    )

    print(
        f"N_FFT            : "
        f"{N_FFT}"
    )

    print(
        f"HOP_LENGTH       : "
        f"{HOP_LENGTH}"
    )

    print(
        f"WIN_LENGTH       : "
        f"{WIN_LENGTH}"
    )

    # -----------------------------------------------------------------------
    # Payload
    # -----------------------------------------------------------------------

    payload = (
        load_payload()
    )

    required_coordinates = (
        payload.numel()
        *
        REDUNDANCY
    )

    print()
    print(
        f"Payload Length   : "
        f"{payload.numel()} bits"
    )

    print(
        f"Redundancy       : "
        f"{REDUNDANCY}x"
    )

    print(
        f"Required Coordinates : "
        f"{required_coordinates}"
    )

    # -----------------------------------------------------------------------
    # Semantic scores
    # -----------------------------------------------------------------------

    semantic_rows = (
        load_semantic_scores()
    )

    semantic_count = (
        max(
            row["frame"]
            for row in semantic_rows
        )
        +
        1
    )

    selected_count = sum(
        1
        for row in semantic_rows
        if row["selected"]
    )

    print()
    print(
        f"Semantic Score Rows : "
        f"{len(semantic_rows)}"
    )

    print(
        f"Selected Semantic Frames : "
        f"{selected_count}"
    )

    # -----------------------------------------------------------------------
    # Semantic regions
    # -----------------------------------------------------------------------

    regions = (
        merge_selected_regions(
            semantic_rows
        )
    )

    regions = (
        map_regions(
            regions,
            semantic_count,
            original_stft.shape[1],
        )
    )

    # -----------------------------------------------------------------------
    # Semantic-spectral parameters
    # -----------------------------------------------------------------------

    regions = (
        calculate_region_parameters(
            original_stft,
            regions,
        )
    )

    print()
    print(
        "Joint Semantic-Spectral "
        "Region Parameters"
    )

    print(
        "-" * 68
    )

    for region in regions:

        print(
            f"Region "
            f"{region['region_id']:02d} | "
            f"Semantic: "
            f"{region['semantic_score']:.6f} | "
            f"Semantic Suit.: "
            f"{region['semantic_suitability']:.6f} | "
            f"Spectral Suit.: "
            f"{region['spectral_suitability']:.6f} | "
            f"Joint Suit.: "
            f"{region['joint_suitability']:.6f} | "
            f"Delta: "
            f"{region['adaptive_strength']:.8f}"
        )

    # -----------------------------------------------------------------------
    # Raw capacity
    # -----------------------------------------------------------------------

    raw_capacity = (
        calculate_capacity(
            regions
        )
    )

    print()
    print(
        f"Raw Region Capacity : "
        f"{raw_capacity}"
    )

    print(
        f"Required Redundant Capacity : "
        f"{required_coordinates}"
    )

    if raw_capacity < required_coordinates:

        raise ValueError(
            "\nInsufficient raw semantic-region capacity.\n"
            f"Required : {required_coordinates}\n"
            f"Available: {raw_capacity}\n"
        )

    # -----------------------------------------------------------------------
    # Candidate generation
    # -----------------------------------------------------------------------

    candidates = (
        build_candidates(
            original_stft,
            regions,
        )
    )

    print(
        f"Eligible STFT Candidates : "
        f"{len(candidates)}"
    )

    if len(candidates) < required_coordinates:

        raise ValueError(
            "\nInsufficient candidate coefficients "
            "after filtering.\n"
            f"Required : {required_coordinates}\n"
            f"Available: {len(candidates)}\n\n"
            "The candidate filter was relaxed automatically, "
            "but the semantic-region capacity is still insufficient."
        )

    print(
        "Candidate Capacity Check : SUFFICIENT"
    )

    # -----------------------------------------------------------------------
    # Redundant allocation
    # -----------------------------------------------------------------------

    coordinates = (
        allocate_redundant_coordinates(
            payload,
            candidates,
        )
    )

    allocation_stats = (
        allocation_statistics(
            coordinates
        )
    )

    print()
    print(
        "Redundant Allocation"
    )

    print(
        "-" * 68
    )

    print(
        f"Payload Bits              : "
        f"{payload.numel()}"
    )

    print(
        f"Replicas per Bit          : "
        f"{REDUNDANCY}"
    )

    print(
        f"Allocated Coordinates     : "
        f"{len(coordinates)}"
    )

    print(
        f"Unique Coordinates        : "
        f"{allocation_stats['unique_coordinates']}"
    )

    print(
        f"Mean Joint Suitability    : "
        f"{allocation_stats['mean_joint_suitability']:.6f}"
    )

    print(
        f"Mean Normalized Magnitude : "
        f"{allocation_stats['mean_normalized_magnitude']:.6f}"
    )

    if (
        allocation_stats[
            "unique_coordinates"
        ]
        != required_coordinates
    ):

        raise RuntimeError(
            "Duplicate STFT coordinates detected."
        )

    # -----------------------------------------------------------------------
    # Embedding
    # -----------------------------------------------------------------------

    watermarked_stft, records = (
        embed_payload(
            original_stft,
            coordinates,
        )
    )

    print()
    print(
        "Embedding Complete"
    )

    print(
        f"Embedded Payload Bits : "
        f"{payload.numel()}"
    )

    print(
        f"Embedded Coordinates  : "
        f"{len(records)}"
    )

    changes = torch.tensor(
        [
            row[
                "delta_change"
            ]
            for row in records
        ],
        dtype=torch.float64,
    )

    print(
        f"Mean Coordinate Change : "
        f"{changes.mean().item():.10f}"
    )

    print(
        f"Maximum Coordinate Change : "
        f"{changes.max().item():.10f}"
    )

    # -----------------------------------------------------------------------
    # Global STFT magnitude change
    # -----------------------------------------------------------------------

    magnitude_change = (
        torch.abs(
            watermarked_stft
        )
        -
        torch.abs(
            original_stft
        )
    ).abs()

    print(
        f"Mean Magnitude Change : "
        f"{magnitude_change.mean().item():.10f}"
    )

    print(
        f"Maximum Magnitude Change : "
        f"{magnitude_change.max().item():.10f}"
    )

    # -----------------------------------------------------------------------
    # Reconstruction
    # -----------------------------------------------------------------------

    watermarked_audio = (
        reconstruct_audio(
            watermarked_stft,
            original_audio.numel(),
        )
    )

    difference_audio = (
        watermarked_audio
        -
        original_audio
    )

    # -----------------------------------------------------------------------
    # Audio quality
    # -----------------------------------------------------------------------

    metrics = (
        quality_metrics(
            original_audio,
            watermarked_audio,
        )
    )

    print()
    print(
        "=" * 68
    )

    print(
        "Audio Quality"
    )

    print(
        "=" * 68
    )

    print(
        f"MSE                  : "
        f"{metrics['mse']:.12e}"
    )

    print(
        f"RMSE                 : "
        f"{metrics['rmse']:.12e}"
    )

    print(
        f"SNR                  : "
        f"{metrics['snr_db']:.6f} dB"
    )

    print(
        f"Peak Absolute Error  : "
        f"{metrics['peak_absolute_error']:.12e}"
    )

    print(
        f"Original RMS         : "
        f"{metrics['original_rms']:.12e}"
    )

    print(
        f"Watermarked RMS      : "
        f"{metrics['watermarked_rms']:.12e}"
    )

    print(
        f"Waveform Correlation : "
        f"{metrics['waveform_correlation']:.12f}"
    )

    # -----------------------------------------------------------------------
    # Save tensors
    # -----------------------------------------------------------------------

    torch.save(
        original_stft.cpu(),
        ORIGINAL_STFT_PATH,
    )

    torch.save(
        watermarked_stft.cpu(),
        WATERMARKED_STFT_PATH,
    )

    torch.save(
        payload.cpu(),
        EMBEDDED_BITS_PATH,
    )

    # -----------------------------------------------------------------------
    # Save CSV files
    # -----------------------------------------------------------------------

    save_csv(
        COORDINATES_PATH,
        records,
    )

    save_csv(
        REGION_PARAMETERS_PATH,
        regions,
    )

    # -----------------------------------------------------------------------
    # Save audio
    # -----------------------------------------------------------------------

    save_audio(
        WATERMARKED_AUDIO_PATH,
        watermarked_audio,
        sample_rate,
    )

    save_audio(
        DIFFERENCE_AUDIO_PATH,
        difference_audio,
        sample_rate,
    )

    # -----------------------------------------------------------------------
    # Save experiment summary
    # -----------------------------------------------------------------------

    summary = {

        "experiment":
            "exp21_redundant_semantic_spectral_embedding",

        "description":
            (
                "Redundant semantic-spectral adaptive "
                "STFT watermark embedding"
            ),

        "payload_bits":
            int(
                payload.numel()
            ),

        "redundancy":
            int(
                REDUNDANCY
            ),

        "required_coordinates":
            int(
                required_coordinates
            ),

        "allocated_coordinates":
            int(
                len(records)
            ),

        "raw_region_capacity":
            int(
                raw_capacity
            ),

        "candidate_capacity":
            int(
                len(candidates)
            ),

        "unique_coordinate_count":
            int(
                allocation_stats[
                    "unique_coordinates"
                ]
            ),

        "alpha":
            ALPHA,

        "beta":
            BETA,

        "delta_min":
            DELTA_MIN,

        "delta_max":
            DELTA_MAX,

        "frequency_start_ratio":
            FREQ_START_RATIO,

        "frequency_end_ratio":
            FREQ_END_RATIO,

        "min_normalized_magnitude":
            MIN_NORMALIZED_MAGNITUDE,

        "preferred_time_separation":
            PREFERRED_TIME_SEPARATION,

        "preferred_frequency_separation":
            PREFERRED_FREQ_SEPARATION,

        "semantic_score_rows":
            int(
                len(
                    semantic_rows
                )
            ),

        "selected_semantic_frames":
            int(
                selected_count
            ),

        "candidate_regions":
            int(
                len(regions)
            ),

        "stft_shape":
            [
                int(
                    original_stft.shape[0]
                ),
                int(
                    original_stft.shape[1]
                ),
            ],

        "stft_configuration":
            {
                "n_fft":
                    N_FFT,

                "hop_length":
                    HOP_LENGTH,

                "win_length":
                    WIN_LENGTH,
            },

        "allocation_statistics":
            allocation_stats,

        "audio_quality":
            metrics,

        "mean_coordinate_change":
            float(
                changes.mean().item()
            ),

        "maximum_coordinate_change":
            float(
                changes.max().item()
            ),

        "mean_global_magnitude_change":
            float(
                magnitude_change
                .mean()
                .item()
            ),

        "maximum_global_magnitude_change":
            float(
                magnitude_change
                .max()
                .item()
            ),

        "output_files":
            {
                "original_stft":
                    str(
                        ORIGINAL_STFT_PATH
                    ),

                "watermarked_stft":
                    str(
                        WATERMARKED_STFT_PATH
                    ),

                "embedded_bits":
                    str(
                        EMBEDDED_BITS_PATH
                    ),

                "coordinates":
                    str(
                        COORDINATES_PATH
                    ),

                "region_parameters":
                    str(
                        REGION_PARAMETERS_PATH
                    ),

                "watermarked_audio":
                    str(
                        WATERMARKED_AUDIO_PATH
                    ),

                "difference_audio":
                    str(
                        DIFFERENCE_AUDIO_PATH
                    ),
            },
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # -----------------------------------------------------------------------
    # Final output
    # -----------------------------------------------------------------------

    print()
    print(
        "Results Saved"
    )

    print(
        ORIGINAL_STFT_PATH
    )

    print(
        WATERMARKED_STFT_PATH
    )

    print(
        EMBEDDED_BITS_PATH
    )

    print(
        COORDINATES_PATH
    )

    print(
        REGION_PARAMETERS_PATH
    )

    print(
        WATERMARKED_AUDIO_PATH
    )

    print(
        DIFFERENCE_AUDIO_PATH
    )

    print(
        SUMMARY_PATH
    )

    print()
    print(
        "=" * 68
    )

    print(
        "Experiment 21 Complete"
    )

    print(
        "=" * 68
    )


if __name__ == "__main__":
    main()