"""
SemBind-Audio

Semantic Region Selection Module
"""

import torch
import numpy as np
import pandas as pd


def compute_semantic_contribution(embeddings: torch.Tensor) -> dict:
    """
    Compute perturbation-based semantic contribution for each HuBERT frame.

    For each frame i:
        G     = (1/N) SUM_j E_j       (global representation)
        G^(-i) = (1/(N-1)) SUM(j!=i) E_j  (global representation without frame i)
        Δ_i    = 1 - cos(G, G^(-i))

    Larger Δ_i → region contributes more → less desirable for watermark embedding.

    Parameters
    ----------
    embeddings : torch.Tensor
        Shape = (num_frames, embedding_dim)

    Returns
    -------
    dict
        - raw_scores: torch.Tensor (num_frames,) Δ_i values
        - normalized_scores: torch.Tensor (num_frames,) Δ_i normalized to [0,1]
        - frame_indices: torch.Tensor (num_frames,) 0-based indices
    """
    contribution = SemanticContribution(embeddings)
    return contribution.compute()


def select_semantic_regions(
    contribution_scores: torch.Tensor,
    percentile: float = 25.0,
    min_region_frames: int = 3,
) -> tuple:
    """
    Select STFT embedding regions based on semantic contribution scores.

    Regions with LOW semantic contribution (low Δ_i) are preferred for watermark
    embedding, because they carry less semantic information.

    Parameters
    ----------
    contribution_scores : torch.Tensor
        Shape = (num_frames,) raw semantic contribution Δ_i values.
    percentile : float
        Percentile threshold for selection. Frames at or below this percentile
        are selected as low-importance regions.
    min_region_frames : int
        Minimum number of consecutive frames to form a valid embedding region.

    Returns
    -------
    selected_frames : torch.Tensor
        Frame indices (1-based) that were selected.
    regions : list of dict
        Each dict contains: region_id, start_frame, end_frame, length,
        mean_score, minimum_score, maximum_score.
    """
    # Threshold: select frames at or below the percentile
    threshold = float(np.percentile(contribution_scores.cpu().numpy(), percentile))

    selected_mask = contribution_scores <= threshold
    selected_frames = torch.arange(1, len(contribution_scores) + 1)[selected_mask]

    # Merge consecutive frames into regions
    if len(selected_frames) == 0:
        return selected_frames, []

    sorted_frames, _ = torch.sort(selected_frames)

    regions = []
    region_id = 1
    start = sorted_frames[0].item()
    previous = sorted_frames[0].item()

    for frame in sorted_frames[1:].tolist():
        if frame == previous + 1:
            previous = frame
        else:
            # Close previous region
            length = previous - start + 1
            if length >= min_region_frames:
                mean_score = float(
                    torch.mean(
                        contribution_scores[start - 1 : previous]
                    ).item()
                )
                regions.append(
                    {
                        "region_id": region_id,
                        "start_frame": start,
                        "end_frame": previous,
                        "length": length,
                        "mean_score": mean_score,
                        "minimum_score": float(
                            torch.min(
                                contribution_scores[start - 1 : previous]
                            ).item()
                        ),
                        "maximum_score": float(
                            torch.max(
                                contribution_scores[start - 1 : previous]
                            ).item()
                        ),
                    }
                )
                region_id += 1
            start = frame
            previous = frame

    # Close last region
    length = previous - start + 1
    if length >= min_region_frames:
        mean_score = float(
            torch.mean(
                contribution_scores[start - 1 : previous + 1]
            ).item()
        )
        regions.append(
            {
                "region_id": region_id,
                "start_frame": start,
                "end_frame": previous,
                "length": length,
                "mean_score": mean_score,
                "minimum_score": float(
                    torch.min(
                        contribution_scores[start - 1 : previous + 1]
                    ).item()
                ),
                "maximum_score": float(
                    torch.max(
                        contribution_scores[start - 1 : previous + 1]
                    ).item()
                ),
            }
        )

    return selected_frames, regions


def select_lowest_contribution_frames(
    contribution_scores: torch.Tensor,
    n_frames: int,
) -> torch.Tensor:
    """
    Select exactly n_frames frames with the lowest semantic contribution.

    Parameters
    ----------
    contribution_scores : torch.Tensor
        Shape = (num_frames,) raw semantic contribution Δ_i values.
    n_frames : int
        Number of frames to select.

    Returns
    -------
    torch.Tensor
        Indices (1-based) of the n_frames lowest-contribution frames.
    """
    _, indices = torch.topk(
        contribution_scores,
        k=n_frames,
        largest=False,
    )

    return indices + 1  # 1-based indexing


def compute_joint_suitability(
    semantic_scores: torch.Tensor,
    spectral_scores: torch.Tensor,
    alpha: float = 0.70,
    beta: float = 0.30,
) -> torch.Tensor:
    """
    Compute joint suitability score for each candidate region.

    J_r = (S_r^alpha) * (P_r^beta)

    where:
        S_r = semantic suitability (inverse of semantic contribution)
        P_r = spectral suitability

    lower semantic contribution → higher suitability

    Parameters
    ----------
    semantic_scores : torch.Tensor
        Raw semantic contribution scores (Δ_i) per region.
    spectral_scores : torch.Tensor
        Spectral suitability scores per region.
    alpha : float
        Exponent for semantic suitability. Default 0.70.
    beta : float
        Exponent for spectral suitability. Default 0.30.

    Returns
    -------
    torch.Tensor
        Joint suitability scores J_r per region.
    """
    # Semantic suitability: inverse of contribution, normalized to [0,1]
    sem_min = semantic_scores.min().item()
    sem_max = semantic_scores.max().item()

    if abs(sem_max - sem_min) < 1e-8:
        s_norm = torch.ones_like(semantic_scores) * 0.5
    else:
        s_norm = (semantic_scores - sem_min) / (sem_max - sem_min)

    # Semantic suitability: lower contribution → higher suitability
    semantic_suitability = 1.0 - s_norm

    # Spectral suitability: normalize to [0,1]
    spec_min = spectral_scores.min().item()
    spec_max = spectral_scores.max().item()

    if abs(spec_max - spec_min) < 1e-8:
        p_norm = torch.ones_like(spectral_scores) * 0.5
    else:
        p_norm = (spectral_scores - spec_min) / (spec_max - spec_min + 1e-8)

    # Joint suitability (multiplicative, per mission spec)
    joint = (s_norm ** alpha) * (p_norm ** beta)

    return joint