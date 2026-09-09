"""
Semantic Contribution Module
---------------------------
Computes perturbation-based semantic contribution scores for HuBERT embeddings.

For each temporal region i:
    G     = (1/N) SUM_j E_j       (global representation)
    G^(-i) = (1/(N-1)) SUM(j!=i) E_j  (global representation without region i)
    Δ_i    = 1 - cos(G, G^(-i))

Larger Δ_i → region contributes more strongly to the global semantic representation.
Therefore: higher Δ_i → higher semantic importance → less desirable for watermark embedding.
Lower Δ_i → lower semantic importance → more desirable for embedding.

NOTE: This is NOT the same as cos(E_i, G) (semantic alignment). The repository previously
used alignment; this has been corrected to the perturbation-based definition per the
SemBind-Audio research methodology.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F


class SemanticContribution:
    """
    Computes perturbation-based semantic contribution scores for HuBERT embeddings.
    """

    def __init__(self, embeddings: torch.Tensor):
        """
        Parameters
        ----------
        embeddings : torch.Tensor
            Shape = (num_frames, embedding_dim)
        """

        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("Embeddings must be a torch.Tensor.")

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must have shape (num_frames, embedding_dim)."
            )

        self.embeddings = embeddings
        self.num_frames = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]

    def compute_global_embedding(self) -> torch.Tensor:
        """
        Compute the global semantic embedding G = (1/N) SUM E_i.

        Returns
        -------
        torch.Tensor
            Shape = (embedding_dim,)
        """
        return torch.mean(self.embeddings, dim=0)

    def compute_perturbed_embedding(self, index: int) -> torch.Tensor:
        """
        Compute the global embedding without frame at given index.

        G^(-i) = (1/(N-1)) SUM(j!=i) E_j

        Returns
        -------
        torch.Tensor
            Shape = (embedding_dim,)
        """
        mask = torch.ones(
            self.num_frames,
            dtype=torch.bool,
        )
        mask[index] = False

        remaining = self.embeddings[mask]
        return torch.mean(remaining, dim=0)

    def compute_semantic_shift(self, original_global: torch.Tensor, perturbed_global: torch.Tensor) -> float:
        """
        Compute Δ_i = 1 - cos(G, G^(-i)).

        Larger shift → region contributes more → less desirable for embedding.

        Returns
        -------
        float
            Semantic shift in [0, 2].
        """
        similarity = F.cosine_similarity(
            original_global.unsqueeze(0),
            perturbed_global.unsqueeze(0),
            dim=1,
        )

        shift = 1.0 - similarity
        return shift.item()

    def compute(self) -> dict:
        """
        Complete semantic contribution pipeline.

        For each frame i:
            1. Compute global embedding G
            2. Compute perturbed embedding G^(-i) (without frame i)
            3. Compute Δ_i = 1 - cos(G, G^(-i))

        Returns
        -------
        dict
            - original_global_embedding: torch.Tensor (embedding_dim,)
            - perturbed_global_embeddings: torch.Tensor (num_frames, embedding_dim)
            - raw_scores: torch.Tensor (num_frames,) Δ_i values
            - normalized_scores: torch.Tensor (num_frames,) Δ_i normalized to [0,1]
            - frame_indices: torch.Tensor (num_frames,) 0-based indices
        """
        original_global = self.compute_global_embedding()

        raw_scores = []
        perturbed_embeddings = []

        for i in range(self.num_frames):
            perturbed = self.compute_perturbed_embedding(i)
            shift = self.compute_semantic_shift(original_global, perturbed)
            raw_scores.append(shift)
            perturbed_embeddings.append(perturbed)

        raw_scores = torch.tensor(
            raw_scores,
            dtype=torch.float32,
        )

        # Normalize to [0, 1] for interpretability
        # Lower raw Δ_i = lower semantic importance = more desirable for embedding
        min_score = raw_scores.min()
        max_score = raw_scores.max()

        if torch.isclose(min_score, max_score):
            normalized_scores = torch.zeros_like(raw_scores)
        else:
            normalized_scores = (
                raw_scores - min_score
            ) / (max_score - min_score + 1e-8)

        perturbed_embeddings = torch.stack(perturbed_embeddings)
        frame_indices = torch.arange(self.num_frames)

        return {
            "original_global_embedding": original_global,
            "perturbed_global_embeddings": perturbed_embeddings,
            "raw_scores": raw_scores,
            "normalized_scores": normalized_scores,
            "frame_indices": frame_indices,
        }

    def compute_alignment_scores(self) -> torch.Tensor:
        """
        Compute cosine similarity between each frame and the global embedding.
        This is semantic ALIGNMENT, NOT semantic contribution.

        Returns
        -------
        torch.Tensor
            Shape = (num_frames,)
        """
        global_embedding = self.compute_global_embedding()

        scores = F.cosine_similarity(
            self.embeddings,
            global_embedding.unsqueeze(0),
            dim=1,
        )

        return scores
    