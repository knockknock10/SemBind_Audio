import torch
import torch.nn.functional as F


class SemanticContribution:

    def __init__(self, embeddings):

        self.embeddings = embeddings
        self.num_frames = embeddings.shape[0]

    def compute_global_embedding(self):

        return torch.mean(
            self.embeddings,
            dim=0,
        )

    def compute_perturbed_embedding(
        self,
        index,
    ):

        mask = torch.ones(
            self.num_frames,
            dtype=torch.bool,
        )

        mask[index] = False

        remaining = self.embeddings[
            mask
        ]

        return torch.mean(
            remaining,
            dim=0,
        )

    def compute_semantic_shift(
        self,
        original,
        perturbed,
    ):

        similarity = F.cosine_similarity(
            original.unsqueeze(0),
            perturbed.unsqueeze(0),
            dim=1,
        )

        shift = 1.0 - similarity

        return shift.item()

    def compute(self):

        original_global = (
            self.compute_global_embedding()
        )

        perturbed_embeddings = []

        contribution_scores = []

        for i in range(
            self.num_frames
        ):

            perturbed = (
                self.compute_perturbed_embedding(
                    i
                )
            )

            shift = (
                self.compute_semantic_shift(
                    original_global,
                    perturbed,
                )
            )

            perturbed_embeddings.append(
                perturbed
            )

            contribution_scores.append(
                shift
            )

        contribution_scores = torch.tensor(
            contribution_scores,
            dtype=torch.float32,
        )

        normalized_scores = (
            contribution_scores
            - contribution_scores.min()
        ) / (
            contribution_scores.max()
            - contribution_scores.min()
            + 1e-8
        )

        perturbed_embeddings = torch.stack(
            perturbed_embeddings
        )

        frame_indices = torch.arange(
            self.num_frames
        )

        return {
            "original_global_embedding": original_global,
            "perturbed_global_embeddings": perturbed_embeddings,
            "raw_scores": contribution_scores,
            "normalized_scores": normalized_scores,
            "frame_indices": frame_indices,
        }