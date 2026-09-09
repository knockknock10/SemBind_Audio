"""
SemBind-Audio

HuBERT Embedding Extractor
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import numpy as np
import torch

from transformers import (
    HubertModel,
    AutoFeatureExtractor,
)

from configs.config import (
    HUBERT_MODEL,
    SAMPLE_RATE,
)


class HuBERTEmbedding:
    """
    Extract semantic embeddings using a pretrained HuBERT model.
    """

    def __init__(self):

        # Device Selection

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")

        elif torch.cuda.is_available():
            self.device = torch.device("cuda")

        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        # Feature Extractor

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            HUBERT_MODEL
        )

        # HuBERT Model

        self.model = HubertModel.from_pretrained(
            HUBERT_MODEL
        )

        self.model.to(self.device)
        self.model.eval()

    def extract(self, audio: np.ndarray):

        """
        Extract semantic embeddings from an audio waveform.
        """

        inputs = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )

        input_values = inputs["input_values"].to(self.device)

        with torch.no_grad():

            outputs = self.model(
                input_values
            )

        embeddings = outputs.last_hidden_state

        return embeddings.squeeze(0).cpu()