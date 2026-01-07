import torch
import torch.nn as nn
import torch.nn.functional as F


class TTAEvalModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

    def forward(
        self, audio_feats: torch.Tensor, text_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity score using Transformer architecture.

        Args:
            audio_feats: Audio features from MSCLAP [B, D]
            text_feats: Text features from MSCLAP [B, D]

        Returns:
            Similarity scores [B, 1]
        """
        audio_feats = F.normalize(audio_feats, p=2, dim=-1)
        text_feats = F.normalize(text_feats, p=2, dim=-1)
        similarity = torch.sum(audio_feats * text_feats, dim=-1)
        return similarity
