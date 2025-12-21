import torch
import torch.nn as nn
import torch.nn.functional as F


class TTAEvalModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, audio_feats: torch.Tensor, text_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between audio and text features.

        Args:
            audio_feats: Audio features from MSCLAP [B, D]
            text_feats: Text features from MSCLAP [B, D]

        Returns:
            Similarity scores [B]
        """
        audio_feats = F.normalize(audio_feats, p=2, dim=-1)
        text_feats = F.normalize(text_feats, p=2, dim=-1)
        similarity = torch.sum(audio_feats * text_feats, dim=-1)
        return similarity
