import torch
import torch.nn as nn


class TTAEvalModel(nn.Module):
    def __init__(self):
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
        audio_feats = audio_feats / audio_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        logits_per_audio = torch.matmul(
            audio_feats, text_feats.t()
        )  # [B, B] cosine similarity matrix
        logits_per_text = torch.matmul(
            text_feats, audio_feats.t()
        )  # [B, B] cosine similarity matrix

        scores = (logits_per_audio.diag() + logits_per_text.diag()) / 2  # [B]

        return scores
