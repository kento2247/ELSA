import torch
import torch.nn as nn


class TTAEvalModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Head MLP
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

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
        B = audio_feats.size(0)

        hadamard_product = audio_feats * text_feats  # [B, D]
        diff = audio_feats - text_feats  # [B, D]

        features = torch.cat(
            [audio_feats, text_feats, hadamard_product, diff], dim=-1
        )  # [B, 4D]
        preds = self.mlp(features)  # [B, D]

        return preds
