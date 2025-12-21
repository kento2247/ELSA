import torch
import torch.nn as nn


class TTAEvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        embedding_dim = 512
        self.head_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim // 2),
            nn.Linear(embedding_dim // 2, 1),
        )

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
        hadamard_product = audio_feats * text_feats  # [B, D]
        diff = audio_feats - text_feats  # [B, D]
        feats = torch.cat([hadamard_product, diff], dim=-1)  # [B, D*2]
        preds = self.head_mlp(feats)  # [B, 1]
        return preds  # [B, 1]
