import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioTextSimilarityModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        embedding_dim = 512
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.prediction_head = nn.Linear(embedding_dim, 1)

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
        hadamard_product = audio_feats * text_feats
        diff = audio_feats - text_feats
        features = torch.cat(
            [
                self.cls_token,
                hadamard_product.unsqueeze(-1),
                diff.unsqueeze(-1),
                audio_feats.unsqueeze(-1),
                text_feats.unsqueeze(-1),
            ],
            dim=-1,
        )  # [B, 4, D]
        features = self.transformer(features)  # [B, 4, D]
        cls_features = features[:, 0, :]  # [B, D]
        preds = self.prediction_head(cls_features)  # [B, 1]
        return preds  # [B, 1]
