import torch
import torch.nn as nn


class TTAEvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        embedding_dim = 512
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
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
        batch_size = audio_feats.size(0)
        hadamard_product = audio_feats * text_feats  # [B, D]
        diff = audio_feats - text_feats  # [B, D]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        features = torch.cat(
            [
                cls_tokens,
                hadamard_product.unsqueeze(1),
                diff.unsqueeze(1),
                # audio_feats.unsqueeze(1),
                # text_feats.unsqueeze(1),
            ],
            dim=1,
        )  # [B, 5, D]
        features = self.transformer(features)  # [B, 5, D]
        cls_features = features[:, 0, :]  # [B, D]
        preds = self.prediction_head(cls_features)  # [B, 1]
        return preds  # [B, 1]
