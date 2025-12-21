import torch
import torch.nn as nn


class TTAEvalModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Head MLP
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.LayerNorm(embedding_dim // 2),
            nn.Linear(embedding_dim // 2, 1),
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

        sequence = torch.cat(
            [
                self.cls_token.expand(B, -1, -1),  # [B, 1, D]
                hadamard_product.unsqueeze(1),  # [B, 1, D]
                diff.unsqueeze(1),  # [B, 1, D]
                audio_feats.unsqueeze(1),  # [B, 1, D]
                text_feats.unsqueeze(1),  # [B, 1, D]
            ],
            dim=1,
        )
        encoded = self.transformer_encoder(sequence)  # [B, 5, D]

        cls_output = encoded[:, 0, :]  # [B, D]
        preds = self.head(cls_output)  # [B, 1]

        return preds
