import torch
import torch.nn as nn


class TTAEvalModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_metrics: int = 2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_metrics = num_metrics

        # Metric embedding (REL=0, OVL=1)
        self.metric_embedding = nn.Embedding(num_metrics, embedding_dim)
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
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor,
        metric_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity scores conditioned on subjective metric type.

        Args:
            audio_feats: Audio features from CLAP [B, D]
            text_feats: Text features from CLAP [B, D]
            metric_ids: Subjective metric IDs (0=REL, 1=OVL) [B]

        Returns:
            Predictions [B, 1]
        """
        B = audio_feats.size(0)

        # Get metric embedding
        metric_emb = self.metric_embedding(metric_ids)  # [B, D]

        hadamard_product = audio_feats * text_feats  # [B, D]
        diff = audio_feats - text_feats  # [B, D]

        sequence = torch.cat(
            [
                self.cls_token.expand(B, -1, -1),  # [B, 1, D]
                hadamard_product.unsqueeze(1),  # [B, 1, D]
                diff.unsqueeze(1),  # [B, 1, D]
                audio_feats.unsqueeze(1),  # [B, 1, D]
                text_feats.unsqueeze(1),  # [B, 1, D]
                metric_emb.unsqueeze(1),  # [B, 1, D
            ],
            dim=1,
        )
        encoded = self.transformer_encoder(sequence)  # [B, 5, D]

        cls_output = encoded[:, 0, :]  # [B, D]
        preds = self.head(cls_output)  # [B, 1]

        return preds
