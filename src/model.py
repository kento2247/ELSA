import torch
import torch.nn as nn


class TTAEvalModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        dropout: float = 0.1,
        num_metrics: int = 2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_metrics = num_metrics

        # Metric embedding (REL=0, OVL=1)
        self.metric_embedding = nn.Embedding(num_metrics, embedding_dim)

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 5, embedding_dim),
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
        # Get metric embedding
        metric_emb = self.metric_embedding(metric_ids)  # [B, D]

        hadamard_product = audio_feats * text_feats  # [B, D]
        diff = audio_feats - text_feats  # [B, D]

        features = torch.cat(
            [audio_feats, text_feats, hadamard_product, diff, metric_emb], dim=-1
        )  # [B, 4D]

        preds = self.mlp(features)  # [B, 1]

        return preds
