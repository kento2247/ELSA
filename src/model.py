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
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor,
        parsed_audio_feats: torch.Tensor | None = None,
        parsed_text_feats: torch.Tensor | None = None,
        parsed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute similarity score using Transformer architecture.

        Args:
            audio_feats: Audio features from MSCLAP [B, D]
            text_feats: Text features from MSCLAP [B, D]
            parsed_audio_feats: Parsed audio features [B, S, D] (optional)
            parsed_text_feats: Parsed text features [B, S, D] (optional)
            parsed_mask: Parsed feature mask [B, S] (optional)

        Returns:
            Similarity scores [B, 1]
        """
        audio_feats = F.normalize(audio_feats, p=2, dim=-1)
        text_feats = F.normalize(text_feats, p=2, dim=-1)
        base_similarity = torch.sum(audio_feats * text_feats, dim=-1)

        if not all(
            isinstance(x, torch.Tensor)
            for x in (parsed_audio_feats, parsed_text_feats, parsed_mask)
        ):
            return base_similarity

        parsed_audio_feats = F.normalize(parsed_audio_feats, p=2, dim=-1)
        parsed_text_feats = F.normalize(parsed_text_feats, p=2, dim=-1)
        parsed_similarity = torch.sum(parsed_audio_feats * parsed_text_feats, dim=-1)
        mask = parsed_mask.to(parsed_similarity.device).float()
        valid_counts = mask.sum(dim=-1).clamp(min=1.0)
        parsed_mean = (parsed_similarity * mask).sum(dim=-1) / valid_counts

        return (base_similarity + parsed_mean) / 2.0
