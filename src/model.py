import torch
import torch.nn as nn
import torch.nn.functional as F


class TTAEvalModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

    def _aligned_matching(
        self,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        対応するセグメント間の類似度からPrecision, Recall, F1を計算

        Args:
            audio_emb: Audio segment embeddings [B, S, D]
            text_emb: Text phrase embeddings [B, S, D]
            audio_mask: Valid audio segment mask [B, S]
            text_mask: Valid text segment mask [B, S]

        Returns:
            (Precision, Recall, F1) each [B]
        """
        # 対応するセグメント間の類似度 [B, S]
        sim = (audio_emb * text_emb).sum(dim=-1)

        # Precision: テキスト視点での平均類似度
        text_mask_float = text_mask.float().to(sim.device)
        text_valid_counts = text_mask_float.sum(dim=1).clamp(min=1.0)
        precision = (sim * text_mask_float).sum(dim=1) / text_valid_counts

        # Recall: 音声視点での平均類似度
        audio_mask_float = audio_mask.float().to(sim.device)
        audio_valid_counts = audio_mask_float.sum(dim=1).clamp(min=1.0)
        recall = (sim * audio_mask_float).sum(dim=1) / audio_valid_counts

        # F1
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return precision, recall, f1

    def forward(
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor,
        parsed_audio_feats: torch.Tensor | None = None,
        parsed_text_feats: torch.Tensor | None = None,
        parsed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Coarse-Grained (COGR): Global cosine similarity between audio and text embeddings.
        Fine-Grained (FIGR): Aligned matching score between parsed segments.

        Final score = (COGR + FIGR) / 2

        Args:
            audio_feats: [B, D]
            text_feats: [B, D]
            parsed_audio_feats: [B, S, D] (optional)
            parsed_text_feats: [B, S, D] (optional)
            parsed_mask: Valid segment mask [B, S] (optional)

        Returns:
            Similarity scores [B]
        """
        audio_feats = F.normalize(audio_feats, p=2, dim=-1)
        text_feats = F.normalize(text_feats, p=2, dim=-1)
        coarse_grained_sim = torch.sum(audio_feats * text_feats, dim=-1)  # [B]

        if not all(
            isinstance(x, torch.Tensor)
            for x in (parsed_audio_feats, parsed_text_feats, parsed_mask)
        ):
            # If no parsed features, return only Coarse-Grained similarity
            return coarse_grained_sim

        parsed_audio_feats = F.normalize(parsed_audio_feats, p=2, dim=-1)
        parsed_text_feats = F.normalize(parsed_text_feats, p=2, dim=-1)
        mask = parsed_mask.to(parsed_audio_feats.device)

        _, _, fine_grained_f1_score = self._aligned_matching(
            audio_emb=parsed_audio_feats,
            text_emb=parsed_text_feats,
            audio_mask=mask,
            text_mask=mask,
        )

        combined_score = (coarse_grained_sim + fine_grained_f1_score) / 2.0

        return combined_score
