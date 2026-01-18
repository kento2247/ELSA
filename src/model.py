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

    def _kl_from_gaussian(
        self,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        類似度行列とガウスノイズ間のKLダイバージェンスを計算

        類似度行列をsoftmaxで確率分布に変換し、均一分布（ガウスノイズの期待値）との
        KLダイバージェンスを計算。スコアが高いほど類似度行列が構造を持つ。

        Args:
            audio_emb: Audio segment embeddings [B, S, D]
            text_emb: Text phrase embeddings [B, S, D]
            mask: Valid segment mask [B, S]

        Returns:
            score: KLダイバージェンス [B]
        """
        # 類似度行列 [B, S, S]
        sim = torch.bmm(text_emb, audio_emb.transpose(1, 2))

        # マスク作成 [B, S, S]
        mask_float = mask.float().to(sim.device)
        sim_mask = mask_float.unsqueeze(2) * mask_float.unsqueeze(1)

        # マスク外を-infにして softmax から除外
        sim_masked = sim.masked_fill(sim_mask == 0, float("-inf"))

        # 各行を確率分布に変換 [B, S, S]
        p = F.softmax(sim_masked, dim=-1)
        p = p.masked_fill(sim_mask == 0, 0)  # NaN防止

        # 均一分布 q (ガウスノイズの期待値)
        valid_per_row = mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
        q = mask_float.unsqueeze(1) / valid_per_row.unsqueeze(2)  # [B, S, S]

        # KLダイバージェンス: sum(p * log(p / q))
        kl = p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))
        kl = kl.masked_fill(sim_mask == 0, 0)

        # 各バッチの平均KL
        valid_counts = mask_float.sum(dim=1).clamp(min=1.0)
        score = kl.sum(dim=(1, 2)) / valid_counts

        return score

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
        Fine-Grained (FIGR): KL divergence from Gaussian noise.

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
        cogr = torch.sum(audio_feats * text_feats, dim=-1)  # [B]

        if not all(
            isinstance(x, torch.Tensor)
            for x in (parsed_audio_feats, parsed_text_feats, parsed_mask)
        ):
            return cogr

        parsed_audio_feats = F.normalize(parsed_audio_feats, p=2, dim=-1)
        parsed_text_feats = F.normalize(parsed_text_feats, p=2, dim=-1)

        mask = parsed_mask.to(parsed_audio_feats.device)

        figr = self._kl_from_gaussian(
            audio_emb=parsed_audio_feats,
            text_emb=parsed_text_feats,
            mask=mask,
        )

        combined_score = (cogr + figr) / 2.0

        return combined_score
