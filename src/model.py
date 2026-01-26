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

    def _greedy_matching(
        self,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_emb: Audio segment embeddings [B, S_a, D]
            text_emb: Text phrase embeddings [B, S_t, D]
            audio_mask: Valid audio segment mask [B, S_a]
            text_mask: Valid text phrase mask [B, S_t]

        Returns:
            (Precision, Recall, F1) (3, B)
        """
        sim = torch.bmm(text_emb, audio_emb.transpose(1, 2))

        mask = torch.bmm(
            text_mask.unsqueeze(2).float(), audio_mask.unsqueeze(1).float()
        )
        mask = mask.to(sim.device)
        sim = sim * mask

        word_precision = sim.max(dim=2)[0]  # [B, S_t]
        word_recall = sim.max(dim=1)[0]  # [B, S_a]

        # precision
        text_mask_float = text_mask.float().to(word_precision.device)
        text_valid_counts = text_mask_float.sum(dim=1).clamp(min=1.0)
        precision = (word_precision * text_mask_float).sum(dim=1) / text_valid_counts

        # recall
        audio_mask_float = audio_mask.float().to(word_recall.device)
        audio_valid_counts = audio_mask_float.sum(dim=1).clamp(min=1.0)
        recall = (word_recall * audio_mask_float).sum(dim=1) / audio_valid_counts

        # F1
        f1 = 2 * precision * recall / (2 * precision + recall + 1e-8)

        return precision, recall, f1

    def forward(
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor,
        parsed_audio_feats: torch.Tensor,
        parsed_text_feats: torch.Tensor,
        parsed_mask: torch.Tensor,
        subjective_metric_id: int,
    ) -> torch.Tensor:
        """
        Coarse-Grained (COGR): Global cosine similarity between audio and text embeddings.
        Fine-Grained (FIGR): Greedy matching F1 score between parsed segments.

        Final score = (COGR + FIGR_F1) / 2

        Args:
            audio_feats: [B, D]
            text_feats: [B, D]
            parsed_audio_feats: [B, S, D] (optional)
            parsed_text_feats: [B, S, D] (optional)
            parsed_mask: Valid segment mask [B, S] (optional)
            subjective_metric_id: 0 for REL, 1 for OVL

        Returns:
            Similarity scores [B]
        """
        audio_feats = F.normalize(audio_feats, p=2, dim=-1)
        text_feats = F.normalize(text_feats, p=2, dim=-1)
        cogr = torch.sum(audio_feats * text_feats, dim=-1)  # [B]

        parsed_audio_feats = F.normalize(parsed_audio_feats, p=2, dim=-1)
        parsed_text_feats = F.normalize(parsed_text_feats, p=2, dim=-1)

        mask = parsed_mask.to(parsed_audio_feats.device)

        figr_precision, figr_recall, figr_f1 = self._greedy_matching(
            audio_emb=parsed_audio_feats,
            text_emb=parsed_text_feats,
            audio_mask=mask,
            text_mask=mask,
        )

        # For samples with all-zero masks, use only cogr
        has_valid_mask = mask.any(dim=-1)  # [B]
        combined_score = 0.2 * cogr + 0.8 * figr_f1
        score = torch.where(has_valid_mask, combined_score, cogr)

        # snr = torch.sum(text_feats.unsqueeze(1) * parsed_audio_feats, dim=-1)
        # snr = (snr * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        # # snr = (snr * mask).max(dim=-1)[0]
        # snr = F.softmax(torch.cat([snr.unsqueeze(-1), cogr.unsqueeze(-1)], dim=-1), dim=-1)[:, 1]

        # parsed_snr = torch.sum(parsed_text_feats * parsed_audio_feats, dim=-1)
        # orig_snr = torch.sum(parsed_text_feats * audio_feats.unsqueeze(1), dim=-1)
        # snr = F.softmax(
        #     torch.cat([parsed_snr.unsqueeze(-1), orig_snr.unsqueeze(-1)], dim=-1),
        #     dim=-1,
        # )[..., 0]
        # # snr = (snr * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        # snr = (snr * mask).max(dim=-1)[0]

        # result = torch.zeros_like(score)
        # for i in range(score.shape[0]):
        #     if subjective_metric_id[i] == 0:
        #         result[i] = score[i]
        #     else:
        #         result[i] = snr[i]

        return score
