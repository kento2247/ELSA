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

    def _calculate_snr(
        self,
        audio: torch.Tensor,
        residual_audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate Signal-to-Noise Ratio (SNR) in dB.

        SNR = 10 * log10(signal_power / noise_power)
        where noise = residual_audio (difference between generated and reference audio)

        Args:
            audio: Original audio signal [B, T]
            residual_audio: Difference audio (noise) [B, T]

        Returns:
            SNR values in dB [B], normalized to [0, 1] range
        """
        signal_power = torch.mean(audio**2, dim=-1)
        noise_power = torch.mean(residual_audio**2, dim=-1)

        snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-8) + 1e-8)

        snr_normalized = torch.sigmoid(snr_db / 20.0)

        return snr_normalized

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
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return precision, recall, f1

    def forward(
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor,
        parsed_audio_feats: torch.Tensor,
        parsed_text_feats: torch.Tensor,
        parsed_mask: torch.Tensor,
        audio: torch.Tensor = None,
        residual_audio: torch.Tensor = None,
        subjective_metric_id: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Coarse-Grained (COGR): Global cosine similarity between audio and text embeddings.
        Fine-Grained (FIGR): Greedy matching F1 score between parsed segments.
        SNR: Signal-to-Noise Ratio calculated from audio and residual_audio.

        For OVL metric (id=1): score = (SNR + COGR) / 2
        For REL metric (id=0): score = (COGR + FIGR_F1) / 2

        Args:
            audio_feats: [B, D]
            text_feats: [B, D]
            parsed_audio_feats: [B, S, D] (optional)
            parsed_text_feats: [B, S, D] (optional)
            parsed_mask: Valid segment mask [B, S] (optional)
            audio: Original audio signal [B, T] (required for OVL)
            residual_audio: Difference audio [B, T] (required for OVL)
            subjective_metric_id: 0 for REL, 1 for OVL [B]

        Returns:
            Similarity scores [B]
        """
        audio_feats = F.normalize(audio_feats, p=2, dim=-1)
        text_feats = F.normalize(text_feats, p=2, dim=-1)
        cogr = torch.sum(audio_feats * text_feats, dim=-1)  # [B]

        # Calculate SNR score for OVL
        snr = self._calculate_snr(audio, residual_audio)
        ovl_score = (snr + cogr) / 2.0

        # Calculate fine-grained score for REL
        parsed_audio_feats = F.normalize(parsed_audio_feats, p=2, dim=-1)
        parsed_text_feats = F.normalize(parsed_text_feats, p=2, dim=-1)
        mask = parsed_mask.to(parsed_audio_feats.device)

        _, _, figr_f1 = self._greedy_matching(
            audio_emb=parsed_audio_feats,
            text_emb=parsed_text_feats,
            audio_mask=mask,
            text_mask=mask,
        )

        has_valid_mask = mask.any(dim=-1)  # [B]
        rel_score = torch.where(has_valid_mask, (cogr + figr_f1) / 2.0, cogr)

        # Select score based on subjective_metric_id (0=REL, 1=OVL)
        is_ovl = subjective_metric_id.bool().to(cogr.device)
        score = torch.where(is_ovl, ovl_score, rel_score)

        return score
