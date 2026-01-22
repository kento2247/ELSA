import torch
import torch.nn as nn
import torch.nn.functional as F


class TTAEvalModel(nn.Module):
    """TTA Evaluation Model with separate REL/OVL configurations."""

    def __init__(
        self,
        embedding_dim: int = 512,
        # REL feature flags (best: adaptive_fusion + contrastive + cogr_norm)
        rel_use_gaussian_calibration: bool = True,
        rel_use_adaptive_fusion: bool = True,
        rel_use_contrastive: bool = True,
        rel_use_cogr_norm: bool = True,
        # OVL feature flags (best: contrastive + cogr_norm, NO adaptive_fusion)
        ovl_use_gaussian_calibration: bool = False,
        ovl_use_adaptive_fusion: bool = False,
        ovl_use_contrastive: bool = True,
        ovl_use_cogr_norm: bool = True,
        # Hyperparameters
        sigma: float = 0.15,
        cogr_weight: float = 0.2,
        figr_weight: float = 0.8,
    ):
        """Initialize TTAEvalModel.

        Args:
            embedding_dim: Dimension of CLAP embeddings.
            rel_use_*: Feature flags for REL (relevance) scoring.
            ovl_use_*: Feature flags for OVL (overall quality) scoring.
            sigma: Standard deviation for Gaussian prior.
            cogr_weight: Base weight for COGR (when not using adaptive).
            figr_weight: Base weight for FIGR (when not using adaptive).
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # REL feature flags
        self.rel_use_gaussian_calibration = rel_use_gaussian_calibration
        self.rel_use_adaptive_fusion = rel_use_adaptive_fusion
        self.rel_use_contrastive = rel_use_contrastive
        self.rel_use_cogr_norm = rel_use_cogr_norm

        # OVL feature flags
        self.ovl_use_gaussian_calibration = ovl_use_gaussian_calibration
        self.ovl_use_adaptive_fusion = ovl_use_adaptive_fusion
        self.ovl_use_contrastive = ovl_use_contrastive
        self.ovl_use_cogr_norm = ovl_use_cogr_norm

        # Hyperparameters
        self.sigma = sigma
        self.cogr_weight = cogr_weight
        self.figr_weight = figr_weight

        # Quality prompt embeddings (loaded from precomputed)
        self.register_buffer("quality_high_emb", None)
        self.register_buffer("quality_low_emb", None)

        # Negative/unrelated prompt embedding for contrastive scoring
        self.register_buffer("unrelated_emb", None)

    def load_quality_prompts(
        self,
        high_emb: torch.Tensor,
        low_emb: torch.Tensor,
        unrelated_emb: torch.Tensor = None,
    ):
        """Load precomputed quality prompt embeddings.

        Args:
            high_emb: Embedding for high quality prompt [D]
            low_emb: Embedding for low quality prompt [D]
            unrelated_emb: Embedding for unrelated/negative prompt [D] (optional)
        """
        self.quality_high_emb = F.normalize(high_emb.unsqueeze(0), p=2, dim=-1)
        self.quality_low_emb = F.normalize(low_emb.unsqueeze(0), p=2, dim=-1)
        if unrelated_emb is not None:
            self.unrelated_emb = F.normalize(unrelated_emb.unsqueeze(0), p=2, dim=-1)

    def _gaussian_prior_weight(self, score: torch.Tensor) -> torch.Tensor:
        """Compute adaptive weight α based on score proximity to boundaries.

        Args:
            score: Raw scores [B] in range [0, 1]

        Returns:
            Adaptive weights α [B] in range [0, 1]
        """
        boundary_dist = torch.min(score, 1.0 - score)
        alpha = torch.exp(-(boundary_dist**2) / (2 * self.sigma**2))
        return alpha

    def _gaussian_calibration(
        self, raw_score: torch.Tensor, use_calibration: bool
    ) -> torch.Tensor:
        """Apply Gaussian prior score calibration.

        Args:
            raw_score: Raw similarity scores [B]
            use_calibration: Whether to apply calibration

        Returns:
            Calibrated scores [B]
        """
        if not use_calibration:
            return raw_score

        raw_score = torch.clamp(raw_score, 0.0, 1.0)
        alpha = self._gaussian_prior_weight(raw_score)
        prior_mean = 0.5
        calibrated = (1 - alpha * 0.3) * raw_score + (alpha * 0.3) * prior_mean
        return calibrated

    def _confidence_weight(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute confidence weight based on mask coverage.

        Args:
            mask: Valid segment mask [B, S]

        Returns:
            Confidence weights [B] in range [0, 1]
        """
        valid_counts = mask.float().sum(dim=-1)
        confidence = torch.sigmoid((valid_counts - 2) * 0.5)
        confidence = torch.where(valid_counts > 0, confidence, torch.zeros_like(confidence))
        return confidence

    def compute_quality_score(
        self, audio_emb: torch.Tensor, use_calibration: bool = False
    ) -> torch.Tensor:
        """Compute PAM-style audio quality score (text-independent).

        Args:
            audio_emb: Audio embeddings [B, D]
            use_calibration: Whether to apply Gaussian calibration

        Returns:
            Quality scores [B] - probability of "high quality"
        """
        audio_emb = F.normalize(audio_emb, p=2, dim=-1)
        high_sim = torch.sum(audio_emb * self.quality_high_emb, dim=-1)
        low_sim = torch.sum(audio_emb * self.quality_low_emb, dim=-1)
        logits = torch.stack([high_sim, low_sim], dim=-1)
        quality = F.softmax(logits, dim=-1)[:, 0]
        return self._gaussian_calibration(quality, use_calibration)

    def _compute_contrastive_bonus(
        self,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
        use_contrastive: bool,
    ) -> torch.Tensor:
        """Compute contrastive bonus using negative/unrelated prompt.

        Args:
            audio_emb: Normalized audio embeddings [B, D]
            text_emb: Normalized text embeddings [B, D]
            use_contrastive: Whether to compute contrastive bonus

        Returns:
            Contrastive bonus [B] in range [0, 1]
        """
        if not use_contrastive or self.unrelated_emb is None:
            return torch.zeros(audio_emb.shape[0], device=audio_emb.device)

        text_sim = torch.sum(audio_emb * text_emb, dim=-1)
        unrelated_sim = torch.sum(audio_emb * self.unrelated_emb, dim=-1)
        margin = text_sim - unrelated_sim
        bonus = torch.sigmoid(margin * 2)
        return bonus

    def _greedy_matching(
        self,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Greedy matching for fine-grained alignment.

        Args:
            audio_emb: Audio segment embeddings [B, S_a, D]
            text_emb: Text phrase embeddings [B, S_t, D]
            audio_mask: Valid audio segment mask [B, S_a]
            text_mask: Valid text phrase mask [B, S_t]

        Returns:
            (Precision, Recall, F1) each [B]
        """
        sim = torch.bmm(text_emb, audio_emb.transpose(1, 2))

        mask = torch.bmm(
            text_mask.unsqueeze(2).float(), audio_mask.unsqueeze(1).float()
        )
        mask = mask.to(sim.device)
        sim = sim * mask

        word_precision = sim.max(dim=2)[0]
        word_recall = sim.max(dim=1)[0]

        text_mask_float = text_mask.float().to(word_precision.device)
        text_valid_counts = text_mask_float.sum(dim=1).clamp(min=1.0)
        precision = (word_precision * text_mask_float).sum(dim=1) / text_valid_counts

        audio_mask_float = audio_mask.float().to(word_recall.device)
        audio_valid_counts = audio_mask_float.sum(dim=1).clamp(min=1.0)
        recall = (word_recall * audio_mask_float).sum(dim=1) / audio_valid_counts

        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return precision, recall, f1

    def _compute_relevance(
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor,
        parsed_audio_feats: torch.Tensor,
        parsed_text_feats: torch.Tensor,
        parsed_mask: torch.Tensor,
        use_gaussian_calibration: bool,
        use_adaptive_fusion: bool,
        use_contrastive: bool,
        use_cogr_norm: bool,
    ) -> torch.Tensor:
        """Compute semantic relevance score.

        Args:
            audio_feats: [B, D]
            text_feats: [B, D]
            parsed_audio_feats: [B, S, D]
            parsed_text_feats: [B, S, D]
            parsed_mask: Valid segment mask [B, S]
            use_gaussian_calibration: Apply Gaussian calibration
            use_adaptive_fusion: Use confidence-weighted fusion
            use_contrastive: Use contrastive bonus
            use_cogr_norm: Normalize COGR to [0,1]

        Returns:
            Relevance scores [B]
        """
        # Normalize embeddings
        audio_feats_norm = F.normalize(audio_feats, p=2, dim=-1)
        text_feats_norm = F.normalize(text_feats, p=2, dim=-1)

        # COGR: Coarse-grained similarity
        cogr = torch.sum(audio_feats_norm * text_feats_norm, dim=-1)
        if use_cogr_norm:
            cogr = (cogr + 1) / 2  # Map from [-1, 1] to [0, 1]

        # Fine-grained matching
        parsed_audio_feats = F.normalize(parsed_audio_feats, p=2, dim=-1)
        parsed_text_feats = F.normalize(parsed_text_feats, p=2, dim=-1)
        mask = parsed_mask.to(parsed_audio_feats.device)

        precision, recall, figr_f1 = self._greedy_matching(
            audio_emb=parsed_audio_feats,
            text_emb=parsed_text_feats,
            audio_mask=mask,
            text_mask=mask,
        )

        has_valid_mask = mask.any(dim=-1)

        # Compute weights
        if use_adaptive_fusion:
            confidence = self._confidence_weight(mask)
            cogr_w = 0.2 + 0.3 * (1 - confidence)
            figr_w = 0.8 - 0.3 * (1 - confidence)
            figr_combined = 0.7 * figr_f1 + 0.15 * precision + 0.15 * recall
        else:
            cogr_w = self.cogr_weight
            figr_w = self.figr_weight
            figr_combined = figr_f1

        # Fused score
        combined_score = cogr_w * cogr + figr_w * figr_combined

        # Add contrastive bonus
        if use_contrastive and self.unrelated_emb is not None:
            contrastive_bonus = self._compute_contrastive_bonus(
                audio_feats_norm, text_feats_norm, use_contrastive
            )
            combined_score = 0.9 * combined_score + 0.1 * contrastive_bonus

        # Fallback to COGR when no valid segments
        score = torch.where(has_valid_mask, combined_score, cogr)

        # Apply calibration
        return self._gaussian_calibration(score, use_gaussian_calibration)

    def forward(
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor,
        parsed_audio_feats: torch.Tensor,
        parsed_text_feats: torch.Tensor,
        parsed_mask: torch.Tensor,
        metric_id: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute evaluation score based on metric type.

        REL (metric_id=0): Uses REL-optimized config (adaptive_fusion + contrastive + cogr_norm)
        OVL (metric_id=1): Uses OVL-optimized config (contrastive + cogr_norm, NO adaptive_fusion)

        Args:
            audio_feats: [B, D]
            text_feats: [B, D]
            parsed_audio_feats: [B, S, D]
            parsed_text_feats: [B, S, D]
            parsed_mask: Valid segment mask [B, S]
            metric_id: 0 for REL, 1 for OVL [B]

        Returns:
            Evaluation scores [B]
        """
        # Compute REL score with REL-optimized settings
        rel_score = self._compute_relevance(
            audio_feats,
            text_feats,
            parsed_audio_feats,
            parsed_text_feats,
            parsed_mask,
            use_gaussian_calibration=self.rel_use_gaussian_calibration,
            use_adaptive_fusion=self.rel_use_adaptive_fusion,
            use_contrastive=self.rel_use_contrastive,
            use_cogr_norm=self.rel_use_cogr_norm,
        )

        if metric_id is None or self.quality_high_emb is None:
            return rel_score

        # Compute OVL-specific relevance with OVL-optimized settings
        ovl_rel_score = self._compute_relevance(
            audio_feats,
            text_feats,
            parsed_audio_feats,
            parsed_text_feats,
            parsed_mask,
            use_gaussian_calibration=self.ovl_use_gaussian_calibration,
            use_adaptive_fusion=self.ovl_use_adaptive_fusion,
            use_contrastive=self.ovl_use_contrastive,
            use_cogr_norm=self.ovl_use_cogr_norm,
        )

        # Compute quality score
        quality_score = self.compute_quality_score(
            audio_feats, use_calibration=self.ovl_use_gaussian_calibration
        )

        is_ovl = (metric_id == 1).to(audio_feats.device)

        # OVL fusion weights
        if self.ovl_use_adaptive_fusion:
            mask = parsed_mask.to(audio_feats.device)
            confidence = self._confidence_weight(mask)
            rel_weight = 0.5 + 0.1 * confidence
            qual_weight = 1.0 - rel_weight
        else:
            rel_weight = 0.5
            qual_weight = 0.5

        ovl_score = rel_weight * ovl_rel_score + qual_weight * quality_score

        return torch.where(is_ovl, ovl_score, rel_score)
