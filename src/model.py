import torch
import torch.nn as nn
import torch.nn.functional as F


class TTAEvalModel(nn.Module):
    def __init__(self, logit_scale=4.6052):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor([logit_scale]))

    def forward(
        self,
        audio_feats: torch.Tensor,
        pam_prompt_1: torch.Tensor,
        pam_prompt_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity score using Transformer architecture.

        Args:
            audio_feats: Audio features from MSCLAP [B, D]
            text_feats: Text features from MSCLAP [B, D]

        Returns:
            Similarity scores [B, 1]
        """
        audio_feats = F.normalize(audio_feats, p=2, dim=-1)
        pam_prompt_1 = F.normalize(pam_prompt_1, p=2, dim=-1)
        pam_prompt_2 = F.normalize(pam_prompt_2, p=2, dim=-1)

        logit_scale = self.logit_scale.exp()

        sim_1 = torch.sum(audio_feats * pam_prompt_1, dim=-1, keepdim=True)
        sim_2 = torch.sum(audio_feats * pam_prompt_2, dim=-1, keepdim=True)
        logits = torch.cat([sim_1, sim_2], dim=-1) * logit_scale
        similarity = F.softmax(logits, dim=-1)[:, 0]
        similarity = torch.clamp(similarity, min=0.0, max=1.0)
        return similarity
