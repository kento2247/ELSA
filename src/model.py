import torch
import torch.nn as nn
import torch.nn.functional as F


class TTAEvalModel(nn.Module):
    def __init__(self):
        """
        KL Divergence (PaSST) baseline model.

        Args:
        """
        super().__init__()

    def forward(self, audio: torch.Tensor, ref_audio: torch.Tensor) -> torch.Tensor:
        """
        Compute KL Divergence (PaSST) between generated audio and reference audio.

        Args:
            audio: Generated audio waveforms [B, D]
            ref_audio: Reference audio waveforms [B, D]

        Returns:
            Normalized KL Divergence scores [B] in range [0, 1]
        """
        # Compute KL Divergence
        B, _ = audio.shape
        kl_div = torch.zeros(B).to(audio.device)
        for i in range(B):
            kl_div[i] = F.kl_div(
                (ref_audio[i] + 1e-6).log(), audio[i], reduction="sum", log_target=False
            )

        return kl_div
