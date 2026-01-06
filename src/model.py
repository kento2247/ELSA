import torch
import torch.nn as nn

from si_sdr import si_sdr


class TTAEvalModel(nn.Module):
    def __init__(
        self,
        min_db: float = -30.0,
        max_db: float = 30.0,
    ):
        """
        SI-SDR baseline model.

        Args:
            min_db: Minimum SI-SDR value (dB) for normalization
            max_db: Maximum SI-SDR value (dB) for normalization
        """
        super().__init__()
        self.min_db = min_db
        self.max_db = max_db

    def forward(
        self,
        audio: torch.Tensor,
        ref_audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SI-SDR between generated audio and reference audio.

        Args:
            audio: Generated audio waveforms [B, T]
            ref_audio: Reference audio waveforms [B, T]

        Returns:
            Normalized SI-SDR scores [B] in range [0, 1]
        """
        # Compute SI-SDR in dB
        si_sdr_db = si_sdr(audio, ref_audio, zero_mean=True, return_db=True)

        # Normalize to [0, 1] range
        normalized = (si_sdr_db - self.min_db) / (self.max_db - self.min_db)
        normalized = torch.clamp(normalized, 0.0, 1.0)

        return normalized
