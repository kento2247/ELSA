import torch
import torch.nn as nn
import torch.nn.functional as F

from fd_openl3.openl3_fd import calculate_embd_statistics, calculate_frechet_distance


class TTAEvalModel(nn.Module):
    def __init__(
        self,
    ):
        """
        Frechet Distance (openl3, inversed) baseline model.

        Args:
        """
        super().__init__()

    def forward(
        self,
        audio: torch.Tensor,
        ref_audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Fréchet Distance (openl3, inversed) between generated audio and reference audio.

        Args:
            audio: Generated audio waveforms [B, T, D]
            ref_audio: Reference audio waveforms [B, T, D]

        Returns:
            Normalized Fréchet Distance scores [B] in range [0, 1]
        """
        # Compute Fréchet Distance
        audio, ref_audio = audio.cpu().numpy(), ref_audio.cpu().numpy()
        fds = torch.randn(audio.shape[0])
        for i, (audio_sample, ref_sample) in enumerate(zip(audio, ref_audio)):
            audio_sample = audio_sample.squeeze(0)
            ref_sample = ref_sample.squeeze(0)
            audio_mu, audio_sigma = calculate_embd_statistics(audio_sample)
            ref_mu, ref_sigma = calculate_embd_statistics(ref_sample)
            fd = calculate_frechet_distance(audio_mu, audio_sigma, ref_mu, ref_sigma)
            fds[i] = fd

        return fds
