import torch
import torch.nn as nn

from audio_bert_score.audiobertscore import bert_score


class TTAEvalModel(nn.Module):
    def __init__(
        self,
        lam=-3.5,
        p=106.0,
    ):
        """
        AudioBERTScore baseline model.

        Args:
        """
        super().__init__()
        self.lam = lam
        self.p = p

    def forward(self, audio: torch.Tensor, ref_audio: torch.Tensor) -> torch.Tensor:
        """
        Compute AudioBERTScore between generated audio and reference audio.

        Args:
            audio: Generated audio waveforms [B, N, D]
            ref_audio: Reference audio waveforms [B, N, D]

        Returns:
            AudioBERTScore scores [B]
        """
        # Compute AudioBERTScore
        B, N, D = audio.shape
        audio_bert_score = torch.zeros(B).to(audio.device)
        for i in range(B):
            p, r, f1 = bert_score(audio[i], ref_audio[i], self.lam, self.p)
            audio_bert_score[i] = f1

        return audio_bert_score
