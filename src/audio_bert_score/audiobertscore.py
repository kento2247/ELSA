import logging
from typing import Tuple

import torch
import torchaudio.compliance.kaldi as kaldi_fbank

logging.getLogger("transformers").setLevel(logging.ERROR)


def make_features(
    waveform: torch.Tensor,
    sr: int = 16000,
    mel_bins: int = 128,
    target_length: int = 1024,
) -> torch.Tensor:
    if sr != 16000:
        raise ValueError("Waveform must be 16 kHz")

    fbank = kaldi_fbank.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=mel_bins,
        dither=0.0,
        frame_shift=10,
    )
    pad = target_length - fbank.size(0)
    fbank = (
        torch.nn.functional.pad(fbank, (0, 0, 0, pad))
        if pad > 0
        else fbank[:target_length]
    )
    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank.unsqueeze(0)


def bert_score(
    gen: torch.Tensor,
    ref: torch.Tensor,
    lam: float,
    p: float,
) -> Tuple[float, float, float]:
    sim = (gen @ ref.T) / (
        torch.norm(gen, dim=1, keepdim=True) * torch.norm(ref, dim=1).unsqueeze(0)
    )
    sim_pos = torch.clamp(sim, min=0.0)
    term1_p = sim.max(dim=1)[0].mean()
    term1_r = sim.max(dim=0)[0].mean()
    term2_p = ((sim_pos.pow(p).mean(dim=1)).pow(1.0 / p)).mean()
    term2_r = ((sim_pos.pow(p).mean(dim=0)).pow(1.0 / p)).mean()
    global_p = lam * term1_p + (1.0 - lam) * term2_p
    global_r = lam * term1_r + (1.0 - lam) * term2_r
    global_f1 = 2 * global_p * global_r / (global_p + global_r + 1e-8)
    return global_p.item(), global_r.item(), global_f1.item()
