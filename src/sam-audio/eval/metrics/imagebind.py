# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from typing import Optional

import torch
from imagebind.models.imagebind_model import ModalityType, imagebind_huge

from sam_audio.ranking.imagebind import VideoTransform, load_and_transform_audio_data


class ImageBind(torch.nn.Module):
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.model = imagebind_huge(pretrained=checkpoint is None)
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model = self.model.eval()
        self.video_transform = VideoTransform()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

    def __call__(
        self,
        target_wavs: list[torch.Tensor],
        videos: list[torch.Tensor],
        target_wavs_sample_rate: int = 48_000,
        **kwargs,
    ) -> dict[str, list[float]]:
        audio_data = load_and_transform_audio_data(
            target_wavs, input_sample_rate=target_wavs_sample_rate
        )
        durations = [x.size(-1) / target_wavs_sample_rate for x in target_wavs]
        video_data = self.video_transform(videos, durations, audio_data.device)

        inputs = {ModalityType.AUDIO: audio_data, ModalityType.VISION: video_data}
        embs = self.model(inputs)
        audio_embs, video_embs = embs[ModalityType.AUDIO], embs[ModalityType.VISION]
        audio_embs, video_embs = (
            audio_embs / ((audio_embs**2).sum(dim=-1, keepdims=True) ** 0.5),
            video_embs / ((video_embs**2).sum(dim=-1, keepdims=True) ** 0.5),
        )
        bsz = len(target_wavs)
        candidates = len(audio_embs) // bsz
        scores = audio_embs.view(bsz, candidates, -1) @ video_embs.view(bsz, -1, 1)
        return {"ImageBind": scores.squeeze(1, 2).cpu().tolist()}
