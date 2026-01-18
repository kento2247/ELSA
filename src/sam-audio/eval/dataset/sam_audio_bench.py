# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import os
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset
from torchcodec.decoders import AudioDecoder, VideoDecoder


@dataclass
class Item:
    anchors: list[Tuple[str, float, float]]
    masked_video_frames: torch.Tensor
    audio_samples: torch.Tensor
    description: str


class SAMAudioBench(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_path,
        collate_fn,
        span: bool = True,
        visual: bool = True,
        subset: Optional[str] = None,
    ):
        self.dataset = load_dataset("facebook/sam-audio-bench")["test"]
        self.subset = subset
        self._span = span
        self._visual = visual
        if subset is not None:
            self.dataset = self.dataset.filter(lambda x: subset in x["paper_eval_sets"])

        self.cache_path = os.path.join(cache_path, "sam_audio_bench")
        self.collate_fn = collate_fn
        DATA_MSG = (
            f"`SAMAudioBench` requires the user to create a directory named {self.cache_path} "
            "see the README.md file for how to prepare"
        )
        assert os.path.exists(self.cache_path), DATA_MSG

    @property
    def visual(self):
        return self._visual

    def __len__(self):
        return len(self.dataset)

    def _get_path(
        self, video_id: str, source_dataset: str, start_offset: float, end_offset: float
    ) -> str:
        path = f"{self.cache_path}/{source_dataset}/{video_id}.mp4"
        select_frames = True

        if not os.path.exists(path):
            path = f"{self.cache_path}/{source_dataset}/{video_id}_{int(start_offset * 1000)}_{int(end_offset * 1000)}.mp4"
            select_frames = False

        if not os.path.exists(path):
            path = f"{self.cache_path}/{source_dataset}/{video_id}_{int(start_offset)}_{int(end_offset)}.mp4"

        if not os.path.exists(path):
            path = f"{self.cache_path}/{source_dataset}/{video_id}.{int(start_offset * 1000):08d}_{int(end_offset * 1000):08d}.mp4"

        return path, select_frames

    def collate(self, items: list[Item]):
        has_video = any(item.masked_video_frames is not None for item in items)
        return self.collate_fn(
            descriptions=[item.description for item in items],
            audios=[item.audio_samples for item in items],
            anchors=[item.anchors for item in items] if self._span else None,
            masked_videos=[item.masked_video_frames for item in items]
            if has_video and self._visual
            else None,
        )

    def _get_masked_video(self, item, video_path, select_frames):
        if item["mask_bytes"] is None:
            return None

        mask = torch.from_numpy(np.load(BytesIO(item["mask_bytes"]))["video_masklet"])

        video_decoder = VideoDecoder(video_path)
        if select_frames:
            video_frames = video_decoder.get_frames_played_in_range(
                item["start_offset"], item["end_offset"]
            ).data
        else:
            video_frames = video_decoder[:].data

        if mask.size(0) != video_frames.size(0):
            # It's possible that the mask and the video frames differ by a small amount
            # we interpolate the mask frame to match
            idxs = (
                torch.linspace(0, mask.size(0) - 1, video_frames.size(0)).round().long()
            )
            mask = mask[idxs]

        mask = mask.unsqueeze(1)

        if mask.shape[-2:] != video_frames.shape[-2:]:
            mask = F.interpolate(mask, size=video_frames.shape[-2:])

        import torchvision

        torchvision.io.write_video("test.mp4", video_frames.permute(0, 2, 3, 1), 30)
        torchvision.io.write_video(
            "test_mask.mp4", mask.unsqueeze(-1).expand(-1, -1, -1, 3) * 255, 30
        )

        return video_frames * mask

    def __getitem__(self, idx) -> Item:
        item = self.dataset[idx]

        video_path, select_frames = self._get_path(
            item["video_id"],
            item["source_dataset"],
            item["start_offset"],
            item["end_offset"],
        )
        assert os.path.exists(video_path), f"{video_path} does not exist!"

        audio_decoder = AudioDecoder(video_path)
        audio_samples = audio_decoder.get_samples_played_in_range(
            start_seconds=item["start_offset"] if select_frames else 0,
            stop_seconds=item["end_offset"] if select_frames else None,
        )

        if audio_samples.sample_rate != self.collate_fn.audio_sampling_rate:
            resampled_audio = torchaudio.functional.resample(
                audio_samples.data,
                audio_samples.sample_rate,
                self.collate_fn.audio_sampling_rate,
            )
        else:
            resampled_audio = audio_samples.data

        masked_video_frames = self._get_masked_video(item, video_path, select_frames)

        return Item(
            description=item["description"],
            anchors=[("+", start, end) for start, end in item["spans"]],
            masked_video_frames=masked_video_frames,
            audio_samples=resampled_audio.mean(0, keepdim=True),
        )
