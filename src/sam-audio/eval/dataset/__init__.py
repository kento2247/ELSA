# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from typing import Callable

from .musdb import MUSDB
from .sam_audio_bench import SAMAudioBench

SETTINGS = {
    # Text-only settings
    "sfx": (
        SAMAudioBench,
        {"span": False, "visual": False, "subset": "others-50:text-only"},
    ),
    "speech": (
        SAMAudioBench,
        {"span": False, "visual": False, "subset": "speech-clean-50:text-only"},
    ),
    "speaker": (
        SAMAudioBench,
        {"span": False, "visual": False, "subset": "spk-50:text-only"},
    ),
    "music": (
        SAMAudioBench,
        {"span": False, "visual": False, "subset": "music-clean-50:text-only"},
    ),
    "instr-wild": (
        SAMAudioBench,
        {"span": False, "visual": False, "subset": "instr-50:text-only"},
    ),
    "instr-pro": (MUSDB, {}),
    # Span settings
    "sfx-span": (
        SAMAudioBench,
        {"span": True, "visual": False, "subset": "others-50:text+span"},
    ),
    "speech-span": (
        SAMAudioBench,
        {"span": True, "visual": False, "subset": "speech-clean-50:text+span"},
    ),
    "speaker-span": (
        SAMAudioBench,
        {"span": True, "visual": False, "subset": "spk-50:text+span"},
    ),
    "music-span": (
        SAMAudioBench,
        {"span": True, "visual": False, "subset": "music-clean-50:text+span"},
    ),
    "instr-wild-span": (
        SAMAudioBench,
        {"span": True, "visual": False, "subset": "instr-50:text+span"},
    ),
    # Visual settings
    "sfx-visual": (
        SAMAudioBench,
        {"span": False, "visual": True, "subset": "others-onscreen-50:visual-only"},
    ),
    "speaker-visual": (
        SAMAudioBench,
        {"span": False, "visual": True, "subset": "spk-onscreen-50:visual-only"},
    ),
    "instr-wild-visual": (
        SAMAudioBench,
        {"span": False, "visual": True, "subset": "instr-onscreen-50:visual-only"},
    ),
}


def make_dataset(setting: str, cache_path: str, collate_fn: Callable):
    dataset, kwargs = SETTINGS[setting]
    return dataset(cache_path=cache_path, collate_fn=collate_fn, **kwargs)
