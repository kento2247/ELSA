#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Waveformer-main
@File    ：CLAPSep.py
@IDE     ：PyCharm
@Author  ：Aisaka/Hao Ma @SDU
@Date    ：2024/2/28 下午1:12
"""

import copy

import laion_clap
import librosa
import loralib as lora
import torch
import torchaudio
from torch import nn
from torchlibrosa import ISTFT, STFT
from torchlibrosa.stft import magphase

from .CLAPSep_decoder import HTSAT_Decoder


def set_module(model, submodule_key, module):
    tokens = submodule_key.split(".")
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def process_model(model, rank):
    for n, module in model.named_modules():
        if "WindowAttention" in str(type(module)):
            for n_, layer in module.named_modules():
                if isinstance(layer, torch.nn.Linear):
                    lora_layer = lora.Linear(
                        layer.in_features,
                        layer.out_features,
                        r=rank,
                        bias=hasattr(layer, "bias"),
                        merge_weights=True,
                    )
                    lora_layer.weight = layer.weight
                    if hasattr(layer, "bias"):
                        lora_layer.bias = layer.bias
                    set_module(model, n + "." + n_, lora_layer)
    return model


class CLAPSep(nn.Module):
    def __init__(self, model_config, CLAP_path, use_lora=True, rank=16, nfft=1024):
        super().__init__()
        self.resampler = torchaudio.transforms.Resample(32000, 48000)
        self.clap_model = laion_clap.CLAP_Module(
            enable_fusion=False, amodel="HTSAT-base", device="cpu"
        )
        self.clap_model.load_ckpt(CLAP_path)
        for p in self.clap_model.parameters():
            p.requires_grad = False
        self.audio_branch = copy.deepcopy(self.clap_model.model.audio_branch)
        if use_lora:
            process_model(self.audio_branch, rank)
        self.decoder_model = HTSAT_Decoder(**model_config)
        self.stft = STFT(
            n_fft=nfft,
            hop_length=320,
            win_length=nfft,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.istft = ISTFT(
            n_fft=nfft,
            hop_length=320,
            win_length=nfft,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.features = self.install_forward_hooks()

    def wav_reconstruct(self, mask, mag_x, cos_x, sin_x, length):
        mag_y = torch.nn.functional.relu_(mag_x * mask)
        cos_y = cos_x
        sin_y = sin_x
        pred = self.istft(mag_y * cos_y, mag_y * sin_y, length=length)
        return pred

    def inference_from_data(self, mixed, pos_prompt: list[str], neg_prompt: list[str]):
        self.eval()
        real, imag = self.stft(mixed)
        mag, cos, sin = magphase(real, imag)
        self.features.append(mag)
        with torch.no_grad():
            embed_pos = self.clap_model.get_text_embedding(pos_prompt, use_tensor=True)
            if len(neg_prompt) == 0:
                embed_neg = torch.zeros_like(embed_pos)
            else:
                embed_neg = self.clap_model.get_text_embedding(
                    neg_prompt, use_tensor=True
                )
            embed = torch.nn.functional.normalize(
                torch.concat([embed_pos, embed_neg], dim=-1), dim=-1
            )
            self.audio_branch({"waveform": self.resampler(mixed)})
            mask = self.decoder_model(
                hidden_state=self.features[-1],
                skip_features=self.features[:-1],
                embed=embed,
            )
            pred = self.wav_reconstruct(mask, mag, cos, sin, length=mixed.size(-1))
        del self.features[:]
        return pred

    def install_forward_hooks(self):
        features = []

        def get_features_list(_, __, output):
            features.append(output)

        def get_features_list_basic_layer(_, __, output):
            features.append(output[0])

        def spectrogram_padding(_, __, out):
            return torch.nn.functional.pad(out, (0, 0, 0, 1024 - out.size(2)))

        self.audio_branch.spectrogram_extractor.register_forward_hook(
            spectrogram_padding
        )
        self.audio_branch.patch_embed.register_forward_hook(get_features_list)
        for module in self.audio_branch.layers:
            module.register_forward_hook(get_features_list_basic_layer)
        return features


if __name__ == "__main__":
    model_config = {
        "lan_embed_dim": 1024,
        "depths": [1, 1, 1, 1],
        "embed_dim": 128,
        "encoder_embed_dim": 128,
        "phase": False,
        "spec_factor": 8,
        "d_attn": 640,
        "n_masker_layer": 3,
        "conv": False,
    }
    CLAP_path = "./music_audioset_epoch_15_esc_90.14.pt"

    model = CLAPSep(model_config, CLAP_path)
    ckpt = torch.load("best_model.ckpt", map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    audio, fs = librosa.load("./510_25.221254348754883_mixture.wav", sr=32000)
    pred = model.inference_from_data(
        torch.tensor(audio).unsqueeze(0),
        pos_prompt=[""],
        neg_prompt=["A vehicle engine revving then powering down."],
    )
    import soundfile as sf

    sf.write("./pred.wav", pred.squeeze().numpy(), 32000)
