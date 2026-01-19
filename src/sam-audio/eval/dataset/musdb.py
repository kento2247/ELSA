# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import os
from subprocess import check_call

import torchaudio
from datasets import load_dataset
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder


def cache_file(url, outfile):
    if not os.path.exists(outfile):
        print("Downloading musdb18hq dataset...")
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        check_call(["curl", "--url", url, "--output", outfile + ".tmp"])
        os.rename(outfile + ".tmp", outfile)


class MUSDB(Dataset):
    def __init__(
        self,
        collate_fn,
        sample_rate: int = 48_000,
        cache_path: str = os.path.expanduser("~/.cache/sam_audio"),
    ):
        self.cache_path = os.path.join(cache_path, "musdb18hq")
        self.ds = self.get_dataset(cache_path)
        self.captions = ["bass", "drums", "vocals"]
        self.collate_fn = collate_fn
        self.sample_rate = sample_rate

    @property
    def visual(self):
        return False

    def get_dataset(self, cache_path):
        zip_file = os.path.join(cache_path, "musdb18hq.zip")
        url = "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
        cache_file(url, zip_file)
        extracted_dir = os.path.join(cache_path, "musdb18hq")
        if not os.path.exists(extracted_dir):
            check_call(["unzip", zip_file, "-d", extracted_dir + ".tmp"])
            os.rename(extracted_dir + ".tmp", extracted_dir)
        return load_dataset("facebook/sam-audio-musdb18hq-test")["test"]

    def __len__(self):
        return len(self.ds)

    def collate(self, items):
        audios, descriptions = zip(*items, strict=False)
        return self.collate_fn(
            audios=audios,
            descriptions=descriptions,
        )

    def __getitem__(self, idx):
        item = self.ds[idx]
        path = os.path.join(self.cache_path, "test", item["id"], "mixture.wav")
        assert os.path.exists(path), f"{path} does not exist!"
        decoder = AudioDecoder(path)
        data = decoder.get_samples_played_in_range(item["start_time"], item["end_time"])
        wav = data.data
        if data.sample_rate != self.sample_rate:
            wav = torchaudio.functional.resample(
                wav, data.sample_rate, self.sample_rate
            )
        wav = wav.mean(0, keepdim=True)
        return wav, item["description"]


if __name__ == "__main__":
    dataset = MUSDB(lambda **kwargs: None)
    print(len(dataset))
    print(dataset[0])
