import argparse

import torch
import torchaudio
from msclap import CLAP
from tqdm import tqdm

from dataset import AudioCapDataset


class AudioCapPreprocessDataset(AudioCapDataset):
    def __init__(self, data_dir: str, split: str = "train"):
        super().__init__(data_dir, split)

    def __getitem__(self, idx: int) -> dict:
        data = self.database[idx]
        data["audio"] = self._load_wav(data["audio_file_path"])
        return data


def arg_parser():
    parser = argparse.ArgumentParser(description="Audio Captioning Preprocessing")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--feats_dir",
        type=str,
        default="features",
        help="Directory to save the preprocessed features",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=32,
        help="Batch size for DataLoader",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()

    dataset = AudioCapPreprocessDataset(data_dir=args.data_dir, split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True)
