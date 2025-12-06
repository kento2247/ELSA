import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AudioCapDataset
from utils.eval_methods import (
    kendall_tau,
    mse,
    pearson_correlation,
    spearman_correlation,
)


class AudioCapEval:
    def __init__(self): ...

    def train(self): ...

    def evaluate(self) -> float: ...

    def test(self) -> float: ...

    def save_model(self, filename: str = "model.pt"): ...

    def load_model(self, filename: str = "model.pt"): ...


def arg_parser():
    parser = argparse.ArgumentParser(description="Audio Captioning Evaluation")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Mode: train or test",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    return parser


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()

    evaluator = AudioCapEval()

    if args.mode == "train":
        evaluator.train()
        evaluator.test()
    elif args.mode == "test":
        evaluator.load_model("best_model.pt")
        evaluator.test()
