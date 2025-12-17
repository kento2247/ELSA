import argparse
import os

import laion_clap
import torch
from msclap import CLAP
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TTADataset


class TTAPreprocessDataset(TTADataset):
    def __init__(self, data_dir: str, split: str = "train"):
        super().__init__(data_dir=data_dir, split=split)

    def __getitem__(self, idx: int) -> dict:
        data = self.database[idx]
        return data


class MSClapEmbedder:
    def __init__(self, dtype: torch.dtype = torch.float32):
        self.model = CLAP(version="2023", use_cuda=True)
        self.max_text_len = 77  # MSCLAPのテキスト最大長
        self.dtype = dtype

    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """テキストをバッチで埋め込む"""
        truncated_texts = [t[: self.max_text_len] for t in texts]
        with torch.no_grad():
            text_embeddings = self.model.get_text_embeddings(truncated_texts)
        return text_embeddings.to(self.dtype)

    def embed_audios(self, audio_files: list[str]) -> torch.Tensor:
        """オーディオをバッチで埋め込む"""
        with torch.no_grad():
            audio_embeddings = self.model.get_audio_embeddings(audio_files)
        return audio_embeddings.to(self.dtype)


class LaionClapEmbedder:
    def __init__(self, dtype: torch.dtype = torch.float32):
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt("models/630k-audioset-best.pt")
        self.max_text_len = 77  # LaionCLAPのテキスト最大長
        self.dtype = dtype

    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """テキストをバッチで埋め込む（長いテキストはトランケート）"""
        truncated_texts = [t[: self.max_text_len] for t in texts]
        with torch.no_grad():
            text_embeddings = self.model.get_text_embedding(
                truncated_texts, use_tensor=True
            )
        return text_embeddings.to(self.dtype)

    def embed_audios(self, audio_files: list[str]) -> torch.Tensor:
        """オーディオをバッチで埋め込む"""
        with torch.no_grad():
            audio_embeddings = self.model.get_audio_embedding_from_filelist(
                x=audio_files, use_tensor=True
            )
        return audio_embeddings.to(self.dtype)


### feature saving functions ###


def save_feats(
    feats_dir: str,
    feats_name: str,
    dataset: str,
    audio_file_path: str,
    feats: torch.Tensor,
) -> str:
    file_name = os.path.basename(audio_file_path).replace(".wav", ".pt")
    save_path = os.path.join(feats_dir, feats_name, dataset, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(feats, save_path)
    return save_path


def save_batch_feats(
    feats_dir: str,
    datasets: list[str],
    feats_name: str,
    audio_file_paths: list[str],
    feats: torch.Tensor,
):
    """
    Save a batch of features to disk.
    Args:
        feats_dir: Directory to save features.
        datasets: List of dataset names corresponding to each audio file.
        feats_name: Name of the feature type (e.g., "msclap_audio"
        audio_file_paths: List of audio file paths.
        feats: Tensor of features [B, D].
    Returns:
        None
    """
    for i in range(len(audio_file_paths)):
        save_feats(
            feats_dir=feats_dir,
            feats_name=feats_name,
            dataset=datasets[i],
            audio_file_path=audio_file_paths[i],
            feats=feats[i],
        )


### feature extraction main ###


def msclap_extract(dataloader, feats_dir: str):
    msclap_embedder = MSClapEmbedder()

    for batch in tqdm(dataloader, desc="Extracting MSCLAP features"):
        audio_files = batch["audio_file_path"]
        texts = batch["text"]
        datasets = batch["dataset"]

        msclap_audio_embeddings = msclap_embedder.embed_audios(audio_files)
        msclap_text_embeddings = msclap_embedder.embed_texts(texts)

        save_batch_feats(
            feats_dir,
            datasets,
            "msclap_audio",
            audio_files,
            msclap_audio_embeddings,
        )
        save_batch_feats(
            feats_dir,
            datasets,
            "msclap_text",
            audio_files,
            msclap_text_embeddings,
        )


def laionclap_extract(dataloader, feats_dir: str):
    laion_clap_embedder = LaionClapEmbedder()
    for batch in tqdm(dataloader, desc="Extracting LaionCLAP features"):
        audio_files = batch["audio_file_path"]
        texts = batch["text"]
        datasets = batch["dataset"]

        laion_audio_embeddings = laion_clap_embedder.embed_audios(audio_files)
        laion_text_embeddings = laion_clap_embedder.embed_texts(texts)

        save_batch_feats(
            feats_dir,
            datasets,
            "laionclap_audio",
            audio_files,
            laion_audio_embeddings,
        )
        save_batch_feats(
            feats_dir,
            datasets,
            "laionclap_text",
            audio_files,
            laion_text_embeddings,
        )


def main(args):
    dataset = TTAPreprocessDataset(data_dir=args.data_dir, split=args.split)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

    msclap_extract(dataloader, args.feats_dir)
    laionclap_extract(dataloader, args.feats_dir)


### argument parser ###


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
        default="data/features",
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
    splits = ["train", "val", "test"]
    for split in splits:
        args.split = split
        main(args)
