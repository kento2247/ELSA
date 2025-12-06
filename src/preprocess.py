import argparse
import os

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
        return data


class MSClapEmbedder:
    def __init__(self):
        self.model = CLAP(version="2023", use_cuda=True)

    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """Embed a batch of texts."""
        with torch.no_grad():
            text_embeddings = self.model.get_text_embeddings(class_labels=texts)
        return text_embeddings

    def embed_audios(self, audio_files: list[str]) -> torch.Tensor:
        """Embed a batch of audios."""
        with torch.no_grad():
            audio_embeddings = self.model.get_audio_embeddings(audio_files=audio_files)
        return audio_embeddings


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
    audio_file_paths: list[str],
    audio_embs: torch.Tensor,
    text_embs: torch.Tensor,
):
    """
    Save batched audio/text embeddings.
    audio_embs: (B, D)
    text_embs:  (B, D)
    """
    for i in range(len(audio_file_paths)):
        dataset = datasets[i]
        audio_file_path = audio_file_paths[i]
        audio_feat = audio_embs[i]
        text_feat = text_embs[i]
        save_feats(
            feats_dir=feats_dir,
            feats_name="msclap_audio",
            dataset=dataset,
            audio_file_path=audio_file_path,
            feats=audio_feat,
        )
        save_feats(
            feats_dir=feats_dir,
            feats_name="msclap_text",
            dataset=dataset,
            audio_file_path=audio_file_path,
            feats=text_feat,
        )


def main(args):
    dataset = AudioCapPreprocessDataset(data_dir=args.data_dir, split=args.split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True)

    msclap_embedder = MSClapEmbedder()

    for batch in tqdm(dataloader, desc="Preprocessing"):
        audio_files = batch["audio_file_path"]
        texts = batch["text"]
        datasets = batch["dataset"]

        msclap_audio_embeddings = msclap_embedder.embed_audios(audio_files)
        msclap_text_embeddings = msclap_embedder.embed_texts(texts)
        save_batch_feats(
            args.feats_dir,
            datasets,
            audio_files,
            msclap_audio_embeddings,
            msclap_text_embeddings,
        )


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
