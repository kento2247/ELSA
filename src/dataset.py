import os
from typing import Literal

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm


class TTADataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: Literal["train", "val", "test"] = "train",
        bitrate: int = 16000,
        max_len: int = 160000 * 10,
        dataset_name: list = ["relate", "pam_audio", "pam_music"],
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize TTADataset with specified data directory and split."""
        self.data_dir = data_dir
        self.split = split
        self.bitrate = bitrate
        self.max_len = max_len
        self.dtype = dtype
        self.database = []
        if "relate" in dataset_name:
            self._load_relate_data()
        if "pam_audio" in dataset_name:
            self._load_pam_audio_data()
        if "pam_music" in dataset_name:
            self._load_pam_music_data()

    def _load_relate_data(self) -> None:
        """Load RELATE dataset and split into train, val, test sets."""
        relate_rel_path = os.path.join(self.data_dir, "RELATE", "scores", "REL.csv")
        relate_rel_data = pd.read_csv(relate_rel_path)

        if self.split == "train" or self.split == "test":
            data = relate_rel_data[relate_rel_data["in RELATE dataset"] == self.split]
        elif self.split == "val":
            data = relate_rel_data[relate_rel_data["in RELATE dataset"] == "validation"]
        data = data.reset_index(drop=True)
        for _, row in tqdm(
            data.iterrows(), total=len(data), desc=f"Loading {self.split} data"
        ):
            wavname: str = row["wavname"]
            text: str = row["text"]
            score: float = float(row["score"])
            audio_file_path = os.path.join(self.data_dir, f"wav{wavname}")
            ref_audio_file_path = os.path.join(
                self.data_dir,
                "wav",
                "audiocaps",
                self.split if self.split != "val" else "test",
                f"{wavname.split('/')[-1]}",
            )
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Wav file not found: {audio_file_path}")
            if not os.path.exists(ref_audio_file_path):
                ref_audio_file_path = ""
            self.database.append(
                {
                    "dataset": "relate",
                    "audio_file_path": audio_file_path,
                    "ref_audio_file_path": ref_audio_file_path,
                    "text": text,
                    "score": score,
                }
            )

    def _load_pam_audio_data(self) -> None:
        """Load PAM audio dataset as test sets."""
        if self.split != "test":
            return
        pam_audio_data_path = os.path.join(
            self.data_dir, "human_eval", "audio", "scores.csv"
        )
        pam_audio_data = pd.read_csv(pam_audio_data_path)

        for _, row in tqdm(
            pam_audio_data.iterrows(),
            total=len(pam_audio_data),
            desc="Loading PAM audio data",
        ):
            text: str = row["Text"]
            model: str = row["Model"]
            file_name: str = row["File Name"]
            score: float = float(row["REL"])
            self.database.append(
                {
                    "dataset": "pam_audio",
                    "audio_file_path": os.path.join(
                        self.data_dir, "human_eval", "audio", model, f"{file_name}.wav"
                    ),
                    "ref_audio_file_path": "",
                    "text": text,
                    "score": score,
                }
            )

    def _load_pam_music_data(self) -> None:
        """Load PAM music dataset as test sets."""
        if self.split != "test":
            return
        pam_music_data_path = os.path.join(
            self.data_dir, "human_eval", "music", "scores.csv"
        )
        pam_music_data = pd.read_csv(pam_music_data_path)

        for _, row in tqdm(
            pam_music_data.iterrows(),
            total=len(pam_music_data),
            desc="Loading PAM music data",
        ):
            text: str = row["Text"]
            model: str = row["Model"]
            file_name: str = row["File Name"]
            score: float = float(row["REL"])
            self.database.append(
                {
                    "dataset": "pam_music",
                    "audio_file_path": os.path.join(
                        self.data_dir, "human_eval", "music", model, f"{file_name}.wav"
                    ),
                    "ref_audio_file_path": "",
                    "text": text,
                    "score": score,
                }
            )

    def _load_wav(self, file_path: str) -> torch.Tensor:
        """Load wav file based on dataset and filename."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Wav file not found: {file_path}")

        waveform, bitrate = torchaudio.load(file_path)  # 16kHz
        if bitrate != self.bitrate:
            waveform = torchaudio.functional.resample(waveform, bitrate, self.bitrate)
        wav = waveform.squeeze(0)  # [T]

        # truncate/pad
        if wav.shape[0] >= self.max_len:
            wav = wav[: self.max_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.max_len - wav.shape[0]))
        return wav.to(self.dtype)

    def __len__(self):
        return len(self.database)

    def _load_pre_extracted_feats(
        self, feats_name: str, dataset_name: str, file_name: str
    ) -> torch.Tensor:
        """Load pre-extracted features from the specified file path."""
        feats_dir = os.path.join(self.data_dir, "features", feats_name)
        feat_path = os.path.join(feats_dir, dataset_name, file_name)
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Feature file not found: {feat_path}")
        feats = torch.load(feat_path, map_location="cpu")
        return feats.to(self.dtype)

    def __getitem__(self, idx):
        """Get item by index from the dataset."""
        data = self.database[idx]
        data["audio"] = self._load_wav(data["audio_file_path"])

        # load pre-extracted features if exist
        file_name = os.path.basename(data["audio_file_path"]).replace(".wav", ".pt")
        dataset_name = data["dataset"]

        data["msclap_audio"] = self._load_pre_extracted_feats(
            feats_name="msclap_audio", dataset_name=dataset_name, file_name=file_name
        )
        data["msclap_text"] = self._load_pre_extracted_feats(
            feats_name="msclap_text", dataset_name=dataset_name, file_name=file_name
        )
        data["laionclap_audio"] = self._load_pre_extracted_feats(
            feats_name="laionclap_audio", dataset_name=dataset_name, file_name=file_name
        )
        data["laionclap_text"] = self._load_pre_extracted_feats(
            feats_name="laionclap_text", dataset_name=dataset_name, file_name=file_name
        )
        return data


if __name__ == "__main__":
    dataset = TTADataset(data_dir="data", split="train")
    print(f"len(dataset)): {len(dataset)}")
    data = dataset[0]
    print(f"data.keys(): {data.keys()}")
    print(f"data['audio'].shape: {data['audio'].shape}")
    print(f"data['audio_feats'].shape: {data['audio_feats'].shape}")
    print(f"data['text_feats'].shape: {data['text_feats'].shape}")
