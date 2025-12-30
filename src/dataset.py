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
        subjective_metrics: list[Literal["REL", "OVL"]] = ["REL", "OVL"],
        dataset_names: list[Literal["relate", "audiocap", "musiccap", "xacle"]] = [
            "relate",
            "audiocap",
            "musiccap",
            "xacle",
        ],
        split: Literal["train", "val", "test"] = "train",
        bitrate: int = 16000,
        max_len: int = 160000 * 10,
        dtype: torch.dtype = torch.float32,
        pre_load_features: bool = False,
    ):
        """Initialize TTADataset with specified data directory and split."""
        self.data_dir = data_dir
        self.bitrate = bitrate
        self.max_len = max_len
        self.dtype = dtype
        self.pre_load_features = pre_load_features
        self.database = []
        for subjective_metric in subjective_metrics:
            if "relate" in dataset_names:
                self._load_relate_data(split, subjective_metric)
            if "audiocap" in dataset_names:
                self._load_audiocap_data(split, subjective_metric)
            if "musiccap" in dataset_names:
                self._load_musiccap_data(split, subjective_metric)
            if "xacle" in dataset_names:
                self._load_xacle_data(split, subjective_metric)

        if pre_load_features:
            for i in tqdm(
                range(len(self.database)),
                desc="Pre-loading pre-extracted features into memory",
            ):
                self.database[i] = self._load_features(self.database[i])

    def _load_relate_data(self, split: str, subjective_metric: str) -> None:
        """Load RELATE dataset and split into train, val, test sets."""
        max_score = 10.0
        if subjective_metric != "REL":
            # RELATE dataset only supports REL subjective metric
            return

        relate_rel_path = os.path.join(self.data_dir, "RELATE", "scores", "REL.csv")
        relate_rel_data = pd.read_csv(relate_rel_path)

        if split == "train" or split == "test":
            data = relate_rel_data[relate_rel_data["in RELATE dataset"] == split]
        elif split == "val":
            data = relate_rel_data[relate_rel_data["in RELATE dataset"] == "validation"]
        data = data.reset_index(drop=True)

        for index, row in tqdm(
            data.iterrows(),
            total=len(data),
            desc=f"Loading RELATE {split} {subjective_metric} data",
        ):
            text_id: str = f"{split}_{subjective_metric}_{index}"
            wavname: str = row["wavname"]
            text: str = row["text"]
            score: float = float(row["score"]) / max_score  # normalize to [0, 1]
            audio_file_path = os.path.join(self.data_dir, f"wav{wavname}")
            ref_audio_file_path = os.path.join(
                self.data_dir,
                "wav",
                "audiocaps",
                split if split != "val" else "test",
                f"{wavname.split('/')[-1]}",
            )
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Wav file not found: {audio_file_path}")
            if not os.path.exists(ref_audio_file_path):
                ref_audio_file_path = ""
            self.database.append(
                {
                    "dataset": "relate",
                    "text_id": text_id,
                    "audio_file_path": audio_file_path,
                    "ref_audio_file_path": ref_audio_file_path,
                    "text": text,
                    "score": score,
                }
            )

    def _load_audiocap_data(self, split: str, subjective_metric: str) -> None:
        """Load AudioCap dataset as test sets."""
        max_score = 5.0
        if split != "test":
            return
        audiocap_data_path = os.path.join(
            self.data_dir, "human_eval", "audio", "scores.csv"
        )
        audiocap_data = pd.read_csv(audiocap_data_path)

        for index, row in tqdm(
            audiocap_data.iterrows(),
            total=len(audiocap_data),
            desc=f"Loading AudioCap {split} {subjective_metric} data",
        ):
            text_id: str = f"{split}_{subjective_metric}_{index}"
            text: str = row["Text"]
            model: str = row["Model"]
            file_name: str = row["File Name"]
            score: float = (
                float(row[subjective_metric]) / max_score
            )  # normalize to [0, 1]
            self.database.append(
                {
                    "dataset": "audiocap",
                    "text_id": text_id,
                    "audio_file_path": os.path.join(
                        self.data_dir, "human_eval", "audio", model, f"{file_name}.wav"
                    ),
                    "ref_audio_file_path": "",
                    "text": text,
                    "score": score,
                }
            )

    def _load_musiccap_data(self, split: str, subjective_metric: str) -> None:
        """Load MusicCap music dataset as test sets."""
        max_score = 5.0
        if split != "test":
            return
        musiccap_data_path = os.path.join(
            self.data_dir, "human_eval", "music", "scores.csv"
        )
        musiccap_data = pd.read_csv(musiccap_data_path)

        for index, row in tqdm(
            musiccap_data.iterrows(),
            total=len(musiccap_data),
            desc=f"Loading MusicCap {split} {subjective_metric} data",
        ):
            text_id: str = f"{split}_{subjective_metric}_{index}"
            text: str = row["Text"]
            model: str = row["Model"]
            file_name: str = row["File Name"]
            score: float = (
                float(row[subjective_metric]) / max_score
            )  # normalize to [0, 1]
            self.database.append(
                {
                    "dataset": "musiccap",
                    "text_id": text_id,
                    "audio_file_path": os.path.join(
                        self.data_dir, "human_eval", "music", model, f"{file_name}.wav"
                    ),
                    "ref_audio_file_path": "",
                    "text": text,
                    "score": score,
                }
            )

    def _load_xacle_data(self, split: str, subjective_metric: str) -> None:
        """Load XACLE dataset as test sets."""
        max_score = 10.0
        if split != "test":
            return
        if split != "REL":
            return
        xacle_data_path = os.path.join(
            self.data_dir, "XACLE_test_data", "meta_data", "test_with_score.csv"
        )
        xacle_data = pd.read_csv(xacle_data_path)

        for index, row in tqdm(
            xacle_data.iterrows(),
            total=len(xacle_data),
            desc=f"Loading XACLE {split} {subjective_metric} data",
        ):
            text_id: str = f"{split}_{subjective_metric}_{index}"
            wavname: str = row["wav_file_name"]
            text: str = row["text"]
            score: float = (
                float(row["average_score"]) / max_score
            )  # normalize to [0, 1]
            audio_file_path = os.path.join(
                self.data_dir, "XACLE_test_data", "wav", f"{wavname}"
            )
            ref_audio_file_path = ""
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Wav file not found: {audio_file_path}")
            self.database.append(
                {
                    "dataset": "xacle",
                    "text_id": text_id,
                    "audio_file_path": audio_file_path,
                    "ref_audio_file_path": ref_audio_file_path,
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

    def _load_features(self, data):
        """Pre-load all features into memory to speed up data loading."""
        text_file_name = f"{data['text_id']}.pt"
        audio_file_name = os.path.basename(data["audio_file_path"]).replace(
            ".wav", ".pt"
        )
        dataset_name = data["dataset"]

        data["msclap_audio"] = self._load_pre_extracted_feats(
            feats_name="msclap_audio",
            dataset_name=dataset_name,
            file_name=audio_file_name,
        )
        data["pam_prompt_1"] = self._load_pre_extracted_feats(
            feats_name="pam_prompts",
            dataset_name="all",
            file_name="prompt_1.pt",
        )
        data["pam_prompt_2"] = self._load_pre_extracted_feats(
            feats_name="pam_prompts",
            dataset_name="all",
            file_name="prompt_2.pt",
        )
        return data

    def __getitem__(self, idx):
        """Get item by index from the dataset."""
        if self.pre_load_features:
            return self.database[idx]
        else:
            return self._load_features(self.database[idx])


if __name__ == "__main__":
    dataset = TTADataset(data_dir="data", split="train")
    print(f"len(dataset)): {len(dataset)}")
    data = dataset.__getitem__(0)
    print(f"data.keys(): {data.keys()}")
