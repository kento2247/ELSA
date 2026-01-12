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
        dataset_names: list[
            Literal["relate", "audiocap", "musiccap", "xacle", "aishell7b"]
        ] = [
            "relate",
            "audiocap",
            "musiccap",
            "aishell7b",
        ],
        split: Literal["train", "val", "test"] = "train",
        bitrate: int = 16000,
        max_len: int = 16000 * 10,
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
            if "aishell7b" in dataset_names:
                self._load_aishell7b_data(split, subjective_metric)

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
            audio_type = row["audio type"]

            if audio_type == "natural":
                continue

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

            if model == "real":
                continue

            self.database.append(
                {
                    "dataset": "audiocap",
                    "text_id": text_id,
                    "audio_file_path": os.path.join(
                        self.data_dir, "human_eval", "audio", model, f"{file_name}.wav"
                    ),
                    "ref_audio_file_path": os.path.join(
                        self.data_dir, "human_eval", "audio", "real", f"{file_name}.wav"
                    ),
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

            if model == "real":
                continue

            self.database.append(
                {
                    "dataset": "musiccap",
                    "text_id": text_id,
                    "audio_file_path": os.path.join(
                        self.data_dir, "human_eval", "music", model, f"{file_name}.wav"
                    ),
                    "ref_audio_file_path": os.path.join(
                        self.data_dir, "human_eval", "music", "real", f"{file_name}.wav"
                    ),
                    "text": text,
                    "score": score,
                }
            )

    def _load_xacle_data(self, split: str, subjective_metric: str) -> None:
        """Load XACLE dataset as test sets."""
        max_score = 10.0
        if subjective_metric != "REL":
            # XACLE dataset only supports REL subjective metric
            return

        split_name = split if split != "val" else "validation"
        if split != "test":
            dataset_dir = "XACLE_dataset"
            filename = f"{split_name}_average.csv"
        else:
            dataset_dir = "XACLE_test_data"
            filename = "test_with_score.csv"
        xacle_data_path = os.path.join(
            self.data_dir, dataset_dir, "meta_data", filename
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
            if split == "test":
                audio_file_path = os.path.join(
                    self.data_dir, dataset_dir, "wav", f"{wavname}"
                )
            else:
                audio_file_path = os.path.join(
                    self.data_dir, dataset_dir, "wav", split_name, f"{wavname}"
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

    def _load_aishell7b_data(self, split: str, subjective_metric: str) -> None:
        """Load AISHELL-7B (MusicEval-full) dataset."""
        max_score = 5.0

        # Load prompt info (text prompts)
        prompt_info_path = os.path.join(
            self.data_dir, "MusicEval-full", "prompt_info.txt"
        )
        prompt_df = pd.read_csv(prompt_info_path, sep="\t")
        prompt_dict = dict(zip(prompt_df["id"], prompt_df["text"]))

        # Map split name: val -> dev
        split_name = "dev" if split == "val" else split
        mos_list_path = os.path.join(
            self.data_dir, "MusicEval-full", "sets", f"{split_name}_mos_list.txt"
        )
        mos_data = pd.read_csv(
            mos_list_path, header=None, names=["filename", "ovl", "rel"]
        )

        for index, row in tqdm(
            mos_data.iterrows(),
            total=len(mos_data),
            desc=f"Loading AISHELL-7B {split} {subjective_metric} data",
        ):
            text_id: str = f"{split}_{subjective_metric}_{index}"
            filename: str = row["filename"]

            # Extract prompt ID from filename (e.g., audiomos2025-track1-S032_P092.wav -> P092)
            prompt_id = filename.split("_")[-1].replace(".wav", "")
            text: str = prompt_dict.get(prompt_id, "")

            # Select score based on subjective metric
            if subjective_metric == "OVL":
                score = float(row["ovl"]) / max_score
            elif subjective_metric == "REL":
                score = float(row["rel"]) / max_score
            else:
                continue

            audio_file_path = os.path.join(
                self.data_dir, "MusicEval-full", "wav", filename
            )
            ref_audio_file_path = ""

            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Wav file not found: {audio_file_path}")

            self.database.append(
                {
                    "dataset": "aishell7b",
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
        """Load raw audio waveforms for SI-SDR computation."""
        data["audio"] = self._load_wav(data["audio_file_path"])

        # Load reference audio if available
        if data["ref_audio_file_path"] and os.path.exists(data["ref_audio_file_path"]):
            data["ref_audio"] = self._load_wav(data["ref_audio_file_path"])
            data["has_ref_audio"] = True
        else:
            # Placeholder tensor for batching (will be skipped in evaluation)
            data["ref_audio"] = torch.zeros_like(data["audio"])
            data["has_ref_audio"] = False

        # data["audio"] = self._load_wav(data["audio_file_path"])
        data["msclap_audio"] = self._load_pre_extracted_feats(
            feats_name="msclap_audio",
            dataset_name=dataset_name,
            file_name=audio_file_name,
        )
        data["msclap_text"] = self._load_pre_extracted_feats(
            feats_name="msclap_text",
            dataset_name=dataset_name,
            file_name=text_file_name,
        )
        data["laionclap_audio"] = self._load_pre_extracted_feats(
            feats_name="laionclap_audio",
            dataset_name=dataset_name,
            file_name=audio_file_name,
        )
        data["laionclap_text"] = self._load_pre_extracted_feats(
            feats_name="laionclap_text",
            dataset_name=dataset_name,
            file_name=text_file_name,
        )
        data["humanclap_audio"] = self._load_pre_extracted_feats(
            feats_name="humanclap_audio",
            dataset_name=dataset_name,
            file_name=audio_file_name,
        )
        data["humanclap_text"] = self._load_pre_extracted_feats(
            feats_name="humanclap_text",
            dataset_name=dataset_name,
            file_name=text_file_name,
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
