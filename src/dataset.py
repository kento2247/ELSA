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
        data_dir: str = "data",
        features_dir: str = "features",
        subjective_metrics: list[
            Literal[
                "REL",
                "OVL",
                "IS",
                "OS",
                "AttributeText",
                "AttributeAudio",
                "OrderText",
                "OrderAudio",
            ]
        ] = [
            "REL",
            "OVL",
            "IS",
            "OS",
        ],
        dataset_names: list[
            Literal[
                "relate",
                "relate_isos",
                "audiocap",
                "musiccap",
                "xacle",
                "aishell7b",
                "clotho",
            ]
        ] = [
            "relate",
            "relate_isos",
            "audiocap",
            "musiccap",
            "aishell7b",
            "clotho",
        ],
        split: Literal["train", "val", "test"] = "train",
        bitrate: int = 16000,
        max_len: int = 16000 * 10,
        dtype: torch.dtype = torch.float32,
        pre_load_features: bool = False,
        parsed_seq_size: int = 20,
        clap_variant: str = "humanclap",
        device: str = "cpu",
    ):
        """Initialize TTADataset with specified data directory and split."""
        self.data_dir = data_dir
        self.features_dir = features_dir
        self.bitrate = bitrate
        self.max_len = max_len
        self.dtype = dtype
        self.pre_load_features = pre_load_features
        self.parsed_seq_size = parsed_seq_size
        self.subjective_metrics = subjective_metrics
        self.clap_variant = clap_variant
        self.device = device
        self.database = []

        # Load datasets
        for subjective_metric in subjective_metrics:
            if "relate" in dataset_names:
                self._load_relate_data(split, subjective_metric)
            if "relate_isos" in dataset_names:
                self._load_relate_isos_data(split, subjective_metric)
            if "audiocap" in dataset_names:
                self._load_audiocap_data(split, subjective_metric)
            if "musiccap" in dataset_names:
                self._load_musiccap_data(split, subjective_metric)
            if "xacle" in dataset_names:
                self._load_xacle_data(split, subjective_metric)
            if "aishell7b" in dataset_names:
                self._load_aishell7b_data(split, subjective_metric)
            if "clotho" in dataset_names:
                self._load_clotho_data(split, subjective_metric)
            if "compa" in dataset_names:
                self._load_compa_data(split, subjective_metric)

    def _load_relate_data(self, split: str, subjective_metric: str) -> None:
        """Load RELATE dataset and split into train, val, test sets."""
        max_score = 10.0
        if subjective_metric != "REL":
            return

        data_path = os.path.join(self.data_dir, "RELATE", "scores", "REL.csv")
        data = pd.read_csv(data_path)
        if split == "train" or split == "test":
            data = data[data["in RELATE dataset"] == split]
        elif split == "val":
            data = data[data["in RELATE dataset"] == "validation"]

        # Aggregate scores by wavname
        data = (
            data.groupby("wavname")
            .agg(
                {
                    "text": "first",
                    "score": "mean",
                    "audio type": "first",
                }
            )
            .reset_index()
        )

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
                    "subjective_metric_id": 0 if subjective_metric == "REL" else 1,
                    "rev_text": "",  # Not used
                    "rev_audio": "",  # Not used
                }
            )

    def _load_relate_isos_data(self, split: str, subjective_metric: str) -> None:
        """Load RELATE dataset and split into train, val, test sets."""
        max_score = 10.0
        if subjective_metric == "IS":
            data_path = os.path.join(self.data_dir, "RELATE", "scores", "IS.csv")
        elif subjective_metric == "OS":
            data_path = os.path.join(self.data_dir, "RELATE", "scores", "OS.csv")
        else:
            return
        data = pd.read_csv(data_path)
        anchors = data[data["anchor label"]]
        avg = anchors.groupby("listener_id")["score"].mean()

        if split == "train":
            data = data[
                (data["in AudioCaps"] == "train")
                & (data["listener_id"].isin(avg[avg < 2].index))
            ]
        elif split == "test":
            data = data[
                (data["in AudioCaps"] == "test")
                & (data["listener_id"].isin(avg[avg < 1].index))
            ]
        elif split == "val":
            data = data[
                (data["in AudioCaps"] == "validation")
                & (data["listener_id"].isin(avg[avg < 1].index))
            ]

        # Aggregate scores by wavname
        data = (
            data.groupby("wavname")
            .agg(
                {
                    "text": "first",
                    "score": "mean",
                    "audio type": "first",
                }
            )
            .reset_index()
        )

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
                    "dataset": "relate_isos",
                    "text_id": text_id,
                    "audio_file_path": audio_file_path,
                    "ref_audio_file_path": ref_audio_file_path,
                    "text": text,
                    "score": score,
                    "subjective_metric_id": 0 if subjective_metric == "REL" else 1,
                    "rev_text": "",  # Not used
                    "rev_audio": "",  # Not used
                }
            )

    def _load_audiocap_data(self, split: str, subjective_metric: str) -> None:
        """Load AudioCap dataset as test sets."""
        max_score = 5.0
        if split != "test":
            return
        if subjective_metric not in ["REL", "OVL"]:
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
                    "subjective_metric_id": 0 if subjective_metric == "REL" else 1,
                    "rev_text": "",  # Not used
                    "rev_audio": "",  # Not used
                }
            )

    def _load_musiccap_data(self, split: str, subjective_metric: str) -> None:
        """Load MusicCap music dataset as test sets."""
        max_score = 5.0
        if split != "test":
            return
        if subjective_metric not in ["REL", "OVL"]:
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
                    "subjective_metric_id": 0 if subjective_metric == "REL" else 1,
                    "rev_text": "",  # Not used
                    "rev_audio": "",  # Not used
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
                    "subjective_metric_id": 0 if subjective_metric == "REL" else 1,
                    "rev_text": "",  # Not used
                    "rev_audio": "",  # Not used
                }
            )

    def _load_clotho_data(self, split: str, subjective_metric: str) -> None:
        """Load Clotho dataset as test sets."""
        max_score = 5.0
        if split != "test":
            return
        if subjective_metric not in ["REL", "OVL"]:
            return
        clotho_data_path = os.path.join(
            self.data_dir, "clotho", "clotho_ovl_rel_test_set.csv"
        )
        clotho_data = pd.read_csv(clotho_data_path)

        for index, row in tqdm(
            clotho_data.iterrows(),
            total=len(clotho_data),
            desc=f"Loading Clotho {split} {subjective_metric} data",
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
                    "dataset": "clotho",
                    "text_id": text_id,
                    "audio_file_path": os.path.join(
                        self.data_dir, "clotho", "wave_all_16k", model, file_name
                    ),
                    "ref_audio_file_path": os.path.join(
                        self.data_dir, "clotho", "wave_all_16k", "real", file_name
                    ),
                    "text": text,
                    "score": score,
                    "subjective_metric_id": 0 if subjective_metric == "REL" else 1,
                    "rev_text": "",  # Not used
                    "rev_audio": "",  # Not used
                }
            )

    def _load_aishell7b_data(self, split: str, subjective_metric: str) -> None:
        """Load AISHELL-7B (MusicEval-full) dataset."""
        max_score = 5.0
        if subjective_metric not in ["REL", "OVL"]:
            return
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
                    "subjective_metric_id": 0 if subjective_metric == "REL" else 1,
                    "rev_text": "",  # Not used
                    "rev_audio": "",  # Not used
                }
            )

    def _load_compa_data(self, split: str, subjective_metric: str) -> None:
        """Load CompA dataset for multiple choice tasks.

        CompA has two sub-tasks:
        - Attribute: AttributeText (audio -> text) and AttributeAudio (text -> audio)
        - Order: OrderText (audio -> text) and OrderAudio (text -> audio)

        Each instance has 2 audio files and 2 text captions (or 3 for Order triplet).
        The task is to select the correct match from the choices.
        """
        if split != "test":
            return  # CompA is only used for testing
        if subjective_metric not in [
            "AttributeText",
            "AttributeAudio",
            "OrderText",
            "OrderAudio",
        ]:
            return
        # Load Attribute dataset
        if subjective_metric in ["AttributeText", "AttributeAudio"]:
            attribute_csv_path = os.path.join(
                self.data_dir, "CompA_attribute", "Compa-Attribute.csv"
            )

            attribute_data = pd.read_csv(attribute_csv_path)
            attribute_files_dir = os.path.join(
                self.data_dir, "CompA_attribute", "CompA_attribute_files"
            )

            for index, row in tqdm(
                attribute_data.iterrows(),
                total=len(attribute_data),
                desc=f"Loading CompA Attribute {subjective_metric} data",
            ):
                text_id: str = f"{split}_{subjective_metric}_{index}"

                # Get audio files
                audio_file_1 = os.path.join(attribute_files_dir, row["pair_file.1"])
                audio_file_2 = os.path.join(
                    attribute_files_dir, row["reversed_pair_file.1"]
                )

                # Get text captions
                text_1 = row["pair_caption"]
                text_2 = row["reversed_pair_caption"]

                data_item = {
                    "dataset": "compa",
                    "text_id": text_id,
                    "audio_file_path": audio_file_1,  # Correct audio
                    "ref_audio_file_path": "",  # Not used
                    "text": text_1,
                    "score": 0.0,  # Not used for classification
                    "subjective_metric_id": 0,  # Not used
                    "rev_text": text_2,  # Incorrect choice
                    "rev_audio": audio_file_2,  # Incorrect choice
                }
                self.database.append(data_item)

        # Load Order dataset
        if subjective_metric in ["OrderText", "OrderAudio"]:
            order_csv_path = os.path.join(
                self.data_dir, "CompA_order", "CompA_order_benchmark.csv"
            )

            order_data = pd.read_csv(order_csv_path)
            order_files_dir = os.path.join(
                self.data_dir, "CompA_order", "CompA_order_files"
            )

            for index, row in tqdm(
                order_data.iterrows(),
                total=len(order_data),
                desc=f"Loading CompA Order {subjective_metric} data",
            ):
                text_id: str = f"{split}_{subjective_metric}_{index}"

                # Get audio files
                audio_file_1 = os.path.join(order_files_dir, row["pair_file"])
                audio_file_2 = os.path.join(order_files_dir, row["reversed_pair_file"])

                # Get text captions
                text_1 = row["pair_caption"]
                text_2 = row["reversed_pair_caption"]

                data_item = {
                    "dataset": "compa",
                    "text_id": text_id,
                    "audio_file_path": audio_file_1,  # Correct audio
                    "ref_audio_file_path": "",  # Not used
                    "text": text_1,
                    "score": 0.0,  # Not used for classification
                    "subjective_metric_id": 0,  # Not used
                    "rev_text": text_2,  # Incorrect choice
                    "rev_audio": audio_file_2,  # Incorrect choice
                }
                self.database.append(data_item)

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

    def _load_pre_extracted_feats(
        self, feats_name: str, dataset_name: str, file_name: str, dim: int = None
    ) -> torch.Tensor:
        """Load pre-extracted features from the specified file path."""
        feats_dir = os.path.join(self.data_dir, self.features_dir, feats_name)
        feat_path = os.path.join(feats_dir, dataset_name, file_name)
        if not os.path.exists(feat_path):
            return torch.empty(0, dim)
        feats = torch.load(feat_path, map_location=self.device)
        return feats.to(self.dtype)

    def _load_pre_extracted_mask(
        self, feats_name: str, dataset_name: str, file_name: str
    ) -> torch.Tensor:
        """Load pre-extracted mask without dtype casting."""
        feats_dir = os.path.join(self.data_dir, self.features_dir, feats_name)
        feat_path = os.path.join(feats_dir, dataset_name, file_name)
        if not os.path.exists(feat_path):
            return torch.empty(0, dtype=torch.bool)
        return torch.load(feat_path, map_location=self.device)

    def _load_quality_prompts(self):
        feats_dir = os.path.join(
            self.data_dir, "features", f"{self.clap_variant}_quality_prompts"
        )
        high_path = os.path.join(feats_dir, "high.pt")
        low_path = os.path.join(feats_dir, "low.pt")
        unrelated_path = os.path.join(feats_dir, "unrelated.pt")

        if (
            os.path.exists(high_path)
            and os.path.exists(low_path)
            and os.path.exists(unrelated_path)
        ):
            self.high_emb = torch.load(high_path, map_location=self.device)
            self.low_emb = torch.load(low_path, map_location=self.device)
            self.unrelated_emb = torch.load(unrelated_path, map_location=self.device)
        else:
            raise FileNotFoundError(f"Quality prompts not found at {feats_dir}. ")

    def _load_compa_feats(self, data):
        """Load CompA choice features for multiple choice tasks."""
        dataset_name = "compa"
        rev_audio = data["rev_audio"]
        rev_audio_file_name = os.path.basename(rev_audio).replace(".wav", ".pt")
        # Rev Audio embed.
        data[f"{self.clap_variant}_audio_rev"] = self._load_pre_extracted_feats(
            feats_name=f"{self.clap_variant}_audio",
            dataset_name=dataset_name,
            file_name=rev_audio_file_name,
        )
        # Rev Text embed.
        data[f"{self.clap_variant}_text_rev"] = self._load_pre_extracted_feats(
            feats_name=f"{self.clap_variant}_text",
            dataset_name=dataset_name,
            file_name=f"{data['text_id']}_rev.pt",
        )
        # Rev Text -> Audio parsed embed.
        data[f"{self.clap_variant}_parsed_audio_rev"] = self._pad_or_truncate_feats(
            self._load_pre_extracted_feats(
                feats_name=f"{self.clap_variant}_parsed_audio",
                dataset_name=dataset_name,
                file_name=f"{data['text_id']}_rev.pt",
            )
        )
        data[f"{self.clap_variant}_parsed_text_rev"] = self._pad_or_truncate_feats(
            self._load_pre_extracted_feats(
                feats_name=f"{self.clap_variant}_parsed_text",
                dataset_name=dataset_name,
                file_name=f"{data['text_id']}_rev.pt",
            )
        )
        data[f"{self.clap_variant}_parsed_mask_rev"] = self._pad_or_truncate_mask(
            self._load_pre_extracted_mask(
                feats_name=f"{self.clap_variant}_parsed_mask",
                dataset_name=dataset_name,
                file_name=f"{data['text_id']}_rev.pt",
            )
        )
        # Text -> Rev Audio parsed embed.
        data[f"rev_{self.clap_variant}_parsed_audio"] = self._pad_or_truncate_feats(
            self._load_pre_extracted_feats(
                feats_name=f"{self.clap_variant}_parsed_audio",
                dataset_name=dataset_name,
                file_name=f"rev_{data['text_id']}.pt",
            )
        )
        data[f"rev_{self.clap_variant}_parsed_text"] = self._pad_or_truncate_feats(
            self._load_pre_extracted_feats(
                feats_name=f"{self.clap_variant}_parsed_text",
                dataset_name=dataset_name,
                file_name=f"rev_{data['text_id']}.pt",
            )
        )
        data[f"rev_{self.clap_variant}_parsed_mask"] = self._pad_or_truncate_mask(
            self._load_pre_extracted_mask(
                feats_name=f"{self.clap_variant}_parsed_mask",
                dataset_name=dataset_name,
                file_name=f"rev_{data['text_id']}.pt",
            )
        )
        # Rev Text -> Rev Audio parsed embed.
        data[f"rev_{self.clap_variant}_parsed_audio_rev"] = self._pad_or_truncate_feats(
            self._load_pre_extracted_feats(
                feats_name=f"{self.clap_variant}_parsed_audio",
                dataset_name=dataset_name,
                file_name=f"rev_{data['text_id']}_rev.pt",
            )
        )
        data[f"rev_{self.clap_variant}_parsed_text_rev"] = self._pad_or_truncate_feats(
            self._load_pre_extracted_feats(
                feats_name=f"{self.clap_variant}_parsed_text",
                dataset_name=dataset_name,
                file_name=f"rev_{data['text_id']}_rev.pt",
            )
        )
        data[f"rev_{self.clap_variant}_parsed_mask_rev"] = self._pad_or_truncate_mask(
            self._load_pre_extracted_mask(
                feats_name=f"{self.clap_variant}_parsed_mask",
                dataset_name=dataset_name,
                file_name=f"rev_{data['text_id']}_rev.pt",
            )
        )
        return data

    def _load_features(self, data):
        """Pre-load all features into memory to speed up data loading."""
        text_file_name = f"{data['text_id']}.pt"
        audio_file_name = os.path.basename(data["audio_file_path"]).replace(
            ".wav", ".pt"
        )
        dataset_name = data["dataset"]

        # Base features
        data[f"{self.clap_variant}_audio"] = self._load_pre_extracted_feats(
            feats_name=f"{self.clap_variant}_audio",
            dataset_name=dataset_name,
            file_name=audio_file_name,
        )
        data[f"{self.clap_variant}_text"] = self._load_pre_extracted_feats(
            feats_name=f"{self.clap_variant}_text",
            dataset_name=dataset_name,
            file_name=text_file_name,
        )

        return data

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        """Get item by index from the dataset."""
        return self._load_features(self.database[idx])


if __name__ == "__main__":
    dataset = TTADataset(data_dir="data", split="train")
    print(f"len(dataset)): {len(dataset)}")
    data = dataset.__getitem__(0)
    print(f"data.keys(): {data.keys()}")
