from typing import Literal

import torch

from model import TTAEvalModel
from preprocess import AudioSeparator  # Placeholder for unused import
from preprocess import CLAPEmbedder  # Placeholder for unused import
from preprocess import TextParser  # Placeholder for unused import
from preprocess import GPTTextParser, HumanCLAPEmbedder, SAMAudioSeparator


class OneShotTTAEvalModel(TTAEvalModel):
    def __init__(self):
        super().__init__()
        self.embedder: CLAPEmbedder = HumanCLAPEmbedder()
        self.text_parser: TextParser = GPTTextParser()
        self.audio_parser: AudioSeparator = SAMAudioSeparator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(
        self,
        audio_file_path: str,
        text: str,
        metric: Literal["REL", "OVL"] = "REL",
    ) -> torch.Tensor:
        audio_feature: torch.Tensor = self.embedder.embed_audios(
            [audio_file_path]
        )  # [1, D]
        text_feature: torch.Tensor = self.embedder.embed_texts([text])  # [1, D]

        acoustic_events: list[str] = self.text_parser.parse_texts([text])[0]
        separated_audios: list[torch.Tensor] = self.audio_parser.separate_audio(
            audio_file_path, acoustic_events
        )
        num_events = len(acoustic_events)
        acoustic_events_features: torch.Tensor = self.embedder.embed_texts(
            acoustic_events
        )  # [num_events, D]
        separated_audios_features: torch.Tensor = self.embedder.embed_audios(
            separated_audios
        )  # [num_events, D]
        mask = torch.ones(num_events, dtype=torch.bool).unsqueeze(0)  # [1, num_events]

        score = super().forward(
            audio_feature,  # [1, D]
            text_feature,  # [1, D]
            separated_audios_features.unsqueeze(0),  # [1, num_events, D]
            acoustic_events_features.unsqueeze(0),  # [1, num_events, D]
            mask,  # [1, num_events]
            (torch.tensor(1.0 if metric == "OVL" else 0.0).to(self.device)),  # [1]
        )  # [1]
        return score.squeeze(0)  # []


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_file_path",
        type=str,
        default="data/wav/tango/train/23.wav",
        help="Path to the input audio file.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="A dog barking and a car honking.",
        help="Text description of the audio.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="REL",
        choices=["REL", "OVL"],
        help="Evaluation metric to use.",
    )
    args = parser.parse_args()

    model = OneShotTTAEvalModel()

    start_time = time.time()
    score = model(
        audio_file_path=args.audio_file_path,
        text=args.text,
        metric=args.metric,
    )
    print("Predicted score:", score.item())
    end_time = time.time()
    print("Inference time:", (end_time - start_time) * 1000, "msec")
