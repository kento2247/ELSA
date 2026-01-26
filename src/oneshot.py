from typing import Literal

import torch

from model import TTAEvalModel
from preprocess import GPTTextParser, HumanCLAPEmbedder, SamAudio


class OneShotTTAEvalModel(TTAEvalModel):
    def __init__(self):
        super().__init__()
        self.embedder = HumanCLAPEmbedder()
        self.text_parser = GPTTextParser()
        self.audio_parser = SamAudio()
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
    model = OneShotTTAEvalModel()
    import time

    start_time = time.time()
    score = model(
        audio_file_path="data/wav/tango/train/23.wav",
        text="A dog barking and a car honking.",
        metric="REL",
    )
    print("Predicted score:", score.item())
    end_time = time.time()
    print("Inference time:", (end_time - start_time) * 1000, "msec")
