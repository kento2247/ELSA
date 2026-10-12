# ruff: noqa: E402

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
        )  # [1]
        return score.squeeze(0)  # []


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Evaluate an audio-text pair with ELSA.")
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
    args = parser.parse_args()

    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Fine-grained audio-text alignment[/bold cyan]",
            title="[bold]ELSA[/bold]",
            border_style="cyan",
        )
    )

    inputs = Table.grid(padding=(0, 2))
    inputs.add_column(style="bold cyan", no_wrap=True)
    inputs.add_column()
    inputs.add_row("Audio", Text(args.audio_file_path))
    inputs.add_row("Text", Text(args.text))
    console.print(Panel(inputs, title="Input", border_style="dim"))

    with console.status("[bold cyan]Loading models...", spinner="dots"):
        model = OneShotTTAEvalModel()

    inference_start = time.perf_counter()
    with console.status("[bold cyan]Evaluating alignment...", spinner="dots"):
        score = model(
            audio_file_path=args.audio_file_path,
            text=args.text,
        )
    inference_seconds = time.perf_counter() - inference_start

    result = Table.grid(padding=(0, 2))
    result.add_column(style="bold", no_wrap=True)
    result.add_column(justify="right")
    result.add_row(
        "ELSA score",
        Text(f"{score.item():.4f}", style="bold bright_green"),
    )
    result.add_row("Inference", f"{inference_seconds:.2f} s")
    console.print(
        Panel(
            result,
            title="[bold green]Evaluation complete[/bold green]",
            border_style="green",
        )
    )
