import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AudioCapDataset


class ScorePredictor(nn.Module):
    """Simple MLP to predict score from CLAP embeddings."""

    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, text_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([text_emb, audio_emb], dim=-1)
        return self.model(x).squeeze(-1)


class AudioCapEval:
    def __init__(self, data_dir: str, lr: float = 1e-4, epochs: int = 10):
        self.data_dir = data_dir
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ScorePredictor().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.train_dataset = AudioCapDataset(data_dir, split="train")
        self.val_dataset = AudioCapDataset(data_dir, split="val")
        self.test_dataset = AudioCapDataset(data_dir, split="test")

        self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=16, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)

    def train(self):
        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for batch in tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}"
            ):
                text_emb = batch["text_embedding"].to(self.device)
                audio_emb = batch["audio_embedding"].to(self.device)
                score = batch["score"].to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(text_emb, audio_emb)
                loss = self.criterion(pred, score)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            val_loss = self.evaluate()

            print(
                f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model.pt")

    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                text_emb = batch["text_embedding"].to(self.device)
                audio_emb = batch["audio_embedding"].to(self.device)
                score = batch["score"].to(self.device)

                pred = self.model(text_emb, audio_emb)
                loss = self.criterion(pred, score)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def test(self) -> float:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_scores = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                text_emb = batch["text_embedding"].to(self.device)
                audio_emb = batch["audio_embedding"].to(self.device)
                score = batch["score"].to(self.device)

                pred = self.model(text_emb, audio_emb)
                loss = self.criterion(pred, score)
                total_loss += loss.item()

                all_preds.extend(pred.cpu().tolist())
                all_scores.extend(score.cpu().tolist())

        test_loss = total_loss / len(self.test_loader)
        print(f"Test Loss (MSE): {test_loss:.4f}")
        return test_loss

    def save_model(self, filename: str = "model.pt"):
        save_path = os.path.join(self.data_dir, filename)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, filename: str = "model.pt"):
        load_path = os.path.join(self.data_dir, filename)
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        print(f"Model loaded from {load_path}")


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

    evaluator = AudioCapEval(
        data_dir=args.data_dir,
        lr=args.lr,
        epochs=args.epochs,
    )

    if args.mode == "train":
        evaluator.train()
        evaluator.test()
    else:
        evaluator.load_model("best_model.pt")
        evaluator.test()
