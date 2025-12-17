import argparse
import json
import os
import time

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TTADataset
from model import AudioTextSimilarityModel
from utils.eval_methods import (
    kendall_tau,
    mse,
    pearson_correlation,
    spearman_correlation,
)


class TTAEval:
    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        batch_size: int,
        lr: float,
        epochs: int,
        log_wandb: bool,
        save_qualitative: bool,
        subjective_metrics: list[str],
        test_dataset_names: list,
        eval_freq: int,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.log_wandb = log_wandb
        self.save_qualitative = save_qualitative
        self.subjective_metrics = subjective_metrics
        self.test_dataset_names = test_dataset_names
        self.eval_freq = eval_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AudioTextSimilarityModel().to(self.device)

        timestamp: str = time.strftime("%Y%m%d-%H%M%S")
        git_branch: str = os.popen("git rev-parse --abbrev-ref HEAD").read().strip()
        git_commit: str = os.popen("git rev-parse HEAD").read().strip()
        run_command: str = "uv run python ".join(os.sys.argv)
        self.meta_data = {
            "timestamp": timestamp,
            "git_branch": git_branch,
            "git_commit": git_commit,
            "run_command": run_command,
        }
        if self.log_wandb:
            wandb.init(project="TTAEval")
            self.meta_data["wandb_run_id"] = wandb.run.id
            self.meta_data["wandb_url"] = wandb.run.url

    def train(self):
        """Train the model with periodic evaluation on val and test sets."""
        train_dataset = TTADataset(
            data_dir=self.data_dir,
            split="train",
            dataset_names=["relate"],
            subjective_metrics=["REL"],
        )
        val_dataset = TTADataset(
            data_dir=self.data_dir,
            split="val",
            dataset_names=["relate"],
            subjective_metrics=["REL"],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )
        del train_dataset
        del val_dataset

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

        best_val_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch, train_loader)
            print(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f}")

            if self.log_wandb:
                wandb.log({"epoch": epoch, "train_loss": train_loss})

            if epoch % self.eval_freq == 0:
                val_loss = self.evaluate(val_loader)
                print(f"Epoch {epoch}/{self.epochs} - Val Loss: {val_loss:.4f}")

                if self.log_wandb:
                    wandb.log({"epoch": epoch, "val_loss": val_loss})

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model("best_model.pt")
                    print(f"Best model saved with val_loss: {val_loss:.4f}")

                print(f"Running test at epoch {epoch}...")
                self.test()

        self.save_model("final_model.pt")
        print("Training completed. Final model saved.")

    def train_epoch(self, epoch: int, train_loader: DataLoader) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            laionclap_audio: torch.Tensor = batch["laionclap_audio"].to(self.device)
            laionclap_text: torch.Tensor = batch["laionclap_text"].to(self.device)
            scores: torch.Tensor = batch["score"].float().to(self.device)

            self.optimizer.zero_grad()
            preds: torch.Tensor = self.model(laionclap_audio, laionclap_text)
            preds = preds.squeeze(-1)

            loss = self.criterion(preds, scores)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate the model on validation set and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                laionclap_audio: torch.Tensor = batch["laionclap_audio"].to(self.device)
                laionclap_text: torch.Tensor = batch["laionclap_text"].to(self.device)
                scores: torch.Tensor = batch["score"].float().to(self.device)

                preds: torch.Tensor = self.model(laionclap_audio, laionclap_text)
                preds = preds.squeeze(-1)

                loss = self.criterion(preds, scores)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def test(self) -> dict:
        """Test the model and log metrics in npy format."""
        metrics: dict = {}
        lb_header_text = ""
        lb_score_text = ""

        for subjective_metric in self.subjective_metrics:
            for test_dataset_name in self.test_dataset_names:
                test_dataset = TTADataset(
                    data_dir=self.data_dir,
                    split="test",
                    dataset_names=[test_dataset_name],
                    subjective_metrics=[subjective_metric],
                )
                if len(test_dataset) == 0:
                    print(
                        f"Skipping {test_dataset_name} for {subjective_metric} as no data found."
                    )
                    continue

                print(
                    f"Testing on {test_dataset_name} dataset, subjective metrics: {subjective_metric}"
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=4,
                )

                self.model.eval()
                all_preds: list[np.ndarray] = []
                all_scores: list[np.ndarray] = []

                with torch.no_grad():
                    for batch in tqdm(test_loader, desc="Testing"):
                        msclap_audio: torch.Tensor = batch["msclap_audio"].to(
                            self.device
                        )
                        msclap_text: torch.Tensor = batch["msclap_text"].to(self.device)
                        laionclap_audio: torch.Tensor = batch["laionclap_audio"].to(
                            self.device
                        )
                        laionclap_text: torch.Tensor = batch["laionclap_text"].to(
                            self.device
                        )
                        scores: np.ndarray = batch["score"].numpy()

                        preds: torch.Tensor = self.model(msclap_audio, msclap_text)
                        # preds: torch.Tensor = self.model(
                        #     laionclap_audio, laionclap_text
                        # )
                        preds: np.ndarray = preds.squeeze(-1).cpu().numpy()

                        all_preds.append(preds)
                        all_scores.append(scores)

                all_preds = np.concatenate(all_preds)
                all_scores = np.concatenate(all_scores)

                # Calculate metrics
                mse_val: float = mse(all_scores, all_preds)
                pearson_val: float = pearson_correlation(all_scores, all_preds)
                spearman_val: float = spearman_correlation(all_scores, all_preds)
                kendall_val: float = kendall_tau(all_scores, all_preds)

                if subjective_metric not in metrics:
                    metrics[subjective_metric] = {}
                metrics[subjective_metric][test_dataset_name] = {
                    "mse": mse_val,
                    "pearson": pearson_val,
                    "spearman": spearman_val,
                    "kendall_tau": kendall_val,
                }

                for metric_name in ["mse", "pearson", "spearman", "kendall_tau"]:
                    lb_header_text += (
                        f"{subjective_metric}.{test_dataset_name}.{metric_name}, "
                    )
                    lb_score_text += f"{metrics[subjective_metric][test_dataset_name][metric_name]:.4f}, "

        # Log to wandb if enabled
        if self.log_wandb:
            wandb.log(metrics)

        # lb_text for pasteing to leaderboard
        lb_text = lb_header_text.rstrip(", ") + "\n" + lb_score_text.rstrip(", ")
        print(f"Leaderboard Text: {lb_text}")

        # Save as json
        if self.save_qualitative:
            os.makedirs(self.model_dir, exist_ok=True)
            qualitative_data = {
                "metrics": metrics,
                "predictions": all_preds.tolist(),
                "scores": all_scores.tolist(),
                "meta_data": self.meta_data,
                "lb_text": lb_text,
            }
            qualitative_path = os.path.join(self.model_dir, "qualitative_results.json")
            json.dump(qualitative_data, open(qualitative_path, "w"), indent=1)
            print(f"Qualitative results saved to {qualitative_path}")

        return metrics

    def save_model(self, filename: str = "model.pt"):
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, filename)
        torch.save(self.model.state_dict(), path)

    def load_model(self, filename: str = "model.pt"):
        path = os.path.join(self.model_dir, filename)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)


def arg_parser():
    parser = argparse.ArgumentParser(description="Audio Captioning Evaluation")
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Mode: train or test",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--eval_freq", type=int, default=5, help="Evaluation frequency (in epochs)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save/load the model",
    )
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="Whether to log training with Weights & Biases",
    )
    parser.add_argument(
        "--save_qualitative",
        action="store_true",
        help="Whether to save qualitative results as csv during testing",
    )
    parser.add_argument(
        "--subjective_metrics",
        type=str,
        nargs="+",
        default=["REL", "OVL"],
        choices=["REL", "OVL"],
        help="Subjective metric to use from the dataset",
    )
    parser.add_argument(
        "--test_dataset_names",
        type=str,
        nargs="+",
        default=["relate", "audiocap", "musiccap"],
        help="List of dataset names to test on",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()

    evaluator = TTAEval(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        batch_size=args.bs,
        lr=args.lr,
        epochs=args.epochs,
        log_wandb=args.log_wandb,
        save_qualitative=args.save_qualitative,
        subjective_metrics=args.subjective_metrics,
        test_dataset_names=args.test_dataset_names,
        eval_freq=args.eval_freq,
    )

    if args.mode == "train":
        evaluator.train()
        evaluator.test()
    elif args.mode == "test":
        # evaluator.load_model("best_model.pt")
        evaluator.test()
