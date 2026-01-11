import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from dataset import TTADataset
from model import TTAEvalModel
from utils.eval_methods import (
    kendall_tau,
    mse,
    pearson_correlation,
    spearman_correlation,
)
from utils.helper_func import fix_seed
from utils.lb_output import format_leaderboard_text


class TTAEval:
    def __init__(
        self,
        # paths
        data_dir: str,
        model_dir: str,
        # training params
        batch_size: int,
        lr: float,
        epochs: int,
        eval_freq: int,
        main_metric: str,
        # evaluation params
        subjective_metrics: list[str],
        test_dataset_names: list[str],
        # logging
        log_wandb: bool,
        save_qualitative: bool,
    ):
        """Initialize TTAEval with training and evaluation parameters."""
        # paths
        self.data_dir = data_dir
        self.model_dir = model_dir
        # training params
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.eval_freq = eval_freq
        self.main_metric = main_metric
        # evaluation params
        self.subjective_metrics = subjective_metrics
        self.test_dataset_names = test_dataset_names
        # logging
        self.log_wandb = log_wandb
        self.save_qualitative = save_qualitative

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TTAEvalModel().to(self.device)

        self.meta_data = {
            "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "git_branch": os.popen("git rev-parse --abbrev-ref HEAD").read().strip(),
            "git_commit": os.popen("git rev-parse HEAD").read().strip(),
            "run_command": "uv run " + " ".join(os.sys.argv),
            "wandb_url": None,
            "best_epoch": None,
        }
        if self.log_wandb:
            wandb.init(project="TTAEval")
            self.meta_data["wandb_url"] = wandb.run.url

    def train(self):
        """Train the model with periodic evaluation on val and test sets."""
        train_dataset = TTADataset(data_dir=self.data_dir, split="train")
        val_dataset = TTADataset(data_dir=self.data_dir, split="val")
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
        del train_dataset, val_dataset

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

        best_epoch = -1
        best_val_metric = float("-inf")
        best_test_metrics = None

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(epoch, train_loader)
            print(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f}")

            if self.log_wandb:
                wandb.log({"epoch": epoch, "train_loss": train_loss})

            if (epoch - 1) % self.eval_freq == 0:
                val_metrics = self.evaluate(val_loader, desc="Validation")
                print(f"Epoch {epoch}/{self.epochs} - Val Metrics: {val_metrics}")

                if self.log_wandb:
                    wandb.log({"epoch": epoch, "val": val_metrics})

                print(f"Running test at epoch {epoch}...")
                test_metrics = self.test()

                val_metric = val_metrics[self.main_metric]
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_epoch = epoch
                    self.save_model("best_model.pt")
                    print(
                        f"Best model saved with val_{self.main_metric}: {val_metric:.4f}"
                    )
                    best_test_metrics = test_metrics

        print(f"Training completed. Best epoch: {best_epoch}")
        self.meta_data["best_epoch"] = best_epoch
        lb_text = format_leaderboard_text(self.meta_data, best_test_metrics)
        print(f"Best Leaderboard Text:\n{lb_text}")
        return best_test_metrics

    def _train_epoch(self, epoch: int, train_loader: DataLoader) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            laionclap_audio = batch["laionclap_audio"].to(self.device)
            laionclap_text = batch["laionclap_text"].to(self.device)
            scores = batch["score"].float().to(self.device)
            metric_ids = batch["subjective_metric_id"].to(self.device)  # [B]

            self.optimizer.zero_grad()
            preds = self.model(laionclap_audio, laionclap_text, metric_ids).squeeze(-1)  # [B]

            loss = self.criterion(preds, scores)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, data_loader: DataLoader, desc: str = "Evaluating") -> dict:
        """Evaluate the model and return metrics dict."""
        self.model.eval()
        all_preds: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                laionclap_audio = batch["laionclap_audio"].to(self.device)
                laionclap_text = batch["laionclap_text"].to(self.device)
                scores = batch["score"].numpy()
                metric_ids = batch["subjective_metric_id"].to(self.device)  # [B]

                preds = self.model(laionclap_audio, laionclap_text, metric_ids)  # [B, 1]
                preds = preds.squeeze(-1).cpu().numpy()  # [B]

                all_preds.append(preds)
                all_scores.append(scores)

        all_preds = np.concatenate(all_preds)
        all_scores = np.concatenate(all_scores)

        return {
            "mse": mse(all_scores, all_preds),
            "pearson": pearson_correlation(all_scores, all_preds),
            "spearman": spearman_correlation(all_scores, all_preds),
            "kendall_tau": kendall_tau(all_scores, all_preds),
        }

    def test(self) -> dict:
        """Test the model on test datasets and log metrics."""
        metrics: dict = {}

        for subjective_metric in self.subjective_metrics:
            for test_dataset_name in self.test_dataset_names:
                test_dataset = TTADataset(
                    data_dir=self.data_dir,
                    split="test",
                    dataset_names=[test_dataset_name],
                    subjective_metrics=[subjective_metric],
                )
                if len(test_dataset) == 0:
                    continue

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=4,
                )
                del test_dataset

                desc = f"Testing {subjective_metric}/{test_dataset_name}"
                eval_metrics = self.evaluate(test_loader, desc=desc)

                if subjective_metric not in metrics:
                    metrics[subjective_metric] = {}
                metrics[subjective_metric][test_dataset_name] = eval_metrics

        if self.log_wandb:
            wandb.log(metrics)

        if self.save_qualitative:
            os.makedirs(self.model_dir, exist_ok=True)
            qualitative_data = {
                "metrics": metrics,
                "meta_data": self.meta_data,
            }
            qualitative_path = os.path.join(self.model_dir, "qualitative_results.json")
            with open(qualitative_path, "w") as f:
                json.dump(qualitative_data, f, indent=1)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Audio Captioning Evaluation")
    # mode
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "test"],
        help="Mode: train or test",
    )
    # paths
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
    # training params
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument(
        "--eval_freq", type=int, default=3, help="Evaluation frequency (in epochs)"
    )
    parser.add_argument(
        "--main_metric",
        type=str,
        default="kendall_tau",
        choices=["mse", "pearson", "spearman", "kendall_tau"],
        help="Main metric for model selection",
    )
    # evaluation params
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
        default=["relate", "audiocap", "musiccap", "aishell7b"],
        help="List of dataset names to test on",
    )
    # logging
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        default=True,
        help="Whether to log training with Weights & Biases",
    )
    parser.add_argument(
        "--save_qualitative",
        action="store_true",
        help="Whether to save qualitative results as json during testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generator",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fix_seed(args.seed)
    evaluator = TTAEval(
        # paths
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        # training params
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        eval_freq=args.eval_freq,
        main_metric=args.main_metric,
        # evaluation params
        subjective_metrics=args.subjective_metrics,
        test_dataset_names=args.test_dataset_names,
        # logging
        log_wandb=args.log_wandb,
        save_qualitative=args.save_qualitative,
    )

    if args.mode == "train":
        evaluator.train()
    elif args.mode == "test":
        evaluator.load_model("best_model.pt")
        evaluator.test()
