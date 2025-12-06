import argparse
import json
import os
import time

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AudioCapDataset
from model import AudioTextSimilarityModel
from utils.eval_methods import (
    kendall_tau,
    mse,
    pearson_correlation,
    spearman_correlation,
)


class AudioCapEval:
    def __init__(
        self,
        data_dir: str,
        model_dir: str = "models",
        batch_size: int = 32,
        lr: float = 1e-4,
        epochs: int = 10,
        log_wandb: bool = False,
        save_qualitative: bool = False,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.log_wandb = log_wandb
        self.save_qualitative = save_qualitative

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
            wandb.init(project="AudioCapEval")
            self.meta_data["wandb_run_id"] = wandb.run.id
            self.meta_data["wandb_url"] = wandb.run.url

    def train(self): ...

    def evaluate(self) -> float: ...

    def test(self) -> dict:
        """Test the model and log metrics in npy format."""
        test_dataset_names = ["relate", "pam_audio", "pam_music"]
        metrics: dict = {}

        for test_dataset_name in test_dataset_names:
            print(f"Testing on {test_dataset_name} dataset")
            test_dataset = AudioCapDataset(
                data_dir=self.data_dir, split="test", dataset_name=[test_dataset_name]
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
                    msclap_audio: torch.Tensor = batch["msclap_audio"].to(self.device)
                    msclap_text: torch.Tensor = batch["msclap_text"].to(self.device)
                    laionclap_audio: torch.Tensor = batch["laionclap_audio"].to(
                        self.device
                    )
                    laionclap_text: torch.Tensor = batch["laionclap_text"].to(
                        self.device
                    )
                    scores: np.ndarray = batch["score"].numpy()

                    preds: torch.Tensor = self.model(msclap_audio, msclap_text)
                    # preds: torch.Tensor = self.model(laionclap_audio, laionclap_text)
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

            metrics[test_dataset_name] = {
                "mse": mse_val,
                "pearson": pearson_val,
                "spearman": spearman_val,
                "kendall_tau": kendall_val,
            }

        # Log to wandb if enabled
        if self.log_wandb:
            wandb.log(metrics)

        # lb_text for pasteing to leaderboard
        """relate.mse, relate.pearson, relate.spearman, relate.kendall_tau, pam_audio.mse, pam_audio.pearson, pam_audio.spearman, pam_audio.kendall_tau, pam_music.mse, pam_music.pearson, pam_music.spearman, pam_music.kendall_tau =  \n..."""
        lb_score_text = ""
        for dataset_name in test_dataset_names:
            for metric_name in ["mse", "pearson", "spearman", "kendall_tau"]:
                lb_score_text += f"{dataset_name}.{metric_name}, "
        lb_score_text = lb_score_text.strip().rstrip(",") + " =  \n"
        for dataset_name in test_dataset_names:
            for metric_name in ["mse", "pearson", "spearman", "kendall_tau"]:
                lb_score_text += f"{metrics[dataset_name][metric_name]:.4f}, "
        lb_score_text = lb_score_text.strip().rstrip(",")
        print("Leaderboard Score Text:")
        print(lb_score_text)

        # Save as json
        if self.save_qualitative:
            os.makedirs(self.model_dir, exist_ok=True)
            qualitative_data = {
                "metrics": metrics,
                "predictions": all_preds.tolist(),
                "scores": all_scores.tolist(),
                "meta_data": self.meta_data,
                "lb_score_text": lb_score_text,
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

    return parser


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()

    evaluator = AudioCapEval(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        batch_size=args.bs,
        lr=args.lr,
        epochs=args.epochs,
        log_wandb=args.log_wandb,
        save_qualitative=args.save_qualitative,
    )

    if args.mode == "train":
        evaluator.train()
        evaluator.test()
    elif args.mode == "test":
        # evaluator.load_model("best_model.pt")
        evaluator.test()
