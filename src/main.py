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
        features_dir: str,
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
        parse_event_count: str = "all",
        clap_variant: str = "humanclap",
    ):
        self.data_dir = data_dir
        self.features_dir = features_dir
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.eval_freq = eval_freq
        self.main_metric = main_metric
        self.subjective_metrics = subjective_metrics
        self.test_dataset_names = test_dataset_names
        self.parse_event_count = parse_event_count
        self.log_wandb = log_wandb
        self.save_qualitative = save_qualitative
        self.clap_variant = clap_variant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TTAEvalModel().to(self.device)

        self._load_quality_prompts()

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

    def _maybe_to_device(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return None

    def _load_quality_prompts(self):
        feats_dir = os.path.join(self.data_dir, "features", "quality_prompts")
        high_path = os.path.join(feats_dir, "high.pt")
        low_path = os.path.join(feats_dir, "low.pt")
        unrelated_path = os.path.join(feats_dir, "unrelated.pt")

        if os.path.exists(high_path) and os.path.exists(low_path):
            high_emb = torch.load(high_path, map_location=self.device)
            low_emb = torch.load(low_path, map_location=self.device)

            unrelated_emb = None
            if os.path.exists(unrelated_path):
                unrelated_emb = torch.load(unrelated_path, map_location=self.device)
                print(f"Loaded quality + contrast prompts from {feats_dir}")
            else:
                print(f"Loaded quality prompts from {feats_dir} (no contrast prompt)")

            self.model.load_quality_prompts(high_emb, low_emb, unrelated_emb)
        else:
            print(
                f"Quality prompts not found at {feats_dir}. "
                "Run 'uv run src/preprocess.py --quality_prompts' to generate them. "
                "REL and OVL will produce identical predictions until quality prompts are loaded."
            )

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
                val_metrics = self.evaluate(val_loader, desc="Validation")["metrics"]
                val_metric = val_metrics[self.main_metric]
                is_best_epoch = val_metric > best_val_metric

                print(
                    f"Epoch {epoch}/{self.epochs} - Val Metrics: {val_metrics}, Best: {is_best_epoch}"
                )

                if self.log_wandb:
                    wandb.log({"epoch": epoch, "val": val_metrics})

                print(f"Running test at epoch {epoch}...")
                test_metrics = self.test(
                    save_qualitative=self.save_qualitative and is_best_epoch
                )

                if is_best_epoch:
                    best_val_metric = val_metric
                    best_epoch = epoch
                    self.save_model("best_model.pt")
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
            clap_audio = batch[f"{self.clap_variant}_audio"].to(self.device)
            clap_text = batch[f"{self.clap_variant}_text"].to(self.device)
            clap_parsed_audio = self._maybe_to_device(
                batch.get(f"{self.clap_variant}_parsed_audio")
            )
            clap_parsed_text = self._maybe_to_device(
                batch.get(f"{self.clap_variant}_parsed_text")
            )
            parsed_mask = self._maybe_to_device(
                batch.get(f"{self.clap_variant}_parsed_mask")
            )
            metric_id = batch.get("subjective_metric_id")
            if metric_id is not None:
                metric_id = metric_id.to(self.device)
            scores = batch["score"].float().to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(
                clap_audio,
                clap_text,
                clap_parsed_audio,
                clap_parsed_text,
                parsed_mask,
                metric_id,
            ).squeeze(-1)
            loss = self.criterion(preds, scores)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, data_loader: DataLoader, desc: str = "Evaluating") -> dict:
        """Evaluate the model and return metrics dict."""
        # Check if this is a CompA dataset (multiple choice task)
        is_compa = next(iter(data_loader))["dataset"][0] == "compa"
        if is_compa:
            return self._evaluate_compa(data_loader, desc)

        self.model.eval()
        all_preds: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_audio_file_paths: list[str] = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                clap_audio = batch[f"{self.clap_variant}_audio"].to(self.device)
                clap_text = batch[f"{self.clap_variant}_text"].to(self.device)
                clap_parsed_audio = self._maybe_to_device(
                    batch.get(f"{self.clap_variant}_parsed_audio")
                )
                clap_parsed_text = self._maybe_to_device(
                    batch.get(f"{self.clap_variant}_parsed_text")
                )
                parsed_mask = self._maybe_to_device(
                    batch.get(f"{self.clap_variant}_parsed_mask")
                )
                metric_id = batch.get("subjective_metric_id")
                if metric_id is not None:
                    metric_id = metric_id.to(self.device)
                scores = batch["score"].numpy()
                audio_file_path = batch["audio_file_path"]

                preds = (
                    self.model(
                        clap_audio,
                        clap_text,
                        clap_parsed_audio,
                        clap_parsed_text,
                        parsed_mask,
                        metric_id,
                    )
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                )

                # Filter by parse event count
                if self.parse_event_count != "all" and parsed_mask is not None:
                    # Calculate event count for each sample in the batch
                    event_counts = parsed_mask.sum(dim=-1).cpu().numpy()  # [batch_size]

                    # Create filter mask based on parse_event_count
                    if self.parse_event_count == "1":
                        filter_mask = event_counts == 1
                    elif self.parse_event_count == "2":
                        filter_mask = event_counts == 2
                    elif self.parse_event_count == "3+":
                        filter_mask = event_counts >= 3
                    else:
                        filter_mask = np.ones_like(event_counts, dtype=bool)

                    # Apply filter
                    preds = preds[filter_mask]
                    scores = scores[filter_mask]
                    audio_file_path = [
                        audio_file_path[i]
                        for i in range(len(audio_file_path))
                        if filter_mask[i]
                    ]

                all_preds.append(preds)
                all_scores.append(scores)
                all_audio_file_paths.extend(audio_file_path)

        # Filter out empty arrays and concatenate
        all_preds = [p for p in all_preds if len(p) > 0]
        all_scores = [s for s in all_scores if len(s) > 0]

        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds)
            all_scores = np.concatenate(all_scores)
        else:
            all_preds = np.array([])
            all_scores = np.array([])
            print(
                f"Warning: No samples found matching parse_event_count={self.parse_event_count}"
            )
            return {
                "metrics": {
                    "mse": float("nan"),
                    "pearson": float("nan"),
                    "spearman": float("nan"),
                    "kendall_tau": float("nan"),
                },
                "y_list": all_scores,
                "y_hat_list": all_preds,
                "audio_file_paths": all_audio_file_paths,
            }

        return {
            "metrics": {
                "mse": mse(all_scores, all_preds),
                "pearson": pearson_correlation(all_scores, all_preds),
                "spearman": spearman_correlation(all_scores, all_preds),
                "kendall_tau": kendall_tau(all_scores, all_preds),
            },
            "y_list": all_scores,
            "y_hat_list": all_preds,
            "audio_file_paths": all_audio_file_paths,
        }

    def _evaluate_compa(
        self, data_loader: DataLoader, desc: str = "Evaluating"
    ) -> dict:
        """Evaluate CompA dataset as multiple choice classification task.

        For AttributeText/OrderText: Given audio, choose the correct text from choices.
        For AttributeAudio/OrderAudio: Given text, choose the correct audio from choices.
        """

        self.model.eval()
        all_points = 0
        num_samples = 0
        is_text_task = data_loader.dataset.subjective_metrics[0] in [
            "AttributeText",
            "OrderText",
        ]

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                # Extract batch data
                batch_size = batch[f"{self.clap_variant}_audio"].shape[0]

                # AttributeText or OrderText: audio -> text
                if is_text_task:
                    for i in range(batch_size):
                        both_correct = True
                        for correct_idx, (audio_prefix, audio_suffix) in enumerate(
                            [("", ""), ("rev_", "_rev")]
                        ):
                            # Query audio embed.
                            query_audio_feats = (
                                batch[f"{self.clap_variant}_audio{audio_suffix}"][i]
                                .to(self.device)
                                .reshape(1, -1)
                                .repeat(2, 1)
                            )

                            # Target text embed.
                            text_feats_list = []
                            parsed_audio_list = []
                            parsed_text_list = []
                            parsed_mask_list = []
                            for text_suffix in ["", "_rev"]:
                                text_feats_list.append(
                                    batch[f"{self.clap_variant}_text{text_suffix}"][i]
                                    .to(self.device)
                                    .reshape(-1)
                                )
                                parsed_audio_list.append(
                                    batch[
                                        f"{audio_prefix}{self.clap_variant}_parsed_audio{text_suffix}"
                                    ][i].to(self.device)
                                )
                                parsed_text_list.append(
                                    batch[
                                        f"{audio_prefix}{self.clap_variant}_parsed_text{text_suffix}"
                                    ][i].to(self.device)
                                )
                                parsed_mask_list.append(
                                    batch[
                                        f"{audio_prefix}{self.clap_variant}_parsed_mask{text_suffix}"
                                    ][i].to(self.device)
                                )

                            # Stack all choice features: [num_choices, D]
                            text_feats = torch.stack(text_feats_list, dim=0)
                            parsed_text = torch.stack(parsed_text_list, dim=0)
                            parsed_audio = torch.stack(parsed_audio_list, dim=0)
                            parsed_mask = torch.stack(parsed_mask_list, dim=0)

                            preds = (
                                self.model(
                                    query_audio_feats,
                                    text_feats,
                                    parsed_audio,
                                    parsed_text,
                                    parsed_mask,
                                    torch.zeros(2, dtype=torch.long).to(self.device),
                                )
                                .cpu()
                                .numpy()
                            )

                            # Select the choice with highest similarity
                            pred_choice_idx = int(np.argmax(preds))
                            if correct_idx != pred_choice_idx:
                                both_correct = False
                                break

                        all_points += 1 if both_correct else 0
                        num_samples += 1

                # AttributeAudio or OrderAudio: text -> audio
                else:
                    for i in range(batch_size):
                        both_correct = True
                        for correct_idx, text_suffix in enumerate(["", "_rev"]):
                            # Query text embed.
                            query_text_feats = (
                                batch[f"{self.clap_variant}_text{text_suffix}"][i]
                                .to(self.device)
                                .reshape(1, -1)
                                .repeat(2, 1)
                            )

                            # Target audio embed.
                            audio_feats_list = []
                            parsed_audio_list = []
                            parsed_text_list = []
                            parsed_mask_list = []
                            for audio_prefix, audio_suffix in [
                                ("", ""),
                                ("rev_", "_rev"),
                            ]:
                                audio_feats_list.append(
                                    batch[f"{self.clap_variant}_audio{audio_suffix}"][i]
                                    .to(self.device)
                                    .reshape(-1)
                                )
                                parsed_audio_list.append(
                                    batch[
                                        f"{audio_prefix}{self.clap_variant}_parsed_audio{text_suffix}"
                                    ][i].to(self.device)
                                )
                                parsed_text_list.append(
                                    batch[
                                        f"{audio_prefix}{self.clap_variant}_parsed_text{text_suffix}"
                                    ][i].to(self.device)
                                )
                                parsed_mask_list.append(
                                    batch[
                                        f"{audio_prefix}{self.clap_variant}_parsed_mask{text_suffix}"
                                    ][i].to(self.device)
                                )

                            # Stack all choice features: [num_choices, D]
                            audio_feats = torch.stack(audio_feats_list, dim=0)
                            parsed_audio = torch.stack(parsed_audio_list, dim=0)
                            parsed_text = torch.stack(parsed_text_list, dim=0)
                            parsed_mask = torch.stack(parsed_mask_list, dim=0)

                            preds = (
                                self.model(
                                    audio_feats,
                                    query_text_feats,
                                    parsed_audio,
                                    parsed_text,
                                    parsed_mask,
                                    torch.zeros(2, dtype=torch.long).to(self.device),
                                )
                                .cpu()
                                .numpy()
                            )

                            # Select the choice with highest similarity
                            pred_choice_idx = int(np.argmax(preds))
                            if correct_idx != pred_choice_idx:
                                both_correct = False
                                break

                        all_points += 1 if both_correct else 0
                        num_samples += 1

        return {
            "metrics": {
                "accuracy": all_points / num_samples,
            },
            "y_list": np.zeros(num_samples),
            "y_hat_list": np.zeros(num_samples),
            "audio_file_paths": [],
        }

    def test(self, save_qualitative: bool = False) -> dict:
        """Test the model on test datasets and log metrics."""
        metrics: dict = {}
        scores: dict = {}

        for subjective_metric in self.subjective_metrics:
            for test_dataset_name in self.test_dataset_names:
                test_dataset = TTADataset(
                    data_dir=self.data_dir,
                    features_dir=self.features_dir,
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
                    num_workers=0,
                )
                del test_dataset

                desc = f"Testing {subjective_metric}/{test_dataset_name}"
                eval_result = self.evaluate(test_loader, desc=desc)
                eval_metrics = eval_result["metrics"]

                if subjective_metric not in metrics:
                    metrics[subjective_metric] = {}
                if subjective_metric not in scores:
                    scores[subjective_metric] = {}
                metrics[subjective_metric][test_dataset_name] = eval_metrics
                scores[subjective_metric][test_dataset_name] = {
                    "y_list": eval_result["y_list"],
                    "y_hat_list": eval_result["y_hat_list"],
                    "audio_file_paths": eval_result["audio_file_paths"],
                }

        if self.log_wandb:
            wandb.log(metrics)

        if save_qualitative:
            os.makedirs(self.model_dir, exist_ok=True)
            # Convert numpy arrays to lists for JSON serialization
            scores_serializable = {}
            for metric_name, datasets in scores.items():
                scores_serializable[metric_name] = {}
                for dataset_name, data in datasets.items():
                    scores_serializable[metric_name][dataset_name] = {
                        "y_list": [float(y) for y in data["y_list"]],
                        "y_hat_list": [float(y) for y in data["y_hat_list"]],
                        "audio_file_paths": data["audio_file_paths"],
                    }
            qualitative_data = {
                "metrics": metrics,
                "meta_data": self.meta_data,
                "scores": scores_serializable,
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
        default="test",
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
        "--features_dir",
        type=str,
        default="features",
        help="Directory containing the precomputed features",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save/load the model",
    )
    # training params
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
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
        # default=["REL", "OVL", "IS", "OS", "AttributeText", "AttributeAudio", "OrderText", "OrderAudio"],
        default=["AttributeText", "AttributeAudio", "OrderText", "OrderAudio"],
        help="Subjective metric to use from the dataset",
    )
    parser.add_argument(
        "--test_dataset_names",
        type=str,
        nargs="+",
        # default=["relate", "audiocap", "musiccap", "aishell7b", "clotho", "compa"],
        default=["compa"],
        help="List of dataset names to test on",
    )
    parser.add_argument(
        "--parse_event_count",
        type=str,
        default="all",
        choices=["all", "1", "2", "3+"],
        help="Filter samples by parse event count: all, 1, 2, or 3+",
    )
    # logging
    parser.add_argument(
        "--log_wandb",
        action="store_true",
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
        data_dir=args.data_dir,
        features_dir=args.features_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        eval_freq=args.eval_freq,
        main_metric=args.main_metric,
        subjective_metrics=args.subjective_metrics,
        test_dataset_names=args.test_dataset_names,
        parse_event_count=args.parse_event_count,
        log_wandb=args.log_wandb,
        save_qualitative=args.save_qualitative,
    )

    if args.mode == "train":
        raise NotImplementedError("Training mode is currently disabled.")
    elif args.mode == "test":
        test_metrics = evaluator.test()
        lb_text = format_leaderboard_text(evaluator.meta_data, test_metrics)
        print(f"Leaderboard Text:\n{lb_text}")
