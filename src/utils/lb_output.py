def format_leaderboard_text(metrics: dict) -> str:
    """Format metrics dict to leaderboard text format."""
    header_parts: list[str] = []
    score_parts: list[str] = []

    for subjective_metric, datasets in metrics.items():
        for dataset_name, eval_metrics in datasets.items():
            for metric_name, value in eval_metrics.items():
                header_parts.append(f"{subjective_metric}.{dataset_name}.{metric_name}")
                score_parts.append(f"{value:.4f}")

    return ", ".join(header_parts) + "\n" + ", ".join(score_parts)
