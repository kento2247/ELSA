import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Load
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Sankey diagram for GT vs Pred from scores/REL/audiocap."
    )
    parser.add_argument(
        "--json-path",
        required=True,
        help="Path to the qualitative results JSON file.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name to display in the plot title.",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=0.2,
        help="Bin size for scores (default: 0.2).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum score for binning (default: 0.0).",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=1.0,
        help="Maximum score for binning (default: 1.0).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    json_path = args.json_path
    model_name = args.model_name
    bin_size = args.bin_size
    min_score = args.min_score
    max_score = args.max_score

    data = json.loads(Path(json_path).read_text())

    # Extract REL/audiocap
    try:
        audiocap = data["scores"]["REL"]["audiocap"]
        gt = np.array(audiocap["y_list"], dtype=float)
        pred = np.array(audiocap["y_hat_list"], dtype=float)
    except Exception as e:
        raise KeyError(
            "Could not find scores/REL/audiocap with y_list and y_hat_list"
        ) from e

    if gt.shape != pred.shape:
        raise ValueError(f"Length mismatch: y_list={gt.shape} y_hat_list={pred.shape}")

    # Bin helper
    edges = np.arange(min_score, max_score + bin_size, bin_size)
    if edges[-1] < max_score:
        edges = np.append(edges, max_score)

    labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges) - 1)]

    def bin_scores(x):
        # Clip to range and bin by edges
        x = np.clip(x, min_score, max_score)
        # pd.cut right-inclusive on last bin
        return pd.cut(x, bins=edges, labels=labels, include_lowest=True, right=True)

    bin_gt = bin_scores(gt)
    bin_pred = bin_scores(pred)

    # Count transitions
    df = pd.DataFrame({"GT": bin_gt, "Pred": bin_pred})
    flow = df.value_counts().reset_index(name="count")

    # Build Sankey
    src_labels = [f"GT {l}" for l in labels]
    tgt_labels = [f"Pred {l}" for l in labels]
    all_labels = src_labels + tgt_labels
    label_to_idx = {l: i for i, l in enumerate(all_labels)}

    sources = flow["GT"].map(lambda l: label_to_idx[f"GT {l}"]).tolist()
    targets = flow["Pred"].map(lambda l: label_to_idx[f"Pred {l}"]).tolist()
    values = flow["count"].tolist()

    # Enforce top-to-bottom order for bins
    if len(labels) == 1:
        y_positions = [0.5]
    else:
        y_positions = np.linspace(0.05, 0.95, len(labels)).tolist()
    node_y = y_positions + y_positions
    node_x = [0.1] * len(labels) + [0.9] * len(labels)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=all_labels, pad=15, thickness=18, y=node_y, x=node_x),
                link=dict(source=sources, target=targets, value=values),
                arrangement="fixed",
            )
        ]
    )

    fig.update_layout(
        title=f"GT vs Pred (REL/audiocap) — {model_name}",
        font_size=36,
    )

    fig.show()

    # Optional: show table
    flow.sort_values("count", ascending=False).head(20)

    return 0


if __name__ == "__main__":
    main()
