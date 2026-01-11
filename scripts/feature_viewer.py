"""
Feature Viewer - Flask app to analyze separated audio and parsed texts.

Usage:
    uv run --group dev python scripts/feature_viewer.py
    Then open http://localhost:5000 in your browser.
"""

import json
import os
from pathlib import Path

import pandas as pd
from flask import Flask, render_template_string, send_file, request

# Base paths
DATA_DIR = Path(__file__).parent.parent / "data"
FEATURES_DIR = DATA_DIR / "features"
PARSED_TEXTS_DIR = FEATURES_DIR / "parsed_texts"
SEPARATED_AUDIO_DIR = FEATURES_DIR / "separated_audio"

DATASETS = ["relate", "audiocap", "musiccap", "xacle"]

app = Flask(__name__)

# Cache for metadata
_metadata_cache: dict[str, dict] = {}


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AudioCapEval Feature Viewer</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #333; margin-bottom: 20px; }
        .nav-bar {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
        }
        .nav-bar label { font-weight: 600; }
        .nav-bar select, .nav-bar input {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        .nav-bar select { min-width: 150px; }
        .nav-bar input[type="text"] { width: 200px; }
        .nav-buttons { display: flex; gap: 10px; align-items: center; }
        .nav-buttons button, .nav-buttons a {
            padding: 8px 16px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
            font-size: 14px;
        }
        .nav-buttons button:hover, .nav-buttons a:hover { background: #0056b3; }
        .nav-buttons button:disabled { background: #ccc; cursor: not-allowed; }
        .nav-info { color: #666; font-size: 14px; }
        .content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 900px) { .content { grid-template-columns: 1fr; } }
        .card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h2 {
            margin: 0 0 15px 0;
            font-size: 18px;
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        .text-content {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #007bff;
            line-height: 1.6;
        }
        .parsed-list { list-style: none; padding: 0; margin: 0; }
        .parsed-list li {
            padding: 10px;
            background: #f8f9fa;
            margin-bottom: 8px;
            border-radius: 4px;
            border-left: 4px solid #28a745;
        }
        .parsed-list li::before {
            content: counter(item) ". ";
            counter-increment: item;
            font-weight: bold;
            color: #28a745;
        }
        .parsed-list { counter-reset: item; }
        .audio-player {
            width: 100%;
            margin-top: 10px;
        }
        .segment {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 15px;
            border-left: 4px solid #17a2b8;
        }
        .segment-label {
            font-weight: 600;
            color: #17a2b8;
            margin-bottom: 8px;
        }
        .metadata {
            background: #e9ecef;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 13px;
        }
        .metadata-item { margin: 5px 0; }
        .metadata-key { font-weight: 600; color: #495057; }
        .warning {
            color: #856404;
            background: #fff3cd;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #ffc107;
        }
        .text-id-display {
            font-family: monospace;
            background: #e9ecef;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>AudioCapEval Feature Viewer</h1>

        <div class="nav-bar">
            <div>
                <label for="dataset">Dataset:</label>
                <select id="dataset" onchange="changeDataset()">
                    {% for ds in datasets %}
                    <option value="{{ ds }}" {{ 'selected' if ds == current_dataset else '' }}>{{ ds }}</option>
                    {% endfor %}
                </select>
            </div>

            <div>
                <label for="search">Search ID:</label>
                <input type="text" id="search" placeholder="e.g., test_REL_0" value="{{ search_query }}"
                       onkeypress="if(event.key==='Enter')searchId()">
                <button onclick="searchId()">Search</button>
            </div>

            <div class="nav-buttons">
                <a href="/?dataset={{ current_dataset }}&index={{ prev_index }}">&#9664; Prev</a>
                <span class="nav-info">{{ current_index + 1 }} / {{ total_items }}</span>
                <a href="/?dataset={{ current_dataset }}&index={{ next_index }}">Next &#9654;</a>
            </div>

            <div>
                <span class="text-id-display">{{ current_text_id }}</span>
            </div>
        </div>

        <div class="content">
            <div>
                <div class="card">
                    <h2>Original Text</h2>
                    {% if original_text %}
                    <div class="text-content">{{ original_text }}</div>
                    {% if metadata %}
                    <div class="metadata">
                        {% for key, value in metadata.items() %}
                        {% if key not in ['text', 'audio_path'] %}
                        <div class="metadata-item"><span class="metadata-key">{{ key }}:</span> {{ value }}</div>
                        {% endif %}
                        {% endfor %}
                    </div>
                    {% endif %}
                    {% else %}
                    <div class="warning">Metadata not found for this text_id</div>
                    {% endif %}
                </div>

                <div class="card" style="margin-top: 20px;">
                    <h2>Parsed Text (Sound Events)</h2>
                    {% if parsed_text %}
                    <ol class="parsed-list">
                        {% for event in parsed_text %}
                        <li>{{ event }}</li>
                        {% endfor %}
                    </ol>
                    {% else %}
                    <div class="warning">Parsed text not found</div>
                    {% endif %}
                </div>
            </div>

            <div>
                <div class="card">
                    <h2>Original Audio</h2>
                    {% if audio_exists %}
                    <audio controls class="audio-player">
                        <source src="/audio/original/{{ current_dataset }}/{{ current_text_id }}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                    {% else %}
                    <div class="warning">Audio file not found: {{ audio_path }}</div>
                    {% endif %}
                </div>

                <div class="card" style="margin-top: 20px;">
                    <h2>Separated Audio Segments</h2>
                    {% if separated_audio %}
                    {% for i, segment in separated_audio %}
                    <div class="segment">
                        <div class="segment-label">{{ i + 1 }}. {{ segment.label }}</div>
                        <audio controls class="audio-player">
                            <source src="/audio/separated/{{ current_dataset }}/{{ current_text_id }}/{{ i }}" type="audio/wav">
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                    {% endfor %}
                    {% else %}
                    <div class="warning">No separated audio files found</div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>

    <script>
        function changeDataset() {
            const dataset = document.getElementById('dataset').value;
            window.location.href = '/?dataset=' + dataset + '&index=0';
        }

        function searchId() {
            const dataset = document.getElementById('dataset').value;
            const search = document.getElementById('search').value;
            if (search) {
                window.location.href = '/?dataset=' + dataset + '&search=' + encodeURIComponent(search);
            }
        }
    </script>
</body>
</html>
"""


def load_relate_metadata() -> dict[str, dict]:
    """Load RELATE dataset metadata, mapping text_id to (text, audio_path)."""
    csv_path = DATA_DIR / "RELATE" / "scores" / "REL.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)

    metadata = {}
    for split in ["train", "test", "validation"]:
        split_key = "val" if split == "validation" else split
        data = df[df["in RELATE dataset"] == split].reset_index(drop=True)

        for index, row in data.iterrows():
            text_id = f"{split_key}_REL_{index}"
            wavname = row["wavname"]
            audio_path = DATA_DIR / f"wav{wavname}"

            metadata[text_id] = {
                "text": row["text"],
                "audio_path": str(audio_path),
                "score": row["score"],
                "audio_type": row["audio type"],
            }

    return metadata


def load_audiocap_metadata() -> dict[str, dict]:
    """Load AudioCap dataset metadata, mapping text_id to (text, audio_path)."""
    csv_path = DATA_DIR / "human_eval" / "audio" / "scores.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)

    metadata = {}
    for metric in ["OVL", "REL"]:
        for index, row in df.iterrows():
            text_id = f"test_{metric}_{index}"
            model = row["Model"]
            file_name = row["File Name"]
            audio_path = DATA_DIR / "human_eval" / "audio" / model / f"{file_name}.wav"

            metadata[text_id] = {
                "text": row["Text"],
                "audio_path": str(audio_path),
                "model": model,
                "file_name": file_name,
                "ovl_score": row["OVL"],
                "rel_score": row["REL"],
            }

    return metadata


def load_musiccap_metadata() -> dict[str, dict]:
    """Load MusicCap dataset metadata, mapping text_id to (text, audio_path)."""
    csv_path = DATA_DIR / "human_eval" / "music" / "scores.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)

    metadata = {}
    for metric in ["OVL", "REL"]:
        for index, row in df.iterrows():
            text_id = f"test_{metric}_{index}"
            model = row["Model"]
            file_name = row["File Name"]
            audio_path = DATA_DIR / "human_eval" / "music" / model / f"{file_name}.wav"

            metadata[text_id] = {
                "text": row["Text"],
                "audio_path": str(audio_path),
                "model": model,
                "file_name": file_name,
                "ovl_score": row["OVL"],
                "rel_score": row["REL"],
            }

    return metadata


def load_xacle_metadata() -> dict[str, dict]:
    """Load XACLE dataset metadata, mapping text_id to (text, audio_path)."""
    metadata = {}

    # Test data
    csv_path = DATA_DIR / "XACLE_test_data" / "meta_data" / "test_with_score.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for index, row in df.iterrows():
            text_id = f"test_REL_{index}"
            wavname = row["wav_file_name"]
            audio_path = DATA_DIR / "XACLE_test_data" / "wav" / wavname

            metadata[text_id] = {
                "text": row["text"],
                "audio_path": str(audio_path),
                "score": row["average_score"],
            }

    # Train/val data
    for split, split_name in [("train", "train"), ("val", "validation")]:
        csv_path = DATA_DIR / "XACLE_dataset" / "meta_data" / f"{split_name}_average.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for index, row in df.iterrows():
                text_id = f"{split}_REL_{index}"
                wavname = row["wav_file_name"]
                audio_path = DATA_DIR / "XACLE_dataset" / "wav" / split_name / wavname

                metadata[text_id] = {
                    "text": row["text"],
                    "audio_path": str(audio_path),
                    "score": row["average_score"],
                }

    return metadata


def get_metadata(dataset: str) -> dict[str, dict]:
    """Get metadata for a dataset, with caching."""
    if dataset not in _metadata_cache:
        loaders = {
            "relate": load_relate_metadata,
            "audiocap": load_audiocap_metadata,
            "musiccap": load_musiccap_metadata,
            "xacle": load_xacle_metadata,
        }
        _metadata_cache[dataset] = loaders[dataset]()
    return _metadata_cache[dataset]


def get_available_text_ids(dataset: str) -> list[str]:
    """Get list of available text_ids for a dataset from parsed_texts directory."""
    parsed_dir = PARSED_TEXTS_DIR / dataset
    if not parsed_dir.exists():
        return []

    text_ids = []
    for f in parsed_dir.iterdir():
        if f.suffix == ".json":
            text_ids.append(f.stem)

    return sorted(text_ids, key=lambda x: (x.split("_")[0], x.split("_")[1], int(x.split("_")[2])))


def load_parsed_text(dataset: str, text_id: str) -> list[str] | None:
    """Load parsed text for a specific text_id."""
    json_path = PARSED_TEXTS_DIR / dataset / f"{text_id}.json"
    if not json_path.exists():
        return None

    with open(json_path) as f:
        return json.load(f)


def get_separated_audio_files(dataset: str, text_id: str) -> list[Path]:
    """Get list of separated audio files for a specific text_id."""
    audio_dir = SEPARATED_AUDIO_DIR / dataset / text_id
    if not audio_dir.exists():
        return []

    audio_files = list(audio_dir.glob("*.wav"))
    return sorted(audio_files, key=lambda x: int(x.stem))


@app.route("/")
def index():
    """Main page with feature viewer."""
    # Get parameters
    dataset = request.args.get("dataset", "relate")
    if dataset not in DATASETS:
        dataset = "relate"

    search_query = request.args.get("search", "")
    index_param = request.args.get("index", "0")

    # Get available text_ids
    text_ids = get_available_text_ids(dataset)
    if not text_ids:
        return render_template_string(
            "<h1>No data found for dataset: {{ dataset }}</h1>",
            dataset=dataset,
        )

    # Handle search
    if search_query:
        matching = [tid for tid in text_ids if search_query.lower() in tid.lower()]
        if matching:
            current_index = text_ids.index(matching[0])
        else:
            current_index = 0
    else:
        try:
            current_index = int(index_param)
            current_index = max(0, min(current_index, len(text_ids) - 1))
        except ValueError:
            current_index = 0

    current_text_id = text_ids[current_index]

    # Load data
    metadata = get_metadata(dataset)
    item_meta = metadata.get(current_text_id, {})
    parsed_text = load_parsed_text(dataset, current_text_id)
    separated_files = get_separated_audio_files(dataset, current_text_id)

    # Check if audio exists
    audio_path = item_meta.get("audio_path", "")
    audio_exists = audio_path and os.path.exists(audio_path)

    # Prepare separated audio data
    separated_audio = []
    if separated_files:
        parsed_labels = parsed_text if parsed_text else []
        for i, audio_file in enumerate(separated_files):
            label = parsed_labels[i] if i < len(parsed_labels) else f"Segment {i}"
            separated_audio.append((i, {"label": label, "path": str(audio_file)}))

    return render_template_string(
        HTML_TEMPLATE,
        datasets=DATASETS,
        current_dataset=dataset,
        current_index=current_index,
        current_text_id=current_text_id,
        total_items=len(text_ids),
        prev_index=max(0, current_index - 1),
        next_index=min(len(text_ids) - 1, current_index + 1),
        search_query=search_query,
        original_text=item_meta.get("text"),
        metadata=item_meta,
        parsed_text=parsed_text,
        audio_exists=audio_exists,
        audio_path=audio_path,
        separated_audio=separated_audio,
    )


@app.route("/audio/original/<dataset>/<text_id>")
def serve_original_audio(dataset: str, text_id: str):
    """Serve original audio file."""
    metadata = get_metadata(dataset)
    item_meta = metadata.get(text_id, {})
    audio_path = item_meta.get("audio_path", "")

    if audio_path and os.path.exists(audio_path):
        return send_file(audio_path, mimetype="audio/wav")
    return "Audio not found", 404


@app.route("/audio/separated/<dataset>/<text_id>/<int:segment_index>")
def serve_separated_audio(dataset: str, text_id: str, segment_index: int):
    """Serve separated audio segment."""
    audio_dir = SEPARATED_AUDIO_DIR / dataset / text_id
    audio_file = audio_dir / f"{segment_index}.wav"

    if audio_file.exists():
        return send_file(str(audio_file), mimetype="audio/wav")
    return "Audio not found", 404


if __name__ == "__main__":
    print("Starting Feature Viewer...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Features directory: {FEATURES_DIR}")
    print("\nOpen http://localhost:5000 in your browser")
    app.run(debug=True, host="0.0.0.0", port=5000)
