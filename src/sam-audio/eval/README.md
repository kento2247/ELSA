# Evaluation

This directory contains the evaluation code to reproduce the results from the SAM-Audio paper. The evaluation framework supports multiple datasets, prompting modes (text-only, span, visual), and metrics.

## Setup

Before running evaluation, ensure you have:

1. Installed the SAM-Audio package and its dependencies
2. Authenticated with Hugging Face to access the model checkpoints (see main [README](../README.md))

## Quick Start

Run evaluation on the default setting (instr-pro):

```bash
python main.py
```

You can also use multiple GPUs to speed up evaluation:

```bash
torchrun --nproc_per_node=<ngpus> python main.py
```

Evaluate on a specific setting:

```bash
python main.py --setting sfx
```

Evaluate on multiple settings:

```bash
python main.py --setting sfx speech music
```

## Available Evaluation Settings

Run `python main.py --help` to see all available settings

## Command Line Options

```bash
python main.py [OPTIONS]
```

### Options:

- `-s, --setting` - Which setting(s) to evaluate (default: `instr-pro`)
  - Choices: See available settings above
  - Can specify multiple settings: `--setting sfx speech music`

- `--cache-path` - Where to cache downloaded datasets (default: `~/.cache/sam_audio`)

- `-p, --checkpoint-path` - Model checkpoint to evaluate (default: `facebook/sam-audio-1b`)
  - Can use local path or Hugging Face model ID

- `-b, --batch-size` - Batch size for evaluation (default: `1`)

- `-w, --num-workers` - Number of data loading workers (default: `4`)

- `-c, --candidates` - Number of reranking candidates (default: `8`)

## Evaluation Metrics

The evaluation framework computes the following metrics:

- **Judge** - SAM Audio Judge quality assessment metric
- **Aesthetic** - Aesthetic quality metric
- **CLAP** - Audio-text alignment metric (CLAP similarity)
- **ImageBind** - Audio-video alignment metric (for visual settings only)

## Output

Results are saved to the `results/` directory as JSON files, one per setting:

```
results/
├── sfx.json
├── speech.json
└── music.json
```

Each JSON file contains the averaged metric scores across all samples in that setting.

Example output:
```json
{
    "JudgeOverall": "4.386",
    "JudgeFaithfulness": "4.708",
    "JudgeRecall": "4.934",
    "JudgePrecision": "4.451",
    "ContentEnjoyment": "5.296",
    "ContentUsefulness": "6.903",
    "ProductionComplexity": "4.301",
    "ProductionQuality": "7.100",
    "CLAPSimilarity": "0.271"
}
```
