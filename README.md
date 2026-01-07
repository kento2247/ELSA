# TTAEval

Text-to-Audio生成モデルの評価を行うためのツールキット。人間の主観評価スコア（REL/OVL）との相関を測定します。

## 特徴

- **複数のCLAPモデル対応**: Microsoft CLAP、LAION CLAPの埋め込み特徴量を使用
- **複数の評価指標**: MSE、Pearson、Spearman、Kendall's τ
- **複数データセット対応**: RELATE、AudioCap、MusicCap、XACLE
- **高度な前処理パイプライン**: Qwen LLMによるテキスト分割、SAM-Audioによる音源分離

## インストール

```bash
uv sync
```

### 動作環境
- Python 3.10.0
- CUDA 11.8（GPU使用時）
- VRAM 24GB (RTX4090推奨)

## データセットのダウンロード

```bash
bash scripts/download.sh
```

以下のデータセットがダウンロードされます：

- **RELATE**: 音声キャプション評価用データセット（[RELATE](https://github.com/sarulab-speech/RELATE)）
- **HumanEval**: 人間による評価データセット（[PAM](https://github.com/soham97/PAM)）
- **XACLE**: 音声キャプション評価用追加データセット

### ディレクトリ構造

```
data/
├── human_eval/
│   ├── audio/                    # 音声生成モデルの評価データ
│   │   ├── audiogen_m/
│   │   ├── audiolm_l/
│   │   ├── audiolm_l2/
│   │   ├── e2edef/
│   │   ├── real/                 # 実際の音声データ
│   │   └── scores.csv            # 評価スコア
│   └── music/                    # 音楽生成モデルの評価データ
│       ├── audioldm2/
│       ├── musicgen_large/
│       ├── musicgen_melody/
│       ├── musicldm/
│       ├── real/
│       └── scores.csv
├── RELATE/
│   ├── listener_attributes/      # リスナー属性データ
│   │   ├── IS_and_OS.csv
│   │   └── REL.csv
│   └── scores/                   # 評価スコア
│       ├── IS.csv
│       ├── OS.csv
│       └── REL.csv
├── XACLE_dataset/                # XACLEデータセット（train/val）
│   ├── train_average.csv
│   └── validation_average.csv
├── XACLE_test_data/              # XACLEテストデータ
│   └── test_with_score.csv
└── wav/                          # 音声波形データ
    ├── audiocaps/
    ├── audioldm/
    ├── audioldm2/
    ├── tango/
    └── tango2/
```

## 使い方

### 特徴量の事前抽出

テストを実行する前に、CLAP特徴量を事前に抽出しておく必要があります。

```bash
uv run python src/preprocess.py
```

### 学習

```bash
uv run python src/main.py train --data_dir data --epochs 30 --batch_size 32 --lr 1e-5
```

### テスト

```bash
# 全データセット・全指標でテスト
uv run python src/main.py test --data_dir data

# 特定のデータセット・指標を指定
uv run python src/main.py test --data_dir data \
    --subjective_metrics REL OVL \
    --test_dataset_names relate audiocap musiccap xacle

# 結果をJSONで保存
uv run python src/main.py test --data_dir data --save_qualitative
```

### lint チェック

```bash
uv run ruff check # チェックのみ
uv run ruff check --fix # 自動修正
```

### コマンドライン引数

| 引数 | デフォルト | 説明 |
|------|------------|------|
| `mode` | - | 実行モード（`train` or `test`） |
| `--data_dir` | `data` | データセットのディレクトリ |
| `--model_dir` | `models` | モデルの保存/読み込みディレクトリ |
| `--batch_size` | `32` | バッチサイズ |
| `--lr` | `1e-5` | 学習率 |
| `--epochs` | `30` | エポック数 |
| `--eval_freq` | `3` | 評価頻度（エポック単位） |
| `--main_metric` | `kendall_tau` | モデル選択の主指標（`mse`, `pearson`, `spearman`, `kendall_tau`） |
| `--subjective_metrics` | `REL OVL` | 評価する主観指標（`REL`, `OVL`） |
| `--test_dataset_names` | `relate audiocap musiccap xacle` | テストするデータセット名 |
| `--log_wandb` | `True` | Weights & Biasesへのログを有効化 |
| `--save_qualitative` | `False` | テスト結果をJSONで保存 |

### 出力

テスト実行時、各データセット・指標ごとに評価結果が出力されます。

`--save_qualitative` オプションを指定すると、`{model_dir}/qualitative_results.json` に予測結果とメタデータが保存されます：

```json
{
  "metrics": {
    "REL": {
      "relate": {"mse": 0.123, "pearson": 0.456, ...},
      "audiocap": {...}
    }
  },
  "predictions": [...],
  "scores": [...],
  "meta_data": {"timestamp": "...", "git_commit": "..."}
}
```

## 評価指標

| 指標 | 説明 |
|------|------|
| MSE | 平均二乗誤差（Mean Squared Error） |
| Pearson | ピアソン相関係数 |
| Spearman | スピアマンの順位相関係数 |
| Kendall's τ | ケンドールの順位相関係数 |

## Qualitative Results

### REL（Audio-キャプション間の類似度評価）

#### RELATE Dataset
| Model | MSE ↓ | Pearson ↑ | Spearman ↑ | Kendall τ ↑ |
|-------|------:|----------:|-----------:|------------:|
| [Microsoft CLAP](https://github.com/microsoft/CLAP) | 49.4577 | 0.0834 | 0.0905 | 0.0648 |
| [LAION CLAP](https://github.com/LAION-AI/CLAP) | 50.3844 | 0.0771 | 0.0838 | 0.0611 |
| [PAM](https://github.com/soham97/PAM) | 53.9596 | -0.0238 | -0.0207 | -0.0149 |
| Ours | XXX | XXX | XXX | XXX |

#### AudioCap Dataset
| Model | MSE ↓ | Pearson ↑ | Spearman ↑ | Kendall τ ↑ |
|-------|------:|----------:|-----------:|------------:|
| [Microsoft CLAP](https://github.com/microsoft/CLAP) | 9.0860 | 0.0897 | 0.0932 | 0.0647 |
| [LAION CLAP](https://github.com/LAION-AI/CLAP) | 9.9390 | 0.1439 | 0.1517 | 0.1040 |
| [PAM](https://github.com/soham97/PAM) | 9.9202 | -0.0305 | -0.0788 | -0.0517 |
| Ours | XXX | XXX | XXX | XXX |

#### MusicCap Dataset
| Model | MSE ↓ | Pearson ↑ | Spearman ↑ | Kendall τ ↑ |
|-------|------:|----------:|-----------:|------------:|
| [Microsoft CLAP](https://github.com/microsoft/CLAP) | 8.6426 | 0.1518 | 0.1442 | 0.0991 |
| [LAION CLAP](https://github.com/LAION-AI/CLAP) | 9.6082 | 0.1584 | 0.1745 | 0.1190 |
| [PAM](https://github.com/soham97/PAM) | 8.8781 | 0.0537 | 0.0496 | 0.0340 |
| Ours | XXX | XXX | XXX | XXX |

#### XACLE Dataset
| Model | MSE ↓ | Pearson ↑ | Spearman ↑ | Kendall τ ↑ |
|-------|------:|----------:|-----------:|------------:|
| [Microsoft CLAP](https://github.com/microsoft/CLAP) | XXX | XXX | XXX | XXX |
| [LAION CLAP](https://github.com/LAION-AI/CLAP) | XXX | XXX | XXX | XXX |
| [PAM](https://github.com/soham97/PAM) | XXX | XXX | XXX | XXX |
| Ours | XXX | XXX | XXX | XXX |

### OVL（Audio音質評価）
#### AudioCap Dataset
| Model | MSE ↓ | Pearson ↑ | Spearman ↑ | Kendall τ ↑ |
|-------|------:|----------:|-----------:|------------:|
| [Microsoft CLAP](https://github.com/microsoft/CLAP) | 7.2382 | 0.0399 | 0.0523 | 0.0357 |
| [LAION CLAP](https://github.com/LAION-AI/CLAP) | 8.0051 | 0.0857 | 0.0888 | 0.0615 |
| [PAM](https://github.com/soham97/PAM) | 7.9779 | -0.0553 | -0.0949 | -0.0667 |
| Ours | XXX | XXX | XXX | XXX |

#### MusicCap Dataset
| Model | MSE ↓ | Pearson ↑ | Spearman ↑ | Kendall τ ↑ |
|-------|------:|----------:|-----------:|------------:|
| [Microsoft CLAP](https://github.com/microsoft/CLAP) | 5.6764 | 0.0662 | 0.0610 | 0.0405 |
| [LAION CLAP](https://github.com/LAION-AI/CLAP) | 6.4569 | 0.0827 | 0.0804 | 0.0541 |
| [PAM](https://github.com/soham97/PAM) | 5.8532 | 0.0235 | 0.0290 | 0.0198 |
| Ours | XXX | XXX | XXX | XXX |

## プロジェクト構成

```
src/
├── main.py          # エントリーポイント・学習/テストの実行
├── model.py         # TTAEvalModel（MLP Head）
├── dataset.py       # TTADataset（RELATE/AudioCap/MusicCap/XACLE対応）
├── preprocess.py    # CLAP特徴量の事前抽出・テキスト分割・音源分離
└── utils/
    ├── eval_methods.py  # 評価指標（MSE, Pearson, Spearman, Kendall）
    └── lb_output.py     # リーダーボード出力フォーマット
```

### 前処理パイプライン

`preprocess.py`は以下の機能を提供します：

| クラス/関数 | 説明 |
|-------------|------|
| `MSClapEmbedder` | Microsoft CLAPによる音声・テキスト埋め込み抽出 |
| `LaionClapEmbedder` | LAION CLAPによる音声・テキスト埋め込み抽出 |
| `QwenTextParser` | Qwen3-4Bによるキャプションの音声イベント分割 |
| `SamAudio` | SAM-Audioによる音源分離 |
| `msclap_extract()` | MSCLAP特徴量のバッチ抽出 |
| `laionclap_extract()` | LAION CLAP特徴量のバッチ抽出 |
| `text_parse()` | テキストの音声イベント分割 |
| `music_parse()` | 音源分離の実行 |
| `embed_parsed_data()` | 分離音源と分割テキストの埋め込み |

### 特徴量ディレクトリ構造

```
data/features/
├── msclap_audio/           # MSCLAP音声埋め込み
├── msclap_text/            # MSCLAPテキスト埋め込み
├── laionclap_audio/        # LAION CLAP音声埋め込み
├── laionclap_text/         # LAION CLAPテキスト埋め込み
├── parsed_texts/           # Qwenによる分割テキスト（JSON）
├── separated_audio/        # SAM-Audioによる分離音源
├── laionclap_parsed_audio/ # 分離音源の埋め込み
├── laionclap_parsed_text/  # 分割テキストの埋め込み
└── parsed_mask/            # シーケンスマスク
```

## 参考文献

- [RELATE](https://github.com/sarulab-speech/RELATE) - 音声キャプション評価用データセット
- [PAM](https://github.com/soham97/PAM) - Perceptual Audio Metric
- [Microsoft CLAP](https://github.com/microsoft/CLAP)
- [LAION CLAP](https://github.com/LAION-AI/CLAP)
- [SAM-Audio](https://github.com/kento2247/sam-audio) - 音源分離モデル
- [Qwen](https://github.com/QwenLM/Qwen) - テキスト分割用LLM