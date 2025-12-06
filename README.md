# AudioCapEval

音声キャプション（Audio Captioning）の評価を行うためのツールキット。

## インストール

```bash
uv sync
```

### 動作環境
- Python 3.10.0
- CUDA 11.8（GPU使用時）

## データセットのダウンロード

```bash
bash scripts/download.sh
```

以下のデータセットがダウンロードされます：

- **RELATE**: 音声キャプション評価用データセット（[RELATE](https://github.com/sarulab-speech/RELATE)）
- **HumanEval**: 人間による評価データセット（[PAM](https://github.com/soham97/PAM)）

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
└── wav/                          # 音声波形データ
    ├── audiocaps/
    ├── audioldm/
    ├── audioldm2/
    ├── tango/
    └── tango2/
```

## 使い方

### 学習

```bash
uv run python src/main.py train --data_dir data --epochs 10 --bs 32 --lr 1e-4
```

### テスト

```bash
uv run python src/main.py test --data_dir data
```

### コマンドライン引数

| 引数 | デフォルト | 説明 |
|------|------------|------|
| `mode` | `train` | 実行モード（`train` or `test`） |
| `--data_dir` | `data` | データセットのディレクトリ |
| `--model_dir` | `models` | モデルの保存/読み込みディレクトリ |
| `--bs` | `32` | バッチサイズ |
| `--lr` | `1e-4` | 学習率 |
| `--epochs` | `10` | エポック数 |
| `--log_wandb` | `False` | Weights & Biasesへのログを有効化 |
| `--save_qualitative` | `False` | テスト結果をCSVで保存 |

### 出力

テスト実行時、以下の評価指標が出力されます：

```
mse, pearson, spearman, kendall_tau = 0.123, 0.456, 0.789, 0.654
```

`--save_qualitative` オプションを指定すると、`{model_dir}/test_results.csv` に予測結果が保存されます：

```csv
score,pred
0.85,0.82
0.72,0.75
...
```

## 評価指標

| 指標 | 説明 |
|------|------|
| MSE | 平均二乗誤差（Mean Squared Error） |
| Pearson | ピアソン相関係数 |
| Spearman | スピアマンの順位相関係数 |
| Kendall's τ | ケンドールの順位相関係数 |

## 定量結果
| モデル | MSE | Pearson | Spearman | Kendall's τ |
|--------|-----|---------|----------|-------------|
| [Microsoft CLAP](https://github.com/microsoft/CLAP) | 41.29 | 0.266 | 0.246 | 0.174 |
| [Laion CLAP](https://github.com/LAION-AI/CLAP) | 42.10 | 0.316 | 0.299 | 0.210 |
| Ours | XXX | XXX | XXX | XXX |


## 用語集

| 用語 | 説明 |
|------|------|
| OVL | 音声キャプションのオーバーラップ評価指標 |
| REL | 音声キャプション評価用データセット |
| OS | 音声キャプションの一貫性評価指標 |
| IS | 音声キャプションの多様性評価指標 |