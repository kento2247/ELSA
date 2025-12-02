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

## 用語集

| 用語 | 説明 |
|------|------|
| OVL | 音声キャプションのオーバーラップ評価指標 |
| REL | 音声キャプション評価用データセット |
| OS | 音声キャプションの一貫性評価指標 |
| IS | 音声キャプションの多様性評価指標 |