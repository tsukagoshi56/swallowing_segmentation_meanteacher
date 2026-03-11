# 嚥下音検出のための半教師あり実環境適応

以下の論文の実装です：

> **"Semi-Supervised Real-World Adaptation for Swallowing Sound Detection"**
> 塚越俊宏，西田雅史，西村雅史
> NCSP 2026，静岡大学 / 愛知産業大学

---

## 概要

皮膚接触型マイクを用いた実環境（自然な食事場面）での嚥下音検出を目的とした，2段階の半教師あり学習フレームワークの実装です。

**基本的なアイデア：**
WavLM + GRU モデルを少量のラベル付き実環境データで教師あり Fine-tuning（Stage 1）した後，Mean Teacher フレームワークを用いて大量のラベルなし実環境データでさらに適応（Stage 2）します。

### 結果（イベントベース F1，IoU=0.1）

| モデル | ピザ | りんご | クラッカー | 全体 |
|--------|------|--------|------------|------|
| (1) 制御環境録音のみ | 0.451 | 0.450 | 0.609 | 0.470 |
| (2) + 音声データ（30時間） | 0.785 | 0.684 | 0.789 | 0.756 |
| (3) + ラベル付き実環境データ（1時間） | 0.932 | 0.838 | 0.897 | 0.899 |
| **(4) + ラベルなし実環境データ（85時間）【提案手法】** | **0.947** | **0.903** | **0.950** | **0.933** |

---

## データセット準備

### ディレクトリ構成

```
dataset/
├── exist_label/               # ラベル付き実環境データ（Stage 1 学習・Stage 2 検証用）
│   ├── 301.wav                # 4ch合成・HPF済みシングルchWAV（参加者ごと）
│   ├── 301.txt                # アノテーションファイル（参加者ごと）
│   ├── divide_wav/            # 10秒セグメントに分割したWAV（自動生成）
│   ├── divide_txt/            # 対応するアノテーション（自動生成）
│   ├── exist_label_train.json # 学習用JSONリスト（自動生成）
│   └── exist_label_valid.json # 検証用JSONリスト（自動生成）
├── no_label/                  # ラベルなし実環境データ（Stage 2 学習用）
│   └── <収録日>/<参加者ID>/
│       ├── MIC1.WAV           # チャンネル1
│       ├── MIC2.WAV           # チャンネル2
│       ├── MIC3.WAV           # チャンネル3
│       └── MIC4.WAV           # チャンネル4
└── test_data/                 # テストデータ
    ├── <日付>_<ID>_MIC1_MIC2_MIC3_MIC4_HPF.wav
    └── <日付>_<ID>_MIC1_MIC2_MIC3_MIC4_HPF.txt
```

---

### ステップ 1：生録音の前処理（4ch → 1ch + ハイパスフィルタ）

生録音は4チャンネル（MIC1〜MIC4）で収録されています。
これらをチャンネル平均で合成し，100 Hz ハイパスフィルタを適用してシングルチャンネルWAVを生成します。

```bash
python prepare/5_db_combined_hpf.py \
    --input-dir dataset/no_label/<収録日>/<参加者ID>/ \
    --output-dir dataset/exist_label/
```

処理後の出力ファイル名例：
```
301_MIC1_MIC2_MIC3_MIC4_HPF.wav
```

> **注意：ラベルなしデータについて**
> Stage 2 の学習には `--unlabeled-root` で指定したディレクトリ以下の `.wav` ファイルをすべて再帰的に使用します（`rglob("*.wav")`）。
> 多チャンネルWAVはコードが自動的にチャンネル平均してモノラル化するため，`dataset/no_label/` を直接指定することも可能です。

---

### ステップ 2：アノテーションファイルの作成（ラベル付きデータのみ）

アノテーションは以下の形式のテキストファイルです（`.txt`）：

```
5.01    5.53    sw
50.44   51.14   sw
96.02   96.63   sw
```

各行は `<開始秒> <終了秒> <ラベル>` の形式です。

| ラベル | 意味 |
|--------|------|
| `sw` | 嚥下（swallowing） |
| `ch` | 咀嚼（chewing） |

`.wav` と同じファイル名の `.txt` を同じディレクトリに配置してください（例：`301.wav` と `301.txt`）。

---

### ステップ 3：学習用JSONリストの作成（ラベル付きデータのみ）

WAVファイルを10秒セグメントに分割し，学習用JSONを生成します。
`prepare/8_real_data_to_train.ipynb`（または `prepare/9_tsukagoshi_real_data_to_train.ipynb`）を実行してください。

```
BASE_DIR = dataset/exist_label/   # WAV・TXTファイルが置かれているディレクトリ
SEG_LEN  = 10.0 秒
OVERLAP  = 1.0 秒
TARGET_SR = 16000 Hz
```

実行後に以下が生成されます：
- `dataset/exist_label/divide_wav/` ─ 分割済みWAV
- `dataset/exist_label/divide_wav/exist_label_divided.json` ─ セグメントのJSONリスト

その後，train/validに分割したJSONを作成し，`config_FT_real.json` の以下のパスに指定します：

```json
"train_json": "./json/fit1/exist_label_train.json",
"val_json":   "./json/fit1/exist_label_valid.json",
"test_json":  "./json/fit1/exist_label_valid.json"
```

JSONリストの形式：

```json
[
  {
    "path": "dataset/exist_label/divide_wav/301_seg001.wav",
    "timestamps": {
      "swallowing": [[4.13, 4.78], [10.2, 10.9]],
      "chewing":    [],
      "speech":     [],
      "noise":      [],
      "mask":       []
    }
  }
]
```

`"mask"` に時間区間を指定すると，その区間は損失計算から除外されます。

---

## prepare/ ノートブック一覧

| ノートブック | 用途 | 対応する学習設定 |
|-------------|------|-----------------|
| `0_database_preproccessing.ipynb` | 制御環境データセットのダウンロード・整理 | モデル (1) |
| `0_Eat_behavior_dataset_w.ipynb` | 食事行動データセットのセグメント化・16kHzリサンプリング | モデル (1) |
| `0.1_db_combined_HPF.ipynb` | 多チャンネルWAVのHPFフィルタ処理・チャンネル合成 | モデル (1) |
| `2_eating_json.ipynb` | 制御環境データのJSON形式変換 | モデル (1) |
| `1.1_ATR_nhk.ipynb` | ATR503・NHK40 音声コーパスの前処理・VAD JSON生成 | モデル (2) |
| `1_commonvoice_json_.ipynb` | Common Voice 日本語データのVAD JSON生成 | モデル (2) |
| `8_real_data_to_train.ipynb` | ラベル付き実環境データを10秒セグメントに分割してJSON生成 | モデル (3) |
| `9_tsukagoshi_real_data_to_train.ipynb` | ラベルなし実環境データのセグメント化・JSON生成 | モデル (4) |
| `4_threshold_check.ipynb` | 検出閾値の精度-再現率曲線分析 | 評価・分析 |
| `7_test_results.ipynb` | 被験者ごと・IoU閾値ごとの詳細評価 | 評価・分析 |

---

## ファイル構成

```
swallowing_segmentation_meanteacher/
├── main_runner.py                    # Stage 1: 教師あり Fine-tuning のエントリポイント
├── semi_supervised_top.py            # Stage 2: Mean Teacher 半教師あり学習
├── training.py                       # 学習・評価ループ
├── models.py                         # EventDetector モデル（WavLM + GRU）
├── data_utils.py                     # データセットクラス・データローダ
├── metrics.py                        # IoU ベースのイベント検出評価指標
├── exp_config.py                     # 実験設定定数
├── ddp_utils.py                      # 分散学習（DDP）ユーティリティ
├── inference.py                      # テスト評価（main_runner.py から使用）
├── inference_wav.py                  # WAV ファイル推論（semi_supervised_top.py から使用）
├── config_FT_real.json               # Stage 1 設定（lr=1e-7，バランス重み付き）
├── config_FT_real_2.json             # Stage 1 設定バリアント（lr=1e-8）
├── config_FT_meanteacher_real.json   # Stage 2 設定（α=0.5 バリアント）
├── config_FT_meanteacher_real_2.json # Stage 2 設定（α=0.1 バリアント）
├── config_FT_meanteacher_real_3.json # Stage 2 設定（α=0.01 バリアント）
└── prepare/                          # データセット準備ノートブック群
    ├── 0_database_preproccessing.ipynb     # 制御環境データのダウンロード
    ├── 0_Eat_behavior_dataset_w.ipynb      # 食事行動データセットの前処理
    ├── 0.1_db_combined_HPF.ipynb           # 多ch合成・HPFフィルタ
    ├── 2_eating_json.ipynb                 # 制御環境データのJSON変換
    ├── 1.1_ATR_nhk.ipynb                   # ATR503/NHK40 音声コーパス前処理
    ├── 1_commonvoice_json_.ipynb           # Common Voice JSON生成
    ├── 8_real_data_to_train.ipynb          # ラベル付き実環境データのJSON生成
    ├── 9_tsukagoshi_real_data_to_train.ipynb # ラベルなし実環境データのJSON生成
    ├── 4_threshold_check.ipynb             # 閾値分析
    └── 7_test_results.ipynb                # テスト結果評価
```

---

## 学習手順

### 必要ライブラリ

```bash
pip install torch torchaudio transformers tqdm audiomentations soundfile scipy
```

### Stage 1: 教師あり Fine-tuning（`main_runner.py`）

データ準備が完了したら，設定ファイルの JSON パスを実際のパスに更新して実行します。

```bash
# 学習（モデル (3)：+ ラベル付き実環境データ）
python main_runner.py --config config_FT_real.json

# テストのみ
python main_runner.py --config config_FT_real.json --test

# チェックポイントから Fine-tuning 継続
python main_runner.py --config config_FT_real.json --finetune
```

**主要なコマンドラインオプション：**

| オプション | 説明 |
|-----------|------|
| `--config` | JSON 設定ファイルのパス（デフォルト：`config_FT_real.json`） |
| `--test` | テスト評価のみ実施（学習なし） |
| `--finetune` | 既存チェックポイントから patience をリセットして継続学習 |
| `--data-workers` | DataLoader のワーカー数上限（デフォルト：0） |

### Stage 2: Mean Teacher 半教師あり適応（`semi_supervised_top.py`）

Stage 1 で得たモデルを起点に，ラベルなし実環境データで適応します。

```bash
python semi_supervised_top.py \
    --config config_FT_meanteacher_real.json \
    --unlabeled-root dataset/no_label/ \
    --exist-label-dir dataset/exist_label/ \
    --ema-decay 0.999 \
    --confidence-threshold 0.5 \
    --w-unsup 1.0 \
    --epochs 300 \
    --lr 1e-7
```

**主要なコマンドラインオプション：**

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--config` | `config_FT_meanteacher_real.json` | JSON 設定ファイル |
| `--unlabeled-root` | `prepare/downloaded_folder/behavior/wav_sync` | ラベルなしWAVディレクトリ（再帰検索） |
| `--exist-label-dir` | `dataset/exist_label` | ラベル付きデータディレクトリ（検証用） |
| `--ema-decay` | `0.999` | Teacher モデルの EMA 減衰率（論文値：0.999） |
| `--confidence-threshold` | `0.9999` | 擬似ラベルの採用信頼度閾値（論文値：0.5） |
| `--w-unsup` | `1.0` | 教師なし損失の重み α（論文値：1.0） |
| `--epochs` | `100` | 学習エポック数 |
| `--lr` | `1e-7` | 学習率 |
| `--real-eval` | - | 検証時にラベル付き実データで評価する |

**WAVファイルへの推論：**

```bash
python semi_supervised_top.py \
    --inference-wav /path/to/audio.wav \
    --inference-threshold 0.5
```

### マルチ GPU 学習（DDP）

```bash
# Stage 1
torchrun --nproc_per_node=GPU数 main_runner.py --config config_FT_real.json

# Stage 2
torchrun --nproc_per_node=GPU数 semi_supervised_top.py --config config_FT_meanteacher_real.json
```

---

## 手法

### Stage 1: 教師あり Fine-tuning

モデル (1)〜(3) は教師あり学習で学習します：
- **(1)** 制御環境録音のみで学習
- **(2)** + Common Voice 約30時間の音声データで拡張学習
- **(3)** + ラベル付き実環境データ（約1時間）で Fine-tuning

### Stage 2: Mean Teacher 半教師あり適応

ラベルなし実環境データ（約85時間）を Mean Teacher フレームワークで活用します。

**EMA 更新（Teacher モデルの更新）：**
```
θ_T^(k) = β · θ_T^(k-1) + (1 - β) · θ_S^(k)
```

**信頼度マスク付き教師なし損失：**
```
L_unsup = (1/N) Σ m_t · BCE(p_t, p̂_t)
m_t = 1 if p_t ≥ τ, else 0
```

**総損失：**
```
L = L_sup + α · L_unsup
```

**ハイパーパラメータ（論文設定値）：**

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| β | 0.999 | EMA 減衰率 |
| τ | 0.5 | 擬似ラベル信頼度閾値 |
| α | 1.0 | 教師なし損失の重み |
| lr | 1×10⁻⁷ | Adam 学習率 |

---

## モデルアーキテクチャ

```
生波形（16kHz）
    ↓
WavLM Base+（microsoft/wavlm-base）
    ↓ [特徴抽出器を凍結，Transformer 層を Fine-tuning]
フレームレベル埋め込み（768次元，ホップ20ms）
    ↓
GRU（時系列モデリング）
    ↓
Linear + Sigmoid
    ↓
フレームレベル2クラス予測（嚥下 / その他）
```

---

## 評価指標

IoU 閾値を用いたイベントベースの指標を使用します。予測イベントは，Ground Truth との時間的重なりが IoU 閾値を超えた場合に正解とみなします。

主評価指標：**F1（IoU=0.1）**（検出重視，境界の許容度が高い設定）

---

## 引用

```
@inproceedings{tsukagoshi2026ncsp,
  title={Semi-Supervised Real-World Adaptation for Swallowing Sound Detection},
  author={Tsukagoshi, Toshihiro and Nishida, Masafumi and Nishimura, Masafumi},
  booktitle={Proceedings of NCSP 2026},
  year={2026}
}
```

---

## 関連研究

- SSL-based chewing and swallowing detection using multiple skin-contact microphones (APSIPA ASC 2024)
- Simultaneous speech and eating behavior recognition using data augmentation and two-stage fine-tuning (Sensors 2025)
- Swallowing sound segmentation using self-supervised learning-based features (IEEE GCCE 2025)
