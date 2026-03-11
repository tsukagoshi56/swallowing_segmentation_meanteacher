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

## 手法

### Stage 1: 教師あり Fine-tuning（`main_runner.py`）

ラベル付き実環境データを用いてモデルを Fine-tuning します。
アーキテクチャは WavLM Base+ による特徴抽出と，GRU によるフレームレベルの2クラス分類（嚥下 vs. その他）で構成されます。

**論文における学習構成：**

- モデル (1): 制御環境録音のみ
- モデル (2): + 音声データ拡張（Common Voice 約30時間）
- モデル (3): + ラベル付き実環境データ（約1時間）

### Stage 2: Mean Teacher 半教師あり適応（`semi_supervised_top.py`）

約85時間のラベルなし実環境録音を Mean Teacher フレームワークで活用します。

**EMA 更新：**
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
- EMA 減衰率 β = 0.999
- 信頼度閾値 τ = 0.5
- 教師なし損失重み α = 1.0
- 学習率 = 1×10⁻⁷（Adam）

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
└── config_FT_meanteacher_real_3.json # Stage 2 設定（α=0.01 バリアント）
```

---

## 使い方

### 必要ライブラリ

```bash
pip install torch torchaudio transformers tqdm audiomentations
```

### Stage 1: 教師あり Fine-tuning

アノテーション JSON ファイルを準備し，設定ファイル内のパスを適宜更新してください。

```bash
# 学習（モデル (3)：+ ラベル付き実環境データ）
python main_runner.py --config config_FT_real.json

# テストのみ
python main_runner.py --config config_FT_real.json --test

# チェックポイントから Fine-tuning 継続
python main_runner.py --config config_FT_real.json --finetune
```

### Stage 2: Mean Teacher 半教師あり適応

```bash
# ラベルなし実環境データを用いた学習
python semi_supervised_top.py \
    --config config_FT_meanteacher_real.json \
    --unlabeled-root /path/to/unlabeled/wav \
    --ema-decay 0.999 \
    --confidence-threshold 0.5 \
    --w-unsup 1.0 \
    --epochs 300 \
    --lr 1e-7

# WAV ファイルへの推論
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

## データ形式

アノテーション JSON ファイルの形式：

```json
[
  {
    "wav": "/path/to/audio.wav",
    "events": [
      {"label": "swallowing", "start": 1.2, "end": 1.8},
      {"label": "chewing", "start": 2.0, "end": 2.5}
    ]
  }
]
```

2クラス分類のクラスマッピング（嚥下 vs. その他）：
- `"swallowing"` → `"swallowing"`
- `"chewing"`, `"speech"`, `"background"`, `"blank"` → `"others"`

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
