# Semi-Supervised Real-World Adaptation for Swallowing Sound Detection

Implementation of the paper:

> **"Semi-Supervised Real-World Adaptation for Swallowing Sound Detection"**
> Toshihiro Tsukagoshi, Masafumi Nishida, Masafumi Nishimura
> NCSP 2026, Shizuoka University / Aichi Sangyo University

---

## Overview

This repository implements a two-stage semi-supervised learning framework for detecting swallowing sounds from skin-contact microphone recordings in real-world dining environments.

**Key idea:**
A WavLM + GRU model is first fine-tuned on a small labeled real-world dataset (Stage 1), then further adapted using a large unlabeled real-world dataset via the Mean Teacher framework (Stage 2).

### Results (Event-based F1 at IoU=0.1)

| Model | Pizza | Apple | Cracker | Overall |
|-------|-------|-------|---------|---------|
| (1) Controlled recordings only | 0.451 | 0.450 | 0.609 | 0.470 |
| (2) + Speech data (30h) | 0.785 | 0.684 | 0.789 | 0.756 |
| (3) + Labeled real-world data (1h) | 0.932 | 0.838 | 0.897 | 0.899 |
| **(4) + Unlabeled real-world data (85h) [Proposed]** | **0.947** | **0.903** | **0.950** | **0.933** |

---

## Method

### Stage 1: Supervised Fine-tuning (`main_runner.py`)

Fine-tunes the model on labeled real-world data.
The architecture consists of a WavLM Base+ feature extractor and a GRU-based temporal module for frame-level binary classification (swallowing vs. others).

**Training configurations (matching paper):**

- Model (1): Controlled-environment recordings only
- Model (2): + Speech data augmentation (Common Voice ~30h)
- Model (3): + Labeled real-world data (~1h)

### Stage 2: Mean Teacher Semi-supervised Adaptation (`semi_supervised_top.py`)

Leverages ~85h of unlabeled real-world recordings via the Mean Teacher framework.

**EMA update:**
```
θ_T^(k) = β · θ_T^(k-1) + (1 - β) · θ_S^(k)
```

**Confidence-masked unsupervised loss:**
```
L_unsup = (1/N) Σ m_t · BCE(p_t, p̂_t)
m_t = 1 if p_t ≥ τ, else 0
```

**Total loss:**
```
L = L_sup + α · L_unsup
```

**Hyperparameters (paper values):**
- EMA decay β = 0.999
- Confidence threshold τ = 0.5
- Unsupervised loss weight α = 1.0
- Learning rate = 1×10⁻⁷ (Adam)

---

## File Structure

```
swallowing_segmentation_meanteacher/
├── main_runner.py                    # Stage 1: Supervised fine-tuning entry point
├── semi_supervised_top.py            # Stage 2: Mean Teacher semi-supervised training
├── training.py                       # Training/evaluation loop functions
├── models.py                         # EventDetector model (WavLM + GRU)
├── data_utils.py                     # SoundEventDataset and data loading utilities
├── metrics.py                        # Event-based precision/recall/F1 (IoU-based)
├── exp_config.py                     # Experiment configuration constants
├── ddp_utils.py                      # Distributed Data Parallel utilities
├── inference.py                      # Test evaluation functions (used by main_runner.py)
├── inference_wav.py                  # WAV-level inference functions (used by semi_supervised_top.py)
├── config_FT_real.json               # Config for Stage 1 (lr=1e-7, balanced weight)
├── config_FT_real_2.json             # Config for Stage 1 variant (lr=1e-8)
├── config_FT_meanteacher_real.json   # Config for Stage 2 (α=0.5 variant)
├── config_FT_meanteacher_real_2.json # Config for Stage 2 (α=0.1 variant)
└── config_FT_meanteacher_real_3.json # Config for Stage 2 (α=0.01 variant)
```

---

## Usage

### Prerequisites

```bash
pip install torch torchaudio transformers tqdm audiomentations
```

### Stage 1: Supervised Fine-tuning

Prepare annotation JSON files for labeled data and update paths in the config file.

```bash
# Training (model 3: + labeled real-world data)
python main_runner.py --config config_FT_real.json

# Test only
python main_runner.py --config config_FT_real.json --test

# Fine-tuning from existing checkpoint
python main_runner.py --config config_FT_real.json --finetune
```

### Stage 2: Mean Teacher Semi-supervised Adaptation

```bash
# Training with unlabeled real-world data
python semi_supervised_top.py \
    --config config_FT_meanteacher_real.json \
    --unlabeled-root /path/to/unlabeled/wav \
    --ema-decay 0.999 \
    --confidence-threshold 0.5 \
    --w-unsup 1.0 \
    --epochs 300 \
    --lr 1e-7

# Inference on audio folder
python semi_supervised_top.py \
    --inference-wav /path/to/audio.wav \
    --inference-threshold 0.5
```

### Multi-GPU (DDP) Training

```bash
# Stage 1
torchrun --nproc_per_node=NUM_GPUS main_runner.py --config config_FT_real.json

# Stage 2
torchrun --nproc_per_node=NUM_GPUS semi_supervised_top.py --config config_FT_meanteacher_real.json
```

---

## Data Format

Annotation JSON files follow this format:

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

Class mapping for binary classification (swallowing vs. others):
- `"swallowing"` → `"swallowing"`
- `"chewing"`, `"speech"`, `"background"`, `"blank"` → `"others"`

---

## Model Architecture

```
Raw waveform (16kHz)
    ↓
WavLM Base+ (microsoft/wavlm-base)
    ↓ [freeze feature extractor, fine-tune transformer layers]
Frame-level embeddings (768-dim, hop=20ms)
    ↓
GRU (temporal modeling)
    ↓
Linear + Sigmoid
    ↓
Frame-level binary predictions (swallowing / others)
```

---

## Evaluation

Event-based metrics using IoU thresholds. A predicted event is correct if temporal overlap with ground truth exceeds the IoU threshold.

Primary metric: **F1 at IoU=0.1** (detection-oriented, lenient boundary matching)

---

## Citation

```
@inproceedings{tsukagoshi2026ncsp,
  title={Semi-Supervised Real-World Adaptation for Swallowing Sound Detection},
  author={Tsukagoshi, Toshihiro and Nishida, Masafumi and Nishimura, Masafumi},
  booktitle={Proceedings of NCSP 2026},
  year={2026}
}
```

---

## Related Work

- [tsukagoshi2024ssl] SSL-based chewing and swallowing detection using multiple skin-contact microphones (APSIPA ASC 2024)
- [tsukagoshi2025simultaneous] Simultaneous speech and eating behavior recognition using data augmentation and two-stage fine-tuning (Sensors 2025)
- [tsukagoshi2025gcce] Swallowing sound segmentation using self-supervised learning-based features (IEEE GCCE 2025)
