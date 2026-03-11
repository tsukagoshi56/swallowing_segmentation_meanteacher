"""
Experiment configuration constants
"""
import sys
import logging
import torch
from typing import Dict, Any, List

# ロガーの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 省略定数
EPS = sys.float_info.epsilon
IOU_THRESHOLDS = [EPS] + [i * 0.1 for i in range(1, 11)]
LABEL_SHORT = {"chewing": "ch", "swallowing": "sw", "speech": "sp"}

# --- 基本設定 ---
BASE_CONFIG: Dict[str, Any] = {
    # データパス
    "train_json": "./json/fit1/merged_train.json",
    "val_json":   "./json/fit1/merged_val.json",
    "test_json":  "./json/fit1/merged_test.json",
    "base_save_dir": f"./result_1st_fine-tuning",  # 保存先ディレクトリ
    # 処理
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',  # DDP設定で更新される
    "sr": 16000,
    "ssl_hop_length": 320,  # 16 kHzで20msに相当
    # 訓練
    "batch_size": 4,
    "num_epochs": 150,
    "lr": 1e-5,
    "optimizer": "adam",
    "momentum": 0.9,
    "early_stopping_patience": 30,
    # 学習率スケジューリング
    "lr_scheduler": "cosine",           # "cosine", "step", "linear", "plateau", "none"
    "lr_warmup_epochs": 5,              # Warm-up期間のエポック数
    "lr_min_factor": 0.01,              # 最小学習率 = lr * lr_min_factor
    "lr_decay_rate": 0.75,               # 学習率減衰率 (0.5→0.8でさらに緩やかに)
    "lr_decay_epochs": [50, 100],       # Stepスケジューラ用の減衰エポック
    "lr_patience": 10,                   # Plateauスケジューラの忍耐値 (2→3でより長く待つ)
    "lr_threshold": 0.001,              # Plateauスケジューラの閾値 (0.0001→0.001でより寛容に)
    # モデル/タスク
    "classes": ["chewing", "swallowing", "speech"],
    # デフォルトSSL設定
    "default_ssl_model_name": "microsoft/wavlm-base-plus",
    "default_freeze_fe": True,
    "default_freeze_transformer": False,
    # 推論
    "threshold": 0.5,
}

# --- 実験定義 ---
EXPERIMENT_GROUP = "exp"
EXPERIMENTS: Dict[str, List[Dict[str, Any]]] = {
    "exp": [
        {"name": "WavLM-base+GRU",  "ssl_model_name": "microsoft/wavlm-base-plus", "feature_type": "raw", "architecture": "gru"},
        {"name": "WavLM-base+GRU_step", "ssl_model_name": "microsoft/wavlm-base-plus", "feature_type": "raw", "architecture": "gru", "lr_scheduler": "step"},
        {"name": "WavLM-base+GRU_linear", "ssl_model_name": "microsoft/wavlm-base-plus", "feature_type": "raw", "architecture": "gru", "lr_scheduler": "linear"},
        {"name": "WavLM-base+GRU_none", "ssl_model_name": "microsoft/wavlm-base-plus", "feature_type": "raw", "architecture": "gru", "lr_scheduler": "none"},
        {"name": "WavLM-base+GRU_plateau", "ssl_model_name": "microsoft/wavlm-base-plus", "feature_type": "raw", "architecture": "gru", "lr_scheduler": "plateau", "lr_patience": 5, "lr_decay_rate": 0.8, "lr_threshold": 0.001},
    ],
    "class_comparison": [
        {"name": "2class_WavLM+GRU", "ssl_model_name": "microsoft/wavlm-base", "feature_type": "raw", "architecture": "gru", "classes": ["swallowing", "others"]},
        {"name": "3class_WavLM+GRU", "ssl_model_name": "microsoft/wavlm-base", "feature_type": "raw", "architecture": "gru", "classes": ["swallowing", "chewing", "others"]},
        {"name": "4class_WavLM+GRU", "ssl_model_name": "microsoft/wavlm-base", "feature_type": "raw", "architecture": "gru", "classes": ["swallowing", "chewing", "speech", "others"]},
    ]
}
