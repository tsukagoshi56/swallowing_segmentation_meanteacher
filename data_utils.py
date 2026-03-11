"""
Data utilities: Dataset class and collate function.
"""
import os
import math
import random
import sys
from pathlib import Path

# 現在のディレクトリをPythonパスに追加
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import logging
from typing import List, Dict, Tuple, Optional, Union, Any

try:
    import torch
    import torchaudio
    import numpy as np
    from torch import nn
    from torch.utils.data import Dataset
    from torch.utils.data.distributed import DistributedSampler
except ImportError as e:
    print(f"必要なライブラリのインポートに失敗しました: {e}")
    sys.exit(1)

# 同じディレクトリ内のモジュールのインポート
try:
    from ddp_utils import is_main_process
except ImportError as e:
    print(f"ddp_utils モジュールのインポートに失敗しました: {e}")
    # エラーは報告するが終了はしない（テスト実行のため）
    def is_main_process(rank): return rank == 0


# ロガーの設定
logger = logging.getLogger(__name__)

class SoundEventDataset(Dataset):
    def __init__(self,
                 annotations: List[Dict[str, Any]],
                 config: Dict[str, Any],
                 feature_type: str = "raw",
                 dataset_frac: float = 1.0,
                 local_rank: int = 0):
        super().__init__()
        self.sr = config["sr"]
        self.hop_length = config["ssl_hop_length"]
        self.hop_time = self.hop_length / self.sr
        self.classes = config["classes"]
        self.num_classes = len(self.classes)
        self.feature_type = feature_type
        self.local_rank = local_rank
        
        # blankクラス使用設定を保存
        self.use_blank_class = config.get("use_blank_class", False)
        
        # ★★★ クラスマッピング設定を保存 ★★★
        self.class_mapping = config.get("class_mapping", {})

        if not (0.0 < dataset_frac <= 1.0):
            raise ValueError("dataset_frac must be between 0 (exclusive) and 1 (inclusive)")

        ann = annotations.copy()
        if dataset_frac < 1.0:
            num_samples = int(len(ann) * dataset_frac)
            if is_main_process(self.local_rank):
                random.shuffle(ann)
            self.annotations = ann[:num_samples]
            if is_main_process(self.local_rank):
                logger.info(f"Using {num_samples}/{len(annotations)} samples ({dataset_frac:.1%})")
        else:
            self.annotations = ann

        self.feature_transform = None
        if feature_type == "mfcc":
            self.feature_transform = torchaudio.transforms.MFCC(
                sample_rate=self.sr, n_mfcc=40, log_mels=True,
                melkwargs={'n_fft': 400, 'hop_length': self.hop_length, 'n_mels': 64, 'center': False}
            )
            self.feature_dim = 40
        elif feature_type == "mel":
            self.feature_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr, n_mels=128, n_fft=1024, hop_length=self.hop_length, center=False
            )
            self.feature_dim = 128
        elif feature_type == "raw":
            self.feature_dim = 1
        else:
            raise ValueError(f"Unsupported feature_type: {feature_type}")

    def __len__(self) -> int:
        return len(self.annotations)

    def _map_class_name(self, original_class: str) -> Optional[str]:
        """Map original class names to target class names based on configuration"""
        # Get class mapping from config if available
        if hasattr(self, 'class_mapping') and self.class_mapping and original_class in self.class_mapping:
            return self.class_mapping[original_class]
        
        # Default mapping for 2-class, 3-class, 4-class scenarios
        if len(self.classes) == 2:  # 2-class: swallowing vs others
            if original_class == "swallowing":
                return "swallowing"
            else:
                return "others"
        elif len(self.classes) == 3:  # 3-class: swallowing, chewing, others
            if original_class in ["swallowing", "chewing"]:
                return original_class
            else:
                return "others"
        elif len(self.classes) == 4:  # 4-class: swallowing, chewing, speech, others
            if original_class in ["swallowing", "chewing", "speech"]:
                return original_class
            else:
                return "others"
        
        # Default: return original class if in target classes
        return original_class if original_class in self.classes else "others"

    def _make_label_matrix(self,
                           timestamps: Dict[str, List[Tuple[float, float]]],
                           num_frames: int,
                           audio_path: str = "") -> torch.Tensor:
        # 動的クラス数対応のラベル作成（クラスインデックス形式）
        # デフォルトはothersクラスのインデックス（背景音）
        if "others" in self.classes:
            others_idx = self.classes.index("others")
            label_vector = torch.full((num_frames,), others_idx, dtype=torch.long)
        else:
            # othersクラスがない場合は従来通りクラス0
            label_vector = torch.zeros(num_frames, dtype=torch.long)
        
        # 各クラスの時間区間をクラスインデックスとして設定
        for original_class, intervals in timestamps.items():
            # Skip mask annotations
            if original_class == "mask":
                continue
                
            # Map original class to target class
            target_class = self._map_class_name(original_class)
            if target_class is None or target_class not in self.classes:
                continue
                
            class_idx = self.classes.index(target_class)
            
            for start_time, end_time in intervals:
                start_frame = max(0, math.floor(start_time / self.hop_time))
                end_frame = min(num_frames, math.ceil(end_time / self.hop_time))
                end_frame = max(start_frame, end_frame)
                if start_frame < num_frames:
                    label_vector[start_frame:end_frame] = class_idx
        return label_vector

    # ★★★ 変更点 1: __getitem__ の戻り値の型ヒントと本体を修正 ★★★
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        item = self.annotations[idx]
        audio_path = item["path"]

        try:
            wav, sr0 = torchaudio.load(audio_path)
        except Exception as e:
            if is_main_process(self.local_rank):
                logger.error(f"Error loading audio file {audio_path}: {e}")
            # エラー時も正しい数のタプルを返せるようにダミーデータを作成
            dummy_wav = torch.zeros(1, dtype=torch.float32)
            dummy_labels = torch.zeros(1, dtype=torch.long)  # 動的クラス数分類用のクラスインデックス
            dummy_mask = torch.zeros(1, dtype=torch.float32)
            return dummy_wav, dummy_labels, dummy_mask, audio_path

        if sr0 != self.sr:
            resampler = torchaudio.transforms.Resample(sr0, self.sr)
            wav = resampler(wav)

        if wav.dim() > 1 and wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0)
        wav = wav.squeeze()

        num_samples = wav.shape[-1]
        num_frames = num_samples // self.hop_length

        labels = self._make_label_matrix(item["timestamps"], num_frames, audio_path)

        # ★★★ 変更点 2: マスクテンソルを生成する処理を追加 ★★★
        # デフォルトは1（損失を計算する）で初期化
        mask_tensor = torch.ones(num_frames, dtype=torch.float32)
        if "mask" in item.get("timestamps", {}):
            for start_time, end_time in item["timestamps"]["mask"]:
                start_frame = max(0, math.floor(start_time / self.hop_time))
                end_frame = min(num_frames, math.ceil(end_time / self.hop_time))
                end_frame = max(start_frame, end_frame)
                if start_frame < num_frames:
                    # マスク区間の損失を0にする
                    mask_tensor[start_frame:end_frame] = 0.0
        
        # マスクの長さをラベルの長さに合わせる
        min_len_for_mask = min(labels.shape[0], mask_tensor.shape[0])
        labels = labels[:min_len_for_mask]
        mask_tensor = mask_tensor[:min_len_for_mask]

        if self.feature_transform:
            if wav.dim() == 0:
                wav = wav.unsqueeze(0)
            
            # (B, T) or (B, C, T) の形状を期待する変換のためにバッチ次元を追加
            features = self.feature_transform(wav.unsqueeze(0))
            features = features.squeeze(0).permute(1, 0) # (T, F) に変換

            feat_len = features.shape[0]
            label_len = labels.shape[0]
            
            min_len = min(feat_len, label_len)
            features = features[:min_len, :]
            labels = labels[:min_len]
            mask_tensor = mask_tensor[:min_len] # マスクも長さを合わせる
            
            return features, labels, mask_tensor, audio_path
        else:
            # raw audio
            return wav, labels, mask_tensor, audio_path

# ★★★ 変更点 3: pad_collate のシグネチャと処理を修正 ★★★
def pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]],
                config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    
    # バッチにNoneが含まれていないかチェック
    batch = [b for b in batch if b is not None and all(item is not None for item in b)]
    if not batch:
        # 空のバッチの場合はダミーデータを返す
        dummy_feat = torch.zeros(1, 1, dtype=torch.float32)
        dummy_label = torch.zeros(1, 1, dtype=torch.long)  # 3クラス分類用のクラスインデックス
        dummy_mask = torch.zeros(1, 1, dtype=torch.float32)
        return dummy_feat, dummy_label, dummy_mask, []

    features_list, labels_list, masks_list, paths = zip(*batch)

    is_raw_audio = features_list[0].dim() == 1

    if is_raw_audio:  # waveforms
        padded_features = torch.nn.utils.rnn.pad_sequence(features_list, batch_first=True, padding_value=0.0)
    else:  # feature maps (T, F)
        padded_features = torch.nn.utils.rnn.pad_sequence(features_list, batch_first=True, padding_value=0.0)
        padded_features = padded_features.permute(0, 2, 1) # (B, T, F) -> (B, F, T) for model

    # ラベルとマスクのパディング
    # labels are (T), masks are (T) - 3クラス分類用のクラスインデックス
    # Use appropriate padding value for labels based on class configuration
    classes = config.get("classes", [])
    if "others" in classes:
        label_pad_value = classes.index("others")
    else:
        label_pad_value = 0
    
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=label_pad_value)
    masks_padded = torch.nn.utils.rnn.pad_sequence(masks_list, batch_first=True, padding_value=0.0)

    # 特徴量とラベル/マスクのフレーム長を合わせる
    # モデルの出力フレーム長に合わせることが一般的
    # raw audio -> model -> features の変換後のフレーム数を考慮
    if is_raw_audio:
        # hop_lengthはconfigから取得
        hop_length = config.get("ssl_hop_length", 320)
        max_feat_len = padded_features.shape[1]
        max_label_len = max_feat_len // hop_length
    else:
        # 特徴量抽出済みの場合、時間軸はすでにフレームになっている
        max_label_len = padded_features.shape[2] # (B, F, T)

    current_label_len = labels_padded.shape[1]  # (B, T) - 3クラス分類用
    if current_label_len > max_label_len:
        labels_padded = labels_padded[..., :max_label_len]
        masks_padded = masks_padded[..., :max_label_len]
    elif current_label_len < max_label_len:
        pad_size = max_label_len - current_label_len
        # For 2-class classification (swallowing vs others), pad with class 1 (others)
        classes = config.get("classes", [])
        if "others" in classes:
            others_idx = classes.index("others")
            labels_padded = nn.functional.pad(labels_padded, (0, pad_size), value=others_idx)
            # print(f"DEBUG: Padding with others class {others_idx}, classes: {classes}")
        else:
            labels_padded = nn.functional.pad(labels_padded, (0, pad_size))
            # print(f"DEBUG: Padding with default 0, classes: {classes}")
        masks_padded = nn.functional.pad(masks_padded, (0, pad_size))

    return padded_features, labels_padded, masks_padded, list(paths)