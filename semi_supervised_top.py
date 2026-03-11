"""Semi-supervised fine-tuning with on-the-fly pseudo labels."""

from __future__ import annotations



import argparse

import json

import io

import logging

import random

import sys

import time

from dataclasses import dataclass

from itertools import cycle

from pathlib import Path

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch.nn.functional as F

import numpy as np

import torch

import torchaudio

from torch import nn

from torch.utils.data import DataLoader, Dataset



try:

    from audiomentations import Compose, Gain, AddGaussianNoise

    AUDIO_AUG_AVAILABLE = True

except ImportError:

    Compose = Gain = AddGaussianNoise = None  # type: ignore

    AUDIO_AUG_AVAILABLE = False



SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:

    sys.path.insert(0, str(SCRIPT_DIR))



from data_utils import SoundEventDataset, pad_collate

from models import EventDetector

from training import evaluate_epoch

from metrics import (

    calculate_event_metrics,

    calculate_event_metrics_by_class_and_food,

    collect_matched_iou,

    collect_raw_iou,

    collect_raw_iou_by_food,

)

from tqdm.auto import tqdm



logger = logging.getLogger("semi_supervised")



ORIGINAL_STDOUT = sys.stdout

ORIGINAL_STDERR = sys.stderr





try:

    import inference_wav as inference_wav_module

except ImportError:

    inference_wav_module = None  # type: ignore





def safe_torch_load(path: Path,

                    map_location: Optional[torch.device] = None,

                    allow_pickle_fallback: bool = True) -> Dict:

    load_kwargs = {"map_location": map_location}

    if allow_pickle_fallback:

        load_kwargs["weights_only"] = False

    try:

        return torch.load(path, **load_kwargs)

    except TypeError:

        load_kwargs.pop("weights_only", None)

        return torch.load(path, **load_kwargs)





def format_seconds(seconds: float) -> str:

    seconds = max(0, int(round(seconds)))

    minutes, secs = divmod(seconds, 60)

    hours, minutes = divmod(minutes, 60)

    if hours:

        return f"{hours}h{minutes:02d}m{secs:02d}s"

    if minutes:

        return f"{minutes}m{secs:02d}s"

    return f"{secs}s"





def seed_everything(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)





def load_json(path: Path) -> List[Dict]:

    with open(path, "r", encoding="utf-8") as handle:

        return json.load(handle)





def load_model_weights(model: nn.Module,

                       checkpoint_path: Path,

                       device: torch.device) -> bool:

    """Load model weights from a checkpoint path, handling raw or wrapped state dicts."""

    if not checkpoint_path.exists():

        return False

    checkpoint = safe_torch_load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    else:

        state_dict = checkpoint

    model.load_state_dict(state_dict)

    return True





def resolve_relative_path(cfg_path: Path, value: str) -> Path:

    if value is None:

        return cfg_path.parent

    return (cfg_path.parent / value).resolve()





def _load_config_manifest(cfg_path: Path) -> Dict:

    with open(cfg_path, "r", encoding="utf-8") as handle:

        return json.load(handle)





def list_available_experiments(cfg_path: Path) -> Dict[str, Dict]:

    raw = _load_config_manifest(cfg_path)

    experiments = raw.get("experiments", {})

    if experiments is None:

        return {}

    if not isinstance(experiments, dict):

        raise TypeError("Expected 'experiments' to be a dictionary in configuration file.")

    return experiments





def load_experiment_config(cfg_path: Path, experiment_key: Optional[str]) -> Dict:

    raw = _load_config_manifest(cfg_path)

    if "base_config" not in raw:

        raise KeyError("Config file must contain 'base_config'.")

    base = raw["base_config"].copy()

    experiments = raw.get("experiments", {}) or {}

    if not isinstance(experiments, dict):

        raise TypeError("Expected 'experiments' to be a dictionary in configuration file.")



    exp_cfg: Dict = {}

    if experiment_key is not None:

        if experiment_key not in experiments:

            available = ", ".join(sorted(experiments.keys()))

            raise KeyError(f"Experiment '{experiment_key}' not found in config. Available experiments: {available}")

        exp_cfg = experiments[experiment_key]

        if not isinstance(exp_cfg, dict):

            raise TypeError(f"Experiment '{experiment_key}' must be defined as a dictionary.")

        merged = base

        merged.update(exp_cfg)

    else:

        merged = base

    # Resolve common paths relative to config file

    for key in ("train_json", "val_json", "test_json"):

        if key in merged:

            merged[key] = str(resolve_relative_path(cfg_path, merged[key]))

    if "base_save_dir" in merged:

        merged["base_save_dir"] = str(resolve_relative_path(cfg_path, merged["base_save_dir"]))

    if exp_cfg and "class_mapping" not in merged and exp_cfg.get("class_mapping"):

        merged["class_mapping"] = exp_cfg["class_mapping"]

    if "name" not in merged:

        if exp_cfg.get("name"):

            merged["name"] = exp_cfg["name"]

        elif experiment_key:

            merged["name"] = experiment_key

        else:

            merged["name"] = raw.get("experiment_name", "default")

    merged["experiment_key"] = experiment_key

    merged["_exp_base_save_dir_overridden"] = bool(exp_cfg.get("base_save_dir")) if exp_cfg else False

    return merged





def select_experiment_keys(selection: Optional[str],

                           available_keys: Sequence[str]) -> List[Optional[str]]:

    if not available_keys:

        if selection and selection.strip():

            tokens = [tok.strip() for tok in selection.split(",") if tok.strip()]

            if tokens and tokens != ["all"]:

                raise KeyError("No experiments are defined in the configuration file.")

        return [None]



    if selection is None or not selection.strip() or selection.strip().lower() == "all":

        return list(available_keys)



    chosen: List[Optional[str]] = []

    seen = set()

    for token in selection.split(","):

        key = token.strip()

        if not key:

            continue

        if key.lower() == "all":

            return list(available_keys)

        if key not in available_keys:

            available = ", ".join(sorted(available_keys))

            raise KeyError(f"Experiment '{key}' not found. Available experiments: {available}")

        if key not in seen:

            seen.add(key)

            chosen.append(key)



    return chosen or list(available_keys)





@dataclass

class UnlabeledSample:

    waveform: torch.Tensor

    valid_samples: int

    path: str





@dataclass

class SegmentSpec:

    path: str

    start_sec: float

    duration_sec: float

    orig_sr: int





class UnlabeledSegmentGenerator:

    def __init__(self,

                 root: Path,

                 target_sr: int,

                 min_window_sec: float,

                 max_window_sec: float,

                 overlap_sec: float,

                 seed: int = 42) -> None:

        self.root = Path(root)

        self.target_sr = target_sr

        self.min_window_sec = max(0.1, min_window_sec)

        self.max_window_sec = max(self.min_window_sec, max_window_sec)

        self.overlap_sec = max(0.0, overlap_sec)

        self.seed = seed

        self.file_infos: List[Dict[str, float]] = []



        paths = sorted(p for p in self.root.rglob("*.wav") if p.is_file())

        if not paths:

            raise FileNotFoundError(f"No wav files found under {self.root}")



        for wav_path in paths:

            try:

                info = torchaudio.info(str(wav_path))

                orig_sr = info.sample_rate

                num_frames = info.num_frames

            except Exception:

                try:

                    waveform, orig_sr = torchaudio.load(str(wav_path))

                    num_frames = waveform.shape[-1]

                except Exception as exc:

                    logger.warning(f"Failed to inspect {wav_path}: {exc}")

                    continue

            if orig_sr <= 0 or num_frames <= 0:

                logger.warning(f"Skipping empty file: {wav_path}")

                continue

            duration_sec = num_frames / float(orig_sr)

            if duration_sec <= 0.0:

                logger.warning(f"Skipping zero-duration file: {wav_path}")

                continue

            self.file_infos.append({

                "path": str(wav_path),

                "orig_sr": orig_sr,

                "duration_sec": duration_sec,

            })



        if not self.file_infos:

            raise FileNotFoundError(f"No valid wav files found under {self.root}")



    def generate_segments(self, epoch: int) -> List[SegmentSpec]:

        rng = random.Random(self.seed + epoch * 9973)

        segments: List[SegmentSpec] = []



        for info in self.file_infos:

            duration = info["duration_sec"]

            orig_sr = int(info["orig_sr"])

            path = info["path"]



            if duration <= 0.0:

                continue



            pos = 0.0

            while pos < duration:

                remain = duration - pos

                if remain <= 0.0:

                    break



                window_sec = rng.uniform(self.min_window_sec, self.max_window_sec)



                if window_sec > remain:

                    if remain >= self.min_window_sec or pos == 0.0:

                        window_sec = remain

                    else:

                        pos = max(0.0, duration - self.min_window_sec)

                        window_sec = duration - pos

                        remain = window_sec



                if window_sec <= 0.0:

                    break



                segments.append(SegmentSpec(

                    path=path,

                    start_sec=pos,

                    duration_sec=window_sec,

                    orig_sr=orig_sr,

                ))



                end = pos + window_sec

                if end >= duration:

                    break

                next_start = end - self.overlap_sec

                # Ensure strict progress to avoid infinite loops

                if next_start <= pos:

                    next_start = pos + max(window_sec * 0.5, 0.5)

                pos = next_start



        return segments





class UnlabeledWindowDataset(Dataset):

    def __init__(self,

                 segments: List[SegmentSpec],

                 target_sr: int,

                 min_window_sec: float) -> None:

        self.segments = segments

        self.target_sr = target_sr

        self.min_window_sec = min_window_sec

        self.default_length = max(1, int(round(self.min_window_sec * self.target_sr)))



    def __len__(self) -> int:

        return len(self.segments)



    def __getitem__(self, idx: int) -> UnlabeledSample:

        spec = self.segments[idx]

        start_frame = int(round(spec.start_sec * spec.orig_sr))

        num_frames = max(1, int(round(spec.duration_sec * spec.orig_sr)))



        try:

            waveform, sr = torchaudio.load(spec.path, frame_offset=start_frame, num_frames=num_frames)

        except Exception as exc:

            logger.warning(f"Failed to load segment from {spec.path}: {exc}")

            try:

                waveform, sr = torchaudio.load(spec.path)

                waveform = waveform[..., start_frame:start_frame + num_frames]

            except Exception as exc2:

                logger.error(f"Could not recover audio from {spec.path}: {exc2}")

                return UnlabeledSample(

                    waveform=torch.zeros(self.default_length, dtype=torch.float32),

                    valid_samples=0,

                    path=spec.path,

                )



        if waveform.numel() == 0:

            waveform = torch.zeros(1, num_frames, dtype=torch.float32)



        if sr != self.target_sr:

            try:

                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)

                waveform = resampler(waveform)

            except Exception as exc:

                logger.warning(f"Resample failed for {spec.path}: {exc}")

                waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)



        if waveform.dim() > 1 and waveform.size(0) > 1:

            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)



        valid_samples = waveform.shape[-1]

        if valid_samples == 0:

            waveform = torch.zeros(self.default_length, dtype=torch.float32)

            valid_samples = 0



        return UnlabeledSample(waveform=waveform, valid_samples=valid_samples, path=spec.path)





def collate_unlabeled(batch: List[UnlabeledSample]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:

    waveforms = [item.waveform for item in batch]

    lengths = torch.tensor([item.valid_samples for item in batch], dtype=torch.long)

    padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)

    paths = [item.path for item in batch]

    return padded, lengths, paths





def log_iou_metrics(metrics: Optional[Dict[float, Tuple[float, float, float]]], prefix: str) -> None:

    if not metrics:

        logger.info("%s IoU metrics unavailable", prefix)

        return



    for threshold in sorted(metrics.keys()):

        precision, recall, f1 = metrics[threshold]

        logger.info(

            "%s IoU > %.2f: P=%.3f, R=%.3f, F1=%.3f",

            prefix,

            threshold,

            precision,

            recall,

            f1,

        )





def determine_primary_event_class(config: Dict) -> Tuple[str, int]:

    """Select the primary positive class used for pseudo labeling."""

    classes = list(config.get("classes", []) or [])

    if not classes:

        raise ValueError("Configuration must define at least one class.")



    preferred = config.get("primary_event_class")

    if preferred and preferred in classes:

        return preferred, classes.index(preferred)



    if "swallowing" in classes:

        return "swallowing", classes.index("swallowing")



    non_other = [cls for cls in classes if cls != "others"]

    if non_other:

        target = non_other[0]

        return target, classes.index(target)



    # fallback to first class if all entries are 'others'

    return classes[0], 0





# ---------------------------------------------------------

# 2. Loss計算関数の変更 (One-Hot変換 + BCE)

# ---------------------------------------------------------

def masked_bce_loss(logits: torch.Tensor,

                    targets: torch.Tensor,

                    weights: Optional[torch.Tensor],

                    criterion: nn.Module) -> torch.Tensor:

    """

    BCEWithLogitsLoss用にターゲットをOne-Hot化して損失を計算する関数

    logits: [B, C, T]

    targets: [B, T] (Indices) or [B, C, T] (One-Hot)

    weights: [B, T] (Mask)

    """

    B, C, T_pred = logits.shape

    T_tgt = targets.size(-1)

    

    # 長さ合わせ

    min_len = min(T_pred, T_tgt)

    if weights is not None:

        T_weight = weights.size(-1)

        min_len = min(min_len, T_weight)

        

    if min_len <= 0:

        return torch.zeros((), device=logits.device)

        

    logits_slice = logits[..., :min_len]

    targets_slice = targets[..., :min_len]

    

    # ターゲットがインデックス([B, T])の場合、One-Hot([B, C, T])に変換

    if targets_slice.dim() == 2:

        # インデックスが範囲外でないかクリップ（念のため）

        targets_clamped = targets_slice.long().clamp(0, C - 1)

        targets_one_hot = F.one_hot(targets_clamped, num_classes=C).permute(0, 2, 1).float()

    else:

        targets_one_hot = targets_slice.float()



    # BCE Loss計算 (Pixel-wise / Frame-wise)

    loss_per_pixel = criterion(logits_slice, targets_one_hot) # [B, C, T]

    

    # クラス方向の平均または和を取る（BCEは各クラス独立なので通常は平均）

    loss_per_frame = loss_per_pixel.mean(dim=1) # [B, T]



    if weights is None:

        return loss_per_frame.mean()

        

    weights_slice = weights[..., :min_len] # [B, T]

    

    # マスク適用

    weighted_loss = loss_per_frame * weights_slice

    

    # 正規化（マスクの有効長で割る）

    denom = weights_slice.sum()

    if denom.item() == 0:

        return torch.zeros((), device=logits.device)

        

    return weighted_loss.sum() / denom





def soft_cross_entropy(logits: torch.Tensor,

                       targets: torch.Tensor,

                       mask: Optional[torch.Tensor]) -> torch.Tensor:

    log_probs = torch.log_softmax(logits, dim=1)

    loss = -(targets * log_probs).sum(dim=1)

    if mask is not None:

        mask = mask.to(logits.device, dtype=loss.dtype)

        loss = loss * mask

        denom = mask.sum()

    else:

        denom = torch.tensor(loss.numel(), device=logits.device, dtype=loss.dtype)

    if denom.item() <= 0:

        return torch.zeros((), device=logits.device, dtype=loss.dtype)

    return loss.sum() / denom





def enforce_min_duration(mask: np.ndarray,

                         labels: np.ndarray,

                         min_frames: int) -> np.ndarray:

    if min_frames <= 1:

        return mask

    keep = np.zeros_like(mask, dtype=bool)

    idx = 0

    length = mask.shape[0]

    while idx < length:

        if mask[idx]:

            label = labels[idx]

            end = idx

            while end < length and mask[end] and labels[end] == label:

                end += 1

            if end - idx >= min_frames:

                keep[idx:end] = True

            idx = end

        else:

            idx += 1

    return keep





def sharpen_probabilities(prob_tensor: torch.Tensor, temperature: float) -> torch.Tensor:

    temp = max(float(temperature), 1e-6)

    powered = prob_tensor.pow(1.0 / temp)

    denom = powered.sum(dim=0, keepdim=True).clamp_min(1e-12)

    return powered / denom



# ---------------------------------------------------------

# 3. 疑似ラベル生成ロジックの簡素化 (固定閾値 + Sigmoid)

# ---------------------------------------------------------

def create_pseudo_labels(logits: torch.Tensor,

                         valid_lengths: torch.Tensor,

                         hop_length: int,

                         threshold: float,

                         min_duration_frames: int,

                         swallow_idx: int,

                         temperature: float,

                         use_adaptive_threshold: bool = False) -> Tuple[torch.Tensor, torch.Tensor, int, int]:

    """

    BCEベースに変更：

    1. Sigmoidを使用（クラス独立確率）

    2. Othersのランキング除外はやめて、固定閾値＋嚥下優先ロジック

    """

    with torch.no_grad():

        # BCE前提なので Sigmoid で確率化

        probs = torch.sigmoid(logits)  # [B, C, T]

        B, C, T = probs.shape



        # ---- Sharpening ----

        if temperature != 1.0:

            probs_sharp = probs.pow(1.0 / temperature)

            probs_sharp = probs_sharp / (probs_sharp + (1 - probs).pow(1.0 / temperature) + 1e-8)

            pseudo_distributions = probs_sharp

        else:

            pseudo_distributions = probs



        frame_mask = torch.zeros((B, T), device=probs.device, dtype=torch.float32)

        total_frames = 0

        selected_frames = 0



        # 各フレームで最大の確率を持つクラスとその確率

        max_probs, max_labels = torch.max(probs, dim=1)  # [B, T], [B, T]



        # ★ 嚥下専用の少し低い閾値（例：全体閾値の 0.8 倍、下限0.05）

        threshold_swallow = max(threshold * 0.1, 0.05)



        for idx in range(B):

            valid_frames = int(valid_lengths[idx].item() // hop_length)

            valid_frames = min(valid_frames, T)

            if valid_frames <= 0:

                continue



            total_frames += valid_frames



            # スライス

            prob_slice = max_probs[idx, :valid_frames]      # [T_valid]

            label_slice = max_labels[idx, :valid_frames]    # [T_valid]

            swallow_probs = probs[idx, swallow_idx, :valid_frames]  # [T_valid]



            # ----------------------------------------------------

            # ★ 嚥下優先ロジック

            # 1) 嚥下クラス: threshold_swallow を超えたら採用

            # 2) それ以外のクラス: 従来どおり threshold

            # ----------------------------------------------------

            with torch.no_grad():

                # 嚥下として自信ありのフレーム

                swallow_confident = swallow_probs >= threshold_swallow  # [T_valid]



                # 非嚥下 or 嚥下として自信なしのフレームは従来の閾値

                base_confident = prob_slice >= threshold  # [T_valid]



                # 最終的な confident_mask

                #   嚥下が自信あり → True

                #   それ以外       → base_confident に従う

                confident_mask = base_confident.clone()

                confident_mask[swallow_confident] = True



                # ラベルも嚥下を優先的に上書き

                label_slice = label_slice.clone()

                label_slice[swallow_confident] = swallow_idx



            # NumPy へ変換して duration 処理（CPU）

            confident_mask_np = confident_mask.cpu().numpy()

            label_slice_np = label_slice.cpu().numpy()



            # 最小継続時間フィルタリング

            final_mask_np = enforce_min_duration(

                confident_mask_np,

                label_slice_np,

                min_duration_frames

            )



            if final_mask_np.sum() > 0:

                mask_tensor = torch.from_numpy(final_mask_np).to(frame_mask.device).float()

                frame_mask[idx, :valid_frames] = mask_tensor

                selected_frames += int(final_mask_np.sum())



                if idx == 0:

                    logger.debug(

                        "PseudoLabel SwallowFirst: thr=%.2f, thr_swallow=%.2f, selected=%d/%d",

                        threshold,

                        threshold_swallow,

                        int(final_mask_np.sum()),

                        valid_frames

                    )



        return pseudo_distributions.detach(), frame_mask.detach(), selected_frames, total_frames









class StreamToLogger(io.TextIOBase):

    def __init__(self, logger: logging.Logger, level: int, original_stream):

        super().__init__()

        self.logger = logger

        self.level = level

        self.original_stream = original_stream

        self._buffer = ''



    def write(self, message: str) -> int:

        if not message:

            return 0

        self._buffer += message

        while '\n' in self._buffer:

            line, self._buffer = self._buffer.split('\n', 1)

            line = line.rstrip('\r')

            if line:

                self.logger.log(self.level, line)

        return len(message)



    def flush(self) -> None:

        if self._buffer:

            line = self._buffer.rstrip('\r')

            if line:

                self.logger.log(self.level, line)

            self._buffer = ''

        if hasattr(self.original_stream, 'flush'):

            self.original_stream.flush()







def evaluate_real_data_validation(model: nn.Module,

                                   config: Dict,

                                   device: torch.device,

                                   exist_label_dir: Path,

                                   output_dir: Path,

                                   threshold: float,

                                   chunk_sec: float,

                                   overlap_sec: float) -> Optional[List[Tuple[str, float]]]:

    if inference_wav_module is None:

        logger.warning("Real-data validation skipped: inference_wav module is not available.")

        return None



    try:

        pairs = inference_wav_module.find_exist_label_pairs(str(exist_label_dir))

    except Exception as exc:

        logger.warning("Failed to collect real-data evaluation pairs from %s: %s", exist_label_dir, exc)

        return None



    if not pairs:

        logger.warning("No WAV/TXT pairs found under %s; skipping real-data validation.", exist_label_dir)

        return []



    inference_wav_module.DEVICE = device

    inference_wav_module.ensure_dir(output_dir)



    eval_config = {

        "sr": config.get("sr"),

        "ssl_hop_length": config.get("ssl_hop_length"),

        "classes": config.get("classes", []),

        "feature_type": config.get("feature_type", "raw"),

        "architecture": config.get("architecture", "gru"),

        "threshold": threshold,

        "inference_chunk_sec": chunk_sec,

        "inference_overlap_sec": overlap_sec,

    }



    if eval_config["sr"] is None or eval_config["ssl_hop_length"] is None:

        logger.warning("Incomplete configuration for real-data validation; skipping evaluation.")

        return None



    was_training = model.training

    model.eval()



    results: List[Tuple[str, float]] = []

    for wav_path, txt_path in pairs:

        eval_result = inference_wav_module.run_inference_on_single_wav(

            wav_file_path_str=wav_path,

            model=model,

            config=eval_config,

            output_dir=output_dir,

            experiment_name="real_eval",

            gt_txt_path=txt_path,

        )

        subject = Path(wav_path).stem

        f1_score = 0.0

        if eval_result and eval_result.get('overall'):

            overall_metrics = eval_result['overall']

            precision_recall_f1 = overall_metrics.get(0.3)

            if precision_recall_f1 is not None and len(precision_recall_f1) >= 3:

                f1_score = float(precision_recall_f1[2])

        results.append((subject, f1_score))



    if was_training:

        model.train()



    return results





# ---------------------------------------------------------

# 1. 損失関数の変更 (BCEWithLogitsLossへ)

# ---------------------------------------------------------

def build_loss(config: Dict, device: torch.device) -> Tuple[nn.Module, nn.Module]:

    # Script Aに合わせて BCEWithLogitsLoss を使用

    # reduction='none' にして、マスク処理を後で行えるようにする

    

    # 必要に応じてpos_weightを設定可能ですが、Script Aの構成に合わせて標準設定にします

    criterion_sup = nn.BCEWithLogitsLoss(reduction='none')

    criterion_unsup = nn.BCEWithLogitsLoss(reduction='none')

    

    return criterion_sup.to(device), criterion_unsup.to(device)





def build_test_loader(config: Dict,

                      batch_size: int,

                      num_workers: int,

                      device: torch.device,

                      test_json_path: Optional[Path] = None) -> DataLoader:

    test_path = Path(test_json_path) if test_json_path is not None else Path(config["test_json"])

    test_ann = load_json(test_path)

    dataset_test = SoundEventDataset(test_ann, config, config.get("feature_type", "raw"), 1.0, local_rank=0)

    collate_fn = lambda batch: pad_collate(batch, config=config)

    use_pinned = device.type != 'cpu'

    return DataLoader(

        dataset_test,

        batch_size=batch_size,

        shuffle=False,

        collate_fn=collate_fn,

        num_workers=num_workers,

        pin_memory=use_pinned,

        persistent_workers=num_workers > 0,

    )









def infinite_data_loader(loader: Iterable):

    """Yield batches from loader indefinitely without caching."""

    while True:

        for batch in loader:

            yield batch



def run_test_evaluation(model: nn.Module,

                        config: Dict,

                        device: torch.device,

                        output_dir: Path,

                        num_workers: int,

                        dataset_label: str = "test",

                        test_json_override: Optional[Path] = None,

                        batch_limit: Optional[int] = None) -> None:

    label_display = dataset_label.replace("_", " ")

    logger.info("Preparing %s dataloader...", label_display)

    test_loader = build_test_loader(

        config,

        batch_size=config["batch_size"],

        num_workers=max(0, num_workers),

        device=device,

        test_json_path=test_json_override,

    )

    dataset_size = len(test_loader.dataset)

    if dataset_size == 0:

        logger.warning("%s dataset is empty; skipping evaluation.", label_display.title())

        return

    source_path = Path(test_json_override) if test_json_override is not None else Path(config["test_json"])

    logger.info(

        "%s dataset size: %d samples\nAnnotations: %s",

        label_display.title(),

        dataset_size,

        source_path,

    )



    logger.info("テスト評価を開始します...")

    logger.info("=== テスト評価を開始します... ===")

    all_classes = list(config.get("classes", []) or [])

    start_time = time.time()

    test_loss, _, gt_list, pred_list, file_paths = evaluate_epoch(

        model,

        test_loader,

        nn.CrossEntropyLoss(),  # reduction='none' を削除

        config,

        local_rank=0,

        world_size=1,

        compute_event_metrics=False,

        epoch_num=None,

        num_epochs_total=None,

        limit=batch_limit,

        return_detailed=True,

    )

    elapsed = time.time() - start_time

    num_samples = len(file_paths)

    samples_per_sec = (num_samples / elapsed) if elapsed > 0 else 0.0



    log_prefix = label_display.title()

    logger.info(

        "Inference time: %.2fs for %d samples (%.2f samples/s)",

        elapsed,

        num_samples,

        samples_per_sec,

    )

    logger.info("%s loss: %.4f", log_prefix, test_loss)



    eval_classes = [cls for cls in all_classes if cls != "others"]

    if not eval_classes:

        eval_classes = list(all_classes)



    overall_metrics = calculate_event_metrics(gt_list, pred_list, eval_classes)

    logger.info("=== IoUごとのF1スコア ===")

    for threshold in sorted(overall_metrics.keys()):

        precision, recall, f1 = overall_metrics[threshold]

        logger.info(

            "IoU > %.2f: P=%.3f, R=%.3f, F1=%.3f",

            threshold,

            precision,

            recall,

            f1,

        )



    raw_iou_values = collect_raw_iou(gt_list, pred_list, all_classes)

    raw_iou_summary = {}

    logger.info("=== Raw IoU Summary ===")

    for class_name in all_classes:

        values = raw_iou_values.get(class_name, [])

        mean_value = float(np.mean(values)) if values else 0.0

        raw_iou_summary[class_name] = {

            "mean": mean_value,

            "count": len(values),

        }

        logger.info("%s: %.3f", class_name, mean_value)



    matched_iou_values = collect_matched_iou(gt_list, pred_list, all_classes)

    matched_iou_summary = {}

    logger.info("=== Matched IoU Summary ===")

    for class_name in all_classes:

        values = matched_iou_values.get(class_name, [])

        mean_value = float(np.mean(values)) if values else 0.0

        matched_iou_summary[class_name] = {

            "mean": mean_value,

            "count": len(values),

        }

        logger.info("%s: %.3f", class_name, mean_value)



    detailed_metrics = calculate_event_metrics_by_class_and_food(

        gt_list,

        pred_list,

        file_paths,

        eval_classes,

    )

    logger.info("=== クラス別・食材別詳細評価 ===")

    category_keys = ["all"] + [key for key in sorted(detailed_metrics.keys()) if key != "all"]

    for category in category_keys:

        class_data = detailed_metrics.get(category, {})

        logger.info(f"--- {category} ---")

        for class_name in eval_classes:

            metrics_by_threshold = class_data.get(class_name)

            if metrics_by_threshold is None:

                logger.info("%s: データなし", class_name)

                continue

            logger.info(f"{class_name}:")

            for threshold in sorted(metrics_by_threshold.keys()):

                precision, recall, f1 = metrics_by_threshold[threshold]

                logger.info(

                    "  IoU > %.2f: P=%.3f, R=%.3f, F1=%.3f",

                    threshold,

                    precision,

                    recall,

                    f1,

                )



    raw_iou_by_food_values = collect_raw_iou_by_food(

        gt_list,

        pred_list,

        file_paths,

        eval_classes,

    )

    raw_iou_by_food_summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    logger.info("=== 食材別 Raw IoU ===")

    for food_type in sorted(raw_iou_by_food_values.keys()):

        class_values = raw_iou_by_food_values[food_type]

        food_summary: Dict[str, Dict[str, float]] = {}

        parts = []

        for class_name in eval_classes:

            values = class_values.get(class_name, [])

            mean_value = float(np.mean(values)) if values else 0.0

            food_summary[class_name] = {

                "mean": mean_value,

                "count": len(values),

            }

            parts.append(f"{class_name}={mean_value:.3f}")

        raw_iou_by_food_summary[food_type] = food_summary

        logger.info("%s: %s", food_type, " ".join(parts))



    label_slug = dataset_label.replace(" ", "_").lower()

    summary_path = output_dir / f"{label_slug}_metrics.json"

    summary_export = {

        "loss": float(test_loss),

        "overall_metrics": {

            f"{threshold:.2f}": {

                "precision": float(precision),

                "recall": float(recall),

                "f1": float(f1),

            }

            for threshold, (precision, recall, f1) in sorted(overall_metrics.items())

        },

        "event_classes": eval_classes,

        "num_samples": num_samples,

        "inference_time_s": elapsed,

    }

    with open(summary_path, "w", encoding="utf-8") as handle:

        json.dump(summary_export, handle, ensure_ascii=False, indent=2)



    detailed_filename = f"detailed_{label_slug}_metrics.json"

    detailed_path = output_dir / detailed_filename

    detailed_export = {

        "dataset_label": dataset_label,

        "loss": float(test_loss),

        "event_classes": eval_classes,

        "overall_metrics": summary_export["overall_metrics"],

        "detailed_metrics": {

            category: {

                class_name: {

                    f"{threshold:.2f}": {

                        "precision": float(precision),

                        "recall": float(recall),

                        "f1": float(f1),

                    }

                    for threshold, (precision, recall, f1) in sorted(metrics_by_threshold.items())

                }

                for class_name, metrics_by_threshold in class_data.items()

            }

            for category, class_data in detailed_metrics.items()

        },

        "raw_iou_summary": raw_iou_summary,

        "matched_iou_summary": matched_iou_summary,

        "raw_iou_by_food": raw_iou_by_food_summary,

        "num_samples": num_samples,

        "inference_time_s": elapsed,

    }

    with open(detailed_path, "w", encoding="utf-8") as handle:

        json.dump(detailed_export, handle, ensure_ascii=False, indent=2)

    logger.info("Saved %s metrics.\nOutput file: %s", label_display, summary_path)

    logger.info("詳細メトリクスを保存しました: %s", detailed_path)





def prepare_dataloaders(config: Dict,

                         batch_size: int,

                         num_workers: int,

                         unlabeled_root: Path,

                         min_window_sec: float,

                         max_window_sec: float,

                         overlap_sec: float,

                         device: torch.device,

                         seed: int) -> Tuple[DataLoader, DataLoader, UnlabeledSegmentGenerator]:

    train_ann = load_json(Path(config["train_json"]))

    val_ann = load_json(Path(config["val_json"]))

    dataset_train = SoundEventDataset(train_ann, config, config.get("feature_type", "raw"), 1.0, local_rank=0)

    dataset_val = SoundEventDataset(val_ann, config, config.get("feature_type", "raw"), 1.0, local_rank=0)

    collate_fn = lambda batch: pad_collate(batch, config=config)

    use_pinned = device.type != 'cpu'

    dataloader_train = DataLoader(dataset_train,

                                  batch_size=batch_size,

                                  shuffle=True,

                                  collate_fn=collate_fn,

                                  num_workers=num_workers,

                                  pin_memory=use_pinned,

                                  persistent_workers=num_workers > 0)

    dataloader_val = DataLoader(dataset_val,

                                batch_size=batch_size,

                                shuffle=False,

                                collate_fn=collate_fn,

                                num_workers=num_workers,

                                pin_memory=use_pinned,

                                persistent_workers=num_workers > 0)

    segment_generator = UnlabeledSegmentGenerator(

        root=unlabeled_root,

        target_sr=config["sr"],

        min_window_sec=min_window_sec,

        max_window_sec=max_window_sec,

        overlap_sec=overlap_sec,

        seed=seed,

    )

    return dataloader_train, dataloader_val, segment_generator





def compute_lambda(epoch: int, lambda_max: float, rampup_epochs: int) -> float:

    if rampup_epochs <= 0:

        return lambda_max

    progress = min(1.0, (epoch + 1) / float(rampup_epochs))

    return lambda_max * progress





def update_teacher_ema(student: nn.Module,

                       teacher: Optional[nn.Module],

                       ema_decay: float,

                       global_step: int) -> None:

    if teacher is None:

        return

    decay = float(ema_decay)

    decay = min(max(decay, 0.0), 1.0)

    step = max(global_step, 0)

    momentum = min(decay, 1.0 - 1.0 / float(step + 1))



    with torch.no_grad():

        student_params = dict(student.named_parameters())

        for name, teacher_param in teacher.named_parameters():

            student_param = student_params.get(name)

            if student_param is None:

                continue

            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)

        # Keep buffers (e.g., BatchNorm running stats) in sync

        student_buffers = dict(student.named_buffers())

        for name, teacher_buffer in teacher.named_buffers():

            student_buffer = student_buffers.get(name)

            if student_buffer is None:

                continue

            teacher_buffer.data.copy_(student_buffer.data)





def build_augmentation_pipelines(sample_rate: int,

                                 args: argparse.Namespace) -> Tuple[Optional[Callable], Optional[Callable]]:

    if args.no_augmentation or not AUDIO_AUG_AVAILABLE:

        if args.no_augmentation:

            logger.info("Data augmentation disabled via command-line flag.")

        elif not AUDIO_AUG_AVAILABLE:

            logger.warning("audiomentations not available; proceeding without data augmentation.")

        return None, None



    def make_compose(transforms: List) -> Optional[Compose]:

        valid = [t for t in transforms if t is not None]

        if not valid:

            return None

        return Compose(valid)



    weak_transforms: List = []

    strong_transforms: List = []

    if Gain is not None:

        weak_transforms.append(Gain(min_gain_db=-3.0, max_gain_db=3.0, p=0.9))

        strong_transforms.append(Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.9))

    if AddGaussianNoise is not None:

        strong_transforms.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.02, p=0.8))



    weak_compose = make_compose(weak_transforms)

    strong_compose = make_compose(strong_transforms)



    def random_time_shift_numpy(samples: np.ndarray,

                                max_fraction: float,

                                probability: float) -> np.ndarray:

        if max_fraction <= 0.0 or probability <= 0.0 or probability > 1.0:

            return samples

        if random.random() > probability:

            return samples

        length = samples.shape[0]

        if length <= 1:

            return samples

        max_shift = int(round(length * max_fraction))

        max_shift = min(max_shift, length - 1)

        if max_shift <= 0:

            return samples

        shift = random.randint(-max_shift, max_shift)

        if shift == 0:

            return samples

        shifted = np.empty_like(samples)

        if shift > 0:

            shifted[:shift] = 0.0

            shifted[shift:] = samples[:-shift]

        else:

            shift = abs(shift)

            shifted[-shift:] = 0.0

            shifted[:-shift] = samples[shift:]

        return shifted



    def build_fn(compose_obj: Optional[Compose],

                 shift_fraction: float,

                 shift_probability: float) -> Callable[[np.ndarray, int], np.ndarray]:

        def augment(samples: np.ndarray,

                    sample_rate: int,

                    **kwargs: Dict) -> np.ndarray:

            augmented = samples

            if compose_obj is not None:

                augmented = compose_obj(samples=augmented, sample_rate=sample_rate)

            augmented = random_time_shift_numpy(augmented, shift_fraction, shift_probability)

            return augmented

        return augment



    weak_pipeline = build_fn(weak_compose, shift_fraction=0.02, shift_probability=0.8)

    strong_pipeline = build_fn(strong_compose, shift_fraction=0.05, shift_probability=0.8)

    logger.info(

        "Data augmentation enabled (weak & strong pipelines) at %d Hz sample rate.",

        sample_rate,

    )

    return weak_pipeline, strong_pipeline





def apply_augmentation_batch(waveforms: torch.Tensor,

                             valid_lengths: Sequence[int],

                             augment_fn: Optional[Callable],

                             sample_rate: int) -> torch.Tensor:

    if augment_fn is None or waveforms.numel() == 0:

        return waveforms



    augmented: List[torch.Tensor] = []

    for waveform, valid_len in zip(waveforms, valid_lengths):

        valid_samples = int(valid_len)

        valid_samples = max(0, min(valid_samples, waveform.shape[0]))

        prefix = waveform[:valid_samples]

        suffix = waveform[valid_samples:]

        prefix_np = prefix.detach().cpu().numpy().astype(np.float32, copy=False)

        augmented_np = augment_fn(samples=prefix_np, sample_rate=sample_rate)

        if not isinstance(augmented_np, np.ndarray):

            augmented_np = np.asarray(augmented_np, dtype=np.float32)

        if augmented_np.shape[0] != prefix_np.shape[0]:

            if augmented_np.shape[0] > prefix_np.shape[0]:

                augmented_np = augmented_np[:prefix_np.shape[0]]

            else:

                pad_width = prefix_np.shape[0] - augmented_np.shape[0]

                augmented_np = np.pad(augmented_np, (0, pad_width), mode="constant")

        augmented_tensor = torch.from_numpy(augmented_np).type_as(waveform)

        if suffix.numel() > 0:

            augmented_tensor = torch.cat([augmented_tensor, suffix.clone()], dim=0)

        augmented.append(augmented_tensor)

    return torch.stack(augmented, dim=0)





def save_checkpoint(model: nn.Module,

                    teacher_model: Optional[nn.Module],

                    optimizer: torch.optim.Optimizer,

                    epoch: int,

                    out_dir: Path,

                    best_val: float,

                    tag: str,

                    global_step: int,

                    best_monitor: float,

                    monitor_metric: str) -> None:

    state = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "teacher_state_dict": teacher_model.state_dict() if teacher_model is not None else None,

        "optimizer_state_dict": optimizer.state_dict(),

        "best_val_loss": best_val,

        "global_step": int(global_step),

        "best_monitor_value": best_monitor,

        "monitor_metric": monitor_metric,

        "rng_state": {

            "python": random.getstate(),

            "numpy": np.random.get_state(),

            "torch": torch.get_rng_state(),

            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,

        },

    }

    torch.save(state, out_dir / f"{tag}.pth")





def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:

    for state in optimizer.state.values():

        for key, value in list(state.items()):

            if torch.is_tensor(value):

                state[key] = value.to(device)





def load_training_state(model: nn.Module,

                        teacher_model: Optional[nn.Module],

                        optimizer: torch.optim.Optimizer,

                        checkpoint_path: Path,

                        device: torch.device) -> Tuple[int, float, int, float, str]:

    checkpoint = safe_torch_load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint or "optimizer_state_dict" not in checkpoint:

        raise KeyError(f"Checkpoint at {checkpoint_path} missing required keys to resume training.")

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    move_optimizer_state_to_device(optimizer, device)



    resume_epoch = int(checkpoint.get("epoch", -1)) + 1

    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))

    global_step = int(checkpoint.get("global_step", 0))

    best_monitor_value = float(checkpoint.get("best_monitor_value", best_val_loss))

    monitor_metric = str(checkpoint.get("monitor_metric", "loss")).lower()



    teacher_state = checkpoint.get("teacher_state_dict")

    if teacher_model is not None:

        if teacher_state:

            teacher_model.load_state_dict(teacher_state)

        else:

            teacher_model.load_state_dict(model.state_dict())

        teacher_model.eval()



    rng_state = checkpoint.get("rng_state", {})

    python_state = rng_state.get("python")

    numpy_state = rng_state.get("numpy")

    torch_state = rng_state.get("torch")

    cuda_state = rng_state.get("cuda")



    if python_state is not None:

        random.setstate(python_state)

    if numpy_state is not None:

        np.random.set_state(numpy_state)

    if torch_state is not None:

        try:

            if isinstance(torch_state, torch.ByteTensor):

                torch.set_rng_state(torch_state)

            else:

                converted_cpu_state = torch.as_tensor(torch_state, dtype=torch.uint8)

                torch.set_rng_state(converted_cpu_state)

        except (TypeError, RuntimeError, ValueError) as exc:

            logger.warning("Failed to restore torch RNG state: %s", exc)

    if cuda_state is not None and torch.cuda.is_available():

        try:

            converted_states = []

            for device_idx, state in enumerate(cuda_state):

                if isinstance(state, torch.ByteTensor) and state.is_cuda:

                    converted_states.append(state)

                else:

                    target_device = torch.device(f"cuda:{device_idx % torch.cuda.device_count()}")

                    converted_states.append(torch.as_tensor(state, dtype=torch.uint8, device=target_device))

            torch.cuda.set_rng_state_all(converted_states)

        except Exception as exc:

            logger.warning("Failed to restore CUDA RNG state from checkpoint: %s", exc)



    return max(0, resume_epoch), best_val_loss, max(0, global_step), best_monitor_value, monitor_metric





def resolve_closed_test_path(request: Optional[str],

                             config: Dict,

                             config_path: Path) -> Optional[Path]:

    if not request:

        return None



    def normalise_path(path_like: Path) -> Path:

        path_like = Path(path_like)

        if not path_like.is_absolute():

            return (config_path.parent / path_like).resolve()

        return path_like.resolve()



    if request != "auto":

        candidate = normalise_path(Path(request).expanduser())

        if candidate.exists():

            return candidate

        logger.error("Closed-test JSON not found at %s", candidate)

        return None



    candidates: List[Optional[Path]] = [

        Path(config.get("closed_test_json")) if config.get("closed_test_json") else None,

        Path(config["test_json"]).with_name("merged_test.json") if "test_json" in config else None,

        config_path.parent / "json/fit1/merged_test.json",

        SCRIPT_DIR / "json/fit1/merged_test.json",

    ]



    for candidate in candidates:

        if candidate is None:

            continue

        candidate = normalise_path(candidate)

        if candidate.exists():

            return candidate

    logger.error("Unable to locate closed-test annotations automatically. Provide a JSON path via --closed-test.")

    return None





def train(args: argparse.Namespace) -> None:

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    seed_everything(args.seed)



    config_path = Path(args.config).resolve()

    config = load_experiment_config(config_path, args.experiment_key)

    experiment_label = config.get("name") or args.experiment_key or "default"

    exp_base_override = bool(config.pop("_exp_base_save_dir_overridden", False))



    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config["device"] = str(device)



    # ===== config優先で上書き =====

    # epochs / lr / batch_size は「CLI が指定されていれば上書き、なければ config の base_config を使う」

    if args.batch_size:

        config["batch_size"] = args.batch_size

    if args.lr:

        config["lr"] = args.lr

    config["num_epochs"] = args.epochs



    # unlabeled_root / exist_label_dir も config 優先

    config.setdefault("unlabeled_root", args.unlabeled_root)

    config.setdefault("exist_label_dir", args.exist_label_dir)



    # 推論・real-eval 用のチャンク長も config 優先

    config.setdefault("inference_chunk_sec", args.inference_chunk_sec)

    config.setdefault("inference_overlap_sec", args.inference_overlap_sec)

    config.setdefault("real_eval_threshold", args.real_eval_threshold)

    config.setdefault("real_eval_chunk_sec", config["inference_chunk_sec"])

    config.setdefault("real_eval_overlap_sec", config["inference_overlap_sec"])



    base_save_dir_str = config.get("base_save_dir")

    if args.output_dir:

        output_dir = Path(args.output_dir).expanduser().resolve()

    else:

        if base_save_dir_str:

            base_root = Path(base_save_dir_str).expanduser().resolve()

        else:

            base_root = (SCRIPT_DIR / "results").resolve()

        if exp_base_override or config.get("experiment_key") is None:

            output_dir = base_root

        else:

            output_dir = base_root / experiment_label

    output_dir.mkdir(parents=True, exist_ok=True)



    # semi-supervised 自体の checkpoint を保存する場所

    if args.checkpoint_dir:

        checkpoint_dir_path = Path(args.checkpoint_dir).expanduser().resolve()

    else:

        checkpoint_dir_path = output_dir

    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)



    args.output_dir = str(output_dir)

    args.checkpoint_dir = str(checkpoint_dir_path)

    config["base_save_dir"] = str(output_dir)



    # Reset logger handlers so each run logs to console and file only once

    if logger.handlers:

        for handler in list(logger.handlers):

            logger.removeHandler(handler)

    logger.setLevel(logging.INFO)

    logger.propagate = False



    stream_handler = logging.StreamHandler(ORIGINAL_STDOUT)

    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

    logger.addHandler(stream_handler)



    log_file_path = output_dir / "training.log"

    file_exists = log_file_path.exists()

    append_log = args.test_mode or args.resume or file_exists

    file_mode = "a" if append_log else "w"

    file_handler = logging.FileHandler(log_file_path, mode=file_mode, encoding="utf-8")

    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

    logger.addHandler(file_handler)



    if not isinstance(sys.stdout, StreamToLogger):

        sys.stdout = StreamToLogger(logger, logging.INFO, ORIGINAL_STDOUT)

    if not isinstance(sys.stderr, StreamToLogger):

        sys.stderr = StreamToLogger(logger, logging.ERROR, ORIGINAL_STDERR)



    logger.info("=" * 80)

    logger.info("Starting experiment: %s", experiment_label)

    logger.info("Classes: %s", config.get("classes", []))

    logger.info("Output directory: %s", output_dir)

    logger.info("Checkpoint directory: %s", checkpoint_dir_path)

    logger.info("=" * 80)



    exist_label_dir = Path(config["exist_label_dir"]).resolve()

    real_eval_dir = output_dir / "real_eval"

    quick_real_eval_dir = real_eval_dir / "iter_checks"

    quick_real_eval_interval = 500



    # --- Real-data validation control ---

    # デフォルトは OFF。--real-eval を付けたときだけ ON を検討する

    real_eval_enabled = False



    if args.real_eval:

        if inference_wav_module is None:

            logger.warning(

                "Real-data validation requested via --real-eval, but inference_wav module "

                "could not be imported. Disabling real-data validation."

            )

        elif not exist_label_dir.exists():

            logger.warning(

                "Real-data validation requested via --real-eval, but directory not found at %s. "

                "Disabling real-data validation.",

                exist_label_dir,

            )

        else:

            real_eval_enabled = True

            logger.info("Real-data validation enabled using %s (requested via --real-eval).", exist_label_dir)

    else:

        logger.info("Real-data validation is disabled by default. Use --real-eval to enable it.")



    real_eval_threshold = float(config["real_eval_threshold"])

    real_eval_chunk_sec = float(config["real_eval_chunk_sec"])

    real_eval_overlap_sec = float(config["real_eval_overlap_sec"])

    real_eval_patience = max(1, args.real_eval_patience)

    best_real_avg_f1 = -float('inf')

    best_real_epoch = -1

    real_epochs_since_improvement = 0

    closed_test_path: Optional[Path] = None

    if args.closed_test:

        closed_test_path = resolve_closed_test_path(args.closed_test, config, config_path)

        if closed_test_path:

            logger.info("Closed-test evaluation enabled.\nAnnotations: %s", closed_test_path)



    min_duration_frames = max(

        1,

        int(round((args.min_duration_ms / 1000.0) / (config["ssl_hop_length"] / config["sr"])))

    )



    merge_gap_frames = 0

    if args.merge_gap_ms > 0:

        merge_gap_frames = int(round((args.merge_gap_ms / 1000.0) / (config["ssl_hop_length"] / config["sr"])))

    config["merge_gap_frames"] = merge_gap_frames



    primary_event_class, swallow_idx = determine_primary_event_class(config)

    config["primary_event_class"] = primary_event_class

    config["swallow_idx"] = swallow_idx

    if primary_event_class != "swallowing":

        logger.info(

            "Primary event class set to '%s' (index %d).",

            primary_event_class,

            swallow_idx,

        )



    if args.pseudo_temperature <= 0.0:

        logger.warning("pseudo_temperature must be positive; defaulting to 0.5.")

        args.pseudo_temperature = 0.5

    config["pseudo_temperature"] = float(args.pseudo_temperature)



    logger.info("Loading datasets...")

    train_loader, val_loader, segment_generator = prepare_dataloaders(

        config,

        batch_size=config["batch_size"],

        num_workers=args.num_workers,

        unlabeled_root=Path(config["unlabeled_root"]).resolve(),

        min_window_sec=args.min_segment_duration,

        max_window_sec=args.max_segment_duration,

        overlap_sec=args.overlap_seconds,

        device=device,

        seed=args.seed,

    )

    weak_augment_fn, strong_augment_fn = build_augmentation_pipelines(config["sr"], args)



    logger.info("Initialising model...")

    model = EventDetector(

        config,

        config.get("ssl_model_name"),

        config.get("freeze_feature_extractor", False),

        config.get("freeze_transformer_layers", False),

        config.get("architecture", "gru"),

        config.get("feature_type", "raw"),

        local_rank=0,

    )

    model.to(device)

    teacher_model = EventDetector(

        config,

        config.get("ssl_model_name"),

        config.get("freeze_feature_extractor", False),

        config.get("freeze_transformer_layers", False),

        config.get("architecture", "gru"),

        config.get("feature_type", "raw"),

        local_rank=0,

    )

    teacher_model.to(device)

    for param in teacher_model.parameters():

        param.requires_grad_(False)

    teacher_model.eval()



    # ===== ベースとなる supervised モデルの checkpoint 探索 =====

    base_weights_loaded = False

    base_candidates: List[Path] = []



    pretrained_root = config.get("pretrained_root_dir")

    if pretrained_root:

        root = Path(pretrained_root).expanduser().resolve()

        # よくあるパターンを全部試す

        base_candidates.extend([

            root / experiment_label / "best_model_state.pth",

            root / experiment_label / "best.pth",

            root / "best_model_state.pth",

            root / "best.pth",

        ])

    else:

        # 古い挙動: 同じ output_dir/ checkpoint_dir にある best.pth を使う

        base_candidates.append(checkpoint_dir_path / "best.pth")



    for ckpt in base_candidates:

        if ckpt.exists():

            if load_model_weights(model, ckpt, device):

                base_weights_loaded = True

                logger.info("Loaded base weights from %s", ckpt)

                break

            else:

                logger.warning("Found base checkpoint but failed to load: %s", ckpt)



    if not base_weights_loaded:

        logger.warning(

            "No base checkpoint could be loaded. "

            "Checked candidates: %s",

            ", ".join(str(p) for p in base_candidates),

        )



    if base_weights_loaded:

        teacher_model.load_state_dict(model.state_dict())



    if not base_weights_loaded and not args.resume:

        raise FileNotFoundError(f"Failed to load initial weights and resume was not requested. Checked: {checkpoint_path}")



    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if config.get("optimizer", "adam").lower() == "sgd":

        optimizer = torch.optim.SGD(trainable_params, lr=config["lr"], momentum=config.get("momentum", 0.9))

    else:

        optimizer = torch.optim.Adam(trainable_params, lr=config["lr"])



    best_val_loss = float('inf')

    start_epoch = 0

    global_step = 0

    monitor_metric = args.early_stop_metric.lower()

    maximize_metric = monitor_metric == "f1"

    patience = max(1, args.early_stop_patience)

    best_metric_value = -float('inf') if maximize_metric else float('inf')

    best_metric_epoch = -1

    epochs_since_improvement = 0

    early_stop_triggered = False

    real_eval_stop_triggered = False

    



    # --- 1. 最終的なコマンドライン引数(args)をログに出力 ---

    # (configに上書きされる「前」の、ユーザー入力値)

    logger.info("--- Final Command-Line Arguments (args) ---")

    try:

        # vars() で Namespace を dict に変換

        args_dict = vars(args)

        args_str = json.dumps(args_dict, indent=2, ensure_ascii=False)

        for line in args_str.split('\n'):

            logger.info(line)

    except Exception as e:

        logger.error(f"Failed to serialize and log args: {e}")

        logger.info(str(args)) # 失敗した場合はプレーンテキストで出力

    logger.info("--- End of Arguments ---")

    logger.info("=" * 80)



    # --- 2. 最終的な設定(config)をログに出力 ---

    # (argsやexperimentsで上書きされた「後」の、実際に使われる値)

    logger.info("--- Final Resolved Configuration (config) ---")

    try:

        # config辞書をJSON文字列に変換（読みやすくインデント）

        config_str = json.dumps(config, indent=2, ensure_ascii=False)

        # 1行ずつログに出力する（タイムスタンプを各行につけるため）

        for line in config_str.split('\n'):

            logger.info(line)

    except Exception as e:

        logger.error(f"Failed to serialize and log config: {e}")

        logger.info(str(config)) # 失敗した場合はプレーンテキストで出力

    logger.info("--- End of Configuration ---")

    logger.info("=" * 80)

    

    

    if args.resume:

        resume_raw = args.resume.strip() if isinstance(args.resume, str) else "latest"

        resume_arg = resume_raw.lower()

        resume_candidates: List[Tuple[str, Path]] = []

        attempted_paths: List[str] = []



        if resume_arg == "latest":

            resume_candidates.append(("latest checkpoint", output_dir / "latest.pth"))

        else:

            manual_path = Path(resume_raw).expanduser()

            resume_candidates.append(("specified checkpoint", manual_path))

            if not manual_path.is_absolute():

                resume_candidates.append(("specified checkpoint within output dir", (output_dir / manual_path).resolve()))

            # Also consider explicit latest if user passes custom tokens like "best"

            resume_candidates.append(("latest checkpoint", output_dir / "latest.pth"))



        resume_loaded = False

        for label, candidate in resume_candidates:

            candidate = Path(candidate).expanduser()

            attempted_paths.append(str(candidate.resolve()))

            if not candidate.exists():

                continue

            try:

                start_epoch, best_val_loss, global_step, best_metric_value, checkpoint_metric = load_training_state(

                    model,

                    teacher_model,

                    optimizer,

                    candidate,

                    device,

                )

                resume_loaded = True

                base_weights_loaded = True

                if checkpoint_metric and checkpoint_metric != args.early_stop_metric.lower():

                    logger.warning(

                        "Resume checkpoint uses monitor metric '%s' differing from requested '%s'. Using checkpoint metric.",

                        checkpoint_metric,

                        args.early_stop_metric,

                    )

                monitor_metric = (checkpoint_metric or monitor_metric).lower()

                maximize_metric = (monitor_metric == "f1")

                if np.isinf(best_metric_value):

                    best_metric_value = -float("inf") if maximize_metric else float("inf")

                best_metric_epoch = max(0, start_epoch - 1)

                epochs_since_improvement = 0

                last_completed = max(0, start_epoch - 1)

                if np.isinf(best_val_loss):

                    logger.info(

                        "Resumed training from %s (%s).\nLast completed epoch: %d\nNext epoch: %d",

                        candidate,

                        label,

                        last_completed,

                        start_epoch,

                    )

                else:

                    logger.info(

                        "Resumed training from %s (%s).\nLast completed epoch: %d\nNext epoch: %d\nBest val loss: %.4f",

                        candidate,

                        label,

                        last_completed,

                        start_epoch,

                        best_val_loss,

                    )

                break

            except Exception as exc:

                logger.error("Failed to resume from %s (%s): %s", candidate, label, exc)

                resume_loaded = False

                break



        if not resume_loaded:

            logger.warning(

                "Resume requested but no valid checkpoint was loaded.\nAttempted paths: %s",

                ", ".join(attempted_paths) if attempted_paths else "none",

            )

            start_epoch = 0

            best_val_loss = float('inf')

            best_metric_value = -float('inf') if maximize_metric else float('inf')

            best_metric_epoch = -1

            epochs_since_improvement = 0

            teacher_model.load_state_dict(model.state_dict())

    else:

        teacher_model.load_state_dict(model.state_dict())



    criterion_sup, criterion_unsup = build_loss(config, device)

    eval_criterion = nn.CrossEntropyLoss()  # reduction='none' を削除



    def load_for_evaluation(preference: str,

                            student_model: nn.Module,

                            teacher: Optional[nn.Module],

                            checkpoint_dir: Path) -> Optional[Path]:  # ← 引数追加

        preference = preference.lower()

        # output_dir → checkpoint_dir に変更

        best_path = checkpoint_dir / "best.pth"

        latest_path = checkpoint_dir / "latest.pth"

        best_teacher_path = checkpoint_dir / "best_teacher.pth"

        latest_teacher_path = checkpoint_dir / "latest_teacher.pth"

        candidates: List[Tuple[str, Path]] = []

        if preference == "latest":

            candidates.append(("latest semi-supervised checkpoint", latest_path))

            candidates.append(("best semi-supervised model", best_path))

        else:

            candidates.append(("best semi-supervised model", best_path))

            candidates.append(("latest semi-supervised checkpoint", latest_path))

    

        teacher_loaded = False



        tried_paths: List[str] = []

        for label, path in candidates:

            tried_paths.append(str(path))

            if load_model_weights(student_model, path, device):

                logger.info("Loaded %s from %s", label, path)

                if teacher is not None:

                    teacher_candidate: Optional[Path] = None

                    if path == latest_path and latest_teacher_path.exists():

                        teacher_candidate = latest_teacher_path

                    elif path == best_path and best_teacher_path.exists():

                        teacher_candidate = best_teacher_path

                    # Attempt to recover teacher weights from combined checkpoint if dedicated file missing

                    if teacher_candidate is None and path.exists() and path.suffix == ".pth":

                        try:

                            state = safe_torch_load(path, map_location=device, allow_pickle_fallback=False)

                            teacher_state = state.get("teacher_state_dict") if isinstance(state, dict) else None

                            if teacher_state:

                                teacher.load_state_dict(teacher_state)

                                teacher.eval()

                                logger.info("Loaded teacher weights embedded in %s", path)

                                teacher_loaded = True

                        except Exception as exc:

                            logger.warning("Failed to extract teacher weights from %s: %s", path, exc)

                    if teacher_candidate and not teacher_loaded:

                        if load_model_weights(teacher, teacher_candidate, device):

                            teacher.eval()

                            logger.info("Loaded teacher weights from %s", teacher_candidate)

                            teacher_loaded = True

                        else:

                            logger.warning("Failed to load teacher weights from %s", teacher_candidate)

                    if not teacher_loaded:

                        teacher.load_state_dict(student_model.state_dict())

                        teacher.eval()

                        logger.warning(

                            "Teacher weights missing for %s; falling back to student parameters.",

                            path,

                        )

                return path

        logger.error("Unable to locate a usable checkpoint. Tried: %s", ", ".join(tried_paths))

        return None



    if args.inference:

        logger.info("Inference mode enabled.")

        if not args.inference_wav:

            logger.error("--inference-wav must be specified in inference mode.")

            return

        if inference_wav_module is None:

            logger.error("Inference mode requires inference_wav module, but it could not be imported.")

            return

        

        wav_path = Path(args.inference_wav).resolve()

        if not wav_path.exists():

            logger.error("Specified WAV file does not exist: %s", wav_path)

            return

        

        eval_checkpoint_dir = checkpoint_dir_path

        selected_path = load_for_evaluation(args.test_checkpoint, model, teacher_model, eval_checkpoint_dir)

        if not selected_path:

            logger.error("Inference aborted due to missing checkpoints.")

            return

        

        eval_model = teacher_model if teacher_model is not None else model

        eval_model.eval()

        

        inference_wav_module.DEVICE = device

        inference_output_dir = output_dir / "inference_results"

        inference_wav_module.ensure_dir(inference_output_dir)

        

        inference_config = {

            "sr": config.get("sr"),

            "ssl_hop_length": config.get("ssl_hop_length"),

            "classes": config.get("classes", []),

            "feature_type": config.get("feature_type", "raw"),

            "architecture": config.get("architecture", "gru"),

            # 閾値だけは CLI で上書きできるようにしておく

            "threshold": args.inference_threshold,

            "inference_chunk_sec": config["inference_chunk_sec"],

            "inference_overlap_sec": config["inference_overlap_sec"],

        }

        

        logger.info("Running inference on: %s", wav_path)

        logger.info("Model: %s | Classes: %s", config.get("ssl_model_name", "unknown"), config.get("classes", []))

        logger.info("Threshold: %.2f | Chunk: %.1fs | Overlap: %.1fs", 

                   args.inference_threshold, args.inference_chunk_sec, args.inference_overlap_sec)

        

        # デバッグモード: inference_wavモジュールを使わず直接推論（チャンク処理あり）

        if args.inference_debug:

            logger.info("DEBUG MODE: Using direct model inference with chunking")

            try:

                import torchaudio

                waveform, sr = torchaudio.load(str(wav_path))

                if waveform.size(0) > 1:

                    waveform = waveform.mean(dim=0, keepdim=True)

                if sr != config["sr"]:

                    resampler = torchaudio.transforms.Resample(sr, config["sr"])

                    waveform = resampler(waveform)

                

                waveform = waveform.squeeze(0)  # [T]

                total_samples = waveform.size(0)

                

                # チャンク処理パラメータ

                chunk_sec = config["inference_chunk_sec"]

                overlap_sec = config["inference_overlap_sec"]

                chunk_samples = int(chunk_sec * config["sr"])

                step_samples = int((chunk_sec - overlap_sec) * config["sr"])

                

                logger.info("Audio duration: %.2f seconds", total_samples / config["sr"])

                logger.info("Chunk: %.1fs (%d samples) | Step: %.1fs (%d samples)", 

                           chunk_sec, chunk_samples, chunk_sec - overlap_sec, step_samples)

                

                all_swallow_probs = []

                chunk_count = 0

                

                with torch.no_grad():

                    pos = 0

                    while pos < total_samples:

                        end = min(pos + chunk_samples, total_samples)

                        chunk = waveform[pos:end].unsqueeze(0).to(device)  # [1, T]

                        

                        logits = eval_model(chunk)  # [1, C, T_frames]

                        probs = torch.softmax(logits, dim=1)

                        swallow_prob = probs[:, swallow_idx, :].cpu().numpy()  # [1, T_frames]

                        all_swallow_probs.append(swallow_prob.squeeze(0))

                        

                        chunk_count += 1

                        pos += step_samples

                        if pos >= total_samples:

                            break

                

                # 統計情報

                all_probs_concat = np.concatenate(all_swallow_probs, axis=0)

                logger.info("DEBUG MODE statistics:")

                logger.info("  Processed %d chunks", chunk_count)

                logger.info("  Total frames: %d", all_probs_concat.shape[0])

                target_label = config.get(

                    "primary_event_class",

                    config["classes"][swallow_idx] if (swallow_idx < len(config.get("classes", []))) else "swallowing",

                )

                logger.info("  %s probability:", target_label.capitalize() if target_label else "Target")

                logger.info("    Mean: %.4f | Std: %.4f", all_probs_concat.mean(), all_probs_concat.std())

                logger.info("    Min: %.4f | Max: %.4f", all_probs_concat.min(), all_probs_concat.max())

                logger.info("    Median: %.4f | 90th percentile: %.4f", 

                           np.median(all_probs_concat), np.percentile(all_probs_concat, 90))

                

                # 閾値を超えるフレームの検出

                threshold = args.inference_threshold

                confident_frames = (all_probs_concat > threshold).sum()

                total_frames = all_probs_concat.size

                logger.info("  Frames above threshold %.2f: %d/%d (%.2f%%)", 

                           threshold, confident_frames, total_frames,

                           100.0 * confident_frames / total_frames if total_frames > 0 else 0.0)

                

                if all_probs_concat.max() < 0.01:

                    logger.warning("=" * 80)

                    logger.warning(

                        "CRITICAL: Model is NOT predicting '%s' in DEBUG MODE!",

                        target_label,

                    )

                    logger.warning("  This confirms the issue is with the model itself, not inference_wav module")

                    logger.warning("=" * 80)

                

                logger.info("DEBUG MODE completed.")

                return

                

            except Exception as exc:

                logger.error("DEBUG MODE failed: %s", exc)

                import traceback

                traceback.print_exc()

                return

        

        # 複数の閾値で試行

        thresholds_to_try = [args.inference_threshold, 0.3, 0.2, 0.1]

        thresholds_to_try = sorted(set(thresholds_to_try), reverse=True)

        

        best_result = None

        best_event_count = 0

        best_threshold = args.inference_threshold

        

        for trial_threshold in thresholds_to_try:

            inference_config["threshold"] = trial_threshold

            

            logger.info("Trying threshold: %.2f", trial_threshold)

            

            # デバッグ: 最初の閾値で予測確率の統計を取得

            if trial_threshold == thresholds_to_try[0]:

                try:

                    # 簡易的な予測確率チェック

                    waveform, sr = torchaudio.load(str(wav_path))

                    if waveform.size(0) > 1:

                        waveform = waveform.mean(dim=0, keepdim=True)

                    if sr != config["sr"]:

                        resampler = torchaudio.transforms.Resample(sr, config["sr"])

                        waveform = resampler(waveform)

                    

                    # 最初の10秒のチャンクで予測

                    chunk_samples = min(config["sr"] * 10, waveform.size(1))

                    chunk = waveform[:, :chunk_samples].to(device)

                    

                    with torch.no_grad():

                        logits = eval_model(chunk)

                        probs = torch.softmax(logits, dim=1)

                        

                        # ロジット値の統計（デバッグ用）

                        logits_np = logits.cpu().numpy()

                        target_label = config.get(

                            "primary_event_class",

                            config["classes"][swallow_idx] if (swallow_idx < len(config.get("classes", []))) else "swallowing",

                        )

                        logger.info("  Logits statistics for '%s' class:", target_label)

                        swallow_logits = logits_np[:, swallow_idx, :]

                        logger.info("    Mean: %.4f | Std: %.4f", swallow_logits.mean(), swallow_logits.std())

                        logger.info("    Min: %.4f | Max: %.4f", swallow_logits.min(), swallow_logits.max())

                        

                        # swallowing クラスの確率統計

                        swallow_probs = probs[:, swallow_idx, :].cpu().numpy()

                        logger.info("  Probability statistics for '%s' class (first 10s):", target_label)

                        logger.info("    Mean: %.4f | Std: %.4f", swallow_probs.mean(), swallow_probs.std())

                        logger.info("    Min: %.4f | Max: %.4f", swallow_probs.min(), swallow_probs.max())

                        logger.info("    Median: %.4f | 90th percentile: %.4f", 

                                   np.median(swallow_probs), np.percentile(swallow_probs, 90))

                        

                        # 各クラスの平均確率とロジット

                        for cls_idx, cls_name in enumerate(config.get("classes", [])):

                            cls_logit_mean = logits_np[:, cls_idx, :].mean()

                            cls_prob_mean = probs[:, cls_idx, :].mean().item()

                            logger.info("    Class '%s': mean logit = %.4f, mean prob = %.4f", 

                                       cls_name, cls_logit_mean, cls_prob_mean)

                        

                        # 音声ファイルの基本統計情報

                        logger.info("  Audio file statistics:")

                        logger.info("    Duration: %.2f seconds", waveform.size(1) / config["sr"])

                        logger.info("    Sample rate: %d Hz", sr)

                        audio_rms = torch.sqrt(torch.mean(chunk ** 2)).item()

                        audio_max = torch.abs(chunk).max().item()

                        logger.info("    RMS level: %.6f | Max amplitude: %.6f", audio_rms, audio_max)

                        

                        # 警告: swallowing確率が極端に低い場合

                        if swallow_probs.max() < 0.01:

                            logger.warning("=" * 80)

                            logger.warning(

                                "CRITICAL: Model is NOT predicting '%s' class!",

                                target_label,

                            )

                            logger.warning(

                                "  Max %s probability: %.6f (threshold: 0.01)",

                                target_label,

                                swallow_probs.max(),

                            )

                            logger.warning(

                                "  %s logit range: %.4f to %.4f",

                                target_label,

                                swallow_logits.min(),

                                swallow_logits.max(),

                            )

                            logger.warning("  This indicates one of the following issues:")

                            logger.warning("    1. Mismatch between training and inference audio characteristics")

                            logger.warning("    2. Different audio preprocessing (filtering, normalization)")

                            logger.warning("    3. Severe class imbalance during training")

                            logger.warning("=" * 80)

                            logger.warning("Suggestions:")

                            logger.warning("  - Compare audio characteristics with training data")

                            logger.warning("  - Check if audio preprocessing matches training")

                            logger.warning("  - Verify w_sup=%.2f / w_unsup=%.2f balance", args.w_sup, args.w_unsup)

                            logger.warning("  - Consider re-training with w_sup=1.0 / w_unsup=0.1")

                            logger.warning("=" * 80)

                except Exception as exc:

                    logger.warning("Failed to compute probability statistics: %s", exc)

            

            result = inference_wav_module.run_inference_on_single_wav(

                wav_file_path_str=str(wav_path),

                model=eval_model,

                config=inference_config,

                output_dir=inference_output_dir,

                experiment_name=f"inference_th{trial_threshold:.2f}",

                gt_txt_path=None,

            )

            

            # 結果ファイルを確認

            output_txt = inference_output_dir / f"inference_th{trial_threshold:.2f}_{wav_path.stem}.txt"

            event_count = 0

            if output_txt.exists():

                with open(output_txt, 'r') as f:

                    lines = f.readlines()

                event_count = len(lines)

                logger.info("  Detected events with threshold %.2f: %d", trial_threshold, event_count)

                

                if event_count > best_event_count:

                    best_event_count = event_count

                    best_result = result

                    best_threshold = trial_threshold

                    # ベスト結果を標準名でコピー

                    best_output = inference_output_dir / f"inference_{wav_path.stem}.txt"

                    import shutil

                    shutil.copy2(output_txt, best_output)

            

            # 最初の試行で十分なイベントが検出されたら終了

            if trial_threshold == args.inference_threshold and event_count > 0:

                logger.info("  Sufficient events detected at initial threshold; stopping search.")

                break

        

        logger.info("Inference completed.")

        logger.info("Results saved to: %s", inference_output_dir)

        logger.info("Best threshold: %.2f with %d events", best_threshold, best_event_count)

        

        # ベスト結果の表示

        best_output_txt = inference_output_dir / f"inference_{wav_path.stem}.txt"

        if best_output_txt.exists():

            with open(best_output_txt, 'r') as f:

                lines = f.readlines()

            logger.info("Final detected events: %d", len(lines))

            if len(lines) > 0:

                logger.info("Preview (first 5 events):")

                for line in lines[:5]:

                    logger.info("  %s", line.strip())

                if len(lines) > 5:

                    logger.info("  ... and %d more events", len(lines) - 5)

        else:

            logger.warning("No events detected at any threshold.")

        

        # メトリクスがあれば表示

        if best_result and isinstance(best_result, dict):

            if best_result.get('overall'):

                overall = best_result['overall']

                logger.info("Overall metrics: Precision=%.3f, Recall=%.3f, F1=%.3f",

                           overall.get('precision', 0.0),

                           overall.get('recall', 0.0),

                           overall.get('f1', 0.0))

            if best_result.get('event_count') is not None:

                logger.info("Total events detected: %d", best_result['event_count'])

        

        return



    if args.test_mode:

        logger.info("Test mode enabled; skipping training and running evaluation only.")

        eval_checkpoint_dir = checkpoint_dir_path

        selected_path = load_for_evaluation(args.test_checkpoint, model, teacher_model, eval_checkpoint_dir)

        if not selected_path:

            logger.error("Test mode aborted due to missing checkpoints.")

            return

        eval_model = teacher_model if teacher_model is not None else model

        run_test_evaluation(eval_model, config, device, output_dir, args.test_num_workers, dataset_label="test")

        if closed_test_path:

            primary_test_path = Path(config["test_json"]).resolve()

            if primary_test_path != closed_test_path.resolve():

                closed_label = closed_test_path.stem or "closed_test"

                if closed_label.lower() == "test":

                    closed_label = "closed_test"

                run_test_evaluation(

                    eval_model,

                    config,

                    device,

                    output_dir,

                    args.test_num_workers,

                    dataset_label=closed_label,

                    test_json_override=closed_test_path,

                )

        return



    use_pinned = device.type != 'cpu'

    epoch_durations: List[float] = []



    if start_epoch >= args.epochs:

        logger.info(

            "Requested epochs (%d) already completed according to resume checkpoint. Skipping training loop.",

            args.epochs,

        )

    else:

        if start_epoch > 0:

            if np.isinf(best_val_loss):

                logger.info(

                    "Resuming training.\nNext epoch: %d/%d",

                    start_epoch + 1,

                    args.epochs,

                )

            else:

                logger.info(

                    "Resuming training.\nNext epoch: %d/%d\nCurrent best val loss: %.4f",

                    start_epoch + 1,

                    args.epochs,

                    best_val_loss,

                )

        else:

            logger.info("Starting semi-supervised training for %d epochs", args.epochs)



        for epoch in range(start_epoch, args.epochs):

            logger.info("Epoch %d/%d: generating unlabeled segments", epoch + 1, args.epochs)

            segments = segment_generator.generate_segments(epoch)

            if not segments:

                logger.warning(f"No unlabeled segments generated for epoch {epoch}. Skipping further training.")

                break



            logger.info("Epoch %d/%d: generated %d unlabeled segments", epoch + 1, args.epochs, len(segments))



            unlabeled_dataset = UnlabeledWindowDataset(

                segments=segments,

                target_sr=config["sr"],

                min_window_sec=args.min_segment_duration,

            )

            logger.info(

                "Epoch %d/%d: building unlabeled loader with %d segments (batch size %d)",

                epoch + 1,

                args.epochs,

                len(unlabeled_dataset),

                config["batch_size"],

            )

            unlabeled_loader = DataLoader(

                unlabeled_dataset,

                batch_size=config["batch_size"],

                shuffle=True,

                collate_fn=collate_unlabeled,

                num_workers=args.num_workers,

                pin_memory=use_pinned,

                persistent_workers=args.num_workers > 0,

            )

            model.train()

            epoch_sup_loss = 0.0

            epoch_unsup_loss = 0.0

            selected_frames_total = 0

            total_frames_considered = 0

            batches_run = 0

            epoch_start_time = time.time()

            total_batches_epoch = len(unlabeled_loader)

            if args.max_unsupervised_batches:

                total_batches_epoch = min(total_batches_epoch, args.max_unsupervised_batches)



            progress_bar = None

            if total_batches_epoch > 0:

                progress_bar = tqdm(

                    total=total_batches_epoch,

                    desc=f"Epoch {epoch + 1}/{args.epochs}",

                    unit="batch",

                    leave=False,

                    dynamic_ncols=True,

                )

                last_pbar_step = 0  # ★ 追加



            # ★ ここから追加：50イテ区間用の累積

            interval_sup_loss = 0.0

            interval_unsup_loss = 0.0

            interval_batches = 0



            if hasattr(train_loader.sampler, 'set_epoch'):

                train_loader.sampler.set_epoch(epoch)

            supervised_iter = infinite_data_loader(train_loader)

            supervised_batches_used = 0



            for batch_idx, unlabeled_batch in enumerate(unlabeled_loader):

                if args.max_unsupervised_batches and batch_idx >= args.max_unsupervised_batches:

                    break

                if args.max_supervised_batches and supervised_batches_used >= args.max_supervised_batches:

                    break



                features, labels, masks, _ = next(supervised_iter)

                supervised_batches_used += 1

                sup_count = int(features.size(0))



                if weak_augment_fn is not None:

                    hop = config["ssl_hop_length"]

                    valid_frame_counts = masks.sum(dim=1)

                    supervised_lengths = []

                    for waveform, frame_count in zip(features, valid_frame_counts):

                        valid_samples = int(frame_count.item()) * hop

                        if valid_samples <= 0:

                            valid_samples = waveform.shape[0]

                        supervised_lengths.append(min(waveform.shape[0], valid_samples))

                    features = apply_augmentation_batch(features, supervised_lengths, weak_augment_fn, config["sr"])



                features = features.to(device)

                labels = labels.to(device)

                masks = masks.to(device)



                unlabeled_features, unlabeled_lengths, _ = unlabeled_batch

                length_list = [int(v.item()) for v in unlabeled_lengths]

                teacher_unlabeled = unlabeled_features.clone()

                if weak_augment_fn is not None:

                    teacher_unlabeled = apply_augmentation_batch(

                        teacher_unlabeled, length_list, weak_augment_fn, config["sr"])

                if strong_augment_fn is not None:

                    student_unlabeled = apply_augmentation_batch(

                        unlabeled_features, length_list, strong_augment_fn, config["sr"])

                else:

                    student_unlabeled = unlabeled_features

                unlabeled_lengths = unlabeled_lengths.to(device)

                teacher_unlabeled = teacher_unlabeled.to(device)

                student_unlabeled = student_unlabeled.to(device)

                unsup_count = int(student_unlabeled.size(0))



                optimizer.zero_grad()

                logits_sup = model(features)

                sup_loss = masked_bce_loss(logits_sup, labels, masks, criterion_sup)



                with torch.no_grad():

                    teacher_logits = teacher_model(teacher_unlabeled)

                pseudo_distributions, pseudo_mask, selected_frames, total_frames = create_pseudo_labels(

                    teacher_logits,

                    unlabeled_lengths,

                    config["ssl_hop_length"],

                    args.confidence_threshold,

                    min_duration_frames,

                    swallow_idx,

                    args.pseudo_temperature,

                    use_adaptive_threshold=args.adaptive_threshold,

                )



                # --- ★追加ここから: 疑似ラベルの内訳集計ロジック ---

                with torch.no_grad():

                    # 1. Teacherが「これだ」と思ったクラス（最大確率のインデックス）

                    teacher_pred_indices = torch.argmax(teacher_logits, dim=1) # [B, T]

                    

                    # 2. マスクが1（採用）の部分だけのインデックスを取り出す

                    valid_mask_bool = pseudo_mask.bool()

                    selected_labels = teacher_pred_indices[valid_mask_bool]



                    # 3. クラスごとのカウント

                    n_selected = selected_labels.numel()

                    n_swallow_pseudo = (selected_labels == swallow_idx).sum().item()

                    n_others_pseudo = n_selected - n_swallow_pseudo # 単純化のためswallow以外をothers扱い



                    # 4. 無視された（マスクされた）フレーム数

                    # total_frames は create_pseudo_labels が返す「パディングを除いた有効フレーム総数」

                    n_masked = max(0, total_frames - n_selected)



                # ログ表示用に割合を計算（バッチ単位）

                ratio_swallow = (n_swallow_pseudo / total_frames * 100) if total_frames > 0 else 0

                ratio_others = (n_others_pseudo / total_frames * 100) if total_frames > 0 else 0

                ratio_masked = (n_masked / total_frames * 100) if total_frames > 0 else 0

                

                logits_unsup = model(student_unlabeled)

                total_frames_considered += total_frames

                selected_frames_total += selected_frames



                pseudo_distributions = pseudo_distributions.to(device)

                pseudo_mask = pseudo_mask.to(device)

                unsup_loss = masked_bce_loss(logits_unsup, pseudo_distributions, pseudo_mask, criterion_unsup)

                total_loss = args.w_sup * sup_loss + args.w_unsup * unsup_loss



                if torch.isnan(total_loss):

                    logger.warning("Encountered NaN loss; skipping update.")

                    continue



                total_loss.backward()

                if args.grad_clip > 0:

                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

                optimizer.step()

                global_step += 1

                update_teacher_ema(model, teacher_model, args.ema_decay, global_step)

                teacher_model.eval()



                # ★ 区間用の累積（log_interval ごとに平均を出す）

                interval_sup_loss += sup_loss.item()

                interval_unsup_loss += unsup_loss.item()

                interval_batches += 1



                # ★ N step ごとの interim validation + early stopping + checkpoint

                if args.val_interval_steps and (global_step % args.val_interval_steps == 0):

                    logger.info(

                        "Global step %d: running interim validation (up to %s batches).",

                        global_step,

                        str(args.max_val_batches) if args.max_val_batches is not None else "all",

                    )

                    val_limit = args.max_val_batches

                    val_loss_tmp, val_metrics_tmp = evaluate_epoch(

                        model,

                        val_loader,

                        eval_criterion,  # 上で定義した nn.CrossEntropyLoss()

                        config,

                        local_rank=0,

                        world_size=1,

                        compute_event_metrics=True,

                        epoch_num=None,

                        num_epochs_total=args.epochs,

                        limit=val_limit,

                    )

                    val_f1_tmp = None

                    if val_metrics_tmp:

                        _, _, val_f1_tmp = val_metrics_tmp.get(0.5, (0.0, 0.0, 0.0))



                    if val_f1_tmp is not None:

                        logger.info(

                            "Interim validation @ step %d: loss=%.4f | F1@0.5=%.3f",

                            global_step,

                            val_loss_tmp,

                            val_f1_tmp,

                        )

                    else:

                        logger.info(

                            "Interim validation @ step %d: loss=%.4f | F1@0.5=N/A",

                            global_step,

                            val_loss_tmp,

                        )

                    log_iou_metrics(val_metrics_tmp, f"Validation (step {global_step})")



                    # --- ここから early stopping & checkpoint 更新も実行 ---

                    prev_best_loss = best_val_loss

                    best_val_loss = min(prev_best_loss, val_loss_tmp)



                    if monitor_metric == "f1":

                        if val_f1_tmp is None:

                            current_metric_tmp = float('-inf')

                            logger.warning(

                                "Interim validation F1@0.5 unavailable; treating as -inf for early stopping."

                            )

                        else:

                            current_metric_tmp = float(val_f1_tmp)

                    else:

                        current_metric_tmp = float(val_loss_tmp)



                    if maximize_metric:

                        metric_improved = current_metric_tmp > best_metric_value

                    else:

                        metric_improved = current_metric_tmp < best_metric_value



                    # latest checkpoint は常に更新

                    save_checkpoint(

                        model,

                        teacher_model,

                        optimizer,

                        epoch,          # 現在の epoch index

                        output_dir,

                        best_val_loss,

                        "latest",

                        global_step,

                        best_metric_value,

                        monitor_metric,

                    )

                    torch.save(teacher_model.state_dict(), output_dir / "latest_teacher.pth")



                    if metric_improved:

                        best_metric_value = current_metric_tmp

                        best_metric_epoch = epoch

                        epochs_since_improvement = 0

                        logger.info(

                            "Validation(%s) %s improved to %.4f (epoch %d, step %d)",

                            f"step {global_step}",

                            monitor_metric.upper(),

                            current_metric_tmp,

                            epoch + 1,

                            global_step,

                        )

                        torch.save(model.state_dict(), output_dir / "best.pth")

                        torch.save(teacher_model.state_dict(), output_dir / "best_teacher.pth")

                        logger.info("New best model checkpointed based on %s (interim).", monitor_metric.upper())

                    else:

                        epochs_since_improvement += 1

                        logger.info(

                            "No improvement in %s (interim). %d/%d validations since last best.",

                            monitor_metric.upper(),

                            epochs_since_improvement,

                            patience,

                        )



                    # ★ validation が終わったら training mode に戻す

                    model.train()

                    teacher_model.eval()



                    if epochs_since_improvement >= patience:

                        early_stop_triggered = True

                        logger.info(

                            "Early stopping triggered by interim validation "

                            "after %d validations without improvement in %s.",

                            patience,

                            monitor_metric.upper(),

                        )

                        break  # バッチループを抜ける



                if real_eval_enabled and quick_real_eval_interval > 0 and global_step % quick_real_eval_interval == 0:

                    quick_eval_model = teacher_model if teacher_model is not None else model

                    quick_results = evaluate_real_data_validation(

                        quick_eval_model,

                        config,

                        device,

                        exist_label_dir,

                        quick_real_eval_dir,

                        real_eval_threshold,

                        real_eval_chunk_sec,

                        real_eval_overlap_sec,

                    )

                    if quick_results is None:

                        logger.warning(

                            "Quick real-data validation skipped at step %d due to configuration issues.",

                            global_step,

                        )

                    elif not quick_results:

                        logger.warning("Quick real-data validation returned no evaluable subjects at step %d.", global_step)

                    else:

                        scores = [score for _, score in quick_results]

                        avg_quick_f1 = float(np.mean(scores)) if scores else 0.0

                        logger.info("Quick real-data validation @ step %d: avg F1@0.3 = %.3f", global_step, avg_quick_f1)



                epoch_sup_loss += sup_loss.item()

                epoch_unsup_loss += unsup_loss.item()

                batches_run += 1



                # tqdm と logger の更新を統合（50イテ平均を使用）

                if args.log_interval and ((batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == total_batches_epoch):

                    pseudo_ratio_running = (

                        selected_frames_total / total_frames_considered

                        if total_frames_considered > 0 else 0.0

                    )

                    

                    # 直近 interval_batches 分の平均を計算

                    avg_sup_interval = interval_sup_loss / max(1, interval_batches)

                    avg_unsup_interval = interval_unsup_loss / max(1, interval_batches)



                    # tqdm 更新

                    if progress_bar is not None:

                        step_inc = (batch_idx + 1) - last_pbar_step

                        if step_inc > 0:

                            progress_bar.update(step_inc)

                            last_pbar_step = batch_idx + 1



                        progress_bar.set_postfix({

                            "sup": sup_count,

                            "unsup": unsup_count,

                            "L_sup": f"{avg_sup_interval:.4f}",   # ★ 50イテ平均

                            "L_uns": f"{avg_unsup_interval:.4f}", # ★ 50イテ平均

                            "w_sup": f"{args.w_sup:.2f}",

                            "w_uns": f"{args.w_unsup:.2f}",

                            "pseudo": f"{pseudo_ratio_running:.3f}",

                        })



                    # logger 出力

                    # logger出力の更新

                    logger.info(

                        "Epoch %d/%d | Batch %d | L_sup %.4f | L_uns %.4f | Pseudo: Swal %.1f%% / Oth %.1f%% / Mask %.1f%%",

                        epoch + 1,

                        args.epochs,

                        batch_idx + 1,

                        avg_sup_interval,

                        avg_unsup_interval,

                        ratio_swallow,  # 嚥下になった割合

                        ratio_others,   # othersになった割合

                        ratio_masked,   # ラベルなし(無視)になった割合

                    )



                    # ★ この区間はリセット（次の50イテ用）

                    interval_sup_loss = 0.0

                    interval_unsup_loss = 0.0

                    interval_batches = 0



                if args.max_unsupervised_batches and batches_run >= args.max_unsupervised_batches:

                    break



            if early_stop_triggered:

                if progress_bar is not None:

                    progress_bar.close()

                logger.info(

                    "Early stopping was triggered during interim validation in epoch %d. "

                    "Skipping epoch-end validation.",

                    epoch + 1,

                )

                break



            if batches_run == 0:

                logger.warning("No batches processed this epoch.")

                if progress_bar is not None:

                    progress_bar.close()

                continue



            if progress_bar is not None:

                progress_bar.close()

            epoch_duration = time.time() - epoch_start_time



            avg_sup_loss = epoch_sup_loss / batches_run

            avg_unsup_loss = epoch_unsup_loss / max(1, batches_run)

            pseudo_ratio = (selected_frames_total / total_frames_considered) if total_frames_considered > 0 else 0.0

            logger.info(

                "Epoch %d/%d | L_sup %.4f | L_unsup %.4f | w_sup %.2f | w_uns %.2f | pseudo_ratio %.3f",

                epoch + 1,

                args.epochs,

                avg_sup_loss,

                avg_unsup_loss,

                args.w_sup,

                args.w_unsup,

                pseudo_ratio,

            )

            if total_frames_considered > 0:

                logger.info(

                    "Epoch %d summary: pseudo_frames=%d/%d (%.2f%%), batches=%d",

                    epoch + 1,

                    selected_frames_total,

                    total_frames_considered,

                    100.0 * selected_frames_total / total_frames_considered,

                    batches_run,

                )

            else:

                logger.warning(

                    "Epoch %d summary: No unlabeled frames were considered!",

                    epoch + 1,

                )

            epoch_durations.append(epoch_duration)

            remaining_epochs = args.epochs - (epoch + 1)

            if remaining_epochs > 0:

                mean_epoch = sum(epoch_durations) / len(epoch_durations)

                eta_epochs = mean_epoch * remaining_epochs

                logger.info(

                    "Epoch %d/%d completed in %s | approx %d epochs remaining (~%s)",

                    epoch + 1,

                    args.epochs,

                    format_seconds(epoch_duration),

                    remaining_epochs,

                    format_seconds(eta_epochs),

                )

            else:

                logger.info(

                    "Epoch %d/%d completed in %s",

                    epoch + 1,

                    args.epochs,

                    format_seconds(epoch_duration),

                )

            logger.info("Epoch %d/%d: running validation", epoch + 1, args.epochs)



            val_limit = args.max_val_batches

            val_loss, val_metrics = evaluate_epoch(

                model,

                val_loader,

                nn.CrossEntropyLoss(),  # reduction='none' を削除

                config,

                local_rank=0,

                world_size=1,

                compute_event_metrics=True,

                epoch_num=epoch + 1,

                num_epochs_total=args.epochs,

                limit=val_limit,

            )

            val_f1 = None

            if val_metrics:

                _, _, val_f1 = val_metrics.get(0.5, (0.0, 0.0, 0.0))

            if val_f1 is not None:

                logger.info("Validation loss: %.4f | F1@0.5: %.3f", val_loss, val_f1)

            else:

                logger.info("Validation loss: %.4f | F1@0.5: N/A", val_loss)

            log_iou_metrics(val_metrics, "Validation")



            if real_eval_enabled:

                real_eval_results = evaluate_real_data_validation(

                    teacher_model,

                    config,

                    device,

                    exist_label_dir,

                    real_eval_dir,

                    real_eval_threshold,

                    real_eval_chunk_sec,

                    real_eval_overlap_sec,

                )

                if real_eval_results is None:

                    logger.warning("Real-data validation skipped for epoch %d due to configuration issues.", epoch + 1)

                elif not real_eval_results:

                    logger.warning("Real-data validation returned no evaluable subjects for epoch %d.", epoch + 1)

                else:

                    scores = [score for _, score in real_eval_results]

                    for subject, f1_score in real_eval_results:

                        logger.info("Real validation %s F1@0.3: %.3f", subject, f1_score)

                    avg_real_f1 = float(np.mean(scores)) if scores else 0.0

                    logger.info("Real validation average F1@0.3: %.3f", avg_real_f1)

                    if avg_real_f1 > best_real_avg_f1 + 1e-6:

                        best_real_avg_f1 = avg_real_f1

                        best_real_epoch = epoch

                        real_epochs_since_improvement = 0

                        logger.info("Real validation average improved to %.3f (epoch %d)", avg_real_f1, epoch + 1)

                    else:

                        real_epochs_since_improvement += 1

                        comparison_best = best_real_avg_f1 if best_real_avg_f1 > -float('inf') else float('nan')

                        logger.info(

                            "Real validation average did not improve (current %.3f vs best %.3f) [%d/%d]",

                            avg_real_f1,

                            comparison_best,

                            real_epochs_since_improvement,

                            real_eval_patience,

                        )

                    if real_epochs_since_improvement >= real_eval_patience:

                        early_stop_triggered = True

                        real_eval_stop_triggered = True

                        logger.info(

                            "Early stopping triggered by real-data validation after %d epochs without improvement.",

                            real_eval_patience,

                        )



            prev_best_loss = best_val_loss

            best_val_loss = min(prev_best_loss, val_loss)



            if monitor_metric == "f1":

                if val_f1 is None:

                    current_metric = float('-inf')

                    logger.warning("Validation F1@0.5 unavailable; treating as -inf for early stopping.")

                else:

                    current_metric = float(val_f1)

            else:

                current_metric = float(val_loss)



            if maximize_metric:

                metric_improved = current_metric > best_metric_value

            else:

                metric_improved = current_metric < best_metric_value



            if metric_improved:

                best_metric_value = current_metric

                best_metric_epoch = epoch

                epochs_since_improvement = 0

                logger.info(

                    "Validation %s improved to %.4f (epoch %d)",

                    monitor_metric.upper(),

                    current_metric,

                    epoch + 1,

                )

            else:

                epochs_since_improvement += 1



            save_checkpoint(

                model,

                teacher_model,

                optimizer,

                epoch,

                output_dir,

                best_val_loss,

                "latest",

                global_step,

                best_metric_value,

                monitor_metric,

            )

            torch.save(teacher_model.state_dict(), output_dir / "latest_teacher.pth")

            if metric_improved:

                torch.save(model.state_dict(), output_dir / "best.pth")

                torch.save(teacher_model.state_dict(), output_dir / "best_teacher.pth")

                logger.info("New best model checkpointed based on %s.", monitor_metric.upper())



          #  if real_eval_stop_triggered:

          #      break



            if epochs_since_improvement >= patience:

                early_stop_triggered = True

                logger.info(

                    "Early stopping triggered after %d epochs without improvement in %s.",

                    patience,

                    monitor_metric.upper(),

                )

                break



    if monitor_metric == "f1" and np.isinf(best_metric_value):

        best_metric_value = float('-inf')

    display_metric = best_metric_value if not np.isinf(best_metric_value) else float('nan')

    if best_metric_epoch >= 0:

        logger.info(

            "Training finished. Best %s %.4f achieved at epoch %d.",

            monitor_metric.upper(),

            display_metric,

            best_metric_epoch + 1,

        )

    else:

        logger.info("Training finished. No improvement recorded for %s.", monitor_metric.upper())



    if real_eval_enabled and best_real_epoch >= 0 and best_real_avg_f1 > -float('inf'):

        logger.info("Best real-data F1@0.3 %.4f achieved at epoch %d.", best_real_avg_f1, best_real_epoch + 1)

    elif real_eval_enabled:

        logger.info("Real-data validation was enabled but no improvement was observed.")



    selected_path = load_for_evaluation(args.test_checkpoint, model, teacher_model, output_dir)

    if not selected_path:

        logger.warning("Proceeding with current in-memory weights; no checkpoint matched preference '%s'.", args.test_checkpoint)

    eval_model = teacher_model if teacher_model is not None else model

    run_test_evaluation(eval_model, config, device, output_dir, args.test_num_workers, dataset_label="test")

    if closed_test_path:

        primary_test_path = Path(config["test_json"]).resolve()

        closed_resolved = closed_test_path.resolve()

        if primary_test_path != closed_resolved:

            closed_label = closed_test_path.stem or "closed_test"

            if closed_label.lower() == "test":

                closed_label = "closed_test"

            run_test_evaluation(

                eval_model,

                config,

                device,

                output_dir,

                args.test_num_workers,

                dataset_label=closed_label,

                test_json_override=closed_test_path,

            )









def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Semi-supervised fine-tuning with pseudo labels")

    parser.add_argument("--config", default=str(SCRIPT_DIR / "config_FT_meanteacher_real.json"))

    parser.add_argument("--experiment-key", default=None,

                        help="Experiment key(s) defined in the config. "

                             "Provide a comma-separated list or leave unset/\"all\" to run every experiment.")

    parser.add_argument("--checkpoint-dir", default=None,

                        help="Root directory for checkpoints. Defaults to the output directory of each experiment.")

    parser.add_argument("--unlabeled-root", default=str(SCRIPT_DIR / "prepare/downloaded_folder/behavior/wav_sync"))

    parser.add_argument("--output-dir", default=None,

                        help="Root directory for outputs. Defaults to <base_save_dir>/<experiment name> per experiment.")

    parser.add_argument("--inference", action="store_true",

                        help="Run inference mode on a specified WAV file")

    parser.add_argument("--inference-wav", type=str, default=None,

                        help="Path to WAV file for inference mode")

    parser.add_argument("--inference-threshold", type=float, default=0.5,

                        help="Detection threshold for inference mode (default: 0.5)")

    parser.add_argument("--inference-chunk-sec", type=float, default=10.0,

                        help="Chunk duration in seconds for inference (default: 10.0)")

    parser.add_argument("--inference-overlap-sec", type=float, default=1.0,

                        help="Overlap duration in seconds for inference (default: 1.0)")

    parser.add_argument("--inference-debug", action="store_true",

                        help="Use direct model inference instead of inference_wav module for debugging")

    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--batch-size", type=int, default=None)

    parser.add_argument("--lr", type=float, default=1e-07)

    parser.add_argument("--lambda-max", type=float, default=0.8)

    parser.add_argument("--lambda-rampup-epochs", type=int, default=8)

    parser.add_argument("--w-sup", type=float, default=1.0,

                        help="Supervised loss weight (default: 1.0)")

    parser.add_argument("--w-unsup", type=float, default=1.0,

                        help="Unsupervised loss weight (default: 1.0)")

    parser.add_argument("--ema-decay", type=float, default=0.999,

                        help="EMA decay applied to the teacher model parameters (default: 0.999)")

    parser.add_argument("--no-augmentation", action="store_true",

                        help="Disable weak/strong audio data augmentation (default: enabled when audiomentations is installed).")

    parser.add_argument("--confidence-threshold", type=float, default=0.9999)

    parser.add_argument("--adaptive-threshold", dest="adaptive_threshold", action="store_true", default=False,

                        help="Use adaptive confidence threshold based on pseudo-label statistics")

    parser.add_argument("--no-adaptive-threshold", dest="adaptive_threshold", action="store_false",

                        help="Disable adaptive confidence thresholding")

    parser.add_argument("--exist-label-dir", type=str, default=str(SCRIPT_DIR / "dataset/exist_label"),

                        help="Directory containing real-data WAV/TXT pairs for validation")

    parser.add_argument("--real-eval-threshold", type=float, default=0.5,

                        help="Detection threshold used during real-data validation (default: 0.5)")

    parser.add_argument("--real-eval-chunk-sec", type=float, default=10.0,

                        help="Chunk duration in seconds for real-data validation inference (default: 10.0)")

    parser.add_argument("--real-eval-overlap-sec", type=float, default=1.0,

                        help="Overlap duration in seconds for real-data validation inference (default: 3.0)")

    parser.add_argument("--real-eval-patience", type=int, default=10,

                        help="Epochs without real-data F1 improvement before triggering early stop (default: 10)")

    parser.add_argument("--real-eval", action="store_true",

                        help="Enable real-data validation using WAV/TXT pairs under --exist-label-dir.")

    parser.add_argument("--min-duration-ms", type=float, default=80.0)

    parser.add_argument("--min-segment-duration", type=float, default=10.0,

                        help="Minimum unlabeled window length in seconds (default: 10.0)")

    parser.add_argument("--max-segment-duration", type=float, default=15.0,

                        help="Maximum unlabeled window length in seconds (default: 15.0)")

    parser.add_argument("--segment-duration", type=float, default=None,

                        help=argparse.SUPPRESS)

    parser.add_argument("--overlap-seconds", type=float, default=1.0,

                        help="Overlap between consecutive unlabeled windows in seconds (default: 1.0)")

    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--max-supervised-batches", type=int, default=None)

    parser.add_argument("--max-unsupervised-batches", type=int, default=None)

    parser.add_argument("--max-val-batches", type=int, default=None)

    parser.add_argument("--log-interval", type=int, default=50,

                        help="How many batches between progress logs (default: 50)")

    parser.add_argument("--test-mode", action="store_true",

                        help="Skip training and run test evaluation only using the available checkpoints.")

    parser.add_argument("--test-num-workers", type=int, default=0,

                        help="Number of DataLoader workers to use during test evaluation (default: 0)")

    parser.add_argument("--test-checkpoint", type=str, default="best", choices=("best", "latest"),

                        help="Select which checkpoint to evaluate ('best' or 'latest').")

    parser.add_argument("--merge-gap-ms", type=float, default=100.0,

                        help="Merge predicted gaps shorter than this duration (ms) during evaluation (default: 100 ms)")

    parser.add_argument("--pseudo-temperature", type=float, default=1,

                        help="Sharpen temperature applied to unlabeled predictions before computing unsupervised loss (lower is sharper).")

    parser.add_argument("--resume", nargs="?", const="latest", default=None, metavar="CKPT",

                        help="Resume training from a checkpoint. If no path is provided, uses <output_dir>/latest.pth.")

    parser.add_argument("--closed-test", nargs="?", const="auto", default=None, metavar="JSON",

                        help="Also evaluate on a closed/merged test set. Provide the annotation JSON or leave blank to auto-detect.")

    parser.add_argument("--early-stop-patience", type=int, default=10,

                        help="Number of epochs without validation improvement before early stopping triggers (default: 10).")

    parser.add_argument("--early-stop-metric", type=str, default="loss",

                        choices=("loss", "f1"),

                        help="Metric used to decide early stopping: 'loss' minimizes, 'f1' maximizes (default: f1).")

    parser.add_argument("--val-interval-steps", type=int, default=300,

                        help="Run interim validation every N global steps during training (0 disables).")



    args = parser.parse_args(argv)



    if args.segment_duration is not None:

        args.max_segment_duration = args.segment_duration



    if args.max_segment_duration < args.min_segment_duration:

        logger.warning("max_segment_duration was smaller than min_segment_duration; clamping to min value.")

        args.max_segment_duration = args.min_segment_duration



    if args.log_interval is not None and args.log_interval <= 0:

        logger.warning("log_interval must be positive; defaulting to 50.")

        args.log_interval = 50



    return args





if __name__ == "__main__":

    args = parse_args()

    config_path = Path(args.config).resolve()

    experiments_dict = list_available_experiments(config_path)

    available_keys = list(experiments_dict.keys())

    selected_keys = select_experiment_keys(args.experiment_key, available_keys)



    multi_run = len(selected_keys) > 1

    user_output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    user_checkpoint_root = Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else None



    for key in selected_keys:

        cloned_args = argparse.Namespace(**vars(args))

        cloned_args.experiment_key = key



        if user_output_root is not None:

            target_output = user_output_root / (key or "default") if multi_run else user_output_root

            cloned_args.output_dir = str(target_output)

        else:

            cloned_args.output_dir = None



        if user_checkpoint_root is not None:

            target_checkpoint = user_checkpoint_root / (key or "default") if multi_run else user_checkpoint_root

            cloned_args.checkpoint_dir = str(target_checkpoint)

        else:

            cloned_args.checkpoint_dir = None



        train(cloned_args)