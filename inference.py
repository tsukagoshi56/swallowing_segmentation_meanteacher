"""
Inference and testing functions for sound event detection models
"""
import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
import logging
from pathlib import Path
import json
import numpy as np
from functools import partial
import time
import datetime
from training import evaluate_epoch
from metrics import (
    print_event_statistics, collect_raw_iou, collect_matched_iou,
    calculate_event_metrics_by_class_and_food, collect_raw_iou_by_food
)
from data_utils import SoundEventDataset, pad_collate
from ddp_utils import is_main_process
from models import EventDetector
from metrics import (
    calculate_event_metrics, print_event_statistics, model_output_to_intervals,
    calculate_event_metrics_by_class_and_food, collect_raw_iou, collect_matched_iou, collect_raw_iou_by_food
)
from exp_config import IOU_THRESHOLDS, BASE_CONFIG
# Override LABEL_SHORT for 2-class model
LABEL_SHORT = {"chewing": "ch", "swallowing": "sw", "speech": "sp", "others": "ot"}
import sys

# インファレンス用の追加設定
OUTPUT_DIR = './output/inference_pre'
BASE_SAVE_DIR = './save'
INPUT_FOLDER = '/home/tsukagoshitoshihiro/workspace/ICASSP/Semi_supervise/dataset/exist_label'  # 推論したいWAVファイルが入っているフォルダのパス

# モデルとコンフィグの指定（簡単に変更可能）
MODEL_PATH = '/home/tsukagoshitoshihiro/workspace/ICASSP/Semi_supervise/results_classification_comparison_re/2class_swallowing_vs_others/best.pth'
CONFIG_PATH = None  # Noneの場合は自動検出、または'config_file.json'を指定

logger = logging.getLogger(__name__)

def test_model(base_config: Dict[str, Any], 
               experiment_config: Dict[str, Any], 
               local_rank: int, world_size: int, 
               test_batch_limit: Optional[int] = None):
    """
    Test a model with the given configuration
    
    Args:
        base_config: Base configuration dictionary
        experiment_config: Experiment-specific configuration dictionary
        local_rank: Process rank
        world_size: Total number of processes
        test_batch_limit: Optional batch limit for testing
    """
    # 1. 設定を組み合わせる
    config = {**base_config, **experiment_config}
    exp_name = config["name"]
    save_dir = Path(config["base_save_dir"]) / exp_name
    
    # Setup logger only on main process for file writing
    test_logger = logging.getLogger(f"{exp_name}_test_rank{local_rank}")
    test_logger.propagate = False 
    test_logger.handlers.clear()
    test_logger.setLevel(logging.INFO)

    # Formatter for test results logging (file and console)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    if is_main_process(local_rank):
        save_dir.mkdir(parents=True, exist_ok=True)
        log_file = save_dir / f"{exp_name}_test_results.log"
        
        # File handler for test results
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(file_formatter)
        test_logger.addHandler(fh)

        # Console handler for test_logger only
        console_sh = logging.StreamHandler()
        console_sh.setFormatter(console_formatter)
        test_logger.addHandler(console_sh)

    # Log test mode banner
    if is_main_process(local_rank):
        test_logger.info(f"=== テストモード: {exp_name} ===")
        test_logger.info(f"結果はこのパスに保存されます: {save_dir}")

    # Model initialization
    try:
        model_instance = EventDetector(
            config=config,
            ssl_model_name=config.get("ssl_model_name"),
            freeze_feature_extractor=config.get("freeze_fe", config["default_freeze_fe"]),
            freeze_transformer_layers=config.get("freeze_transformer", config["default_freeze_transformer"]),
            architecture=config.get("architecture", "gru"),
            feature_type=config.get("feature_type", "raw"),
            local_rank=local_rank
        ).to(config["device"])

        if world_size > 1:
            model_instance = nn.parallel.DistributedDataParallel(
                model_instance, device_ids=[config["device"]], output_device=config["device"], find_unused_parameters=True
            )
        
        if is_main_process(local_rank):
            test_logger.info("モデルが正常に初期化されました。")
            actual_model = model_instance.module if world_size > 1 else model_instance
            num_params = sum(p.numel() for p in actual_model.parameters())
            num_trainable = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
            test_logger.info(f"Total parameters: {num_params:,}")
            test_logger.info(f"Trainable parameters: {num_trainable:,}")

    except Exception as e:
        if is_main_process(local_rank):
            test_logger.exception(f"モデルの初期化に失敗しました: {e}")
        return

    # Load test data
    try:
        test_json_path = Path(config["test_json"])
        with open(test_json_path, 'r') as f:
            test_ann = json.load(f)
        
        dataset_test = SoundEventDataset(test_ann, config, config.get("feature_type", "raw"), 1.0, local_rank)
        
        sampler_test = None
        shuffle_test = False
        if world_size > 1:
            sampler_test = DistributedSampler(dataset_test, num_replicas=world_size, rank=local_rank, shuffle=False)
            
        collate_fn = partial(pad_collate, config=config)
        num_loader_workers = min(4, (os.cpu_count() or 1) // max(1, world_size))
        if is_main_process(local_rank):
            test_logger.info(f"Using {num_loader_workers} workers per DDP process for DataLoader.")

        dataloader_test = DataLoader(
            dataset_test, batch_size=config["batch_size"], shuffle=shuffle_test, 
            sampler=sampler_test, collate_fn=collate_fn, 
            num_workers=num_loader_workers, pin_memory=True,
            persistent_workers=num_loader_workers > 0
        )
        if is_main_process(local_rank):
            test_logger.info(f"テストデータセットをロード。Rank0サンプル概算: {len(dataset_test) // world_size}")
    except Exception as e:
        if is_main_process(local_rank):
            test_logger.exception(f"テストデータのロードに失敗: {e}")
        return

    # Load model weights
    ckpt_to_load_path_str = None
    # Define save_dir_path here so it's available for logging outside the main_process block if needed,
    # and for the error message if no checkpoint is found.
    save_dir_path = Path(config["base_save_dir"]) / exp_name

    if is_main_process(local_rank):
        best_ckpt_path = save_dir_path / "best.pth"
        latest_ckpt_path = save_dir_path / "latest.pth"

        if best_ckpt_path.exists():
            test_logger.info(f"最良モデルの重みを読み込みます: {best_ckpt_path}")
            ckpt_to_load_path_str = str(best_ckpt_path)
        elif latest_ckpt_path.exists():
            test_logger.warning(f"最良モデル ({best_ckpt_path}) が見つからず。最新モデル ({latest_ckpt_path}) を読み込みます。")
            ckpt_to_load_path_str = str(latest_ckpt_path)
        # If neither exists, ckpt_to_load_path_str remains None.
    
    if world_size > 1:
        # Broadcast the path string (or None) from rank 0 to all other ranks
        path_list = [ckpt_to_load_path_str] 
        dist.broadcast_object_list(path_list, src=0)
        ckpt_to_load_path_str = path_list[0]

    # Check if a checkpoint path was determined and broadcasted successfully.
    if not ckpt_to_load_path_str:
        if is_main_process(local_rank):
            # Log the specific directory that was checked.
            test_logger.error(f"テスト用のモデル重みファイル (best.pth または latest.pth) がディレクトリ '{save_dir_path}' に見つかりませんでした。このモデルのテストを中止します。")
        if world_size > 1 and dist.is_initialized():
            dist.barrier() 
        return # Abort test if no checkpoint file was found.

    # If we reach here, ckpt_to_load_path_str is a non-None string.
    try:
        if is_main_process(local_rank): # Log intent to load only on main process
             test_logger.info(f"モデル重み {ckpt_to_load_path_str} のロードを試みます...")
        
        # Proper map_location: if device is int, convert to torch.device string
        dev = config.get("device")
        if isinstance(dev, int):
            map_loc = f"cuda:{dev}" if torch.cuda.is_available() else "cpu"
        else:
            map_loc = dev
        checkpoint = torch.load(ckpt_to_load_path_str, map_location=map_loc)
        state_dict_to_load = checkpoint.get("model_state_dict", checkpoint)
        
        model_to_load = model_instance.module if world_size > 1 else model_instance
        model_to_load.load_state_dict(state_dict_to_load) 
        
        if is_main_process(local_rank):
            test_logger.info(f"モデル重みを {ckpt_to_load_path_str} から正常にロード完了。")

    except FileNotFoundError as e_fnf: 
        if is_main_process(local_rank):
            test_logger.error(f"モデル重みファイルが見つかりません ({ckpt_to_load_path_str}): {e_fnf}。このモデルのテストを中止します。")
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        return # Abort test
    except Exception as e: 
        if is_main_process(local_rank):
            test_logger.error(f"モデル重みのロード中にエラーが発生 ({ckpt_to_load_path_str}): {e}。このモデルのテストを中止します。")
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        return # Abort test

    # Test execution
    criterion = nn.BCEWithLogitsLoss()
    if is_main_process(local_rank):
        test_logger.info("テスト評価を開始します...")

    # Detailed Test execution and evaluation
    criterion = nn.BCEWithLogitsLoss()
    test_logger.info("=== テスト評価を開始します... ===")
    start_time = time.time()
    model_instance.eval()
    test_loss, overall_metrics, gt_list, pred_list, file_paths = evaluate_epoch(
        model_instance, dataloader_test, criterion, config,
        local_rank, world_size, compute_event_metrics=True,
        return_detailed=True, limit=test_batch_limit
    )
    elapsed = time.time() - start_time
    num_samples = len(file_paths)
    test_logger.info(f"Inference time: {elapsed:.2f}s for {num_samples} samples ({num_samples/elapsed:.2f} samples/s)")

    # Overall metrics
    test_logger.info(f"テスト損失: {test_loss:.4f}")
    test_logger.info("=== IoUごとのF1スコア ===")
    if overall_metrics:
        for thr, (p, r, f1) in sorted(overall_metrics.items()):
            test_logger.info(f"IoU > {thr:.2f}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
    else:
        test_logger.warning("Overall metrics are None, skipping IoU summary.")

    # Event statistics
    print_event_statistics(gt_list, pred_list, file_paths, config["classes"])

    # Raw/Matched IoU summaries
    raw_iou = collect_raw_iou(gt_list, pred_list, config["classes"])
    matched_iou = collect_matched_iou(gt_list, pred_list, config["classes"])
    test_logger.info("=== Raw IoU Summary ===")
    for cls, vals in raw_iou.items():
        avg = float(np.mean(vals)) if vals else 0.0
        test_logger.info(f"{cls}: {avg:.3f}")
    test_logger.info("=== Matched IoU Summary ===")
    for cls, vals in matched_iou.items():
        avg = float(np.mean(vals)) if vals else 0.0
        test_logger.info(f"{cls}: {avg:.3f}")

    # Class & Food detailed metrics (イベントクラスのみ)
    event_classes = [cls for cls in config["classes"] if cls != "others"]
    detailed = calculate_event_metrics_by_class_and_food(gt_list, pred_list, file_paths, event_classes)
    test_logger.info("=== クラス別・食材別詳細評価 ===")
    for category, class_data in detailed.items():
        test_logger.info(f"--- {category} ---")
        for cls, metrics_dict in class_data.items():
            test_logger.info(f"{cls}:")
            for thr, (p, r, f1) in sorted(metrics_dict.items()):
                test_logger.info(f"  IoU > {thr:.2f}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

    # Raw IoU by food and class (イベントクラスのみ)
    raw_food = collect_raw_iou_by_food(gt_list, pred_list, file_paths, event_classes)
    test_logger.info("=== 食材別 Raw IoU ===")
    for food, cls_vals in raw_food.items():
        line = f"{food}:"
        for cls in event_classes:
            vals = cls_vals.get(cls, [])
            avg = float(np.mean(vals)) if vals else 0.0
            line += f" {cls}={avg:.3f}"
        test_logger.info(line)

    # メトリクス保存
    metrics_file = save_dir / "detailed_test_metrics.json"
    try:
        out_data = {
            "test_loss": test_loss,
            "overall_metrics": {str(k): {"precision":p, "recall":r, "f1":f1} for k,(p,r,f1) in overall_metrics.items()},
            "detailed": {
                cat: {cls: {str(k):{"precision":p, "recall":r, "f1":f1} for k,(p,r,f1) in md.items()} for cls,md in cd.items()}
                for cat,cd in detailed.items()
            },
            "raw_iou_summary": raw_iou,
            "matched_iou_summary": matched_iou,
            "raw_iou_by_food": raw_food,
            "inference_time_s": elapsed
        }
        with open(metrics_file, 'w') as f:
            json.dump(out_data, f, indent=2)
        test_logger.info(f"詳細メトリクスを保存しました: {metrics_file}")
    except Exception as e:
        test_logger.error(f"メトリクス保存に失敗しました: {e}")

def inference_model(base_config: Dict[str, Any],
                    experiment_config: Dict[str, Any],
                    local_rank: int, world_size: int,
                    batch_limit: Optional[int] = None) -> None:
    """
    Run inference with a trained model
    
    Args:
        base_config: Base configuration dictionary
        experiment_config: Experiment-specific configuration dictionary
        local_rank: Process rank
        world_size: Total number of processes
        batch_limit: Optional batch limit
    """
    cfg = {**base_config, **experiment_config}
    exp_name = cfg["name"]
    
    # Output directories created by rank 0
    if is_main_process(local_rank):
        save_dir = Path(cfg["base_save_dir"]) / exp_name
        out_dir = save_dir / "test_inference"
        prob_dir = save_dir / "test_frame_probs"
        out_dir.mkdir(parents=True, exist_ok=True)
        prob_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{exp_name}] Inference output to: {out_dir} and {prob_dir}")

    # Model Loading
    model_instance = EventDetector(
        config=cfg, ssl_model_name=cfg.get("ssl_model_name"),
        freeze_feature_extractor=cfg.get("freeze_fe", cfg["default_freeze_fe"]),
        freeze_transformer_layers=cfg.get("freeze_transformer", cfg["default_freeze_transformer"]),
        architecture=cfg.get("architecture", "gru"), feature_type=cfg.get("feature_type", "raw"),
        local_rank=local_rank
    ).to(cfg["device"])

    if world_size > 1:
        model_instance = nn.parallel.DistributedDataParallel(
            model_instance, device_ids=[cfg["device"]], output_device=cfg["device"], find_unused_parameters=True
        )

    model_to_load = model_instance.module if world_size > 1 else model_instance
    
    ckpt_to_load_path_str_inf = None
    save_dir_path_inf = Path(cfg["base_save_dir"]) / exp_name # For logging

    if is_main_process(local_rank):
        best_ckpt_path = save_dir_path_inf / "best.pth"
        if best_ckpt_path.exists():
            logger.info(f"[{exp_name}] 推論用にモデルの重みを読み込みます: {best_ckpt_path}")
            ckpt_to_load_path_str_inf = str(best_ckpt_path)
        # If not found, ckpt_to_load_path_str_inf remains None.
    
    if world_size > 1:
        path_list_inf = [ckpt_to_load_path_str_inf]
        dist.broadcast_object_list(path_list_inf, src=0)
        ckpt_to_load_path_str_inf = path_list_inf[0]

    if not ckpt_to_load_path_str_inf: # Check if 'best.pth' was found and broadcasted
        if is_main_process(local_rank):
            logger.error(f"[{exp_name}] 推論用のモデル重みファイル (best.pth) が {save_dir_path_inf} に見つかりませんでした。このモデルの推論を中止します。")
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        return # Abort inference if no 'best.pth' is found.

    try:
        if is_main_process(local_rank): # Log intent to load
            logger.info(f"[{exp_name}] モデル重み {ckpt_to_load_path_str_inf} のロードを試みます...")

        checkpoint = torch.load(ckpt_to_load_path_str_inf, map_location=cfg["device"])
        # Ensure consistency with how checkpoints are saved (e.g. if they always contain 'model_state_dict')
        state_dict_to_load = checkpoint.get('model_state_dict', checkpoint) 
        model_to_load.load_state_dict(state_dict_to_load)
        
        if is_main_process(local_rank):
            logger.info(f"[{exp_name}] モデル重みを {ckpt_to_load_path_str_inf} から正常にロード完了。")

    except FileNotFoundError as e_fnf:
        if is_main_process(local_rank):
            logger.error(f"[{exp_name}] モデル重みファイルが見つかりません ({ckpt_to_load_path_str_inf}): {e_fnf}。このモデルの推論を中止します。")
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        return # Abort inference
    except Exception as e:
        if is_main_process(local_rank):
            logger.error(f"[{exp_name}] モデル重みのロード中にエラーが発生 ({ckpt_to_load_path_str_inf}): {e}。このモデルの推論を中止します。")
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        return # Abort inference

    # Data Loading
    with open(cfg["test_json"]) as f:
        test_ann_inf = json.load(f)
    ds_inf = SoundEventDataset(test_ann_inf, cfg, cfg.get("feature_type", "raw"), local_rank=local_rank)
    
    sampler_inf = None
    if world_size > 1:
        sampler_inf = DistributedSampler(ds_inf, num_replicas=world_size, rank=local_rank, shuffle=False)
    
    num_loader_workers_inf = min(4, (os.cpu_count() or 1) // max(1, world_size))
    dl_inf = DataLoader(
        ds_inf, batch_size=cfg["batch_size"], shuffle=False, sampler=sampler_inf,
        collate_fn=partial(pad_collate, config=cfg),
        num_workers=num_loader_workers_inf, pin_memory=True,
        persistent_workers=num_loader_workers_inf > 0
    )

    hop_time = cfg["ssl_hop_length"] / cfg["sr"]
    threshold = cfg["threshold"]
    classes = cfg["classes"]
    
    model_instance.eval()
    
    # Store results from this rank
    results_for_rank: List[Tuple[str, str, str]] = []

    with torch.no_grad():
        pbar_desc_inf = f"[{exp_name} Rank {local_rank}] Inference"
        pbar_inf = tqdm(dl_inf, desc=pbar_desc_inf, disable=not is_main_process(local_rank))

        for feats, _, paths_batch_rank in pbar_inf:
            feats = feats.to(cfg["device"])
            probs_batch_rank = torch.softmax(model_instance(feats), dim=1).cpu()

            for b_idx in range(probs_batch_rank.size(0)):
                wav_path_original = Path(paths_batch_rank[b_idx])
                current_probs = probs_batch_rank[b_idx]

                # Intervals
                intervals = model_output_to_intervals(
                    current_probs, threshold, hop_time, classes, local_rank
                )
                
                interval_lines = []
                for cls_name in classes:
                    short_label = LABEL_SHORT.get(cls_name, cls_name)
                    for st, ed in intervals[cls_name]:
                        interval_lines.append(f"{st:.6f}\t{ed:.6f}\t{short_label}")
                interval_lines.sort(key=lambda s: float(s.split()[0]))
                interval_lines_str = "\n".join(interval_lines)

                # Frame probabilities
                prob_header = "time_s\t" + "\t".join(classes)
                prob_np = current_probs.T.numpy()
                times_np = np.arange(prob_np.shape[0]) * hop_time
                out_mat_np = np.column_stack((times_np, prob_np))
                
                frame_probs_rows = [prob_header]
                for row in out_mat_np:
                    frame_probs_rows.append("\t".join(map(lambda x: f"{x:.6f}", row)))
                frame_probs_str = "\n".join(frame_probs_rows)
                
                results_for_rank.append((wav_path_original.stem, interval_lines_str, frame_probs_str))

    # Gather all results to rank 0
    all_gathered_results: List[List[Tuple[str, str, str]]] = []
    if world_size > 1 and dist.is_initialized():
        gathered_obj_list = [None] * world_size
        dist.all_gather_object(gathered_obj_list, results_for_rank)
        if is_main_process(local_rank):
            for rank_results in gathered_obj_list:
                all_gathered_results.extend(rank_results)
    else:
        all_gathered_results = results_for_rank
        
    # Rank 0 writes all files
    if is_main_process(local_rank):
        save_dir_main = Path(cfg["base_save_dir"]) / exp_name
        out_dir_main = save_dir_main / "test_inference"
        prob_dir_main = save_dir_main / "test_frame_probs"

        processed_stems = set()
        for wav_stem, interval_str, frame_prob_str in all_gathered_results:
            if wav_stem in processed_stems:
                continue
            
            txt_path = out_dir_main / f"{wav_stem}.txt"
            with open(txt_path, "w") as fp_txt:
                fp_txt.write(interval_str)

            prob_path = prob_dir_main / f"{wav_stem}_probs.tsv"
            with open(prob_path, "w") as fp_prob:
                fp_prob.write(frame_prob_str)
            processed_stems.add(wav_stem)
        logger.info(f"[{exp_name}] Inference writing complete. Processed {len(processed_stems)} unique files.")
EXPERIMENTS = [
    {
        "name": "WavLM-base+GRU",
        "ssl_model_name": "microsoft/wavlm-base-plus",
        "feature_type": "raw",
        "architecture": "gru",
        "classes": ["swallowing", "others"]  # Match the 2-class model
    }
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def process_folder_wavs(folder_path: str) -> List[str]:
    """
    指定されたフォルダ内のすべてのWAVファイルのパスを取得
    
    Args:
        folder_path: WAVファイルが入っているフォルダのパス
        
    Returns:
        List[str]: WAVファイルのパスのリスト
    """
    wav_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"フォルダが見つかりません: {folder_path}")
        return wav_files
    
    if not folder.is_dir():
        print(f"指定されたパスはフォルダではありません: {folder_path}")
        return wav_files
    
    # WAVファイルを検索（再帰的に）
    for pattern in ['*.wav', '*.WAV']:
        wav_files.extend(folder.rglob(pattern))
    
    # パスを文字列に変換してソート
    wav_files = [str(f) for f in wav_files]
    wav_files.sort()
    
    print(f"フォルダ内で {len(wav_files)} 個のWAVファイルを発見しました")
    
    return wav_files


def main():
    # コマンドライン引数からフォルダパスを取得
    if len(sys.argv) < 2:
        print("使用方法: python inference.py <WAVファイルが入ったフォルダのパス>")
        print("例: python inference.py ./dataset/audio_files/")
        return
    
    input_folder = sys.argv[1]
    ensure_dir(OUTPUT_DIR)

    # フォルダ内のWAVファイルを取得
    wav_paths = process_folder_wavs(input_folder)
    
    if not wav_paths:
        print("推論対象のWAVファイルが見つかりませんでした")
        return

    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        print(f"\n--- Inference for {exp_name} ---")

        # Merge base config and experiment-specific settings
        config = {**BASE_CONFIG, **exp}
        config["base_save_dir"] = BASE_SAVE_DIR

        # Initialize and load the model
        model = EventDetector(
            config=config,
            ssl_model_name=config.get("ssl_model_name"),
            freeze_feature_extractor=config.get("freeze_fe", config["default_freeze_fe"]),
            freeze_transformer_layers=config.get("freeze_transformer", config["default_freeze_transformer"]),
            architecture=config.get("architecture"),
            feature_type=config.get("feature_type"),
        ).to(DEVICE)
        model.eval()

        # Load checkpoint from specified MODEL_PATH
        ckpt_path = MODEL_PATH
        if not os.path.isfile(ckpt_path):
            print(f"チェックポイントが見つかりません: {ckpt_path}")
            continue
        
        try:
            checkpoint = torch.load(ckpt_path, map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
            print(f"モデル重みをロードしました: {ckpt_path}")
        except Exception as e:
            print(f"モデル重みのロードに失敗: {e}")
            continue

        # Inference parameters
        sr = config["sr"]
        hop_length = config["ssl_hop_length"]  # samples per frame
        hop_time = hop_length / sr               # seconds per frame
        threshold = config["threshold"]
        classes = config["classes"]

        for wav_path in wav_paths:
            print(f"処理中: {os.path.basename(wav_path)}")
            
            try:
                # Load and preprocess audio
                wav, orig_sr = torchaudio.load(wav_path)
                if orig_sr != sr:
                    wav = torchaudio.transforms.Resample(orig_sr, sr)(wav)
                if wav.dim() > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                wav = wav.squeeze(0)

                total_samples = wav.shape[0]
                chunk_sec = 10.0
                overlap_sec = 3.0
                step_sec = chunk_sec - overlap_sec
                chunk_samples = int(chunk_sec * sr)
                step_samples = int(step_sec * sr)
                n_chunks = math.ceil((total_samples - chunk_samples) / step_samples) + 1

                # Prepare global probability buffer (C x total_frames)
                total_frames = math.ceil(total_samples / hop_length)
                global_probs = np.zeros((len(classes), total_frames), dtype=np.float32)

                # Slide window inference with overlap
                for idx in range(n_chunks):
                    start_samp = idx * step_samples
                    end_samp = min(start_samp + chunk_samples, total_samples)
                    segment = wav[start_samp:end_samp]
                    if segment.numel() == 0:
                        continue

                    tensor_seg = segment.unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits = model(tensor_seg)               # (1, C, T_seg)
                        probs = torch.softmax(logits[0], dim=0).cpu().numpy()  # (C, T_seg)

                    # Map segment frames into global buffer
                    seg_frames = probs.shape[1]
                    start_frame = start_samp // hop_length
                    for c in range(len(classes)):
                        for f in range(seg_frames):
                            g_idx = start_frame + f
                            if g_idx < total_frames:
                                # keep max probability across overlaps
                                global_probs[c, g_idx] = max(global_probs[c, g_idx], probs[c, f])

                # Convert merged global_probs → intervals
                merged_probs_tensor = torch.from_numpy(global_probs)
                merged_intervals = model_output_to_intervals(
                    merged_probs_tensor, threshold, hop_time, classes
                )

                # Write results to file in tab-separated format
                basename = os.path.splitext(os.path.basename(wav_path))[0]
                output_file = os.path.join(OUTPUT_DIR, f"{exp_name}_{basename}.txt")

                lines = []
                for cls in classes:
                    if cls == "others":  # Skip "others" class output
                        continue
                    short = LABEL_SHORT.get(cls, cls)
                    for st, ed in merged_intervals[cls]:
                        lines.append(f"{st:.6f}\t{ed:.6f}\t{short}")
                lines.sort(key=lambda l: float(l.split("\t")[0]))

                with open(output_file, 'w') as f:
                    f.write("\n".join(lines))

                print(f"  結果を保存しました: {output_file}")
                
            except Exception as e:
                print(f"  エラー: {wav_path} の処理中にエラーが発生しました: {e}")
                continue

if __name__ == "__main__":
    main()
