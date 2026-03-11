"""
Event-based metrics for sound event detection evaluation
"""
import os
import re
import numpy as np
import torch
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

from ddp_utils import is_main_process
from exp_config import EPS, IOU_THRESHOLDS, LABEL_SHORT

logger = logging.getLogger(__name__)

def interval_iou(interval_a: Tuple[float, float], interval_b: Tuple[float, float]) -> float:
    """
    Calculate IoU between two time intervals
    
    Args:
        interval_a: (start_time, end_time) for first interval
        interval_b: (start_time, end_time) for second interval
        
    Returns:
        float: IoU value between 0 and 1
    """
    start_a, end_a = interval_a
    start_b, end_b = interval_b
    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)
    intersection_duration = max(0.0, intersection_end - intersection_start)
    union_duration = (end_a - start_a) + (end_b - start_b) - intersection_duration
    return intersection_duration / (union_duration + EPS) if union_duration > EPS else 0.0

def sequence_to_intervals(binary_sequence: np.ndarray, hop_time: float, local_rank: int = 0) -> List[Tuple[float, float]]:
    """
    Convert binary sequence to list of time intervals
    
    Args:
        binary_sequence: Binary array (0s and 1s)
        hop_time: Time in seconds per frame
        local_rank: Process rank for logging
        
    Returns:
        List of (start_time, end_time) tuples
    """
    intervals = []
    in_event = False
    start_frame = 0
    
    # 修正点：スカラー値の入力を処理できるよう、シーケンスが最低でも1次元であることを保証する
    binary_sequence = np.atleast_1d(binary_sequence)
    
    # Pad with 0 at the end to catch events ending at the last frame
    padded_sequence = np.pad(binary_sequence, (0, 1), mode='constant', constant_values=0)

    for frame_index, value in enumerate(padded_sequence):
        if value > 0 and not in_event:
            in_event = True
            start_frame = frame_index
        elif value == 0 and in_event:
            in_event = False
            end_frame = frame_index  # This is exclusive end frame
            # Ensure start_frame is less than end_frame for a valid interval
            if start_frame < end_frame:
                 intervals.append((start_frame * hop_time, end_frame * hop_time))

    return intervals

def model_output_to_intervals(
    output_tensor: torch.Tensor, threshold: float, hop_time: float, class_names: List[str], local_rank: int = 0
    ) -> Dict[str, List[Tuple[float, float]]]:
    """
    Convert model output tensor to dictionary of intervals by class
    
    Args:
        output_tensor: Model output tensor (C, T)
        threshold: Threshold for binary prediction
        hop_time: Time in seconds per frame
        class_names: List of class names
        local_rank: Process rank for logging
        
    Returns:
        Dict mapping class names to lists of (start_time, end_time) tuples
    """
    if output_tensor.dim() != 2 or len(class_names) != output_tensor.shape[0]:
        raise ValueError(f"output_tensor shape error or class_names mismatch. Got {output_tensor.shape}, expected C={len(class_names)}")
    
    binary_predictions = (output_tensor.detach().cpu().numpy() >= threshold).astype(np.int8)
    predicted_intervals = {}
    for i, class_name in enumerate(class_names):
        predicted_intervals[class_name] = sequence_to_intervals(binary_predictions[i], hop_time, local_rank)
    return predicted_intervals

def extract_food_type(file_path: str) -> str:
    """
    Extract food type from file path
    
    Args:
        file_path: Path to audio file
        
    Returns:
        str: Food type
    """
    food_types = ["cbg", "gum", "rtz", "w20"]
    file_name = os.path.basename(file_path).lower()
    for food in food_types:
        if food in file_name:
            return food
    return "unknown"

def calculate_class_specific_metrics(
    ground_truth_list: List[Dict[str, List[Tuple[float, float]]]],
    prediction_list: List[Dict[str, List[Tuple[float, float]]]],
    class_names: List[str],
    iou_thresholds: List[float] = IOU_THRESHOLDS,
    local_rank: int = 0
    ) -> Dict[float, Tuple[float, float, float]]:
    """
    Calculate precision, recall, and F1 for specific classes
    
    Args:
        ground_truth_list: List of ground truth intervals by class
        prediction_list: List of predicted intervals by class
        class_names: List of class names to evaluate
        iou_thresholds: List of IoU thresholds
        local_rank: Process rank for logging
        
    Returns:
        Dict mapping IoU thresholds to (precision, recall, F1) tuples
    """
    results = {}
    if not ground_truth_list or not prediction_list:
        if is_main_process(local_rank):
             logger.warning("Ground truth or prediction list is empty in calculate_class_specific_metrics.")
        return {thr: (0.0, 0.0, 0.0) for thr in iou_thresholds}

    for iou_thr in iou_thresholds:
        total_tp, total_n_pred, total_n_gt = 0, 0, 0
        for gt_intervals_dict, pred_intervals_dict in zip(ground_truth_list, prediction_list):
            for class_name in class_names:
                gt_intervals = gt_intervals_dict.get(class_name, [])
                pred_intervals = pred_intervals_dict.get(class_name, [])
                n_gt, n_pred = len(gt_intervals), len(pred_intervals)
                total_n_gt += n_gt
                total_n_pred += n_pred

                if n_gt == 0 or n_pred == 0:
                    continue

                iou_matrix = np.zeros((n_gt, n_pred))
                for gt_idx in range(n_gt):
                    for pred_idx in range(n_pred):
                        iou_matrix[gt_idx, pred_idx] = interval_iou(gt_intervals[gt_idx], pred_intervals[pred_idx])
                
                # Greedy matching prioritizes GT events
                matched_pred_indices = set()
                tp_for_this_class_file = 0
                for gt_idx in range(n_gt):
                    best_iou_for_gt = -1.0
                    best_pred_idx_for_gt = -1
                    for pred_idx in range(n_pred):
                        if pred_idx not in matched_pred_indices:
                            current_iou = iou_matrix[gt_idx, pred_idx]
                            if current_iou >= iou_thr and current_iou > best_iou_for_gt:
                                best_iou_for_gt = current_iou
                                best_pred_idx_for_gt = pred_idx
                    
                    if best_pred_idx_for_gt != -1:
                        tp_for_this_class_file += 1
                        matched_pred_indices.add(best_pred_idx_for_gt)
                total_tp += tp_for_this_class_file
        
        precision = total_tp / (total_n_pred + EPS)
        recall = total_tp / (total_n_gt + EPS)
        f1_score = 2 * precision * recall / (precision + recall + EPS)
        results[iou_thr] = (
            float(np.clip(precision, 0, 1)),
            float(np.clip(recall, 0, 1)),
            float(np.clip(f1_score, 0, 1))
        )
    return results

def calculate_event_metrics(
    ground_truth_list: List[Dict[str, List[Tuple[float, float]]]],
    prediction_list: List[Dict[str, List[Tuple[float, float]]]],
    class_names: List[str],
    iou_thresholds: List[float] = IOU_THRESHOLDS,
    local_rank: int = 0
    ) -> Dict[float, Tuple[float, float, float]]:
    """
    Calculate overall event metrics
    
    Args:
        ground_truth_list: List of ground truth intervals by class
        prediction_list: List of predicted intervals by class
        class_names: List of class names to evaluate
        iou_thresholds: List of IoU thresholds
        local_rank: Process rank for logging
        
    Returns:
        Dict mapping IoU thresholds to (precision, recall, F1) tuples
    """
    if not ground_truth_list or not prediction_list:
        if is_main_process(local_rank):
            logger.warning("Ground truth or prediction list is empty, cannot calculate event_metrics.")
        return {thr: (0.0, 0.0, 0.0) for thr in iou_thresholds}

    return calculate_class_specific_metrics(
        ground_truth_list, prediction_list, class_names, iou_thresholds, local_rank
    )

def calculate_event_metrics_by_class_and_food(
    ground_truth_list: List[Dict[str, List[Tuple[float, float]]]],
    prediction_list: List[Dict[str, List[Tuple[float, float]]]],
    file_paths: List[str],
    class_names: List[str],
    iou_thresholds: List[float] = IOU_THRESHOLDS,
    local_rank: int = 0
    ) -> Dict[str, Dict[str, Dict[float, Tuple[float, float, float]]]]:
    """
    Calculate metrics by class and food type
    
    Args:
        ground_truth_list: List of ground truth intervals by class
        prediction_list: List of predicted intervals by class
        file_paths: List of file paths
        class_names: List of class names to evaluate
        iou_thresholds: List of IoU thresholds
        local_rank: Process rank for logging
        
    Returns:
        Nested dict mapping food type -> class name -> IoU threshold -> (precision, recall, F1)
    """
    results = {"all": {}}
    food_files_indices: Dict[str, List[int]] = {}
    
    for i, path in enumerate(file_paths):
        food_type = extract_food_type(path)
        if food_type not in food_files_indices:
            food_files_indices[food_type] = []
        food_files_indices[food_type].append(i)
    
    # Calculate metrics for "all" food types (overall per class)
    for class_name in class_names:
        class_metrics = calculate_class_specific_metrics(
            ground_truth_list, prediction_list, [class_name], iou_thresholds, local_rank
        )
        results["all"][class_name] = class_metrics
        
    # Calculate metrics for each specific food type
    for food_type, indices in food_files_indices.items():
        if not indices:
            continue
        
        food_specific_gt = [ground_truth_list[i] for i in indices]
        food_specific_pred = [prediction_list[i] for i in indices]
        
        results[food_type] = {}
        for class_name in class_names:
            food_class_metrics = calculate_class_specific_metrics(
                food_specific_gt, food_specific_pred, [class_name], iou_thresholds, local_rank
            )
            results[food_type][class_name] = food_class_metrics
            
    return results

def print_event_statistics(
    ground_truth_list: List[Dict[str, List[Tuple[float, float]]]],
    prediction_list: List[Dict[str, List[Tuple[float, float]]]],
    file_paths: List[str],
    class_names: List[str]
    ):
    """
    Print event statistics
    
    Args:
        ground_truth_list: List of ground truth intervals by class
        prediction_list: List of predicted intervals by class
        file_paths: List of file paths
        class_names: List of class names
    """
    food_sample_counts: Dict[str, int] = {}
    total_event_counts = {cn: {"gt": 0, "pred": 0} for cn in class_names}
    food_event_counts: Dict[str, Dict[str, Dict[str, int]]] = {}

    print("\n=== イベント統計情報 ===")
    print(f"テストサンプル数: {len(file_paths)}")

    for i, (gt_dict, pred_dict, path) in enumerate(zip(ground_truth_list, prediction_list, file_paths)):
        food_type = extract_food_type(path)
        
        food_sample_counts[food_type] = food_sample_counts.get(food_type, 0) + 1
        if food_type not in food_event_counts:
            food_event_counts[food_type] = {cn: {"gt": 0, "pred": 0} for cn in class_names}
            
        for class_name in class_names:
            gt_events = len(gt_dict.get(class_name, []))
            pred_events = len(pred_dict.get(class_name, []))
            
            total_event_counts[class_name]["gt"] += gt_events
            total_event_counts[class_name]["pred"] += pred_events
            food_event_counts[food_type][class_name]["gt"] += gt_events
            food_event_counts[food_type][class_name]["pred"] += pred_events

    print("\n--- クラス別イベント数 (全体) ---")
    for class_name, counts in total_event_counts.items():
        print(f"{class_name}: GT={counts['gt']}, 予測={counts['pred']}")
    
    print("\n--- 食材別サンプル数 ---")
    for food_type, count in food_sample_counts.items():
        print(f"{food_type}: {count}サンプル")
        
    print("\n--- 食材・クラス別イベント数 ---")
    for food_type, class_counts_dict in food_event_counts.items():
        print(f"\n{food_type}:")
        for class_name, counts in class_counts_dict.items():
            print(f"  {class_name}: GT={counts['gt']}, 予測={counts['pred']}")

def collect_raw_iou(
    gt_list: List[Dict[str, List[Tuple[float, float]]]],
    pred_list: List[Dict[str, List[Tuple[float, float]]]],
    class_names: List[str]
) -> Dict[str, List[float]]:
    """
    Collect raw IoU values between all GT and prediction pairs
    
    Args:
        gt_list: List of ground truth intervals by class
        pred_list: List of predicted intervals by class
        class_names: List of class names
        
    Returns:
        Dict mapping class names to lists of IoU values
    """
    raw = {cn: [] for cn in class_names}
    for gt, pred in zip(gt_list, pred_list):
        for cn in class_names:
            for a in gt.get(cn, []):
                for b in pred.get(cn, []):
                    raw[cn].append(interval_iou(a, b))
    return raw

def collect_raw_iou_by_food(
    gt_list: List[Dict[str, List[Tuple[float, float]]]],
    pred_list: List[Dict[str, List[Tuple[float, float]]]],
    file_paths: List[str],
    class_names: List[str]
) -> Dict[str, Dict[str, List[float]]]:
    """
    Collect raw IoU values by food type
    
    Args:
        gt_list: List of ground truth intervals by class
        pred_list: List of predicted intervals by class
        file_paths: List of file paths
        class_names: List of class names
        
    Returns:
        Nested dict mapping food type -> class name -> list of IoU values
    """
    food_iou: Dict[str, Dict[str, List[float]]] = {}
    for gt, pred, path in zip(gt_list, pred_list, file_paths):
        ft = extract_food_type(path)
        if ft not in food_iou:
            food_iou[ft] = {cn: [] for cn in class_names}
        for cn in class_names:
            for a in gt.get(cn, []):
                for b in pred.get(cn, []):
                    food_iou[ft][cn].append(interval_iou(a, b))
    return food_iou

def collect_matched_iou(
    gt_list: List[Dict[str, List[Tuple[float, float]]]],
    pred_list: List[Dict[str, List[Tuple[float, float]]]],
    class_names: List[str],
    iou_threshold: float = EPS
) -> Dict[str, List[float]]:
    """
    Collect IoU values for matched GT-prediction pairs
    
    Args:
        gt_list: List of ground truth intervals by class
        pred_list: List of predicted intervals by class
        class_names: List of class names
        iou_threshold: Minimum IoU threshold for considering a match
        
    Returns:
        Dict mapping class names to lists of IoU values for matched pairs
    """
    matched_ious_per_class = {cn: [] for cn in class_names}

    for gt_intervals_dict, pred_intervals_dict in zip(gt_list, pred_list):
        for class_name in class_names:
            gt_intervals = gt_intervals_dict.get(class_name, [])
            pred_intervals = pred_intervals_dict.get(class_name, [])

            if not gt_intervals or not pred_intervals:
                continue

            # Create IoU matrix: rows are GT, cols are Pred
            iou_matrix = np.array(
                [[interval_iou(gt, pred) for pred in pred_intervals] for gt in gt_intervals]
            )
            
            num_gt = len(gt_intervals)
            num_pred = len(pred_intervals)
            matched_pred_indices = np.zeros(num_pred, dtype=bool)

            for gt_idx in range(num_gt):
                best_iou_for_gt = -1.0
                best_pred_idx_for_gt = -1
                
                for pred_idx in range(num_pred):
                    if not matched_pred_indices[pred_idx]:
                        current_iou = iou_matrix[gt_idx, pred_idx]
                        if current_iou >= iou_threshold and current_iou > best_iou_for_gt:
                            best_iou_for_gt = current_iou
                            best_pred_idx_for_gt = pred_idx
                
                if best_pred_idx_for_gt != -1:
                    matched_ious_per_class[class_name].append(best_iou_for_gt)
                    matched_pred_indices[best_pred_idx_for_gt] = True

    return matched_ious_per_class

def load_patience_from_log(log_path: Path, patience_total: int) -> int:
    """
    Load patience counter from log file
    
    Args:
        log_path: Path to log file
        patience_total: Total patience value
        
    Returns:
        int: Current patience counter
    """
    if not log_path.exists():
        return 0
    try:
        text = log_path.read_text(encoding='utf-8', errors='ignore')
        pattern = re.compile(r'Patience:\s*(\d+)\s*/\s*{}'.format(patience_total))
        matches = pattern.findall(text)
        return int(matches[-1]) if matches else 0
    except Exception:
        return 0

def has_smoke_test_finished(log_path: Path) -> bool:
    """
    Check if smoke test has finished
    
    Args:
        log_path: Path to log file
        
    Returns:
        bool: True if smoke test has finished
    """
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding='utf-8', errors='ignore')
        return "Smoke test epoch finished." in text
    except Exception:
        return False
