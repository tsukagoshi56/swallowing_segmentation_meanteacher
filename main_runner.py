#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main runner script for sound event detection experiments.
"""
import sys
import os
from pathlib import Path

# システムパスの設定
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
    
# 環境情報を表示
print(f"Python version: {sys.version}")
print(f"Current directory: {current_dir}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

import argparse
import json
import logging
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.utils.data as tud
import torch.distributed as dist
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data.distributed import DistributedSampler

# モジュールをフルパスでインポート
sys.path.append(str(current_dir))
from ddp_utils import setup_ddp, cleanup_ddp, is_main_process
from exp_config import BASE_CONFIG, EXPERIMENTS, EXPERIMENT_GROUP, logger
from training import expand_all_experiments, train_and_evaluate
from inference import test_model, inference_model
import training as training_module
import inference as inference_module

_DATALOADER_PATCHED = False
_EVALUATE_PATCHED = False


def configure_dataloader_workers(max_workers: int) -> None:
    """
    Limit DataLoader workers across modules to avoid multiprocessing permission issues.

    Args:
        max_workers: Maximum number of workers to allow (-1 to keep default behaviour)
    """
    global _DATALOADER_PATCHED
    if _DATALOADER_PATCHED or max_workers < 0:
        return

    base_loader_cls = tud.DataLoader

    class SafeDataLoader(base_loader_cls):
        """DataLoader that caps worker count and disables persistence when unused."""

        def __init__(self, dataset, *args, **kwargs):
            requested = kwargs.get("num_workers", max_workers)
            if requested is None:
                requested = max_workers
            capped_workers = max(0, min(requested, max_workers))
            kwargs["num_workers"] = capped_workers

            if kwargs["num_workers"] == 0:
                if kwargs.get("persistent_workers", False):
                    kwargs["persistent_workers"] = False
                if kwargs.get("pin_memory", False) and not torch.cuda.is_available():
                    kwargs["pin_memory"] = False
            else:
                kwargs.setdefault("pin_memory", torch.cuda.is_available())

            original_collate = kwargs.get("collate_fn")
            num_classes = getattr(dataset, "num_classes", None) or len(getattr(dataset, "classes", []) or [])
            sampler = kwargs.get("sampler")
            is_train_loader = kwargs.get("shuffle", False) or bool(getattr(sampler, "shuffle", False))

            if original_collate is not None:
                def adapted_collate(*c_args, **c_kwargs):
                    batch = original_collate(*c_args, **c_kwargs)
                    if isinstance(batch, tuple) and len(batch) == 4:
                        features, labels, masks, paths = batch
                        if is_train_loader:
                            labels_tensor = labels
                            try:
                                if hasattr(labels_tensor, "dim") and labels_tensor.dim() == 2 and num_classes:
                                    labels_tensor = F.one_hot(labels_tensor.long(), num_classes=num_classes).permute(0, 2, 1).float()
                                else:
                                    labels_tensor = labels_tensor.float()
                            except Exception as exc:
                                logger.warning("Collate conversion to one-hot failed: %s", exc)
                                labels_tensor = labels_tensor.float()

                            if isinstance(masks, torch.Tensor) and labels_tensor.dim() == 3:
                                labels_tensor = labels_tensor * masks.float().unsqueeze(1)

                            return features, labels_tensor.contiguous(), paths

                        return features, labels, masks, paths
                    return batch

                kwargs["collate_fn"] = adapted_collate

            super().__init__(dataset, *args, **kwargs)

    training_module.DataLoader = SafeDataLoader  # type: ignore[attr-defined]
    inference_module.DataLoader = SafeDataLoader  # type: ignore[attr-defined]
    _DATALOADER_PATCHED = True
    logger.info("DataLoader workers capped at %d via main_runner safeguard.", max_workers)


def patch_evaluate_epoch() -> None:
    """Patch training.evaluate_epoch to handle class-index labels with masks."""
    global _EVALUATE_PATCHED
    if _EVALUATE_PATCHED:
        return

    tm = training_module
    metric_mod = training_module
    tm_logger = tm.logger
    iou_thresholds = tm.IOU_THRESHOLDS

    def _ensure_one_hot(labels_tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
        if labels_tensor.dim() == 2:
            return F.one_hot(
                labels_tensor.long().clamp(min=0, max=num_classes - 1),
                num_classes=num_classes
            ).permute(0, 2, 1).float()
        if labels_tensor.dim() == 3 and labels_tensor.size(1) == num_classes:
            return labels_tensor.float()
        raise ValueError(f"Unsupported label shape {labels_tensor.shape} for one-hot conversion.")

    def evaluate_epoch_patched(model: torch.nn.Module,
                               dataloader: tud.DataLoader,
                               criterion: torch.nn.Module,
                               config: dict,
                               local_rank: int,
                               world_size: int,
                               compute_event_metrics: bool = False,
                               epoch_num: Optional[int] = None,
                               num_epochs_total: Optional[int] = None,
                               limit: Optional[int] = None,
                               return_detailed: bool = False):
        model.eval()
        total_loss_rank = 0.0
        batches_processed_rank = 0

        device = config["device"]
        hop_time = config["ssl_hop_length"] / config["sr"]
        threshold = config["threshold"]
        class_names = config["classes"]
        num_classes = len(class_names)

        gt_list_rank: List[Dict[str, List[Tuple[float, float]]]] = []
        pred_list_rank: List[Dict[str, List[Tuple[float, float]]]] = []
        paths_list_rank: List[str] = []

        if hasattr(dataloader.sampler, 'set_epoch') and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch_num if epoch_num is not None else 0)

        num_batches = len(dataloader)
        batches_to_run = min(limit, num_batches) if limit is not None else num_batches
        if batches_to_run == 0:
            empty_metrics = {thr: (0.0, 0.0, 0.0) for thr in iou_thresholds} if compute_event_metrics else None
            if return_detailed:
                return 0.0, empty_metrics, [], [], []
            return 0.0, empty_metrics

        desc_prefix = f"Epoch {epoch_num}/{num_epochs_total} [Val]" if epoch_num else "[Test Eval]"
        pbar = tqdm(dataloader, total=batches_to_run, desc=desc_prefix, disable=not tm.is_main_process(local_rank))

        with torch.no_grad():
            for i, batch_data in enumerate(pbar):
                if limit is not None and i >= limit:
                    break
                try:
                    features, labels, masks, paths = batch_data
                except ValueError as e:
                    if tm.is_main_process(local_rank):
                        tm_logger.error(f"Error unpacking eval batch {i}: {e}. Skipping.")
                    continue

                features = features.to(device)
                labels = labels.to(device)
                masks = masks.to(device)

                try:
                    logits = model(features)
                    output_len = logits.size(-1)
                    label_len = labels.size(-1) if labels.dim() >= 2 else labels.numel()
                    min_len = min(output_len, label_len)

                    if min_len <= 0:
                        if tm.is_main_process(local_rank):
                            tm_logger.warning(f"Skipping eval batch {i} due to zero/negative min_len ({min_len})")
                        continue

                    labels_one_hot = _ensure_one_hot(labels, num_classes)[..., :min_len]
                    mask_slice = masks[..., :min_len]
                    mask_expanded = mask_slice.unsqueeze(1).float()

                    logits_slice = logits[..., :min_len]

                    if mask_expanded is not None:
                        loss = F.binary_cross_entropy_with_logits(
                            logits_slice, labels_one_hot, weight=mask_expanded, reduction='sum'
                        )
                        normalizer = mask_expanded.sum().clamp_min(1.0)
                        loss = loss / normalizer
                    else:
                        loss = criterion(logits_slice, labels_one_hot)

                    if torch.isnan(loss):
                        if tm.is_main_process(local_rank):
                            tm_logger.warning(f"NaN loss in eval batch {i}.")
                    else:
                        total_loss_rank += loss.item()
                        batches_processed_rank += 1
                        if tm.is_main_process(local_rank):
                            pbar.set_postfix(loss=f"{loss.item():.4f}")

                    probabilities = torch.softmax(logits_slice, dim=1)

                    if compute_event_metrics or return_detailed:
                        probs_matched = probabilities.detach().cpu()
                        labels_matched = (labels_one_hot * mask_expanded).detach().cpu()
                        masks_matched = mask_slice.detach().cpu()

                        others_idx = class_names.index("others") if "others" in class_names else None

                        max_probs, pred_indices = probs_matched.max(dim=1)
                        pred_one_hot = F.one_hot(pred_indices, num_classes=num_classes).permute(0, 2, 1).float()
                        pred_one_hot *= masks_matched.unsqueeze(1)

                        below_threshold = (max_probs < threshold) & (masks_matched > 0)
                        if below_threshold.any():
                            pred_one_hot_btc = pred_one_hot.permute(0, 2, 1).contiguous()
                            pred_one_hot_btc[below_threshold] = 0.0
                            if others_idx is not None:
                                pred_one_hot_btc[below_threshold, others_idx] = 1.0
                            pred_one_hot = pred_one_hot_btc.permute(0, 2, 1)

                        for b_idx in range(probs_matched.size(0)):
                            gt_intervals_sample: Dict[str, List[Tuple[float, float]]] = {}
                            label_np_sample = labels_matched[b_idx].numpy()
                            for c_idx, class_name in enumerate(class_names):
                                gt_sequence = label_np_sample[c_idx]
                                gt_intervals_sample[class_name] = metric_mod.sequence_to_intervals(
                                    gt_sequence, hop_time, local_rank
                                )
                            gt_list_rank.append(gt_intervals_sample)

                            pred_intervals_sample = metric_mod.model_output_to_intervals(
                                pred_one_hot[b_idx], threshold, hop_time, class_names, local_rank
                            )
                            pred_list_rank.append(pred_intervals_sample)

                            if return_detailed:
                                paths_list_rank.append(paths[b_idx])

                except Exception as e:
                    if tm.is_main_process(local_rank):
                        tm_logger.exception(f"Error during evaluation batch {i}: {e}.")

        avg_loss = 0.0
        if world_size > 1 and dist.is_initialized():
            loss_stats = torch.tensor([total_loss_rank, float(batches_processed_rank)], device=device)
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
            global_total_loss, global_batches_processed = loss_stats[0].item(), loss_stats[1].item()
            avg_loss = global_total_loss / global_batches_processed if global_batches_processed > 0 else 0.0
        elif batches_processed_rank > 0:
            avg_loss = total_loss_rank / batches_processed_rank
        else:
            avg_loss = 0.0

        final_gt_list_rank0: List[Dict] = []
        final_pred_list_rank0: List[Dict] = []
        final_paths_list_rank0: List[str] = []
        event_metrics_output: Optional[Dict[float, Tuple[float, float, float]]] = None

        if world_size > 1 and dist.is_initialized():
            gathered_gts_obj = [None] * world_size
            gathered_preds_obj = [None] * world_size
            gathered_paths_obj = [None] * world_size if return_detailed else []

            dist.all_gather_object(gathered_gts_obj, gt_list_rank)
            dist.all_gather_object(gathered_preds_obj, pred_list_rank)
            if return_detailed:
                dist.all_gather_object(gathered_paths_obj, paths_list_rank)

            if tm.is_main_process(local_rank):
                for grp_idx in range(world_size):
                    final_gt_list_rank0.extend(gathered_gts_obj[grp_idx])
                    final_pred_list_rank0.extend(gathered_preds_obj[grp_idx])
                    if return_detailed:
                        final_paths_list_rank0.extend(gathered_paths_obj[grp_idx])
        else:
            final_gt_list_rank0 = gt_list_rank
            final_pred_list_rank0 = pred_list_rank
            if return_detailed:
                final_paths_list_rank0 = paths_list_rank

        if tm.is_main_process(local_rank) and compute_event_metrics:
            if len(final_gt_list_rank0) > 0:
                try:
                    tm_logger.info(f"Calculating event metrics for {len(final_gt_list_rank0)} samples on rank 0...")
                    event_metrics_output = metric_mod.calculate_event_metrics(
                        final_gt_list_rank0, final_pred_list_rank0, class_names, iou_thresholds, local_rank
                    )
                except Exception as e:
                    tm_logger.error(f"Error calculating event metrics on rank 0: {e}")
                    event_metrics_output = None
            else:
                tm_logger.warning("No samples gathered on rank 0 for event metrics calculation.")
                event_metrics_output = {thr: (0.0, 0.0, 0.0) for thr in iou_thresholds}

        if return_detailed:
            if not tm.is_main_process(local_rank):
                final_gt_list_rank0, final_pred_list_rank0, final_paths_list_rank0 = [], [], []
                event_metrics_output = None
            return avg_loss, event_metrics_output, final_gt_list_rank0, final_pred_list_rank0, final_paths_list_rank0

        if not tm.is_main_process(local_rank):
            event_metrics_output = None
        return avg_loss, event_metrics_output

    training_module.evaluate_epoch = evaluate_epoch_patched  # type: ignore
    inference_module.evaluate_epoch = evaluate_epoch_patched  # type: ignore
    _EVALUATE_PATCHED = True
    logger.info("evaluate_epoch patched to handle index labels with BCE loss.")

def main():
    """Main function to parse arguments and run experiments"""
    parser = argparse.ArgumentParser(description="Sound Event Detection Experiment Runner")
    parser.add_argument("--test", action="store_true", help="Run in test mode only (no training)")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model name to test/inference (default: all)")
    parser.add_argument("--inference", action="store_true", help="Run inference-only mode")
    parser.add_argument("--finetune", action="store_true", help="Run in fine-tuning mode, resetting patience and continuing training from the previous best model")
    parser.add_argument("--data-workers", type=int, default=0,
                        help="Cap DataLoader worker count (-1 to keep module defaults)")
    parser.add_argument("--config", type=str, default="./config_FT_real.json",
                        help="Path to external JSON experiment configuration (overrides exp_config settings)")
    args = parser.parse_args()

    configure_dataloader_workers(args.data_workers)
    patch_evaluate_epoch()

    base_config: Dict[str, Any] = dict(BASE_CONFIG)
    list_of_experiments: List[Dict[str, Any]] = []

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r") as fp:
            external_conf = json.load(fp)
        base_config.update(external_conf.get("base_config", {}))
        experiments_conf = external_conf.get("experiments", {})
        for key, exp_conf in experiments_conf.items():
            exp_entry = dict(exp_conf)
            exp_entry.setdefault("name", exp_conf.get("name", key))
            list_of_experiments.append(exp_entry)
        logger.info("Loaded %d experiments from %s", len(list_of_experiments), config_path)
        if not list_of_experiments:
            logger.warning("No experiments defined in %s.", config_path)

    # DDP setup
    local_rank, world_size = setup_ddp()
    
    # Device assignment
    if world_size > 1:
        base_config["device"] = local_rank
    else:
        requested_device = base_config.get("device")
        if requested_device:
            if isinstance(requested_device, str) and requested_device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA not available; falling back to CPU.")
                base_config["device"] = "cpu"
        else:
            base_config["device"] = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Root logger setup for rank 0
    if is_main_process(local_rank):
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    # Select experiments
    if not args.config:
        if EXPERIMENT_GROUP == "all":
            list_of_experiments = expand_all_experiments(EXPERIMENTS)
        else:
            list_of_experiments = EXPERIMENTS.get(EXPERIMENT_GROUP, []).copy()
    
    if args.model:
        list_of_experiments = [e for e in list_of_experiments if e['name'] == args.model]

    if not list_of_experiments and is_main_process(local_rank):
        logger.warning("No experiments to run. Exiting.")
        sys.exit(0)

    # Mode selection
    if args.inference:
        logger.info("Inference-only mode")
        for exp_conf in list_of_experiments:
            if is_main_process(local_rank): 
                print(f"=== Inference: {exp_conf['name']} ===")
            inference_model(base_config, exp_conf, local_rank, world_size)
    elif args.test:
        logger.info("Test-only mode")
        for exp_conf in list_of_experiments:
            if is_main_process(local_rank): 
                print(f"=== Test: {exp_conf['name']} ===")
            test_model(base_config, exp_conf, local_rank, world_size)
    elif args.finetune:
        logger.info("Fine-tuning mode (continue training with reset patience)")
        for exp_conf in list_of_experiments:
            if is_main_process(local_rank): 
                print(f"=== Fine-tuning: {exp_conf['name']} ===")
            # Set fine-tuning flag to true
            exp_conf['finetune'] = True
            train_and_evaluate(base_config, exp_conf, local_rank, world_size)
    else:
        logger.info("Training and Evaluation mode")
        for exp_conf in list_of_experiments:
            if is_main_process(local_rank): 
                print(f"=== Experiment: {exp_conf['name']} ===")
            # Set fine-tuning flag to false (default)
            exp_conf['finetune'] = False
            train_and_evaluate(base_config, exp_conf, local_rank, world_size)

    cleanup_ddp()

if __name__ == '__main__':
    main()
