"""
Training and evaluation functions for sound event detection models
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import itertools
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
import json
from functools import partial
import time
from pathlib import Path

from data_utils import SoundEventDataset, pad_collate
from ddp_utils import is_main_process
from models import EventDetector
from metrics import (
    calculate_event_metrics, model_output_to_intervals, sequence_to_intervals,
    load_patience_from_log, has_smoke_test_finished
)
from exp_config import IOU_THRESHOLDS

logger = logging.getLogger(__name__)


def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: Union[torch.device, int],
                    epoch_num: int,
                    num_epochs_total: int,
                    local_rank: int,
                    limit: Optional[int] = None) -> float:
    """
    Train model for one epoch
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use for training
        epoch_num: Current epoch number
        num_epochs_total: Total number of epochs
        local_rank: Process rank
        limit: Optional batch limit (for smoke testing)
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    batches_processed = 0
    
    if hasattr(dataloader.sampler, 'set_epoch') and isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch_num)

    num_batches = len(dataloader)
    batches_to_run = min(limit, num_batches) if limit is not None else num_batches
    if batches_to_run == 0:
        return 0.0

    pbar_desc = f"Epoch {epoch_num}/{num_epochs_total} [Train]"
    pbar = tqdm(dataloader, total=batches_to_run, desc=pbar_desc, disable=not is_main_process(local_rank))

    for i, batch_data in enumerate(pbar):
        if limit is not None and i >= limit:
            break
        try:
            # DataLoader が返す要素数に応じて柔軟に受け取る
            if isinstance(batch_data, (list, tuple)):
                if len(batch_data) == 4:
                    # (features, labels, masks, paths)
                    features, labels, masks, paths = batch_data
                    # 現状の train では masks / paths は利用していないので捨てる
                    # （将来使いたくなったらここで使えばOK）
                elif len(batch_data) == 3:
                    # (features, labels, masks)
                    features, labels, masks = batch_data
                    paths = None
                else:
                    raise ValueError(
                        f"Unexpected number of elements in train batch: {len(batch_data)} "
                        "(expected 3 or 4)"
                    )
            else:
                raise ValueError(f"Unexpected batch type: {type(batch_data)}")

        except ValueError as e:
            if is_main_process(local_rank):
                logger.error(f"Error unpacking train batch {i}: {e}. Skipping.")
            continue

        features, labels = features.to(device), labels.to(device)
        # masks, paths は現状使わないので device にも乗せない

        try:
            optimizer.zero_grad()
            logits = model(features)

            output_len = logits.size(-1)
            label_len = labels.size(-1)
            min_len = min(output_len, label_len)

            if min_len <= 0:
                if is_main_process(local_rank):
                    logger.warning(f"Skipping train batch {i} due to zero/negative min_len ({min_len}) between output ({output_len}) and labels ({label_len})")
                continue
            
            loss = criterion(logits[..., :min_len], labels[..., :min_len])

            if torch.isnan(loss):
                if is_main_process(local_rank):
                    logger.warning(f"NaN loss in train batch {i}. Skipping backward.")
                optimizer.zero_grad()
                continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches_processed += 1
            if is_main_process(local_rank):
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        except Exception as e:
            if is_main_process(local_rank):
                logger.exception(f"Error during training batch {i}: {e}. Skipping.")
            optimizer.zero_grad()

    if batches_processed == 0:
        if is_main_process(local_rank):
            logger.warning("No batches processed in train_one_epoch.")
        return 0.0
    
    avg_loss = total_loss / batches_processed
    if dist.is_initialized() and dist.get_world_size() > 1:
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
        
    return avg_loss

def evaluate_epoch(model: nn.Module,
                   dataloader: DataLoader,
                   criterion: nn.Module,
                   config: Dict[str, Any],
                   local_rank: int,
                   world_size: int,
                   compute_event_metrics: bool = False,
                   epoch_num: Optional[int] = None,
                   num_epochs_total: Optional[int] = None,
                   limit: Optional[int] = None,
                   return_detailed: bool = False
                   ) -> Union[Tuple[float, Optional[Dict[float, Tuple[float, float, float]]]],
                              Tuple[float, Optional[Dict[float, Tuple[float, float, float]]], List[Dict], List[Dict], List[str]]]:
    """
    Evaluate model on validation or test data
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for validation/test data
        criterion: Loss function
        config: Configuration dictionary
        local_rank: Process rank
        world_size: Total number of processes
        compute_event_metrics: Whether to compute event-based metrics
        epoch_num: Current epoch number (optional)
        num_epochs_total: Total number of epochs (optional)
        limit: Optional batch limit (for smoke testing)
        return_detailed: Whether to return detailed results
        
    Returns:
        Tuple containing:
            - Average loss
            - Dictionary of event metrics (if compute_event_metrics=True)
            - List of ground truth intervals (if return_detailed=True)
            - List of predicted intervals (if return_detailed=True)
            - List of file paths (if return_detailed=True)
    """
    model.eval()
    total_loss_rank = 0.0
    batches_processed_rank = 0
    
    device = config["device"]
    hop_time = config["ssl_hop_length"] / config["sr"]
    threshold = config["threshold"]
    class_names = config["classes"]

    gt_list_rank: List[Dict[str, List[Tuple[float, float]]]] = []
    pred_list_rank: List[Dict[str, List[Tuple[float, float]]]] = []
    paths_list_rank: List[str] = []

    if hasattr(dataloader.sampler, 'set_epoch') and isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch_num if epoch_num is not None else 0)

    num_batches = len(dataloader)
    batches_to_run = min(limit, num_batches) if limit is not None else num_batches
    if batches_to_run == 0:
        empty_metrics = {thr: (0.0, 0.0, 0.0) for thr in IOU_THRESHOLDS} if compute_event_metrics else None
        if return_detailed:
            return 0.0, empty_metrics, [], [], []
        return 0.0, empty_metrics

    desc_prefix = f"Epoch {epoch_num}/{num_epochs_total} [Val]" if epoch_num else "[Test Eval]"
    pbar = tqdm(dataloader, total=batches_to_run, desc=desc_prefix, disable=not is_main_process(local_rank))

    with torch.no_grad():
        for i, batch_data in enumerate(pbar):
            if limit is not None and i >= limit:
                break
            try:
                # データローダーが返す4つの値を受け取るように修正
                features, labels, masks, paths = batch_data
            except ValueError as e:
                if is_main_process(local_rank):
                    logger.error(f"Error unpacking eval batch {i}: {e}. Skipping.")
                continue

            features, labels = features.to(device), labels.to(device)
            masks = masks.to(device)

            try:
                logits = model(features)
                probabilities = torch.softmax(logits, dim=1)

                output_len = probabilities.size(-1)
                label_len = labels.size(-1)
                min_len = min(output_len, label_len)

                if min_len <= 0:
                    if is_main_process(local_rank):
                        logger.warning(f"Skipping eval batch {i} due to zero/negative min_len ({min_len})")
                    continue
                
                loss = criterion(logits[..., :min_len], labels[..., :min_len])

                if not torch.isnan(loss):
                    total_loss_rank += loss.item()
                    batches_processed_rank += 1
                    if is_main_process(local_rank):
                        pbar.set_postfix(loss=f"{loss.item():.4f}")
                elif is_main_process(local_rank):
                    logger.warning(f"NaN loss in eval batch {i}.")

                if compute_event_metrics or return_detailed:
                    probs_matched = probabilities[..., :min_len].detach().cpu()
                    labels_matched = labels[..., :min_len].detach().cpu()
                    masks_matched = masks[..., :min_len].detach().cpu()

                    num_classes = len(class_names)
                    others_idx = class_names.index("others") if "others" in class_names else None

                    label_one_hot = F.one_hot(labels_matched.long(), num_classes=num_classes).permute(0, 2, 1).float()
                    label_one_hot *= masks_matched.unsqueeze(1)

                    max_probs, pred_indices = probs_matched.max(dim=1)
                    pred_one_hot = F.one_hot(pred_indices, num_classes=num_classes).permute(0, 2, 1).float()

                    mask_expanded = masks_matched.unsqueeze(1)
                    pred_one_hot *= mask_expanded

                    below_threshold = (max_probs < threshold) & (masks_matched > 0)
                    if below_threshold.any():
                        pred_one_hot_btc = pred_one_hot.permute(0, 2, 1).contiguous()
                        pred_one_hot_btc[below_threshold] = 0.0
                        if others_idx is not None:
                            pred_one_hot_btc[below_threshold, others_idx] = 1.0
                        pred_one_hot = pred_one_hot_btc.permute(0, 2, 1)

                    for b_idx in range(probs_matched.size(0)):
                        gt_intervals_sample = {}
                        for c_idx, class_name in enumerate(class_names):
                            gt_sequence = label_one_hot[b_idx, c_idx].numpy()
                            gt_intervals_sample[class_name] = sequence_to_intervals(
                                gt_sequence, hop_time, local_rank
                            )
                        gt_list_rank.append(gt_intervals_sample)

                        pred_intervals_sample = model_output_to_intervals(
                            pred_one_hot[b_idx], threshold, hop_time, class_names, local_rank
                        )
                        pred_list_rank.append(pred_intervals_sample)

                        if return_detailed:
                            paths_list_rank.append(paths[b_idx])

            except Exception as e:
                if is_main_process(local_rank):
                    logger.exception(f"Error during evaluation batch {i}: {e}.")

    # Aggregate loss across all ranks
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

    # Gather results for metrics calculation on rank 0
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

        if is_main_process(local_rank):
            for i in range(world_size):
                final_gt_list_rank0.extend(gathered_gts_obj[i])
                final_pred_list_rank0.extend(gathered_preds_obj[i])
                if return_detailed:
                    final_paths_list_rank0.extend(gathered_paths_obj[i])
    else:
        final_gt_list_rank0 = gt_list_rank
        final_pred_list_rank0 = pred_list_rank
        if return_detailed:
            final_paths_list_rank0 = paths_list_rank

    # Calculate metrics only on rank 0
    if is_main_process(local_rank) and compute_event_metrics:
        if len(final_gt_list_rank0) > 0:
            try:
                logger.info(f"Calculating event metrics for {len(final_gt_list_rank0)} samples on rank 0...")
                event_metrics_output = calculate_event_metrics(
                    final_gt_list_rank0, final_pred_list_rank0, class_names, IOU_THRESHOLDS, local_rank
                )
            except Exception as e:
                logger.error(f"Error calculating event metrics on rank 0: {e}")
                event_metrics_output = None
        else:
            logger.warning("No samples gathered on rank 0 for event metrics calculation.")
            event_metrics_output = {thr: (0.0, 0.0, 0.0) for thr in IOU_THRESHOLDS}

    if return_detailed:
        # Only rank 0 will have the full lists and metrics
        if not is_main_process(local_rank):
            final_gt_list_rank0, final_pred_list_rank0, final_paths_list_rank0 = [], [], []
            event_metrics_output = None 
        return avg_loss, event_metrics_output, final_gt_list_rank0, final_pred_list_rank0, final_paths_list_rank0
    else:
        if not is_main_process(local_rank):
            event_metrics_output = None
        return avg_loss, event_metrics_output

def expand_all_experiments(experiments_dict: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Expand all experiments to create a list of experiment configurations
    
    Args:
        experiments_dict: Dictionary of experiment configurations
        
    Returns:
        List of expanded experiment configurations
    """
    all_factors = list(experiments_dict.values())
    valid_factors = [f for f in all_factors if isinstance(f, list) and all(isinstance(item, dict) for item in f)]
    if len(valid_factors) != len(all_factors):
        logger.warning("Some items in EXPERIMENTS dict are not lists of dictionaries, skipping them for expansion.")

    if not valid_factors:
        return []
    combinations = list(itertools.product(*valid_factors))
    expanded_experiments = []
    seen_names = set()
    for combo in combinations:
        exp_config = {}
        for factor_dict in combo:
            exp_config.update(factor_dict)
        
        # Auto-generate name if not present
        if "name" not in exp_config:
            ssl_part = "no_ssl"
            if exp_config.get("ssl_model_name"):
                ssl_part = exp_config["ssl_model_name"].split('/')[-1]
            elif exp_config.get("feature_type"):
                ssl_part = exp_config["feature_type"]
            name_parts = [
                ssl_part, exp_config.get("architecture", "gru"),
                f"frac{exp_config.get('dataset_frac', '1.0')}"
            ]
            exp_name_gen = "_".join(map(str, name_parts)).replace('-', '_').replace('.', 'p')
            exp_config["name"] = exp_name_gen

        original_name = exp_config["name"]
        count = 1
        while exp_config["name"] in seen_names:
            exp_config["name"] = f"{original_name}_{count}"
            count += 1
        seen_names.add(exp_config["name"])
        expanded_experiments.append(exp_config)
    
    if is_main_process(0):
        logger.info(f"Generated {len(expanded_experiments)} experiment configurations from 'all'.")
    return expanded_experiments

def train_and_evaluate(base_config: Dict[str, Any],
                       experiment_config: Dict[str, Any],
                       local_rank: int, world_size: int,
                       train_batch_limit: Optional[int] = None,
                       eval_batch_limit: Optional[int] = None):
    """
    Train and evaluate a model with the given configuration
    
    Args:
        base_config: Base configuration dictionary
        experiment_config: Experiment-specific configuration dictionary
        local_rank: Process rank
        world_size: Total number of processes
        train_batch_limit: Optional batch limit for training (for smoke testing)
        eval_batch_limit: Optional batch limit for evaluation (for smoke testing)
    """
    config = {**base_config, **experiment_config}
    exp_name = config["name"]
    save_dir = Path(config["base_save_dir"]) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ★ fine-tune モード判定
    finetune_mode = bool(config.get("finetune", False))

    # Logger setup for this experiment, rank-specific handling
    exp_logger = logging.getLogger(exp_name)
    exp_logger.propagate = False
    exp_logger.handlers.clear()
    exp_logger.setLevel(logging.INFO)

    # File handler only on rank 0
    if is_main_process(local_rank):
        log_path = save_dir / f"{exp_name}.log"
        
        fh = logging.FileHandler(log_path, mode="a")
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        exp_logger.addHandler(fh)
        
        # Console output for rank 0 for this experiment's logger
        sh_exp = logging.StreamHandler()
        sh_exp.setFormatter(formatter)
        exp_logger.addHandler(sh_exp)

    # Patience counter loading (only on rank 0)
    patience_counter_val = 0
    if is_main_process(local_rank) and 'log_path' in locals():
        patience_counter_val = load_patience_from_log(log_path, config["early_stopping_patience"])
        exp_logger.info(f"[RESUME] {exp_name}: patience_counter={patience_counter_val}/{config['early_stopping_patience']}")

    if world_size > 1:
        patience_tensor = torch.tensor([patience_counter_val], dtype=torch.int, device=config["device"])
        dist.broadcast(patience_tensor, src=0)
        patience_counter = patience_tensor.item()
    else:
        patience_counter = patience_counter_val
        
    # ★ fine-tune のときは「もう patience に達してるから SKIP」を無効化
    if patience_counter >= config["early_stopping_patience"] and not finetune_mode:
        if is_main_process(local_rank):
            exp_logger.info(f"[SKIP] {exp_name}: already reached patience -> test only")
        from inference import test_model
        test_model(base_config, experiment_config, local_rank, world_size)
        if world_size > 1:
            dist.barrier()
        return

    # Log experiment details on rank 0
    is_smoke_test = train_batch_limit is not None or eval_batch_limit is not None
    if is_main_process(local_rank):
        test_type_str = "Smoke Test" if is_smoke_test else "Experiment"
        exp_logger.info(f"--- Starting {test_type_str}: {exp_name} (Rank {local_rank}/{world_size}) ---")
        if is_smoke_test:
            exp_logger.info(f"Smoke test limits: Train batches={train_batch_limit}, Eval batches={eval_batch_limit}")
        try:
            config_str = json.dumps(config, indent=2, default=str)
        except TypeError:
            config_str = str(config)
        exp_logger.info(f"Full Configuration (Rank 0 view): {config_str}")
        if 'save_dir' in locals():
            exp_logger.info(f"Results will be saved in: {save_dir}")
        exp_logger.info(f"Using device (this rank): {config['device']}")

    # Model initialization
    try:
        model = EventDetector(
            config=config, ssl_model_name=config.get("ssl_model_name"),
            freeze_feature_extractor=config.get("freeze_fe", config["default_freeze_fe"]),
            freeze_transformer_layers=config.get("freeze_transformer", config["default_freeze_transformer"]),
            architecture=config.get("architecture", "gru"),
            feature_type=config.get("feature_type", "raw"), local_rank=local_rank
        ).to(config["device"])

        # Ensure all ranks have completed model initialization before DDP wrapping
        if world_size > 1:
            dist.barrier()
            if is_main_process(local_rank):
                exp_logger.info("All ranks completed model initialization, proceeding with DDP wrapping")
            
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[config["device"]], output_device=config["device"], find_unused_parameters=True
            )
            
            # Verify DDP model parameter consistency
            if is_main_process(local_rank):
                actual_model_params = model.module
                num_params = sum(p.numel() for p in actual_model_params.parameters())
                num_trainable_params = sum(p.numel() for p in actual_model_params.parameters() if p.requires_grad)
                exp_logger.info(f"Rank 0 - Total parameters: {num_params:,}")
                exp_logger.info(f"Rank 0 - Trainable parameters: {num_trainable_params:,}")
            else:
                # Check parameter count on other ranks for debugging
                actual_model_params = model.module
                num_params = sum(p.numel() for p in actual_model_params.parameters())
                print(f"Rank {local_rank} - Total parameters: {num_params:,}")
        
        if is_main_process(local_rank):
            exp_logger.info("Model initialized successfully.")
            if world_size == 1:
                num_params = sum(p.numel() for p in model.parameters())
                num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                exp_logger.info(f"Total parameters: {num_params:,}")
                exp_logger.info(f"Trainable parameters: {num_trainable_params:,}")
    except Exception as e:
        if is_main_process(local_rank):
            exp_logger.exception(f"Failed to initialize model: {e}")
        if world_size > 1:
            dist.barrier()
        return

    # Optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if config["optimizer"].lower() == "adam":
        optimizer = optim.Adam(trainable_params, lr=config["lr"])
    elif config["optimizer"].lower() == "sgd":
        optimizer = optim.SGD(trainable_params, lr=config["lr"], momentum=config["momentum"])
    else:
        if is_main_process(local_rank):
            exp_logger.error(f"Unsupported optimizer: {config['optimizer']}")
        if world_size > 1:
            dist.barrier()
        return
    criterion = nn.BCEWithLogitsLoss()

    # Data loading
    try:
        with open(Path(config["train_json"]), 'r') as f:
            train_ann = json.load(f)
        with open(Path(config["val_json"]), 'r') as f:
            val_ann = json.load(f)

        dataset_train = SoundEventDataset(
            train_ann, config, config.get("feature_type", "raw"), config.get("dataset_frac", 1.0), local_rank
        )
        dataset_val = SoundEventDataset(
            val_ann, config, config.get("feature_type", "raw"), 1.0, local_rank
        )

        train_sampler, val_sampler = None, None
        shuffle_train = True
        if world_size > 1:
            train_sampler = DistributedSampler(
                dataset_train, num_replicas=world_size, rank=local_rank, shuffle=True, drop_last=True
            )
            val_sampler = DistributedSampler(
                dataset_val, num_replicas=world_size, rank=local_rank, shuffle=False
            )
            shuffle_train = False

        collate_fn = partial(pad_collate, config=config)
        num_loader_workers_main = min(4, (os.cpu_count() or 1) // max(1, world_size))
        if is_main_process(local_rank):
            exp_logger.info(f"Using {num_loader_workers_main} workers per DDP process for DataLoaders.")

        dataloader_train = DataLoader(
            dataset_train, batch_size=config["batch_size"], shuffle=shuffle_train, sampler=train_sampler,
            collate_fn=collate_fn, num_workers=num_loader_workers_main, pin_memory=True,
            persistent_workers=num_loader_workers_main > 0
        )
        dataloader_val = DataLoader(
            dataset_val, batch_size=config["batch_size"], shuffle=False, sampler=val_sampler,
            collate_fn=collate_fn, num_workers=num_loader_workers_main, pin_memory=True,
            persistent_workers=num_loader_workers_main > 0
        )

        if is_main_process(local_rank):
            exp_logger.info("Datasets and DataLoaders created successfully.")
            exp_logger.info(f"Total train samples: {len(dataset_train)}, Val samples: {len(dataset_val)}")
            try:
                batch_sample = next(iter(dataloader_train))
                if isinstance(batch_sample, (list, tuple)):
                    if len(batch_sample) >= 2:
                        feat_sample = batch_sample[0]
                        label_sample = batch_sample[1]
                    else:
                        raise ValueError(
                            f"Unexpected number of elements in sample batch: {len(batch_sample)}"
                        )
                else:
                    # 万が一タプルでない場合
                    raise ValueError(f"Unexpected batch type: {type(batch_sample)}")

                exp_logger.info(f"Sample batch feature shape (rank 0 view): {feat_sample.shape}")
                exp_logger.info(f"Sample batch label shape (rank 0 view): {label_sample.shape}")
            except Exception as e_dl:
                exp_logger.warning(f"Could not get sample batch shape: {e_dl}")
                
    except FileNotFoundError as e_fnf:
        if is_main_process(local_rank):
            exp_logger.error(f"Data JSON file not found: {e_fnf}")
        if world_size > 1:
            dist.barrier()
        return
    except Exception as e_data:
        if is_main_process(local_rank):
            exp_logger.exception(f"Error setting up data: {e_data}")
        if world_size > 1:
            dist.barrier()
        return
    # Resume / Fine-tune preparations
    # Default starting values
    start_epoch = 0
    best_val_iou = -1.0
    patience_counter = 0  # default patience counter

    # ★ 書き出し先 (今回の run の出力先)
    save_root = Path(config["base_save_dir"]) / exp_name
    resume_log = setup_resume_logging(save_root, local_rank)
    latest_ckpt_path = save_root / "latest.pth"

    # ★ 読み込み元 (事前学習モデルの所在。指定がなければ base_save_dir を使う)
    pretrained_root = Path(config.get("pretrained_root_dir", config["base_save_dir"])) / exp_name
    best_state_path = pretrained_root / "best_model_state.pth"  # 読み込み元のベストモデル

    if finetune_mode:
        # ★ Fine-tuning: 古いベストモデルの重みだけ読み込んで、epoch/patience はリセット
        target_model = model.module if world_size > 1 else model
        if best_state_path.exists():
            try:
                # PyTorch 2.6 以降対応
                try:
                    state_dict = torch.load(best_state_path, map_location=config["device"], weights_only=False)
                except TypeError:
                    # 古い PyTorch で weights_only がない場合
                    state_dict = torch.load(best_state_path, map_location=config["device"])
                target_model.load_state_dict(state_dict)
                if is_main_process(local_rank):
                    exp_logger.info(f"[FINETUNE] Loaded best model state from {best_state_path}")
            except Exception as e:
                if is_main_process(local_rank):
                    exp_logger.warning(f"[FINETUNE] Failed to load {best_state_path}: {e}. Starting from scratch.")
        else:
            if is_main_process(local_rank):
                exp_logger.warning(
                    f"[FINETUNE] best_model_state.pth not found at {best_state_path}, starting from scratch."
                )
        # start_epoch = 0, patience_counter = 0, best_val_iou = -1.0 のまま

    else:
        # ★ 通常の再開: latest.pth から epoch / best_iou / patience を復元
        try:
            epoch_loaded, best_loaded, patience_loaded = load_checkpoint(
                model.module if world_size > 1 else model,
                optimizer,
                latest_ckpt_path
            ) if latest_ckpt_path.exists() else (None, None, None)
            if epoch_loaded is not None:
                start_epoch = epoch_loaded + 1
                best_val_iou = best_loaded
                patience_counter = patience_loaded
                if is_main_process(local_rank):
                    exp_logger.info(
                        f"Resuming from epoch {start_epoch}, best IoU: {best_val_iou:.4f}, patience: {patience_counter}"
                    )
            else:
                start_epoch = 0
                best_val_iou = -1.0

        except Exception as e_resume:
            if is_main_process(local_rank):
                exp_logger.warning(f"Failed to load checkpoint: {e_resume}. Starting from scratch.")
            start_epoch = 0
            best_val_iou = -1.0


    # Skip smoke test if already done
    if is_smoke_test and resume_log:
        from metrics import has_smoke_test_finished
        if has_smoke_test_finished(resume_log):
            if is_main_process(local_rank):
                exp_logger.info("Smoke test already completed. Skipping training.")
            if world_size > 1:
                dist.barrier()
            return

    # Broadcast resume state to all ranks
    if world_size > 1:
        resume_tensor = torch.tensor([start_epoch, best_val_iou, float(patience_counter)], dtype=torch.float32, device=config["device"])
        dist.broadcast(resume_tensor, src=0)
        start_epoch = int(resume_tensor[0].item())
        best_val_iou = resume_tensor[1].item()
        patience_counter = int(resume_tensor[2].item())

    # Log the starting epoch after resume
    if is_main_process(local_rank):
        exp_logger.info(f"Training will start from epoch {start_epoch}")

    # Training loop
    if is_main_process(local_rank):
        exp_logger.info(f"=== Starting Training: {exp_name} ===")
    
    patience = patience_counter
    best_iou = best_val_iou

    for epoch in range(start_epoch, config["num_epochs"]):
        train_loss = train_one_epoch(
            model, dataloader_train, criterion, optimizer, config["device"],
            epoch + 1, config["num_epochs"], local_rank, train_batch_limit
        )
        
        eval_results = evaluate_epoch(
            model, dataloader_val, criterion, config,
            local_rank, world_size, compute_event_metrics=True,
            epoch_num=epoch + 1, num_epochs_total=config["num_epochs"],
            limit=eval_batch_limit
        )
        
        val_loss, val_metrics = eval_results
        
        if is_main_process(local_rank):
            exp_logger.info(f"Epoch {epoch + 1}/{config['num_epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # ---- ここから監視指標 (IoUベース) の計算 ----
            current_iou_score = 0.0

            if val_metrics:
                # IoU閾値0.5 の F1 をそのまま「IoUスコア」として使う
                prec_05, rec_05, f1_05 = val_metrics.get(0.5, (0.0, 0.0, 0.0))
                current_iou_score = f1_05

                exp_logger.info(
                    f"IoU > 0.5: Precision={prec_05:.3f}, Recall={rec_05:.3f}, F1={f1_05:.3f}"
                )

                # 他の IoU 閾値もログだけは出す
                for iou_thr, (precision, recall, f1) in sorted(val_metrics.items()):
                    if iou_thr != 0.5:
                        exp_logger.info(
                            f"IoU > {iou_thr:.2f}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}"
                        )

            # ---- ここまで current_iou_score が「監視するスコア」 ----
            
            # Save checkpoint
            save_dir = Path(config["base_save_dir"]) / exp_name
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": (model.module if world_size > 1 else model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_iou": best_iou,
                "current_val_iou": current_iou_score,
                "patience_counter": patience,
                "timestamp": time.time()
            }
            
            torch.save(checkpoint, save_dir / "latest.pth")
            
            # Simulate interruption for resume smoke test
            if config.get('stop_after') is not None and (epoch + 1) >= config['stop_after']:
                exp_logger.info(f"Simulated interruption at epoch {epoch + 1}. Exiting for resume test.")
                import sys
                sys.exit(2)

            # Update best model if improved (IoUベース指標で判定)
            if current_iou_score > best_iou:
                best_iou = current_iou_score
                checkpoint["best_val_iou"] = best_iou
                torch.save(checkpoint, save_dir / "best.pth")
                torch.save((model.module if world_size > 1 else model).state_dict(), save_dir / "best_model_state.pth")
                exp_logger.info(f"New best model saved! IoU-score: {best_iou:.4f}")
                patience = 0
            else:
                patience += 1
                exp_logger.info(
                    f"Patience (no IoU improvement): {patience}/{config['early_stopping_patience']}"
                )
            
            # Early stopping
            if patience >= config["early_stopping_patience"]:
                exp_logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Mark smoke test as completed if this is a smoke test
            if is_smoke_test:
                exp_logger.info("Smoke test epoch finished.")
        
        # Broadcast best_iou and patience to all ranks for synchronized stopping
        if world_size > 1:
            es_info = torch.tensor([best_iou, float(patience)], dtype=torch.float32, device=config["device"])
            dist.broadcast(es_info, src=0)
            best_iou = es_info[0].item()
            patience = int(es_info[1].item())
            
            if patience >= config["early_stopping_patience"]:
                break
    
    if is_main_process(local_rank):
        exp_logger.info(f"Training completed for {exp_name}")
        exp_logger.info(f"Best validation IoU-score: {best_iou:.4f}")
    
    # Run test evaluation
    # Run test evaluation on main process only
    if is_main_process(local_rank):
        from inference import test_model
        test_model(base_config, experiment_config, local_rank, world_size)
     
    # Ensure all ranks synchronize before exit
    if world_size > 1:
        dist.barrier()

def save_checkpoint(model, optimizer, epoch, best_iou, patience_counter, save_dir, filename="latest.pth"):
    """チェックポイントを保存"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_iou': best_iou,
        'patience_counter': patience_counter,
        'timestamp': time.time()
    }
    torch.save(checkpoint, save_dir / filename)
    
def load_checkpoint(model, optimizer, checkpoint_path):
    """チェックポイントを読み込み"""
    try:
        # PyTorch 2.6 以降: weights_only=False を明示
        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=False  # ← これを追加
        )
    except TypeError:
        # 古い PyTorch で weights_only 引数自体がない場合
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # チェックポイントに best_iou がある場合はそれを使い、なければ best_f1 を使う（後方互換性）
    best_score = checkpoint.get('best_iou', checkpoint.get('best_f1', -1.0))
    
    return checkpoint['epoch'], best_score, checkpoint['patience_counter']

def setup_resume_logging(save_dir, rank=0):
    """学習再開時のログ設定"""
    log_file = save_dir / f"{save_dir.name}.log"
    
    if rank == 0:  # メインプロセスのみログ設定
        # 既存のハンドラーをクリア
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # ファイルハンドラー（追記モード）とコンソールハンドラーを設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a'),  # 追記モード
                logging.StreamHandler()
            ]
        )
        
        # 再開時の区切り線
        if log_file.exists():
            logging.info("=" * 50)
            logging.info("TRAINING RESUMED")
            logging.info("=" * 50)
    
    return log_file

if __name__ == "__main__":
    import argparse, sys
    import json
    parser = argparse.ArgumentParser(description="Train and evaluate sound event detection model with resume support.")
    parser.add_argument('--base_config', '-b', required=True, help='Path to base config JSON file')
    parser.add_argument('--exp_config', '-e', required=True, help='Path to experiment config JSON file')
    parser.add_argument('--train_limit', type=int, default=None, help='Smoke test train batch limit')
    parser.add_argument('--eval_limit', type=int, default=None, help='Smoke test eval batch limit')
    parser.add_argument('--stop_after', type=int, default=None, help='Stop after N epochs to simulate interruption')
    args = parser.parse_args()
    # Load configurations
    try:
        with open(args.base_config) as f:
            base_cfg = json.load(f)
        with open(args.exp_config) as f:
            exp_cfg = json.load(f)
    except Exception as e_cfg:
        print(f"Failed to load config: {e_cfg}")
        sys.exit(1)
    # Inject stop_after into experiment config if provided
    if args.stop_after is not None:
        exp_cfg['stop_after'] = args.stop_after
    # Run training
    train_and_evaluate(base_cfg, exp_cfg, local_rank=0, world_size=1, train_batch_limit=args.train_limit, eval_batch_limit=args.eval_limit)
