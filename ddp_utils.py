"""
DDP utility functions for distributed training
"""
import os
import torch
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)

def setup_ddp():
    """
    Setup DistributedDataParallel (DDP) for training.
    
    Returns:
        Tuple[int, int]: local_rank and world_size
    """
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")  # 'nccl' for GPU, 'gloo' for CPU
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return local_rank, world_size
    return 0, 1  # Defaults for non-DDP run (rank 0, world_size 1)

def cleanup_ddp():
    """
    Clean up distributed process group
    """
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(local_rank: int) -> bool:
    """
    Check if current process is the main process (rank 0)
    
    Args:
        local_rank: Current process rank
        
    Returns:
        bool: True if main process, False otherwise
    """
    return local_rank == 0
