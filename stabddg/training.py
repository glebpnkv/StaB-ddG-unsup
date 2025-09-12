import torch
from torch import distributed as dist


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    return (not _is_dist_initialized()) or dist.get_rank() == 0


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    # Unwrap DDP model to access underlying module (for saving state dicts, etc.)
    return model.module if hasattr(model, "module") else model
