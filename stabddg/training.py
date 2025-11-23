import torch
from torch import distributed as dist


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    return (not _is_dist_initialized()) or dist.get_rank() == 0


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    # Unwrap DDP model to access underlying module (for saving state dicts, etc.)
    return model.module if hasattr(model, "module") else model


def _distributed_concat_1d(world_size, t: torch.Tensor) -> torch.Tensor:
    """Concatenate variable-length 1D tensors from all ranks."""
    if not _is_dist_initialized():
        return t
    local_len = torch.tensor([t.numel()], device=t.device, dtype=torch.long)
    lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(lens, local_len)
    max_len = int(torch.max(torch.stack(lens)).item())
    # pad to max_len
    pad_len = max_len - t.numel()
    if pad_len > 0:
        t_padded = torch.cat([t, torch.empty(pad_len, device=t.device, dtype=t.dtype)])
    else:
        t_padded = t
    gathered = [torch.empty_like(t_padded) for _ in range(world_size)]
    dist.all_gather(gathered, t_padded)
    # trim per rank by its true length and concat
    outs = []
    for g, cur_len in zip(gathered, lens):
        outs.append(g[: int(cur_len.item())])
    return torch.cat(outs, dim=0)