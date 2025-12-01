import random
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm

from stabddg.model import StaBddG
from stabddg.ppi_dataset import SKEMPIDataset
from stabddg.training import _is_dist_initialized, _is_main_process


def skempi_eval(
    model: StaBddG,
    dataset: SKEMPIDataset,
    device,
    ensemble: int = 20,
    batch_size: int = 10000,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """
    Distributed-aware SKEMPI/Yeast evaluation.

    - In single-process / single-GPU mode, iterates over all samples.
    - In multi-process (torchrun) mode, each rank processes a strided
      subset of indices [rank, rank+world_size, ...].
    - Predictions from all ranks are gathered and concatenated on rank 0.
    """
    if _is_dist_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # Local predictions as a list of DataFrames per rank
    local_pred_dfs: list[pd.DataFrame] = []

    # Index-based striding over the dataset so each rank sees a disjoint subset
    n_items = len(dataset)
    iterator = range(rank, n_items, world_size)
    if sample_size is not None:
        sample_size = min(sample_size, n_items)
        idx_list = random.sample(iterator, sample_size)
    else:
        idx_list = list(iterator)

    for idx in tqdm(
        idx_list,
        desc=f"Interface (rank {rank})",
        disable=not _is_main_process(),
    ):
        sample = dataset[idx]

        complex, binder1, binder2 = (
            sample["complex"],
            sample["binder1"],
            sample["binder2"],
        )
        complex_mut_seqs = sample["complex_mut_seqs"].to(device)
        binder1_mut_seqs = sample["binder1_mut_seqs"].to(device)
        binder2_mut_seqs = sample["binder2_mut_seqs"].to(device)
        ddG = sample["ddG"]

        binding_ddG_pred_ensemble = []
        for _ in range(ensemble):
            N = complex_mut_seqs.shape[0]
            # convert number of tokens to number of sequences per batch
            M = batch_size // complex_mut_seqs.shape[1]

            binding_ddG_pred_ = []
            for batch_idx in range(0, N, M):
                B = min(N - batch_idx, M)
                with torch.no_grad():
                    batch_binding_ddG_pred = model(
                        complex,
                        binder1,
                        binder2,
                        complex_mut_seqs[batch_idx : batch_idx + B],
                        binder1_mut_seqs[batch_idx : batch_idx + B],
                        binder2_mut_seqs[batch_idx : batch_idx + B],
                    )
                binding_ddG_pred_.append(batch_binding_ddG_pred)

            binding_ddG_pred_ = torch.cat(binding_ddG_pred_)
            binding_ddG_pred_ensemble.append(binding_ddG_pred_.squeeze())

        binding_ddG_pred = torch.stack(binding_ddG_pred_ensemble).mean(dim=0).cpu()
        ddG_cpu = ddG.cpu()

        name, mutations = sample["name"], sample["mutation_list"]
        data = {
            "#Pdb": [name] * len(mutations),  # Repeat the name for all rows
            "Mutation": mutations,
            "ddG": ddG_cpu.detach().numpy(),
            "ddG_pred": binding_ddG_pred.detach().numpy(),
        }

        df = pd.DataFrame(data)
        local_pred_dfs.append(df)

    # Concatenate local predictions on each rank (may be empty)
    if local_pred_dfs:
        local_df = pd.concat(local_pred_dfs, ignore_index=True)
        local_records = local_df.to_dict(orient="records")
    else:
        local_records = []

    # If we're not in a distributed context, just return the single-process DataFrame
    if not _is_dist_initialized():
        return pd.DataFrame(local_records)

    # Distributed gather of python objects (lists of dicts) from all ranks
    world_size = dist.get_world_size()
    gathered_lists: list[list[dict]] = [None] * world_size  # type: ignore[assignment]
    dist.all_gather_object(gathered_lists, local_records)

    # Only the main process assembles and returns the full DataFrame
    if _is_main_process():
        all_records: list[dict] = []
        for recs in gathered_lists:
            if recs:
                all_records.extend(recs)
        return pd.DataFrame(all_records)

    # Non-main ranks return an empty DataFrame; caller should ignore it via _is_main_process
    return pd.DataFrame()
