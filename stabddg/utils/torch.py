import torch


def get_device(local_rank: int | None):
    device = torch.device(f"cuda:{local_rank}") \
        if torch.cuda.is_available() and local_rank not in (-1, None) \
        else (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    return device
