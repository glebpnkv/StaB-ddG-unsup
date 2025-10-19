import torch

from model import StaBddG
from mpnn_utils import ProteinMPNN


def _mpnn_forward_from_dict(
    pmpnn: ProteinMPNN,
    device: torch.device,
    dom_dict: dict,
    seqs: torch.Tensor,
    use_antithetic_variates: bool,
    noise_level: float,
    decoding_order: torch.Tensor | None = None,
    backbone_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Minimal wrapper that feeds already-featurized dicts (like IntactDataset outputs) to ProteinMPNN.
    dom_dict must provide keys: 'X', 'S', 'mask', 'chain_M', 'residue_idx', 'chain_encoding_all'
    If some keys are named differently in saved tensors, we derive what's missing:
      - chain_M: 1 where mask>0 else 0
    Shapes:
      X: [B,L,atoms,3] or [L,atoms,3] (we will expand to batch)
      S: [B,L] or [L] (ignored for WT unless needed)
      mask, chain_M, residue_idx, chain_encoding_all: [B,L] or [L]
    seqs: [B,L] mutated or WT sequences (int64)
    Returns log_probs: [B,L,21]
    """
    # Pull tensors and ensure torch types on device
    def _t(x, dtype=None):
        t = x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
        if dtype is not None:
            t = t.to(dtype)
        return t.to(device)

    X = _t(dom_dict["X"], torch.float32)
    mask = _t(dom_dict.get("mask", dom_dict.get("chain_M", None)))
    if mask is None:
        raise ValueError("mask/chain_M missing in dom_dict")
    chain_M = _t(dom_dict.get("chain_M", (mask > 0).to(torch.int32)), torch.int64)
    residue_idx = _t(dom_dict["residue_idx"], torch.int64)
    chain_encoding_all = _t(dom_dict["chain_encoding_all"], torch.int64)

    # Ensure batch dimension
    B = seqs.shape[0]
    if X.dim() == 3:
        X = X.unsqueeze(0).expand(B, -1, -1, -1)        # [B,L,atoms,3]
    elif X.dim() == 4 and X.shape[0] != B:
        X = X.expand(B, -1, -1, -1)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0).expand(B, -1)          # [B,L]
    if chain_M.dim() == 1:
        chain_M = chain_M.unsqueeze(0).expand(B, -1)
    if residue_idx.dim() == 1:
        residue_idx = residue_idx.unsqueeze(0).expand(B, -1)
    if chain_encoding_all.dim() == 1:
        chain_encoding_all = chain_encoding_all.unsqueeze(0).expand(B, -1)

    # Fix orders/noise if antithetic variates are used
    if use_antithetic_variates:
        if decoding_order is None:
            decoding_order = torch.argsort(torch.abs(torch.randn_like(chain_M, dtype=torch.float32)))
        else:
            decoding_order = decoding_order.to(device)
        if backbone_noise is None:
            backbone_noise = noise_level * torch.randn_like(X)
        else:
            backbone_noise = backbone_noise.to(device)
    else:
        decoding_order = None
        backbone_noise = None

    log_probs = pmpnn(
        X,
        seqs.to(device),
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        fix_order=decoding_order,
        fix_backbone_noise=backbone_noise,
    )

    del X, mask, chain_M, residue_idx, chain_encoding_all
    import gc as _gc
    _gc.collect()
    return log_probs


def folding_dG_from_dict(
    pmpnn: ProteinMPNN,
    device: torch.device,
    dom_dict: dict,
    seqs: torch.Tensor,
    use_antithetic_variates: bool = True,
    noise_level: float = 0.1,
    decoding_order: torch.Tensor | None = None,
    backbone_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Equivalent to StaBddG.folding_dG but uses pre-featurized dom_dict (as produced by IntactDataset).
    Returns dG: [B]
    """
    log_probs = _mpnn_forward_from_dict(
        pmpnn=pmpnn,
        device=device,
        dom_dict=dom_dict,
        seqs=seqs,
        use_antithetic_variates=use_antithetic_variates,
        noise_level=noise_level,
        decoding_order=decoding_order,
        backbone_noise=backbone_noise,
    )
    seq_oh = torch.nn.functional.one_hot(seqs.to(device), 21).to(device)
    dG = torch.sum(seq_oh * log_probs, dim=(1, 2))

    del log_probs, seq_oh
    import gc as _gc
    _gc.collect()

    return dG


def folding_ddG_from_dict(
    pmpnn: ProteinMPNN,
    device: torch.device,
    dom_dict: dict,
    mut_seqs: torch.Tensor,
    use_antithetic_variates: bool = True,
    noise_level: float = 0.1,
    set_wt_seq: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mirror StaBddG.folding_ddG but consume pre-featurized dicts.
    dom_dict provides keys matching featurize outputs mapping:
      X <- dom_dict['X']
      wt_seq (S) <- dom_dict['S']
      mask <- dom_dict['mask']
      chain_M <- dom_dict['mask'] (same semantics here)
      residue_idx <- dom_dict['residue_idx']
      chain_encoding_all <- dom_dict['chain_encoding_all']
    """
    # Extract tensors (single structure, later repeated inside folding_dG_from_dict)
    X = dom_dict["X"].to(device) if isinstance(dom_dict["X"], torch.Tensor) else torch.from_numpy(dom_dict["X"]).to(device)
    S = dom_dict["S"].to(device) if isinstance(dom_dict["S"], torch.Tensor) else torch.from_numpy(dom_dict["S"]).to(device)
    mask = dom_dict["mask"].to(device) if isinstance(dom_dict["mask"], torch.Tensor) else torch.from_numpy(dom_dict["mask"]).to(device)
    # In our stored tensors, mask serves the role of chain_M in ProteinMPNN usage
    chain_M = mask
    residue_idx = dom_dict["residue_idx"].to(device) if isinstance(dom_dict["residue_idx"], torch.Tensor) else torch.from_numpy(dom_dict["residue_idx"]).to(device)
    chain_encoding_all = dom_dict["chain_encoding_all"].to(device) if isinstance(dom_dict["chain_encoding_all"], torch.Tensor) else torch.from_numpy(dom_dict["chain_encoding_all"]).to(device)

    # WT sequence selection
    wt_seq = set_wt_seq.to(device) if set_wt_seq is not None else S.to(device)

    # Build antithetic components (like StaBddG.folding_ddG)
    decoding_order = (
        torch.argsort(torch.abs(torch.randn(chain_M.shape, device=device)))
        if use_antithetic_variates else None
    )
    backbone_noise = (
        noise_level * torch.randn_like(X, device=device)
        if use_antithetic_variates else None
    )

    # Compute dG for WT and mutant using folding_dG_from_dict logic:
    # emulate StaBddG.folding_dG batching: repeat structure tensors to batch size inside _mpnn_forward_from_dict
    wt_dG = folding_dG_from_dict(
        pmpnn=pmpnn,
        device=device,
        dom_dict={
            "X": X,
            "S": wt_seq,  # not used internally except for types; seqs is passed separately
            "mask": mask,
            "chain_M": chain_M,
            "residue_idx": residue_idx,
            "chain_encoding_all": chain_encoding_all,
        },
        seqs=wt_seq.to(torch.long).to(device),
        use_antithetic_variates=use_antithetic_variates,
        noise_level=noise_level,
        decoding_order=decoding_order,
        backbone_noise=backbone_noise,
    )

    mut_dG = folding_dG_from_dict(
        pmpnn=pmpnn,
        device=device,
        dom_dict={
            "X": X,
            "S": S,
            "mask": mask,
            "chain_M": chain_M,
            "residue_idx": residue_idx,
            "chain_encoding_all": chain_encoding_all,
        },
        seqs=mut_seqs.to(torch.long).to(device),
        use_antithetic_variates=use_antithetic_variates,
        noise_level=noise_level,
        decoding_order=decoding_order,
        backbone_noise=backbone_noise,
    )

    # Explicitly freeing up memory
    del X, S, mask, chain_M, residue_idx, chain_encoding_all, decoding_order, backbone_noise, wt_seq, mut_seqs
    import gc as _gc
    _gc.collect()

    return mut_dG - wt_dG


def binding_ddG_from_intact_datapoint(
    model: StaBddG,
    item: dict,
) -> torch.Tensor:
    """
    Adapted binding_ddG that consumes a batched IntactDataset item branch (e.g., out['anchor']).
    item must contain:
      - 'complex' dict, 'binder1' dict, 'binder2' dict
      - 'complex_mut_seqs' [B,L], 'binder1_mut_seqs' [B,L], 'binder2_mut_seqs' [B,L]
    Returns ddG: [B]
    """
    device = model.device if isinstance(model.device, torch.device) else torch.device(model.device)
    complex_dict = item["complex"]
    binder1_dict = item["binder1"]
    binder2_dict = item["binder2"]
    complex_mut = item["complex_mut_seqs"].to(torch.long).to(device)
    binder1_mut = item["binder1_mut_seqs"].to(torch.long).to(device)
    binder2_mut = item["binder2_mut_seqs"].to(torch.long).to(device)

    ddG_complex = folding_ddG_from_dict(
        pmpnn=model.pmpnn,
        device=device,
        dom_dict=complex_dict,
        mut_seqs=complex_mut,
        use_antithetic_variates=model.use_antithetic_variates,
        noise_level=model.noise_level,
    )
    ddG_b1 = folding_ddG_from_dict(
        pmpnn=model.pmpnn,
        device=device,
        dom_dict=binder1_dict,
        mut_seqs=binder1_mut,
        use_antithetic_variates=model.use_antithetic_variates,
        noise_level=model.noise_level,
    )
    ddG_b2 = folding_ddG_from_dict(
        pmpnn=model.pmpnn,
        device=device,
        dom_dict=binder2_dict,
        mut_seqs=binder2_mut,
        use_antithetic_variates=model.use_antithetic_variates,
        noise_level=model.noise_level,
    )

    # Explicitly freeing up memory
    del complex_dict, binder1_dict, binder2_dict, complex_mut, binder1_mut, binder2_mut
    import gc as _gc
    _gc.collect()

    return ddG_complex - (ddG_b1 + ddG_b2)


def forward_whole_intact_datapoint(
    model: StaBddG,
    dp: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Consume a full IntactDataset datapoint (with keys 'anchor','same','opp','neutral'),
    compute binding ddG embeddings for each, and return a tuple suitable for ContrastiveLoss.forward:
      (z_anchor, z_same, z_opp, z_neutral)
    Each z_* is a 1D tensor per sample (binding ddG), shape [B] for each branch.
    """
    z_anchor = binding_ddG_from_intact_datapoint(model, dp["anchor"])
    z_same = binding_ddG_from_intact_datapoint(model, dp["same"])
    z_opp = binding_ddG_from_intact_datapoint(model, dp["opp"])
    z_neu = binding_ddG_from_intact_datapoint(model, dp["neutral"])
    return z_anchor, z_same, z_opp, z_neu
