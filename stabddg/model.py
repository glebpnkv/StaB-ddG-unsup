import torch
from torch import nn

from .mpnn_utils import featurize, ProteinMPNN


class StaBddG(nn.Module):
    def __init__(
        self,
        pmpnn: ProteinMPNN,
        use_antithetic_variates=True,
        noise_level=0.1,
        device="cuda"
    ):
        super(StaBddG, self).__init__()
        self.pmpnn: ProteinMPNN = pmpnn
        self.use_antithetic_variates = use_antithetic_variates
        self.noise_level = noise_level
        self.device = device

    def get_wt_seq(self, domain):
        """Returns the wild type sequence of a protein."""
        _, wt_seq, *_ = featurize([domain], self.device)
        return wt_seq

    def folding_dG(self, domain, seqs, decoding_order=None, backbone_noise=None):
        """Predicts the folding stability (dG) for a list of sequences."""
        B = seqs.shape[0]

        X_, _, mask_, _, chain_M_, residue_idx_, _, chain_encoding_all_ = featurize(
            [domain], self.device
        )
        X_, S_, mask_ = X_.repeat(B, 1, 1, 1), seqs.to(self.device), mask_.repeat(B, 1)
        chain_M_ = chain_M_.repeat(B, 1)
        residue_idx_, chain_encoding_all_ = (
            residue_idx_.repeat(B, 1),
            chain_encoding_all_.repeat(B, 1),
        )

        order = decoding_order.repeat(B, 1) if self.use_antithetic_variates else None
        backbone_noise = (
            backbone_noise.repeat(B, 1, 1, 1) if self.use_antithetic_variates else None
        )

        log_probs = self.pmpnn(
            X_,
            S_,
            mask_,
            chain_M_,
            residue_idx_,
            chain_encoding_all_,
            fix_order=order,
            fix_backbone_noise=backbone_noise,
        )

        seq_oh = torch.nn.functional.one_hot(seqs, 21).to(self.device)
        dG = torch.sum(seq_oh * log_probs, dim=(1, 2))

        return dG

    def folding_ddG(self, domain, mut_seqs, set_wt_seq=None):
        """Predicts the folding ddG."""
        X, wt_seq, _, _, chain_M, _, _, _ = featurize([domain], self.device)

        if set_wt_seq is not None:
            wt_seq = set_wt_seq

        decoding_order = (
            self._get_decoding_order(chain_M) if self.use_antithetic_variates else None
        )
        backbone_noise = (
            self._get_backbone_noise(X) if self.use_antithetic_variates else None
        )

        wt_dG = self.folding_dG(
            domain=domain,
            seqs=wt_seq,
            decoding_order=decoding_order,
            backbone_noise=backbone_noise
        )
        mut_dG = self.folding_dG(
            domain=domain,
            seqs=mut_seqs,
            decoding_order=decoding_order,
            backbone_noise=backbone_noise,
        )

        ddG = mut_dG - wt_dG

        return ddG

    def binding_ddG(
        self,
        complex,
        binder1,
        binder2,
        complex_mut_seqs,
        binder1_mut_seqs,
        binder2_mut_seqs,
    ):
        """We calculate the binding ddG by decomposing it into three folding ddG terms,
        corresponding to the entire complex and each individual binders.
        """
        complex_ddG_fold = self.folding_ddG(complex, complex_mut_seqs)
        binder1_ddG_fold = self.folding_ddG(binder1, binder1_mut_seqs)
        binder2_ddG_fold = self.folding_ddG(binder2, binder2_mut_seqs)

        ddG = complex_ddG_fold - (binder1_ddG_fold + binder2_ddG_fold)

        return ddG

    def forward(
        self,
        complex,
        binder1,
        binder2,
        complex_mut_seqs,
        binder1_mut_seqs,
        binder2_mut_seqs,
    ):
        return self.binding_ddG(
            complex,
            binder1,
            binder2,
            complex_mut_seqs,
            binder1_mut_seqs,
            binder2_mut_seqs,
        )

    def _get_decoding_order(self, chain_M):
        """Generate a random decoding order with the same shape as chain_M."""
        return torch.argsort(torch.abs(torch.randn(chain_M.shape, device=self.device)))

    def _get_backbone_noise(self, X):
        """Generate random backbone noise. Defaults to 0.1A."""
        return self.noise_level * torch.randn_like(X, device=self.device)

    def fused_forward_intact_datapoint(self, dp: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fused forward that consumes a full IntactDataset datapoint with keys:
          - 'anchor', 'same', 'opp', 'neutral'
        Each branch contains dict:
          - 'complex', 'binder1', 'binder2'
          - 'complex_mut_seqs', 'binder1_mut_seqs', 'binder2_mut_seqs'
        Returns four tensors (z_anchor, z_same, z_opp, z_neu), each shape [B].
        Gradients flow through self.pmpnn as usual.
        """
        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)

        def _fold_ddG(dom_dict, mut_seqs, cached):
            # Build per-structure antithetic buffers once per call to reuse across branches
            X = dom_dict["X"].to(device, non_blocking=True)
            mask = dom_dict["mask"].to(device, non_blocking=True)
            chain_M = mask  # same semantics
            residue_idx = dom_dict["residue_idx"].to(device, non_blocking=True)
            chain_encoding_all = dom_dict["chain_encoding_all"].to(device, non_blocking=True)
            wt_seq = dom_dict["S"].to(device, non_blocking=True).to(torch.long)
            mut_seq = mut_seqs.to(device, non_blocking=True).to(torch.long)

            if self.use_antithetic_variates:
                # Cache shared noise/order in 'cached' dict
                if "order" not in cached:
                    # Ensure order has shape [B,L] when expanded later; here we store base [L] or [1,L]
                    base_chain = chain_M if chain_M.dim() == 2 else chain_M.unsqueeze(0)
                    cached["order"] = torch.argsort(
                        torch.abs(torch.randn_like(base_chain, dtype=torch.float32, device=device)),
                        dim=-1
                    )
                if "noise" not in cached:
                    # Create noise with same shape as X; do not bake in batch dim here
                    cached["noise"] = (self.noise_level * torch.randn_like(X))
                order = cached["order"]
                noise = cached["noise"]
            else:
                order = None
                noise = None

            # Inner: forward once for WT and once for MUT using same cached buffers
            def _mpnn(seqs):
                B = seqs.shape[0]

                # Expand structural tensors to batch B without materializing copies
                def _expand_to_batch(t, target_dim, expand_dims):
                    # t: tensor, target_dim: expected rank after expansion at front, expand_dims: sizes tuple for leading dims
                    if t.dim() == target_dim and t.shape[0] == B:
                        return t
                    if t.dim() == (target_dim - 1):
                        return t.unsqueeze(0).expand(expand_dims + t.shape)
                    # Already batched but different B: expand leading
                    if t.dim() == target_dim and t.shape[0] != B:
                        shape = (B,) + t.shape[1:]
                        return t.expand(shape)
                    return t

                # X can be [L,atoms,3], [1,L,atoms,3], or already batched; make [B,L,atoms,3] while preserving trailing dims
                if X.dim() >= 3:
                    Xb = _expand_to_batch(X, target_dim=X.dim() if X.dim() > 3 else 4, expand_dims=(B,))
                else:
                    raise RuntimeError("X has unexpected rank; expected >=3.")
                # mask/chain_M/residue_idx/chain_encoding_all to [B,L]
                mb = mask if (mask.dim() == 2 and mask.shape[0] == B) else (
                    mask.unsqueeze(0).expand(B, -1) if mask.dim() == 1 else mask.expand(B, -1)
                )
                cb = chain_M if (chain_M.dim() == 2 and chain_M.shape[0] == B) else (
                    chain_M.unsqueeze(0).expand(B, -1) if chain_M.dim() == 1 else chain_M.expand(B, -1)
                )
                rb = residue_idx if (residue_idx.dim() == 2 and residue_idx.shape[0] == B) else (
                    residue_idx.unsqueeze(0).expand(B, -1) if residue_idx.dim() == 1 else residue_idx.expand(B, -1)
                )
                eb = chain_encoding_all if (chain_encoding_all.dim() == 2 and chain_encoding_all.shape[0] == B) else (
                    chain_encoding_all.unsqueeze(0).expand(B, -1) if chain_encoding_all.dim() == 1 else chain_encoding_all.expand(B, -1)
                )

                # Order to [B,L]
                if order is not None:
                    ob = order
                    if ob.dim() == 1:
                        ob = ob.unsqueeze(0).expand(B, -1)
                    elif ob.dim() == 2 and ob.shape[0] != B:
                        ob = ob.expand(B, -1)
                else:
                    ob = None

                # Noise: match leading batch dimension B regardless of original rank
                if noise is not None:
                    if noise.shape[0] == B:
                        nb = noise
                    else:
                        # Insert a batch dim at the front if missing
                        nb = noise
                        if nb.shape[0] != B:
                            # If noise has no batch dim (e.g., [L,atoms,3] or [1,L,atoms,3] or [1,1,L,atoms,3]),
                            # expand its first dim to B while keeping other dims as-is.
                            # If first dim is 1 (or absent), expand to B; otherwise, expand leading dims appropriately.
                            if nb.dim() == X.dim() and nb.shape[0] in (1,):
                                nb = nb.expand((B,) + nb.shape[1:])
                            elif nb.dim() == (X.dim() - 1):
                                nb = nb.unsqueeze(0).expand((B,) + nb.shape)
                            else:
                                # General case: ensure front batch dim B
                                if nb.shape[0] != B:
                                    # Try unsqueeze then expand
                                    nb = nb
                                    if nb.dim() == 3:  # [L,atoms,3]
                                        nb = nb.unsqueeze(0).expand(B, -1, -1, -1)
                                    elif nb.dim() == 4:  # [1,L,atoms,3] or [*,L,atoms,3]
                                        if nb.shape[0] == 1:
                                            nb = nb.expand(B, -1, -1, -1)
                                        else:
                                            nb = nb.expand(B, -1, -1, -1)
                                    else:
                                        # Fallback: align first dim to B
                                        shape = (B,) + nb.shape[1:]
                                        nb = nb.expand(shape)
                else:
                    nb = None

                log_probs = self.pmpnn(
                    Xb, seqs, mb, cb, rb, eb,
                    fix_order=ob,
                    fix_backbone_noise=nb
                )
                seq_oh = torch.nn.functional.one_hot(seqs, 21).to(log_probs.dtype)
                return torch.sum(seq_oh * log_probs, dim=(1, 2))

            wt_dG = _mpnn(wt_seq)
            mut_dG = _mpnn(mut_seq)
            return (mut_dG - wt_dG)

        # Build shared caches per unique structure across branches to reuse antithetic buffers
        caches = {
            "complex": {},
            "binder1": {},
            "binder2": {},
        }

        def _branch(b):
            item = dp[b]
            ddG_complex = _fold_ddG(item["complex"], item["complex_mut_seqs"], caches["complex"])
            ddG_b1 = _fold_ddG(item["binder1"], item["binder1_mut_seqs"], caches["binder1"])
            ddG_b2 = _fold_ddG(item["binder2"], item["binder2_mut_seqs"], caches["binder2"])
            return ddG_complex - (ddG_b1 + ddG_b2)

        z_anchor = _branch("anchor")
        z_same = _branch("same")
        z_opp = _branch("opp")
        z_neu = _branch("neutral")

        return z_anchor, z_same, z_opp, z_neu


class LinearModel(nn.Module):
    def __init__(self, num_features):
        super(LinearModel, self).__init__()
        # Linear layer with num_features inputs and 1 output (with bias term)
        self.linear = nn.Linear(num_features, 1, bias=True)

    def forward(self, x):
        return self.linear(x).squeeze(-1)
