import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class ContrastiveLoss(_Loss):
    """
    Standard supervised contrastive loss (SupCon) over a contrast pool of (+) and (-) samples,
    with neutrals used only to robustly centre/scale z, plus an optional sign-direction penalty.

    Inputs:
      z: (B, 1) anchors
      z_sign: (B, 1) anchor signs in {-1, +1} (0 allowed -> anchor ignored in SupCon/sign)
      z_pos: (B_pos, 1) positives (label +1)
      z_neg: (B_neg, 1) negatives (label -1)
      z_neu: (B_neu, 1) neutrals (used for normalisation + optional neutral penalty)
    """

    __constants__ = ["reduction"]

    def __init__(
        self,
        tau: float = 1.0,
        similarity: str = "cosine",  # "distance" or "cosine" (also accepts "dot")
        lambda_supcon: float = 1.0,
        lambda_sign: float = 10.0,
        sign_margin: float = 0.2,       # margin in *neutral-normalised* units
        lambda_neutral: float = 0.0,    # optional: keep neutrals near 0 in normalised space
        eps: float = 1e-1,
        reduction: str = "mean",
        use_neutral_normalizer: bool = False,
        robust_neutral_stats: bool = True,  # median / MAD vs mean / std
    ):
        super().__init__(reduction=reduction)
        self.tau = float(tau)
        self.similarity = similarity
        self.lambda_supcon = float(lambda_supcon)
        self.lambda_sign = float(lambda_sign)
        self.sign_margin = float(sign_margin)
        self.lambda_neutral = float(lambda_neutral)
        self.eps = float(eps)
        self.use_neutral_normalizer = bool(use_neutral_normalizer)
        self.robust_neutral_stats = bool(robust_neutral_stats)

    def _neutral_center_scale(self, z_neu: torch.Tensor, dtype, device):
        # Returns (mu0, sigma0) as scalars on correct device/dtype (detached).
        if (z_neu is None) or (z_neu.numel() == 0) or (not self.use_neutral_normalizer):
            mu0 = torch.zeros((), dtype=dtype, device=device)
            sigma0 = torch.ones((), dtype=dtype, device=device)
            return mu0, sigma0

        z0 = z_neu.view(-1).to(dtype=dtype)

        if self.robust_neutral_stats:
            mu0 = z0.median()
            mad = (z0 - mu0).abs().median()
            sigma0 = mad + self.eps
        else:
            mu0 = z0.mean()
            sigma0 = z0.std(unbiased=False) + self.eps

        return mu0.detach(), sigma0.detach()

    def _normalize(self, x: torch.Tensor, mu0: torch.Tensor, sigma0: torch.Tensor) -> torch.Tensor:
        return (x - mu0) / sigma0

    def _sim_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a: (B, 1), b: (N, 1) -> returns logits (B, N) WITHOUT temperature applied yet.
        similarity:
          - "distance": negative squared distance
          - "cosine": cosine similarity
          - "dot": dot product
        """
        B = a.shape[0]
        N = b.shape[0]
        a2 = a.view(B, 1)
        b2 = b.view(1, N)

        if self.similarity == "distance":
            # logits ~ -||a-b||^2
            return -((a2 - b2) ** 2)
        if self.similarity == "cosine":
            num = a2 * b2
            den = (a2.abs() * b2.abs()).clamp_min(self.eps)
            return num / den
        if self.similarity == "dot":
            return a2 * b2

        raise ValueError(f"Unknown similarity='{self.similarity}'. Use 'distance', 'cosine', or 'dot'.")

    def _masked_logsumexp(self, x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        # x: (B,N), mask: (B,N) bool
        x_masked = x.masked_fill(~mask, float("-inf"))
        return torch.logsumexp(x_masked, dim=dim)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return x.mean()
        if self.reduction == "sum":
            return x.sum()
        return x  # "none"

    def forward(self, z, z_sign, z_pos, z_neg, z_neu):
        loss, _ = self.return_losses_and_metrics(z, z_sign, z_pos, z_neg, z_neu)
        return loss

    @torch.no_grad()
    def _per_anchor_masked_mean(self, mat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Returns (B,) mean over selected entries per row; 0 where count==0
        counts = mask.sum(dim=1).clamp_min(1)
        s = (mat * mask.to(mat.dtype)).sum(dim=1) / counts
        s = s * (mask.sum(dim=1) > 0).to(mat.dtype)
        return s

    def return_losses_and_metrics(self, z, z_sign, z_pos, z_neg, z_neu):
        dtype = z.dtype
        device = z.device

        # Flatten to (B,1) shape consistently
        z = z.view(-1, 1)
        z_sign = z_sign.view(-1, 1)

        z_pos = z_pos.view(-1, 1)
        z_neg = z_neg.view(-1, 1)
        z_neu = z_neu.view(-1, 1)

        # Neutral-based robust centering/scaling
        mu0, sigma0 = self._neutral_center_scale(z_neu, dtype=dtype, device=device)

        z_hat = self._normalize(z, mu0, sigma0)
        z_pos_hat = self._normalize(z_pos, mu0, sigma0)
        z_neg_hat = self._normalize(z_neg, mu0, sigma0)
        z_neu_hat = self._normalize(z_neu, mu0, sigma0) if z_neu.numel() > 0 else z_neu

        # Build contrast pool: (+) then (-)
        pool = torch.cat([z_pos_hat, z_neg_hat], dim=0)  # (N,1)
        pool_sign = torch.cat(
            [
                torch.ones_like(z_pos_hat, dtype=dtype, device=device),
                - torch.ones_like(z_neg_hat, dtype=dtype, device=device),
                # torch.ones((z_pos_hat.shape[0], 1), dtype=z_sign.dtype, device=device),
                # -torch.ones((z_neg_hat.shape[0], 1), dtype=z_sign.dtype, device=device),
            ],
            dim=0,
        )  # (N,1)

        # SupCon only defined for anchors with sign in {-1,+1}
        anchor_is_labeled = (z_sign.abs() == 1).view(-1)  # (B,)

        # Similarity logits (B,N) with temperature
        logits_raw = self._sim_matrix(z_hat, pool)  # (B,N)
        logits = logits_raw / self.tau

        # Masks
        # same sign: y_i * y_j > 0 ; opposite: < 0
        w = (z_sign.to(dtype) * pool_sign.t())  # (B,N)
        pos_mask = (w > 0) & anchor_is_labeled[:, None]
        neg_mask = (w < 0) & anchor_is_labeled[:, None]

        pos_count = pos_mask.sum(dim=1)  # (B,)
        neg_count = neg_mask.sum(dim=1)  # (B,)
        valid = anchor_is_labeled & (pos_count > 0) & (neg_count > 0) & (pool.shape[0] > 0)

        # ---- SupCon loss:  -mean_pos_logits + logsumexp(all_logits)
        # mean over positives per anchor
        pos_sum = (logits * pos_mask.to(dtype)).sum(dim=1)  # (B,)
        neg_sum = (logits * neg_mask.to(dtype)).sum(dim=1)  # (B,)
        pos_count_safe = pos_count.clamp_min(1).to(dtype)
        neg_count_safe = neg_count.clamp_min(1).to(dtype)
        mean_pos = pos_sum / pos_count_safe  # (B,)
        mean_neg = neg_sum / neg_count_safe

        lse_all = torch.logsumexp(logits, dim=1)  # (B,)

        supcon_vec = (-mean_pos + lse_all) * valid.to(dtype)

        if valid.any():
            supcon_loss = self._reduce(supcon_vec[valid])
        else:
            supcon_loss = z_hat.new_zeros(())

        # ---- Sign-direction loss on all weakly-labeled points: anchors + sampled pos/neg
        # Uses normalised z_hat, so margin is in neutral-normalised units.
        y_all = torch.cat(
            [z_sign, torch.ones_like(z_pos_hat), -torch.ones_like(z_neg_hat)], dim=0
        ).view(-1, 1)
        z_all = torch.cat([z_hat, z_pos_hat, z_neg_hat], dim=0).view(-1, 1)

        labeled_all = (y_all.abs() == 1).view(-1)
        sign_loss = z_hat.new_zeros(())
        # Computes sign loss when labels exist and regularisation is enabled
        if self.lambda_sign != 0.0 and labeled_all.any():
            yz = (y_all * z_all).view(-1)
            sign_loss = F.softplus(self.sign_margin - yz).mean()

        # Optional: keep neutrals near 0 in normalised space
        neutral_reg = z_hat.new_zeros(())
        # Computes neutral regularization loss if enabled
        if self.lambda_neutral != 0.0 and z_neu_hat is not None and z_neu_hat.numel() > 0:
            neutral_reg = (z_neu_hat.view(-1) ** 2).mean()

        # Total loss
        loss = self.lambda_supcon * supcon_loss + self.lambda_sign * sign_loss + self.lambda_neutral * neutral_reg

        # Metrics (always report cosine + distance regardless of training similarity)
        with torch.no_grad():
            B = z_hat.shape[0]
            N = pool.shape[0]

            # Cosine matrix (B,N)
            a = z_hat.view(B, 1)
            b = pool.view(1, N) if N > 0 else pool.view(1, 0)
            # Handles zero‑sized pool by creating zero tensors
            if N > 0:
                cos_mat = (a * b) / (a.abs() * b.abs()).clamp_min(self.eps)
                dist_mat = (a - b).abs()
                sqdist_mat = (a - b) ** 2
            else:
                cos_mat = z_hat.new_zeros((B, 0))
                dist_mat = z_hat.new_zeros((B, 0))
                sqdist_mat = z_hat.new_zeros((B, 0))

            # Per-anchor means
            cos_pos = self._per_anchor_masked_mean(cos_mat, pos_mask) if N > 0 else z_hat.new_zeros((B,))
            cos_neg = self._per_anchor_masked_mean(cos_mat, neg_mask) if N > 0 else z_hat.new_zeros((B,))
            dist_pos = self._per_anchor_masked_mean(dist_mat, pos_mask) if N > 0 else z_hat.new_zeros((B,))
            dist_neg = self._per_anchor_masked_mean(dist_mat, neg_mask) if N > 0 else z_hat.new_zeros((B,))
            sqdist_pos = self._per_anchor_masked_mean(sqdist_mat, pos_mask) if N > 0 else z_hat.new_zeros((B,))
            sqdist_neg = self._per_anchor_masked_mean(sqdist_mat, neg_mask) if N > 0 else z_hat.new_zeros((B,))

            # SupCon breakdown pieces
            pos_term_vec = (-mean_pos) * valid.to(dtype)       # (B,)
            neg_term_vec = (-mean_neg) * valid.to(dtype)
            den_term_vec = (lse_all) * valid.to(dtype)         # (B,)

            # Denominator contribution fractions (how much of exp-sum comes from pos vs neg)
            if valid.any() and N > 0:
                lse_pos = self._masked_logsumexp(logits, pos_mask, dim=1)
                lse_neg = self._masked_logsumexp(logits, neg_mask, dim=1)
                # fractions in [0,1]
                pos_frac = torch.exp(lse_pos - lse_all).masked_fill(~valid, 0.0)
                neg_frac = torch.exp(lse_neg - lse_all).masked_fill(~valid, 0.0)
            else:
                pos_frac = z_hat.new_zeros((B,))
                neg_frac = z_hat.new_zeros((B,))

            # Sign stats
            sign_violation_rate = 0.0
            # Computes sign violation rate over labeled data
            if labeled_all.any():
                yz = (y_all * z_all).view(-1)
                sign_violation_rate = (yz < 0).to(torch.float32).mean().item()

            def mean_over_valid(v):
                return v[valid].mean().item() if valid.any() else 0.0

            metrics = {
                "loss_total": float(loss.detach().item()) if loss.numel() == 1 else float("nan"),
                "loss_supcon": float(supcon_loss.detach().item()) if supcon_loss.numel() == 1 else float("nan"),
                "loss_sign": float(sign_loss.detach().item()),
                "loss_neutral_reg": float(neutral_reg.detach().item()),
                "supcon_pos_term": mean_over_valid(pos_term_vec),
                "supcon_den_term": mean_over_valid(den_term_vec),
                "supcon_neg": mean_over_valid(neg_term_vec),
                "supcon_denom_pos_frac": mean_over_valid(pos_frac),
                "supcon_denom_neg_frac": mean_over_valid(neg_frac),
                "cos_pos_mean": mean_over_valid(cos_pos),
                "cos_neg_mean": mean_over_valid(cos_neg),
                "dist_pos_mean": mean_over_valid(dist_pos),
                "dist_neg_mean": mean_over_valid(dist_neg),
                "sqdist_pos_mean": mean_over_valid(sqdist_pos),
                "sqdist_neg_mean": mean_over_valid(sqdist_neg),
                "neutral_mu0": float(mu0.detach().item()),
                "neutral_sigma0": float(sigma0.detach().item()),
                "sign_violation_rate": float(sign_violation_rate),
                "num_valid_anchors": int(valid.sum().item()),
            }

        return loss, metrics
