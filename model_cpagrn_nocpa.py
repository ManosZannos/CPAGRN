"""
model_cpagrn_nocpa.py — CPA-GRN v4 WITHOUT CPA features (ablation)

TCPA/DCPA, only geometric features:
    edge(i→j) = [dist, sin(bearing), cos(bearing), dhdg, dhdg.abs()]   (5 dims)
instead of:
    edge(i→j) = [TCPA, DCPA, dist, sin(bearing), cos(bearing), dhdg, dhdg.abs()]  (7 dims)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Edge Features — NO CPA (geometry only)
class CPAFeatures(nn.Module):

    EDGE_DIM = 5

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pos: torch.Tensor,   # [B, N, 2]
        vel: torch.Tensor,   # [B, N, 2]  unused (kept for signature parity)
        hdg: torch.Tensor,   # [B, N]
    ) -> torch.Tensor:
        """Returns [B, N, N, 5]"""
        B, N, _ = pos.shape

        pos_i = pos.unsqueeze(2).expand(B, N, N, 2)
        pos_j = pos.unsqueeze(1).expand(B, N, N, 2)
        hdg_i = hdg.unsqueeze(2).expand(B, N, N)
        hdg_j = hdg.unsqueeze(1).expand(B, N, N)

        r = pos_j - pos_i

        dist    = r.norm(dim=-1)
        bearing = torch.atan2(r[..., 1], r[..., 0])
        dhdg    = hdg_j - hdg_i

        return torch.stack([
            dist,
            torch.sin(bearing),
            torch.cos(bearing),
            dhdg,
            dhdg.abs(),
        ], dim=-1)  # [B, N, N, 5]


# 2. Sparse Neighbor Aggregation (identical logic to v4, edge_dim=5)

class NeighborAggregation(nn.Module):

    def __init__(self, d_model: int, edge_dim: int = 5, top_k: int = 10):
        super().__init__()
        self.top_k = top_k

        self.attn_mlp = nn.Sequential(
            nn.Linear(d_model + edge_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.msg_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, h, edges, mask):
        B, N, D = h.shape

        h_j    = h.unsqueeze(1).expand(B, N, N, D)
        scores = self.attn_mlp(
            torch.cat([h_j, edges], dim=-1)
        ).squeeze(-1)

        if mask is not None:
            mask_j = mask.unsqueeze(1).expand(B, N, N)
            scores = scores.masked_fill(~mask_j, float('-inf'))

        dist = edges[..., 0]  # dist is now index 0 (no TCPA/DCPA prefix)
        if mask is not None:
            dist_masked = dist.masked_fill(~mask_j, float('inf'))
        else:
            dist_masked = dist

        if self.top_k < N:
            k = min(self.top_k, N)
            kth, _ = dist_masked.topk(k, dim=-1, largest=False)
            threshold = kth[..., -1].unsqueeze(-1)
            scores = scores.masked_fill(dist_masked > threshold, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        msgs = self.msg_proj(h)
        agg  = torch.einsum('bij,bjd->bid', weights, msgs)

        return self.norm(self.out_proj(agg))

# 3. Main Model — identical structure to v4, edge_dim=5

class CPAGRN(nn.Module):
    """CPA-GRN v4 architecture WITHOUT CPA features (ablation)."""

    def __init__(
        self,
        feature_size: int   = 4,
        d_model:      int   = 64,
        gru_layers:   int   = 1,
        pred_len:     int   = 5,
        dropout:      float = 0.0,
        top_k:        int   = 10,
    ):
        super().__init__()
        self.d_model  = d_model
        self.pred_len = pred_len
        self.top_k    = top_k

        self.embed = nn.Sequential(
            nn.Linear(feature_size, d_model),
            nn.LayerNorm(d_model),
        )

        self.cpa_features = CPAFeatures()  # actually geometry-only, see above
        self.neighbor_agg = NeighborAggregation(d_model, edge_dim=5, top_k=top_k)

        self.gru = nn.GRU(
            d_model, d_model,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )

        self.final_spatial = NeighborAggregation(d_model, edge_dim=5, top_k=top_k)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, pred_len * 2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs, mask=None, stats=None):
        B, N, T, _ = obs.shape

        x = self.embed(obs)

        fused_steps = []
        for t in range(T):
            pos_t = obs[:, :, t, :2]
            hdg_t = obs[:, :, t, 3]
            vel_t = obs[:, :, t, :2] - obs[:, :, t-1, :2] if t > 0 \
                    else torch.zeros_like(pos_t)

            edges_t = self.cpa_features(pos_t, vel_t, hdg_t)  # [B,N,N,5]
            x_t     = x[:, :, t, :]
            nbr_t   = self.neighbor_agg(x_t, edges_t, mask)
            fused_steps.append(x_t + nbr_t)

        fused_seq = torch.stack(fused_steps, dim=2)

        gru_in = fused_seq.reshape(B * N, T, self.d_model)
        _, h_n = self.gru(gru_in)
        h      = h_n[-1].reshape(B, N, self.d_model)

        if mask is not None:
            h = h * mask.float().unsqueeze(-1)

        pos_last   = obs[:, :, -1, :2]
        vel_last   = obs[:, :, -1, :2] - obs[:, :, -2, :2] if T >= 2 \
                     else torch.zeros_like(pos_last)
        hdg_last   = obs[:, :, -1, 3]
        edges_last = self.cpa_features(pos_last, vel_last, hdg_last)
        h = h + self.final_spatial(h, edges_last, mask)

        out = self.decoder(h).reshape(B, N, self.pred_len, 2)

        if mask is not None:
            out = out * mask.float().unsqueeze(-1).unsqueeze(-1)

        return out


def cpagrn_loss(pred_disp, target_disp, mask):
    sq_err = (pred_disp - target_disp) ** 2
    sq_err = sq_err.sum(dim=-1)
    m      = mask.unsqueeze(-1).expand_as(sq_err)
    return sq_err[m].mean()
