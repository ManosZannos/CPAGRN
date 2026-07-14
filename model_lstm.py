"""
model_lstm.py — Vanilla LSTM baseline for vessel trajectory prediction.

Architecture (Sekhon & Fleming 2020, Table 2 'Vanilla LSTM'):
  - Per-vessel LSTM encoder (no social interaction)
  - Linear decoder → predicted displacement from last observed position
  - Deterministic output (single prediction, not probabilistic)

Input:  obs  [B, N, obs_len, 4]   (LON, LAT, SOG, Heading — z-score)
Output: pred [B, N, pred_len, 2]  (LON, LAT displacement — z-score)

"""

import torch
import torch.nn as nn


class VanillaLSTM(nn.Module):

    def __init__(
        self,
        feature_size: int = 4,       # LON, LAT, SOG, Heading
        hidden_size:  int = 64,
        num_layers:   int = 1,
        pred_len:     int = 5,
        dropout:      float = 0.0,
    ):
        super().__init__()
        self.pred_len    = pred_len
        self.hidden_size = hidden_size

        # Input embedding
        self.input_proj = nn.Linear(feature_size, hidden_size)

        # LSTM encoder (per vessel, over obs_len steps)
        self.lstm = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # Decoder: hidden state → pred_len × 2 displacements
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, pred_len * 2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        obs:  torch.Tensor,            # [B, N, obs_len, 4]
        mask: torch.Tensor | None = None,  # [B, N] bool (True = real vessel)
    ) -> torch.Tensor:
    
        #Returns: pred_disp: [B, N, pred_len, 2]  predicted displacement (z-score)
        
        B, N, T, F = obs.shape

        # Reshape to process all vessels in parallel: [B*N, T, F]
        x = obs.reshape(B * N, T, F)

        # Embed
        x = self.input_proj(x)         # [B*N, T, hidden]

        # LSTM
        _, (h_n, _) = self.lstm(x)    # h_n: [num_layers, B*N, hidden]
        h = h_n[-1]                    # [B*N, hidden]  (last layer)

        # Decode
        out = self.decoder(h)          # [B*N, pred_len*2]
        out = out.reshape(B, N, self.pred_len, 2)

        # Zero out padded vessels
        if mask is not None:
            out = out * mask.float().unsqueeze(-1).unsqueeze(-1)

        return out  # [B, N, pred_len, 2]
# Loss
def lstm_loss(
    pred_disp:   torch.Tensor,   # [B, N, pred_len, 2]  predicted displacement
    target_disp: torch.Tensor,   # [B, N, pred_len, 2]  true displacement
    mask:        torch.Tensor,   # [B, N] bool
) -> torch.Tensor:
    """MSE loss on displacement, averaged over valid (non-padded) vessels."""
    # [B, N, pred_len, 2]
    sq_err = (pred_disp - target_disp) ** 2  # [B, N, pred_len, 2]
    sq_err = sq_err.sum(dim=-1)              # [B, N, pred_len]  (sum over x,y)

    # Mask padded vessels
    m = mask.unsqueeze(-1).expand_as(sq_err)  # [B, N, pred_len]
    loss = sq_err[m].mean()

    return loss
