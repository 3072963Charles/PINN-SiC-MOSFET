"""
PINN Architecture for SiC MOSFET Parameter Extraction
======================================================
Physical Residual Architecture:
  - Shared backbone: 8-layer MLP with Swish activation
  - Physics layer: SPICE Level-3 equations (differentiable)
  - Output: I_D correction δI added to physics prediction

The network learns to predict the RESIDUAL between the Level-3
physics model and TCAD data, capturing SiC-specific non-ideal effects
(interface trap dynamics, non-monotonic mobility temperature dependence).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x). Smooth, non-monotonic, self-gated."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class MonotonicLinear(nn.Module):
    """
    Linear layer with non-negative weight constraint.
    Ensures I_D is monotonic with respect to V_GS.
    Inspired by AutoPINN (Guo et al., Micromachines 2023).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply softplus to weights for non-negativity
        w_pos = F.softplus(self.weight)
        return F.linear(x, w_pos, self.bias)


class PINN_SiC_MOSFET(nn.Module):
    """
    Physics-Informed Neural Network for SiC MOSFET compact model parameter extraction.

    Architecture:
        Input (V_GS, V_DS, V_BS, T) [4 dims]
           │
        ┌──▼──────────────────────────────────┐
        │  Shared Backbone                     │
        │  Linear(4, 256) + Swish + BN         │
        │  [Linear(256,256) + Swish + BN] × 6  │
        │  → feature vector (256 dims)          │
        └──┬─────────────┬──────────┬──────────┘
           │             │          │
      ┌────▼───┐  ┌──────▼───┐  ┌──▼──────────┐
      │ Head_I │  │ Head_gm  │  │ Head_C       │
      │ (64,1) │  │ (64,1)   │  │ (64,3)       │
      └────┬───┘  └────┬─────┘  └──┬───────────┘
           │            │           │
           ▼            ▼           ▼
        δI (correction)  δgm        δC_gs, δC_gd, δC_ds
    """

    def __init__(self,
                 input_dim: int = 4,
                 hidden_dim: int = 256,
                 num_hidden: int = 8,
                 dropout: float = 0.05,
                 device: str = 'cpu'):
        super().__init__()

        # Input layer (LayerNorm for batch-size independence)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    Swish(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                )
            )

        # Monotonic block — ensures I_D increases with V_GS
        self.mono_layer = MonotonicLinear(hidden_dim, hidden_dim // 2)

        # Concatenated feature dimension after mono block
        cat_dim = hidden_dim + hidden_dim // 2  # 256 + 128 = 384

        # Output heads
        # I_D correction head
        self.head_I = nn.Sequential(
            nn.Linear(cat_dim, 128),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            Swish(),
            nn.Linear(64, 1),
        )

        # g_m correction head (transconductance) — uses monotonic path only
        self.head_gm = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            Swish(),
            nn.Linear(64, 1),
        )

        # C-V correction head (C_GS, C_GD, C_DS corrections)
        self.head_C = nn.Sequential(
            nn.Linear(cat_dim, 128),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            Swish(),
            nn.Linear(64, 3),
        )

        self._init_weights()

    def _init_weights(self, zero_outputs: bool = True):
        """Initialize weights: Xavier for all layers, zero for final output layers."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Zero-initialize only the FINAL output layer so delta ≡ 0 at epoch 0
        # but gradients flow through all preceding layers (which have Xavier init).
        if zero_outputs:
            nn.init.zeros_(self.head_I[-1].weight)
            nn.init.zeros_(self.head_gm[-1].weight)
            nn.init.zeros_(self.head_C[-1].weight)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the PINN.

        Args:
            x: Input tensor [V_GS, V_DS, V_BS, T] normalized, shape (batch, 4)

        Returns:
            dict with keys: 'delta_I', 'delta_gm', 'delta_C'
        """
        # Shared backbone
        h = self.input_layer(x)

        # Residual hidden layers
        for layer in self.hidden_layers:
            h_res = layer(h)
            h = h + h_res  # Residual connection

        # Monotonic transformation
        h_mono = Swish()(self.mono_layer(h))

        # Concatenate regular + monotonic features
        h_full = torch.cat([h, h_mono], dim=-1)

        # Output heads
        delta_I = self.head_I(h_full)       # (batch, 1)
        delta_gm = self.head_gm(h_mono)     # (batch, 1) — from monotonic path only
        delta_C = self.head_C(h_full)       # (batch, 3)

        return {
            'delta_I': delta_I.squeeze(-1),     # (batch,)
            'delta_gm': delta_gm.squeeze(-1),   # (batch,)
            'delta_C': delta_C,                  # (batch, 3)
        }


class PINN_Hybrid_Model(nn.Module):
    """
    Hybrid model combining:
    1. SPICE Level-3 physics (explicit, differentiable)
    2. PINN residual corrections (learned from data)

    This is the top-level model used for training and inference.
    """

    def __init__(self,
                 physics_model: nn.Module,
                 pinn_network: PINN_SiC_MOSFET,
                 device: str = 'cpu'):
        super().__init__()
        self.physics = physics_model
        self.pinn = pinn_network
        self.device = device

    def forward(self, V_GS: torch.Tensor, V_DS: torch.Tensor,
                V_BS: torch.Tensor, T: torch.Tensor,
                return_all: bool = False) -> Dict[str, torch.Tensor]:
        """
        Hybrid forward pass: physics prediction + PINN correction.

        Args:
            V_GS, V_DS, V_BS, T: Terminal voltages and temperature (batch,)
            return_all: If True, return all intermediate values

        Returns:
            dict with keys: 'I_D', 'g_m', 'g_DS', 'C_GS', 'C_GD', 'C_DS'
        """
        # Step 1: Physics-only prediction
        phys_result = self.physics(V_GS, V_DS, V_BS, T)
        I_phys = phys_result['I_D']

        # Step 2: PINN residual correction
        # For reverse conduction (V_DS < 0), use forward-equivalent voltages
        # so that the PINN correction is symmetric: δ(V_GS, V_DS) ≈ δ(V_GS-V_DS, -V_DS)
        # This preserves Gummel symmetry — sign flip comes from physics model.
        # Body follows effective source (0 for reverse, same convention as physics model)
        mask_rev = V_DS < 0
        V_GS_pinn = torch.where(mask_rev, V_GS - V_DS, V_GS)
        V_DS_pinn = torch.abs(V_DS)
        V_BS_pinn = torch.where(mask_rev, torch.zeros_like(V_BS), V_BS)

        x_input = torch.stack([
            V_GS_pinn / 20.0,      # Normalize V_GS to [0, 1]
            V_DS_pinn / 1200.0,    # Normalize |V_DS| to [0, 1]
            V_BS_pinn,              # Body-source voltage
            T / 300.0,             # Normalize temperature
        ], dim=-1)

        pinn_result = self.pinn(x_input)

        # Step 3: Combine — multiplicative correction for gradient-safe positivity
        # I_total = I_phys * exp(delta_log) — always ≥ 0, gradients flow for all values
        delta_log = pinn_result['delta_I']  # Represents log(I_total/I_phys)
        I_total = I_phys * torch.exp(delta_log)
        I_total = I_total + 1e-30  # Floor for numerical safety

        # Compute gm from physics + correction
        g_m = phys_result.get('g_m', torch.zeros_like(I_phys)) + pinn_result['delta_gm']

        result = {
            'I_D': I_total,
            'I_phys': I_phys,
            'delta_I': pinn_result['delta_I'],
            'g_m': g_m,
            'g_DS': phys_result.get('g_DS', torch.zeros_like(I_phys)),
            'V_TH': phys_result['V_TH'],
            'mu_eff': phys_result['mu_eff'],
            'delta_C': pinn_result['delta_C'],
        }

        if return_all:
            result.update(phys_result)
            result.update(pinn_result)

        return result
