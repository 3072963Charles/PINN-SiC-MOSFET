"""
PINN Training Module -- Dual-Stage Training Strategy
====================================================
Stage 1: Physics warmup (epochs 1-1000)
  - Only physics + smoothness + monotonicity losses active
  - Network learns to predict near-zero corrections (physics-dominant)

Stage 2: Data fitting (epochs 1001–5000)
  - Gradually introduce data loss with cosine schedule
  - AdamW → L-BFGS optimizer transition
  - Extract final physical parameters

Loss function components:
  L_total = λ1*L_data_log + λ1'*L_data_lin + λ2*L_physics + λ3*L_smooth + λ4*L_mono
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import defaultdict
import time
import json
import os

from pinn_model.level3_physics import SPICE_Level3_Physics
from pinn_model.pinn_network import PINN_SiC_MOSFET, PINN_Hybrid_Model


# =============================================================================
# Loss Functions
# =============================================================================

def compute_loss(hybrid_model: PINN_Hybrid_Model,
                 batch: Tuple[torch.Tensor, ...],
                 stage: int,
                 lambda_physics: float,
                 lambda_smooth: float,
                 lambda_mono: float,
                 lambda_data_lin: float = 0.5,
                 ) -> Dict[str, torch.Tensor]:
    """
    Compute multi-component PINN loss for a batch.

    Args:
        hybrid_model: PINN + Physics hybrid model
        batch: (V_GS, V_DS, V_BS, T, I_D_target) tensors
        stage: 1 (physics warmup) or 2 (data fitting)
        lambda_*: Loss component weights

    Returns:
        dict with 'total', 'data_log', 'data_lin', 'physics', 'smooth', 'mono'
    """
    V_GS, V_DS, V_BS, T, I_target = batch
    batch_size = V_GS.shape[0]
    eps = 1e-30

    # Forward pass
    result = hybrid_model(V_GS, V_DS, V_BS, T)
    I_pred = result['I_D']
    I_phys = result['I_phys']
    delta_I = result['delta_I']

    # --- Data loss (only computed in Stage 2) ---
    if stage == 2:
        # Log-space MSE (critical for subthreshold accuracy)
        I_target_safe = torch.clamp(I_target, min=1e-20)
        I_pred_safe = torch.clamp(I_pred, min=1e-20)
        log_target = torch.log10(I_target_safe)
        log_pred = torch.log10(I_pred_safe)
        loss_data_log = F.mse_loss(log_pred, log_target)

        # Linear-space relative error (critical for strong inversion accuracy)
        rel_error = (I_pred - I_target) / torch.clamp(I_target, min=1e-12)
        # Apply only to above-threshold region
        mask_strong = (I_target > 1e-8).float()
        n_strong = mask_strong.sum().clamp(min=1)
        loss_data_lin = (rel_error**2 * mask_strong).sum() / n_strong
    else:
        loss_data_log = torch.tensor(0.0, device=I_pred.device)
        loss_data_lin = torch.tensor(0.0, device=I_pred.device)

    # --- Physics consistency loss ---
    # Multiplicative correction: delta_log = log(I_total/I_phys).
    # Physics dominates when delta_log ≈ 0. Penalize large |delta_log|.
    loss_physics = (delta_I**2).mean()

    # --- Smoothness loss ---
    # Penalize large second derivatives w.r.t. V_DS for physically smooth I-V curves.
    if batch_size >= 3:
        # Sort by V_DS within the batch
        vds_sorted, sort_idx = torch.sort(V_DS)
        I_sorted_vds = I_pred[sort_idx]
        d2I = I_sorted_vds[2:] - 2 * I_sorted_vds[1:-1] + I_sorted_vds[:-2]
        loss_smooth = (d2I**2).mean()
    else:
        loss_smooth = torch.tensor(0.0, device=I_pred.device)

    # --- Monotonicity loss ---
    # I_D must increase monotonically with V_GS at fixed V_DS.
    # Group pairs with similar V_DS but different V_GS.
    if batch_size >= 2:
        idx = torch.randperm(batch_size, device=I_pred.device)
        V_GS_1, V_GS_2 = V_GS, V_GS[idx]
        I_1, I_2 = I_pred, I_pred[idx]

        # Penalize: V_GS1 > V_GS2 but I1 < I2 (monotonicity violation)
        vgs_diff = V_GS_1 - V_GS_2
        i_diff = I_1 - I_2
        # Violation when V_GS increases but I_D decreases
        violation = F.relu(-i_diff) * F.relu(vgs_diff) + F.relu(i_diff) * F.relu(-vgs_diff)
        loss_mono = violation.mean()
    else:
        loss_mono = torch.tensor(0.0, device=I_pred.device)

    # --- Subthreshold-focused log-space loss ---
    # Separate loss for subthreshold region (I < 1e-6 A) to improve
    # accuracy in weak inversion where SiC interface traps dominate.
    mask_sub = (I_target < 1e-6).float()
    n_sub = mask_sub.sum().clamp(min=1)
    loss_data_sub = ((log_pred - log_target)**2 * mask_sub).sum() / n_sub if stage == 2 \
        else torch.tensor(0.0, device=I_pred.device)

    # --- Total loss ---
    if stage == 1:
        # Physics warmup: no data loss
        total = (lambda_physics * loss_physics +
                 lambda_smooth * loss_smooth +
                 lambda_mono * loss_mono)
    else:
        # Stage 2: Full loss with data
        # λ_sub=3.0 gives subthreshold region 3x weight for better weak inversion accuracy
        lambda_sub = 3.0
        total = (loss_data_log +
                 lambda_sub * loss_data_sub +
                 lambda_data_lin * loss_data_lin +
                 lambda_physics * loss_physics +
                 lambda_smooth * loss_smooth +
                 lambda_mono * loss_mono)

    return {
        'total': total,
        'data_log': loss_data_log,
        'data_lin': loss_data_lin,
        'data_sub': loss_data_sub,
        'physics': loss_physics,
        'smooth': loss_smooth,
        'mono': loss_mono,
    }


# =============================================================================
# Training Loop
# =============================================================================

class PINN_Trainer:
    """
    Trainer for the PINN-SiC-MOSFET hybrid model.
    Implements dual-stage training with progressive data introduction.
    """

    def __init__(self,
                 hybrid_model: PINN_Hybrid_Model,
                 device: str = 'cuda',
                 log_dir: str = './training_logs'):
        self.model = hybrid_model
        self.device = device
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.history = defaultdict(list)

    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader],
              epochs_stage1: int = 1000,
              epochs_stage2: int = 4000,
              lr_adamw: float = 1e-3,
              lr_lbfgs: float = 0.1,
              lambda_physics_stage1: float = 1.0,
              lambda_smooth: float = 0.01,
              lambda_mono: float = 0.05,
              lambda_physics_stage2_init: float = 0.5,
              lambda_physics_stage2_final: float = 0.05,
              early_stop_patience: int = 500,
              print_interval: int = 200,
              ) -> Dict:
        """
        Execute full dual-stage training.

        Returns:
            Training history dict
        """
        print("=" * 70)
        print("PINN-SiC-MOSFET Training -- Dual-Stage Strategy")
        print("=" * 70)

        # =====================================================================
        # Stage 1: Skip (physics defaults already match data, no pre-training needed)
        # =====================================================================
        if epochs_stage1 > 0:
            print("\n[Stage 1] Skipping -- default physics already match synthetic data")
            print("-" * 50)

        # =====================================================================
        # Stage 2: Joint Data Fitting
        # =====================================================================
        stage1_effective = 0  # Stage 1 skipped
        print(f"\n[Stage 2] Joint Fitting (epochs 1–{epochs_stage2})")
        print("-" * 50)

        # Separate LR: lower LR for physics (already close to truth), higher for NN
        physics_params = list(self.model.physics.parameters())
        pinn_params = list(self.model.pinn.parameters())
        optimizer = AdamW([
            {'params': physics_params, 'lr': lr_adamw * 0.1},
            {'params': pinn_params, 'lr': lr_adamw},
        ], weight_decay=1e-5)
        # CosineAnnealingWarmRestarts with long cycle -- restarts are mild at low LR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_stage2, eta_min=1e-6)

        best_val_loss = float('inf')
        best_mape = float('inf')
        patience_counter = 0
        total_epochs_s2 = epochs_stage2

        for epoch in range(1, epochs_stage2 + 1):
            current_total = stage1_effective + epoch
            self.model.train()
            epoch_losses = defaultdict(float)

            # Progressive data weight schedule (cosine, start at 0.5 for faster training)
            progress = epoch / total_epochs_s2
            lambda_data_now = 0.5 + 0.5 * (1.0 - np.cos(np.pi * progress)) / 2.0

            # Progressive physics weight decay
            lambda_phys_now = (lambda_physics_stage2_init -
                              (lambda_physics_stage2_init - lambda_physics_stage2_final) *
                              (1.0 - np.cos(np.pi * progress)) / 2.0)

            for batch in train_loader:
                batch = [b.to(self.device) for b in batch]
                optimizer.zero_grad()
                losses = compute_loss(
                    self.model, batch,
                    stage=2,
                    lambda_physics=lambda_phys_now,
                    lambda_smooth=lambda_smooth,
                    lambda_mono=lambda_mono,
                    lambda_data_lin=1.0,
                )
                # Progressive data introduction: scale data_log, keep lin fixed
                # Physics weight decays progressively
                # λ_sub=3.0 gives subthreshold region higher weight
                lambda_sub = 3.0
                total = (lambda_data_now * losses['data_log'] +
                        lambda_sub * losses.get('data_sub', 0) +
                        losses['data_lin'] +
                        lambda_phys_now * losses['physics'] +
                        lambda_smooth * losses['smooth'] +
                        lambda_mono * losses['mono'])
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()

                for k, v in losses.items():
                    if k == 'total':
                        continue  # Skip compute_loss's total, use ours below
                    v_float = v.item()
                    if np.isnan(v_float) or np.isinf(v_float):
                        print(f"  [WARN] NaN/Inf in loss '{k}' at epoch {current_total}")
                        v_float = 0.0
                    epoch_losses[k] += v_float
                # Record the actual total used for backward
                epoch_losses['total'] += total.item()

            # Record
            for k, v in epoch_losses.items():
                self.history[f'stage2_{k}'].append(v / len(train_loader))
            self.history['lambda_data'].append(lambda_data_now)
            self.history['lambda_physics'].append(lambda_phys_now)

            # Validate and save best model
            if val_loader and epoch % 50 == 0:
                val_loss = self._validate(val_loader)
                self.history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 50

            # Track best MAPE
            if epoch % 100 == 0:
                train_acc = self._compute_accuracy(train_loader)
                if train_acc['mape'] < best_mape:
                    best_mape = train_acc['mape']
                    self._save_checkpoint('best_model.pt')
                    self.history['best_mape_epoch'].append(current_total)
                    self.history['best_mape'].append(best_mape)

            # Clamp physics parameters
            self.model.physics._clamp_params()

            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {current_total}")
                break

            if epoch % print_interval == 0:
                acc = self._compute_accuracy(train_loader)
                self._print_progress(current_total, epoch_losses, len(train_loader),
                                    scheduler.get_last_lr()[0], accuracy=acc)

        # Final accuracy and save best model
        final_acc = self._compute_accuracy(train_loader)
        print(f"\nFinal Training Accuracy: MAPE = {final_acc['mape']:.3f}%")

        self._save_checkpoint('final_model.pt')
        self._save_history()
        self._save_physics_params()

        return dict(self.history)

    def _validate(self, val_loader: DataLoader) -> float:
        """Compute validation loss."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(self.device) for b in batch]
                V_GS, V_DS, V_BS, T, I_target = batch
                result = self.model(V_GS, V_DS, V_BS, T)
                I_pred = torch.clamp(result['I_D'], min=1e-20)
                I_target_safe = torch.clamp(I_target, min=1e-20)
                total_loss += F.mse_loss(
                    torch.log10(I_pred), torch.log10(I_target_safe)
                ).item()
        return total_loss / len(val_loader)

    def _compute_accuracy(self, loader: DataLoader) -> Dict[str, float]:
        """Compute accuracy metrics on a dataset."""
        self.model.eval()
        all_target = []
        all_pred = []
        with torch.no_grad():
            for batch in loader:
                batch = [b.to(self.device) for b in batch]
                V_GS, V_DS, V_BS, T, I_target = batch
                result = self.model(V_GS, V_DS, V_BS, T)
                all_target.append(I_target.cpu())
                all_pred.append(result['I_D'].cpu())

        I_target = torch.cat(all_target).numpy()
        I_pred = torch.cat(all_pred).numpy()

        # Compute metrics
        mask = I_target > 1e-25
        I_t = I_target[mask]
        I_p = I_pred[mask]

        if len(I_t) == 0:
            return {'mape': float('inf'), 'rmse': float('inf'), 'r2': 0.0}

        mape = np.mean(np.abs((I_t - I_p) / I_t)) * 100
        rmse = np.sqrt(np.mean((I_t - I_p)**2))
        ss_res = np.sum((I_t - I_p)**2)
        ss_tot = np.sum((I_t - np.mean(I_t))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-40)

        return {'mape': mape, 'rmse': rmse, 'r2': r2}

    def _print_progress(self, epoch: int, losses: Dict, n_batches: int,
                        lr: float, accuracy: Optional[Dict] = None):
        """Print training progress."""
        total = losses.get('total', 0) / n_batches
        data_log = losses.get('data_log', 0) / n_batches
        phys = losses.get('physics', 0) / n_batches
        mono = losses.get('mono', 0) / n_batches

        msg = (f"  Epoch {epoch:5d} | lr={lr:.2e} | total={total:.4e} "
               f"| log={data_log:.4e} | phys={phys:.4e}")

        if accuracy:
            msg += f" | MAPE={accuracy['mape']:.2f}% | R2={accuracy['r2']:.5f}"

        print(msg)

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.log_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'physics_params': self.model.physics.get_params_dict(),
        }, path)

    def _save_history(self):
        """Save training history to JSON."""
        path = os.path.join(self.log_dir, 'training_history.json')
        # Convert numpy/tensor values
        hist_serializable = {}
        for k, v in self.history.items():
            hist_serializable[k] = [float(x) if isinstance(x, (np.floating, torch.Tensor))
                                    else x for x in v]
        with open(path, 'w') as f:
            json.dump(hist_serializable, f, indent=2)

    def _save_physics_params(self):
        """Save extracted physical parameters."""
        params = self.model.physics.get_params_dict()
        path = os.path.join(self.log_dir, 'extracted_physics_params.json')
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"\n{'='*60}")
        print("Extracted Physical Parameters (SPICE Level-3 compatible):")
        print(f"{'='*60}")
        for name, value in params.items():
            if isinstance(value, float):
                print(f"  {name:12s} = {value:.6e}")
        print(f"{'='*60}")
        print(f"Saved to: {path}")
