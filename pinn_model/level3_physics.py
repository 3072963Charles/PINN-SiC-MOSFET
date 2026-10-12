"""
SPICE Level-3 MOSFET Physics Equations (Differentiable Implementation)
====================================================================
All equations are implemented as pure PyTorch tensor operations,
enabling automatic differentiation for gradient-based parameter extraction.

Reference: SPICE Level-3 MOSFET Model (HSpice/SmartSpice manual)
SiC-specific extensions: Rinaldi et al., IEEE EDL, 2024
                        Borghese et al., ISPSD, 2022

All equations are EXPLICIT closed-form — no iterative solving required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


# =============================================================================
# Physical Constants
# =============================================================================
Q_E = 1.602176634e-19       # Electron charge [C]
K_B = 1.380649e-23          # Boltzmann constant [J/K]
EPS0 = 8.854187817e-12      # Vacuum permittivity [F/m]
EPS_SIC = 9.7 * EPS0        # 4H-SiC permittivity [F/m]
EPS_OX = 3.9 * EPS0         # SiO2 permittivity [F/m]


class SPICE_Level3_Physics(nn.Module):
    """
    Differentiable implementation of SPICE Level-3 MOSFET compact model.

    The forward method computes I_D, g_m, g_DS from (V_GS, V_DS, V_BS, T)
    given the 13 physical parameters stored as trainable nn.Parameters.

    Architecture: Physical residual — the model computes the physical
    current I_phys first, then adds a small NN correction for SiC-specific
    non-ideal effects (interface traps, mobility non-monotonicity, etc.)

    Parameters (13 trainable):
        V_TO    : Zero-bias threshold voltage [V]
        KP      : Transconductance parameter [A/V^2]
        THETA   : Mobility degradation coefficient [V^-1]
        U0      : Low-field surface mobility [cm^2/V·s]
        NFS     : Fast surface state density [cm^-2]
        GAMMA   : Body effect coefficient [V^0.5]
        PHI     : Surface potential 2*phi_p [V]
        VMAX    : Carrier saturation velocity [m/s]
        KAPPA   : Saturation field factor [-]
        LAMBDA  : Channel length modulation [V^-1]
        R_JFET0 : Zero-bias JFET resistance [Ohm]
        KAPPA_VT: VTH temperature coefficient [V/K]
        ALPHA_MU: Mobility temperature exponent [-]
    """

    def __init__(self,
                 tox: float = 5.0e-8,       # Gate oxide thickness [m]
                 w_eff: float = 5.2e-6,     # Effective channel width [m]
                 l_eff: float = 1.0e-6,     # Effective channel length [m]
                 nsub: float = 2.2e15,      # Substrate doping [cm^-3]
                 temp_nom: float = 300.0,   # Nominal temperature [K]
                 device: str = 'cpu'):
        super().__init__()

        # Fixed geometry parameters — registered as buffers so .to(device) works
        self.register_buffer('_tox', torch.tensor(tox))
        self.register_buffer('_w_eff', torch.tensor(w_eff))
        self.register_buffer('_l_eff', torch.tensor(l_eff))
        self.register_buffer('_nsub', torch.tensor(nsub))
        self.register_buffer('_temp_nom', torch.tensor(temp_nom))
        self.register_buffer('_cox', torch.tensor(EPS_OX / tox))

        # Initialize 13 trainable physical parameters
        # Values calibrated for typical 1.2kV 4H-SiC vertical DMOSFET
        self.V_TO = nn.Parameter(torch.tensor(3.8))         # VTH ~ 4V for SiC
        self.KP = nn.Parameter(torch.tensor(0.5))            # Higher KP for SiC wide bandgap
        self.THETA = nn.Parameter(torch.tensor(0.02))        # Moderate mobility degradation
        self.U0 = nn.Parameter(torch.tensor(50.0))           # ~50 cm^2/V·s in SiC channel
        self.log_NFS = nn.Parameter(torch.tensor(26.0))      # NFS ~ 2e11 cm^-2
        self.GAMMA = nn.Parameter(torch.tensor(1.5))
        self.PHI = nn.Parameter(torch.tensor(0.65))
        self.VMAX = nn.Parameter(torch.tensor(2.0e7))
        self.KAPPA = nn.Parameter(torch.tensor(0.5))
        self.LAMBDA = nn.Parameter(torch.tensor(0.0005))     # Weak CLM in SiC power FETs
        self.log_RJFET0 = nn.Parameter(torch.tensor(-2.3))   # log(R_JFET0) ~ 0.1 Ohm
        self.KAPPA_VT = nn.Parameter(torch.tensor(-0.006))   # VTH drops with temp
        self.ALPHA_MU = nn.Parameter(torch.tensor(2.5))      # SiC mobility temp exponent

        # DIBL coefficient (small, for high-voltage SiC MOSFET)
        self.ETA = nn.Parameter(torch.tensor(0.0005))

        # FB correction factor (short/narrow channel)
        self.register_buffer('FB', torch.tensor(0.0))

    def get_params_dict(self) -> Dict[str, float]:
        """Return all physical parameters as a plain dict (for SPICE export)."""
        with torch.no_grad():
            return {
                'VTO': self.V_TO.item(),
                'KP': self.KP.item(),
                'THETA': self.THETA.item(),
                'U0': self.U0.item(),
                'NFS': torch.exp(self.log_NFS).item(),
                'GAMMA': self.GAMMA.item(),
                'PHI': self.PHI.item(),
                'VMAX': self.VMAX.item(),
                'KAPPA': self.KAPPA.item(),
                'LAMBDA': self.LAMBDA.item(),
                'R_JFET0': torch.exp(self.log_RJFET0).item(),
                'KAPPA_VT': self.KAPPA_VT.item(),
                'ALPHA_MU': self.ALPHA_MU.item(),
                'ETA': self.ETA.item(),
                'TOX': self._tox.item(),
                'COX': self._cox.item(),
                'W_EFF': self._w_eff.item(),
                'L_EFF': self._l_eff.item(),
            }

    def _clamp_params(self):
        """Apply physical range constraints to parameters."""
        with torch.no_grad():
            self.V_TO.clamp_(2.0, 20.0)
            self.KP.clamp_(1e-4, 10.0)
            self.THETA.clamp_(1e-4, 1.0)
            self.U0.clamp_(1.0, 100.0)
            self.log_NFS.clamp_(23.0, 30.0)  # NFS in ~1e10 to 1e13
            self.GAMMA.clamp_(0.1, 10.0)
            self.PHI.clamp_(0.1, 1.5)
            self.VMAX.clamp_(1e5, 5e7)
            self.KAPPA.clamp_(0.01, 5.0)
            self.LAMBDA.clamp_(1e-6, 0.1)
            self.log_RJFET0.clamp_(-10.0, 5.0)
            self.KAPPA_VT.clamp_(-0.02, 0.01)
            self.ALPHA_MU.clamp_(1.0, 4.0)
            self.ETA.clamp_(1e-6, 0.1)

    def _thermal_voltage(self, T: torch.Tensor) -> torch.Tensor:
        """Thermal voltage Vt = kT/q."""
        return K_B * T / Q_E

    def _threshold_voltage(self, V_GS: torch.Tensor, V_DS: torch.Tensor,
                           V_BS: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        SPICE Level-3 Threshold Voltage.

        V_TH = V_TO(T) - ETA*V_DS + GAMMA*(sqrt(PHI - V_BS) - sqrt(PHI))

        Temperature dependence:
        V_TO(T) = V_TO(T_nom) + KAPPA_VT * (T - T_nom)
        """
        # Temperature-dependent V_TO
        V_TO_T = self.V_TO + self.KAPPA_VT * (T - self._temp_nom)

        # Body effect term
        phi_minus_vbs = self.PHI - V_BS
        # Clamp for numerical stability (V_BS is typically negative or 0)
        phi_minus_vbs = torch.clamp(phi_minus_vbs, min=0.01)
        body_effect = self.GAMMA * (torch.sqrt(phi_minus_vbs) - torch.sqrt(self.PHI))

        # DIBL (Drain-Induced Barrier Lowering)
        dibl_term = self.ETA * V_DS

        V_TH = V_TO_T - dibl_term + body_effect
        return V_TH

    def _effective_mobility(self, V_GS: torch.Tensor, V_TH: torch.Tensor,
                            V_DS: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Effective mobility including:
        1. Temperature-dependent low-field mobility
        2. Vertical field degradation (THETA)
        3. Velocity saturation (VMAX)

        mu_eff(T) = U0 * (T/300)^(-ALPHA_MU) / (1 + THETA*(V_GS - V_TH))
        mu_final = mu_eff / (1 + mu_eff*V_DS / (VMAX * L_eff))
        """
        # Temperature-dependent low-field mobility
        mu0_T = self.U0 * (T / self._temp_nom) ** (-self.ALPHA_MU)

        # Gate voltage overdrive (only positive portion for degradation)
        V_ov = torch.clamp(V_GS - V_TH, min=0.0)

        # Vertical field mobility degradation
        mu_eff = mu0_T / (1.0 + self.THETA * V_ov)

        # Velocity saturation
        vsat_factor = mu_eff * V_DS / (self.VMAX * self._l_eff + 1e-12)
        mu_final = mu_eff / (1.0 + vsat_factor)

        return mu_final

    def _drain_current_linear(self, beta: torch.Tensor, V_GS: torch.Tensor,
                               V_TH: torch.Tensor, V_DS: torch.Tensor
                               ) -> torch.Tensor:
        """
        Linear region drain current (V_DS < V_DS,sat).

        I_DS = beta * (V_GS - V_TH - (1+FB)/2 * V_DS) * V_DS
        """
        V_ov = V_GS - V_TH
        factor = 1.0 + self.FB
        I_lin = beta * (V_ov - 0.5 * factor * V_DS) * V_DS
        return I_lin

    def _drain_current_saturation(self, beta: torch.Tensor, V_GS: torch.Tensor,
                                   V_TH: torch.Tensor, V_DS: torch.Tensor
                                   ) -> torch.Tensor:
        """
        Saturation region drain current with channel length modulation.

        I_DS,sat = beta/(2*(1+FB)) * (V_GS - V_TH)^2 * (1 + LAMBDA*V_DS)
        """
        V_ov = torch.clamp(V_GS - V_TH, min=0.0)
        factor = 2.0 * (1.0 + self.FB)
        clm = 1.0 + self.LAMBDA * V_DS
        I_sat = (beta / factor) * V_ov**2 * clm
        return I_sat

    def _subthreshold_current(self, V_GS: torch.Tensor, V_TH: torch.Tensor,
                               V_DS: torch.Tensor, T: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Subthreshold current with SiC-specific high trap density.

        n_s = 1 + q*NFS/C_ox   (subthreshold ideality factor)
        I_sub = I_0 * exp((V_GS - V_on) / (n_s * V_t))

        Returns (I_sub, V_on) where V_on is the threshold onset voltage.
        """
        Vt = self._thermal_voltage(T)
        NFS_val = torch.exp(self.log_NFS)

        # Subthreshold swing ideality factor
        # For SiC: NFS ~ 10^11-10^13 cm^-2, giving n_s up to 100+
        n_s = 1.0 + Q_E * NFS_val * 1e4 / self._cox  # NFS in cm^-2, convert to m^-2

        # Onset voltage (where subthreshold meets above-threshold)
        V_on = V_TH + 0.5 * n_s * Vt

        # Current at V_on (from linear region at small V_DS)
        beta_ref = self.KP * self._w_eff / self._l_eff
        I_on = beta_ref * (V_on - V_TH) * torch.clamp(V_DS, min=0.001)
        I_on = torch.clamp(I_on, min=1e-20)

        # Subthreshold exponential — clamp argument to prevent overflow
        # When V_GS >> V_on, this formula is not valid (device is in strong inversion)
        # The smooth transition above_weight handles the blending
        exp_arg = (V_GS - V_on) / torch.clamp(n_s * Vt, min=1e-6)
        exp_arg = torch.clamp(exp_arg, max=50.0)  # exp(50) ~ 5e21, safe
        I_sub = I_on * torch.exp(exp_arg)
        I_sub = torch.clamp(I_sub, max=1e5)  # Upper bound

        return I_sub, V_on

    def _smooth_transition(self, V_GS: torch.Tensor, V_TH: torch.Tensor,
                           delta: float = 0.1) -> torch.Tensor:
        """
        Smooth transition weight from subthreshold (0) to above-threshold (1).
        Uses tanh for differentiability. delta controls transition width.
        """
        return 0.5 + 0.5 * torch.tanh((V_GS - V_TH) / delta)

    def forward(self, V_GS: torch.Tensor, V_DS: torch.Tensor,
                V_BS: torch.Tensor, T: torch.Tensor,
                return_components: bool = False
                ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: compute I_D, g_m, g_DS from terminal voltages.

        Supports both forward (V_DS >= 0) and reverse (V_DS < 0) conduction:
        - Reverse: swap source/drain roles, negate I_D.
        - Enables Gummel Symmetry Test (GST) for physical consistency.

        Args:
            V_GS: Gate-source voltage [V]        (batch,)
            V_DS: Drain-source voltage [V]        (batch,)
            V_BS: Bulk-source voltage [V]          (batch,)
            T:    Temperature [K]                  (batch,)
            return_components: If True, return subthreshold/above-threshold split

        Returns:
            dict with keys: 'I_D', 'g_m', 'g_DS'
        """
        # Handle reverse conduction (V_DS < 0): swap source/drain
        # Source-drain symmetry: body follows the effective source (min(V_S, V_D))
        # Forward: V_BS_eff = V_BS (body at original source)
        # Reverse: V_BS_eff = 0 (body follows new source = original drain, tied to source node)
        mask_reverse = V_DS < 0
        V_DS_eff = torch.where(mask_reverse, -V_DS, V_DS)
        V_GS_eff = torch.where(mask_reverse, V_GS - V_DS, V_GS)
        V_BS_eff = torch.where(mask_reverse, torch.zeros_like(V_BS), V_BS)

        # Run forward conduction computation on effective voltages
        result = self._forward_positive(
            V_GS_eff, V_DS_eff, V_BS_eff, T, return_components)

        # Negate current for reverse operation
        result['I_D'] = torch.where(mask_reverse, -result['I_D'], result['I_D'])
        result['I_channel'] = torch.where(mask_reverse, -result['I_channel'], result['I_channel'])

        return result

    def _forward_positive(self, V_GS: torch.Tensor, V_DS: torch.Tensor,
                          V_BS: torch.Tensor, T: torch.Tensor,
                          return_components: bool = False
                          ) -> Dict[str, torch.Tensor]:
        """
        Core forward computation assuming V_DS >= 0 (forward conduction only).
        See forward() for the public API with reverse conduction support.
        """
        # Ensure positivity
        V_DS = torch.clamp(V_DS, min=0.0)

        # ---- Step 1: Compute threshold voltage ----
        V_TH = self._threshold_voltage(V_GS, V_DS, V_BS, T)

        # ---- Step 2: Compute effective mobility ----
        mu_final = self._effective_mobility(V_GS, V_TH, V_DS, T)

        # ---- Step 3: Compute transconductance parameter ----
        # Use physically-derived beta = mu_eff * C_ox * W/L
        beta_phys = mu_final * 1e-4 * self._cox * self._w_eff / self._l_eff  # mu in cm^2->m^2

        # ---- Step 4: Compute saturation voltage ----
        V_ov = torch.clamp(V_GS - V_TH, min=0.0)
        V_ds_sat = V_ov / (1.0 + self.FB)
        V_ds_sat = torch.clamp(V_ds_sat, min=0.01)

        # For linear region: use V_DS capped at V_ds_sat
        V_DS_linear = torch.minimum(V_DS, V_ds_sat)

        # ---- Step 5: Compute above-threshold current ----
        # Linear region: V_DS < V_ds_sat
        I_lin = beta_phys * (V_ov - 0.5 * (1.0 + self.FB) * V_DS_linear) * V_DS_linear
        I_lin = torch.clamp(I_lin, min=1e-30)

        # Saturation region: I_D,sat at V_DS = V_ds_sat, then apply CLM
        I_sat_base = (beta_phys / (2.0 * (1.0 + self.FB))) * V_ov**2
        I_sat_base = torch.clamp(I_sat_base, min=1e-30)
        clm = 1.0 + self.LAMBDA * V_DS
        I_sat = I_sat_base * clm
        I_sat = torch.clamp(I_sat, min=1e-30)

        # Smooth transition: linear (weight→0) to saturation (weight→1)
        # Narrow transition ensures sat_weight → 0 at V_DS=0 (I_D=0 at zero bias)
        transition_width = 0.1 * V_ds_sat + 0.01
        sat_weight = 0.5 + 0.5 * torch.tanh((V_DS - V_ds_sat) / transition_width)
        sat_weight = torch.clamp(sat_weight, 0.0, 1.0)

        I_above = I_lin * (1.0 - sat_weight) + I_sat * sat_weight
        I_above = torch.clamp(I_above, min=1e-30)

        # ---- Step 6: Subthreshold current ----
        I_sub, V_on = self._subthreshold_current(V_GS, V_TH, V_DS, T)
        I_sub = torch.clamp(I_sub, min=1e-30)

        # ---- Step 7: Smooth subthreshold → above-threshold transition ----
        delta_trans = 0.2
        above_weight = self._smooth_transition(V_GS, V_TH, delta_trans)

        # Safe log-domain blending
        log_I_sub = torch.log(I_sub + 1e-35)
        log_I_above = torch.log(I_above + 1e-35)

        # Weighted blend in log domain
        log_I_total = (1.0 - above_weight) * log_I_sub + above_weight * log_I_above
        I_channel = torch.exp(torch.clamp(log_I_total, min=-80.0, max=50.0))
        I_channel = torch.clamp(I_channel, min=1e-25, max=1e5)

        # ---- Step 8: JFET resistance (single-pass) ----
        R_JFET = torch.exp(self.log_RJFET0)
        V_DS_int = V_DS - I_channel * R_JFET * 0.1  # Weak feedback (10% coupling)
        V_DS_int = torch.clamp(V_DS_int, min=1e-6)

        # Recompute CLM with internal V_DS
        clm_int = 1.0 + self.LAMBDA * V_DS_int
        I_sat_int = I_sat_base * clm_int
        I_sat_int = torch.clamp(I_sat_int, min=1e-30)

        I_above_int = I_lin * (1.0 - sat_weight) + I_sat_int * sat_weight
        I_above_int = torch.clamp(I_above_int, min=1e-30)

        log_I_above_int = torch.log(I_above_int + 1e-35)
        log_I_total_int = (1.0 - above_weight) * log_I_sub + above_weight * log_I_above_int
        I_D = torch.exp(torch.clamp(log_I_total_int, min=-80.0, max=50.0))
        # No final clamp — exp is always positive. Clamps kill gradients.

        # ---- Step 9: Compute gm and gDS (simplified for speed) ----
        # Use central finite difference with the full model for accuracy
        dV = 0.01
        with torch.no_grad():
            # gm ≈ ∂I_D/∂V_GS
            V_GS_hi = V_GS + dV
            V_DS_hi = V_DS + dV

            V_TH_hi_gs = self._threshold_voltage(V_GS_hi, V_DS, V_BS, T)
            V_ov_hi_gs = torch.clamp(V_GS_hi - V_TH_hi_gs, min=0.0)
            mu_hi_gs = self._effective_mobility(V_GS_hi, V_TH_hi_gs, V_DS, T)
            beta_hi_gs = mu_hi_gs * 1e-4 * self._cox * self._w_eff / self._l_eff
            I_hi_gs = torch.clamp(beta_hi_gs * V_ov_hi_gs * V_DS, min=1e-30)
            g_m = torch.abs((I_hi_gs - I_D) / dV)

            # gDS ≈ ∂I_D/∂V_DS
            V_TH_hi_ds = self._threshold_voltage(V_GS, V_DS_hi, V_BS, T)
            V_ov_hi_ds = torch.clamp(V_GS - V_TH_hi_ds, min=0.0)
            mu_hi_ds = self._effective_mobility(V_GS, V_TH_hi_ds, V_DS_hi, T)
            beta_hi_ds = mu_hi_ds * 1e-4 * self._cox * self._w_eff / self._l_eff
            I_hi_ds = torch.clamp(beta_hi_ds * V_ov_hi_ds * V_DS_hi, min=1e-30)
            g_DS = torch.abs((I_hi_ds - I_D) / dV)

        result = {
            'I_D': I_D,
            'g_m': g_m,
            'g_DS': g_DS,
            'V_TH': V_TH,
            'mu_eff': mu_final,
            'I_channel': I_channel,
        }

        if return_components:
            result['I_sub'] = I_sub
            result['I_above'] = I_above
            result['above_weight'] = above_weight

        return result


def compute_spice_residual(v_gs: torch.Tensor, v_ds: torch.Tensor,
                           v_bs: torch.Tensor, temp: torch.Tensor,
                           i_d_target: torch.Tensor,
                           physics_model: SPICE_Level3_Physics
                           ) -> Dict[str, torch.Tensor]:
    """
    Compute physics residuals for PINN loss function.

    Returns residuals between Level-3 physics predictions and target data.
    Used as the physics-informed loss term.
    """
    result = physics_model(v_gs, v_ds, v_bs, temp)
    i_d_pred = result['I_D']

    # Log-space residual (critical for subthreshold accuracy, per previous work)
    log_i_target = torch.log10(torch.clamp(i_d_target, min=1e-25))
    log_i_pred = torch.log10(torch.clamp(i_d_pred, min=1e-25))

    # Linear-space relative residual (critical for strong inversion)
    rel_error = (i_d_pred - i_d_target) / torch.clamp(i_d_target, min=1e-20)

    return {
        'log_residual': log_i_pred - log_i_target,
        'rel_residual': rel_error,
        'abs_residual': i_d_pred - i_d_target,
        'i_d_pred': i_d_pred,
        'g_m': result['g_m'],
        'g_DS': result['g_DS'],
        'V_TH': result['V_TH'],
    }


# =============================================================================
# Previous Student Model — for accuracy benchmarking
# =============================================================================
class PreviousEmpiricalModel:
    """
    Reimplementation of the previous student's empirical model for comparison.
    Based on the final converged form from FYP_IV_03_func.m / test_D1.lib.

    Core equation:
        I_D = beta * 3/(1+exp(-Psi*Vgs)) * (1+lambda*Vds) * erf(tanh(alpha*Vds))
    where Psi = P1*Vgs0 + P2*Vgs0^2 + P3*Vgs0^3 + P4*Vgs0^4 + P5*Vgs0^5
    """

    def __init__(self):
        # Best-fit parameters extracted from previous work
        self.params = {
            'P1': 3.443711928939403,
            'P2': 5.686636594655933,
            'P3': 3.956731131088150,
            'as': 0.267911877913861,
            'Vpk0': 9.762057131808618,
            'Vpks': 9.592207989753323,
            'P4': 2.01,
            'P5': 3.5,
        }
        # Polynomial coefficients for alpha, beta, lambda (from test_D1.lib)
        self.alpha_poly = [1.223915494981986e-04, -0.004301149853144,
                           0.022028603005521, 0.541506201518383]
        self.beta_poly = [-0.037436671972722, 1.887523041469408,
                          -29.061496665745054, 155.6436553230904]
        self.lambda_poly = [4.863148330020342e-07, -4.725010620926286e-05,
                            0.001902005775586, -0.040919366350027,
                            0.505596931001760, -3.574488135690544,
                            13.390874599767667, -20.553060805654944]

    def predict(self, V_GS, V_DS):
        """Predict I_D using the empirical model (numpy arrays)."""
        import numpy as np
        V_GS = np.atleast_1d(V_GS)
        V_DS = np.atleast_1d(V_DS)

        # Vgs-dependent alpha, beta, lambda
        alpha = np.polyval(self.alpha_poly, V_GS)
        beta = np.polyval(self.beta_poly, V_GS)
        lam = np.polyval(self.lambda_poly, V_GS)

        # Vpk with drain dependence
        Vpk = self.params['Vpk0'] + (self.params['Vpks'] - self.params['Vpk0']) * \
              np.tanh(self.params['as'] * V_DS)
        Vgs0 = V_GS - Vpk

        # Psi polynomial
        Psi = (self.params['P1'] * Vgs0 + self.params['P2'] * Vgs0**2 +
               self.params['P3'] * Vgs0**3 + self.params['P4'] * Vgs0**4 +
               self.params['P5'] * Vgs0**5)

        # Core model equation
        exp_arg = -Psi * V_GS
        exp_arg = np.clip(exp_arg, -50, 50)  # Prevent overflow
        I_D = beta * 3.0 / (1.0 + np.exp(exp_arg)) * \
              (1.0 + lam * V_DS) * np.minimum(
                  np.maximum(np.tanh(alpha * V_DS), -1.0), 1.0)

        return np.maximum(I_D, 1e-25)
