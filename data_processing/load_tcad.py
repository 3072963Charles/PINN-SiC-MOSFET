"""
TCAD Data Loading and Preprocessing Module
===========================================
Handles Silvaco ATLAS CSV output parsing, cleaning, feature extraction,
normalization, and dataset splitting for PINN training.

Inherits data processing architecture from previous work (FYP_IV_TCAD.m)
but extended for full V_GS/V_DS/T parametric coverage.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import warnings


def load_silvaco_csv(filepath: str) -> pd.DataFrame:
    """
    Load a single Silvaco ATLAS CSV output file.

    Silvaco exports CSV with quoted headers like "Gate Voltage", "Drain Current".
    This parser handles automatic column detection.
    """
    df = pd.read_csv(filepath)

    # Rename columns to standardized names
    col_map = {}
    for col in df.columns:
        col_lower = col.strip().lower().replace('"', '')
        if 'gate voltage' in col_lower and 'int' not in col_lower:
            col_map[col] = 'V_GS'
        elif 'drain voltage' in col_lower and 'int' not in col_lower:
            col_map[col] = 'V_DS'
        elif 'drain current' in col_lower:
            col_map[col] = 'I_D'
        elif 'source voltage' in col_lower and 'int' not in col_lower:
            col_map[col] = 'V_S'
        elif 'source current' in col_lower:
            col_map[col] = 'I_S'

    df = df.rename(columns=col_map)

    # Ensure current is positive
    if 'I_D' in df.columns:
        df['I_D'] = df['I_D'].abs()

    return df


def load_multi_vgs_csvs(csv_files: List[str]) -> pd.DataFrame:
    """
    Load multiple Silvaco CSV files (one per V_GS value) and combine into
    a single DataFrame with V_GS, V_DS, I_D columns.

    This mirrors the approach in FYP_IV_TCAD.m lines 4-36.
    """
    all_data = []

    for fpath in csv_files:
        df = load_silvaco_csv(fpath)
        if 'V_GS' in df.columns and 'V_DS' in df.columns and 'I_D' in df.columns:
            all_data.append(df[['V_GS', 'V_DS', 'I_D']])

    if not all_data:
        raise ValueError("No valid CSV files loaded.")

    combined = pd.concat(all_data, ignore_index=True)

    # Sort by V_GS, then V_DS for consistent processing
    combined = combined.sort_values(['V_GS', 'V_DS']).reset_index(drop=True)

    return combined


def generate_synthetic_tcad_data(
    physics_model,
    vgs_range: Tuple[float, float] = (0.0, 20.0),
    vds_range: Tuple[float, float] = (0.0, 1200.0),
    temp_values: List[float] = [298.15, 348.15, 398.15, 448.15],
    n_vgs: int = 15,
    n_vds: int = 80,
    noise_level: float = 0.001,
) -> pd.DataFrame:
    """
    Generate synthetic TCAD-like data using the SPICE Level-3 physics model.

    Used when actual Silvaco data is not yet available — enables PINN
    development and testing before TCAD runs complete.

    Args:
        physics_model: SPICE_Level3_Physics instance with target parameters
        vgs_range: (min, max) V_GS sweep range [V]
        vds_range: (min, max) V_DS sweep range [V]
        temp_values: List of temperature values [K]
        n_vgs, n_vds: Number of sweep points
        noise_level: Gaussian noise std in log-space (0.001 = 0.1%)

    Returns:
        DataFrame with columns: V_GS, V_DS, I_D, T
    """
    import torch

    vgs_vals = np.linspace(vgs_range[0], vgs_range[1], n_vgs)
    vds_vals = np.linspace(vds_range[0], vds_range[1], n_vds)

    records = []

    with torch.no_grad():
        for vgs in vgs_vals:
            for vds in vds_vals:
                for temp in temp_values:
                    result = physics_model(
                        torch.tensor([vgs]),
                        torch.tensor([vds]),
                        torch.tensor([0.0]),  # V_BS = 0
                        torch.tensor([temp]),
                    )
                    i_d = result['I_D'].item()

                    # Add small log-space noise for realism
                    log_id = np.log10(max(i_d, 1e-25))
                    log_id += np.random.normal(0, noise_level)
                    i_d_noisy = 10 ** log_id

                    records.append({
                        'V_GS': vgs,
                        'V_DS': vds,
                        'I_D': i_d_noisy,
                        'T': temp,
                    })

    return pd.DataFrame(records)


def extract_physical_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract physical features from I-V data for PINN parameter initialization.

    Implements the feature extraction strategy from Phase II-2 of the roadmap:
      - V_TH via max-gm method
      - SS (subthreshold swing)
      - R_ON
      - lambda (channel length modulation)

    Returns:
        dict of extracted feature values
    """
    features = {}

    # Group by temperature for temperature-dependent features
    for temp, group in df.groupby('T'):
        temp_key = f"T{int(temp)}"

        # ---- V_TH extraction via max-gm method ----
        # Find the V_GS where dI_D/dV_GS is maximum (at low V_DS)
        low_vds = group[group['V_DS'] < 1.0].copy()
        if len(low_vds) > 5:
            low_vds = low_vds.sort_values('V_GS')
            gm = np.gradient(low_vds['I_D'].values, low_vds['V_GS'].values)
            max_gm_idx = np.argmax(gm)

            if max_gm_idx > 0 and max_gm_idx < len(gm) - 1:
                # Linear extrapolation to I_D=0
                vgs_at_max = low_vds['V_GS'].values[max_gm_idx]
                id_at_max = low_vds['I_D'].values[max_gm_idx]
                gm_max = gm[max_gm_idx]
                vth_extracted = vgs_at_max - id_at_max / (gm_max + 1e-15)
                features[f'{temp_key}_VTH'] = vth_extracted

        # ---- Subthreshold Swing ----
        sub_data = group[(group['V_GS'] < max(1.0, features.get(f'{temp_key}_VTH', 5.0) - 0.5))]
        if len(sub_data) > 5:
            # SS = dV_GS / d(log10 I_D)
            log_id = np.log10(np.clip(sub_data['I_D'].values, 1e-25, None))
            log_id = log_id[np.isfinite(log_id)]
            vgs_sub = sub_data['V_GS'].values[:len(log_id)]
            if len(vgs_sub) > 5:
                try:
                    slope, _ = np.polyfit(vgs_sub, log_id, 1)
                    features[f'{temp_key}_SS'] = 1.0 / max(slope, 0.01)
                except (np.linalg.LinAlgError, ValueError):
                    features[f'{temp_key}_SS'] = 200.0  # Default SiC SS in mV/dec

        # ---- R_ON extraction ----
        high_vgs = group[group['V_GS'] > group['V_GS'].max() * 0.7]
        low_vds_ron = high_vgs[high_vgs['V_DS'] < 0.5]
        if len(low_vds_ron) > 3:
            ron = np.mean(low_vds_ron['V_DS'] / low_vds_ron['I_D'].clip(lower=1e-15))
            features[f'{temp_key}_RON'] = ron

        # ---- Lambda extraction ----
        vgs_mid = group[(group['V_GS'] > group['V_GS'].max() * 0.4) &
                        (group['V_GS'] < group['V_GS'].max() * 0.6)]
        sat_data = vgs_mid[vgs_mid['V_DS'] > 50.0]
        if len(sat_data) > 5:
            try:
                lam = np.polyfit(sat_data['V_DS'].values,
                                sat_data['I_D'].values, 1)[0] / max(sat_data['I_D'].mean(), 1e-15)
                features[f'{temp_key}_LAMBDA'] = lam
            except (np.linalg.LinAlgError, ValueError):
                features[f'{temp_key}_LAMBDA'] = 0.001

    return features


def prepare_pinn_dataset(df: pd.DataFrame,
                         train_ratio: float = 0.70,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15,
                         random_state: int = 42,
                         ) -> Tuple[np.ndarray, ...]:
    """
    Prepare normalized datasets for PINN training.

    Stratified sampling by temperature to ensure each temperature
    is represented in all splits.

    Returns:
        train_data, val_data, test_data as tuples of arrays
        Each: (V_GS, V_DS, V_BS, T, I_D)
        Plus: normalization parameters dict
    """
    np.random.seed(random_state)

    # V_BS is always 0 for power MOSFET (source tied to bulk)
    V_BS = np.zeros(len(df))

    # Split data by temperature for stratification
    train_parts, val_parts, test_parts = [], [], []

    for temp in df['T'].unique():
        temp_mask = df['T'] == temp
        temp_df = df[temp_mask]
        n = len(temp_df)
        indices = np.random.permutation(n)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_parts.append(temp_df.iloc[indices[:train_end]])
        val_parts.append(temp_df.iloc[indices[train_end:val_end]])
        test_parts.append(temp_df.iloc[indices[val_end:]])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=random_state)
    val_df = pd.concat(val_parts).sample(frac=1, random_state=random_state)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=random_state)

    def to_arrays(df):
        return (
            df['V_GS'].values.astype(np.float32),
            df['V_DS'].values.astype(np.float32),
            V_BS[:len(df)].astype(np.float32),
            df['T'].values.astype(np.float32),
            df['I_D'].values.astype(np.float32),
        )

    train_data = to_arrays(train_df)
    val_data = to_arrays(val_df)
    test_data = to_arrays(test_df)

    # Normalization parameters (for reference, not applied in-place)
    norm_params = {
        'V_GS_mean': float(train_data[0].mean()),
        'V_GS_std': float(train_data[0].std()),
        'V_DS_mean': float(train_data[1].mean()),
        'V_DS_std': float(train_data[1].std()),
        'T_mean': float(train_data[3].mean()),
        'T_std': float(train_data[3].std()),
        'log_I_mean': float(np.log10(np.clip(train_data[4], 1e-25, None)).mean()),
        'log_I_std': float(np.log10(np.clip(train_data[4], 1e-25, None)).std()),
    }

    return train_data, val_data, test_data, norm_params


def main():
    """Test the data loading pipeline with existing Silvaco data."""
    import sys

    csv_path = Path(__file__).parent.parent / 'Silvaco_IV01_csv1.csv'

    if csv_path.exists():
        print(f"Loading: {csv_path}")
        df = load_silvaco_csv(str(csv_path))
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  V_GS range: {df['V_GS'].min():.2f} – {df['V_GS'].max():.2f}")
        print(f"  V_DS range: {df['V_DS'].min():.4f} – {df['V_DS'].max():.4f}")
        print(f"  I_D range:  {df['I_D'].min():.4e} – {df['I_D'].max():.4e}")
    else:
        print(f"File not found: {csv_path}")
        print("Generating synthetic data for PINN development...")

        from pinn_model.level3_physics import SPICE_Level3_Physics
        import torch

        phys = SPICE_Level3_Physics()
        df = generate_synthetic_tcad_data(phys)

        print(f"  Generated {len(df)} synthetic data points")
        print(f"  V_GS range: {df['V_GS'].min():.2f} – {df['V_GS'].max():.2f}")
        print(f"  V_DS range: {df['V_DS'].min():.2f} – {df['V_DS'].max():.2f}")
        print(f"  I_D range:  {df['I_D'].min():.4e} – {df['I_D'].max():.4e}")

        # Extract features
        features = extract_physical_features(df)
        print(f"\n  Extracted features: {json.dumps(features, indent=2)}")

        # Prepare datasets
        train, val, test, norm = prepare_pinn_dataset(df)
        print(f"\n  Train: {len(train[0])}, Val: {len(val[0])}, Test: {len(test[0])}")
        print(f"  Norm params: {json.dumps(norm, indent=2)}")


if __name__ == '__main__':
    import json
    main()
