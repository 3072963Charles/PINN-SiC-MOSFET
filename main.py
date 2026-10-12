#!/usr/bin/env python3
"""
======================================================================
PINN-SiC-MOSFET: Physics-Informed Neural Network for SiC MOSFET
                 SPICE Compact Model Parameter Extraction
======================================================================

Complete pipeline:
  1. Load / generate TCAD data
  2. Build PINN + Level-3 physics hybrid model
  3. Train with dual-stage strategy
  4. Extract physical parameters
  5. Generate SPICE model
  6. Validate accuracy vs previous work

Usage:
  python main.py                          # Full pipeline (synthetic data)
  python main.py --data path/to/tcad.csv  # User-provided Silvaco data
  python main.py --train-only             # Only train, skip SPICE gen
  python main.py --validate-only          # Only validate using saved model

Author: PINN Project — 4H-SiC Power MOSFET Modeling
Date: 2026-06-11
======================================================================
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pinn_model.level3_physics import SPICE_Level3_Physics
from pinn_model.pinn_network import PINN_SiC_MOSFET, PINN_Hybrid_Model
from pinn_model.train import PINN_Trainer
from data_processing.load_tcad import (
    generate_synthetic_tcad_data,
    extract_physical_features,
    prepare_pinn_dataset,
    load_multi_vgs_csvs,
)
from spice_gen.generate_spice import generate_spice_model
from validation.validate import run_full_validation


def parse_args():
    parser = argparse.ArgumentParser(
        description='PINN-SiC-MOSFET: Physics-Informed Parameter Extraction'
    )
    parser.add_argument('--data', type=str, default=None,
                        help='Path to Silvaco CSV data files (glob pattern or single file)')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Output directory for models and results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Training device')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Training batch size')
    parser.add_argument('--epochs-s1', type=int, default=1000,
                        help='Stage 1 (physics warmup) epochs')
    parser.add_argument('--epochs-s2', type=int, default=4000,
                        help='Stage 2 (data fitting) epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--noise', type=float, default=0.001,
                        help='Synthetic data noise level (log-space std)')
    parser.add_argument('--train-only', action='store_true',
                        help='Only train, skip SPICE generation')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate using saved model checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(data, batch_size: int):
    """Convert numpy arrays to PyTorch DataLoader."""
    tensor_data = [torch.tensor(d).float() for d in data]
    dataset = TensorDataset(*tensor_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                     pin_memory=True, drop_last=False)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Device selection
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"\nDevice: {device.upper()}")
    print(f"Output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # Step 1: Data Preparation
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 1: Data Preparation")
    print(f"{'='*60}")

    if args.data:
        # Load user-provided Silvaco TCAD data
        import glob
        csv_files = glob.glob(args.data)
        if not csv_files:
            print(f"Error: No CSV files found matching: {args.data}")
            sys.exit(1)
        print(f"Loading {len(csv_files)} Silvaco CSV files...")
        df = load_multi_vgs_csvs(csv_files)

        # Add temperature column if not present
        if 'T' not in df.columns:
            df['T'] = 300.0
            print("  (No temperature data — assuming T=300K)")
    else:
        # Generate synthetic data for development
        print("No data provided — generating synthetic TCAD data...")
        phys_ref = SPICE_Level3_Physics()
        df = generate_synthetic_tcad_data(
            phys_ref,
            vgs_range=(0.0, 20.0),
            vds_range=(0.0, 1200.0),
            temp_values=[298.15, 348.15, 398.15, 448.15],
            n_vgs=15, n_vds=80,
            noise_level=args.noise,
        )

    print(f"  Data shape:     {df.shape}")
    print(f"  V_GS range:     {df['V_GS'].min():.2f} – {df['V_GS'].max():.2f} V")
    print(f"  V_DS range:     {df['V_DS'].min():.2f} – {df['V_DS'].max():.2f} V")
    print(f"  I_D range:      {df['I_D'].min():.4e} – {df['I_D'].max():.4e} A")
    if 'T' in df.columns:
        print(f"  Temperatures:   {sorted(df['T'].unique())} K")

    # Extract physical features for parameter initialization
    features = extract_physical_features(df)
    print(f"\n  Extracted physical features:")
    for k, v in sorted(features.items()):
        print(f"    {k:20s} = {v:.6e}")

    # Prepare datasets
    train_data, val_data, test_data, norm_params = prepare_pinn_dataset(df)
    print(f"\n  Train set: {len(train_data[0])} samples")
    print(f"  Val set:   {len(val_data[0])} samples")
    print(f"  Test set:  {len(test_data[0])} samples")

    # Build DataLoaders
    train_loader = build_dataloader(train_data, args.batch_size)
    val_loader = build_dataloader(val_data, args.batch_size)
    test_loader = build_dataloader(test_data, args.batch_size)

    # =========================================================================
    # Step 2: Build PINN Hybrid Model
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 2: Building PINN-SiC-MOSFET Hybrid Model")
    print(f"{'='*60}")

    physics_model = SPICE_Level3_Physics()

    # Initialize physical parameters from extracted features if available
    if 'T298_VTH' in features:
        with torch.no_grad():
            physics_model.V_TO.copy_(torch.tensor(features['T298_VTH']))
            print(f"  Initialized V_TO from data: {features['T298_VTH']:.2f} V")

    pinn_network = PINN_SiC_MOSFET()
    hybrid_model = PINN_Hybrid_Model(physics_model, pinn_network, device=device)
    hybrid_model = hybrid_model.to(device)

    total_params = sum(p.numel() for p in hybrid_model.parameters())
    trainable_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Physics parameters:   13 (Level-3 SPICE model)")

    # =========================================================================
    # Step 3: Train PINN
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 3: Training PINN (Dual-Stage)")
    print(f"{'='*60}")

    trainer = PINN_Trainer(
        hybrid_model,
        device=device,
        log_dir=os.path.join(args.output_dir, 'training_logs'),
    )

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs_stage1=args.epochs_s1,
        epochs_stage2=args.epochs_s2,
        lr_adamw=args.lr,
    )

    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump({k: v for k, v in history.items() if isinstance(v, list)},
                  f, indent=2, default=float)

    # =========================================================================
    # Step 4: Validate Accuracy
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 4: Validating Model Accuracy")
    print(f"{'='*60}")

    report = run_full_validation(
        hybrid_model,
        test_data,
        output_dir=os.path.join(args.output_dir, 'validation'),
        compare_with_prev=True,
    )

    # =========================================================================
    # Step 5: Generate SPICE Model
    # =========================================================================
    if not args.train_only:
        print(f"\n{'='*60}")
        print("STEP 5: Generating SPICE Model")
        print(f"{'='*60}")

        params = physics_model.get_params_dict()

        spice_path = os.path.join(args.output_dir, 'SiC_MOSFET_PINN.lib')
        spice_netlist = generate_spice_model(
            params,
            output_path=spice_path,
            include_validation_circuit=True,
        )

        # Also save params as JSON
        params_path = os.path.join(args.output_dir, 'extracted_params.json')
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"Parameters saved: {params_path}")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Output directory:  {args.output_dir}")
    print(f"  Accuracy:          MAPE = {report['overall_metrics']['MAPE']:.3f}%")
    print(f"  Target:            MAPE < 3%")
    print(f"  Previous work:     ~5% (limited range)")
    print(f"  Verdict:           {report['accuracy_verdict']}")

    verdict = report['accuracy_verdict']
    if verdict == 'PASS':
        print(f"\n  TARGET MET: PINN-SPICE model exceeds previous work accuracy.")
    else:
        print(f"\n  FURTHER OPTIMIZATION NEEDED.")

    print(f"\n  Deliverable files:")
    for f in ['SiC_MOSFET_PINN.lib', 'extracted_params.json',
              'training_history.json', 'validation/validation_report.json']:
        fp = os.path.join(args.output_dir, f)
        if os.path.exists(fp):
            print(f"    {fp}")


if __name__ == '__main__':
    main()
