"""
Evaluate saved checkpoints on the validation dataset to find the best performing epoch.
"""

import os
import re
import glob
import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on validation set")
    parser.add_argument("--config_file", default="configs/train_configs/MMPD_PHYSFORMER_ONLINE.yaml", type=str)
    parser.add_argument("--checkpoint_dir", default="/content/drive/MyDrive/rPPG_Checkpoints", type=str)
    parser.add_argument("--model_prefix", default="MMPD_PhysFormer_Online_Epoch", type=str)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    args, unknown = parser.parse_known_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from config import get_config
    from dataset import data_loader
    from neural_methods.trainer.PhysFormerTrainer import PhysFormerTrainer
    from neural_methods.trainer.DeepPhysTrainer import DeepPhysTrainer

    # Load configuration
    config = get_config(args, unknown)
    config.defrost()
    config.DEVICE = args.device if torch.cuda.is_available() and args.device == 'cuda' else ('mps' if torch.backends.mps.is_available() else 'cpu')
    config.NUM_WORKERS = args.num_workers
    config.TRAIN.BATCH_SIZE = args.batch_size
    config.INFERENCE.BATCH_SIZE = args.batch_size
    config.TEST.USE_LAST_EPOCH = False
    config.freeze()

    print(f"Device: {config.DEVICE}")
    print(f"Loading config from: {args.config_file}")
    print(f"Scanning checkpoint dir: {args.checkpoint_dir} with prefix '{args.model_prefix}'")

    # Find checkpoints
    pattern = os.path.join(args.checkpoint_dir, f"{args.model_prefix}*.pth")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        # Fallback to local MODEL_DIR if not found in backup dir
        alt_pattern = os.path.join(config.MODEL.MODEL_DIR, f"{args.model_prefix}*.pth")
        print(f"No files matching {pattern}. Trying {alt_pattern}...")
        checkpoint_files = glob.glob(alt_pattern)

    if not checkpoint_files:
        print(f"ERROR: No checkpoints found matching {pattern} or {alt_pattern}!")
        return

    # Extract epoch numbers and sort
    checkpoints = []
    for f in checkpoint_files:
        match = re.search(r'Epoch(\d+)\.pth', os.path.basename(f))
        if match:
            epoch_num = int(match.group(1))
            checkpoints.append((epoch_num, f))

    checkpoints.sort(key=lambda x: x[0])
    print(f"Found {len(checkpoints)} checkpoints (Epoch {checkpoints[0][0]} to Epoch {checkpoints[-1][0]}):")

    # Initialize Validation Dataloader
    print("\nInitializing validation dataloader...")
    if config.VALID.DATA.DATASET == "MMPD":
        valid_loader_cls = data_loader.MMPDLoader.MMPDLoader
    elif config.VALID.DATA.DATASET == "PURE":
        valid_loader_cls = data_loader.PURELoader.PURELoader
    elif config.VALID.DATA.DATASET == "UBFC-rPPG":
        valid_loader_cls = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
    else:
        raise ValueError(f"Dataset {config.VALID.DATA.DATASET} not supported in script.")

    valid_data = valid_loader_cls(
        name="valid",
        data_path=config.VALID.DATA.DATA_PATH,
        config_data=config.VALID.DATA,
        device=config.DEVICE
    )
    valid_dataloader = DataLoader(
        dataset=valid_data,
        num_workers=args.num_workers,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False
    )
    data_loader_dict = {"train": [], "valid": valid_dataloader, "test": None}

    # Initialize Trainer
    if config.MODEL.NAME == "PhysFormer":
        trainer = PhysFormerTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "DeepPhys":
        trainer = DeepPhysTrainer(config, data_loader_dict)
    else:
        raise ValueError(f"Model {config.MODEL.NAME} not supported in script.")

    trainer.model.eval()

    # Evaluation loop
    results = []
    min_val = float('inf')
    best_epoch = None
    best_path = None

    print("\n" + "="*60)
    print(f"{'Epoch':<8} | {'Val RMSE (BPM)':<16} | {'Status'}")
    print("="*60)

    for epoch_num, ckpt_path in checkpoints:
        # Load state dict
        state_dict = torch.load(ckpt_path, map_location=config.DEVICE)
        if hasattr(trainer.model, 'module') and not any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif not hasattr(trainer.model, 'module') and any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        trainer.model.load_state_dict(state_dict)

        # Run valid
        val_rmse = trainer.valid(data_loader_dict)
        is_best = val_rmse < min_val
        if is_best:
            min_val = val_rmse
            best_epoch = epoch_num
            best_path = ckpt_path
            status_str = f"⭐ NEW BEST (min: {min_val:.3f})"
        else:
            diff = val_rmse - min_val
            status_str = f"+{diff:.3f} above best"

        print(f"{epoch_num:<8} | {val_rmse:<16.4f} | {status_str}")
        results.append((epoch_num, val_rmse))

    print("="*60)
    print("\n🏆 SUMMARY ANALYSIS:")
    print(f"  Total Epochs Evaluated: {len(results)}")
    print(f"  Best Epoch:             Epoch {best_epoch}")
    print(f"  Best Validation RMSE:   {min_val:.4f} BPM")
    print(f"  Best Checkpoint Path:   {best_path}")

    # Trend check
    recent_epochs = results[-5:]
    print("\n📈 RECENT 5 EPOCHS TREND:")
    for ep, r in recent_epochs:
        print(f"  Epoch {ep:2d}: {r:.4f} BPM")

    last_ep, last_rmse = results[-1]
    if best_epoch == last_ep or best_epoch >= last_ep - 2:
        print("\n💡 RECOMMENDATION: Validation loss was still improving or near minimum in the latest epochs!")
        print(f"   You should resume training from Epoch {last_ep} to complete epochs 27-30.")
        print(f"   Command to resume:")
        print(f"   !python main.py --config_file {args.config_file} --device cuda --num_workers {args.num_workers} --batch_size {args.batch_size} --resume {best_path} --backup_dir {args.checkpoint_dir}")
    else:
        print(f"\n💡 RECOMMENDATION: Model already converged or started overfitting after Epoch {best_epoch}!")
        print(f"   No need to train further. Test directly using the best model (Epoch {best_epoch}):")
        print(f"   Command to test best model:")
        print(f"   !python main.py --config_file {args.config_file} --toolbox_mode only_test --model_path {best_path} --device cuda --num_workers {args.num_workers} --batch_size {args.batch_size} --backup_dir {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
