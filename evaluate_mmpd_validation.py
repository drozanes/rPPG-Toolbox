import os
import sys
import glob
import re
import gc
import io
import time
import resource
import torch
import numpy as np
from tqdm import tqdm
from config import get_config
from neural_methods.model.ContrastPhysNet import ContrastPhysNet
from dataset.data_loader.MMPDLoader import MMPDLoader
from evaluation.metrics import calculate_metrics
from neural_methods.loss.ContrastPhysLoss import IrrelevantPowerRatio

class Args:
    config_file = 'configs/train_configs/MMPD_CONTRASTPHYS_ONLINE.yaml'

def get_ram_mb():
    # On macOS Darwin ru_maxrss is in bytes, on Linux in kilobytes
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return raw / (1024 * 1024)
    return raw / 1024

def main():
    print("=" * 80)
    print("  Memory-Safe Fast Evaluation of MMPD Validation Checkpoints (Epoch 0-16)")
    print("=" * 80, flush=True)

    ckpt_dir = "/Users/drozanes/Downloads/exp/MMPD_cf0dc246c7/PreTrainedModels"
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "MMPD_ContrastPhys_Online_Epoch*.pth")),
                        key=lambda x: int(re.search(r'Epoch(\d+)', x).group(1)))

    if not ckpt_files:
        print(f"❌ Error: No checkpoints found in {ckpt_dir}!")
        return

    print(f"🔍 Found {len(ckpt_files)} checkpoints to evaluate.", flush=True)

    # 1. Load configuration
    cfg = get_config(Args())
    cfg.defrost()
    cfg.VALID.DATA.CACHE_SIZE = 1  # Keep max 1 video in cache at a time to prevent memory blowup
    cfg.VALID.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE = False  # mini_MMPD is already cropped (80x60)
    cfg.freeze()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using Compute Device: {device}", flush=True)

    # 2. Pre-load all 17 checkpoint state dictionaries onto CPU (only ~220 MB RAM total)
    print("Loading 17 checkpoint weights into CPU RAM...", flush=True)
    state_dicts = {}
    for f in ckpt_files:
        ep = int(re.search(r'Epoch(\d+)', f).group(1))
        state_dicts[ep] = torch.load(f, map_location='cpu')
    
    print(f"✅ Loaded {len(state_dicts)} model state dicts. Current Peak RAM: {get_ram_mb():.1f} MB", flush=True)

    # 3. Initialize Model and Loss
    model = ContrastPhysNet(in_ch=3, S=cfg.MODEL.CONTRAST_PHYS.S).to(device)
    model.eval()

    ipr_eval = IrrelevantPowerRatio(
        Fs=cfg.VALID.DATA.FS,
        high_pass=cfg.MODEL.CONTRAST_PHYS.HIGH_PASS,
        low_pass=cfg.MODEL.CONTRAST_PHYS.LOW_PASS
    )

    # 4. Initialize MMPD Validation Dataset (Subjects 27, 28, 29)
    print("Initializing MMPD Validation Dataset (Subjects 27, 28, 29)...", flush=True)
    val_dataset = MMPDLoader('MMPD', cfg.VALID.DATA.DATA_PATH, cfg.VALID.DATA)
    total_samples = len(val_dataset)
    print(f"Loaded {total_samples} chunks across 60 validation videos.", flush=True)

    # Group sample indices by video filename to process video-by-video
    video_groups = {}
    for idx, sample in enumerate(val_dataset.samples):
        fn = sample['saved_filename']
        if fn not in video_groups:
            video_groups[fn] = []
        video_groups[fn].append(idx)

    num_videos = len(video_groups)
    print(f"Grouped into {num_videos} videos (11 chunks each).", flush=True)

    # Dictionaries to store outputs: predictions[ep][video_fn][chunk_idx] = 1D tensor
    epochs = sorted(state_dicts.keys())
    predictions_per_epoch = {ep: {} for ep in epochs}
    labels_dict = {}
    iprs_per_epoch = {ep: [] for ep in epochs}

    print("\n🚀 Starting Video-by-Video Inference Loop across all 17 epochs...", flush=True)
    t_start = time.time()

    for v_idx, (video_fn, chunk_indices) in enumerate(tqdm(video_groups.items(), desc="Evaluating Videos")):
        # Load the 11 chunks for this video
        chunk_tensors = []
        chunk_labels = []
        chunk_ids = []
        for c_idx in chunk_indices:
            data, label, fn, c = val_dataset[c_idx]
            chunk_tensors.append(torch.from_numpy(data))
            chunk_labels.append(torch.from_numpy(label))
            chunk_ids.append(int(c))

        # Record labels once
        if video_fn not in labels_dict:
            labels_dict[video_fn] = {}
            for i, c_id in enumerate(chunk_ids):
                labels_dict[video_fn][c_id] = chunk_labels[i].cpu()

        # Run inference across all 17 epochs using mini-batches of size 2
        # (Mini-batch size 2 keeps PyTorch Metal activations under 3.8 GB instead of 20+ GB)
        with torch.no_grad():
            for ep in epochs:
                model.load_state_dict(state_dicts[ep])
                preds_list = []
                for b_start in range(0, len(chunk_tensors), 2):
                    mb_tensor = torch.stack(chunk_tensors[b_start:b_start + 2]).to(device)
                    out = model(mb_tensor)
                    pred_mb = out[0] if isinstance(out, (tuple, list)) else out
                    preds_list.append(pred_mb.cpu())
                    del mb_tensor, out, pred_mb

                pred_ppg = torch.cat(preds_list, dim=0)  # [11, 160] on CPU

                # Compute IPR for this video
                ipr_val = torch.mean(ipr_eval(pred_ppg.to(device))).item()
                iprs_per_epoch[ep].append(ipr_val)

                # Store 1D predictions
                if video_fn not in predictions_per_epoch[ep]:
                    predictions_per_epoch[ep][video_fn] = {}
                for i, c_id in enumerate(chunk_ids):
                    predictions_per_epoch[ep][video_fn][c_id] = pred_ppg[i]

        # Memory cleanup after each video
        del chunk_tensors, chunk_labels
        torch.mps.empty_cache()
        gc.collect()

        if (v_idx + 1) % 5 == 0 or (v_idx + 1) == num_videos:
            elapsed = time.time() - t_start
            print(f" [Progress: {v_idx+1}/{num_videos} videos | Elapsed: {elapsed/60:.1f} min | Peak RAM: {get_ram_mb():.1f} MB]", flush=True)

    total_time = time.time() - t_start
    print(f"\n✅ All video inferences finished in {total_time/60:.2f} minutes!", flush=True)
    print(f"Peak RAM: {get_ram_mb():.1f} MB (strictly bounded).", flush=True)

    # 5. Calculate Metrics for each Epoch
    print("\n" + "=" * 80)
    print(f"{'Epoch':<8} | {'MAE (BPM)':<12} | {'RMSE (BPM)':<12} | {'Pearson (r)':<12} | {'Val IPR':<10} | {'Status'}")
    print("-" * 80)

    best_mae = float('inf')
    best_epoch = None
    best_metrics = None
    results = []

    for ep in epochs:
        mean_ipr = np.mean(iprs_per_epoch[ep])

        # Suppress verbose stdout during metric calculation
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            metrics = calculate_metrics(predictions_per_epoch[ep], labels_dict, cfg)
        finally:
            sys.stdout = old_stdout

        mae = metrics.get('MAE', float('nan'))
        rmse = metrics.get('RMSE', float('nan'))
        pearson = metrics.get('Pearson', float('nan'))

        star = ""
        if mae < best_mae:
            best_mae = mae
            best_epoch = ep
            best_metrics = metrics
            star = "⭐ (Best MAE)"

        status_str = f"Epoch {ep:<2}  | {mae:<12.3f} | {rmse:<12.3f} | {pearson:<12.3f} | {mean_ipr:<10.4f} | {star}"
        print(status_str, flush=True)

        results.append({
            'epoch': ep,
            'mae': mae,
            'rmse': rmse,
            'pearson': pearson,
            'val_ipr': mean_ipr
        })

    print("=" * 80)
    print(f"🏆 BEST VALIDATION CHECKPOINT: Epoch {best_epoch}")
    print(f"   Validation MAE:     {best_metrics.get('MAE', 0.0):.3f} BPM")
    print(f"   Validation RMSE:    {best_metrics.get('RMSE', 0.0):.3f} BPM")
    print(f"   Validation Pearson: {best_metrics.get('Pearson', 0.0):.3f}")
    print(f"   Validation SNR:     {best_metrics.get('SNR', 0.0):.3f} dB")
    print("=" * 80, flush=True)

    # Save results to CSV
    csv_path = "mmpd_validation_results.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,mae,rmse,pearson,val_ipr\n")
        for r in results:
            f.write(f"{r['epoch']},{r['mae']:.4f},{r['rmse']:.4f},{r['pearson']:.4f},{r['val_ipr']:.6f}\n")
    print(f"📁 Results saved to {csv_path}", flush=True)

if __name__ == "__main__":
    main()
