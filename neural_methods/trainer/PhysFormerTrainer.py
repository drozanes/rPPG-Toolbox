try:
    import wandb
except ImportError:
    pass
import sys
"""Trainer for Physformer.

Based on open-source code from the original PhysFormer authors below:
https://github.com/ZitongYu/PhysFormer/blob/main/train_Physformer_160_VIPL.py

We also thank the PhysBench authors for their open-source code based on the code
of the original authors. Their code below provided a better reference for tuning loss
parameters of interest and utilizing RSME as a validation loss:
https://github.com/KegangWangCCNU/PhysBench/blob/main/benchmark_addition/PhysFormer_pure.ipynb

"""

import os
import numpy as np
import math
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.loss.PhysFormerLossComputer import TorchLossComputer
from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
from scipy.signal import welch, butter, filtfilt
import scipy.signal

class PhysFormerTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.dropout_rate = config.MODEL.DROP_RATE
        self.patch_size = config.MODEL.PHYSFORMER.PATCH_SIZE
        self.dim = config.MODEL.PHYSFORMER.DIM
        self.ff_dim = config.MODEL.PHYSFORMER.FF_DIM
        self.num_heads = config.MODEL.PHYSFORMER.NUM_HEADS
        self.num_layers = config.MODEL.PHYSFORMER.NUM_LAYERS
        self.theta = config.MODEL.PHYSFORMER.THETA
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.frame_rate = config.TRAIN.DATA.FS
        self.frame_rate_valid = config.VALID.DATA.FS
        self.config = config 
        self.min_valid_loss = None
        self.best_epoch = 0
        self.start_epoch = 0
        self.patience = getattr(config.TRAIN, 'PATIENCE', 3)
        self.patience_counter = 0

        use_fusion_stem = getattr(config.MODEL, "USE_FUSION_STEM", False) or \
            (hasattr(config.MODEL, "PHYSFORMER") and getattr(config.MODEL.PHYSFORMER, "USE_FUSION_STEM", False))
        alpha = getattr(getattr(config.MODEL, "FUSION_STEM", None), "ALPHA", 0.5)
        beta = getattr(getattr(config.MODEL, "FUSION_STEM", None), "BETA", 0.5)

        if config.TOOLBOX_MODE == "train_and_test":
            self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
            self.model = ViT_ST_ST_Compact3_TDC_gra_sharp(
                image_size=(self.chunk_len,config.TRAIN.DATA.PREPROCESS.RESIZE.H,config.TRAIN.DATA.PREPROCESS.RESIZE.W), 
                patches=(self.patch_size,) * 3, dim=self.dim, ff_dim=self.ff_dim, num_heads=self.num_heads, num_layers=self.num_layers, 
                dropout_rate=self.dropout_rate, theta=self.theta,
                use_fusion_stem=use_fusion_stem, fusion_stem_alpha=alpha, fusion_stem_beta=beta).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            resume_path = getattr(config.MODEL, 'RESUME', '')
            if resume_path and os.path.exists(resume_path):
                print(f"Loading checkpoint to resume: {resume_path}")
                state_dict = torch.load(resume_path, map_location=self.device)
                if hasattr(self.model, 'module') and not any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {'module.' + k: v for k, v in state_dict.items()}
                elif not hasattr(self.model, 'module') and any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict)

                import re
                match = re.search(r'Epoch(\d+)', resume_path)
                if match:
                    resumed_epoch = int(match.group(1))
                    self.start_epoch = resumed_epoch + 1
                    self.best_epoch = resumed_epoch
                    print(f"Resuming PhysFormer starting from Epoch {self.start_epoch} (previous checkpoint was Epoch {self.best_epoch})")

            self.num_train_batches = len(data_loader["train"])
            self.criterion_reg = torch.nn.MSELoss()
            self.criterion_L1loss = torch.nn.L1Loss()
            self.criterion_class = torch.nn.CrossEntropyLoss()
            self.criterion_Pearson = Neg_Pearson()
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0.00005)
            # TODO: In both the PhysFormer repo's training example and other implementations of a PhysFormer trainer, 
            # a step_size that doesn't end up changing the LR always seems to be used. This seems to defeat the point
            # of using StepLR in the first place. Consider investigating and using another approach (e.g., OneCycleLR).
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        elif config.TOOLBOX_MODE == "only_test":
            self.chunk_len = config.TEST.DATA.PREPROCESS.CHUNK_LENGTH
            self.model = ViT_ST_ST_Compact3_TDC_gra_sharp(
                image_size=(self.chunk_len,config.TRAIN.DATA.PREPROCESS.RESIZE.H,config.TRAIN.DATA.PREPROCESS.RESIZE.W), 
                patches=(self.patch_size,) * 3, dim=self.dim, ff_dim=self.ff_dim, num_heads=self.num_heads, num_layers=self.num_layers, 
                dropout_rate=self.dropout_rate, theta=self.theta,
                use_fusion_stem=use_fusion_stem, fusion_stem_alpha=alpha, fusion_stem_beta=beta).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("Physformer trainer initialized in incorrect toolbox mode!")


    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        # a --> Pearson loss (alpha = 0.1 in paper); b --> frequency loss (beta in [1.0, 5.0] exponential ascent)
        a = 0.1
        b_start = 1.0
        exp_b = 5.0

        # TODO: Expand tracking and subsequent plotting of these losses for PhysFormer
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []

        for epoch in range(self.start_epoch, self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            loss_rPPG_avg = []
            loss_peak_avg = []
            loss_kl_avg_test = []
            loss_hr_mae = []

            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                hr = torch.tensor([self.get_hr(i, sr=self.frame_rate) for i in batch[1]]).float().to(self.device)
                data, label = batch[0].float().to(self.device), batch[1].float().to(self.device)

                self.optimizer.zero_grad()

                gra_sharp = 2.0
                rPPG, _, _, _ = self.model(data, gra_sharp)
                rPPG = (rPPG-torch.mean(rPPG, axis=-1).view(-1, 1))/torch.std(rPPG, axis=-1).view(-1, 1)    # normalize
                loss_rPPG = self.criterion_Pearson(rPPG, label)

                fre_loss = 0.0
                kl_loss = 0.0
                train_mae = 0.0
                for bb in range(data.shape[0]):
                    loss_distribution_kl, \
                    fre_loss_temp, \
                    train_mae_temp = TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2(
                        rPPG[bb],
                        hr[bb],
                        self.frame_rate,
                        std=1.0
                    )
                    fre_loss = fre_loss+fre_loss_temp
                    kl_loss = kl_loss+loss_distribution_kl
                    train_mae = train_mae+train_mae_temp
                fre_loss /= data.shape[0]
                kl_loss /= data.shape[0]
                train_mae /= data.shape[0]

                # Exponential increment as in paper Eq. 7: beta = beta_0 * (eta ** (epoch / total_epochs))
                b = b_start * math.pow(exp_b, epoch / float(self.max_epoch_num))

                loss = a * loss_rPPG + b * (fre_loss + kl_loss)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f'\nWarning: NaN/Inf detected in loss at epoch {epoch}, batch {idx + 1}. Skipping step to protect weights.')
                    self.optimizer.zero_grad()
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                n = data.size(0)
                loss_rPPG_avg.append(float(loss_rPPG.data))
                loss_peak_avg.append(float(fre_loss.data))
                loss_kl_avg_test.append(float(kl_loss.data))
                loss_hr_mae.append(float(train_mae))
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(f'\nepoch:{epoch}, batch:{idx + 1}, total:{len(data_loader["train"])}, '
                        f'lr:0.0001, sharp:{gra_sharp:.3f}, a:{a:.3f}, NegPearson:{np.mean(loss_rPPG_avg[-2000:]):.4f}, '
                        f'\nb:{b:.3f}, kl:{np.mean(loss_kl_avg_test[-2000:]):.3f}, fre_CEloss:{np.mean(loss_peak_avg[-2000:]):.3f}, '
                        f'hr_mae:{np.mean(loss_hr_mae[-2000:]):.3f}')
                    
            # Append the current learning rate to the list
            lrs.append(self.scheduler.get_last_lr())
            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(loss_rPPG_avg))
            try:
                if 'wandb' in sys.modules:
                    wandb.log({'Train Loss': mean_training_losses[-1], 'Epoch': epoch})
            except Exception:
                pass
            self.save_model(epoch)
            self.scheduler.step()
            self.model.eval()

            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                try:
                    if 'wandb' in sys.modules:
                        wandb.log({'Validation Loss': mean_valid_losses[-1], 'Epoch': epoch})
                except Exception:
                    pass
                print(f'Validation RMSE:{valid_loss:.3f}, batch:{idx+1}')
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    print(f"Update best model! Best epoch: {self.best_epoch}, min_val_loss (RMSE): {self.min_valid_loss:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"Validation loss did not improve ({valid_loss:.4f} >= {self.min_valid_loss:.4f}). Early stopping patience: {self.patience_counter}/{self.patience}")
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch}! Best epoch was {self.best_epoch} with val loss {self.min_valid_loss:.4f}")
                        break
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validating===")
        self.optimizer.zero_grad()
        with torch.no_grad():
            hrs = []
            vbar = tqdm(data_loader["valid"], ncols=80)
            for val_idx, val_batch in enumerate(vbar):
                data, label = val_batch[0].float().to(self.device), val_batch[1].float().to(self.device)
                gra_sharp = 2.0
                rPPG, _, _, _ = self.model(data, gra_sharp)
                rPPG = (rPPG-torch.mean(rPPG, axis=-1).view(-1, 1))/torch.std(rPPG, axis=-1).view(-1, 1)
                for _1, _2 in zip(rPPG, label):
                    hrs.append((self.get_hr(_1.cpu().detach().numpy(), sr=self.frame_rate_valid), self.get_hr(_2.cpu().detach().numpy(), sr=self.frame_rate_valid)))
            RMSE = np.mean([(i-j)**2 for i, j in hrs])**0.5
        return RMSE

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")

        # Change chunk length to be test chunk length
        self.chunk_len = self.config.TEST.DATA.PREPROCESS.CHUNK_LENGTH

        predictions = dict()
        labels = dict()

        def load_checkpoint(path):
            state_dict = torch.load(path, map_location=self.device)
            if hasattr(self.model, 'module') and not any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
            elif not hasattr(self.model, 'module') and any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            load_checkpoint(self.config.INFERENCE.MODEL_PATH)
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                load_checkpoint(last_epoch_model_path)
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                load_checkpoint(best_model_path)

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                gra_sharp = 2.0
                pred_ppg_test, _, _, _ = self.model(data, gra_sharp)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]

        print('')
        metrics = calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
            self.save_test_outputs(predictions, labels, self.config)

        # Auto-append test results to rPPG_results_database.csv if present
        if metrics and os.path.exists("rPPG_results_database.csv"):
            try:
                import csv
                with open("rPPG_results_database.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.config.TEST.DATA.DATASET,
                        "Supervised",
                        "PhysFormer",
                        f"{metrics.get('MAE', 0.0):.3f}",
                        f"{metrics.get('RMSE', 0.0):.3f}",
                        f"{metrics.get('MAPE', 0.0):.3f}",
                        f"{metrics.get('Pearson', 0.0):.3f}",
                        f"{metrics.get('SNR', 0.0):.3f}",
                        "0.0",
                        "Moderate",
                        "Yes",
                        f"Spatiotemporal Vision Transformer (12 layers, 160 frames, theta={self.theta})."
                    ])
                print("Appended PhysFormer benchmark results to rPPG_results_database.csv!")
                backup_dir = getattr(self.config.MODEL, 'BACKUP_DIR', '') or (
                    '/content/drive/MyDrive/rPPG_Checkpoints' if os.path.exists('/content/drive/MyDrive') else ''
                )
                if backup_dir and os.path.exists(backup_dir):
                    import shutil
                    shutil.copy2("rPPG_results_database.csv", os.path.join(backup_dir, "rPPG_results_database.csv"))
                    print(f"--> [Google Drive / Backup Sync] Mirrored rPPG_results_database.csv to {backup_dir}")
            except Exception as e:
                print(f"Note: Could not auto-append to database CSV: {e}")

        if metrics:
            try:
                if 'wandb' in sys.modules:
                    wandb.log({
                        'Test MAE': metrics.get('MAE', 0.0),
                        'Test RMSE': metrics.get('RMSE', 0.0),
                        'Test MAPE': metrics.get('MAPE', 0.0),
                        'Test Pearson': metrics.get('Pearson', 0.0),
                        'Test SNR': metrics.get('SNR', 0.0)
                    })
            except Exception:
                pass

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

        # Automatic real-time mirroring to Google Drive or custom backup directory
        backup_dir = getattr(self.config.MODEL, 'BACKUP_DIR', '') or (
            '/content/drive/MyDrive/rPPG_Checkpoints' if os.path.exists('/content/drive/MyDrive') else ''
        )
        if backup_dir:
            try:
                import shutil
                os.makedirs(backup_dir, exist_ok=True)
                backup_file = os.path.join(backup_dir, self.model_file_name + f'_Epoch{index}.pth')
                shutil.copy2(model_path, backup_file)
                print(f"--> [Google Drive / Backup Sync] Successfully mirrored checkpoint: {backup_file}")
            except Exception as e:
                print(f"Warning: Could not mirror checkpoint to backup dir {backup_dir}: {e}")

    # HR calculation based on ground truth label / prediction
    def get_hr(self, y, sr=30, min_bpm=40, max_bpm=180):
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        y = np.asarray(y, dtype=np.float64).flatten()
        y = y - np.mean(y)
        low_hz = max(min_bpm / 60.0, 0.6)
        high_hz = min(max_bpm / 60.0, 3.0)
        [b, a] = butter(2, [low_hz / sr * 2, high_hz / sr * 2], btype='bandpass')
        y = filtfilt(b, a, y)
        p, q = welch(y, sr, nfft=1e5/sr, nperseg=len(y))
        mask = (p >= min_bpm / 60.0) & (p <= max_bpm / 60.0)
        if np.sum(mask) == 0:
            return 60.0
        return float(p[mask][np.argmax(q[mask])] * 60.0)
