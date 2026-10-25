try:
    import wandb
except ImportError:
    pass
import sys
"""Trainer for DeepPhys."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class DeepPhysTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.patience = getattr(config.TRAIN, 'PATIENCE', 3)
        self.patience_counter = 0
        
        self.start_epoch = 0
        use_fusion_stem = getattr(config.MODEL, "USE_FUSION_STEM", False) or \
            (hasattr(config.MODEL, "DEEPPHYS") and getattr(config.MODEL.DEEPPHYS, "USE_FUSION_STEM", False))
        self.use_fusion_stem = use_fusion_stem
        alpha = getattr(getattr(config.MODEL, "FUSION_STEM", None), "ALPHA", 0.5)
        beta = getattr(getattr(config.MODEL, "FUSION_STEM", None), "BETA", 0.5)

        if config.TOOLBOX_MODE == "train_and_test":
            self.model = DeepPhys(
                img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H,
                use_fusion_stem=use_fusion_stem,
                fusion_stem_alpha=alpha,
                fusion_stem_beta=beta
            ).to(self.device)
            if config.NUM_OF_GPU_TRAIN > 1 and self.device.type == 'cuda':
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
                    print(f"Resuming training starting from Epoch {self.start_epoch} (previous checkpoint was Epoch {self.best_epoch})")

            self.num_train_batches = len(data_loader["train"]) if (data_loader is not None and data_loader.get("train") is not None) else 0
            self.criterion = torch.nn.MSELoss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.model = DeepPhys(
                img_size=config.TEST.DATA.PREPROCESS.RESIZE.H,
                use_fusion_stem=use_fusion_stem,
                fusion_stem_alpha=alpha,
                fusion_stem_beta=beta
            ).to(self.device)
            if config.NUM_OF_GPU_TRAIN > 1 and self.device.type == 'cuda':
                self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("DeepPhys trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        # Align learning rate scheduler and compute initial baseline validation loss if resuming
        if self.start_epoch > 0:
            print(f"Advancing scheduler for {self.start_epoch} epochs ({self.start_epoch * self.num_train_batches} steps)...")
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(self.start_epoch * self.num_train_batches):
                    self.scheduler.step()
            print("Evaluating initial validation loss of resumed model...")
            self.min_valid_loss = self.valid(data_loader)
            print(f"Resumed baseline validation loss: {self.min_valid_loss:.4f}")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.start_epoch, self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data = batch[0].to(self.device).contiguous()
                labels = batch[1].to(self.device).contiguous()
                N, D, C, H, W = data.shape
                if not self.use_fusion_stem:
                    data = data.view(N * D, C, H, W).contiguous()
                labels = labels.view(-1, 1).contiguous()
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                loss = self.criterion(pred_ppg, labels)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss encountered at epoch {epoch}, batch {idx}. Skipping step.")
                    self.optimizer.zero_grad()
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix({"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]})
                try:
                    if 'wandb' in sys.modules and idx % 20 == 0:
                        wandb.log({'Batch Loss': loss.item(), 'lr': self.optimizer.param_groups[0]["lr"]})
                except Exception:
                    pass

            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))
            try:
                if 'wandb' in sys.modules:
                    wandb.log({'Train Loss': mean_training_losses[-1], 'Epoch': epoch})
            except Exception:
                pass

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                try:
                    if 'wandb' in sys.modules:
                        wandb.log({'Validation Loss': mean_valid_losses[-1], 'Epoch': epoch})
                except Exception:
                    pass
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    print("Update best model! Best epoch: {}, min_val_loss: {:.4f}".format(self.best_epoch, self.min_valid_loss))
                else:
                    self.patience_counter += 1
                    print(f"Validation loss did not improve ({valid_loss:.4f} >= {self.min_valid_loss:.4f}). Early stopping patience: {self.patience_counter}/{self.patience}")
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch}! Best epoch was {self.best_epoch} with val loss {self.min_valid_loss:.4f}")
                        break
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid = valid_batch[0].to(self.device).contiguous()
                labels_valid = valid_batch[1].to(self.device).contiguous()
                N, D, C, H, W = data_valid.shape
                if not self.use_fusion_stem:
                    data_valid = data_valid.view(N * D, C, H, W).contiguous()
                labels_valid = labels_valid.view(-1, 1).contiguous()
                pred_ppg_valid = self.model(data_valid)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        config = self.config
        
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
                data_test = test_batch[0].to(self.config.DEVICE).contiguous()
                labels_test = test_batch[1].to(self.config.DEVICE).contiguous()
                N, D, C, H, W = data_test.shape
                if not self.use_fusion_stem:
                    data_test = data_test.view(N * D, C, H, W).contiguous()
                labels_test = labels_test.view(-1, 1).contiguous()
                pred_ppg_test = self.model(data_test)

                # Always detach and move to CPU to avoid GPU VRAM leaks and ensure numpy compatibility
                labels_test = labels_test.detach().cpu()
                pred_ppg_test = pred_ppg_test.detach().cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
        
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
                        "DeepPhys",
                        f"{metrics.get('MAE', 0.0):.3f}",
                        f"{metrics.get('RMSE', 0.0):.3f}",
                        f"{metrics.get('MAPE', 0.0):.3f}",
                        f"{metrics.get('Pearson', 0.0):.3f}",
                        f"{metrics.get('SNR', 0.0):.3f}",
                        "0.0",
                        "Fast",
                        "Yes",
                        f"Online training with Early Stopping (best epoch {self.best_epoch}, val loss {self.min_valid_loss:.4f})."
                    ])
                print("Appended DeepPhys benchmark results to rPPG_results_database.csv!")
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
        """Inits parameters from args and the writer for TensorboardX."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)

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

 