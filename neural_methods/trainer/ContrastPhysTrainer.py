try:
    import wandb
except ImportError:
    pass
import sys
import os
import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.ContrastPhysLoss import ContrastPhysLoss, IrrelevantPowerRatio
from neural_methods.model.ContrastPhysNet import ContrastPhysNet
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm

class ContrastPhysTrainer(BaseTrainer):
    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.config = config
        self.min_valid_loss = None
        self.min_valid_ipr = None
        self.best_epoch = 0
        self.patience = getattr(config.TRAIN, 'PATIENCE', 4)
        self.patience_counter = 0

        frames = config.MODEL.PHYSNET.FRAME_NUM
        contrast_cfg = getattr(config.MODEL, 'CONTRAST_PHYS', None)
        S = getattr(contrast_cfg, 'S', 2) if contrast_cfg else 2
        use_fusion_stem = getattr(config.MODEL, "USE_FUSION_STEM", False) or \
            (hasattr(config.MODEL, "CONTRAST_PHYS") and getattr(config.MODEL.CONTRAST_PHYS, "USE_FUSION_STEM", False))
        alpha = getattr(getattr(config.MODEL, "FUSION_STEM", None), "ALPHA", 0.5)
        beta = getattr(getattr(config.MODEL, "FUSION_STEM", None), "BETA", 0.5)

        self.model = ContrastPhysNet(
            frames=frames, S=S,
            use_fusion_stem=use_fusion_stem,
            fusion_stem_alpha=alpha,
            fusion_stem_beta=beta
        ).to(self.device)
        self.ipr_eval = IrrelevantPowerRatio(
            Fs=config.TRAIN.DATA.FS, high_pass=40, low_pass=250)

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"]) if (data_loader is not None and data_loader.get("train") is not None) else 0
            self.loss_model = ContrastPhysLoss(config)
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=5e-5)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            pass

    def train(self, data_loader):
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print(f"====Training Epoch: {epoch}====")
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data = batch[0].to(torch.float32).to(self.device)
                labels = batch[1].to(torch.float32).to(self.device)
                
                # Model returns rPPG (averaged) and st_block
                rPPG, st_block, _, _ = self.model(data)
                
                contrast_cfg = getattr(self.config.MODEL, 'CONTRAST_PHYS', None)
                weak_sup = getattr(contrast_cfg, 'WEAK_SUPERVISION', False) if contrast_cfg else False
                if weak_sup:
                    loss = self.loss_model(st_block, labels)
                else:
                    loss = self.loss_model(st_block)
                
                loss.backward()
                train_loss.append(loss.item())
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                tbar.set_postfix(loss=loss.item())

            mean_training_losses.append(np.mean(train_loss))
            try:
                if 'wandb' in sys.modules:
                    wandb.log({'Train Loss': mean_training_losses[-1], 'Epoch': epoch})
            except Exception:
                pass
            self.save_model(epoch)
            
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss, valid_ipr = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                try:
                    if 'wandb' in sys.modules:
                        wandb.log({'Validation Loss': valid_loss, 'Validation IPR': valid_ipr, 'Epoch': epoch})
                except Exception:
                    pass

                # Model selection guided by lowest IPR (TPAMI 2024 paper specification)
                contrast_cfg = getattr(self.config.MODEL, 'CONTRAST_PHYS', None)
                selection_metric = getattr(contrast_cfg, 'MODEL_SELECTION', 'IPR') if contrast_cfg else 'IPR'
                
                improved = False
                if selection_metric == 'IPR':
                    if self.min_valid_ipr is None or valid_ipr < self.min_valid_ipr:
                        self.min_valid_ipr = valid_ipr
                        self.min_valid_loss = valid_loss
                        self.best_epoch = epoch
                        self.patience_counter = 0
                        improved = True
                        print(f"Update best model! Best epoch: {self.best_epoch} (IPR: {valid_ipr:.4f}, Loss: {valid_loss:.6f})")
                else:
                    if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                        self.min_valid_loss = valid_loss
                        self.min_valid_ipr = valid_ipr
                        self.best_epoch = epoch
                        self.patience_counter = 0
                        improved = True
                        print(f"Update best model! Best epoch: {self.best_epoch} (Loss: {valid_loss:.6f}, IPR: {valid_ipr:.4f})")
                
                if not improved:
                    self.patience_counter += 1
                    print(f"Validation did not improve. Early stopping patience: {self.patience_counter}/{self.patience}")
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch}! Best epoch was {self.best_epoch}")
                        break
                    
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        if data_loader["valid"] is None:
            return 0.0, 1.0
        self.model.eval()
        valid_loss = []
        valid_iprs = []
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for batch in vbar:
                data = batch[0].to(torch.float32).to(self.device)
                labels = batch[1].to(torch.float32).to(self.device)
                rPPG, st_block, _, _ = self.model(data)
                
                contrast_cfg = getattr(self.config.MODEL, 'CONTRAST_PHYS', None)
                weak_sup = getattr(contrast_cfg, 'WEAK_SUPERVISION', False) if contrast_cfg else False
                if weak_sup:
                    loss = self.loss_model(st_block, labels)
                else:
                    loss = self.loss_model(st_block)
                valid_loss.append(loss.item())

                # Calculate IPR on predicted rPPG
                ipr_val = torch.mean(self.ipr_eval(rPPG)).item()
                valid_iprs.append(ipr_val)

        return np.mean(valid_loss), np.mean(valid_iprs)

    def test(self, data_loader):
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH, map_location=self.device))
        else:
            epoch_to_load = self.max_epoch_num - 1 if self.config.TEST.USE_LAST_EPOCH else self.best_epoch
            model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(epoch_to_load) + '.pth')
            print(f"Loading checkpoint for testing from: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader["test"], ncols=80):
                data = batch[0].to(self.device)
                label = batch[1].to(self.device)
                pred_ppg, _, _, _ = self.model(data)

                for idx in range(data.shape[0]):
                    subj_index = batch[2][idx]
                    sort_index = int(batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg[idx].cpu()
                    labels[subj_index][sort_index] = label[idx].cpu()

        metrics = calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(predictions, labels, self.config)

        # Log test metrics to wandb if active
        try:
            if 'wandb' in sys.modules and wandb.run is not None:
                wandb.log({f"Test/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
        except Exception:
            pass

        # Auto-append test results to rPPG_results_database.csv if present
        if metrics and os.path.exists("rPPG_results_database.csv"):
            try:
                import csv
                model_name = "ContrastPhys+ (FusionStem)" if (hasattr(self.config.MODEL, "CONTRAST_PHYS") and getattr(self.config.MODEL.CONTRAST_PHYS, "USE_FUSION_STEM", False)) else "ContrastPhys+"
                with open("rPPG_results_database.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.config.TEST.DATA.DATASET,
                        "Weakly-Supervised",
                        model_name,
                        f"{metrics.get('MAE', 0.0):.3f}",
                        f"{metrics.get('RMSE', 0.0):.3f}",
                        f"{metrics.get('MAPE', 0.0):.3f}",
                        f"{metrics.get('Pearson', 0.0):.3f}",
                        f"{metrics.get('SNR', 0.0):.3f}",
                        "0.0",
                        "3.1",
                        "Yes",
                        "Contrast-Phys+ TPAMI 2024 setup: 2x2 ST-rPPG pooling, active PSD [40, 250], pairwise MSE loss."
                    ])
                print(f"Appended {model_name} benchmark results to rPPG_results_database.csv!")
            except Exception as e:
                print(f"Error appending to rPPG_results_database.csv: {e}")

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
