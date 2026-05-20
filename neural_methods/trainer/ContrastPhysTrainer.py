import os
import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.ContrastPhysLoss import ContrastPhysLoss
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
        self.best_epoch = 0

        self.model = ContrastPhysNet(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.loss_model = ContrastPhysLoss(config)
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR)
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
                
                # In Contrast-Phys+, we use st_block for loss
                # If unsupervised, pass labels=None. If weakly supervised, pass labels.
                if self.config.MODEL.CONTRAST_PHYS.WEAK_SUPERVISION:
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
            self.save_model(epoch)
            
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print(f"Update best model! Best epoch: {self.best_epoch}")
                    
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        if data_loader["valid"] is None:
            return 0.0
        self.model.eval()
        valid_loss = []
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for batch in vbar:
                data = batch[0].to(torch.float32).to(self.device)
                labels = batch[1].to(torch.float32).to(self.device)
                rPPG, st_block, _, _ = self.model(data)
                
                if self.config.MODEL.CONTRAST_PHYS.WEAK_SUPERVISION:
                    loss = self.loss_model(st_block, labels)
                else:
                    loss = self.loss_model(st_block)
                valid_loss.append(loss.item())
        return np.mean(valid_loss)

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

        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
