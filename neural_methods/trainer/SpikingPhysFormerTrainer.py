try:
    import wandb
except ImportError:
    pass
import sys
import os
import numpy as np
import math
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.loss.PhysFormerLossComputer import TorchLossComputer
from neural_methods.model.SpikingPhysFormer import SpikingPhysFormer
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
from scipy.signal import welch, butter, filtfilt
from spikingjelly.activation_based import functional

class SpikingPhysFormerTrainer(BaseTrainer):
    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.dropout_rate = config.MODEL.DROP_RATE
        self.patch_size = config.MODEL.SPIKINGPHYSFORMER.PATCH_SIZE
        self.dim = config.MODEL.SPIKINGPHYSFORMER.DIM
        self.ff_dim = config.MODEL.SPIKINGPHYSFORMER.FF_DIM
        self.num_heads = config.MODEL.SPIKINGPHYSFORMER.NUM_HEADS
        self.num_layers = config.MODEL.SPIKINGPHYSFORMER.NUM_LAYERS
        self.theta = config.MODEL.SPIKINGPHYSFORMER.THETA
        self.T_s = config.MODEL.SPIKINGPHYSFORMER.T_S
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.frame_rate = config.TRAIN.DATA.FS
        self.frame_rate_valid = config.VALID.DATA.FS
        self.config = config 
        self.min_valid_loss = None
        self.best_epoch = 0

        if config.TOOLBOX_MODE == "train_and_test":
            self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
            self.model = SpikingPhysFormer(
                num_layers=self.num_layers, dim=self.dim, num_heads=self.num_heads, 
                ff_dim=self.ff_dim, theta=self.theta, T_s=self.T_s, frame=self.chunk_len
            ).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            # Load pre-trained PhysFormer patch embedding if specified (Two-Phase Training, arXiv:2402.04798)
            pe_path = getattr(config.MODEL.SPIKINGPHYSFORMER, "PRETRAINED_PE_PATH", "")
            if pe_path and os.path.exists(pe_path):
                print(f"Loading pre-trained PhysFormer Patch Embedding (PE) from: {pe_path}")
                state_dict = torch.load(pe_path, map_location=self.device)
                pe_dict = {}
                for k, v in state_dict.items():
                    clean_k = k.replace('module.', '')
                    if any(prefix in clean_k for prefix in ['Stem0', 'Stem1', 'Stem2', 'patch_embedding']):
                        pe_dict[clean_k] = v
                m = self.model.module if hasattr(self.model, 'module') else self.model
                m.load_state_dict(pe_dict, strict=False)
                print(f"Successfully transferred {len(pe_dict)} pre-trained PE parameters to SpikingPhysFormer!")

            self.num_train_batches = len(data_loader["train"])
            self.criterion_reg = torch.nn.MSELoss()
            self.criterion_L1loss = torch.nn.L1Loss()
            self.criterion_class = torch.nn.CrossEntropyLoss()
            self.criterion_Pearson = Neg_Pearson()
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0.00005)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        elif config.TOOLBOX_MODE == "only_test":
            self.chunk_len = config.TEST.DATA.PREPROCESS.CHUNK_LENGTH
            self.model = SpikingPhysFormer(
                num_layers=self.num_layers, dim=self.dim, num_heads=self.num_heads, 
                ff_dim=self.ff_dim, theta=self.theta, T_s=self.T_s, frame=self.chunk_len
            ).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("SpikingPhysFormer trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        a_start = 1.0
        b_start = 1.0
        exp_b = 1.0

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []

        for epoch in range(self.max_epoch_num):
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

                rPPG = self.model(data)
                # Reset SNN state
                functional.reset_net(self.model)
                
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

                if epoch>10:
                    a = 0.05
                    b = 5.0
                else:
                    a = a_start
                    b = b_start*math.pow(exp_b, epoch/10.0)

                loss = a*loss_rPPG + b*(fre_loss+kl_loss)
                loss.backward()
                self.optimizer.step()
                # Reset SNN state again after backward
                functional.reset_net(self.model)

                loss_rPPG_avg.append(float(loss_rPPG.data))
                loss_peak_avg.append(float(fre_loss.data))
                loss_kl_avg_test.append(float(kl_loss.data))
                loss_hr_mae.append(float(train_mae))
                if idx % 100 == 99:
                    print(f'\nepoch:{epoch}, batch:{idx + 1}, NegPearson:{np.mean(loss_rPPG_avg[-2000:]):.4f}, '
                        f'kl:{np.mean(loss_kl_avg_test[-2000:]):.3f}, fre_CEloss:{np.mean(loss_peak_avg[-2000:]):.3f}, '
                        f'hr_mae:{np.mean(loss_hr_mae[-2000:]):.3f}')
                    
            lrs.append(self.scheduler.get_last_lr())
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
                if self.min_valid_loss is None or (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validating===")
        self.model.eval()
        with torch.no_grad():
            hrs = []
            vbar = tqdm(data_loader["valid"], ncols=80)
            for val_idx, val_batch in enumerate(vbar):
                data, label = val_batch[0].float().to(self.device), val_batch[1].float().to(self.device)
                rPPG = self.model(data)
                functional.reset_net(self.model)
                rPPG = (rPPG - torch.mean(rPPG, axis=-1, keepdim=True)) / torch.std(rPPG, axis=-1, keepdim=True)
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
        self.chunk_len = self.config.TEST.DATA.PREPROCESS.CHUNK_LENGTH
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH, map_location=self.device))
        else:
            epoch = self.max_epoch_num - 1 if self.config.TEST.USE_LAST_EPOCH else self.best_epoch
            model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(epoch) + '.pth')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.eval()
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(self.device), test_batch[1].to(self.device)
                pred_ppg_test = self.model(data)
                functional.reset_net(self.model)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]

        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)

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
