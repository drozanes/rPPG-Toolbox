import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import time
import torch
from torch.utils.data import DataLoader
from config import _C
from dataset import data_loader
from neural_methods.trainer.PhysFormerTrainer import PhysFormerTrainer
import neural_methods.trainer.PhysFormerTrainer as pft
from contextlib import contextmanager

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

@contextmanager
def isolate_database_csv():
    targets = [
        "csvs/rPPG_results_database.csv",
        "rPPG_results_database.csv",
    ]
    renamed = []
    for t in targets:
        if os.path.exists(t):
            tmp = t + ".eval_tmp"
            os.rename(t, tmp)
            renamed.append((t, tmp))
    try:
        yield
    finally:
        for t, tmp in renamed:
            if os.path.exists(tmp):
                os.rename(tmp, t)

def get_config(test_dataset_name, window_size=6, use_smaller_window=True):
    cfg = _C.clone()
    cfg.TOOLBOX_MODE = 'only_test'
    cfg.DEVICE = device
    cfg.INFERENCE.BATCH_SIZE = 4
    cfg.INFERENCE.EVALUATION_METHOD = 'FFT'
    cfg.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW = use_smaller_window
    cfg.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE = window_size
    cfg.TEST.METRICS = ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
    cfg.TEST.USE_LAST_EPOCH = True
    cfg.TEST.OUTPUT_SAVE_DIR = ""
    cfg.TEST.DATA.FS = 30
    cfg.MODEL.NAME = 'PhysFormer'
    cfg.MODEL.PHYSFORMER.PATCH_SIZE = 4
    cfg.MODEL.PHYSFORMER.DIM = 96
    cfg.MODEL.PHYSFORMER.FF_DIM = 144
    cfg.MODEL.PHYSFORMER.NUM_HEADS = 4
    cfg.MODEL.PHYSFORMER.NUM_LAYERS = 12
    cfg.MODEL.PHYSFORMER.THETA = 0.7

    if test_dataset_name == 'UBFC-rPPG':
        cfg.TEST.DATA.DATASET = 'UBFC-rPPG'
        cfg.TEST.DATA.DATA_PATH = 'UBFC-rPPG_Raw/DATASET_2'
        cfg.TEST.DATA.BEGIN = 0.8
        cfg.TEST.DATA.END = 1.0
        cfg.TEST.DATA.DATA_FORMAT = 'NCDHW'
        cfg.TEST.DATA.CACHED_PATH = 'PreprocessedData'
        cfg.TEST.DATA.EXP_DATA_NAME = 'UBFC-rPPG_b9115a3e49'
        cfg.TEST.DATA.FILE_LIST_PATH = 'PreprocessedData/DataFileLists/UBFC-rPPG_b9115a3e49_0.8_1.0.csv'
        cfg.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 160
        cfg.TEST.DATA.PREPROCESS.RESIZE.H = 128
        cfg.TEST.DATA.PREPROCESS.RESIZE.W = 128
        cfg.TEST.DATA.PREPROCESS.DATA_TYPE = ['Standardized']
        cfg.TEST.DATA.PREPROCESS.LABEL_TYPE = 'Standardized'
        ds = data_loader.UBFCrPPGLoader.UBFCrPPGLoader('test', cfg.TEST.DATA.DATA_PATH, cfg.TEST.DATA, device=cfg.DEVICE)
    elif test_dataset_name == 'PURE':
        cfg.TEST.DATA.DATASET = 'PURE'
        cfg.TEST.DATA.DATA_PATH = 'PURE_dataset'
        cfg.TEST.DATA.BEGIN = 0.8
        cfg.TEST.DATA.END = 1.0
        cfg.TEST.DATA.DATA_FORMAT = 'NCDHW'
        cfg.TEST.DATA.CACHED_PATH = 'PreprocessedData'
        cfg.TEST.DATA.EXP_DATA_NAME = 'PURE_b91e5fa8e7'
        cfg.TEST.DATA.FILE_LIST_PATH = 'PreprocessedData/DataFileLists/PURE_b91e5fa8e7_0.8_1.0.csv'
        cfg.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 160
        cfg.TEST.DATA.PREPROCESS.RESIZE.H = 128
        cfg.TEST.DATA.PREPROCESS.RESIZE.W = 128
        cfg.TEST.DATA.PREPROCESS.DATA_TYPE = ['Standardized']
        cfg.TEST.DATA.PREPROCESS.LABEL_TYPE = 'Standardized'
        ds = data_loader.PURELoader.PURELoader('test', cfg.TEST.DATA.DATA_PATH, cfg.TEST.DATA, device=cfg.DEVICE)
    elif test_dataset_name == 'MMPD':
        cfg.TEST.DATA.DATASET = 'MMPD'
        cfg.TEST.DATA.DATA_PATH = 'MMPD_dataset'
        cfg.TEST.DATA.ONLINE = True
        cfg.TEST.DATA.BEGIN = 0.9
        cfg.TEST.DATA.END = 1.0
        cfg.TEST.DATA.INFO.LIGHT = [1, 2, 3, 4]
        cfg.TEST.DATA.INFO.MOTION = [1, 2, 3, 4]
        cfg.TEST.DATA.INFO.EXERCISE = [1, 2]
        cfg.TEST.DATA.INFO.SKIN_COLOR = [1, 2, 3, 4, 5, 6]
        cfg.TEST.DATA.INFO.GENDER = [1, 2]
        cfg.TEST.DATA.INFO.GLASSER = [1, 2]
        cfg.TEST.DATA.INFO.HAIR_COVER = [1, 2]
        cfg.TEST.DATA.INFO.MAKEUP = [1, 2]
        cfg.TEST.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE = True
        cfg.TEST.DATA.PREPROCESS.CROP_FACE.BACKEND = 'MP'
        cfg.TEST.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX = True
        cfg.TEST.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF = 1.5
        cfg.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION = True
        cfg.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY = 30
        cfg.TEST.DATA.DATA_FORMAT = "NCDHW"
        cfg.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 160
        cfg.TEST.DATA.PREPROCESS.RESIZE.H = 128
        cfg.TEST.DATA.PREPROCESS.RESIZE.W = 128
        cfg.TEST.DATA.PREPROCESS.DATA_TYPE = ['Standardized']
        cfg.TEST.DATA.PREPROCESS.LABEL_TYPE = 'Standardized'
        ds = data_loader.MMPDLoader.MMPDLoader('test', cfg.TEST.DATA.DATA_PATH, cfg.TEST.DATA, device=cfg.DEVICE)
    else:
        raise ValueError(f"Unknown test dataset: {test_dataset_name}")
    return cfg, ds

def run_evaluation(ckpt_path, test_dataset_name, window_size=6, use_smaller_window=True):
    print("=" * 80)
    print(f"Evaluating: {os.path.basename(ckpt_path)} -> {test_dataset_name}")
    print(f"Protocol: Window Size = {window_size}s (USE_SMALLER_WINDOW={use_smaller_window})")
    print("=" * 80)

    cfg, ds = get_config(test_dataset_name, window_size=window_size, use_smaller_window=use_smaller_window)
    loader = DataLoader(ds, batch_size=cfg.INFERENCE.BATCH_SIZE, shuffle=False)
    trainer = PhysFormerTrainer(cfg, {"test": loader})

    captured = {}
    orig_calc = pft.calculate_metrics
    def intercepted(p, l, c):
        m = orig_calc(p, l, c)
        if m: captured.update(m)
        return m
    pft.calculate_metrics = intercepted
    cfg.INFERENCE.MODEL_PATH = ckpt_path

    t0 = time.time()
    try:
        with isolate_database_csv():
            trainer.test({"test": loader})
    finally:
        pft.calculate_metrics = orig_calc
    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"--> Result: MAE={captured.get('MAE'):.3f}, RMSE={captured.get('RMSE'):.3f}, Pearson={captured.get('Pearson'):.3f}, SNR={captured.get('SNR'):.3f}")
    return captured

if __name__ == "__main__":
    # Cross 1: PURE-trained -> UBFC-rPPG
    pure_ckpts = [
        ("runs/exp/PURE_b91e5fa8e7/PreTrainedModels/PURE_PhysFormer_Epoch10.pth", "PURE_Epoch10"),
        ("runs/exp/PURE_b91e5fa8e7/PreTrainedModels/PURE_PhysFormer_Epoch3.pth", "PURE_Epoch3"),
    ]

    print("\n" + "#" * 80)
    print("PART 1.1: PURE-trained Corrected PhysFormer -> UBFC-rPPG Test Set")
    print("#" * 80)
    for ckpt, tag in pure_ckpts:
        if os.path.exists(ckpt):
            # 6s window (apples to apples with thesis protocol)
            m6 = run_evaluation(ckpt, "UBFC-rPPG", window_size=6, use_smaller_window=True)
            # full video (toolbox baseline protocol)
            m_full = run_evaluation(ckpt, "UBFC-rPPG", window_size=10, use_smaller_window=False)

    # Cross 2: UBFC-trained -> PURE and MMPD
    ubfc_ckpts = [
        ("trained_models/UBFC-rPPG/best_checkpoints/PhysFormer_best_Epoch10.pth", "UBFC_Epoch10"),
        ("runs/exp/UBFC-rPPG_b9115a3e49/PreTrainedModels/UBFC-rPPG_PhysFormer_Epoch29.pth", "UBFC_Epoch29"),
    ]

    print("\n" + "#" * 80)
    print("PART 1.2: UBFC-trained Corrected PhysFormer -> PURE Test Set")
    print("#" * 80)
    for ckpt, tag in ubfc_ckpts:
        if os.path.exists(ckpt):
            m6 = run_evaluation(ckpt, "PURE", window_size=6, use_smaller_window=True)
            m_full = run_evaluation(ckpt, "PURE", window_size=10, use_smaller_window=False)

    print("\n" + "#" * 80)
    print("PART 1.3: UBFC-trained Corrected PhysFormer -> MMPD Test Set")
    print("#" * 80)
    for ckpt, tag in ubfc_ckpts:
        if os.path.exists(ckpt):
            m6 = run_evaluation(ckpt, "MMPD", window_size=6, use_smaller_window=True)
            m_full = run_evaluation(ckpt, "MMPD", window_size=10, use_smaller_window=False)
