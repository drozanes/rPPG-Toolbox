#!/usr/bin/env python3
"""
Official Pretrained Models Benchmark Runner (Intra & Cross Dataset)
Repository: drozanes/rPPG-Toolbox

Benchmarks the official pretrained weights in final_model_release/
matching the architectures evaluated in the thesis:
- DeepPhys (UBFC-rPPG, PURE)
- PhysFormer (UBFC-rPPG, PURE)
- PhysNet (UBFC-rPPG, PURE)

Evaluates on:
- UBFC-rPPG (Intra & Cross)
- PURE (Intra & Cross)
- MMPD (Cross)

Logs results incrementally to csvs/official_pretrained_models_benchmark.csv
"""

import os
import sys
import csv
import time
import argparse
import traceback
from datetime import datetime
from contextlib import contextmanager
import numpy as np
import torch
from torch.utils.data import DataLoader

# Enable MPS fallback for operators not natively implemented on Apple Silicon MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add repo root to python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import _C
from dataset import data_loader
import evaluation.metrics

CSV_PATH = os.path.join(REPO_ROOT, "csvs", "official_pretrained_models_benchmark.csv")
CSV_PATH_6S = os.path.join(REPO_ROOT, "csvs", "official_pretrained_models_benchmark_6s.csv")
CSV_HEADERS = [
    "Train_Dataset", "Test_Dataset", "Evaluation_Type", "Model_Name",
    "Architecture", "Weights_File", "MAE_BPM", "RMSE_BPM",
    "MAPE_Percent", "Pearson_r", "SNR_dB", "Status", "Timestamp"
]


@contextmanager
def isolate_database_csv():
    """Temporarily hide thesis master database so official benchmark results stay isolated."""
    targets = [
        os.path.join(REPO_ROOT, "csvs", "rPPG_results_database.csv"),
        os.path.join(REPO_ROOT, "rPPG_results_database.csv"),
    ]
    renamed = []
    for t in targets:
        if os.path.exists(t):
            tmp = t + ".benchmark_isolation"
            os.rename(t, tmp)
            renamed.append((t, tmp))
    try:
        yield
    finally:
        for t, tmp in renamed:
            if os.path.exists(tmp):
                os.rename(tmp, t)


def init_csv(csv_path=None):
    if csv_path is None:
        csv_path = CSV_PATH
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        print(f"📄 Initialized new benchmark CSV at: {csv_path}")


def is_already_completed(train_ds, test_ds, model_name, csv_path=None):
    if csv_path is None:
        csv_path = CSV_PATH
    if not os.path.exists(csv_path):
        return False
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("Train_Dataset") == train_ds and
                row.get("Test_Dataset") == test_ds and
                row.get("Model_Name") == model_name and
                row.get("Status") == "Success"):
                return True
    return False


def save_benchmark_result(train_ds, test_ds, model_name, architecture, weights_file, metrics, status="Success", csv_path=None):
    if csv_path is None:
        csv_path = CSV_PATH
    init_csv(csv_path)
    eval_type = "Intra-Dataset" if train_ds == test_ds else "Cross-Dataset"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            train_ds,
            test_ds,
            eval_type,
            model_name,
            architecture,
            weights_file,
            f"{metrics.get('MAE', 0.0):.3f}" if metrics else "N/A",
            f"{metrics.get('RMSE', 0.0):.3f}" if metrics else "N/A",
            f"{metrics.get('MAPE', 0.0):.3f}" if metrics else "N/A",
            f"{metrics.get('Pearson', 0.0):.3f}" if metrics else "N/A",
            f"{metrics.get('SNR', 0.0):.3f}" if metrics else "N/A",
            status,
            now_str
        ])
    print(f"\n💾 [Saved Result -> {os.path.basename(csv_path)}] [{train_ds} -> {test_ds} | {model_name}] MAE={metrics.get('MAE', 'N/A')}, Pearson={metrics.get('Pearson', 'N/A')}, SNR={metrics.get('SNR', 'N/A')} dB")


def get_base_config(device="mps", window_size=6, use_smaller_window=True):
    config = _C.clone()
    config.TOOLBOX_MODE = "only_test"
    config.DEVICE = device
    config.INFERENCE.BATCH_SIZE = 4
    config.INFERENCE.EVALUATION_METHOD = "FFT"
    config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW = use_smaller_window
    config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE = window_size
    config.TEST.METRICS = ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
    config.TEST.USE_LAST_EPOCH = True
    config.TEST.OUTPUT_SAVE_DIR = ""  # Disable saving giant raw frame folders
    return config


def setup_test_dataset(config, test_dataset_name, model_type):
    config.TEST.DATA.DATASET = test_dataset_name
    config.TEST.DATA.DO_PREPROCESS = False
    config.TEST.DATA.FS = 30

    if test_dataset_name == "PURE":
        config.TEST.DATA.DATA_PATH = os.path.join(REPO_ROOT, "test_data/PURE")
        config.TEST.DATA.BEGIN = 0.8
        config.TEST.DATA.END = 1.0

        if model_type == "DeepPhys":
            config.TEST.DATA.DATA_FORMAT = "NDCHW"
            config.TEST.DATA.CACHED_PATH = os.path.join(REPO_ROOT, "PreprocessedData/PURE_af43ac9745")
            config.TEST.DATA.FILE_LIST_PATH = os.path.join(REPO_ROOT, "PreprocessedData/DataFileLists/PURE_af43ac9745_0.8_1.0.csv")
            config.TEST.DATA.EXP_DATA_NAME = "PURE_af43ac9745"
            config.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 180
            config.TEST.DATA.PREPROCESS.RESIZE.H = 72
            config.TEST.DATA.PREPROCESS.RESIZE.W = 72
            config.TEST.DATA.PREPROCESS.DATA_TYPE = ['DiffNormalized', 'Raw']
            config.TEST.DATA.PREPROCESS.LABEL_TYPE = 'DiffNormalized'
        elif model_type == "PhysFormer":
            config.TEST.DATA.DATA_FORMAT = "NCDHW"
            config.TEST.DATA.CACHED_PATH = os.path.join(REPO_ROOT, "PreprocessedData/PURE_b91e5fa8e7")
            config.TEST.DATA.FILE_LIST_PATH = os.path.join(REPO_ROOT, "PreprocessedData/DataFileLists/PURE_b91e5fa8e7_0.8_1.0.csv")
            config.TEST.DATA.EXP_DATA_NAME = "PURE_b91e5fa8e7"
            config.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 160
            config.TEST.DATA.PREPROCESS.RESIZE.H = 128
            config.TEST.DATA.PREPROCESS.RESIZE.W = 128
            config.TEST.DATA.PREPROCESS.DATA_TYPE = ['Standardized']
            config.TEST.DATA.PREPROCESS.LABEL_TYPE = 'Standardized'
        elif model_type in ["Physnet", "PhysNet"]:
            config.TEST.DATA.DATA_FORMAT = "NCDHW"
            config.TEST.DATA.CACHED_PATH = os.path.join(REPO_ROOT, "PreprocessedData/PURE_608f362849")
            config.TEST.DATA.FILE_LIST_PATH = os.path.join(REPO_ROOT, "PreprocessedData/DataFileLists/PURE_608f362849_0.8_1.0.csv")
            config.TEST.DATA.EXP_DATA_NAME = "PURE_608f362849"
            config.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 128
            config.TEST.DATA.PREPROCESS.RESIZE.H = 128
            config.TEST.DATA.PREPROCESS.RESIZE.W = 128
            config.TEST.DATA.PREPROCESS.DATA_TYPE = ['DiffNormalized']
            config.TEST.DATA.PREPROCESS.LABEL_TYPE = 'DiffNormalized'
        return data_loader.PURELoader.PURELoader('test', config.TEST.DATA.DATA_PATH, config.TEST.DATA, device=config.DEVICE)

    elif test_dataset_name == "UBFC-rPPG":
        config.TEST.DATA.DATA_PATH = os.path.join(REPO_ROOT, "test_data/UBFC-rPPG")
        config.TEST.DATA.BEGIN = 0.8
        config.TEST.DATA.END = 1.0

        if model_type == "DeepPhys":
            config.TEST.DATA.DATA_FORMAT = "NDCHW"
            config.TEST.DATA.CACHED_PATH = os.path.join(REPO_ROOT, "PreprocessedData/UBFC-rPPG_f9fd2719b6")
            config.TEST.DATA.FILE_LIST_PATH = os.path.join(REPO_ROOT, "PreprocessedData/DataFileLists/UBFC-rPPG_f9fd2719b6_0.8_1.0.csv")
            config.TEST.DATA.EXP_DATA_NAME = "UBFC-rPPG_f9fd2719b6"
            config.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 180
            config.TEST.DATA.PREPROCESS.RESIZE.H = 72
            config.TEST.DATA.PREPROCESS.RESIZE.W = 72
            config.TEST.DATA.PREPROCESS.DATA_TYPE = ['DiffNormalized', 'Raw']
            config.TEST.DATA.PREPROCESS.LABEL_TYPE = 'DiffNormalized'
        elif model_type == "PhysFormer":
            config.TEST.DATA.DATA_FORMAT = "NCDHW"
            config.TEST.DATA.CACHED_PATH = os.path.join(REPO_ROOT, "PreprocessedData/UBFC-rPPG_b9115a3e49")
            config.TEST.DATA.FILE_LIST_PATH = os.path.join(REPO_ROOT, "PreprocessedData/DataFileLists/UBFC-rPPG_b9115a3e49_0.8_1.0.csv")
            config.TEST.DATA.EXP_DATA_NAME = "UBFC-rPPG_b9115a3e49"
            config.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 160
            config.TEST.DATA.PREPROCESS.RESIZE.H = 128
            config.TEST.DATA.PREPROCESS.RESIZE.W = 128
            config.TEST.DATA.PREPROCESS.DATA_TYPE = ['Standardized']
            config.TEST.DATA.PREPROCESS.LABEL_TYPE = 'Standardized'
        elif model_type in ["Physnet", "PhysNet"]:
            config.TEST.DATA.DATA_FORMAT = "NCDHW"
            config.TEST.DATA.CACHED_PATH = os.path.join(REPO_ROOT, "PreprocessedData/UBFC-rPPG_8ac63bff29")
            config.TEST.DATA.FILE_LIST_PATH = os.path.join(REPO_ROOT, "PreprocessedData/DataFileLists/UBFC-rPPG_8ac63bff29_0.8_1.0.csv")
            config.TEST.DATA.EXP_DATA_NAME = "UBFC-rPPG_8ac63bff29"
            config.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 128
            config.TEST.DATA.PREPROCESS.RESIZE.H = 128
            config.TEST.DATA.PREPROCESS.RESIZE.W = 128
            config.TEST.DATA.PREPROCESS.DATA_TYPE = ['DiffNormalized']
            config.TEST.DATA.PREPROCESS.LABEL_TYPE = 'DiffNormalized'
        return data_loader.UBFCrPPGLoader.UBFCrPPGLoader('test', config.TEST.DATA.DATA_PATH, config.TEST.DATA, device=config.DEVICE)

    elif test_dataset_name == "MMPD":
        config.TEST.DATA.DATA_PATH = os.path.join(REPO_ROOT, "MMPD_dataset")
        config.TEST.DATA.ONLINE = True
        config.TEST.DATA.BEGIN = 0.9
        config.TEST.DATA.END = 1.0
        config.TEST.DATA.INFO.LIGHT = [1, 2, 3, 4]
        config.TEST.DATA.INFO.MOTION = [1, 2, 3, 4]
        config.TEST.DATA.INFO.EXERCISE = [1, 2]
        config.TEST.DATA.INFO.SKIN_COLOR = [1, 2, 3, 4, 5, 6]
        config.TEST.DATA.INFO.GENDER = [1, 2]
        config.TEST.DATA.INFO.GLASSER = [1, 2]
        config.TEST.DATA.INFO.HAIR_COVER = [1, 2]
        config.TEST.DATA.INFO.MAKEUP = [1, 2]
        config.TEST.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE = True
        config.TEST.DATA.PREPROCESS.CROP_FACE.BACKEND = 'MP'
        config.TEST.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX = True
        config.TEST.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF = 1.5
        config.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION = True
        config.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY = 30

        if model_type == "DeepPhys":
            config.TEST.DATA.DATA_FORMAT = "NDCHW"
            config.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 180
            config.TEST.DATA.PREPROCESS.RESIZE.H = 72
            config.TEST.DATA.PREPROCESS.RESIZE.W = 72
            config.TEST.DATA.PREPROCESS.DATA_TYPE = ['DiffNormalized', 'Raw']
            config.TEST.DATA.PREPROCESS.LABEL_TYPE = 'DiffNormalized'
        elif model_type == "PhysFormer":
            config.TEST.DATA.DATA_FORMAT = "NCDHW"
            config.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 160
            config.TEST.DATA.PREPROCESS.RESIZE.H = 128
            config.TEST.DATA.PREPROCESS.RESIZE.W = 128
            config.TEST.DATA.PREPROCESS.DATA_TYPE = ['Standardized']
            config.TEST.DATA.PREPROCESS.LABEL_TYPE = 'Standardized'
        elif model_type in ["Physnet", "PhysNet"]:
            config.TEST.DATA.DATA_FORMAT = "NCDHW"
            config.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 128
            config.TEST.DATA.PREPROCESS.RESIZE.H = 128
            config.TEST.DATA.PREPROCESS.RESIZE.W = 128
            config.TEST.DATA.PREPROCESS.DATA_TYPE = ['DiffNormalized']
            config.TEST.DATA.PREPROCESS.LABEL_TYPE = 'DiffNormalized'
        return data_loader.MMPDLoader.MMPDLoader('test', config.TEST.DATA.DATA_PATH, config.TEST.DATA, device=config.DEVICE)

    else:
        raise ValueError(f"Unknown target dataset: {test_dataset_name}")


def get_all_benchmark_specs():
    specs = [
        # 1-6: DeepPhys
        {
            "id": 0,
            "train_dataset": "UBFC-rPPG",
            "test_dataset": "UBFC-rPPG",
            "model_type": "DeepPhys",
            "model_name": "DeepPhys (Official Pretrained)",
            "architecture": "2D-CNN (Appearance + Motion)",
            "checkpoint": "final_model_release/UBFC-rPPG_DeepPhys.pth",
            "batch_size": 8
        },
        {
            "id": 1,
            "train_dataset": "UBFC-rPPG",
            "test_dataset": "PURE",
            "model_type": "DeepPhys",
            "model_name": "DeepPhys (Official Pretrained)",
            "architecture": "2D-CNN (Appearance + Motion)",
            "checkpoint": "final_model_release/UBFC-rPPG_DeepPhys.pth",
            "batch_size": 8
        },
        {
            "id": 2,
            "train_dataset": "UBFC-rPPG",
            "test_dataset": "MMPD",
            "model_type": "DeepPhys",
            "model_name": "DeepPhys (Official Pretrained)",
            "architecture": "2D-CNN (Appearance + Motion)",
            "checkpoint": "final_model_release/UBFC-rPPG_DeepPhys.pth",
            "batch_size": 8
        },
        {
            "id": 3,
            "train_dataset": "PURE",
            "test_dataset": "PURE",
            "model_type": "DeepPhys",
            "model_name": "DeepPhys (Official Pretrained)",
            "architecture": "2D-CNN (Appearance + Motion)",
            "checkpoint": "final_model_release/PURE_DeepPhys.pth",
            "batch_size": 8
        },
        {
            "id": 4,
            "train_dataset": "PURE",
            "test_dataset": "UBFC-rPPG",
            "model_type": "DeepPhys",
            "model_name": "DeepPhys (Official Pretrained)",
            "architecture": "2D-CNN (Appearance + Motion)",
            "checkpoint": "final_model_release/PURE_DeepPhys.pth",
            "batch_size": 8
        },
        {
            "id": 5,
            "train_dataset": "PURE",
            "test_dataset": "MMPD",
            "model_type": "DeepPhys",
            "model_name": "DeepPhys (Official Pretrained)",
            "architecture": "2D-CNN (Appearance + Motion)",
            "checkpoint": "final_model_release/PURE_DeepPhys.pth",
            "batch_size": 8
        },

        # 7-12: PhysFormer
        {
            "id": 6,
            "train_dataset": "UBFC-rPPG",
            "test_dataset": "UBFC-rPPG",
            "model_type": "PhysFormer",
            "model_name": "PhysFormer (Official Pretrained)",
            "architecture": "Spatio-Temporal ViT (12-layer)",
            "checkpoint": "final_model_release/UBFC-rPPG_PhysFormer_DiffNormalized.pth",
            "batch_size": 4
        },
        {
            "id": 7,
            "train_dataset": "UBFC-rPPG",
            "test_dataset": "PURE",
            "model_type": "PhysFormer",
            "model_name": "PhysFormer (Official Pretrained)",
            "architecture": "Spatio-Temporal ViT (12-layer)",
            "checkpoint": "final_model_release/UBFC-rPPG_PhysFormer_DiffNormalized.pth",
            "batch_size": 4
        },
        {
            "id": 8,
            "train_dataset": "UBFC-rPPG",
            "test_dataset": "MMPD",
            "model_type": "PhysFormer",
            "model_name": "PhysFormer (Official Pretrained)",
            "architecture": "Spatio-Temporal ViT (12-layer)",
            "checkpoint": "final_model_release/UBFC-rPPG_PhysFormer_DiffNormalized.pth",
            "batch_size": 4
        },
        {
            "id": 9,
            "train_dataset": "PURE",
            "test_dataset": "PURE",
            "model_type": "PhysFormer",
            "model_name": "PhysFormer (Official Pretrained)",
            "architecture": "Spatio-Temporal ViT (12-layer)",
            "checkpoint": "final_model_release/PURE_PhysFormer_DiffNormalized.pth",
            "batch_size": 4
        },
        {
            "id": 10,
            "train_dataset": "PURE",
            "test_dataset": "UBFC-rPPG",
            "model_type": "PhysFormer",
            "model_name": "PhysFormer (Official Pretrained)",
            "architecture": "Spatio-Temporal ViT (12-layer)",
            "checkpoint": "final_model_release/PURE_PhysFormer_DiffNormalized.pth",
            "batch_size": 4
        },
        {
            "id": 11,
            "train_dataset": "PURE",
            "test_dataset": "MMPD",
            "model_type": "PhysFormer",
            "model_name": "PhysFormer (Official Pretrained)",
            "architecture": "Spatio-Temporal ViT (12-layer)",
            "checkpoint": "final_model_release/PURE_PhysFormer_DiffNormalized.pth",
            "batch_size": 4
        },

        # 13-18: PhysNet
        {
            "id": 12,
            "train_dataset": "UBFC-rPPG",
            "test_dataset": "UBFC-rPPG",
            "model_type": "Physnet",
            "model_name": "PhysNet (Official Pretrained)",
            "architecture": "3D-CNN Spatio-Temporal",
            "checkpoint": "final_model_release/UBFC-rPPG_PhysNet_DiffNormalized.pth",
            "batch_size": 4
        },
        {
            "id": 13,
            "train_dataset": "UBFC-rPPG",
            "test_dataset": "PURE",
            "model_type": "Physnet",
            "model_name": "PhysNet (Official Pretrained)",
            "architecture": "3D-CNN Spatio-Temporal",
            "checkpoint": "final_model_release/UBFC-rPPG_PhysNet_DiffNormalized.pth",
            "batch_size": 4
        },
        {
            "id": 14,
            "train_dataset": "UBFC-rPPG",
            "test_dataset": "MMPD",
            "model_type": "Physnet",
            "model_name": "PhysNet (Official Pretrained)",
            "architecture": "3D-CNN Spatio-Temporal",
            "checkpoint": "final_model_release/UBFC-rPPG_PhysNet_DiffNormalized.pth",
            "batch_size": 4
        },
        {
            "id": 15,
            "train_dataset": "PURE",
            "test_dataset": "PURE",
            "model_type": "Physnet",
            "model_name": "PhysNet (Official Pretrained)",
            "architecture": "3D-CNN Spatio-Temporal",
            "checkpoint": "final_model_release/PURE_PhysNet_DiffNormalized.pth",
            "batch_size": 4
        },
        {
            "id": 16,
            "train_dataset": "PURE",
            "test_dataset": "UBFC-rPPG",
            "model_type": "Physnet",
            "model_name": "PhysNet (Official Pretrained)",
            "architecture": "3D-CNN Spatio-Temporal",
            "checkpoint": "final_model_release/PURE_PhysNet_DiffNormalized.pth",
            "batch_size": 4
        },
        {
            "id": 17,
            "train_dataset": "PURE",
            "test_dataset": "MMPD",
            "model_type": "Physnet",
            "model_name": "PhysNet (Official Pretrained)",
            "architecture": "3D-CNN Spatio-Temporal",
            "checkpoint": "final_model_release/PURE_PhysNet_DiffNormalized.pth",
            "batch_size": 4
        },
    ]
    return specs


def run_benchmark(spec, device="mps", force=False, window_size=6, use_smaller_window=True, csv_path=None):
    if csv_path is None:
        csv_path = CSV_PATH_6S if use_smaller_window else CSV_PATH

    train_ds = spec["train_dataset"]
    test_ds = spec["test_dataset"]
    model_type = spec["model_type"]
    model_name = spec["model_name"]
    arch = spec["architecture"]
    ckpt_rel = spec["checkpoint"]
    ckpt_path = os.path.join(REPO_ROOT, ckpt_rel)
    batch_size = spec.get("batch_size", 4)
    eval_type = "Intra-Dataset" if train_ds == test_ds else "Cross-Dataset"
    win_str = f"{window_size}s Window" if use_smaller_window else "Full Video"

    print("\n" + "=" * 80)
    print(f"🚀 [{spec['id'] + 1}/18] Benchmark: {eval_type} [{train_ds} -> {test_ds}] ({win_str})")
    print(f"   Model: {model_name} ({arch})")
    print(f"   Checkpoint: {ckpt_rel}")
    print(f"   Device: {device} | Batch size: {batch_size}")
    print("=" * 80)

    if not force and is_already_completed(train_ds, test_ds, model_name, csv_path=csv_path):
        print(f"⏭️  Already completed successfully in {csv_path}. Skipping (use --force to re-run).")
        return

    if not os.path.exists(ckpt_path):
        err_msg = f"Checkpoint not found at: {ckpt_path}"
        print(f"❌ {err_msg}")
        save_benchmark_result(train_ds, test_ds, model_name, arch, ckpt_rel, {}, status=f"Error: Checkpoint missing", csv_path=csv_path)
        return

    config = get_base_config(device=device, window_size=window_size, use_smaller_window=use_smaller_window)
    config.MODEL.NAME = model_type
    config.INFERENCE.MODEL_PATH = ckpt_path
    config.INFERENCE.BATCH_SIZE = batch_size

    if model_type == "DeepPhys":
        config.MODEL.USE_FUSION_STEM = False
        config.MODEL.DEEPPHYS.USE_FUSION_STEM = False
    elif model_type == "PhysFormer":
        config.MODEL.USE_FUSION_STEM = False
        config.MODEL.PHYSFORMER.USE_FUSION_STEM = False
        config.MODEL.PHYSFORMER.PATCH_SIZE = 4
        config.MODEL.PHYSFORMER.DIM = 96
        config.MODEL.PHYSFORMER.FF_DIM = 144
        config.MODEL.PHYSFORMER.NUM_HEADS = 4
        config.MODEL.PHYSFORMER.NUM_LAYERS = 12
        config.MODEL.PHYSFORMER.THETA = 0.7
    elif model_type in ["Physnet", "PhysNet"]:
        config.MODEL.USE_FUSION_STEM = False
        config.MODEL.PHYSNET.FRAME_NUM = 128

    try:
        start_t = time.time()
        test_dataset = setup_test_dataset(config, test_ds, model_type)
        print(f"   Loaded Test Split Size: {len(test_dataset)} samples")
        loader_dict = {"test": DataLoader(test_dataset, batch_size=config.INFERENCE.BATCH_SIZE, shuffle=False)}

        captured_metrics = {}
        orig_calc = evaluation.metrics.calculate_metrics

        def intercepted_calc(predictions, labels, cfg):
            m = orig_calc(predictions, labels, cfg)
            if m:
                captured_metrics.update(m)
            return m

        # Monkey-patch calculate_metrics in trainers and module
        evaluation.metrics.calculate_metrics = intercepted_calc

        if model_type == "DeepPhys":
            import neural_methods.trainer.DeepPhysTrainer as dpt
            dpt.calculate_metrics = intercepted_calc
            trainer = dpt.DeepPhysTrainer(config, loader_dict)
        elif model_type == "PhysFormer":
            import neural_methods.trainer.PhysFormerTrainer as pft
            pft.calculate_metrics = intercepted_calc
            trainer = pft.PhysFormerTrainer(config, loader_dict)
        elif model_type in ["Physnet", "PhysNet"]:
            import neural_methods.trainer.PhysnetTrainer as pnt
            pnt.calculate_metrics = intercepted_calc
            trainer = pnt.PhysnetTrainer(config, loader_dict)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        try:
            with isolate_database_csv():
                trainer.test(loader_dict)
        finally:
            evaluation.metrics.calculate_metrics = orig_calc
            if model_type == "DeepPhys":
                dpt.calculate_metrics = orig_calc
            elif model_type == "PhysFormer":
                pft.calculate_metrics = orig_calc
            elif model_type in ["Physnet", "PhysNet"]:
                pnt.calculate_metrics = orig_calc

        elapsed = time.time() - start_t
        print(f"⏱️ Evaluation completed in {elapsed:.1f}s")
        save_benchmark_result(train_ds, test_ds, model_name, arch, ckpt_rel, captured_metrics, status="Success", csv_path=csv_path)

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        traceback.print_exc()
        save_benchmark_result(train_ds, test_ds, model_name, arch, ckpt_rel, {}, status=f"Error: {e}", csv_path=csv_path)


def main():
    parser = argparse.ArgumentParser(description="Official Pretrained Models Benchmark Runner")
    parser.add_argument("--models", nargs="+", choices=["DeepPhys", "PhysFormer", "Physnet", "PhysNet"],
                        default=None, help="Filter by model type")
    parser.add_argument("--train_datasets", nargs="+", choices=["UBFC-rPPG", "PURE"],
                        default=None, help="Filter by train dataset")
    parser.add_argument("--test_datasets", nargs="+", choices=["UBFC-rPPG", "PURE", "MMPD"],
                        default=None, help="Filter by test dataset")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu",
                        help="Device to run on (mps, cpu, cuda)")
    parser.add_argument("--window_size", type=int, default=6,
                        help="Evaluation window size in seconds (default: 6)")
    parser.add_argument("--full_video", action="store_true",
                        help="Use full video FFT instead of 6s window")
    parser.add_argument("--csv_file", type=str, default=None,
                        help="Custom output CSV path")
    parser.add_argument("--force", action="store_true", help="Re-run even if already completed in CSV")
    parser.add_argument("--dry_run", action="store_true", help="Inspect specs and files without executing")
    parser.add_argument("--list", action="store_true", help="List all 18 specs and exit")
    parser.add_argument("--run_id", type=int, default=None, help="Run only a specific spec ID (0-17)")

    args = parser.parse_args()

    use_smaller_window = not args.full_video
    target_csv = args.csv_file if args.csv_file else (CSV_PATH if args.full_video else CSV_PATH_6S)
    init_csv(target_csv)
    all_specs = get_all_benchmark_specs()

    if args.list:
        print("\n📋 Full Matrix of 18 Official Pretrained Benchmarks:")
        print(f"{'ID':<4} {'Train':<10} {'Test':<10} {'Type':<8} {'Model':<12} {'Weights'}")
        print("-" * 75)
        for s in all_specs:
            kind = "Intra" if s['train_dataset'] == s['test_dataset'] else "Cross"
            print(f"{s['id']:<4} {s['train_dataset']:<10} {s['test_dataset']:<10} {kind:<8} {s['model_type']:<12} {s['checkpoint']}")
        return

    selected = []
    for s in all_specs:
        if args.run_id is not None and s["id"] != args.run_id:
            continue
        if args.models and s["model_type"] not in args.models:
            continue
        if args.train_datasets and s["train_dataset"] not in args.train_datasets:
            continue
        if args.test_datasets and s["test_dataset"] not in args.test_datasets:
            continue
        selected.append(s)

    win_desc = "Full-Video" if args.full_video else f"{args.window_size}s Window"
    print(f"\n🎯 Selected {len(selected)} benchmark evaluations under {win_desc} protocol.")
    print(f"   Destination CSV: {target_csv}")

    if args.dry_run:
        print("\n🔍 DRY RUN - Checking checkpoints:")
        for s in selected:
            p = os.path.join(REPO_ROOT, s["checkpoint"])
            exists = os.path.exists(p)
            print(f"[{'✓' if exists else '✗'}] ID {s['id']}: {s['train_dataset']} -> {s['test_dataset']} ({s['model_type']}) - {p}")
        return

    for s in selected:
        run_benchmark(s, device=args.device, force=args.force, window_size=args.window_size,
                      use_smaller_window=use_smaller_window, csv_path=target_csv)

    print(f"\n🎉 All planned benchmarks finished! Results logged to:\n   {target_csv}\n")


if __name__ == "__main__":
    main()
