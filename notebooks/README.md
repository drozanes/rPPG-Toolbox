# 📓 Google Colab Notebooks for rPPG-Toolbox

This directory contains standalone, reproducible Google Colab notebooks configured for high-performance GPU training and evaluation on standard remote photoplethysmography (rPPG) benchmark datasets.

---

## 🚀 Available Notebooks

| Notebook | Target Architecture | Dataset | Colab Badge | Key Features |
| :--- | :--- | :--- | :---: | :--- |
| **`MMPD_DeepPhys_Colab.ipynb`** | DeepPhys (Convolutional Attention) | **MMPD** (Compressed Mobile) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drozanes/rPPG-Toolbox/blob/main/notebooks/MMPD_DeepPhys_Colab.ipynb) | Zero-disk streaming loader, automated checkpoint mirroring to Google Drive, automatic test evaluation. |
| **`MMPD_PhysFormer_SpikingPhysFormer_Colab.ipynb`** | PhysFormer (ViT) & Spiking-PhysFormer (SNN) | **MMPD** (Compressed Mobile) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drozanes/rPPG-Toolbox/blob/main/notebooks/MMPD_PhysFormer_SpikingPhysFormer_Colab.ipynb) | Multi-temporal difference FusionStem, MediaPipe BlazeFace online tracking, SpikingJelly LIF neurons, WandB tracking. |
| **`ContrastPhys_PURE_Colab.ipynb`** | ContrastPhys+ (Self-Supervised 3D-CNN) | **PURE** (Motion & Rotation) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drozanes/rPPG-Toolbox/blob/main/notebooks/ContrastPhys_PURE_Colab.ipynb) | Self-supervised contrastive learning on PURE, Spatio-Temporal rPPG pooling (TPAMI 2024 setup), cloud checkpoint backup. |

---

## ⚙️ Usage Instructions

1. Click any **Open In Colab** badge above or open the `.ipynb` file in [Google Colab](https://colab.research.google.com/).
2. Select a GPU runtime: **Runtime -> Change runtime type -> Hardware accelerator -> T4 / L4 / A100 GPU**.
3. (Optional but recommended) Enable **High-RAM** in the runtime settings for accelerated CPU data loading.
4. Run each cell sequentially. All checkpoints and evaluation tables will sync to Google Drive (`MyDrive/rPPG_Checkpoints`).
