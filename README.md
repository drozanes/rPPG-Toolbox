<p align="center">
:fire: Please remember to :star: this repo if you find it useful and cite our work if you end up using it in your work! :fire:
</p>
<p align="center">
:fire: If you have any questions or concerns, please create an <a href="https://github.com/ubicomplab/rPPG-Toolbox/issues">issue</a> :memo:! :fire:
</p>


![rPPG-Toolbox Logo](./figures/toolbox_logo.png)

# :wave: Introduction

**rPPG-Toolbox** is an open-source platform designed for camera-based physiological sensing, also known as remote photoplethysmography (rPPG). 

![Overview of the rPPG](./figures/algorithm.png)


rPPG-Toolbox not only benchmarks the **existing state-of-the-art neural and unsupervised methods**, but it also supports flexible and rapid development of your own algorithms.
![Overview of the toolbox](./figures/toolbox_overview.png)


# :notebook: Algorithms
rPPG-Toolbox currently supports the following algorithms: 

* Traditional Unsupervised Algorithms
  - [Remote plethysmographic imaging using ambient light (GREEN)](https://pdfs.semanticscholar.org/7cb4/46d61a72f76e774b696515c55c92c7aa32b6.pdf?_gl=1*1q7hzyz*_ga*NTEzMzk5OTY3LjE2ODYxMDg1MjE.*_ga_H7P4ZT52H5*MTY4NjEwODUyMC4xLjAuMTY4NjEwODUyMS41OS4wLjA), by Verkruysse *et al.*, 2008
  - [Advancements in noncontact multiparameter physiological measurements using a webcam (ICA)](https://affect.media.mit.edu/pdfs/11.Poh-etal-TBME.pdf), by Poh *et al.*, 2011
  - [Robust pulse rate from chrominance-based rppg (CHROM)](https://ieeexplore.ieee.org/document/6523142), by Haan *et al.*, 2013
  - [Local group invariance for heart rate estimation from face videos in the wild (LGI)](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w27/Pilz_Local_Group_Invariance_CVPR_2018_paper.pdf), by Pilz *et al.*, 2018
  - [Improved motion robustness of remote-PPG by using the blood volume pulse signature (PBV)](https://iopscience.iop.org/article/10.1088/0967-3334/35/9/1913), by Haan *et al.*, 2014
  - [Algorithmic principles of remote ppg (POS)](https://ieeexplore.ieee.org/document/7565547), by Wang *et al.*, 2016
  - [Face2PPG: An Unsupervised Pipeline for Blood Volume Pulse Extraction From Faces (OMIT)](https://ieeexplore.ieee.org/document/10227326), by Álvarez *et al.*, 2023


* Supervised Neural Algorithms 
  - [DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks (DeepPhys)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Weixuan_Chen_DeepPhys_Video-Based_Physiological_ECCV_2018_paper.pdf), by Chen *et al.*, 2018
  - [Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks (PhysNet)](https://bmvc2019.org/wp-content/uploads/papers/0186-paper.pdf), by Yu *et al.*, 2019
  - [Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement (TS-CAN)](https://papers.nips.cc/paper/2020/file/e1228be46de6a0234ac22ded31417bc7-Paper.pdf), by Liu *et al.*, 2020
  - [EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Cardiac Measurement (EfficientPhys)](https://openaccess.thecvf.com/content/WACV2023/papers/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.pdf), by Liu *et al.*, 2023
  - [BigSmall: Efficient Multi-Task Learning for Disparate Spatial and Temporal Physiological Measurements
 (BigSmall)](https://arxiv.org/abs/2303.11573), by Narayanswamy *et al.*, 2023
  - [PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer (PhysFormer)](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_PhysFormer_Facial_Video-Based_Physiological_Measurement_With_Temporal_Difference_Transformer_CVPR_2022_paper.pdf), by Yu *et al.*, 2022
  - [iBVPNet: 3D-CNN architecture introduced in iBVP dataset paper](https://doi.org/10.3390/electronics13071334), by Joshi *et al.*, 2024
  - [PhysMamba: Efficient Remote Physiological Measurement with SlowFast Temporal Difference Mamba](https://doi.org/10.48550/arXiv.2409.12031), by Luo *et al.*, 2024
  - [RhythmFormer: Extracting rPPG Signals Based on Hierarchical Temporal Periodic Transformer](https://doi.org/10.48550/arXiv.2402.12788), by Zou *et al.*, 2024
  - [FactorizePhys: Matrix Factorization for Multidimensional Attention in Remote Physiological Sensing](https://proceedings.neurips.cc/paper_files/paper/2024/hash/af1c61e4dd59596f033d826419870602-Abstract-Conference.html), by Joshi *et al.*, 2024

# :file_folder: Datasets
The toolbox supports seven datasets, namely SCAMPS, UBFC-rPPG, PURE, BP4D+, UBFC-Phys, MMPD and iBVP. Please cite the corresponding papers when using these datasets. For now, we recommend training with UBFC-rPPG, PURE, iBVP or SCAMPS due to the level of synchronization and volume of the datasets. **To use these datasets in a deep learning model, you should organize the files as follows.**
* [MMPD](https://github.com/McJackTang/MMPD_rPPG_dataset)
    * Jiankai Tang, Kequan Chen, Yuntao Wang, Yuanchun Shi, Shwetak Patel, Daniel McDuff, Xin Liu, "MMPD: Multi-Domain Mobile Video Physiology Dataset", IEEE EMBC, 2023
    -----------------
         data/MMPD/
         |   |-- subject1/
         |       |-- p1_0.mat
         |       |-- p1_1.mat
         |       |...
         |       |-- p1_19.mat
         |   |-- subject2/
         |       |-- p2_0.mat
         |       |-- p2_1.mat
         |       |...
         |...
         |   |-- subjectn/
         |       |-- pn_0.mat
         |       |-- pn_1.mat
         |       |...
    -----------------
    
* [SCAMPS](https://arxiv.org/abs/2206.04197)
    * D. McDuff, M. Wander, X. Liu, B. Hill, J. Hernandez, J. Lester, T. Baltrusaitis, "SCAMPS: Synthetics for Camera Measurement of Physiological Signals", NeurIPS, 2022
    -----------------
         data/SCAMPS/Train/
            |-- P00001.mat
            |-- P00002.mat
         |...
         data/SCAMPS/Val/
            |-- P00001.mat
            |-- P00002.mat
         |...
         data/SCAMPS/Test/
            |-- P00001.mat
            |-- P00002.mat
         |...
    -----------------

* [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg)
    * S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
    -----------------
         data/UBFC-rPPG/
         |   |-- subject1/
         |       |-- vid.avi
         |       |-- ground_truth.txt
         |   |-- subject2/
         |       |-- vid.avi
         |       |-- ground_truth.txt
         |...
         |   |-- subjectn/
         |       |-- vid.avi
         |       |-- ground_truth.txt
    -----------------
   
* [PURE](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure)
    * Stricker, R., Müller, S., Gross, H.-M.Non-contact "Video-based Pulse Rate Measurement on a Mobile Service Robot"
in: Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
    -----------------
         data/PURE/
         |   |-- 01-01/
         |      |-- 01-01/
         |      |-- 01-01.json
         |   |-- 01-02/
         |      |-- 01-02/
         |      |-- 01-02.json
         |...
         |   |-- ii-jj/
         |      |-- ii-jj/
         |      |-- ii-jj.json
    -----------------
    
* [BP4D+](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)
    * Zhang, Z., Girard, J., Wu, Y., Zhang, X., Liu, P., Ciftci, U., Canavan, S., Reale, M., Horowitz, A., Yang, H., Cohn, J., Ji, Q., Yin, L. "Multimodal Spontaneous Emotion Corpus for Human Behavior Analysis", IEEE International Conference on Computer Vision and Pattern Recognition (CVPR) 2016.   
    -----------------
        RawData/
         |   |-- 2D+3D/
         |       |-- F001.zip/
         |       |-- F002.zip
         |       |...
         |   |-- 2DFeatures/
         |       |-- F001_T1.mat
         |       |-- F001_T2.mat
         |       |...
         |   |-- 3DFeatures/
         |       |-- F001_T1.mat
         |       |-- F001_T2.mat
         |       |...
         |   |-- AUCoding/
         |       |-- AU_INT/
         |            |-- AU06/
         |               |-- F001_T1_AU06.csv
         |               |...
         |           |...
         |       |-- AU_OCC/
         |           |-- F00_T1.csv 
         |           |...
         |   |-- IRFeatures/
         |       |-- F001_T1.txt
         |       |...
         |   |-- Physiology/
         |       |-- F001/
         |           |-- T1/
         |               |-- BP_mmHg.txt
         |               |-- microsiemens.txt
         |               |--LA Mean BP_mmHg.txt
         |               |--LA Systolic BP_mmHg.txt
         |               |-- BP Dia_mmHg.txt
         |               |-- Pulse Rate_BPM.txt
         |               |-- Resp_Volts.txt
         |               |-- Respiration Rate_BPM.txt
         |       |...
         |   |-- Thermal/
         |       |-- F001/
         |           |-- T1.mv
         |           |...
         |       |...
         |   |-- BP4D+UserGuide_v0.2.pdf
    -----------------

* [UBFC-Phys](https://sites.google.com/view/ybenezeth/ubfc-phys)
    * Sabour, R. M., Benezeth, Y., De Oliveira, P., Chappe, J., & Yang, F. (2021). Ubfc-phys: A multimodal database for psychophysiological studies of social stress. IEEE Transactions on Affective Computing.  
    -----------------
          RawData/
          |   |-- s1/
          |       |-- vid_s1_T1.avi
          |       |-- vid_s1_T2.avi
          |       |...
          |       |-- bvp_s1_T1.csv
          |       |-- bvp_s1_T2.csv
          |   |-- s2/
          |       |-- vid_s2_T1.avi
          |       |-- vid_s2_T2.avi
          |       |...
          |       |-- bvp_s2_T1.csv
          |       |-- bvp_s2_T2.csv
          |...
          |   |-- sn/
          |       |-- vid_sn_T1.avi
          |       |-- vid_sn_T2.avi
          |       |...
          |       |-- bvp_sn_T1.csv
          |       |-- bvp_sn_T2.csv
    -----------------

* [iBVP](https://github.com/PhysiologicAILab/iBVP-Dataset)
    * Joshi, J.; Cho, Y. iBVP Dataset: RGB-Thermal rPPG Dataset with High Resolution Signal Quality Labels. Electronics 2024, 13, 1334.
    -----------------
          iBVP_Dataset/
          |   |-- p01_a/
          |      |-- p01_a_rgb/
          |      |-- p01_a_t/
          |      |-- p01_a_bvp.csv
          |   |-- p01_b/
          |      |-- p01_b_rgb/
          |      |-- p01_b_t/
          |      |-- p01_b_bvp.csv
          |...
          |   |-- pii_x/
          |      |-- pii_x_rgb/
          |      |-- pii_x_t/
          |      |-- pii_x_bvp.csv
    -----------------

  * [PhysDrive](https://github.com/WJULYW/PhysDrive-Dataset)
    * Jiyao Wang, Xiao Yang, Qingyong Hu, Jiankai Tang, Can Liu, Dengbo He, Yuntao Wang, Ying-Cong Chen, Kaishun Wu. （2025） PhysDrive: A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring
    -----------------
         On-Road-rPPG/
         |   |-- AFH1/
         |      |-- A1/
         |          |-- Align/
         |              |-- ...png
         |          |-- Label/
         |              |-- BVP.mat
         |              |-- ECG.mat
         |              |-- RESP.mat
         |              |-- SPO2.mat
         |      |-- A2/...
         |      |-- B1/...
         |      |-- B2/...
         |      |-- C1/...
         |      |-- C2/...
         |   |-- AFH2/
         |...
         |   |-- CMZ2/
    -----------------

* [SUMS](https://github.com/thuhci/SUMS/)
    * Ke Liu*, Jiankai Tang*(*Co-first Author), Zhang Jiang, Yuntao Wang, Xiaojing Liu, Dong Li, Yuanchun Shi, "Summit Vitals: Multi-Camera and Multi-Signal Biosensing at High Altitudes", IEEE UIC, 2024
    -----------------
        data/SUMS/
        |   |-- 060200/
        |       |-- v01
        |           |-- BVP.csv
        |           |-- frames_timestamp.csv
        |           |-- HR.csv
        |           |-- RR.csv
        |           |-- video_ZIP_H264_face.avi
        |           |-- video_ZIP_H264_finger.avi
        |       |-- v02
        |       |-- v03
        |       |-- v04
        |   |-- 060201/
        |       |-- v01
        |       |-- v02
        |       |...
        |...
        |   |-- 0602mn/
        |       |-- v01
        |       |-- v02
        |       |...
    -----------------

* [LADH](https://github.com/McJackTang/FusionVitals/)
    * Xulin Ma, Jiankai Tang, Zhang Jiang, Songqin Cheng, Yuanchun Shi, Dong Li, Xin Liu, Daniel Mcduff, Xiaojing Liu, Yuntao Wang, "Non-Contact Health Monitoring During Daily Personal Care Routines", IEEE-EMBS BSN, 2025
    -----------------
        data/LADH/
        |   |-- 12_05/
        |       |-- p_12_05_caip
        |           |-- v01
        |               |-- BVP.csv
        |               |-- HR.csv
        |               |-- RR.csv
        |               |-- SpO2.csv
        |               |-- frames_timestamp_IR.csv
        |               |-- frames_timestamp_RGB.csv
        |               |-- video_RGB_H264.avi
        |               |-- video_IR_H264.avi
        |           |-- v02
        |           |-- v03
        |           |-- v04
        |           |-- v05
        |       |-- p_12_05_huangxj
        |           |-- v01
        |               |-- ...
        |           |-- v02
        |           |-- v03
        |           |-- v04
        |           |-- v05
        |       |-- p_12_05_liutj
        |       |-- p_12_05_lujg
        |       |-- ...
        |   |-- 12_06/
        |       |-- p_12_06_caip
        |       |-- p_12_06_huangxj
        |       |-- p_12_06_liutj
        |       |-- p_12_06_lujg
        |       |...
        |   |-- ...
        |   
        |  
    -----------------

## :bar_chart: Project Benchmarks & Empirical Highlights

This repository contains a comprehensive suite of remote photoplethysmography (rPPG) models, benchmarked under standardized conditions across **UBFC-rPPG** (clean laboratory), **PURE** (head motion & rotation), and **MMPD** (mobile smartphone, low resolution, H.264 compression).

### Key Empirical Milestones

| Model Architecture | Dataset | Evaluation Protocol | MAE (BPM) ↓ | RMSE (BPM) ↓ | Pearson $r$ ↑ | SNR (dB) ↑ | Key Highlight |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`DeepPhys (FusionStem)`** | **PURE** | Supervised Intra-Dataset | **0.128** | **0.229** | **1.000** | **+5.793** | Multi-temporal difference fusion achieves medical-grade precision on motion. |
| **`SpikingPhysFormer`** | **UBFC $\rightarrow$ PURE** | Zero-Shot Cross-Dataset | **0.311** | **0.748** | **0.999** | **+0.761** | Discrete LIF spike-firing filters non-rigid motion; ultra-low inference energy. |
| **`DeepPhys (FusionStem)`** | **PURE $\rightarrow$ UBFC** | Zero-Shot Cross-Dataset | **0.488** | **0.854** | **0.997** | **-1.001** | Motion-trained representation transfers with near-zero error to static scenes. |
| **`POS (Corrected)`** | **UBFC-rPPG** | Unsupervised Signal Processing | **1.549** | **3.996** | **0.975** | **+0.995** | Strict HSV skin-masking + 60s full-video FFT beats standard DeepPhys (1.946). |
| **`CHROM (Corrected)`** | **UBFC-rPPG** | Unsupervised Signal Processing | **1.569** | **3.401** | **0.982** | **-0.098** | Corrected NDHWC format and full-video FFT resolves legacy toolbox degradation. |
| **`ContrastPhys+`** | **UBFC-rPPG** | Weakly-Supervised 3D-CNN | **1.025** | **2.791** | **0.972** | **+1.388** | Active band $[40, 250]$ BPM + $2 \times 2$ ST-rPPG pooling (TPAMI 2024 setup). |
| **`ContrastPhys`** | **UBFC-rPPG** | 100% Label-Free (Min IPR) | **5.882** | **14.401** | **0.601** | **-0.278** | Selected purely without ground-truth PPG labels via Irrelevant Power Ratio (Epoch 13). |
| **`PhysFormer (FusionStem+MP)`** | **MMPD** | Supervised Mobile | **6.158** | **12.546** | **0.613** | **-4.058** | Sub-pixel EMA MediaPipe tracking + FusionStem sets new SOTA on compressed mobile video. |
| **`SpikingPhysFormer (FusionStem+MP)`** | **MMPD** | Supervised Mobile | **15.106** | **23.739** | **0.096** | **-4.441** | SNN cuts test error from 22.52 to 15.11 BPM on official unseen test split (Subj 30–33). |

> [!TIP]
> **Complete Benchmark Data & Visual Reports**:
> - Master local results database: [`csvs/rPPG_results_database.csv`](./csvs/rPPG_results_database.csv).
> - Cross-dataset zero-shot generalization matrix: [`csvs/cross_dataset_benchmarks.csv`](./csvs/cross_dataset_benchmarks.csv).
> - Detailed per-epoch training scans, ablations, and literature bounds: [`csvs/README.md`](./csvs/README.md).
> - Interactive Google Colab Notebooks: [`notebooks/`](./notebooks/) ([DeepPhys](./notebooks/MMPD_DeepPhys_Colab.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drozanes/rPPG-Toolbox/blob/main/notebooks/MMPD_DeepPhys_Colab.ipynb), [PhysFormer / SpikingPhysFormer](./notebooks/MMPD_PhysFormer_SpikingPhysFormer_Colab.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drozanes/rPPG-Toolbox/blob/main/notebooks/MMPD_PhysFormer_SpikingPhysFormer_Colab.ipynb), [ContrastPhys+](./notebooks/ContrastPhys_PURE_Colab.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drozanes/rPPG-Toolbox/blob/main/notebooks/ContrastPhys_PURE_Colab.ipynb)).
> - Forensic charts and visual analytics are located in [`report_assets/`](./report_assets/):
>   - [`report_assets/chart_forensic_improvements.png`](./report_assets/chart_forensic_improvements.png) (Pipeline fixes & skin-mask forensic breakdown)
>   - [`report_assets/chart_mmpd_comparison.png`](./report_assets/chart_mmpd_comparison.png) (Comparative performance on compressed mobile video)
>   - [`report_assets/chart_ubfc_pure_mae.png`](./report_assets/chart_ubfc_pure_mae.png) (Intra-dataset MAE across UBFC and PURE)
>   - [`report_assets/chart_unsupervised_epochs.png`](./report_assets/chart_unsupervised_epochs.png) (IPR vs. Val MAE unsupervised model selection curve)

# :wrench: Setup

You can use either [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [`uv`](https://docs.astral.sh/uv/getting-started/installation/) with this toolbox. Most users are already familiar with `conda`, but `uv` may be a bit less familiar - check out some highlights about `uv` [here](https://docs.astral.sh/uv/#highlights). If you use `uv`, it's highly recommended you do so independently of `conda`, meaning you should make sure you're not installing anything in the base `conda` environment or any other `conda` environment. If you're having trouble making sure you're not in your base `conda` environment, try setting `conda config --set auto_activate_base false`.

STEP 1: `bash setup.sh conda` or `bash setup.sh uv` 

STEP 2: `conda activate rppg-toolbox` or, when using `uv`, `source .venv/bin/activate`

NOTE: the above setup should work without any issues on machines using Linux or MacOS. If you run into compiler-related issues using `uv` when installing tools related to mamba, try checking to see if `clang++` is in your path using `which clang++`. If nothing shows up, you can install `clang++` using `sudo apt-get install clang` on Linux or `xcode-select --install` on MacOS.

If you use Windows or other operating systems, consider using [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) and following the steps within `setup.sh` independently.

# :rocket: Step-by-Step Reproduction Guide

This guide walks through reproducing each core experimental finding across our benchmark suites.

---

### 1. Classical Unsupervised Benchmarks (POS, CHROM, ICA, PBV)

#### Methodological Findings & Pipeline Corrections:
- **Dimension Format Fix**: Classical unsupervised algorithms in `unsupervised_methods/` require `DATA_FORMAT: NDHWC`. Legacy configs configured `NDCHW`, transposing spatial width into color channels on chunked data and causing artificial degradation (e.g., POS MAE jumping from 1.55 to 21.69 BPM).
- **HSV Skin Segmentation**: Background pixels and hair add non-pulsatile noise. A strict HSV skin filter ensures only active capillary tissue is averaged.
- **Evaluation Window**: Evaluating across a 60s full video with Welsh/FFT peak selection prevents spectral leakage common in short 6s chunks.

#### Reproduction Commands:
```bash
# POS on UBFC-rPPG (achieves MAE 1.549 BPM, Pearson 0.975)
python main.py --config_file configs/unsupervised_configs/UBFCrPPG_POS.yaml

# CHROM on UBFC-rPPG (achieves MAE 1.569 BPM, Pearson 0.982)
python main.py --config_file configs/unsupervised_configs/UBFCrPPG_CHROM_NO_CHUNK.yaml

# ICA on UBFC-rPPG (Cardoso JADE BSS with Welch spectral peak selection)
python main.py --config_file configs/unsupervised_configs/UBFCrPPG_ICA.yaml

# PBV on UBFC-rPPG (Blood Volume Pulse signature vector)
python main.py --config_file configs/unsupervised_configs/UBFCrPPG_PBV.yaml

# PURE Unsupervised Suite (Evaluated on 6s sliding chunks)
python main.py --config_file configs/unsupervised_configs/PURE_UNSUPERVISED_CHUNKED.yaml
```

---

### 2. DeepPhys with FusionStem (Achieving 0.128 BPM on PURE)

#### Architectural Innovation:
The standard DeepPhys network takes single-stride frame differences ($S_t - S_{t-1}$). Under head rotation or rapid motion, this difference signal is dominated by non-rigid motion gradients. Our **Multi-Temporal FusionStem** incorporates dual temporal strides with learnable weighting ($\alpha=0.5, \beta=0.5$):
$$\Delta_{\text{fused}} = \alpha (S_t - S_{t-1}) + \beta (S_t - S_{t-2})$$
This temporal fusion suppresses motion-induced phase spikes while amplifying blood volume pulsation. On PURE, this slashes MAE to an unprecedented **0.128 BPM** with **1.000** Pearson correlation.

#### Reproduction Commands:
```bash
# Train DeepPhys with FusionStem on PURE (30 epochs, AdamW LR 1e-4)
python main.py --config_file configs/train_configs/PURE_DEEPPHYS_FUSIONSTEM.yaml

# Evaluate Pretrained Checkpoint (Epoch 11) on PURE test split:
python main.py --config_file configs/train_configs/PURE_DEEPPHYS_FUSIONSTEM.yaml \
  --model_path runs/exp/PURE_af43ac9745/PreTrainedModels/PURE_DeepPhys_FusionStem_Epoch11.pth

# Train DeepPhys with FusionStem on UBFC-rPPG (achieves MAE 1.367 BPM)
python main.py --config_file configs/train_configs/UBFCrPPG_DEEPPHYS_FUSIONSTEM.yaml
```

---

### 3. PhysFormer & SpikingPhysFormer with MediaPipe EMA Dynamic Tracking

#### Stabilizing Video on MMPD (Mobile Smartphone rPPG):
MMPD poses extreme challenges: low resolution ($80 \times 60$), high H.264 compression artifacts, Fitzpatrick skin types 3–6, and multi-axis movement.
- **MediaPipe BlazeFace Dynamic Tracking**: Configured via `CROP_FACE.BACKEND: 'MP'` and `DO_DYNAMIC_DETECTION: True`. Applies sub-pixel Exponential Moving Average (EMA) center-smoothing with fixed scale, eliminating frame-to-frame box jitter.
- **PhysFormer ViT**: 12-layer Spatio-Temporal Difference Transformer operating on $160 \times 128 \times 128$ clips. Combined with FusionStem and MediaPipe tracking, it sets a **new SOTA on MMPD with MAE 6.158 BPM** and doubles Pearson correlation to **0.613**.
- **SpikingPhysFormer (SNN)**: Neuromorphic architecture with Leaky Integrate-and-Fire (LIF) spike neurons ($T_s = 4$). Discrete spike firing inherently filters continuous motion and compression noise, achieving **0.311 BPM** in cross-dataset transfer (UBFC $\rightarrow$ PURE) and **15.106 BPM** on the unseen MMPD test split.

#### Reproduction Commands:
```bash
# Train / Evaluate PhysFormer with MediaPipe Dynamic Tracking on MMPD:
python main.py --config_file configs/train_configs/MMPD_PHYSFORMER_FUSIONSTEM_MEDIAPIPE_ONLINE.yaml

# Train / Evaluate SpikingPhysFormer with MediaPipe Tracking on MMPD:
python main.py --config_file configs/train_configs/MMPD_SPIKINGPHYSFORMER_FUSIONSTEM_MEDIAPIPE_ONLINE.yaml

# Run Zero-Shot Cross-Dataset Generalization Suite:
python run_spiking_cross_tests.py
```

---

### 4. ContrastPhys+ with Irrelevant Power Ratio (IPR) Model Selection

#### Label-Free Self-Supervised Learning:
ContrastPhys+ trains without contact PPG ground truth by contrasting spatial sub-quadrants of the face ($S=2, 2 \times 2$ ST-rPPG pooling) with active PSD bounded in $[40, 250]$ BPM ($[0.67, 4.17]$ Hz).
- **Protocol A (100% Label-Free IPR Selection)**: Checkpoint selection is governed entirely by minimum **Irrelevant Power Ratio (IPR)** on the validation set without ground truth access (Epoch 13 achieves MAE 5.882 BPM on UBFC).
- **Protocol B (Labeled Calibration)**: Calibrates checkpoints using few-shot validation MAE (Epoch 8 achieves MAE 4.597 BPM on UBFC).

#### Reproduction Commands:
```bash
# Automated 30-Epoch Unsupervised Training and Model Selection Scan:
python compare_unsupervised_selection.py

# Evaluate Pretrained ContrastPhys+ Checkpoint on UBFC-rPPG:
python evaluate_test_candidate_epochs.py
```

# :floppy_disk: Pre-trained Checkpoints & Config File Mappings

| Model Architecture | Target Dataset | Paradigm | Configuration File | Checkpoint Path | Verified MAE | Verified Pearson $r$ |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **DeepPhys (FusionStem)** | **PURE** | Supervised 2D-CNN | [`PURE_DEEPPHYS_FUSIONSTEM.yaml`](./configs/train_configs/PURE_DEEPPHYS_FUSIONSTEM.yaml) | [`PURE_DeepPhys_FusionStem_Epoch11.pth`](./runs/exp/PURE_af43ac9745/PreTrainedModels/PURE_DeepPhys_FusionStem_Epoch11.pth) | **0.128** | **1.000** |
| **DeepPhys (FusionStem)** | **UBFC-rPPG** | Supervised 2D-CNN | [`UBFCrPPG_DEEPPHYS_FUSIONSTEM.yaml`](./configs/train_configs/UBFCrPPG_DEEPPHYS_FUSIONSTEM.yaml) | [`UBFC-rPPG_DeepPhys_FusionStem_Epoch5.pth`](./runs/exp/UBFC-rPPG_f9fd2719b6/PreTrainedModels/UBFC-rPPG_DeepPhys_FusionStem_Epoch5.pth) | **1.367** | **0.990** |
| **DeepPhys (Baseline)** | **PURE** | Supervised 2D-CNN | [`PURE_PHYSFORMER.yaml`](./configs/train_configs/PURE_PHYSFORMER.yaml) | [`DeepPhys_Epoch29.pth`](./trained_models/PURE/best_checkpoints/DeepPhys_Epoch29.pth) | **1.728** | **0.917** |
| **PhysFormer (FusionStem+MP)** | **MMPD** | Supervised Transformer | [`MMPD_PHYSFORMER_FUSIONSTEM_MEDIAPIPE_ONLINE.yaml`](./configs/train_configs/MMPD_PHYSFORMER_FUSIONSTEM_MEDIAPIPE_ONLINE.yaml) | [`MMPD_PhysFormer_FusionStem_MediaPipe_Online_Epoch16.pth`](./runs/exp/MMPD_1fa6608630/PreTrainedModels/MMPD_PhysFormer_FusionStem_MediaPipe_Online_Epoch16.pth) | **6.158** | **0.613** |
| **PhysFormer (Paper Aligned)** | **UBFC-rPPG** | Supervised Transformer | [`UBFCrPPG_PHYSFORMER.yaml`](./configs/train_configs/UBFCrPPG_PHYSFORMER.yaml) | [`PostFix_UBFC-rPPG_PhysFormer_Epoch27.pth`](./trained_models/UBFC-rPPG/best_checkpoints/PostFix_UBFC-rPPG_PhysFormer_Epoch27.pth) | **1.639** | **0.966** |
| **PhysFormer (Paper Aligned)** | **PURE** | Supervised Transformer | [`PURE_PHYSFORMER.yaml`](./configs/train_configs/PURE_PHYSFORMER.yaml) | [`PhysFormer_best_Epoch26.pth`](./trained_models/PURE/best_checkpoints/PhysFormer_best_Epoch26.pth) | **6.795** | **0.701** |
| **SpikingPhysFormer (FusionStem+MP)** | **MMPD** | Supervised Neuromorphic (SNN) | [`MMPD_SPIKINGPHYSFORMER_FUSIONSTEM_MEDIAPIPE_ONLINE.yaml`](./configs/train_configs/MMPD_SPIKINGPHYSFORMER_FUSIONSTEM_MEDIAPIPE_ONLINE.yaml) | [`MMPD_SpikingPhysFormer_FusionStem_MediaPipe_Online_Epoch18.pth`](./runs/exp/MMPD_e894957ce5/PreTrainedModels/MMPD_SpikingPhysFormer_FusionStem_MediaPipe_Online_Epoch18.pth) | **15.106** | **0.096** |
| **SpikingPhysFormer (Transfer)** | **UBFC $\rightarrow$ PURE** | Neuromorphic Zero-Shot | [`PURE_SPIKINGPHYSFORMER.yaml`](./configs/train_configs/PURE_SPIKINGPHYSFORMER.yaml) | [`UBFC-rPPG_trained_snn_part.pth`](./pretrained_models/SpikingPhysFormer_Paper/UBFC-rPPG_trained_snn_part.pth) | **0.311** | **0.999** |
| **SpikingPhysFormer (PE Transfer)** | **UBFC-rPPG** | Supervised (SNN) | [`PURE_SPIKINGPHYSFORMER_PAPER.yaml`](./configs/train_configs/PURE_SPIKINGPHYSFORMER_PAPER.yaml) | [`Paper2025_UBFC_SpikingPhysFormer_Paper_Epoch9.pth`](./trained_models/UBFC-rPPG/best_checkpoints/Paper2025_UBFC_SpikingPhysFormer_Paper_Epoch9.pth) | **1.967** | **0.968** |
| **ContrastPhys+ (TPAMI 2024)** | **UBFC-rPPG** | Weakly-Supervised 3D-CNN | [`UBFCrPPG_CONTRASTPHYS_CORRECTED.yaml`](./configs/train_configs/UBFCrPPG_CONTRASTPHYS_CORRECTED.yaml) | [`ContrastPhys_best_Epoch8.pth`](./trained_models/UBFC-rPPG/best_checkpoints/ContrastPhys_best_Epoch8.pth) | **1.025** | **0.972** |
| **ContrastPhys (Min IPR Winner)** | **UBFC-rPPG** | 100% Label-Free Unsupervised | [`UBFCrPPG_CONTRASTPHYS_UNSUPERVISED.yaml`](./configs/train_configs/UBFCrPPG_CONTRASTPHYS_UNSUPERVISED.yaml) | [`Corrected2024_UBFC-rPPG_ContrastPhys_Epoch13.pth`](./trained_models/UBFC-rPPG/best_checkpoints/Corrected2024_UBFC-rPPG_ContrastPhys_Epoch13.pth) | **5.882** | **0.601** |
| **ContrastPhys (Val MAE Winner)** | **UBFC-rPPG** | Unsupervised Calibrated | [`UBFCrPPG_CONTRASTPHYS_UNSUPERVISED.yaml`](./configs/train_configs/UBFCrPPG_CONTRASTPHYS_UNSUPERVISED.yaml) | [`ContrastPhys_best_Epoch8.pth`](./trained_models/UBFC-rPPG/best_checkpoints/ContrastPhys_best_Epoch8.pth) | **4.597** | **0.597** |
| **ContrastPhys (Val MAE Winner)** | **PURE** | Unsupervised Calibrated | [`PURE_CONTRASTPHYS_UNSUPERVISED.yaml`](./configs/train_configs/PURE_CONTRASTPHYS_UNSUPERVISED.yaml) | [`Corrected2024_PURE_ContrastPhys_Epoch9.pth`](./trained_models/PURE/best_checkpoints/Corrected2024_PURE_ContrastPhys_Epoch9.pth) | **8.862** | **0.577** |
| **ContrastPhys (MediaPipe Test)** | **MMPD** | Weakly-Supervised 3D-CNN | [`MMPD_CONTRASTPHYS_ONLINE.yaml`](./configs/train_configs/MMPD_CONTRASTPHYS_ONLINE.yaml) | [`MMPD_ContrastPhys_MediaPipe_Online_Epoch16.pth`](./MMPD_ContrastPhys_MediaPipe_Online_Epoch16.pth) | **19.347** | **0.350** |

# :eyes: Visualization of Preprocessed Data
A python notebook for visualizing preprocessed data can be found in `tools/preprocessing_viz` along with an associated README. The notebook, `viz_preprocessed_data.ipynb`, automatically detects the preprocessed data format and then plots input image examples and waveforms. 
![Data Visualization Example](./tools/preprocessing_viz/preprocessing_viz_example.png)

# :chart_with_downwards_trend: Plots of Training Losses and LR

This toolbox saves plots of training, and if applicable, validation losses automatically. Plots are saved in `LOG.PATH` (`runs/exp` by default). An example of these plots when training and validating with the UBFC-rPPG dataset and testing on the PURE dataset are shown below.

<img src="./figures/example_losses_plot.png" alt="drawing" width="600"/>
<img src="./figures/example_lr_schedule_plot.png" alt="drawing" width="400"/>

# :straight_ruler: Bland-Altman Plots

By default, this toolbox produces Bland-Altman plots as a part of its metrics evaluation process for both supervised and unsupervised methods. These plots are saved in the `LOG.PATH` (`runs/exp` by default). An example of these plots after training and validating with the UBFC-rPPG dataset and testing on the PURE dataset are shown below.

<img src="./figures/example_scatter_plot.png" alt="drawing" width="450"/> <img src="./figures/example_difference_plot.png" alt="drawing" width="450"/>

# :eyes: Visualization of Neural Method Predictions

A python notebook for visualizing test-set neural method output predictions and labels can be found in `tools/output_signal_viz` along with an associated README. The notebook, `data_out_viz.ipynb`, given a `.pickle` output file, generated by setting `TEST.OUTPUT_SAVE_DIR` assists in plotting predicted PPG signals against ground-truth PPG signals.
![Prediction Visualization Example](./tools/output_signal_viz/sample_model_output_viz.png)

# :scroll: YAML File Setting
The rPPG-Toolbox uses yaml file to control all parameters for training and evaluation. 
You can modify the existing yaml files to meet your own training and testing requirements.

Here are some explanation of parameters:
* #### TOOLBOX_MODE: 
  * `train_and_test`: train on the dataset and use the newly trained model to test.
  * `only_test`: you need to set INFERENCE-MODEL_PATH, and it will use pre-trained model initialized with the MODEL_PATH to test.
* #### TRAIN / VALID / TEST / UNSUPERVISED DATA:
  * `PLOT_LOSSES_AND_LR`: If `True`, save plots of the training loss and validation loss, as well as the learning rate, to `LOG.PATH` (`runs/exp` by default). Currently, only a basic training loss and validation loss are plotted, but in the future additional losses utilized in certain trainer files (e.g., PhysFormer and BigSmall) will also be captured.
  * `USE_EXCLUSION_LIST`: If `True`, utilize a provided list to exclude preprocessed videos
  * `SELECT_TASKS`: If `True`, explicitly select tasks to load 
  * `DATA_PATH`: The input path of raw data
  * `CACHED_PATH`: The output path to preprocessed data. This path also houses a directory of .csv files containing data paths to files loaded by the dataloader. This filelist (found in default at CACHED_PATH/DataFileLists). These can be viewed for users to understand which files are used in each data split (train/val/test)
  * `EXP_DATA_NAME` If it is "", the toolbox generates a EXP_DATA_NAME based on other defined parameters. Otherwise, it uses the user-defined EXP_DATA_NAME.  
  * `BEGIN" & "END`: The portion of the dataset used for training/validation/testing. For example, if the `DATASET` is PURE, `BEGIN` is 0.0 and `END` is 0.8 under the TRAIN, the first 80% PURE is used for training the network. If the `DATASET` is PURE, `BEGIN` is 0.8 and `END` is 1.0 under the VALID, the last 20% PURE is used as the validation set. It is worth noting that validation and training sets don't have overlapping subjects.  
  * `DATA_TYPE`: How to preprocess the video data
  * `DATA_AUG`: If present, the type of generative data augmentation applied to video data
  * `LABEL_TYPE`: How to preprocess the label data
  *  `USE_PSUEDO_PPG_LABEL`: If `True` use POS generated PPG psuedo labels instead of dataset ground truth heart singal waveform
  * `DO_CHUNK`: Whether to split the raw data into smaller chunks
  * `CHUNK_LENGTH`: The length of each chunk (number of frames)
  * `DO_CROP_FACE`: Whether to perform face detection
  * `BACKEND`: Select which backend to use for face detection. Currently, the options are HC (Haar Cascade) or Y5F (YOLO5Face). We recommend using Haar Cascade (the config default) in order to reproduce results from the [NeurIPS 2023 Datasets and Benchmarks paper](https://arxiv.org/abs/2210.00716) that corresponds to this toolbox. If you want to use YOLO5Face in your own custom config, we recommend that you reference configs that use it as a default (e.g., FactorizePhys).
  * `DYNAMIC_DETECTION`: If `False`, face detection is only performed at the first frame and the detected box is used to crop the video for all of the subsequent frames. If `True`, face detection is performed at a specific frequency which is defined by `DYNAMIC_DETECTION_FREQUENCY`. 
  * `DYNAMIC_DETECTION_FREQUENCY`: The frequency of face detection (number of frames) if DYNAMIC_DETECTION is `True`
  * `USE_MEDIAN_FACE_BOX`: If `True` and `DYNAMIC_DETECTION` is `True`, use the detected face boxs throughout each video to create a single, median face box per video.
  * `LARGE_FACE_BOX`: Whether to enlarge the rectangle of the detected face region in case the detected box is not large enough for some special cases (e.g., motion videos)
  * `LARGE_BOX_COEF`: The coefficient to scale the face box if `LARGE_FACE_BOX` is `True`.
  * `INFO`: This is a collection of parameters based on attributes of a dataset, such as gender, motion types, and skin color, that help select videos for inclusion in training, validation, or testing. Currently, only the [MMPD](https://github.com/McJackTang/MMPD_rPPG_dataset) dataset is supported for parameter-based video inclusion. Please refer to one of the config files involving the [MMPD](https://github.com/McJackTang/MMPD_rPPG_dataset) dataset for an example of using these parameters.
  * `EXCLUSION_LIST`: A list that specifies videos to exclude, typically based on a unique identifier to a video such as the combination of a subject ID and a task ID. This is only used if `USE_EXCLUSION_LIST` is set to `True`. Currently this parameter is only tested with the [UBFC-Phys](https://sites.google.com/view/ybenezeth/ubfc-phys) dataset. Please refer to one of the config files involving the [UBFC-Phys](https://sites.google.com/view/ybenezeth/ubfc-phys) dataset for an example of using this parameter.
  * `TASK_LIST`: A list to specify tasks to include when loading a dataset, allowing for selective inclusion of a subset of tasks or a single task in a dataset if desired. This is only used if `SELECT_TASKS` is set to `True`. Currently this parameter is only tested with the [UBFC-Phys](https://sites.google.com/view/ybenezeth/ubfc-phys) dataset. Please refer to one of the config files involving the [UBFC-Phys](https://sites.google.com/view/ybenezeth/ubfc-phys) dataset for an example of using this parameter.

  
* #### MODEL : Set used model (Deepphys, TSCAN, Physnet, EfficientPhys, BigSmall, and PhysFormer and their paramaters are supported).
* #### UNSUPERVISED METHOD: Set used unsupervised method. Example: ["ICA", "POS", "CHROM", "GREEN", "LGI", "PBV"]
* #### METRICS: Set used metrics. Example: ['MAE','RMSE','MAPE','Pearson','SNR','BA']
  * 'BA' metric corresponds to the generation of a Bland-Altman plot to graphically compare two measurement techniques (e.g., differences between measured and ground truth heart rates versus mean of measured and ground truth heart rates). This metric saves the plot in the `LOG.PATH` (`runs/exp` by default).
* #### INFERENCE:
  * `USE_SMALLER_WINDOW`: If `True`, use an evaluation window smaller than the video length for evaluation.

    
# :open_file_folder: Adding a New Dataset

* STEP 1: Create a new python file in `dataset/data_loader`, e.g. MyLoader.py

* STEP 2: Implement the required functions, including:

  ```python
  def preprocess_dataset(self, config_preprocess):
  ```
  ```python
  @staticmethod
  def read_video(video_file):
  ```
  ```python
  @staticmethod
  def read_wave(bvp_file):
  ```

* STEP 3:[Optional] Override optional functions. In principle, all functions in BaseLoader can be override, but we **do not** recommend you to override *\_\_len\_\_, \_\_get\_item\_\_,save,load*.
* STEP 4:Set or add configuration parameters.  To set paramteters, create new yaml files in configs/ .  Adding parameters requires modifying config.py, adding new parameters' definition and initial values.

# :robot: Adding a New Neural Algorithms

* STEP 1: Define a model in a new python file in `neural_methods/model`, e.g. NewModel.py.

* STEP 2: Implement the corresponding training/testing routines in a file `neural_methods/trainer`, e.g. NewModelTrainer.py. Ensure to implement the following functions:

  ```python
  def __init__(self, config, data_loader):
  ```
  ```python
  def train(self, data_loader):
  ```
  ```python
  def valid(self, data_loader):
  ```

  ```python
  def test(self, data_loader)
  ```

  ```python
  def save_model(index)
  ```

* STEP 3: Add logic to `main.py` to use the models in the following `train_and_test` and `test` functions. 

* STEP 4: Create new yaml files in configs/ corresponding to the new algorithm.

# :chart_with_upwards_trend: Adding a New Unsupervised Algorithms

* STEP 1: Define a algorithm in a new python file in `unsupervised_methods/methods`, e.g. NewMethod.py.

* STEP 2: Add logic to `main.py` to use the models in the following `unsupervised_method_inference` function. 

* STEP 4: Create new yaml files in configs/ corresponding to the new algorithm.

# :green_book: Weakly Supervised Training 

Supervised rPPG training requires high fidelity synchronous PPG waveform labels. However not all datasets contain such high quality labels. In these cases we offer the option to train on synchronous PPG "pseudo" labels derived through a signal processing methodology. These labels are produced by using POS-generated PPG waveforms, which are then bandpass filtered around the normal heart-rate frequencies, and finally amplitude normalized using a Hilbert-signal envelope. The tight filtering and envelope normalization results in a strong periodic proxy signal, but at the cost of limited signal morphology.

![pseudo_labels](./figures/rppgtoolbox_ppg_psuedo_labels.png)

# :blue_book: Motion Augmented Training

The usage of synthetic data in the training of machine learning models for medical applications is becoming a key tool that warrants further research. In addition to providing support for the fully synthetic dataset [SCAMPS](https://arxiv.org/abs/2206.04197), we provide provide support for synthetic, motion-augmented versions of the [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg), [PURE](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure), [SCAMPS](https://arxiv.org/abs/2206.04197), and [UBFC-Phys](https://sites.google.com/view/ybenezeth/ubfc-phys) datasets for further exploration toward the use of synthetic data for training rPPG models. The synthetic, motion-augmented datasets are generated using the [MA-rPPG Video Toolbox](https://github.com/Roni-Lab/MA-rPPG-Video-Toolbox), an open-source motion augmentation pipeline targeted for increasing motion diversity in rPPG videos. You can generate and utilize the aforementioned motion-augmented datasets using the steps below.

* STEP 1: Follow the instructions in the [README](https://github.com/Roni-Lab/MA-rPPG-Video-Toolbox/blob/main/README.md) of the [MA-rPPG Video Toolbox](https://github.com/Roni-Lab/MA-rPPG-Video-Toolbox) GitHub repo to generate any of the supported motion-augmented datasets. NOTE: You will have to have an original, unaugmented version of a dataset and driving video to generate a motion-augmented dataset. More information can be found [here](https://github.com/Roni-Lab/MA-rPPG-Video-Toolbox#file_folder-datasets). 

* STEP 2: Using any config file of your choice in this toolbox, modify the `DATA_AUG` parameter (set to `'None'` by default) to `'Motion'`. Currently, only `train_configs` that utilize the UBFC-rPPG or PURE datasets have this parameter visible, but you can also modify other config files to add the `DATA_AUG` parameter below the `DATA_TYPE` parameter that is visible in all config files. This will enable the proper funciton for loading motion-augmented data that is in the `.npy` format.

* STEP 3: Run the corresponding config file. Your saved model's filename will have `MA` appended to the corresponding data splits that are motion-augmented.

If you use the aforementioned functionality, please remember to cite the following in addition to citing the rPPG-Toolbox:
* Paruchuri, A., Liu, X., Pan, Y., Patel, S., McDuff, D., & Sengupta, S. (2023). Motion Matters: Neural Motion Transfer for Better Camera Physiological Sensing. arXiv preprint arXiv:2303.12059.

Refer to this [BibTeX](https://github.com/Roni-Lab/MA-rPPG-Video-Toolbox#scroll-citation) for quick inclusion into a `.bib` file.

<p align="center">
  <img src="./figures/ma_rppg_video_toolbox_teaser.gif" alt="Examples of motion augmentation applied to subjects in the UBFC-rPPG dataset." />
</p>

# :orange_book: Extending the Toolbox to Multi-Tasking With BigSmall

We implement [BigSmall](https://girishvn.github.io/BigSmall/) as an example to show how this toolbox may be extended to support physiological multitasking. If you use this functionality please cite the following publication: 
* Narayanswamy, G., Liu, Y., Yang, Y., Ma, C., Liu, X., McDuff, D., Patel, S. "BigSmall: Efficient Multi-Task Learning For Physiological Measurements" https://arxiv.org/abs/2303.11573

The BigSmall mode multi-tasks pulse (PPG regression), respiration (regression), and facial action (multilabel AU classification). The model is trained and evaluated (in this toolbox) on the AU label subset (described in the BigSmall publication) of the BP4D+ dataset, using a 3-fold cross validation method (using the same folds as in the BigSmall publication).

* STEP 1: Download the BP4D+ by emailing the authors found [here](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html).

* STEP 2: Modify `./configs/train_configs/BP4D_BP4D_BIGSMALL_FOLD1.yaml` to train the first fold (config files also exist for the 2nd and 3rd fold).

* STEP 3: Run `python main.py --config_file ./configs/train_configs/BP4D_BP4D_BIGSMALL_FOLD1.yaml `

<p align="center">
  <img src="./figures/bigsmall_ex1.gif" alt="Example Multi-Task Output From BigSmall." />
</p>

# :page_with_curl: Using Custom Data Splits and Custom File Lists

Best practice for rPPG model evaluation involves training and validating a model on one dataset and then evaluating (testing) the performance on additional datasets (Eg. training on PURE and testing on UBFC). Data splits used for training, validation, and testing are saved as .csv filelists with the default directory path set as `CACHED_PATH/DataFileLists` (this are generally auto generated). In cases where users would like to define their own data splits (Eg. for intra-dataset cross validation), the following steps can be used to achieve this.  

* STEP 1: Collect all file paths for the data splits's input files (An example file path would be of format `CACHED_PATH/PREPROCESSED_DATASET/*input*.npy`, eg. `405_input9.npy`). Ensure the corresponding `*label*.npy` files exists in the same directory as the input files.

* STEP 2: Add all file paths to a .csv with column name `input_files`. We suggest creating a pandas dataframe and then saving this out to the csv. This .csv file is the file list for your custom data split. The file can be named anything as long as it has an .csv extension.

* STEP 3: In the desired experiment config file, enter the file path of the .csv file list using keyword `FILE_LIST_PATH` in the `DATA` field of the desired train/valid/test split. The dataloader will load input and label files specified in this file path. 

# :scroll: Citation
If you find our [paper](https://arxiv.org/abs/2210.00716) or this toolbox useful for your research, please cite our work.

```
@article{liu2022rppg,
  title={rPPG-Toolbox: Deep Remote PPG Toolbox},
  author={Liu, Xin and Narayanswamy, Girish and Paruchuri, Akshay and Zhang, Xiaoyu and Tang, Jiankai and Zhang, Yuzhe and Wang, Yuntao and Sengupta, Soumyadip and Patel, Shwetak and McDuff, Daniel},
  journal={arXiv preprint arXiv:2210.00716},
  year={2022}
}
```

# License
<a href="https://www.licenses.ai/source-code-license">
  <img src="https://images.squarespace-cdn.com/content/v1/5c2a6d5c45776e85d1482a7e/1546750722018-T7QVBTM15DQMBJF6A62M/RAIL+Final.png" alt="License: Responsible AI" width="30%">
</a>

# Acknowledgement 

This research project is supported by a Google PhD Fellowship for Xin Liu and a research grant from Cisco for the University of Washington as well as a career start-up funding grant from the Department of Computer Science at UNC Chapel Hill. This research is also supported by Tsinghua University Initiative Scientific Research Program, Beijing Natural Science Foundation,  and the Natural Science Foundation of China (NSFC). We also would like to acknowledge all the contributors from the open-source community. 

