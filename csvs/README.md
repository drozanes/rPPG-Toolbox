# Centralized Benchmark & Experiment Results (`csvs/`)

This directory houses the structured CSV evaluation data for the rPPG research project.

To keep the workspace clean and easy to navigate, only the **3 Core Master Tables** reside in the root of this directory, while detailed training progression scans, literature comparisons, and raw sensor comparisons are organized into subdirectories.

---

## 🌟 Core Master Tables (Root Level)

| File | Description | Metrics / Content |
| :--- | :--- | :--- |
| **[`rPPG_results_database.csv`](file:///Users/drozanes/dev/rPPG-Toolbox/csvs/rPPG_results_database.csv)** | **Master Results Database**: Consolidated metrics across all **85 benchmark runs** conducted in this project. | `Method`, `Dataset`, `Paradigm`, `MAE`, `RMSE`, `MAPE`, `Pearson_r`, `SNR_dB`, `Hardware`, `FPS` |
| **[`local_benchmarks_intra_dataset.csv`](file:///Users/drozanes/dev/rPPG-Toolbox/csvs/local_benchmarks_intra_dataset.csv)** | **Intra-Dataset Benchmarks**: Head-to-head comparison across UBFC-rPPG, PURE, and MMPD for classical, supervised, neuromorphic (SNN), and contrastive models. | Detailed intra-dataset evaluations with baseline vs. our improvements |
| **[`cross_dataset_benchmarks.csv`](file:///Users/drozanes/dev/rPPG-Toolbox/csvs/cross_dataset_benchmarks.csv)** | **Cross-Dataset Generalization Matrix**: 24 zero-shot domain transfer evaluations (Lab $\leftrightarrow$ Dynamic Motion $\leftrightarrow$ Mobile Phone). | Evaluation of domain gap robustness (e.g. UBFC $\rightarrow$ PURE, PURE $\rightarrow$ UBFC) |

---

## 📁 Subdirectories

### 1. [`detailed_scans/`](file:///Users/drozanes/dev/rPPG-Toolbox/csvs/detailed_scans/)
Per-epoch training logs, model-selection validation scans, and ablation studies:
- `ubfc-rppg_unsupervised_scan_all_epochs.csv`: 30-epoch progression of ContrastPhys+ on UBFC-rPPG.
- `pure_unsupervised_scan_all_epochs.csv`: 30-epoch progression of ContrastPhys+ under head motion on PURE.
- `mmpd_validation_results.csv`: Validation metrics across training epochs on compressed mobile video.
- `mmpd_contrastphys_val_scan_mediaPipe.csv`: Training scan with online MediaPipe BlazeFace tracking.
- `mmpd_dynamic_crop_evaluation.csv`: Quantitative impact of face crop jitter (Static vs. Discrete 1.0Hz step vs. Continuous EMA).
- `unsupervised_model_selection_comparison.csv`: Validation of 100% label-free IPR selection vs. ground-truth MAE.
- `contrastphys_experiments_comparison.csv`: Contrastive loss formulations and spatial grid ablations.

### 2. [`literature_benchmarks/`](file:///Users/drozanes/dev/rPPG-Toolbox/csvs/literature_benchmarks/)
External benchmark databases compiled from published papers:
- `rppg_comprehensive_benchmark_database.csv`: 417 literature benchmark evaluations synthesized from 34 publications.
- `published_intra_dataset_ranges.csv`: Reported literature metric ranges across 9 top peer-reviewed papers.
- `rppg_summary_ranges.csv`: Upper/lower performance bounds across 85 distinct published setups.

### 3. [`raw_sensor_comparisons/`](file:///Users/drozanes/dev/rPPG-Toolbox/csvs/raw_sensor_comparisons/)
Per-recording pulse rate measurements comparing contact pulse oximeter sensors to rPPG algorithms:
- `PURE_Comparison.csv`: Sequence-by-sequence pulse measurements across 60 PURE recordings.
- `UBFC_Comparison.csv`: Reference ground-truth audit across UBFC-rPPG recordings.
- `COHFACE_Comparison.csv`: Audit across 160 COHFACE recordings (noting the initial 30-frame hardware sensor corruption).
