# Trained PURE Models Summary

This folder contains all trained rPPG models on the PURE dataset (30 epochs each).

## Best Checkpoints (`best_checkpoints/`)

| Model | Architecture | Best Epoch | Test MAE (BPM) | Test RMSE | Test SNR (dB) | Weights File |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepPhys** | 2D-CNN (Two-stream) | Epoch 29 | 1.728 | 9.481 | 4.975 | `best_checkpoints/DeepPhys_Epoch29.pth` |
| **SpikingPhysFormer** | SNN + Transformer | Epoch 29 | 2.100 | 10.119 | 3.674 | `best_checkpoints/SpikingPhysFormer_Epoch29.pth` |
| **ContrastPhys** | 3D-CNN (PhysNet backbone) | Epoch 4 | 15.813 | 25.695 | -6.839 | `best_checkpoints/ContrastPhys_best_Epoch4.pth` |
| **PhysFormer** | Spatio-temporal Transformer | Epoch 26 | 23.142 | 38.711 | 0.421 | `best_checkpoints/PhysFormer_best_Epoch26.pth` |

## All Epochs (`all_epochs/`)
Symlinks to full epoch checkpoint directories (Epoch 0 through Epoch 29):
- `all_epochs/DeepPhys/`
- `all_epochs/SpikingPhysFormer/`
- `all_epochs/ContrastPhys/`
- `all_epochs/PhysFormer/`
