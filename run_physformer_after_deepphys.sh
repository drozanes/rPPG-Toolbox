#!/bin/bash
TARGET_PID=857
echo "[$(date)] Auto-scheduler active. Waiting for DeepPhys (PID: $TARGET_PID) to finish..."

while kill -0 $TARGET_PID 2>/dev/null; do
    sleep 30
done

echo ""
echo "================================================================="
echo "[$(date)] DeepPhys completed! Launching PhysFormer + FusionStem..."
echo "================================================================="

.venv/bin/python main.py --config_file configs/train_configs/MMPD_PHYSFORMER_FUSIONSTEM_ONLINE.yaml \
               --device mps --num_workers 0 --batch_size 4 2>&1 | tee local_train_physformer.log

echo "[$(date)] PhysFormer with FusionStem training finished!"
