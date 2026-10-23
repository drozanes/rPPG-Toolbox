#!/bin/bash
CURRENT_PID=83173

echo "=========================================================================="
echo "          RPPG TOOLBOX - WEEKEND CONTINUOUS TRAINING SUITE               "
echo "=========================================================================="
echo "Started at: $(date)"
echo "Active Phase: Monitoring PhysFormer + FusionStem (PID: $CURRENT_PID)"
echo ""

# 1. Wait for PhysFormer to complete
while kill -0 $CURRENT_PID 2>/dev/null; do
    sleep 30
done

echo ""
echo "=========================================================================="
echo "[$(date)] Phase 1 Complete! PhysFormer + FusionStem finished."
echo "=========================================================================="
echo ""

# 2. Launch SpikingPhysFormer (without FusionStem)
echo "=========================================================================="
echo "[$(date)] Phase 2: Launching SpikingPhysFormer on MMPD (without FusionStem)..."
echo "Parameters: batch_size=4, num_workers=0, device=mps, early_stopping=4"
echo "=========================================================================="

.venv/bin/python main.py \
    --config_file configs/train_configs/MMPD_SPIKINGPHYSFORMER_ONLINE.yaml \
    --device mps --num_workers 0 --batch_size 4 \
    2>&1 | tee local_train_spikingphysformer.log

echo ""
echo "=========================================================================="
echo "[$(date)] Phase 2 Complete! SpikingPhysFormer finished."
echo "=========================================================================="
echo ""

# 3. Launch ContrastPhys+ (without FusionStem)
echo "=========================================================================="
echo "[$(date)] Phase 3: Launching ContrastPhys+ on MMPD (without FusionStem)..."
echo "Parameters: batch_size=2, num_workers=0, device=mps, IPR_selection=True"
echo "=========================================================================="

.venv/bin/python main.py \
    --config_file configs/train_configs/MMPD_CONTRASTPHYS_ONLINE.yaml \
    --device mps --num_workers 0 --batch_size 2 \
    2>&1 | tee local_train_contrastphys.log

echo ""
echo "=========================================================================="
echo "[$(date)] ALL WEEKEND TRAINING PHASES COMPLETED SUCCESSFULLY!"
echo "Finished at: $(date)"
echo "=========================================================================="
