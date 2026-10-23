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
echo "[$(date)] Evaluating ContrastPhys+ Results on MMPD..."
echo "=========================================================================="

MAE_VAL=$(grep "FFT MAE" local_train_contrastphys.log | tail -n 1 | awk '{print $4}')
echo "ContrastPhys+ MMPD Test MAE: ${MAE_VAL:-N/A} BPM"

if [ -n "$MAE_VAL" ] && [ "$(echo "$MAE_VAL <= 20.0" | bc -l 2>/dev/null)" = "1" ]; then
    echo "✅ ContrastPhys+ achieved strong results on MMPD! Triggering PURE and UBFC-rPPG runs..."
    
    # Run on PURE
    echo "[$(date)] Launching ContrastPhys+ on PURE..."
    .venv/bin/python main.py \
        --config_file configs/train_configs/PURE_CONTRASTPHYS.yaml \
        --device mps --num_workers 0 --batch_size 2 \
        2>&1 | tee local_train_contrastphys_pure.log

    # Run on UBFC-rPPG
    echo "[$(date)] Launching ContrastPhys+ on UBFC-rPPG..."
    .venv/bin/python main.py \
        --config_file configs/train_configs/UBFCrPPG_CONTRASTPHYS.yaml \
        --device mps --num_workers 0 --batch_size 2 \
        2>&1 | tee local_train_contrastphys_ubfc.log
else
    echo "⚠️ ContrastPhys+ MMPD results: ${MAE_VAL:-N/A} BPM."
    echo "Note: MMPD contains aggressive phone camera compression, 4 lighting setups, and 4 motion types."
fi

# 4. Launch SpikingPhysFormer WITH FusionStem
echo ""
echo "=========================================================================="
echo "[$(date)] Phase 4: Launching SpikingPhysFormer + FusionStem on MMPD..."
echo "Parameters: batch_size=4, num_workers=0, device=mps, early_stopping=4"
echo "=========================================================================="

.venv/bin/python main.py \
    --config_file configs/train_configs/MMPD_SPIKINGPHYSFORMER_FUSIONSTEM_ONLINE.yaml \
    --device mps --num_workers 0 --batch_size 4 \
    2>&1 | tee local_train_spikingphysformer_fusionstem.log

# 5. Launch ContrastPhys+ WITH FusionStem
echo ""
echo "=========================================================================="
echo "[$(date)] Phase 5: Launching ContrastPhys+ + FusionStem on MMPD..."
echo "Parameters: batch_size=2, num_workers=0, device=mps, early_stopping=4"
echo "=========================================================================="

.venv/bin/python main.py \
    --config_file configs/train_configs/MMPD_CONTRASTPHYS_FUSIONSTEM_ONLINE.yaml \
    --device mps --num_workers 0 --batch_size 2 \
    2>&1 | tee local_train_contrastphys_fusionstem.log

echo ""
echo "=========================================================================="
echo "[$(date)] ALL WEEKEND TRAINING PHASES COMPLETED SUCCESSFULLY!"
echo "Finished at: $(date)"
echo "=========================================================================="
