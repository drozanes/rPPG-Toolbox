import pandas as pd
import numpy as np
import cv2
import os
from scipy.interpolate import interp1d

def prepare_subject(subject_dir):
    xmp_path = os.path.join(subject_dir, 'gtdump.xmp')
    vid_path = os.path.join(subject_dir, 'vid.avi')
    gt_out_path = os.path.join(subject_dir, 'ground_truth.txt')
    
    if not os.path.exists(xmp_path) or not os.path.exists(vid_path):
        print(f"Skipping {subject_dir}: Missing files.")
        return

    # 1. Get video frame count
    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # 2. Parse XMP and resample
    df = pd.read_csv(xmp_path, header=None)
    bvp = df.iloc[:, 3].values
    ts = df.iloc[:, 0].values
    ts = ts - ts[0]
    
    interp_func = interp1d(ts, bvp, kind='linear', fill_value='extrapolate')
    target_ts = np.linspace(0, ts[-1], total_frames)
    bvp_resampled = interp_func(target_ts)
    
    # 3. Save as space-separated single line
    with open(gt_out_path, 'w') as f:
        f.write(' '.join(bvp_resampled.astype(str)))
    
    print(f"Prepared {subject_dir}: {total_frames} frames.")

if __name__ == "__main__":
    for s in ['subject8', 'subject10', 'subject11']:
        prepare_subject(f"test_data/{s}")
