import pickle
import numpy as np
import cv2
import torch
import os
import scipy.signal
import argparse
import pandas as pd
from scipy.interpolate import interp1d
from evaluation.post_process import _calculate_fft_hr

def smooth(x, window_len=15):
    if window_len < 2: return x
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    w = np.ones(window_len, 'd')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[int(window_len/2):][:len(x)]

def visualize_results(video_path, pickle_path, xmp_path, output_path, method_name="Model", fs=30):
    # 1. Load the predicted data
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    subj_id = list(data['predictions'].keys())[0]
    preds_dict = data['predictions'][subj_id]
    labels_dict = data['labels'][subj_id]
    
    sorted_indices = sorted(preds_dict.keys())
    all_preds = []
    all_labels = []
    for idx in sorted_indices:
        p = preds_dict[idx]
        l = labels_dict[idx]
        if isinstance(p, torch.Tensor): p = p.cpu().numpy()
        if isinstance(l, torch.Tensor): l = l.cpu().numpy()
        all_preds.append(p)
        all_labels.append(l)
    
    pred_signal = np.concatenate(all_preds).flatten()
    label_signal = np.concatenate(all_labels).flatten()
    
    # 2. Load the original XMP for Sensor HR
    df_xmp = pd.read_csv(xmp_path, header=None)
    sensor_hr_raw = df_xmp.iloc[:, 1].values
    ts_xmp = df_xmp.iloc[:, 0].values
    ts_xmp = ts_xmp - ts_xmp[0]
    
    # Video handling
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Resample Sensor HR to video frames
    f_interp = interp1d(ts_xmp, sensor_hr_raw, kind='linear', fill_value='extrapolate')
    target_ts = np.linspace(0, ts_xmp[-1], total_frames)
    sensor_hr_resampled = f_interp(target_ts)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    
    print(f"Total video frames: {total_frames}")
    print(f"Signal length: {len(pred_signal)}")
    
    window_size = 10 * fs
    bpm_preds = []
    bpm_gts = []
    
    # First pass: calculate all BPMs to allow smoothing
    print("Calculating BPMs...")
    for frame_idx in range(total_frames):
        if frame_idx >= window_size and frame_idx < len(pred_signal):
            start = frame_idx - window_size
            end = frame_idx
            win_pred = pred_signal[start:end]
            win_label = label_signal[start:end]
            
            def clean_sig(sig):
                sig = scipy.signal.detrend(sig)
                b, a = scipy.signal.butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
                return scipy.signal.filtfilt(b, a, sig)

            try:
                # Use higher NFFT for better resolution (2048)
                def calc_hr(sig):
                    sig = clean_sig(sig)
                    sig = np.expand_dims(sig, 0)
                    f, pxx = scipy.signal.periodogram(sig, fs=fs, nfft=2048, detrend=False)
                    mask = (f >= 0.75) & (f <= 2.5)
                    hr = f[mask][np.argmax(pxx[0][mask])] * 60
                    return hr

                bpm_preds.append(calc_hr(win_pred))
                bpm_gts.append(calc_hr(win_label))
            except:
                bpm_preds.append(0)
                bpm_gts.append(0)
        else:
            bpm_preds.append(0)
            bpm_gts.append(0)
            
    # Apply temporal smoothing (moving average)
    bpm_preds_smoothed = smooth(np.array(bpm_preds), window_len=30)
    bpm_gts_smoothed = smooth(np.array(bpm_gts), window_len=30)

    # Second pass: write video
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        b_p = bpm_preds_smoothed[frame_idx]
        b_g = bpm_gts_smoothed[frame_idx]
        s_h = sensor_hr_resampled[frame_idx]
            
        cv2.putText(frame, f"Sensor HR (GT): {s_h:.1f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Wave HR (GT):   {b_g:.1f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{method_name} BPM: {b_p:.1f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Algorithm: {method_name}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        out.write(frame)
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"Writing frames {frame_idx}/{total_frames}...")

    cap.release()
    out.release()
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="test_data/subject1/vid.avi")
    parser.add_argument("--pickle", required=True)
    parser.add_argument("--xmp", default="test_data/subject1/gtdump.xmp")
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="Model")
    args = parser.parse_args()
    
    visualize_results(args.video, args.pickle, args.xmp, args.output, args.name)
