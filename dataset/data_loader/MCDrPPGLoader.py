"""The dataloader for MCD-rPPG dataset."""

import glob
import os
import re
import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader

class MCDrPPGLoader(BaseLoader):
    """The data loader for the MCD-rPPG dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes an MCD-rPPG dataloader.
        """
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories and file paths under the path."""
        db_path = os.path.join(data_path, "db.csv")
        if not os.path.exists(db_path):
            raise ValueError(f"db.csv not found in {data_path}. Is the dataset fully downloaded?")
        
        df = pd.read_csv(db_path)
        dirs = []
        for index, row in df.iterrows():
            video_rel_path = row['video']
            bvp_rel_path = row['ppg_sync']
            video_full_path = os.path.join(data_path, video_rel_path)
            
            # We only append to dirs if the video file is specified (could be empty if some are missing)
            if pd.notna(video_rel_path):
                dirs.append({
                    "index": f"Subject{row['patient_id']}_{row['camera']}_{row['step']}",
                    "path": data_path,
                    "video_path": video_full_path,
                    "bvp_path": os.path.join(data_path, bvp_rel_path) if pd.notna(bvp_rel_path) else None,
                    "subject": int(row['patient_id'])
                })
        
        if not dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values,
        ensuring no overlapping subjects between splits."""
        if begin == 0 and end == 1:
            return data_dirs
            
        data_info = dict()
        for data in data_dirs:
            subject = data['subject']
            if subject not in data_info:
                data_info[subject] = list()
            data_info[subject].append(data)
            
        subj_list = sorted(list(data_info.keys()))
        num_subjs = len(subj_list)
        
        subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))
        
        data_dirs_new = list()
        for i in subj_range:
            subj_num = subj_list[i]
            data_dirs_new += data_info[subj_num]
            
        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        saved_filename = data_dirs[i]['index']

        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            frames = self.read_video(data_dirs[i]['video_path'])
        elif 'Motion' in config_preprocess.DATA_AUG:
            frames = self.read_npy_video(
                glob.glob(data_dirs[i]['video_path'].replace('.avi', '*.npy'))
            )
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL or data_dirs[i]['bvp_path'] is None:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(data_dirs[i]['bvp_path'])
            # Resample BVP to match the number of video frames
            target_length = frames.shape[0]
            bvps = BaseLoader.resample_ppg(bvps, target_length)

        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        VidObj.release()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        # Check if it's a CSV or TXT, and try to extract the signal
        # Assuming the signal is a simple array of values
        try:
            # Try to read as a simple list of floats
            bvp = np.loadtxt(bvp_file)
            
            # If it has multiple columns (e.g., value and timestamp), usually the first column is the value
            # Let's check dimensions
            if len(bvp.shape) > 1:
                bvp = bvp[:, 0]
                
        except Exception as e:
            # Fallback: read line by line
            with open(bvp_file, "r") as f:
                str1 = f.read().strip().split("\n")
                bvp = [float(x.split()[0]) for x in str1 if x.strip()]
            bvp = np.asarray(bvp)
            
        return bvp
