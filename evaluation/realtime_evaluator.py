"""Real-time Remote Photoplethysmography (rPPG) Evaluation Engine.

Provides live streaming inference from webcams or video files, supporting:
- Neural models: DeepPhys, EfficientPhys, TS-CAN, PhysNet, ContrastPhys
- Modular FusionStem support for custom trained models
- Unsupervised algorithms: POS, CHROM, GREEN, ICA
- Motion-resilient Inner Core ROI tracking, deadband spatial damping, and HSV skin masking
- Real-time bandpass filtering, detrending, and FFT-based heart rate (BPM) estimation
"""

import os
import time
import math
import collections
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
import scipy.signal
import torch
import torch.nn as nn

from evaluation.post_process import _calculate_fft_hr, _detrend, _next_power_of_2
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.model.EfficientPhys import EfficientPhys
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.ContrastPhysNet import ContrastPhysNet
from unsupervised_methods.methods.POS_WANG import POS_WANG
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
from unsupervised_methods.methods.GREEN import GREEN
from unsupervised_methods.methods.ICA_POH import ICA_POH


SUPPORTED_NEURAL_MODELS = ["DeepPhys", "EfficientPhys", "TS-CAN", "TSCAN", "PhysNet", "ContrastPhys"]
SUPPORTED_UNSUPERVISED_MODELS = ["POS", "CHROM", "GREEN", "ICA"]
ALL_SUPPORTED_MODELS = ["EfficientPhys", "DeepPhys", "TS-CAN", "ContrastPhys", "PhysNet", "POS", "CHROM", "GREEN", "ICA"]


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Clean checkpoint keys: strip DataParallel 'module.' and map legacy layer names."""
    new_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace("module.", "") if k.startswith("module.") else k
        # Remap legacy ContrastPhysNet layer names
        if clean_key == "end.1.weight":
            clean_key = "conv_end.weight"
        elif clean_key == "end.1.bias":
            clean_key = "conv_end.bias"
        new_dict[clean_key] = v
    return new_dict


def find_default_weights(model_name: str, dataset: str = "UBFC-rPPG", base_dir: Optional[str] = None) -> Optional[str]:
    """Find appropriate pre-trained checkpoint in final_model_release or trained_models."""
    norm_name = "TSCAN" if model_name.upper() in ["TS-CAN", "TSCAN"] else model_name

    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    search_dirs = [
        os.path.join(base_dir, "final_model_release"),
        os.path.join(base_dir, "trained_models", dataset, "best_checkpoints"),
        os.path.join(base_dir, "trained_models", "UBFC-rPPG", "best_checkpoints"),
        os.path.join(base_dir, "trained_models", "PURE", "best_checkpoints"),
    ]

    for s_dir in search_dirs:
        if not os.path.isdir(s_dir):
            continue
        candidates = [
            f"{dataset}_{norm_name}.pth",
            f"{dataset}_{norm_name}_DiffNormalized.pth",
            f"{dataset}_{model_name}.pth",
            f"PURE_{norm_name}.pth",
            f"UBFC-rPPG_{norm_name}.pth",
            f"SCAMPS_{norm_name}.pth",
            f"Corrected2024_{dataset}_{model_name}_Epoch29.pth",
            f"ContrastPhys_best_Epoch8.pth",
        ]
        for c in candidates:
            full_p = os.path.join(s_dir, c)
            if os.path.isfile(full_p):
                return full_p

        # Partial match - check for files starting with model name or containing it
        for f in sorted(os.listdir(s_dir), reverse=True):
            if f.endswith(".pth") and norm_name.lower() in f.lower():
                return os.path.join(s_dir, f)

    return None


class FaceTracker:
    """Motion-resilient face detector with deadband spatial damping and inner core ROI cropping."""

    def __init__(
        self,
        haarcascade_path: Optional[str] = None,
        smoothing: float = 0.88,
        roi_mode: str = "core",
        deadband_px: int = 5,
    ):
        self.smoothing = smoothing
        self.roi_mode = roi_mode
        self.deadband_px = deadband_px
        self.current_box: Optional[Tuple[int, int, int, int]] = None
        self.missing_count = 0
        self.max_missing = 45  # Keep last stable box for up to 1.5s on head turns

        cascade_candidates = [
            haarcascade_path,
            os.path.join(os.path.dirname(__file__), "..", "dataset", "haarcascade_frontalface_default.xml"),
            os.path.join("dataset", "haarcascade_frontalface_default.xml"),
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml" if hasattr(cv2, "data") else None,
        ]
        xml_path = None
        for p in cascade_candidates:
            if p and os.path.isfile(p):
                xml_path = p
                break

        if xml_path is None:
            raise FileNotFoundError("Could not locate haarcascade_frontalface_default.xml.")

        self.detector = cv2.CascadeClassifier(xml_path)

    def detect_and_smooth(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=3,
            minSize=(35, 35),
        )

        if len(faces) > 0:
            areas = [bw * bh for (_, _, bw, bh) in faces]
            raw_box = faces[int(np.argmax(areas))]
            rx, ry, rw, rh = raw_box

            target_box = (rx, ry, rw, rh)

            if self.current_box is None:
                self.current_box = target_box
            else:
                cx_cur, cy_cur, cw_cur, ch_cur = self.current_box
                dx = abs(target_box[0] - cx_cur)
                dy = abs(target_box[1] - cy_cur)
                dw = abs(target_box[2] - cw_cur)
                dh = abs(target_box[3] - ch_cur)

                # Deadband: ignore micro-jitter below threshold to avoid introducing artificial motion noise
                if dx > self.deadband_px or dy > self.deadband_px or dw > self.deadband_px or dh > self.deadband_px:
                    smooth_x = int(self.smoothing * cx_cur + (1.0 - self.smoothing) * target_box[0])
                    smooth_y = int(self.smoothing * cy_cur + (1.0 - self.smoothing) * target_box[1])
                    smooth_w = int(self.smoothing * cw_cur + (1.0 - self.smoothing) * target_box[2])
                    smooth_h = int(self.smoothing * ch_cur + (1.0 - self.smoothing) * target_box[3])
                    self.current_box = (smooth_x, smooth_y, smooth_w, smooth_h)

            self.missing_count = 0
            return self.current_box
        else:
            self.missing_count += 1
            if self.missing_count < self.max_missing and self.current_box is not None:
                return self.current_box
            self.current_box = None
            return None

    def extract_roi(self, frame_bgr: np.ndarray, box: Tuple[int, int, int, int], apply_skin_mask: bool = True) -> np.ndarray:
        """Extract inner facial core (cheeks + forehead) and optionally apply skin thresholding."""
        h_frame, w_frame = frame_bgr.shape[:2]
        bx, by, bw, bh = box

        # Bound check
        x1 = max(0, bx)
        y1 = max(0, by)
        x2 = min(w_frame, bx + bw)
        y2 = min(h_frame, by + bh)
        face_roi = frame_bgr[y1:y2, x1:x2]

        if face_roi.size == 0:
            return np.zeros((72, 72, 3), dtype=np.uint8)

        if self.roi_mode == "core":
            # Inner core: cuts off outer 15% (hair outline, ears, neck, and background)
            # Focuses purely on forehead and bilateral cheek zones
            rh, rw = face_roi.shape[:2]
            cy1, cy2 = int(0.12 * rh), int(0.85 * rh)
            cx1, cx2 = int(0.15 * rw), int(0.85 * rw)
            if cy2 > cy1 and cx2 > cx1:
                face_roi = face_roi[cy1:cy2, cx1:cx2]

        if apply_skin_mask and face_roi.size > 0:
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 20, 50]), np.array([25, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 20, 50]), np.array([180, 255, 255]))
            skin_mask = mask1 | mask2
            # Morphological smoothing to avoid patchy masks
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            if np.count_nonzero(skin_mask) > (face_roi.shape[0] * face_roi.shape[1] * 0.35):
                # Replace non-skin background/hair with average skin color to prevent high-frequency gradient edges
                skin_pixels = face_roi[skin_mask == 255]
                if len(skin_pixels) > 0:
                    mean_skin = np.mean(skin_pixels, axis=0).astype(np.uint8)
                    masked_roi = face_roi.copy()
                    masked_roi[skin_mask == 0] = mean_skin
                    face_roi = masked_roi

        return face_roi


class RealtimeRPPG:
    """Core real-time rPPG extraction engine."""

    def __init__(
        self,
        model_name: str = "EfficientPhys",
        dataset: str = "UBFC-rPPG",
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        use_fusion_stem: Optional[bool] = None,
        roi_mode: str = "core",
        img_size: int = 72,
        fs: float = 30.0,
        window_seconds: float = 8.0,
        update_hz: float = 4.0,
    ):
        self.model_name = "TS-CAN" if model_name.upper() in ["TS-CAN", "TSCAN"] else model_name
        self.dataset = dataset
        self.img_size = img_size
        self.fs = fs
        self.window_seconds = window_seconds
        self.window_samples = int(fs * window_seconds)
        self.update_interval = 1.0 / max(1.0, update_hz)
        self.last_update_time = 0.0
        self.roi_mode = roi_mode
        self.use_fusion_stem = use_fusion_stem

        # Device selection: MPS on Apple Silicon, CUDA on Nvidia, or CPU
        if device:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available() and self.model_name not in ["PhysNet", "ContrastPhys"]:
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Initialize motion-resilient tracker
        self.tracker = FaceTracker(roi_mode=self.roi_mode)

        # Buffers for real-time processing
        self.raw_faces = collections.deque(maxlen=self.window_samples)
        self.raw_signals = collections.deque(maxlen=self.window_samples)
        self.sample_timestamps = collections.deque(maxlen=self.window_samples)
        self.frame_times = collections.deque(maxlen=60)
        self.hr_history = collections.deque(maxlen=10)
        self.hr_candidates = collections.deque(maxlen=7)
        self.pending_large_jump_count = 0
        self.pending_jump_target = 0.0

        # EfficientPhys / TS-CAN temporal mini-buffer
        self.mini_buffer_depth = 10
        self.mini_face_buffer = collections.deque(maxlen=self.mini_buffer_depth + 1)
        self.prev_face_float: Optional[np.ndarray] = None

        # State outputs
        self.current_hr = 0.0
        self.smoothed_hr = 0.0
        self.current_snr = 0.0
        self.is_motion_artifact = False
        self.latest_bvp_signal = np.array([])
        self.latest_freqs = np.array([])
        self.latest_spectrum = np.array([])
        self.fps = 0.0
        self.last_latency_ms = 0.0
        self.weights_path: Optional[str] = None

        # Load specified model
        self.model: Optional[nn.Module] = None
        self.load_model(self.model_name, dataset=dataset, weights_path=weights_path)

    def load_model(self, model_name: str, dataset: str = "UBFC-rPPG", weights_path: Optional[str] = None):
        """Load or switch the evaluation model, with auto-detection for FusionStem & ContrastPhys."""
        norm_name = "TS-CAN" if model_name.upper() in ["TS-CAN", "TSCAN"] else model_name
        self.model_name = norm_name
        self.dataset = dataset

        if self.model_name in SUPPORTED_UNSUPERVISED_MODELS:
            self.model = None
            self.weights_path = None
            print(f"[RealtimeRPPG] Loaded unsupervised method: {self.model_name}")
            return

        if self.model_name not in SUPPORTED_NEURAL_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported: {ALL_SUPPORTED_MODELS}")

        # Resolve weights path
        if weights_path is None or not os.path.isfile(weights_path):
            weights_path = find_default_weights(self.model_name, dataset=self.dataset)
        if weights_path is None or not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"No pretrained weights found for model {self.model_name} (dataset: {self.dataset}). "
                f"Please specify --weights <path_to_pth>."
            )
        self.weights_path = weights_path

        state_dict = torch.load(self.weights_path, map_location=self.device)
        cleaned_sd = clean_state_dict(state_dict)

        # Auto-detect FusionStem from checkpoint keys if not explicitly configured
        has_fusion_stem = any("fusion_stem." in k for k in cleaned_sd.keys())
        use_fusion_stem = self.use_fusion_stem if self.use_fusion_stem is not None else has_fusion_stem

        # Neural model architecture construction
        if self.model_name == "DeepPhys":
            net = DeepPhys(img_size=self.img_size, use_fusion_stem=use_fusion_stem).to(self.device)
        elif self.model_name == "EfficientPhys":
            net = EfficientPhys(frame_depth=self.mini_buffer_depth, img_size=self.img_size).to(self.device)
        elif self.model_name in ["TS-CAN", "TSCAN"]:
            net = TSCAN(frame_depth=self.mini_buffer_depth, img_size=self.img_size, use_fusion_stem=use_fusion_stem).to(self.device)
        elif self.model_name == "ContrastPhys":
            net = ContrastPhysNet(frames=128, S=2, use_fusion_stem=use_fusion_stem).to(self.device)
        elif self.model_name == "PhysNet":
            net = PhysNet_padding_Encoder_Decoder_MAX(frames=128, use_fusion_stem=use_fusion_stem).to(self.device)
        else:
            raise ValueError(f"Unknown neural model: {self.model_name}")

        net.load_state_dict(cleaned_sd)
        net.eval()
        self.model = net
        fs_tag = " (with FusionStem)" if use_fusion_stem else ""
        print(f"[RealtimeRPPG] Loaded {self.model_name}{fs_tag} on {self.device} from: {os.path.basename(self.weights_path)}")

    def reset_buffers(self):
        """Clear all historical buffers."""
        self.raw_faces.clear()
        self.raw_signals.clear()
        self.sample_timestamps.clear()
        self.mini_face_buffer.clear()
        self.prev_face_float = None
        self.hr_history.clear()
        self.hr_candidates.clear()
        self.pending_large_jump_count = 0
        self.pending_jump_target = 0.0
        self.current_hr = 0.0
        self.smoothed_hr = 0.0
        self.current_snr = 0.0
        self.is_motion_artifact = False

    def process_frame(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Process a single incoming video/camera frame with motion artifact handling."""
        now = time.time() if timestamp is None else timestamp
        self.frame_times.append(now)
        if len(self.frame_times) >= 2:
            dt = self.frame_times[-1] - self.frame_times[0]
            if dt > 0:
                self.fps = (len(self.frame_times) - 1) / dt

        # 1. Face detection & Inner Core ROI cropping
        box = self.tracker.detect_and_smooth(frame_bgr)
        face_crop_rgb: Optional[np.ndarray] = None

        if box is not None:
            face_roi = self.tracker.extract_roi(frame_bgr, box, apply_skin_mask=True)
            if face_roi.size > 0:
                face_resized = cv2.resize(face_roi, (self.img_size, self.img_size))
                face_crop_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                self.raw_faces.append(face_crop_rgb)

        # 2. Model inference & Motion Artifact Detection
        t_start = time.perf_counter()
        if face_crop_rgb is not None:
            self.sample_timestamps.append(now)
            pulse_val = self._step_inference(face_crop_rgb)
            if pulse_val is not None:
                # If motion artifact is active, softly damp pulse addition
                if self.is_motion_artifact and len(self.raw_signals) > 0:
                    pulse_val = 0.5 * pulse_val + 0.5 * self.raw_signals[-1]
                self.raw_signals.append(pulse_val)
        self.last_latency_ms = (time.perf_counter() - t_start) * 1000.0

        # 3. Periodic Heart Rate calculation
        if (now - self.last_update_time) >= self.update_interval:
            self._update_heart_rate()
            self.last_update_time = now

        return {
            "face_box": box,
            "face_crop": face_crop_rgb,
            "hr": self.current_hr,
            "hr_smoothed": self.smoothed_hr,
            "snr": self.current_snr,
            "is_motion_artifact": self.is_motion_artifact,
            "bvp_signal": self.latest_bvp_signal,
            "freqs": self.latest_freqs,
            "spectrum": self.latest_spectrum,
            "fps": self.fps,
            "latency_ms": self.last_latency_ms,
            "buffer_fill_pct": min(100.0, 100.0 * len(self.raw_signals) / max(1, self.window_samples)),
            "model_name": self.model_name,
            "device": str(self.device) if self.model else "CPU",
        }

    def _step_inference(self, face_rgb: np.ndarray) -> Optional[float]:
        """Compute single-step pulse point depending on model type."""
        face_float = face_rgb.astype(np.float32)

        # Measure inter-frame motion energy
        if self.prev_face_float is not None:
            motion_energy = float(np.mean(np.abs(face_float - self.prev_face_float)))
            self.is_motion_artifact = (motion_energy > 18.0)  # Threshold for rapid head displacement
        else:
            self.is_motion_artifact = False

        # Unsupervised methods operate in rolling batches during _update_heart_rate
        if self.model is None:
            self.prev_face_float = face_float
            return float(np.mean(face_float[:, :, 1]))  # Green channel tracking as proxy

        # Neural models:
        if self.model_name == "DeepPhys":
            if self.prev_face_float is None:
                self.prev_face_float = face_float
                return 0.0

            # Normalized difference frame
            diff = (face_float - self.prev_face_float) / (face_float + self.prev_face_float + 1e-7)
            diff_std = np.std(diff) + 1e-7
            diff_norm = diff / diff_std

            # DeepPhys input: 6 channels (Diff, Raw)
            feat = np.concatenate([diff_norm, face_float], axis=-1)  # (H, W, 6)
            feat_t = torch.from_numpy(feat).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                pred = self.model(feat_t)
            self.prev_face_float = face_float
            return float(pred.squeeze().item())

        elif self.model_name in ["EfficientPhys", "TS-CAN", "TSCAN"]:
            self.mini_face_buffer.append(face_float)
            self.prev_face_float = face_float
            if len(self.mini_face_buffer) < self.mini_buffer_depth + 1:
                return 0.0

            mini_faces = np.array(self.mini_face_buffer)  # (11, H, W, 3)
            if self.model_name == "EfficientPhys":
                std_faces = (mini_faces - np.mean(mini_faces)) / (np.std(mini_faces) + 1e-7)
                tensor_in = torch.from_numpy(std_faces).permute(0, 3, 1, 2).float().to(self.device)
                with torch.no_grad():
                    out = self.model(tensor_in)  # (10, 1)
                return float(out[-1].squeeze().item())
            else:
                # TS-CAN
                diff_frames = np.zeros((self.mini_buffer_depth, self.img_size, self.img_size, 3), dtype=np.float32)
                for j in range(self.mini_buffer_depth):
                    diff_frames[j] = (mini_faces[j + 1] - mini_faces[j]) / (
                        mini_faces[j + 1] + mini_faces[j] + 1e-7
                    )
                diff_frames = diff_frames / (np.std(diff_frames) + 1e-7)
                raw_frames = mini_faces[1:]
                cat_frames = np.concatenate([diff_frames, raw_frames], axis=-1)  # (10, H, W, 6)
                tensor_in = torch.from_numpy(cat_frames).permute(0, 3, 1, 2).float().to(self.device)
                with torch.no_grad():
                    out = self.model(tensor_in)
                return float(out[-1].squeeze().item())

        elif self.model_name in ["ContrastPhys", "PhysNet"]:
            self.prev_face_float = face_float
            # Accumulate 128 frames for 3D convolution
            if len(self.raw_faces) < 128:
                return 0.0
            clip = np.array(list(self.raw_faces)[-128:], dtype=np.float32)

            if self.model_name == "ContrastPhys":
                # ContrastPhys expects [1, C, T, H, W]
                tensor_in = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).float().to(self.device)
                with torch.no_grad():
                    out = self.model(tensor_in)
                    rppg = out[0] if isinstance(out, (tuple, list)) else out
                return float(rppg.squeeze()[-1].item())
            else:
                # PhysNet
                diff = np.zeros_like(clip)
                for j in range(127):
                    diff[j] = (clip[j + 1] - clip[j]) / (clip[j + 1] + clip[j] + 1e-7)
                diff = diff / (np.std(diff) + 1e-7)
                tensor_in = torch.from_numpy(diff).permute(3, 0, 1, 2).unsqueeze(0).float().to(self.device)
                with torch.no_grad():
                    out = self.model(tensor_in)
                    if isinstance(out, tuple):
                        out = out[0]
                return float(out.squeeze()[-1].item())

        return 0.0

    def _update_heart_rate(self):
        """Compute BVP filtering, FFT spectrum, and heart rate estimate with continuity tracking & anti-jitter gating."""
        # Require at least 5.0 seconds of signal for solid frequency resolution before estimating HR
        min_samples = int(self.fs * 5.0)
        if self.model is None:
            if len(self.raw_faces) < min_samples:
                return
            faces = np.array(self.raw_faces)
            if self.model_name == "POS":
                bvp = POS_WANG(faces, self.fs)
            elif self.model_name == "CHROM":
                bvp = CHROME_DEHAAN(faces, self.fs)
            elif self.model_name == "GREEN":
                bvp = GREEN(faces)
            elif self.model_name == "ICA":
                bvp = ICA_POH(faces, self.fs)
            else:
                bvp = faces[:, :, :, 1].mean(axis=(1, 2))
            signal_to_process = np.squeeze(bvp)
        else:
            if len(self.raw_signals) < min_samples:
                return
            raw_sig = np.array(self.raw_signals)
            if self.model_name == "ContrastPhys":
                signal_to_process = raw_sig
            else:
                signal_to_process = np.cumsum(raw_sig)

        if len(signal_to_process) < min_samples:
            return

        # 1. Determine actual effective sampling rate from real frame timestamps
        effective_fs = self.fs
        if len(self.sample_timestamps) >= 30:
            ts_list = list(self.sample_timestamps)
            dt_total = ts_list[-1] - ts_list[0]
            if dt_total > 1.0:
                measured_fs = (len(ts_list) - 1) / dt_total
                if 12.0 <= measured_fs <= 120.0:
                    effective_fs = measured_fs

        # 2. Resample signal onto a strictly uniform temporal grid to eliminate frame delivery jitter
        if len(self.sample_timestamps) >= len(signal_to_process):
            t_raw = np.array(list(self.sample_timestamps)[-len(signal_to_process):])
            if t_raw[-1] > t_raw[0]:
                t_uniform = np.linspace(t_raw[0], t_raw[-1], len(signal_to_process))
                signal_to_process = np.interp(t_uniform, t_raw, signal_to_process)

        # 3. Detrending
        try:
            detrended = _detrend(signal_to_process, lambda_value=100)
        except Exception:
            detrended = scipy.signal.detrend(signal_to_process)

        # 4. Butterworth bandpass filter [0.75, 2.5] Hz (45 - 150 BPM)
        nyq = 0.5 * effective_fs
        low = max(0.01, 0.75 / nyq)
        high = min(0.99, 2.5 / nyq)
        b, a = scipy.signal.butter(1, [low, high], btype="bandpass")
        filtered = scipy.signal.filtfilt(b, a, np.double(detrended))
        self.latest_bvp_signal = filtered

        # 5. High-resolution FFT periodogram
        L = len(filtered)
        N = max(1024, _next_power_of_2(L) * 2)
        f_ppg, pxx_ppg = scipy.signal.periodogram(filtered, fs=effective_fs, nfft=N, detrend=False, window="hann")
        fmask = (f_ppg >= 0.75) & (f_ppg <= 2.5)
        freqs_bpm = f_ppg[fmask] * 60.0  # Converted to BPM
        spectrum = pxx_ppg[fmask]
        self.latest_freqs = freqs_bpm
        self.latest_spectrum = spectrum

        if len(spectrum) == 0 or np.max(spectrum) <= 0:
            return

        # 6. Peak Finding with Continuity Prior & Harmonic Awareness
        peaks, _ = scipy.signal.find_peaks(spectrum, height=np.max(spectrum) * 0.08)
        if len(peaks) == 0:
            best_hr = float(freqs_bpm[np.argmax(spectrum)])
        else:
            peak_hrs = freqs_bpm[peaks]
            peak_powers = spectrum[peaks]

            # If we already have an established stable HR, apply Gaussian continuity prior
            if self.smoothed_hr > 40.0:
                scores = []
                for hr_cand, pwr in zip(peak_hrs, peak_powers):
                    # Harmonic suppression: if candidate is near 2x previous HR, heavily discount it
                    if abs(hr_cand - 2.0 * self.smoothed_hr) < 12.0:
                        pwr_mod = pwr * 0.35
                    # Subharmonic suppression: if candidate is near 0.5x previous HR, discount it
                    elif abs(hr_cand - 0.5 * self.smoothed_hr) < 10.0:
                        pwr_mod = pwr * 0.30
                    else:
                        pwr_mod = pwr

                    # Gaussian continuity prior centered at current smoothed HR with sigma = 18 BPM
                    delta_bpm = abs(hr_cand - self.smoothed_hr)
                    prior = math.exp(-(delta_bpm ** 2) / (2.0 * (18.0 ** 2)))
                    scores.append(pwr_mod * (0.30 + 0.70 * prior))

                best_idx = int(np.argmax(scores))
                best_hr = float(peak_hrs[best_idx])
            else:
                # Initial acquisition: pick the highest spectral peak
                best_idx = int(np.argmax(peak_powers))
                best_hr = float(peak_hrs[best_idx])

        # 7. Compute SNR around the selected peak
        peak_idx = int(np.argmin(np.abs(freqs_bpm - best_hr)))
        peak_pow = spectrum[peak_idx]
        mean_pow = float(np.mean(spectrum) + 1e-7)
        self.current_snr = float(10.0 * np.log10(max(1e-6, peak_pow) / mean_pow))

        # 8. Physiological Gating & Rolling Median Smoothing (Anti-Jitter)
        if 45.0 <= best_hr <= 160.0:
            self.hr_candidates.append(best_hr)
            med_hr = float(np.median(self.hr_candidates))

            if self.smoothed_hr > 40.0:
                jump = abs(med_hr - self.smoothed_hr)
                if jump > 14.0:
                    # Gating: Large jump (> 14 BPM) must persist across multiple windows before accepting
                    if abs(med_hr - self.pending_jump_target) < 6.0:
                        self.pending_large_jump_count += 1
                    else:
                        self.pending_jump_target = med_hr
                        self.pending_large_jump_count = 1

                    if self.pending_large_jump_count >= 4:
                        # Confirmed sustained physiological shift
                        alpha = 0.20
                        self.smoothed_hr = alpha * med_hr + (1.0 - alpha) * self.smoothed_hr
                        self.pending_large_jump_count = 0
                    else:
                        # Hold previous estimate, with only micro-drift
                        self.smoothed_hr = 0.05 * med_hr + 0.95 * self.smoothed_hr
                else:
                    self.pending_large_jump_count = 0
                    alpha = 0.10 if self.is_motion_artifact else 0.18
                    self.smoothed_hr = alpha * med_hr + (1.0 - alpha) * self.smoothed_hr
            else:
                # Initial seed
                self.smoothed_hr = med_hr

            self.current_hr = round(self.smoothed_hr, 1)
            self.hr_history.append(self.current_hr)
