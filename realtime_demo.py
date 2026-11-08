#!/usr/bin/env python3
"""Interactive Real-time Remote PPG Heart Rate Evaluation Tool.

Stream live from a webcam or evaluate any pre-recorded video file in real time using:
- Supervised Deep Learning models: EfficientPhys, DeepPhys, TS-CAN, PhysNet
- Unsupervised algorithms: POS, CHROM, GREEN, ICA

Usage Examples:
  # 1. Live Webcam with EfficientPhys (Default)
  python realtime_demo.py --camera 0

  # 2. Live Webcam with DeepPhys on Apple Silicon MPS
  python realtime_demo.py --camera 0 --model DeepPhys

  # 3. Test on an input video file
  python realtime_demo.py --video PURE_videos/01-01.mp4 --model POS

  # 4. Headless execution saving annotated video & signal CSV
  python realtime_demo.py --video PURE_videos/01-01.mp4 --headless --max_frames 300 --save_signal results.csv
"""

import os
import sys
import time
import argparse
import csv
import math
from typing import Optional

# Prevent OpenCL / MPS driver crashes
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np

# Ensure repository root is on Python module path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.realtime_evaluator import (
    RealtimeRPPG,
    ALL_SUPPORTED_MODELS,
    SUPPORTED_NEURAL_MODELS,
    SUPPORTED_UNSUPERVISED_MODELS,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time rPPG Heart Rate Evaluation using Camera or Video File"
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to an input video file (e.g. .mp4, .avi, .mov).",
    )
    input_group.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera device index (default: 0 if --video is not specified).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EfficientPhys",
        choices=ALL_SUPPORTED_MODELS,
        help="rPPG model to use for inference. (Default: EfficientPhys)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional custom checkpoint (.pth) path. If omitted, uses final_model_release/.",
    )
    parser.add_argument(
        "--roi_mode",
        type=str,
        default="core",
        choices=["core", "full"],
        help="ROI mode: 'core' (inner forehead & cheeks, suppresses motion edges) or 'full'. (Default: core)",
    )
    parser.add_argument(
        "--use_fusion_stem",
        action="store_true",
        help="Explicitly enable modular FusionStem (auto-detected if checkpoint contains fusion_stem).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="UBFC-rPPG",
        help="Pretrained dataset weights to load for neural models (e.g. UBFC-rPPG, PURE, SCAMPS).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (mps, cuda, cpu). Auto-detected by default.",
    )
    parser.add_argument(
        "--window_seconds",
        type=float,
        default=8.0,
        help="Length of sliding temporal window in seconds for FFT HR computation. (Default: 8.0)",
    )
    parser.add_argument(
        "--update_hz",
        type=float,
        default=4.0,
        help="Rate (in Hz) to recalculate the heart rate estimate. (Default: 4.0)",
    )
    parser.add_argument(
        "--save_video",
        type=str,
        default=None,
        help="Optional path to save annotated visualization output video (e.g. out.mp4).",
    )
    parser.add_argument(
        "--save_signal",
        type=str,
        default=None,
        help="Optional path to export estimated BVP signals and HR timeseries to a CSV file.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying an OpenCV GUI window (ideal for servers or automated test runs).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Stop processing after this number of frames.",
    )
    return parser.parse_args()


class HUDVisualizer:
    """Renders a modern, real-time medical telemetry HUD overlay."""

    def __init__(self, target_h: int = 540, hud_width: int = 380):
        self.target_h = target_h
        self.hud_width = hud_width
        self.pulse_phase = 0.0

    def draw_hud(
        self,
        main_frame_bgr: np.ndarray,
        status: dict,
    ) -> np.ndarray:
        # Resize main frame to standard height maintaining aspect ratio
        h, w = main_frame_bgr.shape[:2]
        new_w = int(w * (self.target_h / h))
        video_resized = cv2.resize(main_frame_bgr, (new_w, self.target_h))

        # Draw smoothed face bounding box on video feed
        box = status.get("face_box")
        if box is not None:
            bx, by, bw, bh = box
            scale_x = new_w / w
            scale_y = self.target_h / h
            vx = int(bx * scale_x)
            vy = int(by * scale_y)
            vw = int(bw * scale_x)
            vh = int(bh * scale_y)
            color = (0, 255, 128) if status.get("hr_smoothed", 0) > 0 else (0, 215, 255)
            cv2.rectangle(video_resized, (vx, vy), (vx + vw, vy + vh), color, 2)
            cv2.putText(
                video_resized,
                "ROI LOCKED",
                (vx, max(20, vy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        # Create Dark Theme Telemetry Panel
        panel = np.zeros((self.target_h, self.hud_width, 3), dtype=np.uint8)
        panel[:] = (24, 24, 28)  # Deep Charcoal Background

        # Title Header
        cv2.putText(panel, "rPPG REALTIME MONITOR", (16, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        model_name = status.get("model_name", "N/A")
        device_name = status.get("device", "CPU")
        cv2.putText(panel, f"Model: {model_name} [{device_name}]", (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.line(panel, (16, 64), (self.hud_width - 16, 64), (60, 60, 65), 1)

        # Heart Rate Display & Animated Pulsating Heart
        hr_val = status.get("hr_smoothed", 0.0)
        buffer_pct = status.get("buffer_fill_pct", 0.0)

        # Update heart pulse animation
        if hr_val > 0:
            bpm_freq = hr_val / 60.0
            self.pulse_phase += 0.25 * bpm_freq
            pulse_scale = 1.0 + 0.15 * math.sin(self.pulse_phase * 2 * math.pi)
        else:
            pulse_scale = 1.0

        # Heart indicator icon
        hx, hy = 40, 115
        base_r = int(12 * pulse_scale)
        heart_col = (40, 40, 240) if hr_val > 0 else (100, 100, 100)
        cv2.circle(panel, (hx - base_r // 2, hy), base_r, heart_col, -1)
        cv2.circle(panel, (hx + base_r // 2, hy), base_r, heart_col, -1)
        pts = np.array([[hx - base_r * 1.4, hy + 2], [hx + base_r * 1.4, hy + 2], [hx, hy + base_r * 2]], np.int32)
        cv2.fillPoly(panel, [pts], heart_col)

        # Heart Rate Value Text
        if status.get("is_motion_artifact", False):
            hr_str = f"{hr_val:5.1f}" if hr_val > 0 else "--.-"
            unit_str = "BPM"
            stat_color = (0, 165, 255)  # Orange
            status_tag = "MOTION DAMPED"
        elif hr_val > 0 and buffer_pct >= 50.0:
            hr_str = f"{hr_val:5.1f}"
            unit_str = "BPM"
            stat_color = (0, 255, 128)  # Emerald green
            status_tag = "LOCKED"
        else:
            hr_str = "--.-"
            unit_str = "BPM"
            stat_color = (0, 215, 255)  # Amber
            status_tag = f"ACQUIRING ({buffer_pct:.0f}%)"

        cv2.putText(panel, hr_str, (88, 125), cv2.FONT_HERSHEY_DUPLEX, 1.35, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, unit_str, (215, 123), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1, cv2.LINE_AA)

        # Status Tag Badge
        cv2.putText(panel, f"STATUS: {status_tag}", (16, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, stat_color, 1, cv2.LINE_AA)
        snr_val = status.get("snr", 0.0)
        cv2.putText(panel, f"SNR: {snr_val:+.1f} dB", (220, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.line(panel, (16, 168), (self.hud_width - 16, 168), (60, 60, 65), 1)

        # Live Face ROI thumbnail
        face_crop = status.get("face_crop")
        if face_crop is not None:
            thumb_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            thumb_sized = cv2.resize(thumb_bgr, (60, 60))
            panel[180:240, 16:76] = thumb_sized
            cv2.rectangle(panel, (16, 180), (76, 240), (80, 80, 90), 1)
            cv2.putText(panel, "ROI Preview", (88, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(panel, "72x72 px", (88, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)

        # Waveform Oscilloscope Plot
        cv2.putText(panel, "BLOOD VOLUME PULSE (rPPG)", (16, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        wave_top = 275
        wave_height = 90
        wave_w = self.hud_width - 32
        cv2.rectangle(panel, (16, wave_top), (16 + wave_w, wave_top + wave_height), (35, 35, 42), -1)
        cv2.line(panel, (16, wave_top + wave_height // 2), (16 + wave_w, wave_top + wave_height // 2), (50, 50, 60), 1)

        bvp = status.get("bvp_signal", np.array([]))
        if len(bvp) > 10:
            recent_bvp = bvp[-150:]
            b_min, b_max = np.min(recent_bvp), np.max(recent_bvp)
            rng = max(1e-5, b_max - b_min)
            norm_bvp = (recent_bvp - b_min) / rng
            pts = []
            for idx, val in enumerate(norm_bvp):
                px = int(16 + idx * (wave_w / max(1, len(norm_bvp) - 1)))
                py = int(wave_top + wave_height - (val * (wave_height - 12) + 6))
                pts.append([px, py])
            pts = np.array(pts, np.int32)
            cv2.polylines(panel, [pts], False, (0, 235, 255), 2, cv2.LINE_AA)

        # FFT Power Spectrum Graph
        cv2.putText(panel, "FFT SPECTRUM (BPM)", (16, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        spec_top = 395
        spec_height = 65
        cv2.rectangle(panel, (16, spec_top), (16 + wave_w, spec_top + spec_height), (35, 35, 42), -1)

        freqs = status.get("freqs", np.array([]))
        spectrum = status.get("spectrum", np.array([]))
        if len(spectrum) > 5 and len(freqs) == len(spectrum):
            spec_max = np.max(spectrum)
            if spec_max > 1e-6:
                norm_spec = spectrum / spec_max
                pts_spec = []
                for idx, (f_bpm, s_val) in enumerate(zip(freqs, norm_spec)):
                    # map f_bpm (45 to 150) to width
                    px = int(16 + ((f_bpm - 45.0) / 105.0) * wave_w)
                    px = max(16, min(16 + wave_w, px))
                    py = int(spec_top + spec_height - (s_val * (spec_height - 8) + 4))
                    pts_spec.append([px, py])
                pts_spec = np.array(pts_spec, np.int32)
                cv2.polylines(panel, [pts_spec], False, (255, 128, 0), 1, cv2.LINE_AA)

                # Mark Peak Frequency
                peak_idx = int(np.argmax(spectrum))
                peak_x = int(16 + ((freqs[peak_idx] - 45.0) / 105.0) * wave_w)
                cv2.circle(panel, (peak_x, pts_spec[peak_idx][1]), 3, (0, 255, 255), -1)

        # Footer Metrics: FPS, Latency & Hotkeys Guide
        cv2.line(panel, (16, 475), (self.hud_width - 16, 475), (60, 60, 65), 1)
        fps = status.get("fps", 0.0)
        lat = status.get("latency_ms", 0.0)
        cv2.putText(panel, f"FPS: {fps:4.1f} | Latency: {lat:4.1f} ms", (16, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(panel, "[q] Quit | [m] Next Model | [r] Reset", (16, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1, cv2.LINE_AA)

        # Stitch main video and telemetry panel horizontally
        combined = np.hstack([video_resized, panel])
        return combined


def run():
    args = parse_args()

    # Determine input source
    if args.video is not None:
        source_desc = f"Video: {args.video}"
        cap = cv2.VideoCapture(args.video)
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        fs = input_fps if (input_fps and 10.0 <= input_fps <= 120.0) else 30.0
    else:
        cam_idx = args.camera if args.camera is not None else 0
        source_desc = f"Camera #{cam_idx}"
        cap = cv2.VideoCapture(cam_idx)
        # Request 30 fps
        cap.set(cv2.CAP_PROP_FPS, 30)
        fs = 30.0

    if not cap.isOpened():
        print(f"❌ Error: Could not open video/camera source: {source_desc}")
        sys.exit(1)

    print(f"\n========================================================")
    print(f"🎬 Starting rPPG Real-Time Evaluation")
    print(f"   Source:     {source_desc}")
    print(f"   Model:      {args.model}")
    print(f"   Sampling:   {fs:.1f} FPS")
    print(f"   Window:     {args.window_seconds:.1f} s")
    print(f"   Headless:   {args.headless}")
    print(f"========================================================\n")

    # Initialize rPPG engine
    evaluator = RealtimeRPPG(
        model_name=args.model,
        dataset=args.dataset,
        weights_path=args.weights,
        device=args.device,
        use_fusion_stem=args.use_fusion_stem if args.use_fusion_stem else None,
        roi_mode=args.roi_mode,
        fs=fs,
        window_seconds=args.window_seconds,
        update_hz=args.update_hz,
    )

    visualizer = HUDVisualizer()

    # Video Writer if requested
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Probe frame dimension for HUD
        ret, probe_frame = cap.read()
        if ret:
            sample_status = evaluator.process_frame(probe_frame)
            hud_sample = visualizer.draw_hud(probe_frame, sample_status)
            out_h, out_w = hud_sample.shape[:2]
            writer = cv2.VideoWriter(args.save_video, fourcc, fs, (out_w, out_h))
            writer.write(hud_sample)
            print(f"💾 Saving annotated output video to: {args.save_video}")
        # Reset position if reading file
        if args.video is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            evaluator.reset_buffers()

    # CSV Signal Writer if requested
    csv_file = None
    csv_writer = None
    if args.save_signal:
        csv_file = open(args.save_signal, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame_idx", "timestamp_sec", "model", "instant_hr", "smoothed_hr", "snr_db"])
        print(f"💾 Exporting signal time-series to: {args.save_signal}")

    frame_count = 0
    t0 = time.time()
    model_idx = ALL_SUPPORTED_MODELS.index(evaluator.model_name)

    window_name = f"rPPG Realtime Evaluator - {evaluator.model_name}"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nReached end of video stream.")
                break

            frame_count += 1
            now = time.time()

            # Process frame through rPPG pipeline
            status = evaluator.process_frame(frame, timestamp=now)

            # Export to CSV if enabled
            if csv_writer:
                csv_writer.writerow([
                    frame_count,
                    f"{now - t0:.3f}",
                    status["model_name"],
                    f"{status['hr']:.2f}",
                    f"{status['hr_smoothed']:.2f}",
                    f"{status['snr']:.2f}",
                ])

            # Render HUD visualization
            hud_frame = visualizer.draw_hud(frame, status)

            if writer is not None:
                writer.write(hud_frame)

            # Console output in headless mode or periodic log
            if args.headless:
                if frame_count % 30 == 0:
                    print(
                        f"Frame {frame_count:4d} | HR: {status['hr_smoothed']:5.1f} BPM | "
                        f"SNR: {status['snr']:+4.1f} dB | FPS: {status['fps']:4.1f} | Model: {status['model_name']}"
                    )
            else:
                cv2.imshow(window_name, hud_frame)
                key = cv2.waitKey(1) & 0xFF

                # [q]: Quit
                if key == ord("q") or key == 27:
                    print("\nUser requested exit.")
                    break
                # [m]: Cycle to next model
                elif key == ord("m"):
                    model_idx = (model_idx + 1) % len(ALL_SUPPORTED_MODELS)
                    next_model = ALL_SUPPORTED_MODELS[model_idx]
                    print(f"\n🔄 Switching model to: {next_model}...")
                    try:
                        evaluator.load_model(next_model, dataset=args.dataset)
                        evaluator.reset_buffers()
                        cv2.setWindowTitle(window_name, f"rPPG Realtime Evaluator - {next_model}")
                    except Exception as e:
                        print(f"Could not load {next_model}: {e}")
                # [r]: Reset historical buffer
                elif key == ord("r"):
                    evaluator.reset_buffers()
                    print("\nSignal buffers reset.")
                # [s]: Save snapshot
                elif key == ord("s"):
                    snap_name = f"snapshot_frame_{frame_count}.png"
                    cv2.imwrite(snap_name, hud_frame)
                    print(f"\n📸 Snapshot saved to {snap_name}")

            if args.max_frames and frame_count >= args.max_frames:
                print(f"\nReached max frames limit ({args.max_frames}).")
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        if not args.headless:
            cv2.destroyAllWindows()

    total_time = time.time() - t0
    print("\n--------------------------------------------------------")
    print(f"✅ Evaluation Complete:")
    print(f"   Processed Frames:   {frame_count}")
    print(f"   Elapsed Time:       {total_time:.2f} s ({frame_count / max(0.01, total_time):.1f} avg FPS)")
    print(f"   Final Heart Rate:   {evaluator.smoothed_hr:.1f} BPM")
    print(f"--------------------------------------------------------\n")


if __name__ == "__main__":
    run()
