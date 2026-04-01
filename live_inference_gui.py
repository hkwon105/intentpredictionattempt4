"""
Live Inference GUI — CNN-LSTM Intent Prediction
RealSense camera + trained model -> STOP signal on INTERACTION detection
"""

import argparse
import collections
import json
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import font as tkfont

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
from torchvision import transforms

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[WARN] pyrealsense2 not found — falling back to webcam (cv2)")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import generate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Live Intent Prediction GUI")
    parser.add_argument("--resume_path", required=True)
    parser.add_argument("--annotation_path", default="./data/annotation/ucf101_01.json")
    parser.add_argument("--n_classes", type=int, default=3)
    parser.add_argument("--sample_size", type=int, default=150)
    parser.add_argument("--sample_duration", type=int, default=16)
    parser.add_argument("--confidence_threshold", type=float, default=0.6)
    parser.add_argument("--smoothing_window", type=int, default=5)
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def load_class_labels(annotation_path):
    with open(annotation_path, "r") as f:
        data = json.load(f)
    return data["labels"]


def build_transform(sample_size):
    mean = [114.7748, 107.7354, 99.4750]
    std  = [1.0, 1.0, 1.0]
    return transforms.Compose([
        transforms.Resize((sample_size, sample_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
        transforms.Normalize(mean, std),
    ])


def load_model(args, device):
    model = generate_model(args, device)
    checkpoint = torch.load(args.resume_path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    model.resnet.eval()
    return model


class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(cfg)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color  = frames.get_color_frame()
        if not color:
            return False, None
        return True, np.asanyarray(color.get_data())

    def release(self):
        self.pipeline.stop()


class WebcamCamera:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class InferenceEngine:
    def __init__(self, model, transform, class_labels, args, device):
        self.model        = model
        self.transform    = transform
        self.class_labels = class_labels
        self.sample_dur   = args.sample_duration
        self.conf_thresh  = args.confidence_threshold
        self.smooth_win   = args.smoothing_window
        self.device       = device

        self.frame_buffer = collections.deque(maxlen=self.sample_dur)
        self.pred_history = collections.deque(maxlen=self.smooth_win)

        self.lock             = threading.Lock()
        self.latest_probs     = np.zeros(len(class_labels))
        self.latest_label     = class_labels[-1]
        self.latest_conf      = 0.0
        self.stop_signal      = False
        self.fps              = 0.0
        self._last_infer_time = time.time()

    def push_frame(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        t   = self.transform(pil)
        self.frame_buffer.append(t)
        if len(self.frame_buffer) == self.sample_dur:
            self._infer()

    def _infer(self):
        clip = torch.stack(list(self.frame_buffer))
        clip = clip.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(clip)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        self.pred_history.append(pred_idx)

        # Weighted vote — recent predictions count more
        weights = np.linspace(0.5, 1.0, len(self.pred_history))
        votes   = np.zeros(len(self.class_labels))
        for w, idx in zip(weights, self.pred_history):
            votes[idx] += w
        smoothed_idx = int(np.argmax(votes))

        label = self.class_labels[smoothed_idx]
        conf  = float(probs[smoothed_idx])
        stop  = (label == "INTERACTION") and (conf >= self.conf_thresh)

        now  = time.time()
        fps  = 1.0 / max(now - self._last_infer_time, 1e-6)
        self._last_infer_time = now

        with self.lock:
            self.latest_probs = probs
            self.latest_label = label
            self.latest_conf  = conf
            self.stop_signal  = stop
            self.fps          = fps

    def get_state(self):
        with self.lock:
            return {
                "probs": self.latest_probs.copy(),
                "label": self.latest_label,
                "conf":  self.latest_conf,
                "stop":  self.stop_signal,
                "fps":   self.fps,
            }


class App:
    BG         = "#0d0f14"
    PANEL_BG   = "#13161e"
    ACCENT     = "#e8f0fe"
    STOP_RED   = "#ff3b30"
    SAFE_GREEN = "#34c759"
    WARN_AMBER = "#ffd60a"
    BAR_BG     = "#1e2230"
    TEXT_DIM   = "#5a6070"
    WHITE      = "#f0f4ff"

    CLASS_COLORS = {
        "INTERACTION": "#ff3b30",
        "PASSTHRU":    "#ffd60a",
        "WAIT":        "#34c759",
    }

    def __init__(self, root, engine, class_labels, camera):
        self.root         = root
        self.engine       = engine
        self.class_labels = class_labels
        self.camera       = camera
        self.running      = True
        self._flash_on    = False
        self._flash_after = None
        self._build_ui()
        self._start_camera_thread()
        self._update_loop()

    def _build_ui(self):
        self.root.title("Intent Prediction — Safety Monitor")
        self.root.configure(bg=self.BG)
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)

        try:
            title      = tkfont.Font(family="JetBrains Mono", size=28, weight="bold")
            label_font = tkfont.Font(family="JetBrains Mono", size=13, weight="bold")
            small_font = tkfont.Font(family="JetBrains Mono", size=10)
        except Exception:
            title      = tkfont.Font(family="Courier", size=28, weight="bold")
            label_font = tkfont.Font(family="Courier", size=13, weight="bold")
            small_font = tkfont.Font(family="Courier", size=10)

        top = tk.Frame(self.root, bg=self.BG, pady=12)
        top.pack(fill="x", padx=20)
        tk.Label(top, text="INTENT MONITOR", font=title,
                 bg=self.BG, fg=self.WHITE).pack(side="left")
        self.fps_label = tk.Label(top, text="FPS: —", font=small_font,
                                  bg=self.BG, fg=self.TEXT_DIM)
        self.fps_label.pack(side="right", padx=10)
        self.time_label = tk.Label(top, text="", font=small_font,
                                   bg=self.BG, fg=self.TEXT_DIM)
        self.time_label.pack(side="right", padx=10)

        content = tk.Frame(self.root, bg=self.BG)
        content.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        cam_frame = tk.Frame(content, bg=self.PANEL_BG,
                             highlightbackground=self.BAR_BG, highlightthickness=1)
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.cam_label = tk.Label(cam_frame, bg="#000000")
        self.cam_label.pack(fill="both", expand=True, padx=2, pady=2)

        right = tk.Frame(content, bg=self.BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=0)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=0)
        right.columnconfigure(0, weight=1)

        self.banner_frame = tk.Frame(right, bg=self.SAFE_GREEN, height=110, pady=10)
        self.banner_frame.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.banner_frame.pack_propagate(False)
        self.banner_label = tk.Label(self.banner_frame, text="● SAFE",
                                     font=tkfont.Font(family="Courier", size=32, weight="bold"),
                                     bg=self.SAFE_GREEN, fg="#0a2e14")
        self.banner_label.pack(expand=True)
        self.banner_sub = tk.Label(self.banner_frame, text="No interaction detected",
                                   font=small_font, bg=self.SAFE_GREEN, fg="#1a5c2e")
        self.banner_sub.pack()

        bars_outer = tk.Frame(right, bg=self.PANEL_BG,
                              highlightbackground=self.BAR_BG,
                              highlightthickness=1, pady=16, padx=16)
        bars_outer.grid(row=1, column=0, sticky="nsew", pady=(0, 12))
        tk.Label(bars_outer, text="CONFIDENCE", font=small_font,
                 bg=self.PANEL_BG, fg=self.TEXT_DIM).pack(anchor="w", pady=(0, 12))

        self.bar_labels   = {}
        self.bar_canvases = {}
        self.bar_fills    = {}

        for cls in self.class_labels:
            row   = tk.Frame(bars_outer, bg=self.PANEL_BG)
            row.pack(fill="x", pady=6)
            color = self.CLASS_COLORS.get(cls, self.ACCENT)
            tk.Label(row, text=cls, font=label_font, width=13,
                     anchor="w", bg=self.PANEL_BG, fg=color).pack(side="left")
            canvas = tk.Canvas(row, height=22, bg=self.BAR_BG, highlightthickness=0)
            canvas.pack(side="left", fill="x", expand=True, padx=(8, 8))
            canvas.create_rectangle(0, 0, 2000, 22, fill=self.BAR_BG, outline="")
            fill_id = canvas.create_rectangle(0, 0, 0, 22, fill=color, outline="")
            pct_lbl = tk.Label(row, text="0.0%", font=label_font, width=7,
                               anchor="e", bg=self.PANEL_BG, fg=color)
            pct_lbl.pack(side="right")
            self.bar_canvases[cls] = canvas
            self.bar_fills[cls]    = fill_id
            self.bar_labels[cls]   = pct_lbl

        log_outer = tk.Frame(right, bg=self.PANEL_BG,
                             highlightbackground=self.BAR_BG,
                             highlightthickness=1, pady=10, padx=12)
        log_outer.grid(row=2, column=0, sticky="ew")
        tk.Label(log_outer, text="EVENT LOG", font=small_font,
                 bg=self.PANEL_BG, fg=self.TEXT_DIM).pack(anchor="w", pady=(0, 6))
        self.log_text = tk.Text(log_outer, height=6, bg=self.PANEL_BG,
                                fg=self.TEXT_DIM, font=small_font,
                                relief="flat", state="disabled",
                                wrap="word", insertbackground=self.WHITE)
        self.log_text.pack(fill="x")
        self.log_text.tag_config("STOP", foreground=self.STOP_RED)
        self.log_text.tag_config("SAFE", foreground=self.SAFE_GREEN)
        self.log_text.tag_config("INFO", foreground=self.TEXT_DIM)
        self._last_logged_state = None
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _start_camera_thread(self):
        t = threading.Thread(target=self._camera_loop, daemon=True)
        t.start()

    def _camera_loop(self):
        while self.running:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            self.engine.push_frame(frame)
            h, w  = frame.shape[:2]
            scale = 620 / w
            disp  = cv2.resize(frame, (620, int(h * scale)))
            rgb   = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            self._latest_frame = ImageTk.PhotoImage(Image.fromarray(rgb))

    def _update_loop(self):
        if not self.running:
            return
        state = self.engine.get_state()
        if hasattr(self, "_latest_frame"):
            self.cam_label.configure(image=self._latest_frame)
            self.cam_label.image = self._latest_frame
        self.fps_label.configure(text=f"FPS: {state['fps']:.1f}")
        self.time_label.configure(text=time.strftime("%H:%M:%S"))
        for i, cls in enumerate(self.class_labels):
            prob  = float(state["probs"][i]) if len(state["probs"]) > i else 0.0
            self.bar_labels[cls].configure(text=f"{prob*100:.1f}%")
            canvas = self.bar_canvases[cls]
            canvas.update_idletasks()
            fill_w = max(2, int(canvas.winfo_width() * prob))
            canvas.coords(self.bar_fills[cls], 0, 0, fill_w, 22)
        self._update_banner(state)
        self._maybe_log(state)
        self.root.after(16, self._update_loop)

    def _update_banner(self, state):
        if state["stop"]:
            if not self._flash_on:
                self._flash_on = True
                self._flash_banner()
        else:
            self._flash_on = False
            if self._flash_after:
                self.root.after_cancel(self._flash_after)
                self._flash_after = None
            if state["label"] == "PASSTHRU":
                bg, txt, sub, fg, sfg = self.WARN_AMBER, "● CAUTION", "Pass-through detected", "#3d2e00", "#7a5c00"
            else:
                bg, txt, sub, fg, sfg = self.SAFE_GREEN, "● SAFE", "No interaction detected", "#0a2e14", "#1a5c2e"
            self.banner_frame.configure(bg=bg)
            self.banner_label.configure(text=txt, bg=bg, fg=fg)
            self.banner_sub.configure(text=f"{sub}  ({state['conf']*100:.1f}%)", bg=bg, fg=sfg)

    def _flash_banner(self):
        if not self._flash_on:
            return
        state  = self.engine.get_state()
        toggle = getattr(self, "_flash_toggle", True)
        bg     = self.STOP_RED if toggle else "#7a1a14"
        self.banner_frame.configure(bg=bg)
        self.banner_label.configure(text="⚠  STOP — INTERACTION", bg=bg, fg="#ffffff")
        self.banner_sub.configure(
            text=f"Confidence: {state['conf']*100:.1f}%  |  ARM HALT SIGNAL",
            bg=bg, fg="#ffccc9")
        self._flash_toggle = not toggle
        self._flash_after  = self.root.after(400, self._flash_banner)

    def _maybe_log(self, state):
        cur = state["label"]
        if cur == self._last_logged_state:
            return
        self._last_logged_state = cur
        ts  = time.strftime("%H:%M:%S")
        tag = "STOP" if cur == "INTERACTION" else "INFO" if cur == "PASSTHRU" else "SAFE"
        msg = f"[{ts}]  {cur}  ({state['conf']*100:.1f}%)\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg, tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_close(self):
        self.running = False
        self.camera.release()
        self.root.destroy()


def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    args.model         = "cnnlstm"
    args.pretrain_path = ""
    args.norm_value    = 1
    args.use_cuda      = torch.cuda.is_available()

    print("[INFO] Loading class labels ...")
    class_labels = load_class_labels(args.annotation_path)
    print(f"[INFO] Classes: {class_labels}")

    print("[INFO] Loading model ...")
    model = load_model(args, device)
    print("[INFO] Model ready.")

    transform = build_transform(args.sample_size)

    if REALSENSE_AVAILABLE:
        print("[INFO] Starting RealSense camera ...")
        try:
            camera = RealSenseCamera()
        except Exception as e:
            print(f"[WARN] RealSense failed ({e}), falling back to webcam")
            camera = WebcamCamera(args.camera_index)
    else:
        print(f"[INFO] Starting webcam (index {args.camera_index}) ...")
        camera = WebcamCamera(args.camera_index)

    engine = InferenceEngine(model, transform, class_labels, args, device)
    root   = tk.Tk()
    App(root, engine, class_labels, camera)
    root.mainloop()


if __name__ == "__main__":
    main()
    