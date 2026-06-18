import time
import math
import json
import socket
import openvr
import threading
import tkinter as tk
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from pathlib import Path

# Setting up main window for Tkinter GUI application
window = tk.Tk()
window.geometry("1000x500")
window.title("Extract Nonverbal Cues For VR")

# Font style for labels in application
custom_font1: tuple = ("Helvetica", 15)

# Frame constants
MAIN_FRAME = "main"
LOGGING_FRAME = "logging_trackers"
EXTRACTING_FRAME = "extracting_nonverbal_cues"

MAX_DEVICES = openvr.k_unMaxTrackedDeviceCount

ROLES = [
    "hmd", "chest", "waist", "left_knee", "right_knee", "left_foot", "right_foot", "left_controller", "right_controller"
]

ROLE_DIM = 7
FEATURE_DIM = len(ROLES) * ROLE_DIM * 2

CUE_KEYS: tuple[str, ...] = (
    "yes_nod_small_fast",
    "yes_nod_small_slow",
    "yes_nod_large_fast",
    "yes_nod_large_slow",
    "no_nod_small_fast",
    "no_nod_small_slow",
    "no_nod_large_fast",
    "no_nod_large_slow",
    "left_hand_wave_high_fast",
    "left_hand_wave_high_slow",
    "left_hand_wave_low_fast",
    "left_hand_wave_low_slow",
    "right_hand_wave_high_fast",
    "right_hand_wave_high_slow",
    "right_hand_wave_low_fast",
    "right_hand_wave_low_slow",
    "both_hand_wave_high_fast",
    "both_hand_wave_high_slow",
    "squat_down",
    "squat_up",
    "left_foot_restless",
    "right_foot_restless"
)


def mat34_to_pos_quat(m) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    r00, r01, r02, tx = m[0]
    r10, r11, r12, ty = m[1]
    r20, r21, r22, tz = m[2]

    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r21 - r12) / s
        qy = (r02 - r20) / s
        qz = (r10 - r01) / s
    elif (r00 > r11) and (r00 > r22):
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s

    return (float(tx), float(ty), float(tz)), (float(qx), float(qy), float(qz), float(qw))


def role_name(vr, idx) -> str:
    role = vr.getControllerRoleForTrackedDeviceIndex(idx)
    if role == openvr.TrackedControllerRole_LeftHand:
        return "left_controller"
    if role == openvr.TrackedControllerRole_RightHand:
        return "right_controller"
    return "unknown_controller"


def get_serial(vr, idx) -> Optional[str]:
    try:
        serial = vr.getStringTrackedDeviceProperty(idx, openvr.Prop_SerialNumber_String)
        return serial.strip() if serial else None
    except openvr.OpenVRError:
        return None


def tracker_name(vr, idx) -> str:
    serial_role = get_serial(vr, idx)
    if not serial_role:
        return f"tracker_{idx}"

    if serial_role.lower().startswith("human://"):
        return serial_role.split("://", 1)[1].lower()

    return serial_role.lower()


def make_all_filename(path_str: str) -> str:
    p = Path(path_str)
    return str(p.with_name("all_" + p.name))


def make_cues_filename(path_str: str) -> str:
    p = Path(path_str)
    return str(p.with_name("cues_" + p.name))


def filter_devices(devices: list[dict], allowed_roles: set[str]) -> list[dict]:
    return [device_filtered for device_filtered in devices if device_filtered.get("role") in allowed_roles]


@dataclass
class TriggerState:
    is_standing: bool = True
    last_fire_ts: float = 0.0


def ema_update(prev: float, new: float, alpha: float) -> float:
    return prev * (1.0 - alpha) + new * alpha


class CuePostProcessor:
    def __init__(
            self,
            alpha: float = 0.25,
            threshold_continuous: float = 0.70,
            threshold_trigger_on: float = 0.70,
            threshold_trigger_off: float = 0.45,
            trigger_cooldown_s: float = 0.75
    ) -> None:
        self.alpha = alpha
        self.threshold_continuous = threshold_continuous
        self.threshold_trigger_on = threshold_trigger_on
        self.threshold_trigger_off = threshold_trigger_off
        self.trigger_cooldown_s = trigger_cooldown_s

        self.smooth: dict[str, float] = {}
        self.trigger = TriggerState()

    def _smooth_prob(self, key: str, p: float) -> float:
        prev = self.smooth.get(key, p)
        s = ema_update(prev, p, self.alpha)
        self.smooth[key] = s
        return s

    def process(self, probs: dict[str, float]) -> dict[str, object]:
        now = time.time()

        def sp(name: str) -> float:
            return self._smooth_prob(name, float(probs.get(name, 0.0)))

        continuous_labels: list[str] = [
            "yes_nod_small_fast", "yes_nod_small_slow", "yes_nod_large_fast", "yes_nod_large_slow",
            "no_nod_small_fast", "no_nod_small_slow", "no_nod_large_fast", "no_nod_large_slow",
            "left_hand_wave_high_fast", "left_hand_wave_high_slow", "left_hand_wave_low_fast", "left_hand_wave_low_slow",
            "right_hand_wave_high_fast", "right_hand_wave_high_slow", "right_hand_wave_low_fast", "right_hand_wave_low_slow",
            "both_hand_wave_high_fast", "both_hand_wave_high_slow",
            "left_foot_restless", "right_foot_restless"
        ]

        scores: dict[str, float] = {}
        active: list[str] = []

        for lbl in continuous_labels:
            s = sp(lbl)
            scores[lbl] = s
            if s >= self.threshold_continuous:
                active.append(lbl)

        squat_down_p = sp("squat_down")
        squat_up_p = sp("squat_up")

        triggers: list[str] = []

        can_fire = (now - self.trigger.last_fire_ts) >= self.trigger_cooldown_s

        if self.trigger.is_standing and can_fire and squat_down_p >= self.threshold_trigger_on:
            triggers.append("squat_down")
            self.trigger.is_standing = False
            self.trigger.last_fire_ts = now

        if (not self.trigger.is_standing) and can_fire and squat_up_p >= self.threshold_trigger_on:
            triggers.append("squat_up")
            self.trigger.is_standing = True
            self.trigger.last_fire_ts = now

        if squat_down_p <= self.threshold_trigger_off and squat_up_p <= self.threshold_trigger_off:
            pass

        scores["squat_down"] = squat_down_p
        scores["squat_up"] = squat_up_p

        return {
            "active": active,
            "triggers": triggers,
            "scores": scores,
            "is_standing": self.trigger.is_standing,
        }


class UdpCueSender:
    def __init__(self, host: str = "127.0.0.1", port: int = 9000) -> None:
        self.addr: tuple[str, int] = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload: dict[str, object]) -> None:
        msg = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.sock.sendto(msg, self.addr)


class GRUGestureNet(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, hidden: int = 256, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits


def safe_pos(item) -> np.ndarray:
    pos = item.get("pos", None)
    if pos is None or len(pos) != 3:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.array(pos, dtype=np.float32)


def safe_quat(item) -> np.ndarray:
    quat = item.get("quat", None)
    if quat is None or len(quat) != 4:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return np.array(quat, dtype=np.float32)


def frame_to_vec(frame: list[dict]) -> np.ndarray:
    out = np.zeros((len(ROLES), ROLE_DIM), dtype=np.float32)
    role_to_idx = {r: i for i, r in enumerate(ROLES)}

    for item in frame:
        role = item.get("role")
        if role not in role_to_idx:
            continue

        i = role_to_idx[role]
        pos = safe_pos(item)
        quat = safe_quat(item)

        out[i, 0:3] = pos
        out[i, 3:7] = quat

    return out.reshape(-1)


class LivePredictor:
    def __init__(
            self,
            model: nn.Module,
            mean: np.ndarray,
            std: np.ndarray,
            seq_len: int,
            label_names: list[str],
            device: str):
        self.model = model.to(device).eval()
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.seq_len = seq_len
        self.label_names = label_names
        self.device = device

        self.buf = np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)
        self.prev_base = np.zeros((len(ROLES)*ROLE_DIM,), dtype=np.float32)
        self.i = 0
        self.filled = 0

    def push_frame(self, frame: list[dict]) -> dict[str, float]:
        base = frame_to_vec(frame)
        vel = base - self.prev_base
        self.prev_base = base
        feat = np.concatenate([base, vel], axis=0).astype(np.float32)

        self.buf[self.i] = feat
        self.i = (self.i + 1) % self.seq_len
        self.filled = min(self.seq_len, self.filled + 1)

        if self.filled < self.seq_len:
            return {}

        x = np.concatenate([self.buf[self.i:], self.buf[:self.i]], axis=0)
        x = (x - self.mean) / self.std
        xt = torch.from_numpy(x[None, :, :]).to(self.device)

        with torch.no_grad():
            logits = self.model(xt)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        return {name: float(p) for name, p in zip(self.label_names, probs)}


def load_predictor(ckpt_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> LivePredictor:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    label_names: list[str] = list(ckpt["label_names"])
    mean: np.ndarray = np.asarray(ckpt["mean"], dtype=np.float32)
    std: np.ndarray = np.asarray(ckpt["std"], dtype=np.float32)
    seq_len: int = int(ckpt["seq_len"])

    hparams = ckpt.get("model_hparams", {})
    hidden: int = int(hparams.get("hidden", 256))
    layers: int = int(hparams.get("layers", 2))
    dropout: float = float(hparams.get("dropout", 0.2))

    model = GRUGestureNet(
        input_dim=FEATURE_DIM,
        num_labels=len(label_names),
        hidden=hidden,
        layers=layers,
        dropout=dropout,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return LivePredictor(model, mean, std, seq_len, label_names, device)


def tracking_loop(
        stop_event: threading.Event,
        selected_roles: Optional[set[str]],
        out_selected_path: str, out_all_path:
        Optional[str]) -> None:
    openvr.init(openvr.VRApplication_Other)
    vr = openvr.VRSystem()

    try:
        file_selected = open(out_selected_path, "a", encoding="utf-8")
        file_all = open(out_all_path, "a", encoding="utf-8") if out_all_path else None

        period = 0.01
        next_tick = time.perf_counter()

        try:
            while not stop_event.is_set():
                poses = vr.getDeviceToAbsoluteTrackingPose(
                    openvr.TrackingUniverseStanding, 0, MAX_DEVICES
                )

                devices: list[dict] = []
                for idx, p in enumerate(poses):
                    if not p.bDeviceIsConnected:
                        continue

                    cls = vr.getTrackedDeviceClass(idx)
                    pos = quat = None
                    if p.bPoseIsValid:
                        pos, quat = mat34_to_pos_quat(p.mDeviceToAbsoluteTracking)

                    if cls == openvr.TrackedDeviceClass_HMD:
                        devices.append({"role": "hmd", "pos": pos, "quat": quat})
                    elif cls == openvr.TrackedDeviceClass_Controller:
                        devices.append({"role": role_name(vr, idx), "pos": pos, "quat": quat})
                    elif cls == openvr.TrackedDeviceClass_GenericTracker:
                        devices.append({"role": tracker_name(vr, idx), "pos": pos, "quat": quat})

                if file_all is not None:
                    file_all.write(json.dumps(devices) + "\n")
                    file_all.flush()

                if selected_roles is None:
                    file_selected.write(json.dumps(devices) + "\n")
                else:
                    filtered_devices = filter_devices(devices, selected_roles)
                    file_selected.write(json.dumps(filtered_devices) + "\n")

                file_selected.flush()

                next_tick += period
                sleep_time = next_tick - time.perf_counter()

                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            file_selected.close()
            if file_all is not None:
                file_all.close()

    finally:
        openvr.shutdown()


def extracting_loop(
        stop_event: threading.Event,
        out_selected_path: str,
        out_cues_path: str,
        unity_udp_host: str = "127.0.0.1",
        unity_udp_port: int = 9100) -> None:
    openvr.init(openvr.VRApplication_Other)
    vr = openvr.VRSystem()

    predictor = load_predictor("gesture_gru_more_data_193.pt")
    post: CuePostProcessor = CuePostProcessor()
    sender: UdpCueSender = UdpCueSender(unity_udp_host, unity_udp_port)

    try:
        file_tracking = open(out_selected_path, "a", encoding="utf-8")
        file_cues = open(out_cues_path, "a", encoding="utf-8")

        period = 0.01
        next_tick = time.perf_counter()

        try:
            while not stop_event.is_set():
                poses = vr.getDeviceToAbsoluteTrackingPose(
                    openvr.TrackingUniverseStanding, 0, MAX_DEVICES
                )

                devices: list[dict] = []
                for idx, p in enumerate(poses):
                    if not p.bDeviceIsConnected:
                        continue

                    cls = vr.getTrackedDeviceClass(idx)
                    pos = quat = None
                    if p.bPoseIsValid:
                        pos, quat = mat34_to_pos_quat(p.mDeviceToAbsoluteTracking)

                    if cls == openvr.TrackedDeviceClass_HMD:
                        devices.append({"role": "hmd", "pos": pos, "quat": quat})
                    elif cls == openvr.TrackedDeviceClass_Controller:
                        devices.append({"role": role_name(vr, idx), "pos": pos, "quat": quat})
                    elif cls == openvr.TrackedDeviceClass_GenericTracker:
                        devices.append({"role": tracker_name(vr, idx), "pos": pos, "quat": quat})

                probs: dict[str, float] = predictor.push_frame(devices)
                if probs:
                    cues = post.process(probs)

                    payload: dict[str, object] = {
                        "timestamp": time.time(),
                        "active": cues["active"],
                        "triggers": cues["triggers"],
                        "is_standing": cues["is_standing"],
                    }

                    file_cues.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    file_cues.flush()

                    sender.send(payload)

                file_tracking.write(json.dumps({"timestamp": time.time(), "devices": devices}) + "\n")
                file_tracking.flush()

                next_tick += period
                sleep_time = next_tick - time.perf_counter()

                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            file_tracking.close()
            file_cues.close()
    finally:
        openvr.shutdown()


def start() -> None:
    frames = {
        MAIN_FRAME: ttk.Frame(window),
        LOGGING_FRAME: ttk.Frame(window),
        EXTRACTING_FRAME: ttk.Frame(window),
    }

    for frame in frames.values():
        frame.grid(row=0, column=0, sticky='nsew')
        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)

    main_frame(frames[MAIN_FRAME], frames)

    frames[MAIN_FRAME].tkraise()
    window.mainloop()


def main_frame(frame, frames):
    ttk.Label(frame, text="Start and go to logging trackers", font=custom_font1, padding=(0, 20)).pack()

    def on_open_logging():
        logging_frame(frames[LOGGING_FRAME], frames)
        frames[LOGGING_FRAME].tkraise()

    def on_open_extracting():
        extracting_frame(frames[EXTRACTING_FRAME], frames)
        frames[EXTRACTING_FRAME].tkraise()

    loggingb = ttk.Button(frame, text="Logging trackers", style="alt.TButton",
                          command=on_open_logging)
    loggingb.pack(pady=5)

    extractingb = ttk.Button(frame, text="Extract nonverbal cues", state="alt.TButton", command=on_open_extracting)
    extractingb.pack(pady=5)


def logging_frame(frame, frames):
    tracking_stop_event = threading.Event()
    tracking_thread: Optional[threading.Thread] = None

    openvr.init(openvr.VRApplication_Other)
    vr = openvr.VRSystem()
    try:
        tracker_list: list[str] = []

        tracker_role_select = vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, MAX_DEVICES
        )

        for idx, p in enumerate(tracker_role_select):
            if not p.bDeviceIsConnected:
                continue

            cls = vr.getTrackedDeviceClass(idx)
            if cls == openvr.TrackedDeviceClass_HMD:
                tracker_list.append("hmd")
            elif cls == openvr.TrackedDeviceClass_Controller:
                tracker_list.append(role_name(vr, idx))
            elif cls == openvr.TrackedDeviceClass_GenericTracker:
                tracker_list.append(tracker_name(vr, idx))

        openvr.shutdown()

        top_frame = ttk.Frame(frame)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        ttk.Label(top_frame, text="Track Trackers", font=custom_font1, padding=(0, 20)).pack()

        left_frame = ttk.Frame(top_frame)
        mid_frame = ttk.Frame(top_frame)
        right_frame = ttk.Frame(top_frame)

        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=40, pady=20)
        mid_frame.pack(side=tk.LEFT, padx=10, pady=20)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=40, pady=20)

        ttk.Label(left_frame, text="Available Trackers", font=custom_font1, padding=(0, 20)).pack()
        list_traker_variable = tk.Variable(value=tracker_list)
        role_listbox = tk.Listbox(left_frame, listvariable=list_traker_variable, selectmode=tk.EXTENDED, height=10,
                                  width=25)
        role_listbox.pack(fill=tk.BOTH, expand=True)

        ttk.Label(right_frame, text="Tracked Trackers", font=custom_font1, padding=(0, 20)).pack()
        selected_listbox = tk.Listbox(right_frame, selectmode=tk.EXTENDED, height=10, width=25)
        selected_listbox.pack(fill=tk.BOTH, expand=True)

        selected_tracker_list = []

        def add_selected() -> None:
            selected_indices_lb = role_listbox.curselection()
            if not selected_indices_lb:
                messagebox.showinfo("Nothing selected", "Select one or more items first.")
                return

            selected_trackers_lb = [role_listbox.get(i) for i in selected_indices_lb]

            for tracker_lb in selected_trackers_lb:
                if tracker_lb not in selected_tracker_list:
                    selected_tracker_list.append(tracker_lb)
                    selected_listbox.insert(tk.END, tracker_lb)

        def remove_selected_from_list() -> None:
            selected_indices_lb = list(selected_listbox.curselection())
            if not selected_indices_lb:
                return

            for i in reversed(selected_indices_lb):
                tracker_lb = selected_listbox.get(i)
                selected_listbox.delete(i)
                if tracker_lb in selected_tracker_list:
                    selected_tracker_list.remove(tracker_lb)

        ttk.Button(mid_frame, text="Add ->", command=add_selected).pack(pady=5)
        ttk.Button(mid_frame, text="<- Remove", command=remove_selected_from_list).pack(pady=5)

        bottom_frame = ttk.Frame(frame)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=15)

        ttk.Label(bottom_frame, text="Output file:").pack(side=tk.LEFT)

        file_var = tk.StringVar(value="trackers.json")
        file_entry = ttk.Entry(bottom_frame, textvariable=file_var, width=50)
        file_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        def brows_file():
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            if path:
                file_var.set(path)

        ttk.Button(bottom_frame, text="Brows...", command=brows_file).pack(side=tk.LEFT, padx=5)

        def start_tracking():
            nonlocal tracking_thread

            output_path = file_var.get().strip()
            if not output_path:
                messagebox.showwarning("No File", "Choose/Create an output file.")
                return

            if len(selected_tracker_list) == 0:
                selected_roles = None
                out_selected = output_path
                out_all = None
            else:
                selected_roles = set(selected_tracker_list)
                out_selected = output_path
                out_all = make_all_filename(output_path)

            if tracking_thread is not None and tracking_thread.is_alive():
                return

            # No Delay function
            tracking_stop_event.clear()

            tracking_thread = threading.Thread(
                target=tracking_loop,
                args=(tracking_stop_event, selected_roles, out_selected, out_all),
                daemon=True
            )
            tracking_thread.start()

            start_btn.config(state="disabled")
            stop_btn.config(state="normal")

            # Delay function
            #def delayed_start():
            #    nonlocal tracking_thread

            #    tracking_stop_event.clear()

            #    tracking_thread = threading.Thread(
            #        target=tracking_loop,
            #        args=(tracking_stop_event, selected_roles, out_selected, out_all),
            #        daemon=True
            #    )
            #    tracking_thread.start()

            #    stop_btn.config(state="normal")

            #    window.after(10000, stop_tracking)

            # Temp
            #window.after(3000, delayed_start)

        def stop_tracking():
            tracking_stop_event.set()
            if tracking_thread is not None:
                tracking_thread.join(timeout=1.0)
            stop_btn.config(state="disabled")
            start_btn.config(state="normal")

        start_btn = ttk.Button(bottom_frame, text="Start tracking", command=start_tracking)
        start_btn.pack(side=tk.RIGHT, padx=(10, 0))

        stop_btn = ttk.Button(bottom_frame, text="Stop", command=stop_tracking, state="disabled")
        stop_btn.pack(side=tk.RIGHT)

    finally:
        openvr.shutdown()


def extracting_frame(frame, frames):
    for window_frame in frame.winfo_children():
        window_frame.destroy()

    tracking_stop_event = threading.Event()
    tracking_thread: Optional[threading.Thread] = None

    openvr.init(openvr.VRApplication_Other)
    vr = openvr.VRSystem()
    try:
        tracker_list: list[str] = []

        tracker_role_select = vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, MAX_DEVICES
        )

        for idx, p in enumerate(tracker_role_select):
            if not p.bDeviceIsConnected:
                continue

            cls = vr.getTrackedDeviceClass(idx)
            if cls == openvr.TrackedDeviceClass_HMD:
                tracker_list.append("hmd")
            elif cls == openvr.TrackedDeviceClass_Controller:
                tracker_list.append(role_name(vr, idx))
            elif cls == openvr.TrackedDeviceClass_GenericTracker:
                tracker_list.append(tracker_name(vr, idx))

        openvr.shutdown()

        top_frame = ttk.Frame(frame)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        ttk.Label(top_frame, text="Extracting Nonverbal Cues", font=custom_font1, padding=(0, 20)).pack()

        mid_top_frame = ttk.Frame(top_frame)
        mid_bottom_frame = ttk.Frame(top_frame)

        mid_top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=40, pady=20)
        mid_bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=40, pady=20)

        ttk.Label(mid_top_frame, text="Available Trackers", font=custom_font1, padding=(0, 20)).pack()
        list_tracker_variable = tk.Variable(value=tracker_list)
        tracker_listbox = tk.Listbox(mid_top_frame, listvariable=list_tracker_variable, height=10, width=25)
        tracker_listbox.pack(fill=tk.BOTH, expand=True)

        ttk.Label(mid_bottom_frame, text="Output file:").pack(side=tk.LEFT)

        file_var = tk.StringVar(value="participant.json")
        file_entry = ttk.Entry(mid_bottom_frame, textvariable=file_var, width=30)
        file_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        def brows_file():
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            if path:
                file_var.set(path)

        ttk.Button(mid_bottom_frame, text="Brows...", command=brows_file).pack(side=tk.LEFT, padx=5)

        def start_extracting():
            nonlocal tracking_thread

            output_path = file_var.get().strip()
            if not output_path:
                messagebox.showwarning("No File", "Choose/Create an output file.")
                return

            out_cues = make_cues_filename(output_path)

            if tracking_thread is not None and tracking_thread.is_alive():
                return

            tracking_stop_event.clear()

            tracking_thread = threading.Thread(
                target=extracting_loop,
                args=(tracking_stop_event, output_path, out_cues),
                daemon=True
            )
            tracking_thread.start()

            start_extract_btn.config(state="disabled")
            stop_extract_btn.config(state="normal")

        def stop_extracting():
            tracking_stop_event.set()
            if tracking_thread is not None:
                tracking_thread.join(timeout=1.0)
            print("---------------------------------------")
            stop_extract_btn.config(state="disabled")
            start_extract_btn.config(state="normal")

        start_extract_btn = ttk.Button(mid_bottom_frame, text="Start extracting", command=start_extracting)
        start_extract_btn.pack(side=tk.RIGHT, padx=(10, 0))

        stop_extract_btn = ttk.Button(mid_bottom_frame, text="Stop extracting", command=stop_extracting, state="disabled")
        stop_extract_btn.pack(side=tk.RIGHT)

        def go_back():
            frames[MAIN_FRAME].tkraise()

        bottom_frame = ttk.Frame(frame)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=15)

        ttk.Button(bottom_frame, text="Return", command=go_back).pack()

    finally:
        openvr.shutdown()


def main():
    start()


if __name__ == "__main__":
    main()