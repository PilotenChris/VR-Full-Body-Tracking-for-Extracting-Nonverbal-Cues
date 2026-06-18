import json
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error, )
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROLES = [ "hmd", "chest", "waist", "left_knee", "right_knee", "left_foot", "right_foot", "left_controller", "right_controller" ]

ROLE_DIM = 7
FEATURE_DIM = len(ROLES) * ROLE_DIM * 2


def label_from_filename(name: str) -> str | None:
    filename = name.lower()
    if "squatd" in filename:
        return "squat_down"
    if "squatu" in filename:
        return "squat_up"
    if "squat" in filename:
        return "squat"

    if "neutral" in filename:
        return "neutral"

    if "nonod" in filename:
        speed = "fast" if "fast" in filename else "slow" if "slow" in filename else "unknown"
        size = "small" if "low" in filename else "large" if "high" in filename else "unknown"
        return f"no_nod_{size}_{speed}"
    if "yesnod" in filename:
        speed = "fast" if "fast" in filename else "slow" if "slow" in filename else "unknown"
        size = "small" if "low" in filename else "large" if "high" in filename else "unknown"
        return f"yes_nod_{size}_{speed}"

    # Hand waves
    def speed_part():
        return "fast" if "fast" in filename else "slow" if "slow" in filename else "unknown"

    def height_part():
        return "high" if "high" in filename else "low" if "low" in filename else "unknown"

    if "lhand" in filename:
        return f"left_hand_wave_{height_part()}_{speed_part()}"

    if "rhand" in filename:
        return f"right_hand_wave_{height_part()}_{speed_part()}"

    if "bhand" in filename:
        return f"both_hand_wave_{height_part()}_{speed_part()}"

    if "lfootrestless" in filename:
        return f"left_foot_restless"

    if "rfootrestless" in filename:
        return f"right_foot_restless"

    return None


def collect_files(base_dir: str) -> list[tuple[str, list[str]]]:
    base = Path(base_dir)
    items: list[tuple[str, list[str]]] = []

    for sub in base.glob("*_tracking_data"):
        for p in sub.glob("all_*.json"):
            lbl = label_from_filename(p.name)
            if lbl is None:
                continue
            items.append((str(p), [lbl]))
    return items


def load_frames(path: str) -> List[List[dict]]:
    frames: list[list[dict]] = []
    with open(path, "r", encoding="utf-8") as file_frame:
        for line in file_frame:
            line = line.strip()
            if not line:
                continue
            frames.append(json.loads(line))
    return frames


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


def frame_to_vec(frame: List[dict]) -> np.ndarray:
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


def frames_to_features(frames: List[List[dict]]) -> np.ndarray:
    base = np.stack([frame_to_vec(fr) for fr in frames], axis=0)
    vel = np.zeros_like(base)
    vel[1:] = base[1:] - base[:-1]
    feat = np.concatenate([base, vel], axis=1)
    return feat


@dataclass
class LabelSpec:
    name_to_idx: Dict[str, int]


def one_hot(label_names: List[str], spec: LabelSpec) -> np.ndarray:
    y = np.zeros((len(spec.name_to_idx),), dtype=np.float32)
    for ln in label_names:
        y[spec.name_to_idx[ln]] = 1.0

    return y


def get_role_y(frame: List[dict], role: str) -> float | None:
    for item in frame:
        if item.get("role") == role:
            pos = item.get("pos")
            if pos is None or len(pos) != 3:
                return None
            return float(pos[1])
    return None


def frame_height_signal(frame: List[dict]) -> float:
    for role in ("waist", "chest", "hmd"):
        y = get_role_y(frame, role)
        if y is not None:
            return y
    return float("nan")


def split_squat_frames(frames: List[List[dict]], min_gap: int = 5) -> tuple[List[List[dict]], List[List[dict]]]:
    if len(frames) < 10:
        return frames, []
    heights = np.array([frame_height_signal(fr) for fr in frames], dtype=np.float32)
    heights = fill_nans_nearest(heights)
    if np.isnan(heights).any():
        return frames, []
    idx_min = int(np.argmin(heights))
    idx_min = max(min_gap, min(len(frames) - min_gap - 1, idx_min))
    down = frames[: idx_min + 1]
    up = frames[idx_min: ]
    return down, up


class GRUGestureNet(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, hidden: int = 256, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False, )
        self.head = nn.Sequential( nn.LayerNorm(hidden), nn.Linear(hidden, num_labels), )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits

def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        lr: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu", ):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * x.size(0)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                va_loss += loss.item() * x.size(0)

        tr_loss /= len(train_loader.dataset)
        va_loss /= len(val_loader.dataset)

        is_best = va_loss < best_val
        if is_best:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        star = " *" if is_best else ""
        print(f"epoch {ep:02d} | train {tr_loss:.4f} | val {va_loss:.4f}{star}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val


class LivePredictor:
    def __init__(self, model: nn.Module, mean: np.ndarray, std: np.ndarray, seq_len: int, label_names: List[str], device: str):
        self.model = model.to(device).eval()
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.seq_len = seq_len
        self.label_names = label_names
        self.device = device

        self.buf = np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)
        self.prev_base = np.zeros((len(ROLES) * ROLE_DIM,), dtype=np.float32)
        self.i = 0
        self.filled = 0

    def push_frame(self, frame: List[dict]) -> Dict[str, float]:
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


def build_label_spec(files_and_labels: list[tuple[str, list[str]]]) -> LabelSpec:
    all_names = sorted({lbl for _, lbls in files_and_labels for lbl in lbls})
    return LabelSpec({name: i for i, name in enumerate(all_names)})


def is_mixed_squat(path: str) -> bool:
    p = Path(path)
    parent = p.parent.name.lower()
    name = p.name.lower()
    return ("squat" in name) and (parent.startswith("02_") or parent.startswith("03_"))


def moving_average(x: np.ndarray, w: int = 9) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(x, kernel, mode="same")


def find_local_extrema(x: np.ndarray):
    mins, maxs = [], []
    for i in range(1, len(x) - 1):
        if x[i] < x[i - 1] and x[i] < x[i + 1]:
            mins.append(i)
        if x[i] > x[i - 1] and x[i] > x[i + 1]:
            maxs.append(i)
    return mins, maxs


def enforce_min_separation(idxs: list[int], min_sep: int) -> list[int]:
    idxs = sorted(idxs)
    kept = []
    last = -10**9
    for i in idxs:
        if i - last >= min_sep:
            kept.append(i)
            last = i
    return kept


def fill_nans_nearest(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    if not np.isnan(x).any():
        return x

    valid = np.where(~np.isnan(x))[0]
    if len(valid) == 0:
        return x

    last = x[valid[0]]
    for i in range(len(x)):
        if np.isnan(x[i]):
            x[i] = last
        else:
            last = x[i]

    first = valid[0]
    x[:first] = x[first]
    return x


def split_multi_squats(
        frames: list[list[dict]],
        smooth_w: int = 9,
        min_depth: float = 0.08,
        min_len_frames: int = 20,
        pad: int = 10) -> list[tuple[list[list[dict]], list[list[dict]]]]:
    if len(frames) < 50:
        return []

    heights = np.array([frame_height_signal(fr) for fr in frames], dtype=np.float32)
    heights = fill_nans_nearest(heights)

    if np.isnan(heights).any():
        return []

    hs = moving_average(heights, w=smooth_w)

    baseline = float(np.percentile(hs, 80))
    thresh = baseline - float(min_depth)

    low = hs < thresh

    reps: list[tuple[list[list[dict]], list[list[dict]]]] = []
    i = 0
    N = len(frames)

    while i < N:
        if not low[i]:
            i += 1
            continue

        start = i
        while i < N and low[i]:
            i += 1
        end = i - 1

        if (end - start + 1) < min_len_frames:
            continue

        seg = hs[start:end + 1]
        mi = start + int(np.argmin(seg))

        left = max(0, start - pad)
        right = min(N - 1, end + pad)

        down = frames[left:mi + 1]
        up = frames[mi:right + 1]

        if len(down) >= min_len_frames and len(up) >= min_len_frames:
            reps.append((down, up))

    return reps


def expand_squat_mixed(files_and_labels: list[tuple[str, list[str]]]) -> list[tuple[str, list[str], list[list[dict]] | None]]:
    expanded = []
    for path, labels in files_and_labels:
        if labels == ["squat"] and is_mixed_squat(path):
            frames = load_frames(path)
            reps = split_multi_squats(frames)

            if not reps:
                down, up = split_squat_frames(frames)
                expanded.append((path + "::down0", ["squat_down"], down))
                if up:
                    expanded.append((path + "::up0", ["squat_up"], up))
                continue

            for k, (down, up) in enumerate(reps):
                expanded.append((path + f"::down{k}", ["squat_down"], down))
                expanded.append((path + f"::up{k}", ["squat_up"], up))
        else:
            expanded.append((path, labels, None))
    return expanded


def load_frames_override(path: str, override: list[list[dict]] | None) -> list[list[dict]]:
    if override is not None:
        return override

    real_path = path.split("::")[0]
    return load_frames(real_path)


def evaluate(
        model: nn.Module,
        loader: DataLoader,
        label_names: list[str],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.5):
    model.eval()
    model.to(device)

    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits: torch.Tensor = model(x)
            probs: np.ndarray = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

            all_targets.append(y.numpy())

    y_true: np.ndarray = np.vstack(all_targets)
    y_prob: np.ndarray = np.vstack(all_probs)
    y_pred: np.ndarray = (y_prob >= threshold).astype(np.int32)

    print("\n==== Evaluation Metrics ====")

    exact_match: float = float(np.mean(np.all(y_pred == y_true, axis=1)))
    print("Exact match accuracy:", exact_match)

    per_label_accuracy: np.ndarray = np.mean(y_pred == y_true, axis=0)
    for name, acc in zip(label_names, per_label_accuracy):
        print(f"{name:30s} accuracy: {float(acc):.3f}")

    macro_precision: float = float( precision_score(y_true, y_pred, average="macro", zero_division=0) )
    macro_recall: float = float( recall_score(y_true, y_pred, average="macro", zero_division=0) )
    macro_f1: float = float( f1_score(y_true, y_pred, average="macro", zero_division=0) )

    print("\nMacro Precision:", macro_precision)
    print("Macro Recall:", macro_recall)
    print("Macro F1:", macro_f1)

    micro_precision: float = float( precision_score(y_true, y_pred, average="micro", zero_division=0) )
    micro_recall: float = float( recall_score(y_true, y_pred, average="micro", zero_division=0) )
    micro_f1: float = float( f1_score(y_true, y_pred, average="micro", zero_division=0) )

    print("\nMicro Precision:", micro_precision)
    print("Micro Recall:", micro_recall)
    print("Micro F1:", micro_f1)

    mse: float = float(mean_squared_error(y_true, y_prob))
    print("\nMSE (probabilities vs labels):", mse)


class GestureWindowDataset(Dataset):
    def __init__(
            self,
            entries: list[tuple[str, list[str], list[list[dict]] | None]],
            label_spec: LabelSpec,
            seq_len: int = 150,
            stride: int = 10,
            mean: np.ndarray | None = None,
            std: np.ndarray | None = None,
            compute_norm: bool = False, ):
        self.seq_len = seq_len
        self.stride = stride
        self.label_spec = label_spec

        self.samples = []
        all_feats = []

        for path, labels, override in entries:
            frames = load_frames_override(path, override)
            feat = frames_to_features(frames)
            y = one_hot(labels, label_spec)

            T = feat.shape[0]
            for start in range(0, max(1, T - seq_len + 1), stride):
                window = feat[start:start + seq_len]
                if window.shape[0] < seq_len:
                    pad = np.zeros((seq_len - window.shape[0], feat.shape[1]), dtype=np.float32)
                    window = np.vstack([window, pad])

                window = window.astype(np.float32)
                self.samples.append((window, y))
                if compute_norm:
                    all_feats.append(window)

        D = FEATURE_DIM
        if compute_norm:
            if len(all_feats) == 0:
                self.mean = np.zeros((D,), dtype=np.float32)
                self.std = np.ones((D,), dtype=np.float32)
            else:
                stacked = np.concatenate(all_feats, axis=0)
                self.mean = stacked.mean(axis=0).astype(np.float32)
                self.std = (stacked.std(axis=0) + 1e-6).astype(np.float32)
        else:
            if mean is None or std is None:
                raise ValueError("VAL/TEST dataset requires mean and std from TRAIN dataset.")
            self.mean = mean.astype(np.float32)
            self.std = std.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        x = (x - self.mean) / self.std
        return torch.from_numpy(x), torch.from_numpy(y)


def train_and_save(base_dir: str, out_path: str = "gesture_gru.pt"):
    files_and_labels = collect_files(base_dir)

    entries = expand_squat_mixed(files_and_labels)

    all_labels = sorted({lbl for _, lbls, _ in entries for lbl in lbls})
    label_spec = LabelSpec({name: i for i, name in enumerate(all_labels)})
    print("Labels:", all_labels)

    train_entries, val_entries = train_test_split(entries, test_size=0.2, random_state=42)

    train_ds = GestureWindowDataset(train_entries, label_spec, seq_len=150, stride=100, compute_norm=True)
    val_ds = GestureWindowDataset(val_entries, label_spec, seq_len=150, stride=100, mean=train_ds.mean, std=train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = GRUGestureNet(input_dim=FEATURE_DIM, num_labels=len(all_labels), hidden=128, layers=1, dropout=0.4)
    best_val = train(model, train_loader, val_loader, epochs=20, lr=1e-3)
    print("Best val:", best_val)

    evaluate(model, val_loader, all_labels)

    save_obj = {
        "model_state": model.state_dict(),
        "label_names": all_labels,
        "label_to_idx": label_spec.name_to_idx,
        "mean": train_ds.mean,
        "std": train_ds.std,
        "seq_len": train_ds.seq_len,
        "roles": ROLES,
        "feature_dim": FEATURE_DIM,
        "model_hparams": {
            "hidden": 128,
            "layers": 1,
            "dropout": 0.4 },
    }
    torch.save(save_obj, out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    train_and_save(base_dir=".", out_path="gesture_gru.pt")