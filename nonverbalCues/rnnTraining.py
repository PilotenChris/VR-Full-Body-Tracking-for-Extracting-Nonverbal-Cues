import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROLES = [
    "hmd", "chest", "waist", "left_knee", "right_knee", "left_foot", "right_foot", "left_controller", "right_controller"
]

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


def load_frames(path: str) -> list[list[dict]]:
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


def frames_to_features(frames: list[list[dict]]) -> np.ndarray:
    base = np.stack([frame_to_vec(fr) for fr in frames], axis=0)
    vel = np.zeros_like(base)
    vel[1:] = base[1:] - base[:-1]
    feat = np.concatenate([base, vel], axis=1)
    return feat


@dataclass
class LabelSpec:
    name_to_idx: dict[str, int]


def one_hot(label_names: list[str], spec: LabelSpec) -> np.ndarray:
    y = np.zeros((len(spec.name_to_idx),), dtype=np.float32)
    for ln in label_names:
        y[spec.name_to_idx[ln]] = 1.0

    return y


def get_role_y(frame: list[dict], role: str) -> float | None:
    for item in frame:
        if item.get("role") == role:
            pos = item.get("pos")
            if pos is None or len(pos) != 3:
                return None
            return float(pos[1])
    return None


def frame_height_signal(frame: list[dict]) -> float:
    for role in ("waist", "chest", "hmd"):
        y = get_role_y(frame, role)
        if y is not None:
            return y
    return float("nan")


def split_squat_frames(frames: list[list[dict]], min_gap: int = 5) -> tuple[list[list[dict]], list[list[dict]]]:
    if len(frames) < 10:
        return frames, []

    heights = np.array([frame_height_signal(fr) for fr in frames], dtype=np.float32)
    heights = fill_nans_nearest(heights)
    if np.isnan(heights).any():
        return frames, []

    idx_min = int(np.argmin(heights))
    idx_min = max(min_gap, min(len(frames) - min_gap - 1, idx_min))

    down = frames[: idx_min + 1]
    up = frames[idx_min:]

    return down, up


class GRUGestureNet(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_labels: int,
            hidden: int = 256,
            layers: int = 2,
            dropout: float = 0.2) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        weight_decay: float = 1e-2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu", ) -> float:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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


def expand_squat_mixed(
        files_and_labels: list[tuple[str, list[str]]]) -> list[tuple[str, list[str], list[list[dict]] | None]]:
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
    print(f"Exact match accuracy: {exact_match:.4f}")

    macro_precision: float = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    macro_recall: float = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    macro_f1: float = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    print(f"\nMacro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    micro_precision: float = float(
        precision_score(y_true, y_pred, average="micro", zero_division=0)
    )
    micro_recall: float = float(
        recall_score(y_true, y_pred, average="micro", zero_division=0)
    )
    micro_f1: float = float(
        f1_score(y_true, y_pred, average="micro", zero_division=0)
    )

    print(f"\nMicro Precision: {micro_precision:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")

    y_true_cls: np.ndarray = np.argmax(y_true, axis=1)
    y_pred_cls: np.ndarray = np.argmax(y_prob, axis=1)

    print("\n==== Confusion Matrix ====")
    conf_matrix = confusion_matrix(y_true_cls, y_pred_cls)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=label_names, columns=label_names)
    print(conf_matrix_df)

    #conf_matrix_df.to_excel("confusion_matrix.xlsx", index=True)

    class_report = classification_report(y_true_cls, y_pred_cls, target_names=label_names, zero_division=0)
    class_report_for_exel = classification_report(y_true_cls, y_pred_cls, target_names=label_names, zero_division=0,
                                                  output_dict=True)
    class_report_df = pd.DataFrame(class_report_for_exel).transpose()

    print("\n==== Classification Report ====")
    print(class_report)

    #class_report_df.to_excel("classification_report.xlsx", index=True)


class GestureWindowDataset(Dataset):
    def __init__(
            self,
            entries: list[tuple[str, list[str], list[list[dict]] | None]],
            label_spec: LabelSpec,
            seq_len: int = 150,
            stride: int = 10,
            mean: np.ndarray | None = None,
            std: np.ndarray | None = None,
            compute_norm: bool = False, ) -> None:
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        x = (x - self.mean) / self.std
        return torch.from_numpy(x), torch.from_numpy(y)


def train_and_save(base_dir: str, out_path: str = "gesture_gru_single.pt"):
    files_and_labels = collect_files(base_dir)
    hidden: int = 512
    layers: int = 3
    dropout: float = 0.3

    seq_len: int = 150
    stride: int = 100

    entries = expand_squat_mixed(files_and_labels)

    all_labels = sorted({lbl for _, lbls, _ in entries for lbl in lbls})
    label_spec = LabelSpec({name: i for i, name in enumerate(all_labels)})
    print("Labels:", all_labels)

    train_entries, val_entries = train_test_split(entries, test_size=0.2, random_state=42)  # may remove shuffle

    train_ds = GestureWindowDataset(
        train_entries,
        label_spec,
        seq_len=seq_len,
        stride=stride,
        compute_norm=True
    )
    val_ds = GestureWindowDataset(
        val_entries,
        label_spec,
        seq_len=seq_len,
        stride=stride,
        mean=train_ds.mean,
        std=train_ds.std
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = GRUGestureNet(
        input_dim=FEATURE_DIM,
        num_labels=len(all_labels),
        hidden=hidden,
        layers=layers,
        dropout=dropout
    )
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
            "hidden": hidden,
            "layers": layers,
            "dropout": dropout
        },
    }
    torch.save(save_obj, out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    train_and_save(base_dir=".", out_path="gesture_gru_more_data_199.pt")
