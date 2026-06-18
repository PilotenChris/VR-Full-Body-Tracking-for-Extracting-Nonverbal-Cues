import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

CUE_TO_TRACKER = {
    # Head movement
    "yes_nod_small_fast": ["hmd"],
    "yes_nod_small_slow": ["hmd"],
    "yes_nod_large_fast": ["hmd"],
    "yes_nod_large_slow": ["hmd"],
    "no_nod_small_fast": ["hmd"],
    "no_nod_small_slow": ["hmd"],
    "no_nod_large_fast": ["hmd"],
    "no_nod_large_slow": ["hmd"],

    # Hand movement
    "left_hand_wave_high_fast": ["left_controller"],
    "left_hand_wave_high_slow": ["left_controller"],
    "left_hand_wave_low_fast": ["left_controller"],
    "left_hand_wave_low_slow": ["left_controller"],

    "right_hand_wave_high_fast": ["right_controller"],
    "right_hand_wave_high_slow": ["right_controller"],
    "right_hand_wave_low_fast": ["right_controller"],
    "right_hand_wave_low_slow": ["right_controller"],

    "both_hand_wave_high_fast": ["left_controller", "right_controller"],
    "both_hand_wave_high_slow": ["left_controller", "right_controller"],

    # Feet movement
    "left_foot_restless": ["left_foot"],
    "right_foot_restless": ["right_foot"],

    # Squat
    "squat_down": ["hmd", "waist", "left_knee", "right_knee"],
    "squat_up": ["hmd", "waist", "left_knee", "right_knee"]
}

FAST_SPEED_THRESHOLD: float = 0.6
MOVEMENT_THRESHOLD: float = 0.10

HIGH_MARGIN: float = 0.25


def load_json_lines(file_path: str):
    rows = []

    with open(file_path, "r") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def load_cues(file_path: str):
    data = load_json_lines(file_path)
    load_cue_df = pd.DataFrame(data)

    if "active" not in load_cue_df.columns:
        load_cue_df["active"] = [[] for _ in range(len(load_cue_df))]

    if "triggers" not in load_cue_df.columns:
        load_cue_df["triggers"] = [[] for _ in range(len(load_cue_df))]

    return load_cue_df


def load_trackers(file_path: str):
    data = load_json_lines(file_path)

    rows = []

    for entry in data:
        timestamp = entry["timestamp"]
        for device in entry["devices"]:
            role = device["role"]
            pos = device["pos"]
            quat = device["quat"]

            if pos is None or quat is None:
                rows.append({
                    "timestamp": timestamp,
                    "role": role,
                    "x": np.nan,
                    "y": np.nan,
                    "z": np.nan,
                    "qx": np.nan,
                    "qy": np.nan,
                    "qz": np.nan,
                    "qw": np.nan,
                    "valid_tracking": False
                })
                continue

            rows.append({
                "timestamp": timestamp,
                "role": role,
                "x": pos[0],
                "y": pos[1],
                "z": pos[2],
                "qx": quat[0],
                "qy": quat[1],
                "qz": quat[2],
                "qw": quat[3],
                "valid_tracking": True
            })

    print(f"{file_path}: Finished loading trackers")

    return pd.DataFrame(rows)


def add_tracker_features(df):
    tracker_feature_df = df.sort_values(["role", "timestamp"]).copy()

    tracker_feature_df["dt"] = tracker_feature_df.groupby("role")["timestamp"].diff()

    tracker_feature_df["dx"] = tracker_feature_df.groupby("role")["x"].diff()
    tracker_feature_df["dy"] = tracker_feature_df.groupby("role")["y"].diff()
    tracker_feature_df["dz"] = tracker_feature_df.groupby("role")["z"].diff()

    tracker_feature_df["distance"] = np.sqrt(tracker_feature_df["dx"]**2 + tracker_feature_df["dy"]**2 + tracker_feature_df["dz"]**2)
    tracker_feature_df["speed"] = tracker_feature_df["distance"] / tracker_feature_df["dt"]

    tracker_feature_df["vertical_speed"] = tracker_feature_df["dy"] / tracker_feature_df["dt"]

    tracker_feature_df = tracker_feature_df.replace([np.inf, -np.inf], np.nan)

    tracker_feature_df.loc[tracker_feature_df["valid_tracking"] == False, ["speed", "vertical_speed"]] = np.nan

    tracker_feature_df[["speed", "vertical_speed"]] = tracker_feature_df[["speed", "vertical_speed"]].fillna(0)

    return tracker_feature_df


def expand_cues(df, cue_names):
    expand_cue_df = df.copy()

    for cue in cue_names:
        if cue == "squat_down" or cue == "squat_up":
            expand_cue_df[cue] = expand_cue_df["triggers"].apply(lambda active: 1 if cue in active else 0)
        else:
            expand_cue_df[cue] = expand_cue_df["active"].apply(lambda active: 1 if cue in active else 0)

    return expand_cue_df


def align_data(cues_df, trackers_df, tolerance=0.03):
    aligned_roles = []

    cues_df = cues_df.sort_values("timestamp").copy()

    for role, role_df in trackers_df.groupby("role"):
        role_df = role_df.sort_values("timestamp").copy()

        merged = pd.merge_asof(
            role_df,
            cues_df,
            on="timestamp",
            direction="nearest",
            tolerance=tolerance
        )

        aligned_roles.append(merged)

    return pd.concat(aligned_roles, ignore_index=True)


def get_height_category(device_df, controller_role):
    hmd_y = device_df.loc[device_df["role"] == "hmd", "y"]
    chest_y = device_df.loc[device_df["role"] == "chest", "y"]
    controller_y = device_df.loc[device_df["role"] == controller_role, "y"]

    if hmd_y.empty or controller_y.empty:
        return "unknown"

    hmd_y = hmd_y.iloc[0]
    controller_y = controller_y.iloc[0]

    chest_y = chest_y.iloc[0] if not chest_y.empty else hmd_y - 0.35

    if controller_y >= hmd_y - HIGH_MARGIN:
        return "high"

    if chest_y - 0.20 <= controller_y < hmd_y - HIGH_MARGIN:
        return "low"

    return "too_low"


def get_speed_category(speed):
    if speed >= FAST_SPEED_THRESHOLD:
        return "fast"
    return "slow"


def evaluate_single_or_hand_foot_cue(aligned_df, cue, expected_trackers):
    cue_rows = aligned_df[aligned_df[cue] == 1].copy()

    results = []

    for timestamp, frame in cue_rows.groupby("timestamp"):
        movement = frame.set_index("role")["speed"].to_dict()

        if not movement:
            continue

        strongest_tracker = max(movement, key=movement.get)
        strongest_speed = movement[strongest_tracker]

        expected_speed = max([movement.get(t, 0) for t in expected_trackers])

        tracker_correct = expected_speed >= MOVEMENT_THRESHOLD

        height_correct = True
        speed_correct = True

        predicted_height = None
        expected_height = None

        predicted_speed_type = None
        expected_speed_type = None

        if "hand_wave" in cue:
            expected_height = "high" if "_high_" in cue else "low"
            expected_speed_type = "fast" if cue.endswith("_fast") else "slow"

            height_results = []
            speed_results = []
            movement_results = []

            controllers_to_check = (
                ["left_controller", "right_controller"]
                if "both_hand_wave" in cue
                else expected_trackers
            )

            for controller in controllers_to_check:
                controller_speed = movement.get(controller, 0)

                predicted_height = get_height_category(frame, controller)
                predicted_speed_type = get_speed_category(controller_speed)

                movement_results.append(controller_speed >= MOVEMENT_THRESHOLD)
                height_results.append(predicted_height == expected_height)
                speed_results.append(predicted_speed_type == expected_speed_type)

            tracker_correct = all(movement_results)
            height_correct = all(height_results)
            speed_correct = all(speed_results)

        correct = tracker_correct and height_correct and speed_correct

        results.append({
            "timestamp": timestamp,
            "cue": cue,
            "expected_trackers": ",".join(expected_trackers),
            "strongest_tracker": strongest_tracker,
            "expected_speed": expected_speed,
            "strongest_speed": strongest_speed,
            "tracker_correct": tracker_correct,
            "height_correct": height_correct,
            "speed_correct": speed_correct,
            "predicted_height": predicted_height,
            "expected_height": expected_height,
            "predicted_speed_type": predicted_speed_type,
            "expected_speed_type": expected_speed_type,
            "correct": correct
        })

    return pd.DataFrame(results)


def evaluate_squat_cue(aligned_df, cue):
    cue_rows = aligned_df[aligned_df[cue] == 1].copy()

    body_trackers = ["hmd", "waist", "left_knee", "right_knee"]
    results = []

    for timestamp, frame in cue_rows.groupby("timestamp"):
        frame = frame[frame["role"].isin(body_trackers)]

        if frame.empty:
            continue

        vertical = frame.set_index("role")["vertical_speed"].to_dict()

        if cue == "squat_down":
            matching_trackers = [role for role, value in vertical.items() if value < 0]
        else:
            matching_trackers = [role for role, value in vertical.items() if value > 0]

        correct = len(matching_trackers) >= 3

        results.append({
            "timestamp": timestamp,
            "cue": cue,
            "expected_trackers": ",".join(body_trackers),
            "strongest_tracker": "multi_body",
            "expected_speed": np.mean([abs(v) for v in vertical.values()]),
            "strongest_speed": np.max([abs(v) for v in vertical.values()]),
            "correct": correct,
            "matching_body_trackers": ",".join(matching_trackers),
        })

    return pd.DataFrame(results)


def evaluate_all_cues(aligned_df):
    all_results = []

    for cue, expected_trackers in CUE_TO_TRACKER.items():
        if cue not in aligned_df.columns:
            continue

        if aligned_df[cue].sum() == 0:
            continue

        if cue in ["squat_down", "squat_up"]:
            result = evaluate_squat_cue(aligned_df, cue)
        else:
            result = evaluate_single_or_hand_foot_cue(aligned_df, cue, expected_trackers)

        all_results.append(result)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def analyze_participant(participant_id, cue_file, tracker_file):
    cues = load_cues(cue_file)
    trackers = load_trackers(tracker_file)

    cues = expand_cues(cues, CUE_TO_TRACKER.keys())
    trackers = add_tracker_features(trackers)

    aligned = align_data(cues, trackers)

    evaluation = evaluate_all_cues(aligned)

    if evaluation.empty:
        return evaluation, aligned

    evaluation["participant"] = participant_id

    return evaluation, aligned


all_evaluations = []
aligned_data_per_participant = {}

for participant_id in range(1, 9):
    cue_file = f"cues_participant_slime_{participant_id}.json"
    tracker_file = f"participant_slime_{participant_id}.json"

    if not os.path.exists(cue_file) or not os.path.exists(tracker_file):
        print(f"Skipping participant {participant_id}: missing file")
        continue

    evaluation, aligned = analyze_participant(
        participant_id,
        cue_file,
        tracker_file
    )

    aligned_data_per_participant[participant_id] = aligned

    if not evaluation.empty:
        all_evaluations.append(evaluation)

results = pd.concat(all_evaluations, ignore_index=True)

cue_accuracy = (
    results.groupby("cue")["correct"]
    .mean().reset_index()
    .rename(columns={"correct": "accuracy"})
    .sort_values("accuracy", ascending=False)
)

participant_accuracy = (
    results.groupby("participant")["correct"]
    .mean().reset_index()
    .rename(columns={"correct": "accuracy"})
)

cue_participant_accuracy = (
    results.groupby(["participant", "cue"])["correct"]
    .mean().reset_index()
    .rename(columns={"correct": "accuracy"})
)

wrong_tracker_counts = (
    results[results["correct"] == False].groupby(["cue", "strongest_tracker"])
    .size().reset_index(name="count")
    .sort_values(["cue", "count"], ascending=[True, False])
)

print("\nAccuracy per cue:")
print(cue_accuracy)

print("\nAccuracy per participant:")
print(participant_accuracy)

print("\nWrong tracker counts:")
print(wrong_tracker_counts)

results.to_csv("plots/all_cue_tracker_evaluations.csv", index=False)
cue_accuracy.to_csv("plots/cue_accuracy_summary.csv", index=False)
participant_accuracy.to_csv("plots/participant_accuracy_summary.csv", index=False)
wrong_tracker_counts.to_csv("plots/wrong_tracker_counts.csv", index=False)

# Plot accuracy per cue
plt.figure(figsize=(14, 6))
plt.bar(cue_accuracy["cue"], cue_accuracy["accuracy"])
plt.xticks(rotation=90)
plt.ylabel("Accuracy")
plt.xlabel("Nonverbal cue")
plt.title("Cue-to-tracker correctness per nonverbal cue")
plt.tight_layout()
plt.savefig("plots/cue_to_tracker_correctness_per_nonverbal_cue.png", dpi=1200)
plt.show()


def plot_all_devices_and_cues(participant_id, aligned_df, cues_to_show=None):
    if cues_to_show is None:
        cues_to_show = list(CUE_TO_TRACKER.keys())

    device_cue_df = aligned_df.copy()

    if device_cue_df.empty:
        print(f"No data for participant {participant_id}")
        return

    device_cue_df["time_sec"] = device_cue_df["timestamp"] - device_cue_df["timestamp"].min()

    devices = sorted(device_cue_df["role"].dropna().unique())

    fig, axes = plt.subplots(2, 1, figsize=(30, 12), sharex=True)

    # Height graph
    for device in devices:
        device_df = device_cue_df[device_cue_df["role"] == device]
        axes[0].plot(device_df["time_sec"], device_df["y"], label=device)

    axes[0].set_title(f"Participant {participant_id}: Device height + cue triggers")
    axes[0].set_ylabel("Height / Y position")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    # Movement graph
    for device in devices:
        device_df = device_cue_df[device_cue_df["role"] == device]
        axes[1].plot(device_df["time_sec"], device_df["speed"], label=device)

    axes[1].set_ylabel("Movement speed")
    axes[1].set_xlabel("Time since recording start (seconds)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    # Cue markers
    height_marker_y = device_cue_df["y"].max()
    speed_marker_y = device_cue_df["speed"].max()

    added_labels = set()

    for cue in cues_to_show:
        if cue not in device_cue_df.columns:
            continue

        cue_times = (device_cue_df[device_cue_df[cue] == 1]["timestamp"].drop_duplicates())

        if cue_times.empty:
            continue

        cue_times_sec = cue_times - device_cue_df["timestamp"].min()

        label = cue if cue not in added_labels else None
        added_labels.add(cue)

        axes[0].scatter(
            cue_times_sec,
            [height_marker_y] * len(cue_times_sec),
            marker="x",
            s=60,
            label=label
        )

        axes[1].scatter(
            cue_times_sec,
            [speed_marker_y] * len(cue_times_sec),
            marker="x",
            s=60,
            label=label
        )

    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    plt.tight_layout()
    plt.savefig(f"plots/Movment_and_cue_participant_{participant_id}_graph.png", format="png", dpi=1200)
    plt.show()


def plot_interactive_devices_and_cues(participant_id, aligned_df, cues_to_show=None):
    if cues_to_show is None:
        cues_to_show = list(CUE_TO_TRACKER.keys())

    device_cue_df = aligned_df.copy()
    device_cue_df["time_sec"] = device_cue_df["timestamp"] - device_cue_df["timestamp"].min()

    fig = go.Figure()

    for device in sorted(device_cue_df["role"].dropna().unique()):
        device_df = device_cue_df[device_cue_df["role"] == device]

        fig.add_trace(go.Scatter(
            x=device_df["time_sec"],
            y=device_df["y"],
            mode="lines",
            name=f"{device} height",
            visible=True
        ))

        fig.add_trace(go.Scatter(
            x=device_df["time_sec"],
            y=device_df["speed"],
            mode="lines",
            name=f"{device} speed",
            visible="legendonly"
        ))

    cue_y = device_cue_df["y"].max()

    for cue in cues_to_show:
        if cue not in device_cue_df.columns:
            continue

        cue_times = device_cue_df[device_cue_df[cue] == 1]["timestamp"].drop_duplicates()
        if cue_times.empty:
            continue

        fig.add_trace(go.Scatter(
            x=cue_times - device_cue_df["timestamp"].min(),
            y=[cue_y] * len(cue_times),
            mode="markers",
            name=cue,
            marker=dict(symbol="x", size=10),
            visible="legendonly"
        ))

    fig.update_layout(
        title=f"Participant {participant_id}: Devices and Nonverbal Cues",
        xaxis_title="Time (seconds)",
        yaxis_title="Height / Speed",
        height=800
    )

    fig.write_html(f"plots/Interactive_movment_and_cue_participant_{participant_id}_graph.html")


#for participant_id, aligned in aligned_data_per_participant.items():
#    plot_all_devices_and_cues(participant_id, aligned)

#for participant_id, aligned in aligned_data_per_participant.items():
#    plot_interactive_devices_and_cues(participant_id, aligned)