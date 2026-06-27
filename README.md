# VR Full-Body Tracking for Extracting Nonverbal Cues

## Project Overview

This project extracts nonverbal cues from a user's full-body movement in VR and uses them as additional context for conversations with an LLM-powered NPC.

The goal is to improve the perceived immersion and naturalness of conversing with NPCs by allowing the NPC to interpret not only what the user says, but also how they move while speaking. The extracted nonverbal cues are sent alongside the verbal conversation, enabling the NPC to better understand the user's intent and behavior.



## Supported Nonverbal Cues

The current implementation supports the following nonverbal cues:
1. **Head nod ("Yes")**
   - Large/Small movement
   - Fast/Slow
2. **Head shake ("No")**
   - Large/Small movement
   - Fast/Slow
3. **Single-hand wave**
   - Shoulder/above head height
   - Fast/Slow
4. **Two-hand wave**
   - Above head
   - Fast/Slow
5. **Squatting**
   - Squat down
   - Stand up
6. **Restless foot movement**
   - While standing
7. **User viewing position**
   - Used only within the Unity experiment

## Technologies Used

- **Python**: Training and running the GRU-based RNN for nonverbal cue extraction.
- **OpenVR**: Collects position and rotation data, along with labels, for each tracker, controller, and HMD.
- **Unity**: Runs the VR experiment and integrates the extracted nonverbal cues with the LLM-powered NPC. [Unity Experiment MVP Project](https://github.com/PilotenChris/Nonverbal-Cues-Unity-Experiment)
- **UDP**: Transmitting extracted nonverbal cues from Python to Unity or other external applications.

## Usage

### 1. Recording nonverbal cues for training
   1. Select either specific trackers or all available trackers.
   2. Two files will be created when recording selected trackers:
      - One containing only the selected trackers.
      - One containing all available trackers.
   3. Choose a filename that clearly describes the recorded nonverbal cue.
   4. The project expects predefined filenames for the six primary cues used during training. If you introduce new cues, 
   you must update both:
      - `rnnTraining.py`
      - `cuesExtractor.py`
   5. Press **Start tracking** to begin recording.
   6. Recording continues either for a predefined duration or until manually stopped, depending on the recording mode 
   configured in `cuesExtractor.py`.

### 2. Extract nonverbal cues
   1. Start the extraction program.
   2. The application continuously reads data from the VR trackers.
   3. The trained RNN predicts the active nonverbal cues.
   4. The extracted cues are transmitted over UDP.
   5. Unity, or any other application capable of receiving UDP packets, can use these cues in real time.

## Project Setup

### Requirements
- Python 3.11
- Conda (recommended)

### Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

## VR Equipment Used

- SlimeVR Full-Body Tracker Enhanced Core Set V1.2 (6+2)
- Meta Quest 3 (development)
- HTC VIVE Pro 2 (experiment)


## Future Work

- Use an avatar skeleton representation instead of raw tracker coordinates to improve nonverbal cue prediction.
  1. Improve robustness of extracting nonverbal cues across different body heights and proportions.
- Add support for face and eye tracking.
- Add support for hand tracking (Haptic Gloves) for sign language and hand movement.
- Add support for EEG cap for controlling more of the avatar, to show nonverbal cues only available to the avatar.
