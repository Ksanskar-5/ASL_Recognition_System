# ASL Recognition System

A real-time **American Sign Language (ASL) letter recognition** application using computer vision. Detects hand gestures via webcam and converts them into spoken words using text-to-speech.

## Features

- Real-time webcam inference with FPS monitoring
- Classifies 26 ASL hand gestures using MediaPipe + a trained KeyPointClassifier
- Supports 2 hands simultaneously
- Gesture hold logic (1.5s steady hold required to reduce false positives)
- Special gestures: `both-A` = space, `both-B` = speak the built sentence aloud
- TTS output via gTTS (plays spoken audio)
- Live sentence builder overlay on video display

## Tech Stack

- **Python 3.x**
- **MediaPipe** — hand landmark detection
- **OpenCV** — video capture & rendering
- **gTTS** — text-to-speech
- **NumPy**, **scikit-learn**

## Installation

```bash
cd Code/ASEP1_FINAL
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
cd Code/ASEP1_FINAL
python Main.py
```

- Press `ESC` to quit
- Hold both hands in `A` shape → adds a space
- Hold both hands in `B` shape → speaks the current sentence

## Project Structure

```
ASL_Recognition_System/
├── Code/
│   └── ASEP1_FINAL/
│       ├── Main.py                 # Main inference loop
│       ├── model/                  # KeyPointClassifier model + labels
│       ├── utils/                  # FPS calculator
│       └── requirements.txt
├── Docs/                           # Documentation
└── README.md
```

## Academic Context

ASEP 1 Project — Mid-Semester Assessment, December 2024
