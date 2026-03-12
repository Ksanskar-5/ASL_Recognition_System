import os
import csv
import time
import argparse
from gtts import gTTS
import cv2 as cv
import numpy as np
import mediapipe as mp

from utils.cvfpscalc import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()


def recognize_gesture_once(gesture, current_gesture, gesture_start_time, last_recognition_time, cooldown=1.6):
    """Recognize a gesture only after it's held steady for 1.5s, with a cooldown."""
    now = time.time()
    if gesture != current_gesture:
        return None, gesture, now, last_recognition_time
    elif now - gesture_start_time >= 1.5 and now - last_recognition_time >= cooldown:
        return gesture, current_gesture, gesture_start_time, now
    return None, current_gesture, gesture_start_time, last_recognition_time


def calc_bounding_rect(image, landmarks):
    h, w = image.shape[:2]
    coords = np.array([(int(l.x * w), int(l.y * h)) for l in landmarks.landmark])
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    return [x_min, y_min, x_max, y_max], coords


def pre_process_landmark(coords):
    """Normalize coordinates for model input."""
    base_x, base_y = coords[0]
    coords = coords - np.array([base_x, base_y])
    coords = coords.flatten()
    max_value = np.max(np.abs(coords)) or 1
    return (coords / max_value).tolist()


def main():
    args = get_args()

    # Initialize video capture
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # Mediapipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    # Load classifier
    classifier = KeyPointClassifier()
    with open("model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig") as f:
        labels = [row[0] for row in csv.reader(f)]

    fps_calc = CvFpsCalc(buffer_len=10)

    # Initialize state
    sentence = []
    current_gesture = ""
    gesture_start_time = time.time()
    last_recognition_time = 0
    last_spoken = ""
    both_A_time = both_B_time = single_A_time = None

    cv.namedWindow("ASL Recognition", cv.WINDOW_NORMAL)
    cv.resizeWindow("ASL Recognition", 1280, 720)

    while True:
        fps = fps_calc.get()
        if cv.waitKey(1) == 27:  # ESC to quit
            break

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv.flip(frame, 1)
        debug_image = frame.copy()

        # Process with mediapipe
        results = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        gestures = []

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect, coords = calc_bounding_rect(debug_image, hand_landmarks)
                processed = pre_process_landmark(coords)
                gesture_id = classifier(processed)
                gesture_label = labels[gesture_id]
                gestures.append(gesture_label)

                cv.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 255), 2)
                cv.putText(debug_image, f"{handedness.classification[0].label}: {gesture_label}",
                           (brect[0], brect[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Gesture logic
            now = time.time()
            if len(gestures) == 2:
                if all(g == "B" for g in gestures):
                    if both_B_time is None:
                        both_B_time = now
                    elif now - both_B_time > 1.5:
                        text = "".join(sentence)
                        if text:
                            print(f"Speaking: {text}")
                            gTTS(text=text, lang='en').save("output.mp3")
                            os.system("afplay output.mp3")
                        sentence.clear()
                        both_B_time = None
                elif all(g == "A" for g in gestures):
                    if both_A_time is None:
                        both_A_time = now
                    elif now - both_A_time > 1.5:
                        sentence.append(" ")
                        print("Added space.")
                        both_A_time = None
            else:
                both_A_time = both_B_time = None
                for g in gestures:
                    if g == "A":
                        if single_A_time is None:
                            single_A_time = now
                        elif now - single_A_time > 1.5:
                            sentence.append("A")
                            print("Added A")
                            single_A_time = None
                    else:
                        single_A_time = None
                        recognized, current_gesture, gesture_start_time, last_recognition_time = recognize_gesture_once(
                            g, current_gesture, gesture_start_time, last_recognition_time, cooldown=2
                        )
                        if recognized:
                            sentence.append(recognized)
                            print(f"Recognized: {recognized}")

        # UI overlay
        background = np.full((720, 1280, 3), (30, 30, 30), dtype=np.uint8)
        resized_debug = cv.resize(debug_image, (960, 540))
        background[90:630, 160:1120] = resized_debug
        cv.rectangle(background, (150, 80), (1130, 640), (200, 200, 200), 2)
        cv.putText(background, "ASL Recognition", (400, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(background, f"Recognized: {''.join(sentence)}", (160, 680), cv.FONT_HERSHEY_SIMPLEX, 1,
                   (255, 255, 0), 2)
        cv.putText(background, f"FPS: {fps:.1f}", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv.imshow("ASL Recognition", background)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
