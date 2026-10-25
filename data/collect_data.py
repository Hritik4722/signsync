"""
Interactive webcam data collector for capturing hand landmarks directly from live camera.
Can be used to record custom signs or build the dataset without external downloads.
"""

import os
import sys
import time
import argparse
import cv2
import pandas as pd

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.hand_detector import HandDetector


def collect_landmarks(letter: str, target_samples: int = 150, output_csv: str = "data/landmarks.csv"):
    letter = letter.upper().strip()
    if not letter:
        print("Please provide a valid letter/label.")
        return

    print(f"\n==========================================")
    print(f"Collecting landmarks for gesture: [{letter}]")
    print(f"Target sample count: {target_samples}")
    print("Press 'q' in the video window to stop early.")
    print("Hold your hand steady in front of the camera...")
    print(f"==========================================\n")

    detector = HandDetector(static_image_mode=False, max_num_hands=1)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Error] Could not open webcam.")
        return

    # Countdown
    for i in range(3, 0, -1):
        print(f"Starting capture in {i}...")
        time.sleep(1)

    collected_samples = []

    while len(collected_samples) < target_samples:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame for natural preview
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        detection = detector.detect(frame)

        status_text = f"Class: {letter} | Samples: {len(collected_samples)}/{target_samples}"
        color = (0, 0, 255)

        if detection["hand_detected"]:
            color = (0, 255, 0)
            collected_samples.append(detection["landmark_list"] + [letter])

            # Draw landmarks on preview
            for lm in detection["landmarks"]:
                cx, cy = int(lm[0] * w), int(lm[1] * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, "Press 'q' to quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("SignSync - Landmark Data Collector", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    if not collected_samples:
        print("No samples were collected.")
        return

    # Column names
    columns = []
    for i in range(21):
        columns.extend([f"x{i}", f"y{i}", f"z{i}"])
    columns.append("label")

    new_df = pd.DataFrame(collected_samples, columns=columns)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        # Remove old rows for this letter if re-recording, or append
        existing_df = existing_df[existing_df["label"] != letter]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(output_csv, index=False)
    print(f"\nSuccessfully saved {len(collected_samples)} samples for [{letter}] to {output_csv}")
    print(f"Total samples currently in dataset: {len(combined_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record live hand landmarks for a letter.")
    parser.add_argument("--letter", type=str, required=True, help="Letter or sign name to record (e.g. A, B, C)")
    parser.add_argument("--samples", type=int, default=150, help="Number of samples to collect (default: 150)")
    parser.add_argument("--output-csv", type=str, default="data/landmarks.csv", help="CSV path to append/save to")

    args = parser.parse_args()
    collect_landmarks(args.letter, args.samples, args.output_csv)
