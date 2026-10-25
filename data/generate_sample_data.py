"""
Generates a baseline sample landmark dataset for testing the recognition pipeline.
Provides realistic landmark poses for common ASL gestures (A, B, C, L, V, Y).
Can be used to immediately verify training and real-time inference.
"""

import os
import numpy as np
import pandas as pd


def generate_baseline_dataset(output_csv: str = "data/landmarks.csv", samples_per_class: int = 100):
    print(f"Generating synthetic baseline landmark dataset ({samples_per_class} samples/gesture)...")

    # Base hand template (wrist at (0.5, 0.8), palm center at (0.5, 0.5))
    # 21 landmarks: 0=wrist; 1-4=thumb; 5-8=index; 9-12=middle; 13-16=ring; 17-20=pinky
    base_template = np.array([
        [0.50, 0.85, 0.00],  # 0: wrist
        [0.44, 0.75, -0.02], # 1: thumb CMC
        [0.40, 0.65, -0.04], # 2: thumb MCP
        [0.37, 0.57, -0.05], # 3: thumb IP
        [0.35, 0.50, -0.06], # 4: thumb tip
        [0.46, 0.58, -0.01], # 5: index MCP
        [0.46, 0.46, -0.03], # 6: index PIP
        [0.46, 0.38, -0.04], # 7: index DIP
        [0.46, 0.30, -0.05], # 8: index tip
        [0.50, 0.56, -0.01], # 9: middle MCP
        [0.50, 0.43, -0.03], # 10: middle PIP
        [0.50, 0.34, -0.04], # 11: middle DIP
        [0.50, 0.25, -0.05], # 12: middle tip
        [0.54, 0.57, -0.01], # 13: ring MCP
        [0.54, 0.46, -0.03], # 14: ring PIP
        [0.54, 0.38, -0.04], # 15: ring DIP
        [0.54, 0.31, -0.05], # 16: ring tip
        [0.58, 0.60, 0.00],  # 17: pinky MCP
        [0.59, 0.52, -0.02], # 18: pinky PIP
        [0.60, 0.46, -0.03], # 19: pinky DIP
        [0.61, 0.40, -0.04], # 20: pinky tip
    ], dtype=np.float32)

    def fold_finger(pts, indices):
        # Bend finger joints down towards palm
        pts = pts.copy()
        mcp = pts[indices[0]]
        for step, idx in enumerate(indices[1:], start=1):
            pts[idx] = mcp + np.array([0.0, 0.06 * step, 0.04 * step], dtype=np.float32)
        return pts

    def extend_finger(pts, indices, offset_y=-0.25, offset_x=0.0):
        pts = pts.copy()
        mcp = pts[indices[0]]
        for step, idx in enumerate(indices[1:], start=1):
            ratio = step / 3.0
            pts[idx] = mcp + np.array([offset_x * ratio, offset_y * ratio, -0.02 * ratio], dtype=np.float32)
        return pts

    # Construct poses
    # Gesture 'A': fist, thumb upright beside index
    pose_a = base_template.copy()
    pose_a = fold_finger(pose_a, [5, 6, 7, 8])
    pose_a = fold_finger(pose_a, [9, 10, 11, 12])
    pose_a = fold_finger(pose_a, [13, 14, 15, 16])
    pose_a = fold_finger(pose_a, [17, 18, 19, 20])
    pose_a[4] = np.array([0.43, 0.48, -0.05], dtype=np.float32) # Thumb along side

    # Gesture 'B': flat hand upright, thumb folded in palm
    pose_b = base_template.copy()
    pose_b = extend_finger(pose_b, [5, 6, 7, 8], -0.28, 0.0)
    pose_b = extend_finger(pose_b, [9, 10, 11, 12], -0.32, 0.0)
    pose_b = extend_finger(pose_b, [13, 14, 15, 16], -0.29, 0.0)
    pose_b = extend_finger(pose_b, [17, 18, 19, 20], -0.25, 0.0)
    pose_b[4] = np.array([0.49, 0.65, 0.03], dtype=np.float32) # Thumb tucked in

    # Gesture 'L': index up, thumb out, others folded
    pose_l = base_template.copy()
    pose_l = extend_finger(pose_l, [5, 6, 7, 8], -0.30, 0.0)
    pose_l = fold_finger(pose_l, [9, 10, 11, 12])
    pose_l = fold_finger(pose_l, [13, 14, 15, 16])
    pose_l = fold_finger(pose_l, [17, 18, 19, 20])
    pose_l[4] = np.array([0.30, 0.65, -0.04], dtype=np.float32) # Thumb pointing left

    # Gesture 'V': index and middle up in V, others folded
    pose_v = base_template.copy()
    pose_v = extend_finger(pose_v, [5, 6, 7, 8], -0.28, -0.06)
    pose_v = extend_finger(pose_v, [9, 10, 11, 12], -0.29, 0.06)
    pose_v = fold_finger(pose_v, [13, 14, 15, 16])
    pose_v = fold_finger(pose_v, [17, 18, 19, 20])
    pose_v[4] = np.array([0.48, 0.62, 0.02], dtype=np.float32) # Thumb holds ring

    # Gesture 'Y': thumb and pinky extended, middle three folded
    pose_y = base_template.copy()
    pose_y = fold_finger(pose_y, [5, 6, 7, 8])
    pose_y = fold_finger(pose_y, [9, 10, 11, 12])
    pose_y = fold_finger(pose_y, [13, 14, 15, 16])
    pose_y[4] = np.array([0.28, 0.62, -0.04], dtype=np.float32) # Thumb out
    pose_y = extend_finger(pose_y, [17, 18, 19, 20], -0.20, 0.08) # Pinky out

    gestures = {
        "A": pose_a,
        "B": pose_b,
        "L": pose_l,
        "V": pose_v,
        "Y": pose_y,
    }

    records = []
    np.random.seed(42)

    for label, base_pose in gestures.items():
        for _ in range(samples_per_class):
            # Realistic sensor jitter & translation
            jitter = np.random.normal(0, 0.012, base_pose.shape)
            shift = np.random.normal(0, 0.02, (1, 3))
            sample = base_pose + jitter + shift
            records.append(sample.flatten().tolist() + [label])

    columns = []
    for i in range(21):
        columns.extend([f"x{i}", f"y{i}", f"z{i}"])
    columns.append("label")

    df = pd.DataFrame(records, columns=columns)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Generated {len(df)} samples across classes: {list(gestures.keys())}")
    print(f"Saved to: {output_csv}")
    return True


if __name__ == "__main__":
    generate_baseline_dataset()
