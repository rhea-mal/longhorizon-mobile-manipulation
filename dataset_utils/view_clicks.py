import pickle
import sys
import numpy as np
import pdb

# def print_clicks(pkl_path):
#     with open(pkl_path, 'rb') as f:
#         demo = pickle.load(f)

#     print(f"\n[INFO] Inspecting: {pkl_path}")
#     found = False
#     count = 0

#     for i, step in enumerate(demo):
#         if 'click' in step and step['click'] is not None:
#             print(f"Step {i}: click = {step['click']}")
#             count += 1
#             found = True

#     if not found:
#         print("[INFO] No 'click' values found in this demo.")
#     else:
#         print(f"[INFO] Total non-None clicks: {count}")

# if __name__ == "__main__":
#     path = "data/dev_cube_wbc/demo00011.pkl"
#     print_clicks(path)

import os
import pickle
import numpy as np
import sys

def check_pointcloud(pkl_path):
    with open(pkl_path, 'rb') as f:
        try:
            demo = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Couldn't load {pkl_path}: {e}")
            return

    print(f"\n[INFO] Checking point clouds in: {pkl_path}")
    for i, step in enumerate(demo):
        pdb.set_trace()
        if 'xyz' not in step:
            print(f"  ⚠️ Step {i}: No 'xyz' key found.")
            continue

        xyz = np.array(step['xyz'])
        if not np.all(np.isfinite(xyz)):
            nan_mask = ~np.isfinite(xyz)
            num_nans = np.sum(nan_mask)
            print(f"  ❌ Step {i}: {num_nans} invalid (NaN or Inf) values in 'xyz'")
            print(f"     Example bad entries: {xyz[nan_mask][:5]}")

def scan_folder(folder):
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".pkl"):
            check_pointcloud(os.path.join(folder, filename))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_pointcloud_nan.py path/to/demo_folder")
        sys.exit(1)

    scan_folder(sys.argv[1])

