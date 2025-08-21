import os
import pickle
import numpy as np
import glob
import argparse
from interactive_scripts.dataset_recorder import ActMode

THRESH = 1e-3

def filter_dir(SRC_DIR):
    DST_DIR = SRC_DIR + "_filtered"
    os.makedirs(DST_DIR, exist_ok=True)
    for pkl_path in sorted(glob.glob(os.path.join(SRC_DIR, "demo*.pkl"))):
        with open(pkl_path, "rb") as f:
            demo = pickle.load(f)

        print(f"{pkl_path}: loaded {len(demo)} steps")

        filtered_demo = []
        removed_count = 0

        for step in demo:
            action = np.array(step["action"])
            # Remove if *any* element is smaller than threshold in absolute value
            if step["mode"] == ActMode.Interpolate and np.any(np.abs(action) < THRESH):
                removed_count += 1
                continue
            filtered_demo.append(step)

        print(f"  removed {removed_count} steps, kept {len(filtered_demo)}")

        # Save new pkl
        dst_path = os.path.join(DST_DIR, os.path.basename(pkl_path))
        with open(dst_path, "wb") as f:
            pickle.dump(filtered_demo, f)

    print("✅ Filtering complete. Saved to", DST_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="dev1_relabeled")
    args = parser.parse_args()

    filter_dir(args.src_dir)
