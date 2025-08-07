import pickle
import sys
import os

def print_clicks(pkl_path):
    print(f"\n[INFO] Inspecting: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        demo = pickle.load(f)

    found = False
    count = 0
    for i, step in enumerate(demo):
        if 'click' in step and step['click'] is not None:
            print(f"Step {i}: click = {step['click']}")
            count += 1
            found = True

    if not found:
        print("[INFO] No 'click' values found in this demo.")
    else:
        print(f"[INFO] Total non-None clicks: {count}")

if __name__ == "__main__":
    root_dir = sys.argv[1]

    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(f"\n[INFO] Entering directory: {dirpath}")
        for file in filenames:
            if file.endswith(".pkl"):
                full_path = os.path.join(dirpath, file)
                print_clicks(full_path)
