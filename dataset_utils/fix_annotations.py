import os
import pickle
from interactive_scripts.dataset_recorder import ActMode


# relabeling for HoMeR
def relabel_drawer_homer(pkl_path, save_path=None):
    print(f"\n[INFO] Processing file: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        demo = pickle.load(f)
    print(f"[DEBUG] Loaded {len(demo)} steps from {pkl_path}")

    # Find waypoint-labeled segments
    mode_list = [step['mode'] for step in demo]
    waypoint_starts = []

    for i in range(len(mode_list)):
        if mode_list[i] in [ActMode.ArmWaypoint, ActMode.BaseWaypoint]:
            waypoint_starts.append(i)

    print(f"[DEBUG] Found {len(waypoint_starts)} waypoint start indices")

    # Collect full segments
    segments = []
    for start in waypoint_starts:
        end = start + 1
        while end < len(demo) and demo[end]['mode'] == ActMode.Interpolate:
            end += 1
        segments.append((start, end))

    print(f"[DEBUG] Found {len(segments)} full segments")

    # Relabel first 4 and last 4 waypoint segments to DENSE
    relabeled = segments[:4] + segments[-4:]
    print(f"[INFO] Relabeling {len(relabeled)} segments to DENSE")

    for (start, end) in relabeled:
        print(f"[DEBUG] Relabeling segment {start}–{end} to DENSE")
        for i in range(start, end):
            demo[i]['mode'] = ActMode.Dense
            demo[i]['waypoint_idx'] = -1

    # Save
    save_path = save_path or pkl_path
    with open(save_path, 'wb') as f:
        pickle.dump(demo, f)
    print(f"[INFO] Relabeled and saved to {save_path}")



## 1 single dense policy
def relabel_drawer_all_dense(pkl_path, save_path=None):
    print(f"\n[INFO] Processing file: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        demo = pickle.load(f)
    print(f"[DEBUG] Loaded {len(demo)} steps from {pkl_path}")

    # Relabel every step to DENSE
    for i, step in enumerate(demo):
        old_mode = step['mode']
        demo[i]['mode'] = ActMode.Dense
        demo[i]['waypoint_idx'] = -1
        print(f"[DEBUG] Step {i}: {old_mode} -> Dense")

    # Save relabeled demo
    save_path = save_path or pkl_path
    with open(save_path, 'wb') as f:
        pickle.dump(demo, f)
    print(f"[INFO] Relabeled and saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="data/dev_drawer_longhorizon_waypoint", help="Path to input drawer directory of .pkl files")
    parser.add_argument("--outdir", default="data/dev_drawer_longhorizon_homer_waypoint", help="Path to save relabeled .pkl files")
    args = parser.parse_args()

    print(f"[INFO] Scanning directory: {args.indir}")
    os.makedirs(args.outdir, exist_ok=True)
    count = 0

    for fname in os.listdir(args.indir):
        if fname.endswith(".pkl"):
            input_path = os.path.join(args.indir, fname)
            output_path = os.path.join(args.outdir, fname)
            print(f"\n[INFO] --- Processing {fname} ---")
            relabel_drawer_homer(input_path, output_path)
            count += 1

    print(f"\n[INFO] Done. Processed {count} files.")
