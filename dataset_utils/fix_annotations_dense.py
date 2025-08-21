import os
import pickle
from tqdm import tqdm
from interactive_scripts.dataset_recorder import ActMode

# Folder paths
input_dir = "data/dev_cube_longhorizon_2cubes_waypoint"
output_root = "data/dev_cube_longhorizon_2cubes_allwaypoint"

# Mapping of segment name to how many chunks it should consume
segment_chunks = {
    "pick_green": 3,
    "place_green": 3,
    "pick_blue": 3,
    "place_blue": 3,
}

def find_segments(annotations):
    # Extract start indices of all waypoint annotations
    indices = [i for i, a in enumerate(annotations) if a['mode'] in [ActMode.ArmWaypoint, ActMode.BaseWaypoint]]
    
    segments = []
    for i in range(len(indices)):
        start = indices[i]
        end = indices[i + 1] if i + 1 < len(indices) else len(annotations)
        segments.append((start, end))
    return segments

def relabel_segment(segment):
    for step in segment:
        step['mode'] = ActMode.Dense
        step['waypoint_idx'] = -1
        step['click'] = None
    return segment

def save_segment(segment, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(segment, f)

def process_demo(demo_path):
    with open(demo_path, "rb") as f:
        annotations = pickle.load(f)

    demo_id = os.path.basename(demo_path)
    segments = find_segments(annotations)

    seg_pointer = 0
    for seg_name, num_chunks in segment_chunks.items():
        combined = []
        for _ in range(num_chunks):
            start, end = segments[seg_pointer]
            combined.extend(annotations[start:end])
            seg_pointer += 1
        # relabeled = relabel_segment(combined) I want waypoint right now
        out_path = os.path.join(output_root, seg_name, demo_id)
        save_segment(combined, out_path)

if __name__ == "__main__":
    for file in tqdm(sorted(os.listdir(input_dir))):
        if file.endswith(".pkl") and file.startswith("demo"):
            print("processing: ", file)
            process_demo(os.path.join(input_dir, file))
