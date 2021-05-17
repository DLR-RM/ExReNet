from collections import defaultdict
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.RMatrix import RMatrix
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('dataset_dir')
parser.add_argument('pairs_output')
parser.add_argument('--min_iou', type=float, default=0.3)
parser.add_argument('--max_dist', type=float, default=0.6)
parser.add_argument('--max_angle', type=float, default=30)
args = parser.parse_args()

dataset = Path(args.dataset_dir)

scene_groups = defaultdict(list)

for path in dataset.iterdir():
    if path.name.startswith("scene"):
        scene_groups[path.name[:path.name.find("_")]].append(path.name)

# Define transformation to correct coordinate frame
frame_change = RMatrix.from_euler(np.array([0, np.pi, np.pi]))
frame_change = np.pad(frame_change, [[0, 1], [0, 1]])
frame_change[3, 3] = 1

# Define the intrinsics the images now have after resizing
target_intrinsics = np.array([
    [105.9726, 0.0000, 63.5000],
    [0.0000, 141.2968, 63.5000],
    [0.0000, 0.0000, 1.0000]
])

with open(args.pairs_output, "w") as o:
    for scene_name, scene_group in tqdm(scene_groups.items(), position=1):
       
        for trajectory in sorted(scene_group):

            # Collect the poses of all frames
            all_filenames = []
            all_poses = []
            for frame in sorted((dataset / trajectory / "color").iterdir()):
                pose = []
                with open(str(dataset / trajectory / "pose" / frame.name.replace("jpg", "txt")), "r") as f:
                    for line in f.readlines():
                        pose.append(line.split())

                pose = np.array(pose, dtype=np.float)
                pose = np.matmul(pose, frame_change)

                if np.isnan(pose).any() or np.isinf(pose).any():
                    print(pose)
                    print("skipping: " + str(frame))
                    continue

                all_filenames.append(frame)
                all_poses.append(pose)

            # Read in intersection measures between frames
            ious = np.load(str(dataset / trajectory / "iou.npz"))["arr_0"]

            for i in tqdm(range(len(all_filenames)), position=0):
                others = []
                for j in range(len(all_filenames)):
                    # Compute translational and rotational distance between pair
                    dist = np.linalg.norm(all_poses[i][:3,3] - all_poses[j][:3,3])
                    angle = RMatrix.rotation_diff(all_poses[i][:3,:3], all_poses[j][:3,:3]) / np.pi * 180
                    # Check if pair is valid
                    if i != j and ious[i, j] > args.min_iou and dist < args.max_dist and angle < args.max_angle:
                        others.append(j)

                # Sample 10 out of these valid pairs
                others = random.sample(others, k=min(len(others), 10))
                for j in others:
                    o.write(str(all_filenames[i]).replace(str(dataset) + "/", "") + " " + str(all_filenames[j]).replace(str(dataset) + "/", "") + " " + " ".join([str(x) for x in all_poses[i].flatten()]) + " " + " ".join([str(x) for x in all_poses[j].flatten()]) + " " + str(target_intrinsics[0, 0]) + " " + str(target_intrinsics[1, 1]) + " " + str(target_intrinsics[0, 2]) + " " + str(target_intrinsics[1, 2]) + "\n")
