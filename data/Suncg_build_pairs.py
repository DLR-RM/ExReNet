from collections import defaultdict
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm
import sys
import argparse
import h5py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.RMatrix import RMatrix
import json

parser = argparse.ArgumentParser(description="")
parser.add_argument('dataset_dir')
parser.add_argument('pairs_output')
parser.add_argument('--min_iou', type=float, default=0.3)
parser.add_argument('--max_dist', type=float, default=0.6)
parser.add_argument('--max_angle', type=float, default=30)
args = parser.parse_args()

dataset = Path(args.dataset_dir)

# Intrinsics used for data generation
intrinsics = np.array([
    [105.9726, 0.0000, 63.5000],
    [0.0000, 141.2968, 63.5000],
    [0.0000, 0.0000, 1.0000]
])

scene_groups = []
poses = []
for house in tqdm(dataset.iterdir()):
    # Go over all frames of that scene and collect all scene coordinates grouped by groups
    frames = house.rglob("*.hdf5")
    scene_groups.append([])
    poses.append([])
    last_group_id = None
    for frame in sorted(frames, key=lambda x: int(x.name[:-len(".hdf5")])):
        with h5py.File(str(frame), "r") as f:
            # Check if the frame has the same group id as the last one
            campose = json.loads(np.array(f["campose"]).tostring())[0]
            if campose["customprop_group_id"] != last_group_id:
                scene_groups[-1].append([])
                poses[-1].append([])
                last_group_id = campose["customprop_group_id"]
            scene_groups[-1][-1].append(frame)

            # Collect pose
            pose = np.identity(4)
            pose[:3, :3] = RMatrix.from_euler(campose["rotation_euler"])
            pose[:3, 3] = campose["location"]
            poses[-1][-1].append(pose)


with open(args.pairs_output, "w") as o:
    for h, house in tqdm(enumerate(scene_groups)):
        for s, group in enumerate(house):

            # Read in intersection measures between frames
            ious = np.load(str(group[0].parent / ("iou_" + str(s) + ".npz")))["arr_0"]
            all_poses = poses[h][s]

            for i in tqdm(range(len(group)), position=0):
                others = []
                for j in range(len(group)):
                    # Compute translational and rotational distance between pair
                    dist = np.linalg.norm(all_poses[i][:3,3] - all_poses[j][:3,3])
                    angle = RMatrix.rotation_diff(all_poses[i][:3,:3], all_poses[j][:3,:3]) / np.pi * 180
                    # Check if pair is valid
                    if i !=j and ious[i,j] > args.min_iou and dist < args.max_dist and angle < args.max_angle:
                        others.append(j)

                # Sample 10 out of these valid pairs
                others = random.sample(others, k=min(len(others), 10))
                for j in others:
                    o.write(str(group[i]).replace(str(dataset) + "/", "").replace(".hdf5", ".jpg") + " " + str(group[j]).replace(str(dataset) + "/", "").replace(".hdf5", ".jpg") + " " + " ".join([str(x) for x in all_poses[i].flatten()]) + " " + " ".join([str(x) for x in all_poses[j].flatten()]) + " " + str(intrinsics[0, 0]) + " " + str(intrinsics[1, 1]) + " " + str(intrinsics[0, 2]) + " " + str(intrinsics[1, 2]) + "\n")
