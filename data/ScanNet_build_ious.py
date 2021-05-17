from collections import defaultdict
from pathlib import Path

import argparse
import imageio
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="")
parser.add_argument('dataset_dir')
args = parser.parse_args()

dataset = Path(args.dataset_dir)

# Collect all frames grouped by scene
scene_groups = defaultdict(list)
for path in dataset.iterdir():
    if path.name.startswith("scene"):
        scene_groups[path.name[:path.name.find("_")]].append(path.name)

for scene_name, scene_group in tqdm(scene_groups.items(), position=1):
    for trajectory in sorted(scene_group):

        # Read in intrinsics
        intrinsics = []
        with open(str(dataset / trajectory / "intrinsic" / "intrinsic_depth.txt"), "r") as f:
            for line in f.readlines():
                intrinsics.append(line.split())
        intrinsics = np.array(intrinsics, dtype=np.float)

        # Collect all frames from the trajectory
        all_poses = []
        all_filenames = []
        all_scene_coords = []
        for frame in sorted((dataset / trajectory / "color").iterdir()):
            # Read depth
            depth_image = imageio.imread(dataset / trajectory / "depth" / frame.name.replace("jpg", "png"))

            # Read in and validate pose
            pose = []
            with open(str(dataset / trajectory / "pose" / frame.name.replace("jpg", "txt")), "r") as f:
                for line in f.readlines():
                    pose.append(line.split())
            pose = np.array(pose, dtype=np.float)
            if np.isnan(pose).any() or np.isinf(pose).any():
                print(pose)
                print("skipping: " + str(frame))
                continue

            all_filenames.append(frame)
            all_poses.append(pose)

            # Project pixels to get some scene coordinates to later calc ious
            stride = 32
            scene_coords = np.zeros([(depth_image.shape[0] // stride) * (depth_image.shape[1] // stride), 4])
            scene_coords[:, 3] = 1
            p = 0
            for i in range(0, depth_image.shape[0], stride):
                for j in range(0, depth_image.shape[1], stride):
                    depth = depth_image[i, j] / 1000.0
                    if depth > 0:
                        scene_coords[p, 0] = depth * (j - intrinsics[0][2]) / intrinsics[0][0]
                        scene_coords[p, 1] = depth * (i - intrinsics[1][2]) / intrinsics[1][1]
                        scene_coords[p, 2] = depth

                        p += 1
            scene_coords = scene_coords[:p]

            # Transform to world coordinates
            scene_coords = np.matmul(pose, scene_coords.T).T
            all_scene_coords.append(scene_coords)

        # Create matrix to store the intersection between all frame combinations
        ious = np.zeros((len(all_filenames), len(all_filenames)), dtype=np.float32)
        for i in tqdm(range(len(all_filenames)), position=0):
            for j in range(len(all_filenames)):
                scene_coords_i = all_scene_coords[i][:,None]
                scene_coords_j = all_scene_coords[j][None]

                # Compute distance between all scene coordinate combinations
                diff = np.linalg.norm(scene_coords_i - scene_coords_j, axis=-1)

                # Find for each scene coordinate the min distance to other scene coordinate
                diff_query = diff.min(axis=1)
                diff_ref = diff.min(axis=0)

                # Count the num scene coords with nearby other scene coords
                intersection_query = (diff_query < 0.2).sum()
                intersection_ref = (diff_ref < 0.2).sum()
                # Compute ratio of points that have points from the other image nearby
                ious[i,j] = (intersection_query + intersection_ref) / float(all_scene_coords[i].shape[0] + all_scene_coords[j].shape[0])

        # Store intersection measures in file
        np.savez(str(dataset / trajectory / "iou"), ious)

