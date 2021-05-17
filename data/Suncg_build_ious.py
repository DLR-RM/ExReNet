from collections import defaultdict
from pathlib import Path

import argparse
import imageio
import numpy as np
from tqdm import tqdm
import h5py
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.RMatrix import RMatrix

parser = argparse.ArgumentParser(description="")
parser.add_argument('dataset_dir')
args = parser.parse_args()

dataset = Path(args.dataset_dir)

# Intrinsics used for data generation
intrinsics = np.array([
    [105.9726, 0.0000, 63.5000],
    [0.0000, 141.2968, 63.5000],
    [0.0000, 0.0000, 1.0000]
])

for house in tqdm(dataset.iterdir()):
    # Go over all frames of that scene and collect all scene coordinates grouped by groups
    frames = house.rglob("*.hdf5")
    scene_groups = []
    last_group_id = None
    for frame in tqdm(sorted(frames, key=lambda x: int(x.name[:-len(".hdf5")]))):
        with h5py.File(str(frame), "r") as f:
            # Check if the frame has the same group id as the last one
            campose = json.loads(np.array(f["campose"]).tostring())[0]
            if campose["customprop_group_id"] != last_group_id:
                scene_groups.append([])
                last_group_id = campose["customprop_group_id"]

            # Read depth image
            depth_image = np.array(f["depth"])

            # Read pose
            pose = np.zeros((4, 4))
            pose[:3,:3] = RMatrix.from_euler(campose["rotation_euler"])
            pose[:3, 3] = campose["location"]

            # Define pixels
            stride = 32
            points = np.stack(np.meshgrid(np.arange(0, 128, stride), np.arange(0, 128, stride)), -1).astype(np.int)

            # Project pixels to local camera coordinates
            depth = depth_image[points[..., 1], points[..., 0]]
            scene_coords = np.stack([
                depth * ((points[..., 0]) - intrinsics[0][2]) / intrinsics[0][0],
                depth * ((128 - points[..., 1]) - intrinsics[1][2]) / intrinsics[1][1],
                -depth,
                np.ones_like(depth)
            ], -1)

            # Transform to world coordinates
            scene_coords = np.reshape(scene_coords, [-1, 4])
            scene_coords = np.matmul(pose, scene_coords.T).T
            scene_coords = np.reshape(scene_coords, [128, 128, 4])[..., :3]

            # Save depth and png image
            depth_image.astype(np.float32).tofile(str(frame).replace("hdf5", "raw"))
            imageio.imwrite(str(frame).replace("hdf5", "jpg"), np.array(f["colors"]))
            scene_groups[-1].append(scene_coords)

    # Go over all grouped frames
    for s, scene_group in enumerate(scene_groups):
        # Create matrix to store the intersection between all frame combinations
        ious = np.zeros((len(scene_group), len(scene_group)), dtype=np.float32)

        for i in tqdm(range(len(scene_group))):
            for j in range(len(scene_group)):
                scene_coords_i = scene_group[i][:, None]
                scene_coords_j = scene_group[j][None]

                # Compute distance between all scene coordinate combinations
                diff = np.linalg.norm(scene_coords_i - scene_coords_j, axis=-1)

                # Find for each scene coordinate the min distance to other scene coordinate
                diff_query = diff.min(axis=1)
                diff_ref = diff.min(axis=0)

                # Count the num scene coords with nearby other scene coords
                intersection_query = (diff_query < 0.2).sum()
                intersection_ref = (diff_ref < 0.2).sum()
                # Compute ratio of points that have points from the other image nearby
                ious[i, j] = (intersection_query + intersection_ref) / float(scene_group[i].shape[0] + scene_group[j].shape[0])

        # Store intersection measures in file
        np.savez(str(house / ("iou_" + str(s))), ious)
