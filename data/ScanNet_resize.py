import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from src.utils.RMatrix import RMatrix

parser = argparse.ArgumentParser(description="")
parser.add_argument('dataset_dir')
parser.add_argument('target_size', type=int)
parser.add_argument('target_focal_length', type=float)
args = parser.parse_args()

dataset = Path(args.dataset_dir)
output = Path(dataset.parent / (dataset.name + "_" + str(args.target_size)))
output.mkdir(exist_ok=True)

# Collect all frame corresponding to one scene
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

        # Compute resize fac to get to desired focal length
        fac = args.target_focal_length / intrinsics[0,0]
        dest_size = (int(fac * 640), int(fac * 480))
        print(dest_size)

        (output / trajectory / "color").mkdir(exist_ok=True, parents=True)

        for frame in sorted((dataset / trajectory / "color").iterdir()):
            # Read rgb
            image = imageio.imread(str(frame))
            # Read depth
            depth_image = imageio.imread(dataset / trajectory / "depth" / frame.name.replace("jpg", "png"))

            # Resize rgb
            image = cv2.resize(image, dest_size)
            # Embed into original resolution, s.t. we get black border around image
            pad_image = np.zeros((480, 640, 3), dtype=np.uint8)
            pad_image[(480 - image.shape[0]) // 2:(480 - image.shape[0]) // 2 + image.shape[0], (640 - image.shape[1]) // 2:(640 - image.shape[1]) // 2 + image.shape[1]] = image

            # Resize image to target resolution
            image = cv2.resize(pad_image, (args.target_size, args.target_size))

            # Store new rgb image
            imageio.imwrite(str(frame).replace(str(dataset), str(output)), image)

            # Compute mapping from pixel coordinates of new resolution to original resolution
            points = np.stack(np.meshgrid(np.arange(args.target_size), np.arange(args.target_size)), -1) + 0.5
            border = np.array([round((640 - dest_size[0]) // 2 * args.target_size / 640), round((480 - dest_size[1]) // 2 * args.target_size / 480)])
            points -= border[None, None]
            points *= np.array([640 / (args.target_size - border[0] * 2), 480 / (args.target_size - border[1] * 2)])[None, None]
            points = np.round(points).astype(np.int)

            # Mask out invalid pixels
            mask = np.stack((np.logical_and(points[..., 0] >= 0, points[..., 0] < 640), np.logical_and(points[..., 1] >= 0, points[..., 1] < 480)), -1).all(axis=-1)[..., None]
            points *= mask

            # Use that mapping to define depth image of resized rgb image
            depth = depth_image[points[..., 1], points[..., 0]] / 1000.0

            # Write depth image to file
            depth.astype(np.float32).tofile(str(frame).replace(str(dataset), str(output)).replace("jpg", "raw"))
