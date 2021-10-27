import os
import cv2
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.spatial.transform import Rotation as R

import imageio
import json
import h5py
import math
import argparse

from src.utils.RMatrix import RMatrix

parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
args = parser.parse_args()

# Go over all scenes
for dataset in ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]:
    path = os.path.join(args.data_dir, dataset)
    if os.path.isdir(path):
        # Collect test and train seq
        with open(os.path.join(args.data_dir, path, "TrainSplit.txt"), "r") as f:
            train_set = [int(s.strip()[len("sequence"):]) for s in f.readlines()]
        with open(os.path.join(args.data_dir, path, "TestSplit.txt"), "r") as f:
            test_set = [int(s.strip()[len("sequence"):]) for s in f.readlines()]

        last_org_mat = None
        last_new_mat = None
        # Go over all train/test seq
        for set_name, seq_ids in {"train": train_set, "test": test_set}.items():
            hdf_id = 0
            for seq_id in seq_ids:
                seq_path = os.path.join(path, "seq-" + ("%02d" % seq_id))
                # Go over all frames
                for frame_id in range(1000):
                    # Make sure frame exists
                    pose_path = os.path.join(seq_path, "frame-" + ("%06d" % frame_id) + ".pose.txt")
                    if os.path.isfile(pose_path):
                        print(pose_path)
                        with open(pose_path, "r") as f:
                            # Read pose
                            mat = []
                            for line in f.readlines():
                                mat.append(line.split())

                            new_mat = np.identity(4)
                            new_mat[:3,:3] = RMatrix.from_euler([np.pi, 0, 0])
                            campose = np.matmul(np.array(mat, np.float32), new_mat)

                            rgb = imageio.imread(os.path.join(seq_path, "frame-" + ("%06d" % frame_id) + ".color.png"))
                            rgb = cv2.resize(rgb, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

                        hdf5_path = os.path.join(args.data_dir, dataset, set_name)
                        if not os.path.exists(hdf5_path):
                            os.makedirs(hdf5_path)

                        # Put everything into one hdf5 file
                        with h5py.File(os.path.join(hdf5_path, str(hdf_id) + ".hdf5"), "w") as f:
                            for key, data in {"colors": rgb, "campose": campose}.items():
                                f.create_dataset(key, data=data, compression='gzip')

                        hdf_id += 1
