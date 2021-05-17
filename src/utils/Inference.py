# coding=utf-8
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

import h5py
import numpy as np

from pathlib import Path
from src.utils.RMatrix import RMatrix
from tqdm import tqdm
from src.model.RansacAbsFromRel import RansacAbsFromRel
import random
import json

import cv2


class Inference:
    def __init__(self, reference_root, query_root, rel_pose_model, scale=False, uncertainty=False, legacy_pose_transform=False):
        print("Scale:" + str(scale) + " Uncertainty: " + str(uncertainty))
        self.reference_root = Path(reference_root)
        self.query_root = Path(query_root)
        self.rel_pose_model = rel_pose_model
        self.scale = scale
        self.uncertainty = uncertainty

        self.retr_a = 0.05
        self.retr_b = 10

        # Prepare cache for image retrieval
        self.ir_cache_path = Path("ir_cache/" + self.query_root.parent.parent.name + "-" + self.query_root.parent.name + ".json")
        print(self.ir_cache_path)
        self.ir_cache = {}
        if self.ir_cache_path.exists():
            with open(str(self.ir_cache_path), "r") as f:
                self.ir_cache = json.load(f)
        self.ir_cache_changed = False
        self.legacy_pose_transform = legacy_pose_transform

    def read_hdf(self, path):
        with h5py.File(str(path), 'r') as f:
            obs = {}

            # Read RGB image and resize it
            image = f['colors'][:]
            if self.rel_pose_model.data.image_size != 448:
                image = cv2.resize(image, dsize=(128, 128) if self.rel_pose_model.data.image_size == 128 else (341, 256), interpolation=cv2.INTER_CUBIC)#341, 256
            obs["image"] = image

            # Read pose
            pose = {"R": np.array(f['campose'])[:3, :3], "t": np.array(f['campose'])[:3, 3]}
            # Read densevlad descriptor if available
            if "densevlad" in f:
                retrieval_encoding = f["densevlad"][:]
            else:
                retrieval_encoding = None

        return obs, pose, retrieval_encoding

    def calc_error(self, label, pred):
        # Compute translational difference
        t = np.linalg.norm(label["t"] - pred["t"])
        # Compute rotational difference between two poses
        angle = RMatrix.rotation_diff(label["R"], pred["R"])
        return t, angle

    def calc_similarity(self, reference_lookup, query_encoding):
        return -np.linalg.norm(reference_lookup - query_encoding, axis=-1)

    def check_dist_to_others(self, location, selected_refs):
        for selected_ref in selected_refs:
            dist = np.linalg.norm(location - selected_ref[1])
            if dist < self.retr_a or dist > self.retr_b:
                return False
        return True

    def select_ref_images(self, query_path, query_encoding, query_index, num_refs=5):
        # If query image is in the cache, just use that
        if str(query_index) in self.ir_cache:
            return self.ir_cache[str(query_index)]

        # Calc similarity between query encoding and the encodings of all reference images
        similarity = self.calc_similarity(self.reference_lookup, query_encoding)

        # Add indices
        similarity = np.stack((similarity, np.arange(len(similarity))), -1)

        # Sort by similarity
        similarity = similarity[np.lexsort((similarity[:, 1], -similarity[:,0]))]

        # Go over all reference images (sorted)
        selected_refs = []
        for i in range(len(similarity)):
            # Build reference path
            ref_index = int(similarity[i][1])
            path = self.reference_root / (str(ref_index) + ".hdf5")
            if path != query_path:
                # Read in location of reference cam pose
                with h5py.File(str(path), 'r') as f:
                    location = np.array(f['campose'])[:3, 3]

                # Check that it is in the right distance to already selected reference images
                if self.check_dist_to_others(location, selected_refs):
                    # Select it
                    selected_refs.append([ref_index, location])

                    # Stop if we have enough
                    if len(selected_refs) > num_refs:
                        break

        # Remember the retrieval in the cache
        selected_refs = [x[0] for x in selected_refs]
        self.ir_cache[query_index] = selected_refs
        self.ir_cache_changed = True
        return selected_refs

    def run(self):
        # Read in densevlad lookup table (necessary if there is no cache)
        if (self.reference_root / ("densevlad_lookup.npz")).exists():
            self.reference_lookup = np.load(str(self.reference_root / ("densevlad_lookup.npz")))["arr_0"]
        else:
            self.reference_lookup = None

        # Collect query images
        query_paths = list(self.query_root.rglob("*.hdf5"))
        query_paths = sorted(query_paths, key=lambda x: x.name)

        # Set seed
        random.seed(42)
        np.random.seed(42)

        # Init error collection
        errors_t, errors_angle = [], []

        # Create absolute pose estimation model
        model = RansacAbsFromRel(self.rel_pose_model, self.scale, self.uncertainty)

        #  Go over all query images
        successful = 0
        total = 0
        for i, query_path in enumerate(tqdm(query_paths)):
            # Read in query image and pose
            query_obs, query_pose, query_encoding = self.read_hdf(query_path)
            # Do image retrieval
            selected_refs = self.select_ref_images(query_path, query_encoding, int(query_path.name[:-len(".hdf5")]))

            # Read in retrieved reference images
            ref_observations = []
            ref_poses = []
            for selected_ref in selected_refs:
                ref_obs, ref_pose, _ = self.read_hdf(self.reference_root / (str(selected_ref) + ".hdf5"))

                ref_observations.append(ref_obs)
                ref_poses.append(ref_pose)

            # Predict pose of query image
            pred_cam_poses = model.predict(ref_observations, ref_poses, query_obs, query_pose, self.legacy_pose_transform)

            # If there is a valid prediction
            if pred_cam_poses is not None:
                # Calc deviation from ground truth
                t, angle = self.calc_error(query_pose, pred_cam_poses)

                # Log deviation
                errors_t.append(t)
                errors_angle.append(angle)
                successful += 1

            total += 1

        # Write ir cache if it has changed
        if self.ir_cache_changed:
            print("Writing ir cache")
            with open(str(self.ir_cache_path), "w") as f:
                json.dump(self.ir_cache, f)

        # Compute median and success rate
        return float(successful) / total, np.median(errors_t), np.median(errors_angle) / np.pi * 180
