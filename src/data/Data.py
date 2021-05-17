import os
import tensorflow as tf
from math import pi

from src.utils.TMatrix import TMatrix
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.set_logical_device_configuration(gpu,[tf.config.LogicalDeviceConfiguration(memory_limit=8000)])

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class Data:
    def __init__(self, config=None):
        self.config = config
        self.image_size = self.config.get_int("image_size")
        self.image_channel_num = 3
        self.cam_pose_dim = 7

        self.inverse_pose_representation = self.config.get_bool("inverse_pose_repr")

    def _prepare_cam_pose(self, reference_cam_poses, query_cam_poses):
        # Just convert rotational part to quaternions
        reference_cam_poses = TMatrix.to_quaternion(reference_cam_poses, 1)
        query_cam_poses = TMatrix.to_quaternion(query_cam_poses, 1)

        return reference_cam_poses, query_cam_poses

    def revert_cam_pose_normalization(self, cam_pose, pose_transform=None, inverse_output=True, return_euler=True):

        if pose_transform is not None:
            # Make sure the given quaternion has unit length
            cam_pose = tf.concat((cam_pose[:, :3], cam_pose[:, 3:7] / tf.linalg.norm(cam_pose[:, 3:7], axis=-1, keepdims=True), cam_pose[:, 7:]), -1)
            # Convert tmat from translation + quaternion
            cam_pose = TMatrix.from_quaternion(cam_pose[:, :self.cam_pose_dim], 1)

            # Invert the pose if necessary
            if not self.inverse_pose_representation and inverse_output:
                cam_pose = TMatrix.inverse(cam_pose, num_batch_dim=1)

            # Undo pose_transform
            pose_transform = TMatrix.inverse(pose_transform)
            cam_pose = TMatrix.apply(pose_transform, cam_pose)

            if return_euler:
                return TMatrix.to_euler(cam_pose, 1)
            else:
                return cam_pose
        else:
            return cam_pose

    def augment_image(self, image):
        # Check if we should do any augmentation
        do_brightness_augm = self.config.get_float("augmentation/brightness") > 0
        do_contrast_augm = self.config.get_list("augmentation/contrast")[0] != 1 or self.config.get_list("augmentation/contrast")[1] != 1
        do_saturation_augm = self.config.get_list("augmentation/saturation")[0] != 1 or self.config.get_list("augmentation/saturation")[1] != 1

        # Apply random brightness
        if do_brightness_augm:
            image = tf.image.random_brightness(image, self.config.get_float("augmentation/brightness"))

        # Apply random contrast
        if do_contrast_augm:
            image = tf.image.random_contrast(image, self.config.get_list("augmentation/contrast")[0], self.config.get_list("augmentation/contrast")[1])

        # Apply random saturation
        if do_saturation_augm:
            image = tf.image.random_saturation(image, self.config.get_list("augmentation/saturation")[0], self.config.get_list("augmentation/saturation")[1])

        # Make sure we get a valid image in the end
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def _augment_data(self, reference_images, reference_cam_poses, query_images, query_cam_poses, iou, room_ids, pose_transform, matching_labels):
        # Check if we should do any augmentation
        do_brightness_augm = self.config.get_float("augmentation/brightness") > 0
        do_contrast_augm = self.config.get_list("augmentation/contrast")[0] != 1 or self.config.get_list("augmentation/contrast")[1] != 1
        do_saturation_augm = self.config.get_list("augmentation/saturation")[0] != 1 or self.config.get_list("augmentation/saturation")[1] != 1

        # Augment reference and query images
        if do_brightness_augm or do_contrast_augm or do_saturation_augm:
            reference_images = self.augment_image(reference_images)
            query_images = self.augment_image(query_images)

        return reference_images, reference_cam_poses, query_images, query_cam_poses, iou, room_ids, pose_transform, matching_labels

    def preprocess_model_input(self, image):
        # Map image to 0 -> 1
        image = tf.cast(image, tf.float32) / 255.0
        # Resize to desired size (usually not necessary)
        image = tf.image.resize(image, [self.image_size, self.image_size])
        return image

    def postprocess_model_output(self, cam_pose, legacy_pose_transform=False):
        # Revert normalization of predicted cam pose
        if self.inverse_pose_representation and legacy_pose_transform:
            pose_transform = TMatrix.from_euler(tf.convert_to_tensor([[0, 0, 0, pi / 2, 0, 0]]))
        else:
            pose_transform = TMatrix.from_euler(tf.convert_to_tensor([[0, 0, 0, 0.0, 0, 0]]))
        cam_pose = self.revert_cam_pose_normalization(cam_pose, pose_transform, inverse_output=False, return_euler=False)

        return cam_pose

    def decode_img(self, img):
        # Decode image
        img = tf.image.decode_png(img, channels=3)
        # Map to [0,1] floats
        img = tf.image.convert_image_dtype(img, tf.float32)
        # Resize
        img = tf.image.resize(img, [self.image_size, self.image_size])
        return img

    def load_pair(self, pair, base_path):
        # Read in both images
        img1 = self.decode_img(tf.io.read_file(base_path + pair[0] + ".jpg"))
        img2 = self.decode_img(tf.io.read_file(base_path + pair[1] + ".jpg"))

        # Build K matrix
        intrinsics = tf.convert_to_tensor([
            [tf.strings.to_number(pair[34]), 0, tf.strings.to_number(pair[36]), 0],
            [0, tf.strings.to_number(pair[35]), tf.strings.to_number(pair[37]), 0],
            [0, 0, 1, 0],
        ])

        # Read in poses
        pose1 = tf.strings.to_number(pair[2:18])
        pose2 = tf.strings.to_number(pair[18:34])
        pose1 = tf.reshape(pose1, [4, 4])
        pose2 = tf.reshape(pose2, [4, 4])

        # Read in screen cooords of reference image
        in_file = tf.io.read_file(base_path + pair[0] + ".raw")
        depth1 = tf.reshape(tf.io.decode_raw(in_file, tf.float32), (128, 128))
        in_file = tf.io.read_file(base_path + pair[1] + ".raw")
        depth2 = tf.reshape(tf.io.decode_raw(in_file, tf.float32), (128, 128))

        scene_coords = self.calc_scene_coords(depth1, intrinsics, pose1)

        # Poses are already stored in inverse form, so if that is not desired, inverse it again
        if not self.inverse_pose_representation:
            pose1 = TMatrix.inverse(pose1, num_batch_dim=0)
            pose2 = TMatrix.inverse(pose2, num_batch_dim=0)

        # Apply inverse of reference pose to get relative pose
        pose_transform = TMatrix.inverse(pose1, num_batch_dim=0)
        pose2 = TMatrix.apply(pose_transform, pose2)
        pose1 = TMatrix.apply(pose_transform, pose1)

        return img1, pose1, img2, pose2, [[1.0]], pair[0] + " - " + pair[1], intrinsics, pose_transform, scene_coords, depth2

    def calc_scene_coords(self, depth, intrinsics, pose):
        # Create pixel mesh grid
        points = tf.cast(tf.stack(tf.meshgrid(tf.range(self.image_size), tf.range(self.image_size)), -1), tf.float32) + 0.5

        # Project pixels to local camera coordinates
        scene_coords = tf.stack([
            depth * ((points[..., 0]) - intrinsics[0][2]) / intrinsics[0][0],
            depth * ((self.image_size - points[..., 1]) - intrinsics[1][2]) / intrinsics[1][1],
            -depth,
            tf.ones_like(depth)
        ], -1)

        # Transform to world coordinates
        scene_coords = tf.reshape(scene_coords, [-1, 4])
        scene_coords = tf.transpose(tf.matmul(pose, scene_coords, transpose_b=True))
        scene_coords = tf.reshape(scene_coords, [128, 128, 4])[..., :3]
        return scene_coords

    def calc_matching_labels(self, reference_images, reference_cam_poses, query_images, query_cam_poses, iou, room_ids, intrinsics, pose_transform, scene_coord, query_depth):
        # Calculate original query pose
        if not self.inverse_pose_representation:
            query_cam_poses_orig = TMatrix.inverse(query_cam_poses, num_batch_dim=0)
        else:
            query_cam_poses_orig = query_cam_poses
        query_cam_poses_orig = TMatrix.apply(TMatrix.inverse(pose_transform, num_batch_dim=0), query_cam_poses_orig)

        # Depth to Dist
        points = tf.cast(tf.stack(tf.meshgrid(tf.range(self.image_size), tf.range(self.image_size)), -1), tf.float32) + 0.5
        points = tf.abs(points - intrinsics[:2, 2])
        query_dist = tf.sqrt(tf.square(query_depth) + tf.square(query_depth * points[..., 0] / intrinsics[0, 0]) + tf.square(query_depth * points[..., 1] / intrinsics[1, 1]))

        # Transform screen coordinates into camera coordinate system of query image
        h_scene_coord = tf.concat((scene_coord, tf.ones_like(scene_coord[..., :1])), -1)
        cam_mat = TMatrix.inverse(query_cam_poses_orig, num_batch_dim=0)
        pos_scree_space = tf.matmul(tf.cast(tf.reshape(h_scene_coord, [-1, 4]), tf.float32), cam_mat, transpose_b=True)

        # Account for different coordinate system in opencv and blender
        pos_scree_space *= [1, 1, -1, 1]

        # Project to screen coordinates
        repr_points = tf.matmul(pos_scree_space, tf.cast(intrinsics, tf.float32), transpose_b=True)
        repr_points /= repr_points[..., -1:]
        repr_points = tf.unstack(repr_points, axis=-1)
        repr_points[1] = self.image_size - repr_points[1]
        repr_points = tf.stack(repr_points, axis=-1)

        # Round the reprojected points to discrete pixels
        coord = tf.round(repr_points[..., :2] - 0.5)

        # Compute distance from scene coord to camera
        actual_depth = tf.linalg.norm(scene_coord - query_cam_poses_orig[:3, 3][None, None], axis=-1)
        # Clip coordinates
        indices = tf.reverse(tf.clip_by_value(tf.cast(coord, tf.int64), 0, self.image_size - 1), [-1])
        # Gather distance data at reprojected coordinates
        depth = tf.gather_nd(query_dist, indices)
        depth = tf.reshape(depth, [128, 128])

        # Compare distance from depth image with distance from scene coord
        diff = actual_depth - depth
        # Match is only valid if distances are similar (otherwise the scene coord is probably not visible from query view)
        valid = tf.abs(diff) < 0.05

        coord = tf.reshape(coord, [128, 128, 2])
        # Remove all invalid matches outside the image
        invalid_dest_mask = tf.reduce_all(tf.logical_and(tf.logical_and(coord >= 0, coord < 128), valid[..., None]), -1)
        coord = tf.where(tf.repeat(tf.logical_and(tf.reduce_all(scene_coord[..., :2] != 0, -1), invalid_dest_mask)[..., None], 2, -1), coord, tf.ones_like(coord) * -1)
        coord = tf.cast(coord, tf.int64)
        return reference_images, reference_cam_poses, query_images, query_cam_poses, iou, room_ids, pose_transform, coord

    def dataset_from_text_file(self, name):
        # Read all lines from the given text file (each line represents a training pair)
        pairs = []
        with open(name, "r") as f:
            pairs.extend([l.split() for l in f.readlines()])
        for pair in pairs:
            pair[0] = pair[0].replace(".jpg", "")
            pair[1] = pair[1].replace(".jpg", "")

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(pairs)
        return dataset

    def _normalize_cam_poses_with_mapping(self, reference_images, reference_cam_poses, query_images, query_cam_poses, iou, room_ids, pose_transform, matching_labels):
        reference_cam_poses, query_cam_poses = self._prepare_cam_pose(reference_cam_poses, query_cam_poses)
        return reference_images, reference_cam_poses, query_images, query_cam_poses, iou, room_ids, pose_transform, matching_labels

    def build_train_dataset(self):
        num_threads = 8
        batch_size = self.config.get_int("batch_size")

        # Read in path and poses of image pairs
        dataset = self.dataset_from_text_file(self.config.get_string("train_pair_file"))

        # Read in pair
        dataset = dataset.map(lambda x: self.load_pair(x, self.config.get_string("train_data_path")), num_parallel_calls=num_threads)

        # Calculate feature match labels
        dataset = dataset.map(self.calc_matching_labels, num_parallel_calls=num_threads)

        # Augment the images
        dataset = dataset.map(self._augment_data, num_parallel_calls=num_threads)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        # Prepare relative pose representation
        dataset = dataset.map(self._normalize_cam_poses_with_mapping, num_parallel_calls=num_threads)

        dataset = dataset.prefetch(1)
        return dataset

    def build_val_dataset(self, augment=False):
        num_threads = 8
        batch_size = self.config.get_int("val_batch_size")

        # Read in path and poses of image pairs
        dataset = self.dataset_from_text_file(self.config.get_string("val_pair_file"))

        # Read in pair
        dataset = dataset.map(lambda x: self.load_pair(x, self.config.get_string("val_data_path")), num_parallel_calls=num_threads)

        # Calculate feature match labels
        dataset = dataset.map(self.calc_matching_labels, num_parallel_calls=num_threads)

        # Augment the images
        if augment:
            dataset = dataset.map(self._augment_data, num_parallel_calls=num_threads)

        dataset = dataset.batch(batch_size, drop_remainder=True)

        # Prepare relative pose representation
        dataset = dataset.map(self._normalize_cam_poses_with_mapping, num_parallel_calls=num_threads)

        dataset = dataset.prefetch(1)
        return dataset
