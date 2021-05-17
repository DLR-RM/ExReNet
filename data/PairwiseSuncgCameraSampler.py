import os
import random
import time

import bpy

from src.camera.SuncgCameraSampler import SuncgCameraSampler
from src.provider.sampler.Sphere import Sphere
import numpy as np

from src.utility.CameraUtility import CameraUtility
from src.utility.EntityUtility import Entity
from src.utility.Utility import Utility


class PairwiseSuncgCameraSampler(SuncgCameraSampler):
    """ Samples valid camera poses inside suncg rooms.
    Works as the standard camera sampler, except the following differences:
    - Always sets the x and y coordinate of the camera location to a value uniformly sampled inside a rooms bounding box
    - The configured z coordinate of the configured camera location is used as relative to the floor
    - All sampled camera locations need to lie straight above the room's floor to be valid

    See parent class CameraSampler for more details.
    """

    def __init__(self, config):
        SuncgCameraSampler.__init__(self, config)

    def _sample_cam_poses(self, config):
        """ Samples camera poses according to the given config
        :param config: The config object
        """
        cam_ob = bpy.context.scene.camera
        cam = cam_ob.data

        # Set global parameters
        self._is_bvh_tree_inited = False
        self.sqrt_number_of_rays = config.get_int("sqrt_number_of_rays", 10)
        self.max_tries = config.get_int("max_tries", 100000000)
        self.proximity_checks = config.get_raw_dict("proximity_checks", {})
        self.excluded_objects_in_proximity_check = config.get_list("excluded_objs_in_proximity_check", [])
        self.min_interest_score = config.get_float("min_interest_score", 0.0)
        self.interest_score_range = config.get_float("interest_score_range", self.min_interest_score)
        self.interest_score_step = config.get_float("interest_score_step", 0.1)
        self.special_objects = config.get_list("special_objects", [])
        self.special_objects_weight = config.get_float("special_objects_weight", 2)
        self._above_objects = config.get_list("check_if_pose_above_object_list", [])
        self.check_visible_objects = config.get_list("check_if_objects_visible", [])

        if self.proximity_checks:
            # needs to build an bvh tree
            self._init_bvh_tree()

        if self.interest_score_step <= 0.0:
            raise Exception("Must have an interest score step size bigger than 0")

        # Determine the number of camera poses to sample
        number_of_poses = config.get_int("number_of_samples", 1)
        print("Sampling " + str(number_of_poses) + " cam poses")

        if self.min_interest_score == self.interest_score_range:
            step_size = 1
        else:
            step_size = (self.interest_score_range - self.min_interest_score) / self.interest_score_step
            step_size += 1  # To include last value
        # Decreasing order
        interest_scores = np.linspace(self.interest_score_range, self.min_interest_score, step_size)
        score_index = 0

        all_tries = 0  # max_tries is now applied per each score
        tries = 0

        self.min_interest_score = interest_scores[score_index]
        print("Trying a min_interest_score value: %f" % self.min_interest_score)
        for i in range(number_of_poses):
            # Do until a valid pose has been found or the max number of tries has been reached
            while tries < self.max_tries:
                tries += 1
                all_tries += 1
                start_frame = bpy.context.scene.frame_end
                # Sample a new cam pose and check if its valid
                if self.sample_and_validate_cam_pose(cam, cam_ob, config, i):
                    location = cam_ob.matrix_world.to_translation()
                    rotation = cam_ob.matrix_world.to_quaternion()

                    # Sample other camposes around it
                    for n in range(5):
                        # Try multiple times
                        for _ in range(100):
                            valid_pose, no_rotation_found = self.sample_cam_pose_nearby(cam, cam_ob, config, location, rotation)

                            # If no rotation can be found, stop right away and go to the next new pose
                            if no_rotation_found:
                                break

                            if valid_pose:
                                break

                    # Go to the next new pose
                    break

            if tries >= self.max_tries:
                if score_index == len(interest_scores) - 1:  # If we tried all score values
                    print("Maximum number of tries reached!")
                    break
                # Otherwise, try a different lower score and reset the number of trials
                score_index += 1
                self.min_interest_score = interest_scores[score_index]
                print("Trying a different min_interest_score value: %f" % self.min_interest_score)
                tries = 0

        print(str(all_tries) + " tries were necessary")


    def sample_cam_pose_nearby(self, cam, cam_ob, config, location, rotation):
        # Compute room id of last sampled pose
        group_id = cam_ob["room_id"]
        room_obj, floor_obj = self.rooms[group_id]

        # Sample/set intrinsics
        self._set_cam_intrinsics(cam, config)

        # Sample camera extrinsics multiple times until rotation diff between the new pose and the last sampled pose is small enough
        for i in range(10000):
            cam2world_matrix = self._cam2world_matrix_from_cam_extrinsics(config)

            # Compute relative rotation angle
            R1 = np.array(cam2world_matrix.to_quaternion().to_matrix())
            R2 = np.array(rotation.to_matrix())
            R_ab = np.matmul(R1.T, R2)
            angle = np.arccos(np.clip((np.trace(R_ab) - 1) / 2, -1, 1)) / np.pi * 180

            # Check if it is small enough
            if angle < 15:
                break
        # If no valid pose could have been found return
        if angle >= 15:
            return False, True

        # Sample location of new pose closely around location of last pose
        cam2world_matrix.translation = Sphere.sample(location, 0.3, "INTERIOR")

        # Check if sampled pose is valid
        if self._is_pose_valid(floor_obj, cam, cam_ob, cam2world_matrix):
            # Set camera extrinsics as the pose is valid
            frame = CameraUtility.add_camera_pose(cam2world_matrix)
            # Set group and room id keyframe (room id stays the same)
            cam_ob.keyframe_insert(data_path='["group_id"]', frame=frame)
            cam_ob.keyframe_insert(data_path='["room_id"]', frame=frame)
            return True, False
        else:
            return False, False

    def sample_and_validate_cam_pose(self, cam, cam_ob, config, group_id):
        """ Samples a new camera pose, sets the parameters of the given camera object accordingly and validates it.
        :param cam: The camera which contains only camera specific attributes.
        :param cam_ob: The object linked to the camera which determines general properties like location/orientation
        :param config: The config object describing how to sample
        :return: True, if the sampled pose was valid
        """
        ret = super().sample_and_validate_cam_pose(cam, cam_ob, config)
        if ret:
            cam_ob["group_id"] = group_id
            cam_ob.keyframe_insert(data_path='["group_id"]', frame=bpy.context.scene.frame_end - 1)
        return ret