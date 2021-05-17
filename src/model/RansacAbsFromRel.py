import itertools

import numpy as np
import tensorflow as tf

from src.utils.RMatrix import RMatrix
from src.utils.TMatrix import TMatrix
from src.utils.triangulation.Triangulation import Triangulation


class RansacAbsFromRel():
    def __init__(self, rel_pose_estimation_model, use_scale=True, use_uncertainty=True, debug=False):
        self.rel_pose_estimation_model = rel_pose_estimation_model
        self.use_scale = use_scale
        self.use_uncertainty = use_uncertainty
        self.debug = debug

    #@tf.function
    def abs_pose_from_pair(self, ref_to_query_1, ref_pose_1, ref_to_query_2, ref_pose_2):
        """ Triangulate an absolute pose estimate from the two given relative pose estimates """
        # Compute all possible estimates for the rotation of the query image (for ExReNet R1 = R2)
        R_r1 = RMatrix.inverse(ref_pose_1["R"])
        dR1_q_r1 = ref_to_query_1["R1"]
        dR2_q_r1 = ref_to_query_1["R2"]

        R1_q_r1 = RMatrix.apply(dR1_q_r1, R_r1)
        R2_q_r1 = RMatrix.apply(dR2_q_r1, R_r1)

        R_r2 = RMatrix.inverse(ref_pose_2["R"])
        dR1_q_r2 = ref_to_query_2["R1"]
        dR2_q_r2 = ref_to_query_2["R2"]

        R1_q_r2 = RMatrix.apply(dR1_q_r2, R_r2)
        R2_q_r2 = RMatrix.apply(dR2_q_r2, R_r2)

        pairs = [(R1_q_r1, R1_q_r2), (R1_q_r1, R2_q_r2), (R2_q_r1, R1_q_r2), (R2_q_r1, R2_q_r2)]

        # Compute difference between all rotation pairs
        diffs = []
        for pair in pairs:
            diffs.append(RMatrix.rotation_diff(pair[0], pair[1]))

        # Translation directions
        dir_1 = np.expand_dims(ref_to_query_1["t"], 1)
        dir_2 = np.expand_dims(ref_to_query_2["t"], 1)

        # Triangulate
        t_abs = Triangulation.triangulate(
            ref_pose_1["t"],
            RMatrix.apply(RMatrix.inverse(R_r1), RMatrix.apply(RMatrix.inverse(dR1_q_r1), dir_1))[:3, 0],
            ref_pose_2["t"],
            RMatrix.apply(RMatrix.inverse(R_r2), RMatrix.apply(RMatrix.inverse(dR1_q_r2), dir_2))[:3, 0],
        )

        # Use rotation estimates which are closest
        R1_abs = np.stack([p[0] for p in pairs], 0)[np.argmin(diffs)]
        R2_abs = np.stack([p[1] for p in pairs], 0)[np.argmin(diffs)]

        return {"t": t_abs, "R1": R1_abs, "R2": R2_abs}

    def _run_model(self, ref_obs, query_obs, use_uncertainty, legacy_pose_transform=False):
        # Prepare input
        selected_query_obs = query_obs["image"]
        selected_ref_obs = [r["image"] for r in ref_obs]

        # Run relative pose estimation model
        ref_to_query_T, uncertainty = self.rel_pose_estimation_model.predict_using_raw_data(selected_ref_obs, selected_query_obs, use_uncertainty, legacy_pose_transform)

        return ref_to_query_T, uncertainty

    def predict(self, ref_obs, reference_cam_poses, query_obs, query_pose, legacy_pose_transform=False):
        # Get relative pose estimates and optional also uncertainty
        ref_to_query, t_uncertainty = self._run_model(ref_obs, query_obs, self.use_uncertainty, legacy_pose_transform)

        # Sort out invalid pose estimations
        valid_pairs = []
        for i, delta_pair in enumerate(ref_to_query):
            if delta_pair is not None:
                valid_pairs.append(i)

        # We need at least two
        if len(valid_pairs) < 2:
            return None


        best_estimate = None
        max_inlier = -1
        max_inlier_mean_uncertainty = None
        with tf.device("/cpu:0"):
            # Go over all combination of pairs
            pairs = list(itertools.combinations(valid_pairs, 2))
            for i, j in pairs:
                # Triangulate absolute pose based on the two estimates
                abs_pose_est = self.abs_pose_from_pair(ref_to_query[i], reference_cam_poses[i], ref_to_query[j], reference_cam_poses[j])

                # Go over all reference images
                inliers = 0
                for k in valid_pairs:
                    third_pair = ref_to_query[k]

                    # Compute relative rotation between pose prediction and ref image
                    R_r3 = RMatrix.inverse(reference_cam_poses[k]["R"])
                    dR1_q_r3 = third_pair["R1"]

                    # Compute relative translation between pose prediction and ref image
                    dir_3 = np.expand_dims(abs_pose_est["t"] - reference_cam_poses[k]["t"], 1)
                    t_pred = RMatrix.apply(R_r3, dir_3)[:3, 0]

                    # Compute predicted relative translation by ref image
                    tk = np.expand_dims(third_pair["t"], 1)
                    tk = -RMatrix.apply(RMatrix.inverse(dR1_q_r3), tk)[:3, 0]

                    # Compuate angle between
                    alpha = np.arccos(np.clip(np.dot(t_pred, tk) / (np.linalg.norm(t_pred) * np.linalg.norm(tk)), -1, 1))
                    alpha = alpha / np.pi * 180

                    # Check if we count it as an inlier:
                    # - check that angle < 15deg
                    # - check that 0.5 < (actual scale / predicted scale) < 2 (optional)
                    s_min = 0.5
                    s_max = 2
                    if alpha < 15 and (not self.use_scale or (s_min < np.linalg.norm(abs_pose_est["t"] - reference_cam_poses[k]["t"]) / np.linalg.norm(third_pair["t"]) < s_max or abs(np.linalg.norm(abs_pose_est["t"] - reference_cam_poses[k]["t"]) - np.linalg.norm(third_pair["t"])) < 0)):
                        inliers += 1

                # Compute uncertainty of hypothesis
                if self.use_uncertainty:
                    mean_uncertainty = t_uncertainty[i] + t_uncertainty[j]
                else:
                    mean_uncertainty = None
                # Check if there has been a hypothesis with more inlier or higher uncertainty
                if inliers > max_inlier or (self.use_uncertainty and inliers == max_inlier and max_inlier_mean_uncertainty > mean_uncertainty):
                    max_inlier = inliers
                    best_estimate = abs_pose_est
                    max_inlier_mean_uncertainty = mean_uncertainty

                if self.debug:
                    print(inliers, i, j, np.linalg.norm(abs_pose_est["t"] - query_pose[:3]), 180.0 / np.pi * RMatrix.rotation_diff(query_pose[:3, :3], RMatrix.inverse(RMatrix.from_quaternion(RMatrix.average_quaternions(np.stack((RMatrix.to_quaternion(abs_pose_est["R1"]), RMatrix.to_quaternion(abs_pose_est["R2"])), 0))))), mean_uncertainty)

            if best_estimate is None:
                return None

            # Take the mean of the two rotation estimates of query image
            best_rot = RMatrix.from_quaternion(RMatrix.average_quaternions(np.stack((RMatrix.to_quaternion(best_estimate["R1"]), RMatrix.to_quaternion(best_estimate["R2"])), 0)))

            # Return the best estimate
            return {"t": best_estimate["t"], "R": RMatrix.inverse(best_rot)}

