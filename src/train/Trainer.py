import tensorflow as tf

from src.train.Metric import Metric

from src.utils.Utility import Utility


class Trainer:

    def __init__(self, config, model, data):
        self.config = config
        self.model = model
        self.data = data

        self._build()

    def _build(self):
        # Build optimizer
        lr = self.config.get_float("optimizer/lr")
        optimizer = self.config.get_string("optimizer/name")
        if optimizer == "adam":
            self.optimizer = tf.keras.optimizers.Adam(lr)
        elif optimizer == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(lr, self.config.get_float("optimizer/mom"))
        else:
            raise Exception("No such optimizer: " + optimizer)
        self.aux_loss_weight = self.config.get_float("aux_loss_weight")

        # Build metrics
        self.train_metric = Metric(self.data, "train/")
        self.val_metric = Metric(self.data, "val/")

        # Init data loaders
        self.train_dataset = self.data.build_train_dataset()
        self.val_dataset = self.data.build_val_dataset()
        self.train_iter = iter(self.train_dataset)
        self.val_iter = iter(self.val_dataset)

    def _regression_pose_loss(self, label_cam_pose, pred_cam_pose, metric, pose_transform):
        # If we predict translation direction
        num_of_location_params = 3
        if self.config.get_bool("provide_translation_scale"):
            # Normalize translational part in label
            len_label = Utility.norm_kdf(label_cam_pose[:, :num_of_location_params])
            unstacked_label = tf.unstack(label_cam_pose, axis=1)
            for i in range(3):
                unstacked_label[i] /= len_label
            label_cam_pose_for_diff = tf.stack(unstacked_label, axis=1)

            # If scale is predicted extra, add it as additional channel
            if self.model.config.get_bool("pred_scale_extra"):
                label_cam_pose_for_diff = tf.concat((label_cam_pose_for_diff, tf.expand_dims(len_label, 1)), axis=-1)
        else:
            label_cam_pose_for_diff = label_cam_pose

        # Check which quaternion is closer to the ground truth: q or -q
        dist_1 = tf.reduce_mean(tf.abs(pred_cam_pose[..., 3:7] / tf.linalg.norm(pred_cam_pose[..., 3:7], axis=-1, keepdims=True) - label_cam_pose_for_diff[..., 3:7]), -1, keepdims=True)
        dist_2 = tf.reduce_mean(tf.abs(-pred_cam_pose[..., 3:7] / tf.linalg.norm(pred_cam_pose[..., 3:7], axis=-1, keepdims=True) - label_cam_pose_for_diff[..., 3:7]), -1, keepdims=True)

        # Use the one closer as prediction
        pred_cam_pose = tf.concat((pred_cam_pose[..., :3], tf.where(dist_1 < dist_2, pred_cam_pose[..., 3:7], -pred_cam_pose[..., 3:7]), pred_cam_pose[..., 7:]), -1)

        # Compute absolute error
        diff = tf.abs(label_cam_pose_for_diff - pred_cam_pose)

        # Square if desired
        if self.config.get_bool("squared_loss"):
            diff = tf.square(diff)

        # Compute translational and rotational loss
        pose_loss = tf.reduce_mean(diff[..., :3]) + tf.reduce_mean(diff[..., 3:])

        # If the translation direction was predicted
        if self.config.get_bool("provide_translation_scale"):
            # Scale up predicted translation direction to get a fully scaled translation estimate
            len_pred = Utility.norm_kdf(pred_cam_pose[:, :num_of_location_params])
            len_label = Utility.norm_kdf(label_cam_pose[:, :num_of_location_params])

            unstacked_pred = tf.unstack(pred_cam_pose, axis=1)
            for i in range(num_of_location_params):
                # If available use predicted scale to scale it up
                if not self.model.config.get_bool("pred_scale_extra"):
                    unstacked_pred[i] *= len_label / len_pred
                else:
                    unstacked_pred[i] *= unstacked_pred[-1] / len_pred

            # Remove scale prediction channel
            if self.model.config.get_bool("pred_scale_extra"):
                unstacked_pred.pop()
            pred_cam_pose = tf.stack(unstacked_pred, axis=1)

        # Log pose prediction metrics
        metric.update_cam_pose_metric(pred_cam_pose, label_cam_pose, False, pose_transform)

        return pose_loss

    def _compute_att_loss(self, images, matching_labels, pred_matched_coordinates, all_pred_dots):
        prev_pred_matched_coord = self.model.coord(images)

        last_layer_res = 1
        loss = 0
        losses = []
        accs = []
        for i, (src_att_size, dest_att_size) in enumerate(zip(self.model.src_att_iters, self.model.dest_att_iters)):
            # Split up match labels according to last feature layer
            match_labels_split = tf.stack(tf.split(tf.stack(tf.split(matching_labels, last_layer_res, 1), 1), last_layer_res, 3), 2)
            # Split each tile again according to resolution of current feature layer
            match_labels_split = tf.stack(tf.split(tf.stack(tf.split(match_labels_split, src_att_size, 3), 3), src_att_size, 5), 4)
            # Collapse x,y dimensions
            match_labels_split = tf.reshape(match_labels_split, [tf.shape(match_labels_split)[0], (last_layer_res) ** 2, src_att_size ** 2, 1, tf.shape(match_labels_split)[-3], tf.shape(match_labels_split)[-2], tf.shape(match_labels_split)[-1]])

            # Split up the possible coordinates to match for each feature vector
            possible_coords_to_match = tf.stack(tf.split(tf.stack(tf.split(prev_pred_matched_coord, last_layer_res, 1), 1), last_layer_res, 3), 2)
            possible_coords_to_match = tf.stack(tf.split(tf.stack(tf.split(possible_coords_to_match, dest_att_size, 3), 3), dest_att_size, 5), 4)
            possible_coords_to_match = tf.reshape(possible_coords_to_match, [tf.shape(possible_coords_to_match)[0], (last_layer_res) ** 2, 1, dest_att_size ** 2, tf.shape(possible_coords_to_match)[-3], tf.shape(possible_coords_to_match)[-2], tf.shape(possible_coords_to_match)[-1]])
            possible_coords_to_match = tf.cast(possible_coords_to_match, tf.int64)

            # For each possible tile that can be matched in the second image, compute min and max coordinates
            x_min = tf.reduce_min(possible_coords_to_match[..., 0], axis=[-1, -2], keepdims=True)
            y_min = tf.reduce_min(possible_coords_to_match[..., 1], axis=[-1, -2], keepdims=True)
            x_max = tf.reduce_max(possible_coords_to_match[..., 0], axis=[-1, -2], keepdims=True)
            y_max = tf.reduce_max(possible_coords_to_match[..., 1], axis=[-1, -2], keepdims=True)

            # Define boolean map over pixelwise match labels to mask the ones that can be matched at all
            pixels_with_matches_mask = tf.logical_and(
                tf.logical_and(
                    x_min <= match_labels_split[..., 0],
                    match_labels_split[..., 0] <= x_max
                ),
                tf.logical_and(
                    y_min <= match_labels_split[..., 1],
                    match_labels_split[..., 1] <= y_max
                )
            )

            # Count the number of matchable pixels per feature vector combination
            num_matchable_pixels = tf.reduce_sum(tf.cast(pixels_with_matches_mask, tf.float32), axis=[-2, -1])
            num_matchable_pixels = tf.reshape(num_matchable_pixels, [tf.shape(num_matchable_pixels)[0], (last_layer_res), (last_layer_res), src_att_size, src_att_size, dest_att_size ** 2])
            num_matchable_pixels = tf.reshape(num_matchable_pixels, [tf.shape(num_matchable_pixels)[0], (last_layer_res) ** 2, src_att_size ** 2, dest_att_size ** 2])

            # Get a one hot encoding of the feature vector combination that contains the most pixel wise matches
            best_match_mask = tf.one_hot(tf.argmax(num_matchable_pixels, -1), tf.shape(num_matchable_pixels)[-1])
            # Check that the one with the most matches has more than zero matches
            any_valid_matches_mask = tf.cast(tf.logical_and(num_matchable_pixels > 0, best_match_mask > 0), tf.float32)
            # Sum up per feature vector in the first map, to get a mask across feature vectors that have any valid match at all in their destination area
            any_valid_matches_mask = tf.reduce_sum(any_valid_matches_mask, -1, keepdims=True) > 0

            # Extract the dot products that correspond to the best feature matches
            max_logit = tf.boolean_mask(all_pred_dots[i], best_match_mask > 0)
            max_logit = tf.reshape(max_logit, tf.shape(all_pred_dots[i])[:-1])

            # Compute the contrastive loss between the dot product of the best match and all other dot products
            single_loss = tf.maximum(0.0, -(max_logit[..., None] - all_pred_dots[i]) + 1.0)
            # Mask out the loss of any dot products that correspond to feature matches that have no matches at all and make
            # sure that we only consider dot products that correspond to zero matching pixels
            mask = tf.logical_and(any_valid_matches_mask, num_matchable_pixels == 0)
            single_loss = tf.boolean_mask(single_loss, mask)
            # Compute the mean
            single_loss = tf.reduce_mean(single_loss)
            losses.append(single_loss)
            # Sum up losses
            loss += single_loss * self.aux_loss_weight

            # Extract the number of matching pixels corresponding to the highest dot product per feature vector in the first image
            hit_points = tf.reduce_sum(num_matchable_pixels * tf.one_hot(tf.argmax(all_pred_dots[i], -1), tf.shape(all_pred_dots[i])[-1]), axis=-1)
            # Mask out any feature vectors that dont have any valid match at all
            hit_points_mask = tf.reduce_sum(num_matchable_pixels, axis=-1) > 0
            hit_points = tf.boolean_mask(hit_points, hit_points_mask)
            # Build binary tensor that denotes all feature vectors that were matched correctly
            hit_points = tf.cast(hit_points > 0, tf.float32)
            accs.append(hit_points)

            # Prepare for next iteration
            last_layer_res *= src_att_size
            prev_pred_matched_coord = pred_matched_coordinates[i]

        return loss, losses, accs

    def _calc_loss(self, label_cam_pose, pred_cam_pose, metric, pose_transform, reference_images, matching_labels, pred_matched_coordinates, all_pred_dots):
        # Calc pose estimation loss
        pose_loss = self._regression_pose_loss(label_cam_pose, pred_cam_pose, metric, pose_transform)
        # Compute auxiliary loss on predicted matches
        att_loss, att_losses, att_accs = self._compute_att_loss(reference_images, matching_labels, pred_matched_coordinates, all_pred_dots)

        # Sum up losses and log them
        loss = pose_loss + (att_loss if self.config.get_bool("aux_loss") else 0)
        metric.update_loss(loss, pose_loss, att_losses, att_accs)

        return loss

    @tf.function()
    def train_step(self, reference_images, reference_cam_poses, query_images, query_cam_poses, matching_labels, pose_transform):
        with tf.name_scope("main"):
            with tf.GradientTape() as tape:
                # Forward pass
                with tf.name_scope("model"):
                    pred_cam_poses, pred_matched_coordinates, all_pred_dots, _ = self.model(reference_images, query_images, True)
                # Loss calculation
                with tf.name_scope("loss"):
                    loss = self._calc_loss(query_cam_poses, pred_cam_poses, self.train_metric, pose_transform, reference_images, matching_labels, pred_matched_coordinates, all_pred_dots)

            # Compute gradients and apply optimizer step
            with tf.name_scope("optimizer"):
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            # Log gradient norm
            with tf.name_scope("metric"):
                self.train_metric.update_gradients(gradients)

    @tf.function()
    def test_step(self, reference_images, reference_cam_poses, query_images, query_cam_poses, matching_labels, pose_transform):
        # Forward pass
        pred_cam_poses, pred_matched_coordinates, all_pred_dots, _ = self.model(reference_images, query_images, False)
        # Calc loss
        self._calc_loss(query_cam_poses, pred_cam_poses, self.val_metric, pose_transform, reference_images, matching_labels, pred_matched_coordinates, all_pred_dots)

    def test_steps(self, steps=100):
        for i in range(steps):
            # Get next batch
            try:
                reference_images, reference_cam_poses, query_images, query_cam_poses, iou, room_ids, pose_transform, matching_labels = next(self.val_iter)
            except StopIteration:
                # If we reached the end of the epoch, start again
                self.val_iter = iter(self.val_dataset)
                reference_images, reference_cam_poses, query_images, query_cam_poses, iou, room_ids, pose_transform, matching_labels = next(self.val_iter)

            self.test_step(reference_images, reference_cam_poses, query_images, query_cam_poses, matching_labels, pose_transform)

    def train_steps(self, steps=10):
        for i in range(steps):
            # Get next batch
            try:
                reference_images, reference_cam_poses, query_images, query_cam_poses, iou, room_ids, pose_transform, matching_labels = next(self.train_iter)
            except StopIteration:
                # If we reached the end of the epoch, start again
                print("Starting new epoch")
                self.train_iter = iter(self.train_dataset)
                reference_images, reference_cam_poses, query_images, query_cam_poses, iou, room_ids, pose_transform, matching_labels = next(self.train_iter)

            self.train_step(reference_images, reference_cam_poses, query_images, query_cam_poses, matching_labels, pose_transform)

    def step(self, tensorboard_writer, current_iteration):
        with tensorboard_writer.as_default():
            # Give iteration number to metrics for correct tensorboard plotting
            self.train_metric.set_current_iteration(current_iteration)
            self.val_metric.set_current_iteration(current_iteration)

            # Run 100 train batches
            self.train_steps(100)
            # Run 10 test batches
            self.test_steps(10)

            # Plot to tensorboard
            self.train_metric.plot()
            self.val_metric.plot()

            # Reset metrics
            self.train_metric.reset()
            self.val_metric.reset()

