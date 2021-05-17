import tensorflow as tf
from math import pi

class Metric:

    def __init__(self, data, prefix):
        self.data = data
        # Prefix for tensorboard: Either train/ or val/
        self.prefix = prefix

        # Init loss metrics
        self.loss = tf.keras.metrics.Mean()
        self.pose_loss = tf.keras.metrics.Mean()

        # Init pose deviation metrics
        self.pose_deviation = {}
        for key in ["x", "y", "z", "pitch", "roll", "yaw"]:
            self.pose_deviation[key] = tf.keras.metrics.Mean()

        # Init gradient norm metric and iterations counter
        self.gradient_norm = tf.keras.metrics.Mean()
        self.current_iteration = tf.Variable(0, dtype=tf.int64)

        # Init aux loss + accuracy metrics (for both correlation layer)
        self.att_losses = []
        self.att_accs = []
        for i in range(2):
            self.att_losses.append(tf.keras.metrics.Mean())
            self.att_accs.append(tf.keras.metrics.Mean())

    def set_current_iteration(self, current_iteration):
        self.current_iteration.assign(current_iteration)

    def update_loss(self, loss, pose_loss, att_losses, att_accs):
        self.loss(loss)
        self.pose_loss(pose_loss)
        for i in range(len(att_losses)):
            self.att_losses[i](att_losses[i])
            self.att_accs[i](att_accs[i])

    def update_gradients(self, gradients):
        for gradient in gradients:
            if gradient is not None:
                self.gradient_norm(tf.norm(gradient))

    def update_cam_pose_metric(self, pred_cam_poses, query_cam_poses, is_first_batch, pose_transform):
        # Bring back pred and query cam pose into world coordinate frame
        pred_real_cam_pose = self.data.revert_cam_pose_normalization(pred_cam_poses, pose_transform)
        query_real_cam_poses = self.data.revert_cam_pose_normalization(query_cam_poses, pose_transform)

        # Compute deviation
        dev = tf.unstack(tf.abs(pred_real_cam_pose - query_real_cam_poses), axis=1)

        # For angles use the smaller diff
        for i in range(3, 6):
            dev[i] = tf.math.mod(dev[i], 2 * pi)
            dev[i] = tf.minimum(2 * pi - dev[i], dev[i])

        # Add them to the metrics
        for i, key in enumerate(["x", "y", "z", "pitch", "roll", "yaw"]):
            self.pose_deviation[key](dev[i])

    def reset(self):
        # Reset all metrics
        self.gradient_norm.reset_states()
        self.loss.reset_states()
        self.pose_loss.reset_states()

        for pose_dev in self.pose_deviation.values():
            pose_dev.reset_states()

        for i in range(len(self.att_losses)):
            self.att_losses[i].reset_states()
            self.att_accs[i].reset_states()

    def plot(self):
        # Plot total + pose loss
        tf.summary.scalar(self.prefix + 'loss', self.loss.result(), step=self.current_iteration)
        tf.summary.scalar(self.prefix + 'pose/loss', self.pose_loss.result(), step=self.current_iteration)

        # Plot pose deviations
        for key, pose_dev in self.pose_deviation.items():
            tf.summary.scalar(self.prefix + 'pose_dev/' + key, pose_dev.result(), step=self.current_iteration)

        # Plot gradient norm
        tf.summary.scalar(self.prefix + 'gradients', self.gradient_norm.result(), step=self.current_iteration)

        # Plot loss / accuracy of matchings
        for i in range(len(self.att_losses)):
            tf.summary.scalar(self.prefix + 'att/loss_' + str(i), self.att_losses[i].result(), step=self.current_iteration)
            tf.summary.scalar(self.prefix + 'att/acc_' + str(i), self.att_accs[i].result(), step=self.current_iteration)
