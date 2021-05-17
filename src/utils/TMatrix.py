import tensorflow as tf
from math import pi
import numpy as np
from math import atan2

class TMatrix:

    @staticmethod
    #http://www.songho.ca/opengl/gl_anglestoaxes.html
    def from_euler(cam_pose, num_batch_dim=1):
        cam_pose, batch_dim = TMatrix.merge_batch_dim(cam_pose, num_batch_dim)

        s0, s1, s2 = tf.sin(cam_pose[:, 3]), tf.sin(cam_pose[:, 4]), tf.sin(cam_pose[:, 5])
        c0, c1, c2 = tf.cos(cam_pose[:, 3]), tf.cos(cam_pose[:, 4]), tf.cos(cam_pose[:, 5])
        zero_col = tf.zeros_like(s0)
        tmat = tf.convert_to_tensor([
            [c2 * c1, -s2 * c0 + c2 * s1 * s0, s2 * s0 + c2 * s1 * c0, cam_pose[:, 0]],
            [s2 * c1, c2 * c0 + s2 * s1 * s0, -c2 * s0 + s2 * s1 * c0, cam_pose[:, 1]],
            [-s1, c1 * s0, c1 * c0, cam_pose[:, 2]],
            [zero_col, zero_col, zero_col, zero_col + 1]
        ])
        tmat = tf.transpose(tmat, [2, 0, 1])
        tmat = TMatrix.unmerge_batch_dim(tmat, batch_dim)

        return tmat


    @staticmethod
    #https://www.geometrictools.com/Documentation/EulerAngles.pdf
    def to_euler(tmat, num_batch_dim=1):
        tmat, batch_dim = TMatrix.merge_batch_dim(tmat, num_batch_dim)

        cam_pose = []
        cam_pose.append(tmat[:, 0, 3])
        cam_pose.append(tmat[:, 1, 3])
        cam_pose.append(tmat[:, 2, 3])

        cam_pose.append(tf.math.mod(tf.atan2(tmat[:, 2, 1], tmat[:, 2, 2]), 2 * pi))
        cam_pose.append(tf.math.mod(tf.asin(tf.clip_by_value(-tmat[:, 2, 0], -1, 1)), 2 * pi))
        cam_pose.append(tf.math.mod(tf.atan2(tmat[:, 1, 0], tmat[:, 0, 0]), 2 * pi))

        cam_pose = tf.stack(cam_pose, axis=-1)
        cam_pose = TMatrix.unmerge_batch_dim(cam_pose, batch_dim)
        return cam_pose

    @staticmethod
    # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    def from_quaternion(cam_pose, num_batch_dim=1):
        cam_pose, batch_dim = TMatrix.merge_batch_dim(cam_pose, num_batch_dim)


        zero_col = tf.zeros_like(cam_pose[:, 2])
        tmat = tf.convert_to_tensor([
            [1 - 2 * tf.square(cam_pose[:, 4]) - 2 * tf.square(cam_pose[:, 5]), 2 * cam_pose[:, 3] * cam_pose[:, 4] - 2 * cam_pose[:, 5] * cam_pose[:, 6], 2 * cam_pose[:, 3] * cam_pose[:, 5] + 2 * cam_pose[:, 4] * cam_pose[:, 6], cam_pose[:, 0]],
            [2 * cam_pose[:, 3] * cam_pose[:, 4] + 2 * cam_pose[:, 5] * cam_pose[:, 6], 1 - 2 * tf.square(cam_pose[:, 3]) - 2 * tf.square(cam_pose[:, 5]), 2 * cam_pose[:, 4] * cam_pose[:, 5] - 2 * cam_pose[:, 3] * cam_pose[:, 6], cam_pose[:, 1]],
            [2 * cam_pose[:, 3] * cam_pose[:, 5] - 2 * cam_pose[:, 4] * cam_pose[:, 6], 2 * cam_pose[:, 4] * cam_pose[:, 5] + 2 * cam_pose[:, 3] * cam_pose[:, 6], 1 - 2 * tf.square(cam_pose[:, 3]) - 2 * tf.square(cam_pose[:, 4]), cam_pose[:, 2]],
            [zero_col, zero_col, zero_col, zero_col + 1]
        ])
        tmat = tf.transpose(tmat, [2, 0, 1])
        fov = cam_pose[0, 7:9]

        tmat = TMatrix.unmerge_batch_dim(tmat, batch_dim)

        return tmat

    @staticmethod
    def from_rotmat(cam_pose, num_batch_dim=1):
        cam_pose, batch_dim = TMatrix.merge_batch_dim(cam_pose, num_batch_dim)

        zero_col = tf.zeros_like(cam_pose[:, 0])
        tmat = tf.convert_to_tensor([
            [zero_col, zero_col, zero_col, cam_pose[:, 0]],
            [zero_col, zero_col, zero_col, cam_pose[:, 1]],
            [zero_col, zero_col, zero_col, cam_pose[:, 2]],
            [zero_col, zero_col, zero_col, zero_col + 1]
        ])
        tmat = tf.transpose(tmat, [2, 0, 1])
        tmat[:, :3, :3] = tf.reshape(cam_pose[:, 3:], [-1, 3, 3])

        tmat = TMatrix.unmerge_batch_dim(tmat, batch_dim)

        return tmat

    @staticmethod
    # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    def to_quaternion(tmat, num_batch_dim=1):
        tmat, batch_dim = TMatrix.merge_batch_dim(tmat, num_batch_dim)

        cam_pose = []
        cam_pose.append(tmat[:, 0, 3])
        cam_pose.append(tmat[:, 1, 3])
        cam_pose.append(tmat[:, 2, 3])
        cam_pose = tf.stack(cam_pose, axis=-1)

        t1 = tf.where(tmat[:, 0, 0] > tmat[:, 1, 1], 1 + tmat[:, 0, 0] - tmat[:, 1, 1] - tmat[:, 2, 2], 1 - tmat[:, 0, 0] + tmat[:, 1, 1] - tmat[:, 2, 2])
        q1 = tf.where(tmat[:, 0, 0, None] > tmat[:, 1, 1, None], tf.stack([t1, tmat[:, 1, 0] + tmat[:, 0, 1], tmat[:, 0, 2] + tmat[:, 2, 0], tmat[:, 2, 1] - tmat[:, 1, 2]], axis=-1), tf.stack([tmat[:, 1, 0] + tmat[:, 0, 1], t1, tmat[:, 2, 1] + tmat[:, 1, 2], tmat[:, 0, 2] - tmat[:, 2, 0]], axis=-1))

        t2 = tf.where(tmat[:, 0, 0] < -tmat[:, 1, 1], 1 - tmat[:, 0, 0] - tmat[:, 1, 1] + tmat[:, 2, 2], 1 + tmat[:, 0, 0] + tmat[:, 1, 1] + tmat[:, 2, 2])
        q2 = tf.where(tmat[:, 0, 0, None] < -tmat[:, 1, 1, None], tf.stack([tmat[:, 0, 2] + tmat[:, 2, 0], tmat[:, 2, 1] + tmat[:, 1, 2], t2, tmat[:, 1, 0] - tmat[:, 0, 1]], axis=-1), tf.stack([tmat[:, 2, 1] - tmat[:, 1, 2], tmat[:, 0, 2] - tmat[:, 2, 0], tmat[:, 1, 0] - tmat[:, 0, 1], t2], axis=-1))

        t = tf.where(tmat[:, 2, 2] < 0, t1, t2)
        q = tf.where(tmat[:, 2, 2, None] < 0, q1, q2)

        q *= 0.5 / tf.sqrt(t[:, None])

        cam_pose = tf.concat((cam_pose, q), axis=-1)
        cam_pose = TMatrix.unmerge_batch_dim(cam_pose, batch_dim)
        return cam_pose

    @staticmethod
    # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    def to_rotmat(tmat, num_batch_dim=1):
        tmat, batch_dim = TMatrix.merge_batch_dim(tmat, num_batch_dim)

        cam_pose = []
        cam_pose.append(tmat[:, 0, 3])
        cam_pose.append(tmat[:, 1, 3])
        cam_pose.append(tmat[:, 2, 3])
        cam_pose = tf.stack(cam_pose, axis=-1)

        cam_pose = tf.concat((cam_pose, tf.reshape(tmat[:, :3, :3], [-1, 9])), axis=-1)
        cam_pose = TMatrix.unmerge_batch_dim(cam_pose, batch_dim)
        return cam_pose

    @staticmethod
    def inverse(tmat, num_batch_dim=1):
        tmat, batch_dim = TMatrix.merge_batch_dim(tmat, num_batch_dim)

        original_tmat = tmat
        tmat = tf.transpose(tmat, [0, 2, 1])

        unstacked_tmat = tf.unstack(tmat, axis=2)

        inv_t = tf.matmul(-tmat[:, :3, :3], original_tmat[:, :3, 3:4])[:, :, 0]
        unstacked_tmat[-1] = tf.concat((inv_t, tf.ones((tf.shape(tmat)[0], 1), tf.float32)), 1)

        tmat = tf.stack(unstacked_tmat, 2)

        unstacked_tmat = tf.unstack(tmat, axis=1)
        unstacked_tmat[-1] = original_tmat[:, -1, :]
        tmat = tf.stack(unstacked_tmat, 1)

        tmat = TMatrix.unmerge_batch_dim(tmat, batch_dim)
        return tmat

    @staticmethod
    def apply(tmat1, tmat2):
        return tf.matmul(tmat1, tmat2)

    @staticmethod
    def merge_batch_dim(tensor, num_batch_dim):
        shape = [-1]
        batch_dim = []
        for i in range(len(tensor.shape)):
            dim = tensor.shape[i] if tensor.shape[i] is not None else tf.shape(tensor)[i]
            if i < num_batch_dim:
                batch_dim.append(dim)
            else:
                shape.append(dim)

        return tf.reshape(tensor, shape), batch_dim

    @staticmethod
    def unmerge_batch_dim(tensor, batch_dim):
        shape = batch_dim

        for i in range(1, len(tensor.shape)):
            shape.append(tensor.shape[i] if tensor.shape[i] is not None else tf.shape(tensor)[i])

        return tf.reshape(tensor, shape)

    @staticmethod
    def rotation_diff(tmat1, tmat2):
        R_ab = tf.matmul(tmat1[:3, :3], tmat2[:3, :3], transpose_a=True)
        angle = tf.acos((tf.linalg.trace(R_ab) - 1) / 2)
        return angle

    @staticmethod
    def quaternion_multiply(quaternion1, quaternion0):
        x0, y0, z0, w0 = tf.unstack(quaternion0)
        x1, y1, z1, w1 = tf.unstack(quaternion1)
        return tf.convert_to_tensor([
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0], dtype=tf.float32)