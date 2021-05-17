from math import pi
import numpy as np
import numpy.matlib as npm

class RMatrix:

    @staticmethod
    #http://www.songho.ca/opengl/gl_anglestoaxes.html
    def from_euler(cam_pose):
        s0, s1, s2 = np.sin(cam_pose[0]), np.sin(cam_pose[1]), np.sin(cam_pose[2])
        c0, c1, c2 = np.cos(cam_pose[0]), np.cos(cam_pose[1]), np.cos(cam_pose[2])
        tmat = np.array([
            [c2 * c1, -s2 * c0 + c2 * s1 * s0, s2 * s0 + c2 * s1 * c0],
            [s2 * c1, c2 * c0 + s2 * s1 * s0, -c2 * s0 + s2 * s1 * c0],
            [-s1, c1 * s0, c1 * c0]
        ])
        return tmat

    @staticmethod
    #https://www.geometrictools.com/Documentation/EulerAngles.pdf
    def to_euler(tmat):
        cam_pose = np.array([
            np.mod(np.arctan2(tmat[2, 1], tmat[2, 2]), 2 * pi),
            np.mod(np.arcsin(-tmat[2, 0]), 2 * pi),
            np.mod(np.arctan2(tmat[1, 0], tmat[0, 0]), 2 * pi)
        ])

        return cam_pose

    @staticmethod
    def inverse(tmat):
        return tmat.T

    @staticmethod
    def apply(tmat1, tmat2):
        return np.matmul(tmat1, tmat2)

    @staticmethod
    def rotation_diff(tmat1, tmat2):
        R_ab = np.matmul(tmat1.T, tmat2)
        angle = np.arccos(np.clip((np.trace(R_ab) - 1) / 2, -1, 1))
        return angle

    @staticmethod
    def average_rotations(tmat_list):
        for i in range(len(tmat_list)):
            tmat_list[i] = RMatrix.to_quaternion(tmat_list[i])

        return RMatrix.from_quaternion(RMatrix.average_quaternions(np.stack(tmat_list, 0)))

    @staticmethod
    # https://github.com/christophhagen/averaging-quaternions/blob/master/averageQuaternions.py
    def average_quaternions(Q):
        # to (w, x, y, z)
        Q = np.roll(Q, 1, axis=-1)

        # Number of quaternions to average
        M = Q.shape[0]
        A = npm.zeros(shape=(4, 4))

        for i in range(0, M):
            q = Q[i, :]
            # multiply q with its transposed version q' and add A
            A = np.outer(q, q) + A

        # scale
        A = (1.0 / M) * A
        # compute eigenvalues and -vectors
        eigenValues, eigenVectors = np.linalg.eig(A)
        # Sort by largest eigenvalue
        eigenVectors = eigenVectors[:, eigenValues.argsort()[::-1]]
        # return the real part of the largest eigenvector (has only real part)
        res = np.real(eigenVectors[:, 0].A1)

        res = np.roll(res, -1, axis=-1)

        return res

    @staticmethod
    # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    def from_quaternion(cam_pose):
        tmat = np.array([
            [1 - 2 * np.square(cam_pose[1]) - 2 * np.square(cam_pose[2]), 2 * cam_pose[0] * cam_pose[1] - 2 * cam_pose[2] * cam_pose[3], 2 * cam_pose[0] * cam_pose[2] + 2 * cam_pose[1] * cam_pose[3]],
            [2 * cam_pose[0] * cam_pose[1] + 2 * cam_pose[2] * cam_pose[3], 1 - 2 * np.square(cam_pose[0]) - 2 * np.square(cam_pose[2]), 2 * cam_pose[1] * cam_pose[2] - 2 * cam_pose[0] * cam_pose[3]],
            [2 * cam_pose[0] * cam_pose[2] - 2 * cam_pose[1] * cam_pose[3], 2 * cam_pose[1] * cam_pose[2] + 2 * cam_pose[0] * cam_pose[3], 1 - 2 * np.square(cam_pose[0]) - 2 * np.square(cam_pose[1])],
        ])

        return tmat

    @staticmethod
    # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    #(x,y,z,w)
    def to_quaternion(tmat):

        t1 = np.where(tmat[0, 0] > tmat[1, 1], 1 + tmat[0, 0] - tmat[1, 1] - tmat[2, 2], 1 - tmat[0, 0] + tmat[1, 1] - tmat[2, 2])
        q1 = np.where(tmat[0, 0] > tmat[1, 1], [t1, tmat[1, 0] + tmat[0, 1], tmat[0, 2] + tmat[2, 0], tmat[2, 1] - tmat[1, 2]], [tmat[1, 0] + tmat[0, 1], t1, tmat[2, 1] + tmat[1, 2], tmat[0, 2] - tmat[2, 0]])

        t2 = np.where(tmat[0, 0] < -tmat[1, 1], 1 - tmat[0, 0] - tmat[1, 1] + tmat[2, 2], 1 + tmat[0, 0] + tmat[1, 1] + tmat[2, 2])
        q2 = np.where(tmat[0, 0] < -tmat[1, 1], [tmat[0, 2] + tmat[2, 0], tmat[2, 1] + tmat[1, 2], t2, tmat[1, 0] - tmat[0, 1]], [tmat[2, 1] - tmat[1, 2], tmat[0, 2] - tmat[2, 0], tmat[1, 0] - tmat[0, 1], t2])

        t = np.where(tmat[2, 2] < 0, t1, t2)
        q = np.where(tmat[2, 2] < 0, q1, q2)

        q *= 0.5 / np.sqrt(t)

        if q[-1] < -1:
            q = -1 * q

        return q
