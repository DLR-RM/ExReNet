# Source: https://github.com/Eliasvan/Multiple-Quadrotor-SLAM

import cv2
import numpy as np

class Triangulation:

    @staticmethod
    def directionToRot(dir, up=[0,1,0]):
        rotmat = np.zeros((3, 3))

        dir /= np.linalg.norm(dir)

        x = np.cross(up, dir)
        x /= np.linalg.norm(x)

        y = np.cross(dir, x)
        y /= np.linalg.norm(y)

        rotmat[0, 0] = x[0]
        rotmat[1, 0] = y[0]
        rotmat[2, 0] = dir[0]

        rotmat[0, 1] = x[1]
        rotmat[1, 1] = y[1]
        rotmat[2, 1] = dir[1]

        rotmat[0, 2] = x[2]
        rotmat[1, 2] = y[2]
        rotmat[2, 2] = dir[2]

        return rotmat

    @staticmethod
    def triangulate(p1, d1, p2, d2):
        K = np.array([
            [1, 0., 0],
            [0., 1, 0],
            [0., 0., 1.]
        ])  # Kamera matrix:

        # define pose 0
        T0 = np.array(p1)  # Translation vector
        RT0 = np.zeros((3, 4))  # combined Rotation/Translation matrix
        RT0[:3, :3] = Triangulation.directionToRot(d1)
        RT0[:3, 3] = np.matmul(-RT0[:3, :3], T0)
        P0 = np.matmul(K, RT0)  # Projection matrix

        # define pose 1
        T1 = np.array(p2)
        RT1 = np.zeros((3, 4))
        RT1[:3, :3] = Triangulation.directionToRot(d2)
        RT1[:3, 3] = np.matmul(-RT1[:3, :3], T1)
        P1 = np.matmul(K, RT1)

        res = iterative_LS_triangulation(np.array([[0.0, 0.0]]), P0, np.array([[0.0, 0.0]]), P1)[0]

        return res[0]


output_dtype = np.float32


# Initialize consts to be used in iterative_LS_triangulation()
iterative_LS_triangulation_C = -np.eye(2, 3)


def iterative_LS_triangulation(u1, P1, u2, P2, tolerance=3.e-5):
    """
    Iterative (Linear) Least Squares based triangulation.
    From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997.
    Relative speed: 0.025

    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    "tolerance" is the depth convergence tolerance.

    Additionally returns a status-vector to indicate outliers:
        1: inlier, and in front of both cameras
        0: outlier, but in front of both cameras
        -1: only in front of second camera
        -2: only in front of first camera
        -3: not in front of any camera
    Outliers are selected based on non-convergence of depth, and on negativity of depths (=> behind camera(s)).

    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    """
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))

    # Create array of triangulated points
    x = np.empty((4, len(u1)));
    x[3, :].fill(1)  # create empty array of homogenous 3D coordinates
    x_status = np.empty(len(u1), dtype=int)

    # Initialize C matrices
    C1 = np.array(iterative_LS_triangulation_C)
    C2 = np.array(iterative_LS_triangulation_C)

    for xi in range(len(u1)):
        # Build C matrices, to construct A and b in a concise way
        C1[:, 2] = u1[xi, :]
        C2[:, 2] = u2[xi, :]

        # Build A matrix
        A[0:2, :] = C1.dot(P1[0:3, 0:3])  # C1 * R1
        A[2:4, :] = C2.dot(P2[0:3, 0:3])  # C2 * R2

        # Build b vector
        b[0:2, :] = C1.dot(P1[0:3, 3:4])  # C1 * t1
        b[2:4, :] = C2.dot(P2[0:3, 3:4])  # C2 * t2
        b *= -1

        # Init depths
        d1 = d2 = 1.

        for i in range(10):  # Hartley suggests 10 iterations at most
            # Solve for x vector
            # x_old = np.array(x[0:3, xi])    # TODO: remove
            cv2.solve(A, b, x[0:3, xi:xi + 1], cv2.DECOMP_SVD)

            # Calculate new depths
            d1_new = P1[2, :].dot(x[:, xi])
            d2_new = P2[2, :].dot(x[:, xi])

            # Convergence criterium
            # print i, d1_new - d1, d2_new - d2, (d1_new > 0 and d2_new > 0)    # TODO: remove
            # print i, (d1_new - d1) / d1, (d2_new - d2) / d2, (d1_new > 0 and d2_new > 0)    # TODO: remove
            # print i, np.sqrt(np.sum((x[0:3, xi] - x_old)**2)), (d1_new > 0 and d2_new > 0)    # TODO: remove
            ##print i, u1[xi, :] - P1[0:2, :].dot(x[:, xi]) / d1_new, u2[xi, :] - P2[0:2, :].dot(x[:, xi]) / d2_new    # TODO: remove
            # print bool(i) and ((d1_new - d1) / (d1 - d_old), (d2_new - d2) / (d2 - d1_old), (d1_new > 0 and d2_new > 0))    # TODO: remove
            ##if abs(d1_new - d1) <= tolerance and abs(d2_new - d2) <= tolerance: print "Orig cond met"    # TODO: remove
            if abs(d1_new - d1) <= tolerance and \
                    abs(d2_new - d2) <= tolerance:
                # if i and np.sum((x[0:3, xi] - x_old)**2) <= 0.0001**2:
                # if abs((d1_new - d1) / d1) <= 3.e-6 and \
                # abs((d2_new - d2) / d2) <= 3.e-6: #and \
                # abs(d1_new - d1) <= tolerance and \
                # abs(d2_new - d2) <= tolerance:
                # if i and 1 - abs((d1_new - d1) / (d1 - d_old)) <= 1.e-2 and \    # TODO: remove
                # 1 - abs((d2_new - d2) / (d2 - d1_old)) <= 1.e-2 and \    # TODO: remove
                # abs(d1_new - d1) <= tolerance and \    # TODO: remove
                # abs(d2_new - d2) <= tolerance:    # TODO: remove
                break

            # Re-weight A matrix and b vector with the new depths
            A[0:2, :] *= 1 / d1_new
            A[2:4, :] *= 1 / d2_new
            b[0:2, :] *= 1 / d1_new
            b[2:4, :] *= 1 / d2_new

            # Update depths
            # d_old = d1    # TODO: remove
            # d1_old = d2    # TODO: remove
            d1 = d1_new
            d2 = d2_new

        # Set status
        x_status[xi] = (i < 10 and  # points should have converged by now
                        (d1_new > 0 and d2_new > 0))  # points should be in front of both cameras
        if d1_new <= 0: x_status[xi] -= 1
        if d2_new <= 0: x_status[xi] -= 2

    return x[0:3, :].T.astype(output_dtype), x_status
