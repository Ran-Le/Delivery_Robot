import numpy as np

EPSILON_OMEGA = 1e-3


def compute_dynamics(x, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                        x: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to x.
        Gu: np.array[3,2] - Jacobian of g with respect ot u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    if abs(u[1]) < EPSILON_OMEGA:
        u[1] = EPSILON_OMEGA
    g = np.zeros(3)
    g[0] = x[0] + u[0] / u[1] * (np.sin(x[2] + u[1] * dt) - np.sin(x[2]))
    g[1] = x[1] + u[0] / u[1] * (np.cos(x[2]) - np.cos(x[2] + u[1] * dt))
    g[2] = x[2] + u[1] * dt
    Gx = np.zeros((3, 3))
    Gx[0, 0] = 1
    Gx[0, 2] = u[0] / u[1] * (np.cos(x[2] + u[1] * dt) - np.cos(x[2]))
    Gx[1, 1] = 1
    Gx[1, 2] = u[0] / u[1] * (-np.sin(x[2]) + np.sin(x[2] + u[1] * dt))
    Gx[2, 2] = 1
    Gu = np.zeros((3, 2))
    Gu[0, 0] = 1 / u[1] * (np.sin(x[2] + u[1] * dt) - np.sin(x[2]))
    Gu[0, 1] = -u[0] / u[1] / u[1] * \
        (np.sin(x[2] + u[1] * dt) - np.sin(x[2])) + \
        u[0] / u[1] * np.cos(x[2] + u[1] * dt) * dt
    Gu[1, 0] = 1 / u[1] * (np.cos(x[2]) - np.cos(x[2] + u[1] * dt))
    Gu[1, 1] = -u[0] / u[1] / u[1] * \
        (np.cos(x[2]) - np.cos(x[2] + u[1] * dt)) + \
        u[0] / u[1] * np.sin(x[2] + u[1] * dt) * dt
    Gu[2, 1] = dt
    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu


def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    h = np.zeros(2)
    h[0] = line[0] - x[2] - tf_base_to_camera[2]
    h[1] = line[1] - x[0] * np.cos(line[0]) - x[1] * np.sin(line[0]) - tf_base_to_camera[0] * \
        np.cos(line[0] - x[2]) - tf_base_to_camera[1] * np.sin(line[0] - x[2])
    Hx = np.zeros((2, 3))
    Hx[0, 2] = -1
    Hx[1, 0] = -np.cos(line[0])
    Hx[1, 1] = -np.sin(line[0])
    Hx[1, 2] = - tf_base_to_camera[0] * \
        np.sin(line[0] - x[2]) + tf_base_to_camera[1] * np.cos(line[0] - x[2])
    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1, :] *= -1
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
