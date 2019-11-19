import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    path = np.asarray(path)
    t_smoothed = np.zeros(len(path))
    t_smoothed[0] = 0
    for i in range(1, len(path)):
        t_smoothed[i] = np.sqrt(
            (path[i, 0] - path[i - 1, 0])**2 + (path[i, 1] - path[i - 1, 1])**2) / V_des + t_smoothed[i - 1]
    splx = scipy.interpolate.splrep(t_smoothed, path[:, 0], s=alpha)
    sply = scipy.interpolate.splrep(t_smoothed, path[:, 1], s=alpha)
    x = scipy.interpolate.splev(t_smoothed, splx, der=0)
    xd = scipy.interpolate.splev(t_smoothed, splx, der=1)
    xdd = scipy.interpolate.splev(t_smoothed, splx, der=2)
    y = scipy.interpolate.splev(t_smoothed, sply, der=0)
    yd = scipy.interpolate.splev(t_smoothed, sply, der=1)
    ydd = scipy.interpolate.splev(t_smoothed, sply, der=2)
    th = np.arctan2(yd, xd)
    traj_smoothed = np.zeros((len(path), 7))
    for i in range(len(path)):
        traj_smoothed[i] = [x[i], y[i], th[i], xd[i], yd[i], xdd[i], ydd[i]]
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed
