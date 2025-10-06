
import numpy as np

def run_ils(z: np.ndarray, a: np.ndarray, x0: np.ndarray|None = None) -> np.ndarray:
    """
    Python/Numpy translation of C++ runILS(Eigen::VectorXd z, Eigen::Matrix2Xd a, Eigen::Vector2d x0)

    Args:
        z: 1D array of size (nSensors-1,)  -- TDOA measurements
        a: 2xN array                      -- sensor positions, each column is [x, y]
        x0: 1D array of size (2,)         -- initial guess of source location

    Returns:
        solution: 1D array (2,)           -- estimated source position
    """

    # Ensure input dimensions are correct
    assert z.ndim == 1, "z must be a 1D array"
    assert a.ndim == 2 and a.shape[0] == 2, "a must be a 2xN array"

    if x0 is None:
        x0 = np.zeros(2, dtype=np.float64)
    assert x0.ndim == 1 and x0.shape[0] == 2, "x0 must be a 1D array of size 2"

    # Number of sensors
    nSensors = z.shape[0] + 1

    # Current solution estimate
    solution = x0.copy()

    # Measurement noise covariance
    sigma = 1.0
    R = np.eye(nSensors - 1) * sigma**2
    Ri = np.linalg.inv(R)

    # Allocate arrays
    ranges = np.zeros(nSensors)
    h = np.zeros(nSensors - 1)
    H = np.zeros((nSensors - 1, 2))
    u = np.zeros((nSensors, 2))

    stop_eps = 0.0025

    # Iterative Least Squares loop
    for iteration in range(50):

        # Compute range from current estimate to each sensor
        for i in range(nSensors):
            sensor_pos = a[:, i]
            ranges[i] = np.linalg.norm(solution - sensor_pos)

        # Compute h vector (range differences)
        for i in range(1, nSensors):
            h[i - 1] = ranges[i] - ranges[0]

        # Compute unit vectors from each sensor to solution
        for i in range(nSensors):
            sensor_pos = a[:, i]
            u[i, :] = (solution - sensor_pos) / ranges[i]

        # Linearized model matrix
        for i in range(1, nSensors):
            H[i - 1, :] = u[i, :] - u[0, :]

        # P = H^T * Ri * H
        P = H.T @ Ri @ H

        # Check invertibility
        if abs(np.linalg.det(P)) <= 1e-2:
            print("Warning: P matrix near singular. Breaking out.")
            break

        Pinv = np.linalg.inv(P)

        # Compute delta and update solution
        # Note '@' is matrix multiplication in numpy
        delta_solution = Pinv @ H.T @ Ri @ (z - h)
        solution += delta_solution

        # Check stopping condition
        if np.linalg.norm(delta_solution) < stop_eps:
            print(f"Stopping criterion met on iteration={iteration}")
            break

    return solution