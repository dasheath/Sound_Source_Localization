import numpy as np

def tdoa_measurements(source_position: np.ndarray, sensors: np.ndarray) -> np.ndarray:
    """
    Python/Numpy translation of C++ runILS(Eigen::VectorXd z, Eigen::Matrix2Xd a, Eigen::Vector2d x0)

    Args:
        source_position: 1D array (2,)    -- source position [x, y]
        sensors: 2xN array                -- sensor positions, each column is [x, y]

    Returns:
        delta-distance measurements to be used for the ILS: 1D array (N-1,)
    """
    
    # Distance from signal to reach reference sensor
    d0 = np.linalg.norm(source_position - sensors[:,0])

    # Distance from signal to reach each sensor 1..n
    d_arr = np.linalg.norm(source_position[:,None] - sensors[:, 1:], axis=0)

    # TDOA values in meters (v*tau)
    #   In a real test, these would come from measured 
    #   time differences multiplied by the speed of sound
    delta_distances = (d_arr - d0)
    return delta_distances