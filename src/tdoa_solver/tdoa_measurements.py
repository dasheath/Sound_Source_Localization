import numpy as np

def tdoa_measurements(source_position: np.ndarray, sensors: np.ndarray) -> np.ndarray:
    """
    Python/Numpy translation of C++ runILS(Eigen::VectorXd z, Eigen::Matrix2Xd a, Eigen::Vector2d x0)

    Args:
        source_position: 1D array (2,)    -- source position [x, y]. One column, two rows.
        sensors: 2xN array                -- sensor positions, each column is [x, y]. First row is x values, second row is y values, and column 1 is the reference sensor.

    Returns:
        delta-distance measurements to be used for the ILS: 1D array (N-1,)
    """
    
    # Input validation
    assert sensors.ndim == 2 and sensors.shape[0] == 2, f"sensors must be a 2xN array, got {sensors.shape=}"
    assert sensors.shape[1] >= 2, f"At least two sensors are required, got {sensors.shape=}"

    if source_position.ndim == 2 and source_position.shape == (2,1):
        source_position = source_position.flatten()
    assert source_position.ndim == 1 and source_position.shape == (2,), f"source_position must be a 1D array of size 2, got {source_position.shape=}"

    # Distance from signal to reach reference sensor
    d0 = np.linalg.norm(source_position - sensors[:,0])
    assert d0.shape == (), f"{d0.shape=} should be a scalar"

    # Distance from signal to reach each sensor 1..n
    d_arr = np.linalg.norm(source_position[:,None] - sensors[:, 1:], axis=0)
    assert d_arr.shape == (sensors.shape[1]-1,), f"{d_arr.shape=} should be (N-1,) where N is number of sensors"

    # TDOA values in meters (v*tau)
    #   In a real test, these would come from measured
    #   time differences multiplied by the speed of sound
    delta_distances = (d_arr - d0)
    return delta_distances