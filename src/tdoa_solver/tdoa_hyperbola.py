import numpy as np

def tdoa_hyperbola(sensor1:np.ndarray, sensor2:np.ndarray, tdoa: float, x_range:tuple[int,int], y_range:tuple[int,int], n_points=400):
    """
    Compute a TDOA hyperbola. A TDOA hyperbola is such that r1 - r2 = tdoa,
    where r1 and r2 are the distances from a point on the hyperbola to sensor1 and sensor2,
    for all points on the hyperbola.

    Args:
        sensor1, sensor2: (2,) arrays with [x, y] coordinates
        tdoa: float, time difference of arrival (distance difference)
        x_range, y_range: tuples like (xmin, xmax), (ymin, ymax)
        n_points: resolution of grid

    Returns:
        X, Y, F: 2D NumPy arrays for contour plotting
                 F = sqrt((X-x1)^2+(Y-y1)^2) - sqrt((X-x2)^2+(Y-y2)^2) - tdoa
    """
    assert sensor1.shape == (2,) # Check that sensor1 is a 2D point
    assert sensor2.shape == (2,) # Check that sensor2 is a 2D point
    assert isinstance(tdoa, (int, float)), "tdoa must be a scalar"
    assert len(x_range) == 2 and len(y_range) == 2, "x_range and y_range must be tuples of length 2"

    x = np.linspace(*x_range, n_points)
    y = np.linspace(*y_range, n_points)
    X, Y = np.meshgrid(x, y)
    r1 = np.sqrt((X - sensor1[0])**2 + (Y - sensor1[1])**2)
    r2 = np.sqrt((X - sensor2[0])**2 + (Y - sensor2[1])**2)

    F = r2 - r1 - tdoa
    return x, y, F