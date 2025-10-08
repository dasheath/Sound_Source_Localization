import numpy as np
import pandas as pd
import streamlit as st

from src.tdoa_solver.run_ils import run_ils
from src.tdoa_solver.tdoa_hyperbola import tdoa_hyperbola
from src.tdoa_solver.tdoa_measurements import tdoa_measurements 

import plotly.graph_objects as go
import plotly.express as px


def plot_tdoa_curves(sensors: np.ndarray, tdoa_vals:np.ndarray, soln: np.ndarray, fig: go.Figure | None = None) -> None:
    """
    Plots TDOA hyperbolas for given sensor pairs and TDOA values.
    Args:
        sensors (np.ndarray): Array of sensor positions in meters, shape (2, N). Each column is a sensor position.
        tdoa_vals (np.ndarray): Array of TDOA values corresponding to sensor pairs in meters (v*tau), shape (N-1,) -> row vector.
        soln (np.ndarray): Estimated source position, shape (2,1) -> row vector.
    """

    assert sensors.shape[0] == 2, "Sensor positions must be 2D."
    assert sensors.shape[1] >= 2, "At least two sensors are required."
    assert tdoa_vals.shape[0] == sensors.shape[1] - 1, "TDOA values must match sensor pairs."

    # Sensor pairs such that each column of 'sensors' is a sensor position
    sensor_pairs = [
        (sensors[:,0], sensors[:,i], tdoa_vals[i-1]) for i in range(1, sensors.shape[1])
    ]

    # If no figure is provided, create a new one
    fig = fig or go.Figure()

    # Annotate estimated source position
    fig.add_annotation(
        x=soln[0], y=soln[1],
        text=f"Estimated Source Position: ({soln[0]:.2f}, {soln[1]:.2f})",
        showarrow=True,
        arrowhead=0,
        ax=180,
        ay=0,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
    )

    # scatter sensors
    for idx, s in enumerate(sensors.T):
        fig.add_trace(go.Scatter(
            x=[s[0]], y=[s[1]],
            mode='markers+text',
            text=[f"S{idx+1}"],
            textposition='top center',
            marker=dict(size=10),
            name=f"S{idx+1}"
        ))

    # Adding state estimate from my project images from back in the day
    fig.add_trace(go.Scatter(
        x=[soln[0]], y=[soln[1]],
        marker=dict(size=15,color='red', symbol='x'),
        showlegend=False))

    colors = px.colors.qualitative.Plotly
    for i, (s1, s2, tdoa) in enumerate(sensor_pairs):
        x,y,F = tdoa_hyperbola(s1, s2, tdoa, (-30, 30), (-30, 30), n_points=200)

        fig.add_trace(go.Contour(
            x=x, y=y, z=F,
            contours=dict(start=0, end=0, size=1, coloring='none'),
            line=dict(width=2, color=colors[i % len(colors)]),
            name=f"τ_{i+2}1"
        ))

    fig.update_layout(
        title="Multiple TDOA Hyperbolas",
        xaxis=dict(scaleanchor='y', scaleratio=1)
    )

    # fig.show()

if __name__ == "__main__":
    st.title("TDOA Localization with ILS")

    # Sensor position DF editor
    default_sensor_pos = pd.DataFrame(
        [
            {"xpos": -8, "ypos":  8},
            {"xpos":  8, "ypos":  8},
            {"xpos":  8, "ypos": -8},
            {"xpos": -8, "ypos": -8},
        ]
    )
    st.markdown("## Sensor Positions \nEdit the sensor positions in the table above. Units are in meters.")
    edited_sensor_pos = st.data_editor(
        default_sensor_pos,
        num_rows="dynamic",
        column_config={
            "xpos": "X Position (m)",
            "ypos": "Y Position (m)",
        },
        hide_index=False,
    )

    # True source position editor
    default_source_pos = pd.DataFrame(
        [
            {"xpos":  -1, "ypos":  2.5},
        ]
    )
    st.markdown("## Sound Source Position \nEdit the source position to be located in the table above. Units are in meters.")

    edited_source_pos = st.data_editor(
        default_source_pos,
        num_rows="fixed",
        column_config={
            "xpos": "X Position (m)",
            "ypos": "Y Position (m)",
        },
        hide_index=True,
    )

    # -----

    # Get updated sensor positions and source position
    sensors = edited_sensor_pos.to_numpy(dtype=np.float64).T
    assert sensors.shape[0] == 2, "Sensors must be a 2xN array"
    source_position = edited_source_pos.to_numpy(dtype=np.float64).T.flatten()  # shape (2,)
    assert source_position.shape == (2,), "Source position must be a 1D array of size 2"

    tdoa_meas_vals = tdoa_measurements(source_position, sensors)
    # st.markdown(f"Computed TDOA measurements in meters (v_sound*tau) from source to sensors: {tdoa_meas_vals}")
    st.markdown("## Computed TDOA Measurements \nThese values are the computed TDOA measurements in meters (v_sound*tau) from source to sensors.")
    st.markdown("Here, d21 = (distance from reference to sensor 2) - (distance from reference to sensor 1).")
    dist_diff_df = pd.DataFrame(tdoa_meas_vals, columns=["TDOA (m)"], index=["d%d1" % (i+2) for i in range(tdoa_meas_vals.shape[0])])
    st.dataframe(dist_diff_df)

    soln = run_ils(tdoa_meas_vals, sensors, x0 = None)
    fig = go.Figure()
    plot_tdoa_curves(sensors, np.array(tdoa_meas_vals), soln, fig)
    st.plotly_chart(fig, use_container_width=True)
