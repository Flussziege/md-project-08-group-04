"""
visualization_functions.py

Funktionen zur interaktiven 3D-Visualisierung einer XYZ-Trajektorie.

Optional können Cluster-IDs aus der von cluster_functions.py erzeugten
CSV-Datei eingelesen und durch unterschiedliche Farben dargestellt werden.
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

from cluster_functions import read_xyz_trajectory


UNCLUSTERED_COLOR = "lightgray"

CLUSTER_COLOR_PALETTE = (
    qualitative.Plotly
    + qualitative.D3
    + qualitative.G10
    + qualitative.Set3
    + qualitative.Dark24
    + qualitative.Light24
)


def read_cluster_assignments(
    csv_path: str | Path,
    n_particles: int,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Liest die Cluster-ID-CSV aus cluster_functions.py ein.

    Returns
    -------
    cluster_data
        Dictionary:
            cluster_data[frame]["cluster_id"]
            cluster_data[frame]["cluster_size"]
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cluster-CSV wurde nicht gefunden:\n{csv_path.resolve()}"
        )

    required_columns = {
        "frame",
        "particle",
        "cluster_id",
        "cluster_size",
    }

    cluster_data: dict[int, dict[str, np.ndarray]] = {}
    missing_value = -999999

    with csv_path.open(
        "r", newline="", encoding="utf-8"
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError("Die Cluster-CSV enthält keine Kopfzeile.")

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                "In der Cluster-CSV fehlen Spalten: "
                f"{sorted(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                frame = int(row["frame"])
                particle = int(row["particle"])
                cluster_id = int(row["cluster_id"])
                cluster_size = int(row["cluster_size"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Ungültiger Zahlenwert in CSV-Zeile {row_number}."
                ) from error

            if not 0 <= particle < n_particles:
                raise ValueError(
                    f"Ungültige Teilchennummer {particle} in "
                    f"CSV-Zeile {row_number}. Erlaubt: 0 bis "
                    f"{n_particles - 1}."
                )

            if frame not in cluster_data:
                cluster_data[frame] = {
                    "cluster_id": np.full(
                        n_particles, missing_value, dtype=int
                    ),
                    "cluster_size": np.full(
                        n_particles, missing_value, dtype=int
                    ),
                }

            if (
                cluster_data[frame]["cluster_id"][particle]
                != missing_value
            ):
                raise ValueError(
                    f"Doppelte Zuordnung für Frame {frame}, "
                    f"Teilchen {particle}."
                )

            cluster_data[frame]["cluster_id"][particle] = cluster_id
            cluster_data[frame]["cluster_size"][particle] = cluster_size

    for frame, frame_data in cluster_data.items():
        missing_particles = np.flatnonzero(
            frame_data["cluster_id"] == missing_value
        )

        if len(missing_particles):
            raise ValueError(
                f"Im Cluster-CSV-Frame {frame} fehlen "
                f"{len(missing_particles)} Teilchen. Erste fehlende "
                f"Indizes: {missing_particles[:10]}"
            )

    return cluster_data


def cluster_ids_to_colors(cluster_ids: np.ndarray) -> list[str]:
    """
    Wandelt Cluster-IDs in diskrete Plotly-Farben um.

    cluster_id = -1 wird grau dargestellt.
    """
    colors: list[str] = []

    for cluster_id in cluster_ids:
        cluster_id = int(cluster_id)

        if cluster_id < 0:
            colors.append(UNCLUSTERED_COLOR)
        else:
            colors.append(
                CLUSTER_COLOR_PALETTE[
                    cluster_id % len(CLUSTER_COLOR_PALETTE)
                ]
            )

    return colors


def make_particle_marker(
    marker_size: float,
    cluster_ids: np.ndarray | None = None,
) -> dict:
    """
    Erstellt das Marker-Dictionary für Plotly.

    Ohne cluster_ids bleibt die ursprüngliche einfarbige Darstellung erhalten.
    """
    marker = {
        "size": marker_size,
        "opacity": 0.8,
    }

    if cluster_ids is not None:
        marker["color"] = cluster_ids_to_colors(cluster_ids)

    return marker


def make_particle_hover_text(
    positions: np.ndarray,
    original_frame_number: int,
    cluster_ids: np.ndarray | None = None,
    cluster_sizes: np.ndarray | None = None,
) -> list[str]:
    """
    Erzeugt die Hovertexte eines Frames.
    """
    hover_text: list[str] = []

    for particle_index, (x, y, z) in enumerate(positions):
        text = (
            f"Frame {original_frame_number}<br>"
            f"Teilchen {particle_index}<br>"
            f"x = {x:.3f} Å<br>"
            f"y = {y:.3f} Å<br>"
            f"z = {z:.3f} Å"
        )

        if cluster_ids is not None and cluster_sizes is not None:
            cluster_id = int(cluster_ids[particle_index])
            cluster_size = int(cluster_sizes[particle_index])

            if cluster_id >= 0:
                text += (
                    f"<br>Cluster-ID = {cluster_id}"
                    f"<br>Clustergröße = {cluster_size}"
                )
            else:
                text += (
                    "<br>Cluster-ID = -1"
                    "<br>nicht als Cluster klassifiziert"
                )

        hover_text.append(text)

    return hover_text


def make_box_trace(box_length_nm: float) -> go.Scatter3d:
    """
    Erstellt die Kanten einer kubischen Simulationsbox.

    box_length_nm wird intern nach Å umgerechnet, weil die XYZ-Datei
    Å-Koordinaten enthält.
    """
    length_angstrom = box_length_nm * 10.0

    edges = [
        ([0, 0, 0], [length_angstrom, 0, 0]),
        ([length_angstrom, 0, 0], [length_angstrom, length_angstrom, 0]),
        ([length_angstrom, length_angstrom, 0], [0, length_angstrom, 0]),
        ([0, length_angstrom, 0], [0, 0, 0]),
        ([0, 0, length_angstrom], [length_angstrom, 0, length_angstrom]),
        (
            [length_angstrom, 0, length_angstrom],
            [length_angstrom, length_angstrom, length_angstrom],
        ),
        (
            [length_angstrom, length_angstrom, length_angstrom],
            [0, length_angstrom, length_angstrom],
        ),
        ([0, length_angstrom, length_angstrom], [0, 0, length_angstrom]),
        ([0, 0, 0], [0, 0, length_angstrom]),
        ([length_angstrom, 0, 0], [length_angstrom, 0, length_angstrom]),
        (
            [length_angstrom, length_angstrom, 0],
            [length_angstrom, length_angstrom, length_angstrom],
        ),
        ([0, length_angstrom, 0], [0, length_angstrom, length_angstrom]),
    ]

    x_box: list[float | None] = []
    y_box: list[float | None] = []
    z_box: list[float | None] = []

    for point_1, point_2 in edges:
        x_box += [point_1[0], point_2[0], None]
        y_box += [point_1[1], point_2[1], None]
        z_box += [point_1[2], point_2[2], None]

    return go.Scatter3d(
        x=x_box,
        y=y_box,
        z=z_box,
        mode="lines",
        line={"width": 6, "color": "orange"},
        name="Simulationsbox",
        hoverinfo="skip",
    )


def plot_xyz_trajectory_with_slider(
    xyz_path: str | Path,
    box_length_nm: float,
    frame_stride: int = 10,
    marker_size: float = 4,
    save_html: bool = True,
    cluster: bool = False,
    cluster_csv_path: str | Path | None = None,
) -> None:
    """
    Interaktive 3D-Visualisierung einer XYZ-Trajektorie mit Slider.

    Parameters
    ----------
    xyz_path
        Exakter Pfad zur XYZ-Datei.
    box_length_nm
        Boxlänge in nm.
    frame_stride
        Nur jeden n-ten Frame darstellen.
    marker_size
        Größe der Teilchenmarker.
    save_html
        Bei True wird zusätzlich eine HTML-Datei gespeichert.
    cluster
        Bei True werden Teilchen anhand der Cluster-CSV eingefärbt.
    cluster_csv_path
        Pfad zur Cluster-ID-CSV. Erforderlich, wenn cluster=True.
    """
    xyz_path = Path(xyz_path)

    if not xyz_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {xyz_path}")
    if frame_stride < 1:
        raise ValueError("frame_stride muss mindestens 1 sein.")

    all_positions, atom_names = read_xyz_trajectory(xyz_path)
    original_n_frames, n_particles, _ = all_positions.shape

    if cluster:
        if cluster_csv_path is None:
            raise ValueError(
                "Bei cluster=True muss cluster_csv_path angegeben werden."
            )

        cluster_data = read_cluster_assignments(
            csv_path=cluster_csv_path,
            n_particles=n_particles,
        )
    else:
        cluster_data = None

    positions = all_positions[::frame_stride]
    n_frames = positions.shape[0]
    box_length_angstrom = box_length_nm * 10.0

    if cluster and cluster_data is not None:
        displayed_original_frames = range(
            0, original_n_frames, frame_stride
        )
        missing_cluster_frames = [
            frame_number
            for frame_number in displayed_original_frames
            if frame_number not in cluster_data
        ]

        if missing_cluster_frames:
            raise ValueError(
                "In der Cluster-CSV fehlen Daten für dargestellte Frames. "
                f"Erste fehlende Frames: {missing_cluster_frames[:10]}. "
                "Erzeuge die CSV mit analysis_frame_stride=1 oder mit "
                "einem passenden Analyse-Stride."
            )

    first_positions = positions[0]
    first_original_frame_number = 0

    if cluster and cluster_data is not None:
        first_cluster_ids = cluster_data[0]["cluster_id"]
        first_cluster_sizes = cluster_data[0]["cluster_size"]
    else:
        first_cluster_ids = None
        first_cluster_sizes = None

    particle_trace = go.Scatter3d(
        x=first_positions[:, 0],
        y=first_positions[:, 1],
        z=first_positions[:, 2],
        mode="markers",
        marker=make_particle_marker(
            marker_size=marker_size,
            cluster_ids=first_cluster_ids,
        ),
        text=make_particle_hover_text(
            positions=first_positions,
            original_frame_number=first_original_frame_number,
            cluster_ids=first_cluster_ids,
            cluster_sizes=first_cluster_sizes,
        ),
        hoverinfo="text",
        name="Teilchen",
    )

    box_trace = make_box_trace(box_length_nm)

    fig = go.Figure(data=[particle_trace, box_trace])

    frames: list[go.Frame] = []

    for frame in range(n_frames):
        frame_positions = positions[frame]
        original_frame_number = frame * frame_stride

        if cluster and cluster_data is not None:
            frame_cluster_ids = cluster_data[
                original_frame_number
            ]["cluster_id"]
            frame_cluster_sizes = cluster_data[
                original_frame_number
            ]["cluster_size"]
        else:
            frame_cluster_ids = None
            frame_cluster_sizes = None

        trace = go.Scatter3d(
            x=frame_positions[:, 0],
            y=frame_positions[:, 1],
            z=frame_positions[:, 2],
            mode="markers",
            marker=make_particle_marker(
                marker_size=marker_size,
                cluster_ids=frame_cluster_ids,
            ),
            text=make_particle_hover_text(
                positions=frame_positions,
                original_frame_number=original_frame_number,
                cluster_ids=frame_cluster_ids,
                cluster_sizes=frame_cluster_sizes,
            ),
            hoverinfo="text",
            name="Teilchen",
        )

        frames.append(
            go.Frame(
                data=[trace],
                traces=[0],
                name=str(frame),
                layout=go.Layout(
                    annotations=[
                        {
                            "text": f"Frame: {original_frame_number}",
                            "x": 0.5,
                            "y": 1.05,
                            "xref": "paper",
                            "yref": "paper",
                            "showarrow": False,
                            "font": {"size": 16},
                        }
                    ]
                ),
            )
        )

    fig.frames = frames

    slider_steps = []

    for frame in range(n_frames):
        original_frame_number = frame * frame_stride

        slider_steps.append(
            {
                "method": "animate",
                "label": str(original_frame_number),
                "args": [
                    [str(frame)],
                    {
                        "mode": "immediate",
                        "frame": {
                            "duration": 0,
                            "redraw": True,
                        },
                        "transition": {"duration": 0},
                    },
                ],
            }
        )

    fig.update_layout(
        title="MD-Trajektorie in der Simulationsbox",
        scene={
            "xaxis": {
                "title": "x / Å",
                "range": [0, box_length_angstrom],
            },
            "yaxis": {
                "title": "y / Å",
                "range": [0, box_length_angstrom],
            },
            "zaxis": {
                "title": "z / Å",
                "range": [0, box_length_angstrom],
            },
            "aspectmode": "cube",
        },
        annotations=[
            {
                "text": "Frame: 0",
                "x": 0.5,
                "y": 1.05,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16},
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Frame: "},
                "steps": slider_steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": 50,
                                    "redraw": True,
                                },
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
    )

    if save_html:
        if cluster:
            html_path = xyz_path.with_name(
                f"{xyz_path.stem}_clusters.html"
            )
        else:
            html_path = xyz_path.with_suffix(".html")

        fig.write_html(html_path)
        print(f"Interaktive HTML-Datei gespeichert unter: {html_path}")

    fig.show()
