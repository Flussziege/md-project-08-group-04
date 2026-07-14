#primer:

#example usage:
#make_box_trace(3.0)
#plot_xyz_trajectory_with_slider("results/2026-07-09_10-50-24/my_simulation_pos.xyz", box_length_nm=5, frame_stride=20)'

#first make box with make_box_trace(box_length_nm))
#then plot with plot_xyz_trajectory_with_slider
#->(path, box_length_nm, frame_stride=1, marker_size=4, save_html=True)
#the more steps the bigger the fram_stride has to be so it does not crash


from pathlib import Path
import numpy as np
import plotly.graph_objects as go


def read_xyz_trajectory(filename):
    """
    Liest eine XYZ-Trajektorie ein.

    Erwartetes Format pro Frame:
    N
    Kommentarzeile
    Atom x y z
    Atom x y z
    ...

    Rückgabe:
    positions : np.ndarray mit shape (n_frames, n_particles, 3)
    atom_names : Liste der Atomsymbole

    Wichtig:
    Die Datei my_simulation_pos.xyz aus eurem Code enthält Koordinaten in Å,
    weil write_xyz_trajectory vorher nm zu Å umrechnet.
    """

    filename = Path(filename)

    with open(filename, "r") as f:
        lines = f.readlines()

    n_particles = int(lines[0].strip())
    lines_per_frame = n_particles + 2
    n_frames = len(lines) // lines_per_frame

    positions = np.zeros((n_frames, n_particles, 3))
    atom_names = []

    for frame in range(n_frames):
        start = frame * lines_per_frame

        for i in range(n_particles):
            line = lines[start + 2 + i].split()

            atom = line[0]
            x = float(line[1])
            y = float(line[2])
            z = float(line[3])

            positions[frame, i, :] = [x, y, z]

            if frame == 0:
                atom_names.append(atom)

    return positions, atom_names


def make_box_trace(box_length_nm):
    """
    Erstellt die Kanten einer kubischen Simulationsbox.

    Input:
    box_length_nm : float
        Boxlänge in nm, so wie sie im Simulationscode steht.

    Wichtig:
    Die XYZ-Koordinaten sind in Å gespeichert.
    Deshalb wird nm → Å umgerechnet.
    """

    # nm in Å umrechnen
    L = box_length_nm * 10.0

    edges = [
        # untere Fläche
        ([0, 0, 0], [L, 0, 0]),
        ([L, 0, 0], [L, L, 0]),
        ([L, L, 0], [0, L, 0]),
        ([0, L, 0], [0, 0, 0]),

        # obere Fläche
        ([0, 0, L], [L, 0, L]),
        ([L, 0, L], [L, L, L]),
        ([L, L, L], [0, L, L]),
        ([0, L, L], [0, 0, L]),

        # vertikale Kanten
        ([0, 0, 0], [0, 0, L]),
        ([L, 0, 0], [L, 0, L]),
        ([L, L, 0], [L, L, L]),
        ([0, L, 0], [0, L, L]),
    ]

    x_box = []
    y_box = []
    z_box = []

    for p1, p2 in edges:
        x_box += [p1[0], p2[0], None]
        y_box += [p1[1], p2[1], None]
        z_box += [p1[2], p2[2], None]

    box_trace = go.Scatter3d(
        x=x_box,
        y=y_box,
        z=z_box,
        mode="lines",
        line=dict(width=6, color="orange"),
        name="Simulationsbox",
        hoverinfo="skip"
    )

    return box_trace

def plot_xyz_trajectory_with_slider(
        xyz_path,
        box_length_nm,
        frame_stride=10,
        marker_size=4,
        save_html=True
    ):
    """
    Interaktive 3D-Visualisierung einer XYZ-Trajektorie mit Slider.

    Parameters
    ----------
    xyz_path : str oder Path
        Exakter Pfad zur XYZ-Datei.

    box_length_nm : float
        Boxlänge in nm, so wie sie im Simulationscode steht.

    frame_stride : int
        Nur jeden n-ten Frame anzeigen.
        Bei vielen Frames z.B. 5, 10 oder 20 verwenden.

    marker_size : float
        Größe der Punktteilchen.

    save_html : bool
        Wenn True, wird der Plot zusätzlich als HTML-Datei gespeichert.
    """

    xyz_path = Path(xyz_path)

    if not xyz_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {xyz_path}")

    print(f"Verwende XYZ-Datei: {xyz_path}")

    # XYZ einlesen
    positions, atom_names = read_xyz_trajectory(xyz_path)

    # Nur jeden n-ten Frame nehmen, falls gewünscht
    positions = positions[::frame_stride]

    n_frames, n_particles, _ = positions.shape

    # Eure XYZ-Datei ist in Å, weil write_xyz_trajectory nm zu Å umrechnet.
    # Deshalb wird die Boxlänge von nm nach Å umgerechnet.
    box_length_angstrom = box_length_nm * 10.0

    # Erster Frame
    first_positions = positions[0]

    particle_trace = go.Scatter3d(
        x=first_positions[:, 0],
        y=first_positions[:, 1],
        z=first_positions[:, 2],
        mode="markers",
        marker=dict(
            size=marker_size,
            opacity=0.8
        ),
        text=[
            f"Frame 0<br>"
            f"Teilchen {i}<br>"
            f"x = {first_positions[i, 0]:.3f} Å<br>"
            f"y = {first_positions[i, 1]:.3f} Å<br>"
            f"z = {first_positions[i, 2]:.3f} Å"
            for i in range(n_particles)
        ],
        hoverinfo="text",
        name="Teilchen"
    )

    box_trace = make_box_trace(box_length_angstrom)

    fig = go.Figure(data=[particle_trace, box_trace])

    # Frames für Slider erzeugen
    frames = []

    for frame in range(n_frames):
        p = positions[frame]
        original_frame_number = frame * frame_stride

        trace = go.Scatter3d(
            x=p[:, 0],
            y=p[:, 1],
            z=p[:, 2],
            mode="markers",
            marker=dict(
                size=marker_size,
                opacity=0.8
            ),
            text=[
                f"Frame {original_frame_number}<br>"
                f"Teilchen {i}<br>"
                f"x = {p[i, 0]:.3f} Å<br>"
                f"y = {p[i, 1]:.3f} Å<br>"
                f"z = {p[i, 2]:.3f} Å"
                for i in range(n_particles)
            ],
            hoverinfo="text",
            name="Teilchen"
        )

        frames.append(
            go.Frame(
                data=[trace],
                traces=[0],
                name=str(frame),
                layout=go.Layout(
                    annotations=[
                        dict(
                            text=f"Frame: {original_frame_number}",
                            x=0.5,
                            y=1.05,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(size=16)
                        )
                    ]
                )
            )
        )

    fig.frames = frames

    # Slider
    slider_steps = []

    for frame in range(n_frames):
        original_frame_number = frame * frame_stride

        slider_steps.append(
            dict(
                method="animate",
                label=str(original_frame_number),
                args=[
                    [str(frame)],
                    dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0)
                    )
                ]
            )
        )

    fig.update_layout(
        title="MD-Trajektorie in der Simulationsbox",
        scene=dict(
            xaxis=dict(title="x / Å", range=[0, box_length_angstrom]),
            yaxis=dict(title="y / Å", range=[0, box_length_angstrom]),
            zaxis=dict(title="z / Å", range=[0, box_length_angstrom]),
            aspectmode="cube"
        ),
        annotations=[
            dict(
                text="Frame: 0",
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Frame: "),
                steps=slider_steps
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=50, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate"
                            )
                        ]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                                mode="immediate"
                            )
                        ]
                    )
                ]
            )
        ]
    )

    if save_html:
        html_path = xyz_path.with_suffix(".html")
        fig.write_html(html_path)
        print(f"Interaktive HTML-Datei gespeichert unter: {html_path}")

    fig.show()

make_box_trace(10.0)
plot_xyz_trajectory_with_slider("results/2026-07-13_21-17-25/my_simulation_pos.xyz", box_length_nm=10, frame_stride=50)