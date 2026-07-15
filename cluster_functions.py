"""
cluster_functions.py

Funktionen zum:
- Einlesen einer XYZ-Trajektorie
- Erkennen räumlicher Cluster unter periodischen Randbedingungen
- Speichern der Cluster-ID jedes Teilchens in einer CSV-Datei
- Speichern einer Zusammenfassung pro Frame

Clusterdefinition
-----------------
Zwei Teilchen gelten als direkte Nachbarn, wenn ihr Abstand kleiner oder
gleich dem gewählten Cluster-Cutoff ist. Alle über solche Nachbarschaften
zusammenhängenden Teilchen bilden einen Cluster.

Die Cluster-IDs werden in jedem Frame nach Clustergröße vergeben:
    cluster_id = 0   größter Cluster
    cluster_id = 1   zweitgrößter Cluster
    cluster_id = -1  Einzelteilchen oder Gruppe kleiner als min_cluster_size
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def read_xyz_trajectory(filename: str | Path) -> tuple[np.ndarray, list[str]]:
    """
    Liest eine XYZ-Trajektorie vollständig ein.

    Parameters
    ----------
    filename
        Pfad zur XYZ-Datei.

    Returns
    -------
    positions
        Array mit Form (n_frames, n_particles, 3).
        Die Koordinaten bleiben in der Einheit der XYZ-Datei.
    atom_names
        Elementsymbole der Teilchen aus dem ersten Frame.
    """
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(
            f"XYZ-Datei wurde nicht gefunden:\n{filename.resolve()}"
        )

    with filename.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    if len(lines) < 3:
        raise ValueError("Die XYZ-Datei enthält nicht genügend Zeilen.")

    try:
        n_particles = int(lines[0].strip())
    except ValueError as error:
        raise ValueError(
            "Die erste Zeile der XYZ-Datei muss die Teilchenzahl enthalten."
        ) from error

    if n_particles <= 0:
        raise ValueError("Die Teilchenzahl muss größer als null sein.")

    lines_per_frame = n_particles + 2

    if len(lines) % lines_per_frame != 0:
        raise ValueError(
            "Die Anzahl der Zeilen passt nicht zum XYZ-Format.\n"
            f"Teilchenzahl: {n_particles}\n"
            f"Zeilen pro Frame: {lines_per_frame}\n"
            f"Gesamtzahl der Zeilen: {len(lines)}"
        )

    n_frames = len(lines) // lines_per_frame
    positions = np.zeros((n_frames, n_particles, 3), dtype=float)
    atom_names: list[str] = []

    for frame_index in range(n_frames):
        frame_start = frame_index * lines_per_frame

        try:
            frame_particle_count = int(lines[frame_start].strip())
        except ValueError as error:
            raise ValueError(
                f"Ungültige Teilchenzahl in Frame {frame_index}."
            ) from error

        if frame_particle_count != n_particles:
            raise ValueError(
                f"Frame {frame_index} enthält laut Datei "
                f"{frame_particle_count} Teilchen; erwartet wurden {n_particles}."
            )

        for particle_index in range(n_particles):
            line_index = frame_start + 2 + particle_index
            columns = lines[line_index].split()

            if len(columns) < 4:
                raise ValueError(
                    f"Ungültige Atomzeile in Frame {frame_index}, "
                    f"Teilchen {particle_index}:\n{lines[line_index]}"
                )

            atom_name = columns[0]

            try:
                coordinates = [
                    float(columns[1]),
                    float(columns[2]),
                    float(columns[3]),
                ]
            except ValueError as error:
                raise ValueError(
                    f"Ungültige Koordinaten in Frame {frame_index}, "
                    f"Teilchen {particle_index}:\n{lines[line_index]}"
                ) from error

            positions[frame_index, particle_index] = coordinates

            if frame_index == 0:
                atom_names.append(atom_name)

    return positions, atom_names


def identify_clusters_in_frame(
    positions: np.ndarray,
    box_length: float,
    cutoff: float,
    min_cluster_size: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Erkennt Cluster in einem einzelnen Frame.

    Periodische Randbedingungen werden über einen periodischen cKDTree
    berücksichtigt.

    Parameters
    ----------
    positions
        Positionen mit Form (n_particles, 3).
    box_length
        Länge der kubischen Box in derselben Einheit wie positions.
    cutoff
        Maximaler Abstand für eine direkte Nachbarschaft.
    min_cluster_size
        Minimale Teilchenzahl, damit eine Komponente als Cluster zählt.

    Returns
    -------
    cluster_ids
        Cluster-ID jedes Teilchens.
    cluster_sizes_per_particle
        Größe des Clusters jedes Teilchens; 0 bei cluster_id = -1.
    neighbor_counts
        Anzahl direkter Nachbarn jedes Teilchens.
    cluster_sizes
        Größen der gültigen Cluster, nach Größe absteigend sortiert.
    """
    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            "positions muss die Form (n_particles, 3) besitzen."
        )
    if box_length <= 0:
        raise ValueError("box_length muss größer als null sein.")
    if cutoff <= 0:
        raise ValueError("cutoff muss größer als null sein.")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size muss mindestens 1 sein.")
    if cutoff >= box_length / 2:
        raise ValueError(
            "cutoff muss kleiner als die halbe Boxlänge sein, damit die "
            "Minimum-Image-Zuordnung eindeutig bleibt."
        )

    n_particles = positions.shape[0]

    # cKDTree mit boxsize erwartet 0 <= Koordinate < box_length.
    wrapped_positions = np.mod(positions, box_length)

    tree = cKDTree(wrapped_positions, boxsize=box_length)
    pairs = tree.query_pairs(r=cutoff, output_type="ndarray")

    if pairs.size == 0:
        adjacency_graph = coo_matrix(
            (n_particles, n_particles), dtype=np.int8
        ).tocsr()
    else:
        particle_i = pairs[:, 0]
        particle_j = pairs[:, 1]

        rows = np.concatenate((particle_i, particle_j))
        columns = np.concatenate((particle_j, particle_i))
        data = np.ones(len(rows), dtype=np.int8)

        adjacency_graph = coo_matrix(
            (data, (rows, columns)),
            shape=(n_particles, n_particles),
        ).tocsr()

    neighbor_counts = np.diff(adjacency_graph.indptr)

    n_components, raw_component_labels = connected_components(
        adjacency_graph,
        directed=False,
        return_labels=True,
    )

    raw_component_sizes = np.bincount(
        raw_component_labels,
        minlength=n_components,
    )

    valid_components = [
        component_id
        for component_id in range(n_components)
        if raw_component_sizes[component_id] >= min_cluster_size
    ]

    # Größte Komponente erhält ID 0. Bei gleicher Größe entscheidet
    # der kleinste enthaltene Teilchenindex.
    def component_sort_key(component_id: int) -> tuple[int, int]:
        members = np.flatnonzero(raw_component_labels == component_id)
        return (
            -int(raw_component_sizes[component_id]),
            int(members.min()),
        )

    valid_components.sort(key=component_sort_key)

    cluster_ids = np.full(n_particles, -1, dtype=int)
    cluster_sizes_per_particle = np.zeros(n_particles, dtype=int)
    cluster_sizes: list[int] = []

    for new_cluster_id, component_id in enumerate(valid_components):
        members = raw_component_labels == component_id
        component_size = int(np.count_nonzero(members))

        cluster_ids[members] = new_cluster_id
        cluster_sizes_per_particle[members] = component_size
        cluster_sizes.append(component_size)

    return (
        cluster_ids,
        cluster_sizes_per_particle,
        neighbor_counts,
        np.asarray(cluster_sizes, dtype=int),
    )


def analyze_xyz_trajectory_clusters(
    xyz_path: str | Path,
    box_length_nm: float,
    cluster_cutoff_nm: float,
    minimum_cluster_size: int = 3,
    time_step_ps: float = 0.001,
    frame_stride: int = 1,
    cluster_csv_path: str | Path | None = None,
    summary_csv_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """
    Analysiert eine XYZ-Trajektorie und schreibt zwei CSV-Dateien.

    Die XYZ-Trajektorie aus LJ_gas.py enthält Å-Koordinaten. Deshalb
    werden Boxlänge und Cutoff von nm nach Å umgerechnet.

    Cluster-ID-CSV
    --------------
    Eine Zeile pro Teilchen und analysiertem Frame.

    Summary-CSV
    -----------
    Eine Zeile pro analysiertem Frame.
    """
    xyz_path = Path(xyz_path)

    if frame_stride < 1:
        raise ValueError("frame_stride muss mindestens 1 sein.")
    if time_step_ps <= 0:
        raise ValueError("time_step_ps muss größer als null sein.")

    positions, atom_names = read_xyz_trajectory(xyz_path)
    n_frames, n_particles, _ = positions.shape

    box_length_angstrom = box_length_nm * 10.0
    cluster_cutoff_angstrom = cluster_cutoff_nm * 10.0

    if cluster_csv_path is None:
        cluster_csv_path = xyz_path.with_name(
            f"{xyz_path.stem}_cluster_ids.csv"
        )
    else:
        cluster_csv_path = Path(cluster_csv_path)

    if summary_csv_path is None:
        summary_csv_path = xyz_path.with_name(
            f"{xyz_path.stem}_cluster_summary.csv"
        )
    else:
        summary_csv_path = Path(summary_csv_path)

    cluster_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    cluster_header = [
        "frame",
        "time_ps",
        "particle",
        "atom",
        "cluster_id",
        "cluster_size",
        "is_clustered",
        "is_largest_cluster",
        "neighbor_count",
        "x_angstrom",
        "y_angstrom",
        "z_angstrom",
    ]

    summary_header = [
        "frame",
        "time_ps",
        "number_of_clusters",
        "largest_cluster_size",
        "largest_cluster_fraction",
        "clustered_particles",
        "clustered_fraction",
        "mean_cluster_size",
        "weighted_mean_cluster_size",
        "isolated_or_small_group_particles",
    ]

    summary_rows: list[list[object]] = []

    with cluster_csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as cluster_file:
        cluster_writer = csv.writer(cluster_file)
        cluster_writer.writerow(cluster_header)

        analyzed_frame_counter = 0

        for frame_index in range(0, n_frames, frame_stride):
            frame_positions = positions[frame_index]

            (
                cluster_ids,
                cluster_sizes_per_particle,
                neighbor_counts,
                cluster_sizes,
            ) = identify_clusters_in_frame(
                positions=frame_positions,
                box_length=box_length_angstrom,
                cutoff=cluster_cutoff_angstrom,
                min_cluster_size=minimum_cluster_size,
            )

            time_ps = frame_index * time_step_ps
            number_of_clusters = len(cluster_sizes)

            if number_of_clusters:
                largest_cluster_size = int(cluster_sizes[0])
                mean_cluster_size = float(np.mean(cluster_sizes))
                weighted_mean_cluster_size = float(
                    np.sum(cluster_sizes**2) / np.sum(cluster_sizes)
                )
            else:
                largest_cluster_size = 0
                mean_cluster_size = 0.0
                weighted_mean_cluster_size = 0.0

            clustered_particles = int(np.count_nonzero(cluster_ids >= 0))
            isolated_or_small_group_particles = (
                n_particles - clustered_particles
            )
            largest_cluster_fraction = largest_cluster_size / n_particles
            clustered_fraction = clustered_particles / n_particles

            for particle_index in range(n_particles):
                cluster_id = int(cluster_ids[particle_index])
                x, y, z = frame_positions[particle_index]

                cluster_writer.writerow(
                    [
                        frame_index,
                        time_ps,
                        particle_index,
                        atom_names[particle_index],
                        cluster_id,
                        int(cluster_sizes_per_particle[particle_index]),
                        cluster_id >= 0,
                        cluster_id == 0,
                        int(neighbor_counts[particle_index]),
                        x,
                        y,
                        z,
                    ]
                )

            summary_rows.append(
                [
                    frame_index,
                    time_ps,
                    number_of_clusters,
                    largest_cluster_size,
                    largest_cluster_fraction,
                    clustered_particles,
                    clustered_fraction,
                    mean_cluster_size,
                    weighted_mean_cluster_size,
                    isolated_or_small_group_particles,
                ]
            )

            analyzed_frame_counter += 1
            print(
                f"Frame {frame_index:>6} | "
                f"Cluster: {number_of_clusters:>4} | "
                f"größter Cluster: {largest_cluster_size:>5} | "
                f"Anteil: {largest_cluster_fraction:>7.3f}"
            )

    with summary_csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow(summary_header)
        summary_writer.writerows(summary_rows)

    print("\n" + "=" * 70)
    print("Clusteranalyse abgeschlossen.")
    print("=" * 70)
    print(f"Analysierte Frames: {analyzed_frame_counter}")
    print(f"Teilchen pro Frame: {n_particles}")
    print(f"Cluster-Cutoff: {cluster_cutoff_nm:.6f} nm")
    print(f"Minimale Clustergröße: {minimum_cluster_size}")
    print(f"Cluster-IDs: {cluster_csv_path.resolve()}")
    print(f"Zusammenfassung: {summary_csv_path.resolve()}")
    print("=" * 70)

    return cluster_csv_path, summary_csv_path
