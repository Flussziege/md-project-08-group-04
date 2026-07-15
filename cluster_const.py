"""
cluster_const.py

Konstante Clusterzuordnung für eine MD-Trajektorie.

Die räumlichen Cluster werden ausschließlich im ersten XYZ-Frame erkannt.
Danach behält jedes Teilchen über die gesamte Trajektorie dieselbe Cluster-ID.

Zweck
-----
Dadurch bleiben die Farben der ursprünglichen Cluster konstant. In der
Visualisierung kann dann beobachtet werden, wie Teilchen verschiedener
Startcluster räumlich miteinander vermischt werden.

Beispiel
--------
Ein Teilchen erhält in Frame 0 cluster_id = 2. Diese ID bleibt in jedem
späteren Frame ebenfalls 2, unabhängig davon, wo sich das Teilchen befindet.

Die erzeugte Cluster-ID-CSV besitzt dasselbe Format wie die dynamische
Clusteranalyse aus cluster_functions.py und kann deshalb direkt mit
visualization_functions.py verwendet werden.
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
from scipy.spatial import cKDTree

from cluster_functions import (
    identify_clusters_in_frame,
    read_xyz_trajectory,
)


def _find_periodic_neighbor_pairs(
    positions: np.ndarray,
    box_length: float,
    cutoff: float,
) -> np.ndarray:
    """
    Findet alle eindeutigen Teilchenpaare innerhalb des Cutoffs.

    Periodische Randbedingungen werden mit einem periodischen cKDTree
    berücksichtigt.

    Returns
    -------
    pairs
        Ganzzahliges Array mit Form (n_pairs, 2).
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
    if cutoff >= box_length / 2:
        raise ValueError(
            "cutoff muss kleiner als die halbe Boxlänge sein."
        )

    wrapped_positions = np.mod(positions, box_length)

    tree = cKDTree(
        wrapped_positions,
        boxsize=box_length,
    )

    pairs = tree.query_pairs(
        r=cutoff,
        output_type="ndarray",
    )

    if pairs.size == 0:
        return np.empty((0, 2), dtype=int)

    return np.asarray(pairs, dtype=int)


def _neighbor_counts_from_pairs(
    pairs: np.ndarray,
    n_particles: int,
) -> np.ndarray:
    """
    Berechnet aus einer Paarliste die Nachbarzahl jedes Teilchens.
    """
    neighbor_counts = np.zeros(n_particles, dtype=int)

    if len(pairs) == 0:
        return neighbor_counts

    np.add.at(neighbor_counts, pairs[:, 0], 1)
    np.add.at(neighbor_counts, pairs[:, 1], 1)

    return neighbor_counts


def _calculate_mixing_metrics(
    pairs: np.ndarray,
    constant_cluster_ids: np.ndarray,
) -> dict[str, float | int]:
    """
    Bestimmt einfache Durchmischungskennzahlen anhand der Startclusterfarben.

    Berücksichtigt werden nur Teilchen, die in Frame 0 einem gültigen
    Cluster zugeordnet wurden, also cluster_id >= 0.

    Kennzahlen
    ----------
    same_cluster_neighbor_pairs
        Nachbarpaare mit derselben konstanten Startcluster-ID.

    foreign_cluster_neighbor_pairs
        Nachbarpaare aus zwei verschiedenen Startclustern.

    foreign_neighbor_pair_fraction
        Anteil der fremdfarbigen Paare an allen Nachbarpaaren zwischen
        ursprünglich geclusterten Teilchen.

    particles_with_foreign_neighbor
        Zahl der ursprünglich geclusterten Teilchen, die mindestens einen
        direkten Nachbarn aus einem anderen Startcluster besitzen.

    particles_with_foreign_neighbor_fraction
        Entsprechender Anteil an allen ursprünglich geclusterten Teilchen.
    """
    constant_cluster_ids = np.asarray(
        constant_cluster_ids,
        dtype=int,
    )

    clustered_mask = constant_cluster_ids >= 0
    number_of_initially_clustered_particles = int(
        np.count_nonzero(clustered_mask)
    )

    if len(pairs) == 0:
        return {
            "same_cluster_neighbor_pairs": 0,
            "foreign_cluster_neighbor_pairs": 0,
            "foreign_neighbor_pair_fraction": 0.0,
            "particles_with_foreign_neighbor": 0,
            "particles_with_foreign_neighbor_fraction": 0.0,
        }

    id_i = constant_cluster_ids[pairs[:, 0]]
    id_j = constant_cluster_ids[pairs[:, 1]]

    both_initially_clustered = (id_i >= 0) & (id_j >= 0)

    same_cluster_pairs_mask = (
        both_initially_clustered
        & (id_i == id_j)
    )

    foreign_cluster_pairs_mask = (
        both_initially_clustered
        & (id_i != id_j)
    )

    same_cluster_neighbor_pairs = int(
        np.count_nonzero(same_cluster_pairs_mask)
    )

    foreign_cluster_neighbor_pairs = int(
        np.count_nonzero(foreign_cluster_pairs_mask)
    )

    considered_pair_count = (
        same_cluster_neighbor_pairs
        + foreign_cluster_neighbor_pairs
    )

    if considered_pair_count > 0:
        foreign_neighbor_pair_fraction = (
            foreign_cluster_neighbor_pairs
            / considered_pair_count
        )
    else:
        foreign_neighbor_pair_fraction = 0.0

    particles_with_foreign_neighbor_mask = np.zeros(
        len(constant_cluster_ids),
        dtype=bool,
    )

    foreign_pairs = pairs[
        foreign_cluster_pairs_mask
    ]

    if len(foreign_pairs) > 0:
        particles_with_foreign_neighbor_mask[
            foreign_pairs[:, 0]
        ] = True

        particles_with_foreign_neighbor_mask[
            foreign_pairs[:, 1]
        ] = True

    particles_with_foreign_neighbor = int(
        np.count_nonzero(
            particles_with_foreign_neighbor_mask
        )
    )

    if number_of_initially_clustered_particles > 0:
        particles_with_foreign_neighbor_fraction = (
            particles_with_foreign_neighbor
            / number_of_initially_clustered_particles
        )
    else:
        particles_with_foreign_neighbor_fraction = 0.0

    return {
        "same_cluster_neighbor_pairs": (
            same_cluster_neighbor_pairs
        ),
        "foreign_cluster_neighbor_pairs": (
            foreign_cluster_neighbor_pairs
        ),
        "foreign_neighbor_pair_fraction": (
            foreign_neighbor_pair_fraction
        ),
        "particles_with_foreign_neighbor": (
            particles_with_foreign_neighbor
        ),
        "particles_with_foreign_neighbor_fraction": (
            particles_with_foreign_neighbor_fraction
        ),
    }


def analyze_xyz_trajectory_clusters_const(
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
    Bestimmt Cluster nur in Frame 0 und behält deren IDs dauerhaft bei.

    Parameters
    ----------
    xyz_path
        Pfad zur XYZ-Trajektorie.

    box_length_nm
        Länge der kubischen Simulationsbox in nm.

    cluster_cutoff_nm
        Abstand in nm, bis zu dem Teilchen als direkte Nachbarn gelten.

    minimum_cluster_size
        Minimale Teilchenzahl eines Startclusters.

    time_step_ps
        Zeit zwischen zwei ursprünglichen XYZ-Frames in ps.

    frame_stride
        Nur jeden n-ten Frame in die CSV schreiben. Für die flexible
        Visualisierung wird frame_stride=1 empfohlen.

    cluster_csv_path
        Optionaler Ausgabepfad der konstanten Cluster-ID-CSV.

    summary_csv_path
        Optionaler Ausgabepfad der Durchmischungs-Zusammenfassung.

    Returns
    -------
    cluster_csv_path, summary_csv_path
        Pfade der beiden erzeugten CSV-Dateien.
    """
    xyz_path = Path(xyz_path)

    if frame_stride < 1:
        raise ValueError("frame_stride muss mindestens 1 sein.")
    if time_step_ps <= 0:
        raise ValueError("time_step_ps muss größer als null sein.")

    positions, atom_names = read_xyz_trajectory(
        xyz_path
    )

    n_frames, n_particles, _ = positions.shape

    # XYZ-Koordinaten aus LJ_gas.py liegen in Å.
    box_length_angstrom = box_length_nm * 10.0
    cluster_cutoff_angstrom = (
        cluster_cutoff_nm * 10.0
    )

    # ------------------------------------------------------------
    # Cluster ausschließlich im ersten Frame bestimmen
    # ------------------------------------------------------------

    (
        constant_cluster_ids,
        constant_cluster_sizes_per_particle,
        first_neighbor_counts,
        initial_cluster_sizes,
    ) = identify_clusters_in_frame(
        positions=positions[0],
        box_length=box_length_angstrom,
        cutoff=cluster_cutoff_angstrom,
        min_cluster_size=minimum_cluster_size,
    )

    number_of_initial_clusters = len(
        initial_cluster_sizes
    )

    initially_clustered_particles = int(
        np.count_nonzero(
            constant_cluster_ids >= 0
        )
    )

    initially_unclustered_particles = (
        n_particles
        - initially_clustered_particles
    )

    if number_of_initial_clusters > 0:
        largest_initial_cluster_size = int(
            initial_cluster_sizes[0]
        )
    else:
        largest_initial_cluster_size = 0

    # ------------------------------------------------------------
    # Standard-Ausgabepfade
    # ------------------------------------------------------------

    if cluster_csv_path is None:
        cluster_csv_path = xyz_path.with_name(
            f"{xyz_path.stem}_cluster_const_ids.csv"
        )
    else:
        cluster_csv_path = Path(
            cluster_csv_path
        )

    if summary_csv_path is None:
        summary_csv_path = xyz_path.with_name(
            f"{xyz_path.stem}_cluster_const_summary.csv"
        )
    else:
        summary_csv_path = Path(
            summary_csv_path
        )

    cluster_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        "number_of_initial_clusters",
        "largest_initial_cluster_size",
        "initially_clustered_particles",
        "initially_unclustered_particles",
        "same_cluster_neighbor_pairs",
        "foreign_cluster_neighbor_pairs",
        "foreign_neighbor_pair_fraction",
        "particles_with_foreign_neighbor",
        "particles_with_foreign_neighbor_fraction",
    ]

    summary_rows: list[list[object]] = []

    with cluster_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as cluster_file:
        cluster_writer = csv.writer(
            cluster_file
        )

        cluster_writer.writerow(
            cluster_header
        )

        analyzed_frame_counter = 0

        for frame_index in range(
            0,
            n_frames,
            frame_stride,
        ):
            frame_positions = positions[
                frame_index
            ]

            current_pairs = (
                _find_periodic_neighbor_pairs(
                    positions=frame_positions,
                    box_length=box_length_angstrom,
                    cutoff=cluster_cutoff_angstrom,
                )
            )

            current_neighbor_counts = (
                _neighbor_counts_from_pairs(
                    pairs=current_pairs,
                    n_particles=n_particles,
                )
            )

            mixing_metrics = (
                _calculate_mixing_metrics(
                    pairs=current_pairs,
                    constant_cluster_ids=(
                        constant_cluster_ids
                    ),
                )
            )

            time_ps = (
                frame_index * time_step_ps
            )

            for particle_index in range(
                n_particles
            ):
                cluster_id = int(
                    constant_cluster_ids[
                        particle_index
                    ]
                )

                cluster_size = int(
                    constant_cluster_sizes_per_particle[
                        particle_index
                    ]
                )

                x, y, z = frame_positions[
                    particle_index
                ]

                cluster_writer.writerow(
                    [
                        frame_index,
                        time_ps,
                        particle_index,
                        atom_names[
                            particle_index
                        ],
                        cluster_id,
                        cluster_size,
                        cluster_id >= 0,
                        cluster_id == 0,
                        int(
                            current_neighbor_counts[
                                particle_index
                            ]
                        ),
                        x,
                        y,
                        z,
                    ]
                )

            summary_rows.append(
                [
                    frame_index,
                    time_ps,
                    number_of_initial_clusters,
                    largest_initial_cluster_size,
                    initially_clustered_particles,
                    initially_unclustered_particles,
                    mixing_metrics[
                        "same_cluster_neighbor_pairs"
                    ],
                    mixing_metrics[
                        "foreign_cluster_neighbor_pairs"
                    ],
                    mixing_metrics[
                        "foreign_neighbor_pair_fraction"
                    ],
                    mixing_metrics[
                        "particles_with_foreign_neighbor"
                    ],
                    mixing_metrics[
                        "particles_with_foreign_neighbor_fraction"
                    ],
                ]
            )

            analyzed_frame_counter += 1

            print(
                f"Frame {frame_index:>6} | "
                f"fremdfarbige Nachbarpaare: "
                f"{mixing_metrics['foreign_cluster_neighbor_pairs']:>6} | "
                f"Durchmischungsanteil: "
                f"{mixing_metrics['foreign_neighbor_pair_fraction']:>7.3f}"
            )

    with summary_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as summary_file:
        summary_writer = csv.writer(
            summary_file
        )

        summary_writer.writerow(
            summary_header
        )

        summary_writer.writerows(
            summary_rows
        )

    print("\n" + "=" * 70)
    print("KONSTANTE CLUSTERZUORDNUNG ABGESCHLOSSEN")
    print("=" * 70)
    print("Die Cluster wurden nur in Frame 0 bestimmt.")
    print("Jedes Teilchen behält seine anfängliche Cluster-ID.")
    print(f"Analysierte Frames: {analyzed_frame_counter}")
    print(f"Teilchen pro Frame: {n_particles}")
    print(f"Anfängliche Clusterzahl: {number_of_initial_clusters}")
    print(
        "Anfangs größter Cluster: "
        f"{largest_initial_cluster_size}"
    )
    print(
        "Anfangs geclusterte Teilchen: "
        f"{initially_clustered_particles}"
    )
    print(
        f"Cluster-Cutoff: {cluster_cutoff_nm:.6f} nm"
    )
    print(
        "Minimale Clustergröße: "
        f"{minimum_cluster_size}"
    )
    print(
        f"Konstante Cluster-IDs: "
        f"{cluster_csv_path.resolve()}"
    )
    print(
        f"Durchmischungs-Zusammenfassung: "
        f"{summary_csv_path.resolve()}"
    )
    print("=" * 70)

    return (
        cluster_csv_path,
        summary_csv_path,
    )
