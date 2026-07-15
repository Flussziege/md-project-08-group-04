# ================================================================
#   C L U S T E R A N A L Y S E   F Ü R   L J - T R A J E K T O R I E N
# ================================================================
#
# Dieses Skript:
#
# 1. liest eine XYZ-Trajektorie ein,
# 2. bestimmt für jeden Frame räumliche Cluster,
# 3. berücksichtigt periodische Randbedingungen,
# 4. ordnet jedem Teilchen eine Cluster-ID zu,
# 5. schreibt eine CSV mit allen Cluster-IDs,
# 6. schreibt zusätzlich eine Zusammenfassung pro Frame.
#
# Definition:
# Zwei Teilchen gelten als verbunden, wenn ihr Abstand kleiner
# als CLUSTER_CUTOFF_NM ist.
#
# Alle über solche Verbindungen zusammenhängenden Teilchen bilden
# einen gemeinsamen Cluster.
#
# ================================================================


from pathlib import Path
import csv

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# ================================================================
#   P A R A M E T E R
# ================================================================

# Pfad zur XYZ-Trajektorie
XYZ_PATH = Path(
    "results/2026-07-09_10-50-24/my_simulation_pos.xyz"
)

# Boxlänge aus der Simulation in nm
BOX_LENGTH_NM = 5.0

# Lennard-Jones-sigma in nm
SIGMA_NM = 0.34

# Clusterabstand in nm
#
# Als erster sinnvoller Testwert:
#
# r_cut = 1.5 * sigma
#
# Für sigma = 0.34 nm:
#
# r_cut = 0.51 nm = 5.1 Å
#
CLUSTER_CUTOFF_NM = 1.5 * SIGMA_NM

# Minimale Anzahl von Teilchen, damit eine zusammenhängende Gruppe
# als Cluster bezeichnet wird.
#
# Bei MIN_CLUSTER_SIZE = 3:
# - Einzelteilchen bekommen cluster_id = -1
# - Teilchenpaare bekommen cluster_id = -1
# - Gruppen ab 3 Teilchen erhalten eine Cluster-ID
MIN_CLUSTER_SIZE = 3

# Zeit zwischen zwei gespeicherten XYZ-Frames in ps.
#
# Da dein Programm jeden Simulationsschritt speichert, entspricht
# das normalerweise deinem dt.
TIME_STEP_PS = 0.001

# Nur jeden n-ten Frame analysieren.
#
# Für die spätere Visualisierung sollte dieser Wert am besten 1 sein,
# damit für jeden darstellbaren Frame Cluster-IDs vorhanden sind.
ANALYSIS_FRAME_STRIDE = 1

# Optional eigene Ausgabepfade angeben.
#
# Bei None werden die Dateien neben der XYZ-Datei gespeichert.
CLUSTER_CSV_PATH = None
SUMMARY_CSV_PATH = None


# ================================================================
#   X Y Z - D A T E I   E I N L E S E N
# ================================================================

def read_xyz_trajectory(filename):
    """
    Liest eine XYZ-Trajektorie vollständig ein.

    Erwartetes Format pro Frame
    ---------------------------
    N
    Kommentarzeile
    Atom x y z
    Atom x y z
    ...

    Parameters
    ----------
    filename : str oder Path
        Pfad zur XYZ-Datei.

    Returns
    -------
    positions : np.ndarray
        Form:
            (n_frames, n_particles, 3)

        Die Koordinaten bleiben in der Einheit, in der sie in der
        XYZ-Datei gespeichert sind.

        Bei deiner Trajektorie ist diese Einheit Ångström.

    atom_names : list[str]
        Elementsymbole der Teilchen.

    Raises
    ------
    FileNotFoundError
        Wenn die XYZ-Datei nicht existiert.

    ValueError
        Wenn das XYZ-Format inkonsistent ist.
    """

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(
            f"XYZ-Datei wurde nicht gefunden:\n{filename.resolve()}"
        )

    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    if len(lines) < 3:
        raise ValueError(
            "Die XYZ-Datei enthält nicht genügend Zeilen."
        )

    try:
        n_particles = int(lines[0].strip())
    except ValueError as error:
        raise ValueError(
            "Die erste Zeile der XYZ-Datei muss die Teilchenzahl enthalten."
        ) from error

    if n_particles <= 0:
        raise ValueError(
            "Die Teilchenzahl muss größer als null sein."
        )

    lines_per_frame = n_particles + 2

    if len(lines) % lines_per_frame != 0:
        raise ValueError(
            "Die Anzahl der Zeilen passt nicht zum erwarteten XYZ-Format.\n"
            f"Teilchenzahl: {n_particles}\n"
            f"Zeilen pro Frame: {lines_per_frame}\n"
            f"Gesamtzahl der Zeilen: {len(lines)}"
        )

    n_frames = len(lines) // lines_per_frame

    positions = np.zeros(
        (n_frames, n_particles, 3),
        dtype=float
    )

    atom_names = []

    for frame_index in range(n_frames):

        frame_start = frame_index * lines_per_frame

        # Erste Zeile des Frames nochmals überprüfen
        frame_particle_count = int(
            lines[frame_start].strip()
        )

        if frame_particle_count != n_particles:
            raise ValueError(
                f"Frame {frame_index} enthält laut XYZ-Datei "
                f"{frame_particle_count} Teilchen, erwartet wurden "
                f"{n_particles}."
            )

        for particle_index in range(n_particles):

            line_index = frame_start + 2 + particle_index
            columns = lines[line_index].split()

            if len(columns) < 4:
                raise ValueError(
                    f"Ungültige Atomzeile in Frame {frame_index}, "
                    f"Teilchen {particle_index}:\n"
                    f"{lines[line_index]}"
                )

            atom_name = columns[0]

            try:
                x = float(columns[1])
                y = float(columns[2])
                z = float(columns[3])
            except ValueError as error:
                raise ValueError(
                    f"Ungültige Koordinaten in Frame {frame_index}, "
                    f"Teilchen {particle_index}:\n"
                    f"{lines[line_index]}"
                ) from error

            positions[
                frame_index,
                particle_index,
                :
            ] = [x, y, z]

            if frame_index == 0:
                atom_names.append(atom_name)

    print("XYZ-Datei erfolgreich eingelesen.")
    print(f"Frames: {n_frames}")
    print(f"Teilchen pro Frame: {n_particles}")
    print(f"Positionsarray: {positions.shape}")

    return positions, atom_names


# ================================================================
#   C L U S T E R   I N   E I N E M   F R A M E
# ================================================================

def identify_clusters_in_frame(
    positions,
    box_length,
    cutoff,
    min_cluster_size=3
):
    """
    Bestimmt räumliche Cluster in einem einzelnen Frame.

    Die Cluster werden als zusammenhängende Komponenten eines
    Nachbarschaftsgraphen definiert.

    Zwei Teilchen i und j sind Nachbarn, wenn gilt:

        r_ij < cutoff

    Periodische Randbedingungen werden über scipy.spatial.cKDTree
    mit boxsize berücksichtigt.

    Parameters
    ----------
    positions : np.ndarray
        Form:
            (n_particles, 3)

    box_length : float
        Boxlänge in derselben Einheit wie positions.

    cutoff : float
        Maximaler Nachbarschaftsabstand in derselben Einheit
        wie positions.

    min_cluster_size : int
        Minimale Anzahl von Teilchen pro Cluster.

    Returns
    -------
    cluster_ids : np.ndarray
        Cluster-ID jedes Teilchens.

        Bedeutungen:
            0  = größter Cluster
            1  = zweitgrößter Cluster
            2  = drittgrößter Cluster
            -1 = keinem ausreichend großen Cluster zugeordnet

    cluster_sizes_per_particle : np.ndarray
        Größe des Clusters jedes Teilchens.

        Für Teilchen mit cluster_id = -1 wird 0 gespeichert.

    neighbor_counts : np.ndarray
        Zahl direkter Nachbarn innerhalb des Cutoffs pro Teilchen.

    cluster_sizes : np.ndarray
        Größen aller gültigen Cluster.
        Die Reihenfolge entspricht den Cluster-IDs.

    adjacency_graph
        Sparse-Nachbarschaftsmatrix.
    """

    positions = np.asarray(
        positions,
        dtype=float
    )

    if positions.ndim != 2:
        raise ValueError(
            "positions muss ein zweidimensionales Array sein."
        )

    if positions.shape[1] != 3:
        raise ValueError(
            "positions muss die Form (n_particles, 3) besitzen."
        )

    if box_length <= 0:
        raise ValueError(
            "box_length muss größer als null sein."
        )

    if cutoff <= 0:
        raise ValueError(
            "cutoff muss größer als null sein."
        )

    if cutoff >= box_length / 2:
        print(
            "WARNUNG: cutoff ist mindestens halb so groß wie die Box.\n"
            "Dadurch können Cluster über sehr große Distanzen verbunden werden."
        )

    if min_cluster_size < 1:
        raise ValueError(
            "min_cluster_size muss mindestens 1 sein."
        )

    n_particles = positions.shape[0]

    # ------------------------------------------------------------
    # Positionen in die periodische Box zurückfalten
    # ------------------------------------------------------------
    #
    # cKDTree mit boxsize erwartet:
    #
    #     0 <= x < box_length
    #
    wrapped_positions = np.mod(
        positions,
        box_length
    )

    # ------------------------------------------------------------
    # Periodischer KD-Tree
    # ------------------------------------------------------------

    tree = cKDTree(
        wrapped_positions,
        boxsize=box_length
    )

    # Alle Teilchenpaare mit Abstand <= cutoff finden.
    #
    # pairs hat die Form:
    #
    #     (n_pairs, 2)
    #
    pairs = tree.query_pairs(
        r=cutoff,
        output_type="ndarray"
    )

    # ------------------------------------------------------------
    # Nachbarschaftsgraph erstellen
    # ------------------------------------------------------------

    if pairs.size == 0:

        # Keine Nachbarschaften gefunden
        adjacency_graph = coo_matrix(
            (n_particles, n_particles),
            dtype=np.int8
        ).tocsr()

    else:

        particle_i = pairs[:, 0]
        particle_j = pairs[:, 1]

        # Für einen ungerichteten Graphen müssen beide Richtungen
        # eingetragen werden:
        #
        # i -> j
        # j -> i
        #
        rows = np.concatenate([
            particle_i,
            particle_j
        ])

        columns = np.concatenate([
            particle_j,
            particle_i
        ])

        data = np.ones(
            len(rows),
            dtype=np.int8
        )

        adjacency_graph = coo_matrix(
            (
                data,
                (rows, columns)
            ),
            shape=(n_particles, n_particles)
        ).tocsr()

    # Zahl direkter Nachbarn pro Teilchen
    neighbor_counts = np.diff(
        adjacency_graph.indptr
    )

    # ------------------------------------------------------------
    # Zusammenhängende Komponenten bestimmen
    # ------------------------------------------------------------
    #
    # Jede zusammenhängende Komponente entspricht zunächst einer
    # Teilchengruppe. Dazu gehören auch einzelne Teilchen.
    #
    n_components, raw_component_labels = connected_components(
        adjacency_graph,
        directed=False,
        return_labels=True
    )

    raw_component_sizes = np.bincount(
        raw_component_labels,
        minlength=n_components
    )

    # Nur Komponenten berücksichtigen, die mindestens
    # min_cluster_size Teilchen besitzen.
    valid_components = [
        component_id
        for component_id in range(n_components)
        if raw_component_sizes[component_id] >= min_cluster_size
    ]

    # ------------------------------------------------------------
    # Cluster nach Größe sortieren
    # ------------------------------------------------------------
    #
    # Dadurch gilt für jeden Frame:
    #
    # Cluster 0 = größter Cluster
    # Cluster 1 = zweitgrößter Cluster
    # usw.
    #
    # Bei gleicher Größe entscheidet die kleinste Teilchennummer,
    # damit die Sortierung reproduzierbar ist.
    #
    def component_sort_key(component_id):

        members = np.where(
            raw_component_labels == component_id
        )[0]

        component_size = raw_component_sizes[
            component_id
        ]

        smallest_particle_index = int(
            np.min(members)
        )

        return (
            -component_size,
            smallest_particle_index
        )

    valid_components.sort(
        key=component_sort_key
    )

    # Standardwert -1:
    #
    # Das Teilchen gehört keinem ausreichend großen Cluster an.
    cluster_ids = np.full(
        n_particles,
        -1,
        dtype=int
    )

    cluster_sizes_per_particle = np.zeros(
        n_particles,
        dtype=int
    )

    cluster_sizes = []

    # Neue Cluster-IDs vergeben
    for new_cluster_id, component_id in enumerate(
        valid_components
    ):

        component_members = (
            raw_component_labels == component_id
        )

        component_size = int(
            np.sum(component_members)
        )

        cluster_ids[
            component_members
        ] = new_cluster_id

        cluster_sizes_per_particle[
            component_members
        ] = component_size

        cluster_sizes.append(
            component_size
        )

    cluster_sizes = np.asarray(
        cluster_sizes,
        dtype=int
    )

    return (
        cluster_ids,
        cluster_sizes_per_particle,
        neighbor_counts,
        cluster_sizes,
        adjacency_graph
    )


# ================================================================
#   G E S A M T E   T R A J E K T O R I E   A N A L Y S I E R E N
# ================================================================

def analyze_xyz_trajectory_clusters(
    xyz_path,
    box_length_nm,
    cluster_cutoff_nm,
    minimum_cluster_size,
    time_step_ps,
    frame_stride=1,
    cluster_csv_path=None,
    summary_csv_path=None
):
    """
    Analysiert die vollständige XYZ-Trajektorie.

    Es werden zwei CSV-Dateien geschrieben.

    1. Cluster-ID-Datei
       Eine Zeile pro Teilchen und analysiertem Frame.

    2. Zusammenfassung
       Eine Zeile pro analysiertem Frame.

    Parameters
    ----------
    xyz_path : str oder Path
        Pfad zur XYZ-Datei.

    box_length_nm : float
        Boxlänge in nm.

    cluster_cutoff_nm : float
        Clusterabstand in nm.

    minimum_cluster_size : int
        Minimale Teilchenzahl pro Cluster.

    time_step_ps : float
        Zeit zwischen zwei ursprünglichen XYZ-Frames in ps.

    frame_stride : int
        Nur jeden n-ten Frame analysieren.

    cluster_csv_path : str, Path oder None
        Ausgabepfad für die Cluster-ID-Datei.

    summary_csv_path : str, Path oder None
        Ausgabepfad für die Zusammenfassung.

    Returns
    -------
    cluster_csv_path : Path

    summary_csv_path : Path
    """

    xyz_path = Path(xyz_path)

    if frame_stride < 1:
        raise ValueError(
            "frame_stride muss mindestens 1 sein."
        )

    positions, atom_names = read_xyz_trajectory(
        xyz_path
    )

    n_frames = positions.shape[0]
    n_particles = positions.shape[1]

    # ------------------------------------------------------------
    # Einheiten umrechnen
    # ------------------------------------------------------------
    #
    # Die XYZ-Trajektorie aus deinem Simulationscode ist in Å.
    #
    # Boxlänge und Clusterabstand werden oben in nm eingegeben und
    # deshalb hier nach Å umgerechnet.
    #
    box_length_angstrom = box_length_nm * 10.0
    cluster_cutoff_angstrom = cluster_cutoff_nm * 10.0

    # ------------------------------------------------------------
    # Ausgabepfade festlegen
    # ------------------------------------------------------------

    if cluster_csv_path is None:
        cluster_csv_path = xyz_path.with_name(
            f"{xyz_path.stem}_cluster_ids.csv"
        )
    else:
        cluster_csv_path = Path(
            cluster_csv_path
        )

    if summary_csv_path is None:
        summary_csv_path = xyz_path.with_name(
            f"{xyz_path.stem}_cluster_summary.csv"
        )
    else:
        summary_csv_path = Path(
            summary_csv_path
        )

    cluster_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Cluster-ID-CSV öffnen
    # ------------------------------------------------------------

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
        "z_angstrom"
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
        "isolated_or_small_group_particles"
    ]

    summary_rows = []

    with open(
        cluster_csv_path,
        "w",
        newline="",
        encoding="utf-8"
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
            frame_stride
        ):

            frame_positions = positions[
                frame_index
            ]

            (
                cluster_ids,
                cluster_sizes_per_particle,
                neighbor_counts,
                cluster_sizes,
                adjacency_graph
            ) = identify_clusters_in_frame(
                positions=frame_positions,
                box_length=box_length_angstrom,
                cutoff=cluster_cutoff_angstrom,
                min_cluster_size=minimum_cluster_size
            )

            time_ps = frame_index * time_step_ps

            number_of_clusters = len(
                cluster_sizes
            )

            if number_of_clusters > 0:

                largest_cluster_size = int(
                    cluster_sizes[0]
                )

                mean_cluster_size = float(
                    np.mean(cluster_sizes)
                )

                weighted_mean_cluster_size = float(
                    np.sum(cluster_sizes**2)
                    / np.sum(cluster_sizes)
                )

            else:

                largest_cluster_size = 0
                mean_cluster_size = 0.0
                weighted_mean_cluster_size = 0.0

            clustered_particles = int(
                np.sum(cluster_ids >= 0)
            )

            isolated_or_small_group_particles = (
                n_particles - clustered_particles
            )

            largest_cluster_fraction = (
                largest_cluster_size
                / n_particles
            )

            clustered_fraction = (
                clustered_particles
                / n_particles
            )

            # ----------------------------------------------------
            # Teilchenweise Daten schreiben
            # ----------------------------------------------------

            for particle_index in range(
                n_particles
            ):

                cluster_id = int(
                    cluster_ids[particle_index]
                )

                cluster_size = int(
                    cluster_sizes_per_particle[
                        particle_index
                    ]
                )

                x, y, z = frame_positions[
                    particle_index
                ]

                cluster_writer.writerow([
                    frame_index,
                    time_ps,
                    particle_index,
                    atom_names[particle_index],
                    cluster_id,
                    cluster_size,
                    cluster_id >= 0,
                    cluster_id == 0,
                    int(neighbor_counts[particle_index]),
                    x,
                    y,
                    z
                ])

            # ----------------------------------------------------
            # Frame-Zusammenfassung speichern
            # ----------------------------------------------------

            summary_rows.append([
                frame_index,
                time_ps,
                number_of_clusters,
                largest_cluster_size,
                largest_cluster_fraction,
                clustered_particles,
                clustered_fraction,
                mean_cluster_size,
                weighted_mean_cluster_size,
                isolated_or_small_group_particles
            ])

            analyzed_frame_counter += 1

            print(
                f"Frame {frame_index:>6} | "
                f"Cluster: {number_of_clusters:>4} | "
                f"größter Cluster: {largest_cluster_size:>5} | "
                f"Anteil: {largest_cluster_fraction:>7.3f}"
            )

    # ------------------------------------------------------------
    # Zusammenfassungs-CSV schreiben
    # ------------------------------------------------------------

    with open(
        summary_csv_path,
        "w",
        newline="",
        encoding="utf-8"
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

    print()
    print("=" * 70)
    print("Clusteranalyse abgeschlossen.")
    print("=" * 70)
    print(f"Analysierte Frames: {analyzed_frame_counter}")
    print(f"Teilchen pro Frame: {n_particles}")
    print(f"Cluster-Cutoff: {cluster_cutoff_nm:.6f} nm")
    print(f"Minimale Clustergröße: {minimum_cluster_size}")
    print()
    print("Cluster-IDs:")
    print(cluster_csv_path.resolve())
    print()
    print("Cluster-Zusammenfassung:")
    print(summary_csv_path.resolve())
    print("=" * 70)

    return (
        cluster_csv_path,
        summary_csv_path
    )


# ================================================================
#   P R O G R A M M S T A R T
# ================================================================

if __name__ == "__main__":

    analyze_xyz_trajectory_clusters(
        xyz_path=XYZ_PATH,
        box_length_nm=BOX_LENGTH_NM,
        cluster_cutoff_nm=CLUSTER_CUTOFF_NM,
        minimum_cluster_size=MIN_CLUSTER_SIZE,
        time_step_ps=TIME_STEP_PS,
        frame_stride=ANALYSIS_FRAME_STRIDE,
        cluster_csv_path=CLUSTER_CSV_PATH,
        summary_csv_path=SUMMARY_CSV_PATH
    )