from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from cluster_functions import read_xyz_trajectory


# ================================================================
#   E I N G A B E N
# ================================================================

XYZ_PATH = Path(
    r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04\results\2026-07-16_18-51-01\my_simulation_pos-300K.xyz"
)

# Boxlänge der kubischen Simulationsbox in nm
BOX_LENGTH_NM = 6.0

# Lennard-Jones-Parameter in nm
SIGMA_NM = 0.34

# 3: Winkel zwischen räumlichen Bindungsvektoren
# 2: Nur x- und y-Komponenten verwenden
DIMENSION = 3

# Ersten Teil der Simulation als Equilibrierungsphase verwerfen.
START_FRAME = 4000

# None bedeutet: bis zum letzten Frame.
STOP_FRAME = None

# Nur jeden n-ten Frame verwenden.
FRAME_STRIDE = 20

# Anzahl der Winkelintervalle zwischen 0° und 180°.
N_ANGLE_BINS = 180

# Nachbarn werden nur innerhalb dieses Radius betrachtet.
#
# None:
#   Der Radius wird automatisch als erstes Minimum der RDF nach
#   dem ersten Maximum gewählt.
#
# Beispiel für einen festen Wert:
#   NEIGHBOR_CUTOFF_NM = 0.52
NEIGHBOR_CUTOFF_NM = None

# Parameter für die RDF, falls der Nachbarradius automatisch
# bestimmt werden soll.
N_RDF_BINS = 300
RDF_R_MAX_NM = None
RDF_SMOOTHING_SIGMA_BINS = 2.0

# Deine XYZ-Datei enthält Å-Koordinaten.
# 1 Å = 0.1 nm
XYZ_COORDINATES_IN_ANGSTROM = True

# Ausgabe speichern
SAVE_CSV = True
SAVE_PLOT = True
SAVE_RDF_DIAGNOSTIC = True

ANGLE_CSV_PATH = XYZ_PATH.with_name(
    f"{XYZ_PATH.stem}_relative_angle_probability.csv"
)

ANGLE_PLOT_PATH = XYZ_PATH.with_name(
    f"{XYZ_PATH.stem}_relative_angle_probability.png"
)

ANGULAR_CORRELATION_PLOT_PATH = XYZ_PATH.with_name(
    f"{XYZ_PATH.stem}_relative_angle_correlation.png"
)

RDF_DIAGNOSTIC_PATH = XYZ_PATH.with_name(
    f"{XYZ_PATH.stem}_rdf_neighbor_cutoff.png"
)


# ================================================================
#   H I L F S F U N K T I O N E N
# ================================================================


def validate_positions(positions):
    """Prüft die Form der Trajektorie."""
    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 3:
        raise ValueError(
            "positions muss die Form "
            "(n_frames, n_particles, 3) besitzen."
        )

    if positions.shape[2] != 3:
        raise ValueError(
            "Die letzte Dimension von positions muss 3 sein."
        )

    if positions.shape[0] == 0:
        raise ValueError("Es wurden keine Frames übergeben.")

    if positions.shape[1] < 3:
        raise ValueError(
            "Für relative Winkel werden mindestens drei Teilchen benötigt."
        )

    return positions


def minimum_image(displacements, box_length):
    """Wendet die Minimum-Image-Konvention an."""
    return displacements - box_length * np.rint(
        displacements / box_length
    )


# ================================================================
#   R D F   F Ü R   N A C H B A R R A D I U S
# ================================================================


def calculate_rdf(
    positions,
    box_length,
    n_bins=300,
    r_max=None,
):
    """
    Berechnet die radiale Verteilungsfunktion g(r).

    Sie wird hier hauptsächlich verwendet, um das erste Minimum
    nach dem ersten Peak und damit die erste Nachbarschale zu finden.
    """
    positions = validate_positions(positions)

    if box_length <= 0:
        raise ValueError("box_length muss größer als null sein.")

    if n_bins < 10:
        raise ValueError("n_bins sollte mindestens 10 sein.")

    n_frames, n_particles, _ = positions.shape

    if r_max is None:
        r_max = box_length / 2.0

    if not 0 < r_max <= box_length / 2.0:
        raise ValueError(
            "r_max muss größer als null und höchstens L/2 sein."
        )

    bin_edges = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pair_histogram = np.zeros(n_bins, dtype=np.int64)

    for frame_index, frame_positions in enumerate(positions):
        wrapped_positions = np.mod(frame_positions, box_length)
        tree = cKDTree(wrapped_positions, boxsize=box_length)

        pairs = tree.query_pairs(
            r=r_max,
            output_type="ndarray",
        )

        if pairs.size > 0:
            displacements = (
                wrapped_positions[pairs[:, 1]]
                - wrapped_positions[pairs[:, 0]]
            )
            displacements = minimum_image(
                displacements,
                box_length,
            )
            distances = np.linalg.norm(displacements, axis=1)
            frame_histogram, _ = np.histogram(
                distances,
                bins=bin_edges,
            )
            pair_histogram += frame_histogram

        if frame_index % 100 == 0 or frame_index == n_frames - 1:
            print(
                f"RDF: Frame {frame_index + 1} von {n_frames}"
            )

    box_volume = box_length**3
    shell_volumes = (
        4.0
        * np.pi
        / 3.0
        * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
    )

    total_number_of_pairs = n_particles * (n_particles - 1) / 2.0
    expected_pairs_all_frames = (
        n_frames
        * total_number_of_pairs
        * shell_volumes
        / box_volume
    )

    g_r = np.divide(
        pair_histogram,
        expected_pairs_all_frames,
        out=np.zeros_like(expected_pairs_all_frames, dtype=float),
        where=expected_pairs_all_frames > 0,
    )

    return r_centers, g_r, pair_histogram


def determine_first_shell_cutoff(
    r_nm,
    g_r,
    sigma_nm,
    smoothing_sigma_bins=2.0,
):
    """
    Bestimmt das erste RDF-Minimum nach dem ersten RDF-Maximum.

    Die Suche wird auf den physikalisch relevanten Bereich um die
    erste Lennard-Jones-Nachbarschale beschränkt.
    """
    r_nm = np.asarray(r_nm, dtype=float)
    g_r = np.asarray(g_r, dtype=float)

    if r_nm.shape != g_r.shape:
        raise ValueError("r_nm und g_r müssen dieselbe Form besitzen.")

    if sigma_nm <= 0:
        raise ValueError("sigma_nm muss größer als null sein.")

    smoothed_g_r = gaussian_filter1d(
        g_r,
        sigma=smoothing_sigma_bins,
        mode="nearest",
    )

    # Erster Peak ungefähr um r_min = 2^(1/6) sigma.
    peak_search_mask = (
        (r_nm >= 0.85 * sigma_nm)
        & (r_nm <= 1.65 * sigma_nm)
    )

    peak_indices = np.flatnonzero(peak_search_mask)
    if peak_indices.size == 0:
        raise RuntimeError(
            "Kein gültiger Suchbereich für den ersten RDF-Peak."
        )

    first_peak_index = peak_indices[
        np.argmax(smoothed_g_r[peak_indices])
    ]

    # Lokale Minima nach dem Peak suchen.
    minimum_candidates, _ = find_peaks(-smoothed_g_r)
    minimum_candidates = minimum_candidates[
        (minimum_candidates > first_peak_index)
        & (r_nm[minimum_candidates] <= 2.5 * sigma_nm)
    ]

    if minimum_candidates.size > 0:
        first_minimum_index = minimum_candidates[0]
    else:
        # Robuster Fallback: kleinstes g(r) in einem Bereich nach dem Peak.
        fallback_mask = (
            (r_nm > r_nm[first_peak_index])
            & (r_nm <= 2.0 * sigma_nm)
        )
        fallback_indices = np.flatnonzero(fallback_mask)

        if fallback_indices.size == 0:
            fallback_cutoff = 1.5 * sigma_nm
            print(
                "WARNUNG: Erstes RDF-Minimum konnte nicht bestimmt "
                f"werden. Verwende {fallback_cutoff:.4f} nm."
            )
            return (
                fallback_cutoff,
                smoothed_g_r,
                first_peak_index,
                None,
            )

        first_minimum_index = fallback_indices[
            np.argmin(smoothed_g_r[fallback_indices])
        ]

    cutoff_nm = float(r_nm[first_minimum_index])

    return (
        cutoff_nm,
        smoothed_g_r,
        first_peak_index,
        first_minimum_index,
    )


# ================================================================
#   R E L A T I V E   W I N K E L V E R T E I L U N G
# ================================================================


def calculate_relative_angle_probability(
    positions,
    box_length,
    neighbor_cutoff,
    n_angle_bins=180,
    dimension=3,
):
    """
    Berechnet die Verteilung der Winkel zwischen zwei Nachbarbindungen.

    Für ein Zentralteilchen i und zwei Nachbarn j und k wird berechnet:

        theta_jik = arccos(
            r_ij dot r_ik / (|r_ij| |r_ik|)
        )

    Es werden alle ungeordneten Nachbarpaare j < k gezählt.

    Parameters
    ----------
    positions : np.ndarray
        Form (n_frames, n_particles, 3).

    box_length : float
        Länge der periodischen kubischen Box.

    neighbor_cutoff : float
        Maximaler Abstand eines Nachbarn vom Zentralteilchen.
        Typischerweise das erste Minimum der RDF.

    n_angle_bins : int
        Zahl der Winkelintervalle von 0° bis 180°.

    dimension : int
        3 für räumliche Winkel, 2 für eine Analyse nur in der xy-Ebene.

    Returns
    -------
    angle_centers_deg : np.ndarray
        Mittelpunkte der Winkelintervalle.

    probability_density_per_degree : np.ndarray
        Normierte Aufenthaltswahrscheinlichkeitsdichte P(theta).
        Das Integral von 0° bis 180° ist ungefähr 1.

    isotropic_density_per_degree : np.ndarray
        Erwartete Referenzdichte für zufällig orientierte Bindungen.

    angular_correlation : np.ndarray
        P(theta) geteilt durch die isotrope Referenz.
        Ein Wert von 1 entspricht einer isotropen Verteilung.

    angle_histogram : np.ndarray
        Absolute Zahl der Winkel pro Intervall.

    mean_coordination_number : float
        Mittlere Zahl von Nachbarn innerhalb des Cutoffs.
    """
    positions = validate_positions(positions)

    if dimension not in (2, 3):
        raise ValueError("dimension muss 2 oder 3 sein.")

    if box_length <= 0:
        raise ValueError("box_length muss größer als null sein.")

    if not 0 < neighbor_cutoff <= box_length / 2.0:
        raise ValueError(
            "neighbor_cutoff muss größer als null und höchstens L/2 sein."
        )

    if n_angle_bins < 1:
        raise ValueError("n_angle_bins muss mindestens 1 sein.")

    n_frames, n_particles, _ = positions.shape

    angle_edges_deg = np.linspace(0.0, 180.0, n_angle_bins + 1)
    angle_centers_deg = 0.5 * (
        angle_edges_deg[:-1] + angle_edges_deg[1:]
    )
    angle_histogram = np.zeros(n_angle_bins, dtype=np.int64)

    coordination_sum = 0
    number_of_central_particles = n_frames * n_particles

    for frame_index, frame_positions in enumerate(positions):
        wrapped_positions = np.mod(frame_positions, box_length)
        tree = cKDTree(wrapped_positions, boxsize=box_length)

        neighbor_lists = tree.query_ball_point(
            wrapped_positions,
            r=neighbor_cutoff,
        )

        for central_index, neighbor_indices in enumerate(neighbor_lists):
            # Der KD-Tree enthält das Zentralteilchen selbst.
            neighbor_indices = np.asarray(
                [
                    index
                    for index in neighbor_indices
                    if index != central_index
                ],
                dtype=int,
            )

            number_of_neighbors = neighbor_indices.size
            coordination_sum += number_of_neighbors

            # Für einen Winkel werden mindestens zwei Nachbarn benötigt.
            if number_of_neighbors < 2:
                continue

            neighbor_vectors = (
                wrapped_positions[neighbor_indices]
                - wrapped_positions[central_index]
            )
            neighbor_vectors = minimum_image(
                neighbor_vectors,
                box_length,
            )

            if dimension == 2:
                neighbor_vectors = neighbor_vectors[:, :2]

            vector_norms = np.linalg.norm(
                neighbor_vectors,
                axis=1,
            )

            valid = vector_norms > 0.0
            neighbor_vectors = neighbor_vectors[valid]
            vector_norms = vector_norms[valid]

            if neighbor_vectors.shape[0] < 2:
                continue

            first_indices, second_indices = np.triu_indices(
                neighbor_vectors.shape[0],
                k=1,
            )

            dot_products = np.einsum(
                "ij,ij->i",
                neighbor_vectors[first_indices],
                neighbor_vectors[second_indices],
            )

            norm_products = (
                vector_norms[first_indices]
                * vector_norms[second_indices]
            )

            cosines = np.divide(
                dot_products,
                norm_products,
                out=np.zeros_like(dot_products),
                where=norm_products > 0,
            )
            cosines = np.clip(cosines, -1.0, 1.0)

            angles_deg = np.degrees(np.arccos(cosines))

            frame_histogram, _ = np.histogram(
                angles_deg,
                bins=angle_edges_deg,
            )
            angle_histogram += frame_histogram

        if frame_index % 100 == 0 or frame_index == n_frames - 1:
            print(
                f"Winkel: Frame {frame_index + 1} von {n_frames}"
            )

    total_angle_count = int(angle_histogram.sum())
    if total_angle_count == 0:
        raise RuntimeError(
            "Es wurden keine Winkel gefunden. "
            "Prüfe den Nachbarradius und die ausgewählten Frames."
        )

    bin_widths_deg = np.diff(angle_edges_deg)
    probability_per_bin = angle_histogram / total_angle_count
    probability_density_per_degree = (
        probability_per_bin / bin_widths_deg
    )

    angle_edges_rad = np.radians(angle_edges_deg)

    if dimension == 3:
        # Für zwei unabhängige, isotrope 3D-Richtungen gilt
        # p(theta) = 1/2 sin(theta), 0 <= theta <= pi.
        # Hier wird die exakte Wahrscheinlichkeit pro Histogrammbin
        # integriert und danach wieder durch die Binbreite geteilt.
        isotropic_probability_per_bin = 0.5 * (
            np.cos(angle_edges_rad[:-1])
            - np.cos(angle_edges_rad[1:])
        )
    else:
        # In 2D ist der gefaltete relative Winkel auf [0, pi]
        # für zufällige Richtungen gleichverteilt.
        isotropic_probability_per_bin = (
            bin_widths_deg / 180.0
        )

    isotropic_density_per_degree = (
        isotropic_probability_per_bin / bin_widths_deg
    )

    angular_correlation = np.divide(
        probability_per_bin,
        isotropic_probability_per_bin,
        out=np.zeros_like(probability_per_bin, dtype=float),
        where=isotropic_probability_per_bin > 0,
    )

    mean_coordination_number = (
        coordination_sum / number_of_central_particles
    )

    return (
        angle_centers_deg,
        probability_density_per_degree,
        isotropic_density_per_degree,
        angular_correlation,
        angle_histogram,
        mean_coordination_number,
    )


# ================================================================
#   X Y Z   E I N L E S E N
# ================================================================

positions, atom_names = read_xyz_trajectory(XYZ_PATH)

print(f"Gesamte Trajektorie: {positions.shape}")

if XYZ_COORDINATES_IN_ANGSTROM:
    positions = positions * 0.1

selected_positions = positions[
    START_FRAME:STOP_FRAME:FRAME_STRIDE
]

if selected_positions.shape[0] == 0:
    raise ValueError(
        "Die Frame-Auswahl ist leer. Prüfe START_FRAME, "
        "STOP_FRAME und FRAME_STRIDE."
    )

print(
    f"Für die Analyse verwendete Frames: "
    f"{selected_positions.shape[0]}"
)
print(
    f"Teilchen pro Frame: "
    f"{selected_positions.shape[1]}"
)


# ================================================================
#   N A C H B A R R A D I U S   B E S T I M M E N
# ================================================================

if NEIGHBOR_CUTOFF_NM is None:
    r_nm, g_r, rdf_counts = calculate_rdf(
        positions=selected_positions,
        box_length=BOX_LENGTH_NM,
        n_bins=N_RDF_BINS,
        r_max=RDF_R_MAX_NM,
    )

    (
        neighbor_cutoff_nm,
        smoothed_g_r,
        first_peak_index,
        first_minimum_index,
    ) = determine_first_shell_cutoff(
        r_nm=r_nm,
        g_r=g_r,
        sigma_nm=SIGMA_NM,
        smoothing_sigma_bins=RDF_SMOOTHING_SIGMA_BINS,
    )

    print(
        "Automatisch bestimmter Nachbarradius "
        f"(erstes RDF-Minimum): {neighbor_cutoff_nm:.5f} nm"
    )

    if SAVE_RDF_DIAGNOSTIC:
        plt.figure(figsize=(9, 6))
        plt.plot(r_nm, g_r, linewidth=1.0, label=r"$g(r)$")
        plt.plot(
            r_nm,
            smoothed_g_r,
            linewidth=1.5,
            label="geglättete RDF",
        )
        plt.axvline(
            r_nm[first_peak_index],
            linestyle=":",
            linewidth=1.2,
            label="erster Peak",
        )
        plt.axvline(
            neighbor_cutoff_nm,
            linestyle="--",
            linewidth=1.2,
            label="Nachbarradius: erstes Minimum",
        )
        plt.xlabel("Abstand r / nm")
        plt.ylabel(r"radiale Verteilungsfunktion $g(r)$")
        plt.title("Bestimmung der ersten Nachbarschale")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            RDF_DIAGNOSTIC_PATH,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(
            "RDF-Diagnose gespeichert unter: "
            f"{RDF_DIAGNOSTIC_PATH}"
        )
else:
    neighbor_cutoff_nm = float(NEIGHBOR_CUTOFF_NM)
    print(
        "Manuell gesetzter Nachbarradius: "
        f"{neighbor_cutoff_nm:.5f} nm"
    )


# ================================================================
#   R E L A T I V E   W I N K E L   B E R E C H N E N
# ================================================================

(
    angle_deg,
    probability_density,
    isotropic_density,
    angular_correlation,
    angle_counts,
    mean_coordination_number,
) = calculate_relative_angle_probability(
    positions=selected_positions,
    box_length=BOX_LENGTH_NM,
    neighbor_cutoff=neighbor_cutoff_nm,
    n_angle_bins=N_ANGLE_BINS,
    dimension=DIMENSION,
)

print(f"Gezählte relative Winkel: {angle_counts.sum()}")
print(
    "Mittlere Koordinationszahl innerhalb des Cutoffs: "
    f"{mean_coordination_number:.3f}"
)


# ================================================================
#   C S V   S P E I C H E R N
# ================================================================

if SAVE_CSV:
    output_data = np.column_stack(
        (
            angle_deg,
            probability_density,
            isotropic_density,
            angular_correlation,
            angle_counts,
        )
    )

    np.savetxt(
        ANGLE_CSV_PATH,
        output_data,
        delimiter=",",
        header=(
            "angle_deg,"
            "probability_density_per_degree,"
            "isotropic_density_per_degree,"
            "angular_correlation,"
            "angle_count"
        ),
        comments="",
    )

    print(
        "Winkel-CSV gespeichert unter: "
        f"{ANGLE_CSV_PATH}"
    )


# ================================================================
#   W I N K E L V E R T E I L U N G   P L O T T E N
# ================================================================

plt.figure(figsize=(9, 6))
plt.plot(
    angle_deg,
    probability_density,
    linewidth=1.5,
    label=r"gemessene $P(\theta)$",
)
plt.plot(
    angle_deg,
    isotropic_density,
    linestyle="--",
    linewidth=1.2,
    label=(
        r"isotrope Referenz $P_0(\theta)$"
        if DIMENSION == 3
        else "gleichverteilte 2D-Referenz"
    ),
)

for reference_angle in (60.0, 90.0, 120.0, 180.0):
    plt.axvline(
        reference_angle,
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
    )

plt.xlabel(r"relativer Winkel $\theta$ / Grad")
plt.ylabel(r"Wahrscheinlichkeitsdichte $P(\theta)$ / Grad$^{-1}$")
plt.title(
    "Relative Winkelverteilung der ersten Nachbarschale\n"
    f"Nachbarradius = {neighbor_cutoff_nm:.4f} nm"
)
plt.xlim(0.0, 180.0)
plt.grid(True)
plt.legend()
plt.tight_layout()

if SAVE_PLOT:
    plt.savefig(
        ANGLE_PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    print(
        "Winkelverteilungs-Plot gespeichert unter: "
        f"{ANGLE_PLOT_PATH}"
    )

plt.show()


# ================================================================
#   I S O T R O P I E - K O R R I G I E R T E   V E R T E I L U N G
# ================================================================

plt.figure(figsize=(9, 6))
plt.plot(
    angle_deg,
    angular_correlation,
    linewidth=1.5,
    label=r"$g_\theta(\theta)=P(\theta)/P_0(\theta)$",
)
plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1.0,
    label="isotrope Referenz",
)

for reference_angle in (60.0, 90.0, 120.0, 180.0):
    plt.axvline(
        reference_angle,
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
    )

plt.xlabel(r"relativer Winkel $\theta$ / Grad")
plt.ylabel(r"Winkelkorrelation $g_\theta(\theta)$")
plt.title(
    "Isotropie-korrigierte relative Winkelverteilung\n"
    f"Nachbarradius = {neighbor_cutoff_nm:.4f} nm"
)
plt.xlim(0.0, 180.0)
plt.grid(True)
plt.legend()
plt.tight_layout()

if SAVE_PLOT:
    plt.savefig(
        ANGULAR_CORRELATION_PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    print(
        "Winkelkorrelations-Plot gespeichert unter: "
        f"{ANGULAR_CORRELATION_PLOT_PATH}"
    )

plt.show()
