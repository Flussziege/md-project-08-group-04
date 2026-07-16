from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from cluster_functions import read_xyz_trajectory


# ================================================================
#   E I N G A B E N
# ================================================================

XYZ_PATH = Path(
    "results/Long_sim/my_simulation_pos.xyz"
)

# Boxlänge der Simulation in nm
BOX_LENGTH_NM = 6.0

# Lennard-Jones-Parameter in nm
SIGMA_NM = 0.34

# Ersten Teil der Simulation als Equilibrierungsphase verwerfen.
#
# Beispiel:
# START_FRAME = 5000
#
# Bei 0 werden alle Frames verwendet.
START_FRAME = 4000

# Optionales Ende.
#
# None bedeutet: bis zum letzten Frame.
STOP_FRAME = None

# Nur jeden n-ten Frame verwenden.
#
# Bei langen Trajektorien beispielsweise 10, 20, 50 oder 100.
FRAME_STRIDE = 20

# Anzahl der Abstandsintervalle
N_BINS = 200

# Maximaler Abstand.
#
# None bedeutet automatisch:
#
# r_max = L / 2
#
# Größere Werte sollten bei einer kubischen periodischen Box nicht
# verwendet werden.
R_MAX_NM = None

# Deine XYZ-Datei enthält Å-Koordinaten.
# 1 Å = 0.1 nm
XYZ_COORDINATES_IN_ANGSTROM = True

# Ausgabe speichern
SAVE_CSV = True
SAVE_PLOT = True

CSV_PATH = XYZ_PATH.with_name(
    f"{XYZ_PATH.stem}_rdf.csv"
)

PLOT_PATH = XYZ_PATH.with_name(
    f"{XYZ_PATH.stem}_rdf.png"
)


# ================================================================
#   R D F   B E R E C H N E N
# ================================================================

def calculate_rdf(
    positions,
    box_length,
    n_bins=200,
    r_max=None,
):
    """
    Berechnet die radiale Verteilungsfunktion g(r).

    Parameters
    ----------
    positions : np.ndarray
        Form:
            (n_frames, n_particles, 3)

        Positionen und box_length müssen dieselbe Einheit besitzen.

    box_length : float
        Länge der kubischen Simulationsbox.

    n_bins : int
        Anzahl der Abstandsintervalle.

    r_max : float oder None
        Maximaler betrachteter Abstand.

        Bei None wird box_length / 2 verwendet.

    Returns
    -------
    r_centers : np.ndarray
        Mittelpunkte der Abstandsintervalle.

    g_r : np.ndarray
        Radiale Verteilungsfunktion.

    coordination_number : np.ndarray
        Mittlere Zahl von Nachbarn innerhalb des Radius r.

    pair_histogram : np.ndarray
        Über alle Frames gezählte Paarabstände.
    """

    positions = np.asarray(
        positions,
        dtype=float
    )

    if positions.ndim != 3:
        raise ValueError(
            "positions muss die Form "
            "(n_frames, n_particles, 3) besitzen."
        )

    if positions.shape[2] != 3:
        raise ValueError(
            "Die letzte Dimension von positions muss 3 sein."
        )

    if box_length <= 0:
        raise ValueError(
            "box_length muss größer als null sein."
        )

    if n_bins < 1:
        raise ValueError(
            "n_bins muss mindestens 1 sein."
        )

    n_frames, n_particles, _ = positions.shape

    if n_frames == 0:
        raise ValueError(
            "Es wurden keine Frames übergeben."
        )

    if n_particles < 2:
        raise ValueError(
            "Für eine RDF werden mindestens zwei Teilchen benötigt."
        )

    if r_max is None:
        r_max = box_length / 2.0

    if r_max <= 0:
        raise ValueError(
            "r_max muss größer als null sein."
        )

    if r_max > box_length / 2.0:
        raise ValueError(
            "Für eine kubische periodische Box sollte "
            "r_max nicht größer als L/2 sein."
        )

    # ------------------------------------------------------------
    # Histogrammgrenzen
    # ------------------------------------------------------------

    bin_edges = np.linspace(
        0.0,
        r_max,
        n_bins + 1
    )

    r_centers = 0.5 * (
        bin_edges[:-1]
        + bin_edges[1:]
    )

    pair_histogram = np.zeros(
        n_bins,
        dtype=np.int64
    )

    # ------------------------------------------------------------
    # Jeden ausgewählten Frame analysieren
    # ------------------------------------------------------------

    for frame_index, frame_positions in enumerate(
        positions
    ):

        # Positionen nach [0, L) zurückfalten
        wrapped_positions = np.mod(
            frame_positions,
            box_length
        )

        # Periodischer KD-Tree
        tree = cKDTree(
            wrapped_positions,
            boxsize=box_length
        )

        # Nur Teilchenpaare bis r_max suchen.
        #
        # query_pairs liefert jedes Paar nur einmal:
        # i < j
        pairs = tree.query_pairs(
            r=r_max,
            output_type="ndarray"
        )

        if len(pairs) > 0:

            displacement_vectors = (
                wrapped_positions[pairs[:, 0]]
                - wrapped_positions[pairs[:, 1]]
            )

            # Minimum-Image-Konvention
            displacement_vectors -= (
                box_length
                * np.rint(
                    displacement_vectors
                    / box_length
                )
            )

            distances = np.linalg.norm(
                displacement_vectors,
                axis=1
            )

            frame_histogram, _ = np.histogram(
                distances,
                bins=bin_edges
            )

            pair_histogram += frame_histogram

        if (
            frame_index % 100 == 0
            or frame_index == n_frames - 1
        ):
            print(
                f"RDF: Frame {frame_index + 1} "
                f"von {n_frames}"
            )

    # ------------------------------------------------------------
    # Normierung
    # ------------------------------------------------------------

    box_volume = box_length**3

    shell_volumes = (
        4.0
        * np.pi
        / 3.0
        * (
            bin_edges[1:]**3
            - bin_edges[:-1]**3
        )
    )

    # Zahl aller eindeutigen Teilchenpaare
    total_number_of_pairs = (
        n_particles
        * (n_particles - 1)
        / 2.0
    )

    # Erwartete Paarzahl pro Frame für eine ideale gleichmäßige
    # Verteilung
    expected_pairs_per_frame = (
        total_number_of_pairs
        * shell_volumes
        / box_volume
    )

    # Erwartete Paarzahl über alle verwendeten Frames
    expected_pairs_all_frames = (
        n_frames
        * expected_pairs_per_frame
    )

    g_r = np.divide(
        pair_histogram,
        expected_pairs_all_frames,
        out=np.zeros_like(
            expected_pairs_all_frames,
            dtype=float
        ),
        where=expected_pairs_all_frames > 0
    )

    # ------------------------------------------------------------
    # Koordinationszahl
    # ------------------------------------------------------------
    #
    # N_coord(r)
    # =
    # rho * Integral_0^r g(r') 4 pi r'^2 dr'
    #
    # Für ein endliches System verwenden wir hier (N-1)/V,
    # weil ein Referenzteilchen nur N-1 mögliche Nachbarn besitzt.
    #
    number_density_of_other_particles = (
        (n_particles - 1)
        / box_volume
    )

    coordination_number = np.cumsum(
        number_density_of_other_particles
        * g_r
        * shell_volumes
    )

    return (
        r_centers,
        g_r,
        coordination_number,
        pair_histogram
    )


# ================================================================
#   X Y Z   E I N L E S E N
# ================================================================

positions, atom_names = read_xyz_trajectory(
    XYZ_PATH
)

print(f"Gesamte Trajektorie: {positions.shape}")

# XYZ: Å nach nm
if XYZ_COORDINATES_IN_ANGSTROM:
    positions = positions * 0.1

# Nur ausgewählte Frames verwenden
selected_positions = positions[
    START_FRAME:STOP_FRAME:FRAME_STRIDE
]

print(
    f"Für RDF verwendete Frames: "
    f"{selected_positions.shape[0]}"
)

print(
    f"Teilchen pro Frame: "
    f"{selected_positions.shape[1]}"
)


# ================================================================
#   R D F   B E R E C H N E N
# ================================================================

r_nm, g_r, coordination_number, counts = calculate_rdf(
    positions=selected_positions,
    box_length=BOX_LENGTH_NM,
    n_bins=N_BINS,
    r_max=R_MAX_NM
)


# ================================================================
#   C S V   S P E I C H E R N
# ================================================================

if SAVE_CSV:

    output_data = np.column_stack(
        (
            r_nm,
            g_r,
            coordination_number,
            counts
        )
    )

    np.savetxt(
        CSV_PATH,
        output_data,
        delimiter=",",
        header=(
            "r_nm,"
            "g_r,"
            "coordination_number,"
            "pair_count"
        ),
        comments=""
    )

    print(
        f"RDF-CSV gespeichert unter: "
        f"{CSV_PATH}"
    )


# ================================================================
#   R D F   P L O T T E N
# ================================================================

plt.figure(figsize=(9, 6))

plt.plot(
    r_nm,
    g_r,
    linewidth=1.5,
    label=r"$g(r)$"
)

# Für große Abstände sollte g(r) bei einem homogenen System
# ungefähr gegen 1 gehen.
plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1,
    label="ideale gleichmäßige Verteilung"
)

# Minimum des Lennard-Jones-Potentials
#
# r_min = 2^(1/6) sigma
lj_minimum_nm = (
    2.0**(1.0 / 6.0)
    * SIGMA_NM
)

plt.axvline(
    lj_minimum_nm,
    linestyle=":",
    linewidth=1.2,
    label=(
        r"$r_\mathrm{min}=2^{1/6}\sigma$"
    )
)

plt.xlabel("Abstand r / nm")
plt.ylabel(r"radiale Verteilungsfunktion $g(r)$")

plt.title(
    "Radiale Verteilungsfunktion "
    "der Lennard-Jones-Trajektorie"
)

plt.grid(True)
plt.legend()
plt.tight_layout()

if SAVE_PLOT:
    plt.savefig(
        PLOT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"RDF-Plot gespeichert unter: "
        f"{PLOT_PATH}"
    )

plt.show()