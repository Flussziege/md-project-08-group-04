"""
Radial distribution function analysis for an XYZ trajectory
============================================================

PURPOSE
-------
This script calculates the radial distribution function g(r) of a molecular
dynamics trajectory stored in XYZ format.

The radial distribution function describes how likely it is to find another
particle at a distance r from a reference particle relative to an ideal,
uniform particle distribution.

For a homogeneous ideal gas:

    g(r) -> 1

For a liquid, g(r) usually shows:

    - a pronounced first maximum,
    - a clear minimum after the first maximum,
    - additional damped maxima at larger distances.

USAGE
-----
1. Set XYZ_PATH to the XYZ trajectory.
2. Set BOX_LENGTH_NM to the cubic simulation-box length.
3. Set START_FRAME to exclude the equilibration phase.
4. Set FRAME_STRIDE to reduce the number of analyzed frames.
5. Run:

       python rdf_analysis.py

EXPECTED UNITS
--------------
The simulation writes XYZ coordinates in angstrom by default.

    1 angstrom = 0.1 nm

Set XYZ_COORDINATES_IN_ANGSTROM = True in this case.

OUTPUT
------
The script can save:

    - a CSV file containing r, g(r), coordination number, and pair counts,
    - a PNG plot of the radial distribution function.

THEORY
------
For a spherical shell between r_i and r_{i+1}, the shell volume is

    Delta V_i = 4*pi/3 * (r_{i+1}^3 - r_i^3)

The RDF is calculated as

    g(r_i) =
        measured pair count in shell i
        ---------------------------------------------
        expected pair count for a uniform distribution

The coordination number is

    N_coord(r) =
        rho * integral_0^r g(r') 4*pi*r'^2 dr'
"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


# ================================================================
# INPUT PARAMETERS
# ================================================================

XYZ_PATH = (
    Path.home()
    / "Documents"
    / "VSCODE"
    / "moldyn_proj"
    / "md-project"
    / "results"
    / "Long_sim_200_new"
    / "my_simulation_pos.xyz"
)

# Cubic simulation-box length in nm
BOX_LENGTH_NM = 6.0

# Lennard-Jones sigma parameter in nm
SIGMA_NM = 0.34

# First trajectory frame used for the analysis
START_FRAME = 3000

# Last trajectory frame used for the analysis.
# None means: continue to the final frame.
STOP_FRAME = None

# Analyze only every nth frame
FRAME_STRIDE = 20

# Number of distance bins
N_BINS = 200

# Maximum analyzed distance in nm.
# None means BOX_LENGTH_NM / 2.
R_MAX_NM = None

# Set True when the XYZ coordinates are stored in angstrom
XYZ_COORDINATES_IN_ANGSTROM = True

# Save calculated data and figure
SAVE_CSV = True
SAVE_PLOT = True

CSV_PATH = XYZ_PATH.with_name(
    f"{XYZ_PATH.stem}_rdf.csv"
)

PLOT_PATH = XYZ_PATH.with_name(
    f"{XYZ_PATH.stem}_rdf.png"
)


# ================================================================
# PLOT APPEARANCE
# ================================================================

TITLE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 17
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 13

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
    }
)


# ================================================================
# XYZ READER
# ================================================================

def read_xyz_trajectory(filename):
    """
    Read a multi-frame XYZ trajectory.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the XYZ trajectory.

    Returns
    -------
    positions : np.ndarray
        Array with shape:
            (n_frames, n_particles, 3)

    atom_names : list[str]
        Atom labels from the first frame.
    """

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(
            f"XYZ file not found:\n{filename.resolve()}"
        )

    with filename.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    if not lines:
        raise ValueError("The XYZ file is empty.")

    try:
        n_particles = int(lines[0].strip())
    except ValueError as error:
        raise ValueError(
            "The first line of the XYZ file must contain "
            "the number of particles."
        ) from error

    if n_particles < 1:
        raise ValueError(
            "The XYZ file must contain at least one particle."
        )

    lines_per_frame = n_particles + 2

    if len(lines) % lines_per_frame != 0:
        raise ValueError(
            "The XYZ file contains an incomplete frame or does not "
            "use a constant particle number."
        )

    n_frames = len(lines) // lines_per_frame

    positions = np.zeros(
        (n_frames, n_particles, 3),
        dtype=float,
    )

    atom_names = []

    for frame_index in range(n_frames):
        frame_start = frame_index * lines_per_frame

        for particle_index in range(n_particles):
            line_index = frame_start + 2 + particle_index
            columns = lines[line_index].split()

            if len(columns) < 4:
                raise ValueError(
                    f"Invalid XYZ line {line_index + 1}: "
                    "expected atom label and three coordinates."
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
                    f"Invalid coordinates in XYZ line "
                    f"{line_index + 1}."
                ) from error

            positions[
                frame_index,
                particle_index,
                :,
            ] = coordinates

            if frame_index == 0:
                atom_names.append(atom_name)

    return positions, atom_names


# ================================================================
# RDF CALCULATION
# ================================================================

def calculate_rdf(
    positions,
    box_length,
    n_bins=200,
    r_max=None,
    progress_label="RDF",
):
    """
    Calculate the radial distribution function g(r).

    Parameters
    ----------
    positions : np.ndarray
        Particle positions with shape:
            (n_frames, n_particles, 3)

        positions and box_length must use the same length unit.

    box_length : float
        Length of the cubic simulation box.

    n_bins : int, default=200
        Number of radial distance intervals.

    r_max : float or None
        Maximum analyzed distance.

        If None:
            r_max = box_length / 2

    progress_label : str, default="RDF"
        Text shown in progress messages.

    Returns
    -------
    r_centers : np.ndarray
        Centers of the radial bins.

    g_r : np.ndarray
        Radial distribution function.

    coordination_number : np.ndarray
        Mean number of neighbors within radius r.

    pair_histogram : np.ndarray
        Total number of measured particle pairs per bin.
    """

    positions = np.asarray(
        positions,
        dtype=float,
    )

    if positions.ndim != 3:
        raise ValueError(
            "positions must have shape "
            "(n_frames, n_particles, 3)."
        )

    if positions.shape[2] != 3:
        raise ValueError(
            "The final positions dimension must have size 3."
        )

    if not np.all(np.isfinite(positions)):
        raise ValueError(
            "positions contains NaN or infinite values."
        )

    if box_length <= 0:
        raise ValueError(
            "box_length must be greater than zero."
        )

    if n_bins < 1:
        raise ValueError(
            "n_bins must be at least 1."
        )

    n_frames, n_particles, _ = positions.shape

    if n_frames == 0:
        raise ValueError(
            "No trajectory frames were provided."
        )

    if n_particles < 2:
        raise ValueError(
            "At least two particles are required for an RDF."
        )

    if r_max is None:
        r_max = box_length / 2.0

    if r_max <= 0:
        raise ValueError(
            "r_max must be greater than zero."
        )

    if r_max > box_length / 2.0:
        raise ValueError(
            "For a cubic periodic box, r_max must not be "
            "greater than box_length / 2."
        )

    # ------------------------------------------------------------
    # Radial bins
    # ------------------------------------------------------------

    bin_edges = np.linspace(
        0.0,
        r_max,
        n_bins + 1,
    )

    r_centers = 0.5 * (
        bin_edges[:-1]
        + bin_edges[1:]
    )

    pair_histogram = np.zeros(
        n_bins,
        dtype=np.int64,
    )

    # ------------------------------------------------------------
    # Analyze every selected frame
    # ------------------------------------------------------------

    for frame_index, frame_positions in enumerate(positions):

        # Wrap all particles into [0, L)
        wrapped_positions = np.mod(
            frame_positions,
            box_length,
        )

        # Periodic KD-tree
        tree = cKDTree(
            wrapped_positions,
            boxsize=box_length,
        )

        # query_pairs returns every pair only once: i < j
        pairs = tree.query_pairs(
            r=r_max,
            output_type="ndarray",
        )

        if pairs.size > 0:
            displacement_vectors = (
                wrapped_positions[pairs[:, 0]]
                - wrapped_positions[pairs[:, 1]]
            )

            # Minimum-image convention
            displacement_vectors -= (
                box_length
                * np.rint(
                    displacement_vectors
                    / box_length
                )
            )

            distances = np.linalg.norm(
                displacement_vectors,
                axis=1,
            )

            frame_histogram, _ = np.histogram(
                distances,
                bins=bin_edges,
            )

            pair_histogram += frame_histogram

        if (
            frame_index % 100 == 0
            or frame_index == n_frames - 1
        ):
            print(
                f"{progress_label}: frame "
                f"{frame_index + 1} of {n_frames}"
            )

    # ------------------------------------------------------------
    # RDF normalization
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

    total_number_of_pairs = (
        n_particles
        * (n_particles - 1)
        / 2.0
    )

    expected_pairs_per_frame = (
        total_number_of_pairs
        * shell_volumes
        / box_volume
    )

    expected_pairs_all_frames = (
        n_frames
        * expected_pairs_per_frame
    )

    g_r = np.divide(
        pair_histogram,
        expected_pairs_all_frames,
        out=np.zeros_like(
            expected_pairs_all_frames,
            dtype=float,
        ),
        where=expected_pairs_all_frames > 0,
    )

    # ------------------------------------------------------------
    # Coordination number
    # ------------------------------------------------------------

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
        pair_histogram,
    )


def analyze_xyz_trajectory(
    xyz_path,
    box_length_nm,
    start_frame=0,
    stop_frame=None,
    frame_stride=1,
    n_bins=200,
    r_max_nm=None,
    coordinates_in_angstrom=True,
    progress_label="RDF",
):
    """
    Read an XYZ trajectory, select frames, and calculate its RDF.

    Returns
    -------
    result : dict
        Dictionary containing the trajectory metadata and RDF arrays.
    """

    xyz_path = Path(xyz_path)

    if frame_stride < 1:
        raise ValueError(
            "frame_stride must be at least 1."
        )

    positions, atom_names = read_xyz_trajectory(
        xyz_path
    )

    print(
        f"{progress_label}: complete trajectory shape = "
        f"{positions.shape}"
    )

    if coordinates_in_angstrom:
        positions = positions * 0.1

    selected_positions = positions[
        start_frame:stop_frame:frame_stride
    ]

    if selected_positions.shape[0] == 0:
        raise ValueError(
            f"{progress_label}: the selected frame range is empty."
        )

    print(
        f"{progress_label}: analyzed frames = "
        f"{selected_positions.shape[0]}"
    )

    print(
        f"{progress_label}: particles per frame = "
        f"{selected_positions.shape[1]}"
    )

    (
        r_nm,
        g_r,
        coordination_number,
        pair_counts,
    ) = calculate_rdf(
        positions=selected_positions,
        box_length=box_length_nm,
        n_bins=n_bins,
        r_max=r_max_nm,
        progress_label=progress_label,
    )

    return {
        "xyz_path": xyz_path,
        "atom_names": atom_names,
        "n_frames": selected_positions.shape[0],
        "n_particles": selected_positions.shape[1],
        "r_nm": r_nm,
        "g_r": g_r,
        "coordination_number": coordination_number,
        "pair_counts": pair_counts,
    }


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():
    result = analyze_xyz_trajectory(
        xyz_path=XYZ_PATH,
        box_length_nm=BOX_LENGTH_NM,
        start_frame=START_FRAME,
        stop_frame=STOP_FRAME,
        frame_stride=FRAME_STRIDE,
        n_bins=N_BINS,
        r_max_nm=R_MAX_NM,
        coordinates_in_angstrom=XYZ_COORDINATES_IN_ANGSTROM,
        progress_label=XYZ_PATH.stem,
    )

    r_nm = result["r_nm"]
    g_r = result["g_r"]
    coordination_number = result["coordination_number"]
    pair_counts = result["pair_counts"]

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------

    if SAVE_CSV:
        output_data = np.column_stack(
            (
                r_nm,
                g_r,
                coordination_number,
                pair_counts,
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
            comments="",
        )

        print(
            f"RDF CSV saved to:\n{CSV_PATH.resolve()}"
        )

    # ------------------------------------------------------------
    # Plot RDF
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.plot(
        r_nm,
        g_r,
        linewidth=2.0,
        label=r"$g(r)$",
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.5,
        label="Uniform ideal-gas reference",
    )

    lj_minimum_nm = (
        2.0**(1.0 / 6.0)
        * SIGMA_NM
    )

    ax.axvline(
        lj_minimum_nm,
        linestyle=":",
        linewidth=1.5,
        label=(
            r"$r_\mathrm{min}=2^{1/6}\sigma$"
        ),
    )

    ax.set_xlabel(
        "Particle distance $r$ / nm"
    )

    ax.set_ylabel(
        r"Radial distribution function $g(r)$"
    )

    ax.set_title(
        "Radial Distribution Function of the "
        "Lennard-Jones Trajectory",
        pad=15,
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_FONT_SIZE,
    )

    ax.grid(
        True,
        alpha=0.4,
    )

    ax.legend()

    fig.tight_layout()

    if SAVE_PLOT:
        fig.savefig(
            PLOT_PATH,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"RDF plot saved to:\n{PLOT_PATH.resolve()}"
        )

    plt.show()


if __name__ == "__main__":
    main()
