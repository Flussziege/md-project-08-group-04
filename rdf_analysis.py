"""
Radial distribution function comparison for XYZ trajectories
=============================================================

PURPOSE
-------
This script calculates and compares the radial distribution functions g(r)
of multiple molecular-dynamics trajectories stored in XYZ format.

All RDF curves are drawn in the same figure.

For a homogeneous ideal gas:

    g(r) -> 1

For a liquid, g(r) usually shows:

    - a pronounced first maximum,
    - a minimum after the first maximum,
    - additional damped maxima at larger distances.

USAGE
-----
1. Add the XYZ trajectories to TRAJECTORIES.
2. Set the cubic box length for every trajectory.
3. Set START_FRAME to exclude equilibration.
4. Set FRAME_STRIDE to reduce the number of analyzed frames.
5. Select the desired y-axis scale.
6. Run:

       python rdf_comparison.py

EXPECTED UNITS
--------------
The simulation writes XYZ coordinates in angstrom by default.

    1 angstrom = 0.1 nm

Set XYZ_COORDINATES_IN_ANGSTROM = True in this case.

OUTPUT
------
The script saves:

    - one RDF CSV file for every trajectory,
    - one PNG figure containing all RDF curves.

Y-AXIS SCALES
-------------
Available options are:

    "linear"
        Standard RDF representation.

    "quadratic"
        Emphasizes large RDF peaks.

    "square_root"
        Compresses large peaks and emphasizes smaller differences.

    "logarithmic"
        Logarithmic scale. Values equal to zero cannot be displayed.
"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


# ================================================================
# INPUT PARAMETERS
# ================================================================


TRAJECTORIES = [
    {
        "path": Path(
            r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04\results\2026-07-16_18-51-01-300K\my_simulation_pos-300K.xyz"
        ),
        "label": "300K",
        "box_length_nm": 6.0,
    },
    {
        "path": Path(
            r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04\results\2026-07-15_19-30-00-5K\my_simulation_pos-5K.xyz"
        ),
        "label": "50 K",
        "box_length_nm": 6.0,
    },
]



# Lennard-Jones sigma parameter in nm
SIGMA_NM = 0.34

# First trajectory frame used for the analysis
START_FRAME = 3000

# Last trajectory frame used for the analysis.
# None means: continue to the final frame.
STOP_FRAME = None

# Analyze only every nth frame
FRAME_STRIDE = 20

# Number of radial-distance bins
N_BINS = 200

# Maximum analyzed distance in nm.
#
# None means that the common maximum distance is calculated as:
#
#     minimum box length / 2
#
# This ensures that all trajectories use the same distance range.
R_MAX_NM = None

# Set True when XYZ coordinates are stored in angstrom
XYZ_COORDINATES_IN_ANGSTROM = True

# Save calculated data and figure
SAVE_CSV = True
SAVE_PLOT = True

# Available options:
#
#     "linear"
#     "quadratic"
#     "square_root"
#     "logarithmic"
#
Y_AXIS_SCALE = "square_root"

# Output path for the comparison figure
COMPARISON_PLOT_PATH = TRAJECTORIES[0]["path"].with_name(
    "rdf_comparison.png"
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
        raise ValueError(
            f"The XYZ file is empty:\n{filename.resolve()}"
        )

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
            f"The XYZ file contains an incomplete frame or does not "
            f"use a constant particle number:\n{filename.resolve()}"
        )

    n_frames = len(lines) // lines_per_frame

    positions = np.zeros(
        (n_frames, n_particles, 3),
        dtype=float,
    )

    atom_names = []

    for frame_index in range(n_frames):
        frame_start = frame_index * lines_per_frame

        # Verify that every frame has the same particle number
        try:
            frame_particle_count = int(
                lines[frame_start].strip()
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid particle count in frame "
                f"{frame_index + 1}."
            ) from error

        if frame_particle_count != n_particles:
            raise ValueError(
                f"Frame {frame_index + 1} contains "
                f"{frame_particle_count} particles, but the first "
                f"frame contains {n_particles} particles."
            )

        for particle_index in range(n_particles):
            line_index = frame_start + 2 + particle_index
            columns = lines[line_index].split()

            if len(columns) < 4:
                raise ValueError(
                    f"Invalid XYZ line {line_index + 1}: "
                    "expected an atom label and three coordinates."
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

        Positions and box_length must use the same length unit.

    box_length : float
        Length of the cubic simulation box.

    n_bins : int, default=200
        Number of radial-distance intervals.

    r_max : float or None
        Maximum analyzed distance.

        If None:

            r_max = box_length / 2

    progress_label : str, default="RDF"
        Text displayed in progress messages.

    Returns
    -------
    r_centers : np.ndarray
        Centers of the radial bins.

    g_r : np.ndarray
        Radial distribution function.

    coordination_number : np.ndarray
        Mean number of neighbors within radius r.

    pair_histogram : np.ndarray
        Total number of measured particle pairs in every bin.
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

        # Wrap particles into the interval [0, box_length)
        wrapped_positions = np.mod(
            frame_positions,
            box_length,
        )

        # Create a periodic KD-tree
        tree = cKDTree(
            wrapped_positions,
            boxsize=box_length,
        )

        # query_pairs returns every pair once, with i < j
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
        Dictionary containing trajectory metadata and RDF arrays.
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
            f"{progress_label}: the selected frame range is empty. "
            f"The complete trajectory contains "
            f"{positions.shape[0]} frames and START_FRAME is "
            f"{start_frame}."
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
# Y-AXIS SCALING
# ================================================================

def apply_y_axis_scale(ax, scale_name, maximum_g_r):
    """
    Apply the selected y-axis scale.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Plot axes.

    scale_name : str
        Name of the selected scale.

    maximum_g_r : float
        Largest RDF value in all analyzed trajectories.
    """

    scale_name = scale_name.lower()

    if scale_name == "linear":
        ax.set_yscale("linear")
        ax.set_ylim(
            bottom=0.0,
            top=maximum_g_r * 1.05,
        )

    elif scale_name == "quadratic":
        ax.set_ylim(
            bottom=0.0,
            top=maximum_g_r * 1.05,
        )

        ax.set_yscale(
            "function",
            functions=(
                lambda y: np.square(y),
                lambda y: np.sqrt(
                    np.maximum(y, 0.0)
                ),
            ),
        )

    elif scale_name == "square_root":
        ax.set_ylim(
            bottom=0.0,
            top=maximum_g_r * 1.05,
        )

        ax.set_yscale(
            "function",
            functions=(
                lambda y: np.sqrt(
                    np.maximum(y, 0.0)
                ),
                lambda y: np.square(y),
            ),
        )

    elif scale_name == "logarithmic":
        # Logarithmic axes cannot display zero.
        positive_lower_limit = max(
            maximum_g_r * 1.0e-4,
            1.0e-6,
        )

        ax.set_yscale("log")

        ax.set_ylim(
            bottom=positive_lower_limit,
            top=maximum_g_r * 1.05,
        )

    else:
        raise ValueError(
            f"Unknown Y_AXIS_SCALE: {scale_name}. "
            "Available options are 'linear', 'quadratic', "
            "'square_root', and 'logarithmic'."
        )


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():
    if not TRAJECTORIES:
        raise ValueError(
            "TRAJECTORIES must contain at least one trajectory."
        )

    # ------------------------------------------------------------
    # Determine a common maximum RDF distance
    # ------------------------------------------------------------

    smallest_half_box_length = min(
        trajectory["box_length_nm"] / 2.0
        for trajectory in TRAJECTORIES
    )

    if R_MAX_NM is None:
        common_r_max_nm = smallest_half_box_length
    else:
        common_r_max_nm = R_MAX_NM

    if common_r_max_nm <= 0:
        raise ValueError(
            "R_MAX_NM must be greater than zero."
        )

    if common_r_max_nm > smallest_half_box_length:
        raise ValueError(
            "R_MAX_NM is larger than half of the smallest "
            "simulation-box length. Use a value not greater than "
            f"{smallest_half_box_length:.6g} nm."
        )

    print(
        "Common maximum RDF distance = "
        f"{common_r_max_nm:.6g} nm"
    )

    # ------------------------------------------------------------
    # Calculate all RDF curves
    # ------------------------------------------------------------

    all_results = []
    maximum_g_r = 1.0

    for trajectory in TRAJECTORIES:
        required_keys = {
            "path",
            "label",
            "box_length_nm",
        }

        missing_keys = (
            required_keys
            - set(trajectory)
        )

        if missing_keys:
            raise ValueError(
                "A trajectory entry is missing the following keys: "
                f"{sorted(missing_keys)}"
            )

        xyz_path = Path(
            trajectory["path"]
        )

        label = str(
            trajectory["label"]
        )

        box_length_nm = float(
            trajectory["box_length_nm"]
        )

        print()
        print("=" * 70)
        print(f"Analyzing: {label}")
        print(f"File: {xyz_path}")
        print("=" * 70)

        result = analyze_xyz_trajectory(
            xyz_path=xyz_path,
            box_length_nm=box_length_nm,
            start_frame=START_FRAME,
            stop_frame=STOP_FRAME,
            frame_stride=FRAME_STRIDE,
            n_bins=N_BINS,
            r_max_nm=common_r_max_nm,
            coordinates_in_angstrom=(
                XYZ_COORDINATES_IN_ANGSTROM
            ),
            progress_label=label,
        )

        result["label"] = label
        result["box_length_nm"] = box_length_nm

        all_results.append(result)

        if result["g_r"].size > 0:
            maximum_g_r = max(
                maximum_g_r,
                float(np.max(result["g_r"])),
            )

        # --------------------------------------------------------
        # Save one CSV file for every trajectory
        # --------------------------------------------------------

        if SAVE_CSV:
            csv_path = xyz_path.with_name(
                f"{xyz_path.stem}_rdf.csv"
            )

            output_data = np.column_stack(
                (
                    result["r_nm"],
                    result["g_r"],
                    result["coordination_number"],
                    result["pair_counts"],
                )
            )

            np.savetxt(
                csv_path,
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
                f"RDF CSV saved to:\n{csv_path.resolve()}"
            )

    # ------------------------------------------------------------
    # Plot all RDF curves
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    for result in all_results:
        ax.plot(
            result["r_nm"],
            result["g_r"],
            linewidth=2.0,
            label=result["label"],
        )

    # Ideal-gas reference
    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.5,
        label="Uniform ideal-gas reference",
    )

    # Position of the Lennard-Jones potential minimum
    lj_minimum_nm = (
        2.0**(1.0 / 6.0)
        * SIGMA_NM
    )

    ax.axvline(
        lj_minimum_nm,
        linestyle=":",
        linewidth=1.5,
        label=r"$r_\mathrm{min}=2^{1/6}\sigma$",
    )

    # Apply selected y-axis transformation
    apply_y_axis_scale(
        ax=ax,
        scale_name=Y_AXIS_SCALE,
        maximum_g_r=maximum_g_r,
    )

    ax.set_xlim(
        0.0,
        common_r_max_nm,
    )

    ax.set_xlabel(
        "Particle distance $r$ / nm"
    )

    ax.set_ylabel(
        r"Radial distribution function $g(r)$"
    )

    ax.set_title(
        "Comparison of Radial Distribution Functions",
        pad=15,
    )

    ax.grid(
        True,
        alpha=0.4,
    )

    ax.legend()

    fig.tight_layout()

    # ------------------------------------------------------------
    # Save comparison plot
    # ------------------------------------------------------------

    if SAVE_PLOT:
        COMPARISON_PLOT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            COMPARISON_PLOT_PATH,
            dpi=300,
            bbox_inches="tight",
        )

        print()
        print(
            "RDF comparison plot saved to:\n"
            f"{COMPARISON_PLOT_PATH.resolve()}"
        )

    plt.show()


if __name__ == "__main__":
    main()