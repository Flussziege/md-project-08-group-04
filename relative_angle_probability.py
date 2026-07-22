"""
Relative-angle distribution comparison for multiple XYZ trajectories
====================================================================

PURPOSE
-------
This script compares the local angular structure of several molecular-
dynamics trajectories stored in XYZ format.

For every trajectory, it:

1. reads and selects trajectory frames,
2. determines the first coordination-shell cutoff from the first minimum
   of the radial distribution function (unless a fixed cutoff is supplied),
3. calculates the relative-angle probability density P(theta),
4. calculates the isotropy-corrected angular correlation

       g_theta(theta) = P(theta) / P_0(theta),

5. saves one CSV file per trajectory, and
6. draws shared comparison plots for all trajectories.

EXPECTED UNITS
--------------
The box lengths and all internal calculations use nanometres.

If the XYZ coordinates are stored in angstroms, set
XYZ_COORDINATES_IN_ANGSTROM = True:

    1 angstrom = 0.1 nm

DEPENDENCY
----------
The function read_xyz_trajectory is imported from cluster_functions.py.
That file must be importable from the current Python environment.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from cluster_functions import read_xyz_trajectory


# ================================================================
# INPUT PARAMETERS
# ================================================================

# Add or remove trajectory dictionaries as needed.
# Replace the example paths with the actual paths on your computer.
TRAJECTORIES = [
    {
        "path": Path(
            r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik"
            r"\md-project-08-group-04\results"
            r"\2026-07-15_19-30-00-5K"
            r"\my_simulation_pos-5K.xyz"
        ),
        "label": "5 K",
        "box_length_nm": 6.0,
    },
    {
        "path": Path(
            r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik"
            r"\md-project-08-group-04\results"
            r"\2026-07-15_19-30-00-50K"
            r"\my_simulation_pos-50K.xyz"
        ),
        "label": "50 K",
        "box_length_nm": 6.0,
    },
    {
        "path": Path(
            r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik"
            r"\md-project-08-group-04\results"
            r"\2026-07-15_19-30-00-100K"
            r"\my_simulation_pos-100K.xyz"
        ),
        "label": "100 K",
        "box_length_nm": 6.0,
    },
]

# Lennard-Jones sigma parameter in nm
SIGMA_NM = 0.34

# 3: angles between three-dimensional bond vectors
# 2: use only the x and y components
DIMENSION = 3

# Discard the first part of each simulation as equilibration.
START_FRAME = 4000

# None means: continue to the final frame.
STOP_FRAME = None

# Analyze only every nth frame.
FRAME_STRIDE = 20

# Number of angular bins between 0 and 180 degrees.
N_ANGLE_BINS = 180

# Global neighbor cutoff in nm.
#
# None:
#   Determine a separate cutoff for every trajectory from the first RDF
#   minimum after the first RDF maximum.
#
# Float:
#   Use the same fixed cutoff for every trajectory.
#
# A trajectory dictionary may optionally contain its own override:
#
#     "neighbor_cutoff_nm": 0.52
#
NEIGHBOR_CUTOFF_NM = None

# RDF parameters used to determine automatic neighbor cutoffs.
N_RDF_BINS = 300
RDF_R_MAX_NM = None
RDF_SMOOTHING_SIGMA_BINS = 2.0

# The XYZ files contain coordinates in angstroms.
XYZ_COORDINATES_IN_ANGSTROM = True

# Save output files.
SAVE_CSV = True
SAVE_PLOT = True
SAVE_INDIVIDUAL_RDF_DIAGNOSTICS = True
SAVE_COMBINED_RDF_DIAGNOSTIC = True

# Available y-axis scales:
#
#     "linear"
#     "quadratic"
#     "square_root"
#     "logarithmic"
#
# The conventional scientific representation is "linear".
# "quadratic" visually emphasizes large peaks.
Y_AXIS_SCALE = "quadratic"

# Add each automatically selected cutoff to its curve label.
SHOW_CUTOFF_IN_LEGEND = True

# Combined plots are saved beside the first trajectory.
OUTPUT_DIRECTORY = TRAJECTORIES[0]["path"].parent

ANGLE_COMPARISON_PLOT_PATH = OUTPUT_DIRECTORY / (
    "relative_angle_probability_comparison.png"
)

ANGULAR_CORRELATION_COMPARISON_PLOT_PATH = OUTPUT_DIRECTORY / (
    "relative_angle_correlation_comparison.png"
)

RDF_COMPARISON_PLOT_PATH = OUTPUT_DIRECTORY / (
    "rdf_neighbor_cutoff_comparison.png"
)


# ================================================================
# PLOT APPEARANCE
# ================================================================

TITLE_FONTSIZE = 17
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 11

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelsize": AXIS_LABEL_FONTSIZE,
        "xtick.labelsize": TICK_LABEL_FONTSIZE,
        "ytick.labelsize": TICK_LABEL_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
    }
)


# ================================================================
# HELPER FUNCTIONS
# ================================================================


def validate_positions(positions):
    """Validate and return a trajectory-position array."""
    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 3:
        raise ValueError(
            "positions must have shape (n_frames, n_particles, 3)."
        )

    if positions.shape[2] != 3:
        raise ValueError(
            "The final dimension of positions must have size 3."
        )

    if positions.shape[0] == 0:
        raise ValueError("No frames were provided.")

    if positions.shape[1] < 3:
        raise ValueError(
            "At least three particles are required for relative angles."
        )

    if not np.all(np.isfinite(positions)):
        raise ValueError(
            "positions contains NaN or infinite values."
        )

    return positions


def minimum_image(displacements, box_length):
    """Apply the minimum-image convention."""
    return displacements - box_length * np.rint(
        displacements / box_length
    )


def apply_y_axis_scale(ax, scale_name, maximum_value):
    """Apply a nonnegative y-axis transformation to a Matplotlib axis."""
    scale_name = str(scale_name).lower()
    maximum_value = max(float(maximum_value), 1.0e-12)
    upper_limit = 1.05 * maximum_value

    if scale_name == "linear":
        ax.set_yscale("linear")
        ax.set_ylim(0.0, upper_limit)

    elif scale_name == "quadratic":
        ax.set_ylim(0.0, upper_limit)
        ax.set_yscale(
            "function",
            functions=(
                lambda y: np.square(np.asarray(y, dtype=float)),
                lambda y: np.sqrt(
                    np.maximum(np.asarray(y, dtype=float), 0.0)
                ),
            ),
        )

    elif scale_name == "square_root":
        ax.set_ylim(0.0, upper_limit)
        ax.set_yscale(
            "function",
            functions=(
                lambda y: np.sqrt(
                    np.maximum(np.asarray(y, dtype=float), 0.0)
                ),
                lambda y: np.square(np.asarray(y, dtype=float)),
            ),
        )

    elif scale_name == "logarithmic":
        positive_lower_limit = max(maximum_value * 1.0e-4, 1.0e-8)
        ax.set_yscale("log")
        ax.set_ylim(positive_lower_limit, upper_limit)

    else:
        raise ValueError(
            f"Unknown y-axis scale: {scale_name}. "
            "Use 'linear', 'quadratic', 'square_root', or 'logarithmic'."
        )


def add_reference_angle_lines(ax):
    """Add common structural reference angles to a plot."""
    for reference_angle in (60.0, 90.0, 120.0, 180.0):
        ax.axvline(
            reference_angle,
            linestyle=":",
            linewidth=0.8,
            alpha=0.5,
        )


def validate_trajectory_configuration(trajectory):
    """Validate one TRAJECTORIES dictionary."""
    required_keys = {"path", "label", "box_length_nm"}
    missing_keys = required_keys - set(trajectory)

    if missing_keys:
        raise ValueError(
            "A trajectory entry is missing keys: "
            f"{sorted(missing_keys)}"
        )

    xyz_path = Path(trajectory["path"])
    label = str(trajectory["label"])
    box_length_nm = float(trajectory["box_length_nm"])

    if not label.strip():
        raise ValueError("Every trajectory label must be non-empty.")

    if box_length_nm <= 0:
        raise ValueError(
            f"The box length for {label!r} must be greater than zero."
        )

    return xyz_path, label, box_length_nm


# ================================================================
# RDF FOR THE NEIGHBOR CUTOFF
# ================================================================


def calculate_rdf(
    positions,
    box_length,
    n_bins=300,
    r_max=None,
    progress_label="RDF",
):
    """
    Calculate the radial distribution function g(r).

    The RDF is used here to locate the first minimum after the first
    maximum and therefore the boundary of the first coordination shell.
    """
    positions = validate_positions(positions)

    if box_length <= 0:
        raise ValueError("box_length must be greater than zero.")

    if n_bins < 10:
        raise ValueError("n_bins should be at least 10.")

    n_frames, n_particles, _ = positions.shape

    if r_max is None:
        r_max = box_length / 2.0

    if not 0 < r_max <= box_length / 2.0:
        raise ValueError(
            "r_max must be greater than zero and no larger than L/2."
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
                f"{progress_label} RDF: frame "
                f"{frame_index + 1} of {n_frames}"
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
    """Determine the first RDF minimum after the first RDF maximum."""
    r_nm = np.asarray(r_nm, dtype=float)
    g_r = np.asarray(g_r, dtype=float)

    if r_nm.shape != g_r.shape:
        raise ValueError("r_nm and g_r must have the same shape.")

    if sigma_nm <= 0:
        raise ValueError("sigma_nm must be greater than zero.")

    if smoothing_sigma_bins < 0:
        raise ValueError(
            "smoothing_sigma_bins must not be negative."
        )

    smoothed_g_r = gaussian_filter1d(
        g_r,
        sigma=smoothing_sigma_bins,
        mode="nearest",
    )

    # The first peak is expected near r_min = 2^(1/6) sigma.
    peak_search_mask = (
        (r_nm >= 0.85 * sigma_nm)
        & (r_nm <= 1.65 * sigma_nm)
    )

    peak_indices = np.flatnonzero(peak_search_mask)
    if peak_indices.size == 0:
        raise RuntimeError(
            "No valid search region for the first RDF peak."
        )

    first_peak_index = peak_indices[
        np.argmax(smoothed_g_r[peak_indices])
    ]

    minimum_candidates, _ = find_peaks(-smoothed_g_r)
    minimum_candidates = minimum_candidates[
        (minimum_candidates > first_peak_index)
        & (r_nm[minimum_candidates] <= 2.5 * sigma_nm)
    ]

    if minimum_candidates.size > 0:
        first_minimum_index = minimum_candidates[0]
    else:
        fallback_mask = (
            (r_nm > r_nm[first_peak_index])
            & (r_nm <= 2.0 * sigma_nm)
        )
        fallback_indices = np.flatnonzero(fallback_mask)

        if fallback_indices.size == 0:
            fallback_cutoff = 1.5 * sigma_nm
            print(
                "WARNING: The first RDF minimum could not be determined. "
                f"Using {fallback_cutoff:.4f} nm."
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
# RELATIVE-ANGLE DISTRIBUTION
# ================================================================


def calculate_relative_angle_probability(
    positions,
    box_length,
    neighbor_cutoff,
    n_angle_bins=180,
    dimension=3,
    progress_label="Angles",
):
    """
    Calculate the distribution of angles between two neighbor bonds.

    For a central particle i and two neighbors j and k:

        theta_jik = arccos(
            r_ij dot r_ik / (|r_ij| |r_ik|)
        )

    All unordered neighbor pairs j < k are counted.
    """
    positions = validate_positions(positions)

    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3.")

    if box_length <= 0:
        raise ValueError("box_length must be greater than zero.")

    if not 0 < neighbor_cutoff <= box_length / 2.0:
        raise ValueError(
            "neighbor_cutoff must be greater than zero and no larger "
            "than L/2."
        )

    if n_angle_bins < 1:
        raise ValueError("n_angle_bins must be at least 1.")

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
            # The periodic KD-tree includes the central particle itself.
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
                f"{progress_label}: frame "
                f"{frame_index + 1} of {n_frames}"
            )

    total_angle_count = int(angle_histogram.sum())
    if total_angle_count == 0:
        raise RuntimeError(
            "No angles were found. Check the neighbor cutoff and "
            "the selected frames."
        )

    bin_widths_deg = np.diff(angle_edges_deg)
    probability_per_bin = angle_histogram / total_angle_count
    probability_density_per_degree = (
        probability_per_bin / bin_widths_deg
    )

    angle_edges_rad = np.radians(angle_edges_deg)

    if dimension == 3:
        # For two independent isotropic 3D directions:
        # p(theta) = 1/2 sin(theta), 0 <= theta <= pi.
        isotropic_probability_per_bin = 0.5 * (
            np.cos(angle_edges_rad[:-1])
            - np.cos(angle_edges_rad[1:])
        )
    else:
        # The folded relative angle in 2D is uniform on [0, pi].
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
# TRAJECTORY ANALYSIS
# ================================================================


def analyze_trajectory(trajectory):
    """Read and analyze one configured trajectory."""
    xyz_path, label, box_length_nm = validate_trajectory_configuration(
        trajectory
    )

    print()
    print("=" * 72)
    print(f"Analyzing: {label}")
    print(f"File: {xyz_path}")
    print("=" * 72)

    positions, atom_names = read_xyz_trajectory(xyz_path)
    positions = validate_positions(positions)

    print(f"{label}: complete trajectory shape = {positions.shape}")

    if XYZ_COORDINATES_IN_ANGSTROM:
        positions = positions * 0.1

    selected_positions = positions[
        START_FRAME:STOP_FRAME:FRAME_STRIDE
    ]

    if selected_positions.shape[0] == 0:
        raise ValueError(
            f"{label}: the frame selection is empty. The complete "
            f"trajectory contains {positions.shape[0]} frames, while "
            f"START_FRAME is {START_FRAME}."
        )

    print(
        f"{label}: frames used for analysis = "
        f"{selected_positions.shape[0]}"
    )
    print(
        f"{label}: particles per frame = "
        f"{selected_positions.shape[1]}"
    )

    trajectory_cutoff = trajectory.get(
        "neighbor_cutoff_nm",
        NEIGHBOR_CUTOFF_NM,
    )

    r_nm = None
    g_r = None
    rdf_counts = None
    smoothed_g_r = None
    first_peak_index = None
    first_minimum_index = None

    if trajectory_cutoff is None:
        r_nm, g_r, rdf_counts = calculate_rdf(
            positions=selected_positions,
            box_length=box_length_nm,
            n_bins=N_RDF_BINS,
            r_max=RDF_R_MAX_NM,
            progress_label=label,
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
            f"{label}: automatically determined neighbor cutoff "
            f"= {neighbor_cutoff_nm:.5f} nm"
        )
    else:
        neighbor_cutoff_nm = float(trajectory_cutoff)

        if not 0 < neighbor_cutoff_nm <= box_length_nm / 2.0:
            raise ValueError(
                f"{label}: neighbor cutoff must be greater than zero "
                "and no larger than half the box length."
            )

        print(
            f"{label}: manually specified neighbor cutoff "
            f"= {neighbor_cutoff_nm:.5f} nm"
        )

    (
        angle_deg,
        probability_density,
        isotropic_density,
        angular_correlation,
        angle_counts,
        mean_coordination_number,
    ) = calculate_relative_angle_probability(
        positions=selected_positions,
        box_length=box_length_nm,
        neighbor_cutoff=neighbor_cutoff_nm,
        n_angle_bins=N_ANGLE_BINS,
        dimension=DIMENSION,
        progress_label=f"{label} angles",
    )

    print(f"{label}: counted relative angles = {angle_counts.sum()}")
    print(
        f"{label}: mean coordination number within cutoff = "
        f"{mean_coordination_number:.3f}"
    )

    return {
        "xyz_path": xyz_path,
        "label": label,
        "box_length_nm": box_length_nm,
        "atom_names": atom_names,
        "n_frames": selected_positions.shape[0],
        "n_particles": selected_positions.shape[1],
        "neighbor_cutoff_nm": neighbor_cutoff_nm,
        "r_nm": r_nm,
        "g_r": g_r,
        "rdf_counts": rdf_counts,
        "smoothed_g_r": smoothed_g_r,
        "first_peak_index": first_peak_index,
        "first_minimum_index": first_minimum_index,
        "angle_deg": angle_deg,
        "probability_density": probability_density,
        "isotropic_density": isotropic_density,
        "angular_correlation": angular_correlation,
        "angle_counts": angle_counts,
        "mean_coordination_number": mean_coordination_number,
    }


def make_curve_label(result):
    """Create a legend label for one trajectory result."""
    if SHOW_CUTOFF_IN_LEGEND:
        return (
            f"{result['label']} "
            f"($r_c$ = {result['neighbor_cutoff_nm']:.3f} nm)"
        )

    return result["label"]


# ================================================================
# OUTPUT FUNCTIONS
# ================================================================


def save_angle_csv(result):
    """Save one angle-analysis CSV file beside its XYZ trajectory."""
    csv_path = result["xyz_path"].with_name(
        f"{result['xyz_path'].stem}_relative_angle_probability.csv"
    )

    output_data = np.column_stack(
        (
            result["angle_deg"],
            result["probability_density"],
            result["isotropic_density"],
            result["angular_correlation"],
            result["angle_counts"],
        )
    )

    np.savetxt(
        csv_path,
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

    print(f"{result['label']}: angle CSV saved to:\n{csv_path.resolve()}")


def save_individual_rdf_diagnostic(result):
    """Save an RDF cutoff diagnostic for one automatically analyzed file."""
    if result["r_nm"] is None:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        result["r_nm"],
        result["g_r"],
        linewidth=1.0,
        label=r"$g(r)$",
    )
    ax.plot(
        result["r_nm"],
        result["smoothed_g_r"],
        linewidth=1.5,
        label="Smoothed RDF",
    )
    ax.axvline(
        result["r_nm"][result["first_peak_index"]],
        linestyle=":",
        linewidth=1.2,
        label="First peak",
    )
    ax.axvline(
        result["neighbor_cutoff_nm"],
        linestyle="--",
        linewidth=1.2,
        label="Neighbor cutoff: first minimum",
    )

    maximum_value = max(
        float(np.max(result["g_r"])),
        float(np.max(result["smoothed_g_r"])),
    )
    apply_y_axis_scale(ax, Y_AXIS_SCALE, maximum_value)

    ax.set_xlabel(r"Distance $r$ / nm")
    ax.set_ylabel(r"Radial distribution function $g(r)$")
    ax.set_title(
        f"First Coordination-Shell Cutoff: {result['label']}",
        pad=12,
    )
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()

    diagnostic_path = result["xyz_path"].with_name(
        f"{result['xyz_path'].stem}_rdf_neighbor_cutoff.png"
    )

    fig.savefig(
        diagnostic_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(
        f"{result['label']}: RDF diagnostic saved to:\n"
        f"{diagnostic_path.resolve()}"
    )


def plot_combined_rdf_diagnostic(results):
    """Plot all available smoothed RDFs and selected cutoffs."""
    rdf_results = [result for result in results if result["r_nm"] is not None]

    if not rdf_results:
        print(
            "Combined RDF diagnostic skipped because all cutoffs were "
            "specified manually."
        )
        return None

    fig, ax = plt.subplots(figsize=(11, 7))
    maximum_value = 0.0

    for result in rdf_results:
        line, = ax.plot(
            result["r_nm"],
            result["smoothed_g_r"],
            linewidth=2.0,
            label=make_curve_label(result),
        )
        ax.axvline(
            result["neighbor_cutoff_nm"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
            color=line.get_color(),
        )
        maximum_value = max(
            maximum_value,
            float(np.max(result["smoothed_g_r"])),
        )

    apply_y_axis_scale(ax, Y_AXIS_SCALE, maximum_value)

    ax.set_xlabel(r"Distance $r$ / nm")
    ax.set_ylabel(r"Smoothed radial distribution function $g(r)$")
    ax.set_title(
        "Comparison of First Coordination-Shell Cutoffs",
        pad=12,
    )
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()

    if SAVE_PLOT and SAVE_COMBINED_RDF_DIAGNOSTIC:
        RDF_COMPARISON_PLOT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            RDF_COMPARISON_PLOT_PATH,
            dpi=300,
            bbox_inches="tight",
        )
        print(
            "Combined RDF diagnostic saved to:\n"
            f"{RDF_COMPARISON_PLOT_PATH.resolve()}"
        )

    return fig


def plot_angle_probability_comparison(results):
    """Plot all relative-angle probability-density curves."""
    fig, ax = plt.subplots(figsize=(11, 7))
    maximum_value = 0.0

    for result in results:
        ax.plot(
            result["angle_deg"],
            result["probability_density"],
            linewidth=2.0,
            label=make_curve_label(result),
        )
        maximum_value = max(
            maximum_value,
            float(np.max(result["probability_density"])),
        )

    # The isotropic reference is identical for every trajectory because
    # DIMENSION and the angular bins are shared.
    reference = results[0]
    ax.plot(
        reference["angle_deg"],
        reference["isotropic_density"],
        linestyle="--",
        linewidth=1.5,
        label=(
            r"Isotropic reference $P_0(\theta)$"
            if DIMENSION == 3
            else "Uniform 2D reference"
        ),
    )
    maximum_value = max(
        maximum_value,
        float(np.max(reference["isotropic_density"])),
    )

    add_reference_angle_lines(ax)
    apply_y_axis_scale(ax, Y_AXIS_SCALE, maximum_value)

    ax.set_xlabel(r"Relative angle $\theta$ / degrees")
    ax.set_ylabel(
        r"Probability density $P(\theta)$ / degree$^{-1}$"
    )
    ax.set_title(
        "Comparison of Relative-Angle Distributions "
        "in the First Coordination Shell",
        pad=12,
    )
    ax.set_xlim(0.0, 180.0)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()

    if SAVE_PLOT:
        ANGLE_COMPARISON_PLOT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            ANGLE_COMPARISON_PLOT_PATH,
            dpi=300,
            bbox_inches="tight",
        )
        print(
            "Angle-probability comparison saved to:\n"
            f"{ANGLE_COMPARISON_PLOT_PATH.resolve()}"
        )

    return fig


def plot_angular_correlation_comparison(results):
    """Plot all isotropy-corrected angular-correlation curves."""
    fig, ax = plt.subplots(figsize=(11, 7))
    maximum_value = 1.0

    for result in results:
        ax.plot(
            result["angle_deg"],
            result["angular_correlation"],
            linewidth=2.0,
            label=make_curve_label(result),
        )
        maximum_value = max(
            maximum_value,
            float(np.max(result["angular_correlation"])),
        )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.2,
        label="Isotropic reference",
    )

    add_reference_angle_lines(ax)
    apply_y_axis_scale(ax, Y_AXIS_SCALE, maximum_value)

    ax.set_xlabel(r"Relative angle $\theta$ / degrees")
    ax.set_ylabel(r"Angular correlation $g_\theta(\theta)$")
    ax.set_title(
        "Comparison of Isotropy-Corrected Relative-Angle Distributions",
        pad=12,
    )
    ax.set_xlim(0.0, 180.0)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()

    if SAVE_PLOT:
        ANGULAR_CORRELATION_COMPARISON_PLOT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            ANGULAR_CORRELATION_COMPARISON_PLOT_PATH,
            dpi=300,
            bbox_inches="tight",
        )
        print(
            "Angular-correlation comparison saved to:\n"
            f"{ANGULAR_CORRELATION_COMPARISON_PLOT_PATH.resolve()}"
        )

    return fig


# ================================================================
# MAIN PROGRAM
# ================================================================


def main():
    if not TRAJECTORIES:
        raise ValueError(
            "TRAJECTORIES must contain at least one trajectory."
        )

    if FRAME_STRIDE < 1:
        raise ValueError("FRAME_STRIDE must be at least 1.")

    if DIMENSION not in (2, 3):
        raise ValueError("DIMENSION must be 2 or 3.")

    results = []

    for trajectory in TRAJECTORIES:
        result = analyze_trajectory(trajectory)
        results.append(result)

        if SAVE_CSV:
            save_angle_csv(result)

        if SAVE_PLOT and SAVE_INDIVIDUAL_RDF_DIAGNOSTICS:
            save_individual_rdf_diagnostic(result)

    print()
    print("=" * 72)
    print("Analysis summary")
    print("=" * 72)

    for result in results:
        print(
            f"{result['label']}: "
            f"cutoff = {result['neighbor_cutoff_nm']:.5f} nm, "
            f"mean coordination = "
            f"{result['mean_coordination_number']:.3f}, "
            f"angles = {int(result['angle_counts'].sum())}"
        )

    figures = []

    if SAVE_COMBINED_RDF_DIAGNOSTIC:
        rdf_figure = plot_combined_rdf_diagnostic(results)
        if rdf_figure is not None:
            figures.append(rdf_figure)

    figures.append(
        plot_angle_probability_comparison(results)
    )
    figures.append(
        plot_angular_correlation_comparison(results)
    )

    plt.show()


if __name__ == "__main__":
    main()