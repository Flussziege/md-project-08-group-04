from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from cluster_functions import read_xyz_trajectory


# ================================================================
#   I N P U T   P A R A M E T E R S
# ================================================================

XYZ_PATH = Path(
    r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04\results\2026-07-15_19-30-00-5K\my_simulation_pos-5K.xyz"
)

# Side length of the cubic simulation box in nm
BOX_LENGTH_NM = 6.0

# Lennard-Jones parameter in nm
SIGMA_NM = 0.34

# 3: Angles between three-dimensional bond vectors
# 2: Use only the x and y components
DIMENSION = 3

# Discard the first part of the simulation as an equilibration phase.
START_FRAME = 4000

# None means: continue to the final frame.
STOP_FRAME = None

# Use only every nth frame.
FRAME_STRIDE = 20

# Number of angular bins between 0° and 180°.
N_ANGLE_BINS = 180

# Neighbors are considered only within this radius.
#
# None:
#   The radius is automatically chosen as the first RDF minimum after
#   the first maximum.
#
# Example of a fixed value:
#   NEIGHBOR_CUTOFF_NM = 0.52
NEIGHBOR_CUTOFF_NM = None

# RDF parameters used when the neighbor cutoff is determined
# automatically.
N_RDF_BINS = 300
RDF_R_MAX_NM = None
RDF_SMOOTHING_SIGMA_BINS = 2.0

# The XYZ file contains coordinates in angstroms.
# 1 Å = 0.1 nm
XYZ_COORDINATES_IN_ANGSTROM = True

# Save output
SAVE_CSV = True
SAVE_PLOT = True
SAVE_RDF_DIAGNOSTIC = True

# Plot font sizes
TITLE_FONTSIZE = 17
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 11

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
#   H E L P E R   F U N C T I O N S
# ================================================================


def validate_positions(positions):
    """Validate the shape of the trajectory array."""
    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 3:
        raise ValueError(
            "positions must have the shape "
            "(n_frames, n_particles, 3) besitzen."
        )

    if positions.shape[2] != 3:
        raise ValueError(
            "The final dimension of positions must be 3."
        )

    if positions.shape[0] == 0:
        raise ValueError("No frames were provided.")

    if positions.shape[1] < 3:
        raise ValueError(
            "At least three particles are required for relative angles."
        )

    return positions


def minimum_image(displacements, box_length):
    """Apply the minimum-image convention."""
    return displacements - box_length * np.rint(
        displacements / box_length
    )


# ================================================================
#   R D F   F O R   N E I G H B O R   C U T O F F
# ================================================================


def calculate_rdf(
    positions,
    box_length,
    n_bins=300,
    r_max=None,
):
    """
    Calculate the radial distribution function g(r).

    It is mainly used here to locate the first minimum
    after the first peak and therefore the first coordination shell.
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
                f"RDF: frame {frame_index + 1} of {n_frames}"
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
    Determine the first RDF minimum after the first RDF maximum.

    The search is restricted to the physically relevant region around
    the first Lennard-Jones coordination shell.
    """
    r_nm = np.asarray(r_nm, dtype=float)
    g_r = np.asarray(g_r, dtype=float)

    if r_nm.shape != g_r.shape:
        raise ValueError("r_nm and g_r must have the same shape.")

    if sigma_nm <= 0:
        raise ValueError("sigma_nm must be greater than zero.")

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

    # Search for local minima after the peak.
    minimum_candidates, _ = find_peaks(-smoothed_g_r)
    minimum_candidates = minimum_candidates[
        (minimum_candidates > first_peak_index)
        & (r_nm[minimum_candidates] <= 2.5 * sigma_nm)
    ]

    if minimum_candidates.size > 0:
        first_minimum_index = minimum_candidates[0]
    else:
        # Robust fallback: smallest g(r) in a region after the peak.
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
#   R E L A T I V E   A N G L E   D I S T R I B U T I O N
# ================================================================


def calculate_relative_angle_probability(
    positions,
    box_length,
    neighbor_cutoff,
    n_angle_bins=180,
    dimension=3,
):
    """
    Calculate the distribution of angles between two neighbor bonds.

    For a central particle i and two neighbors j and k, calculate:

        theta_jik = arccos(
            r_ij dot r_ik / (|r_ij| |r_ik|)
        )

    All unordered neighbor pairs j < k are counted.

    Parameters
    ----------
    positions : np.ndarray
        Shape (n_frames, n_particles, 3).

    box_length : float
        Side length of the periodic cubic box.

    neighbor_cutoff : float
        Maximum distance of a neighbor from the central particle.
        Typically the first minimum of the RDF.

    n_angle_bins : int
        Number of angular bins from 0° to 180°.

    dimension : int
        3 for three-dimensional angles, 2 for an analysis in the xy plane.

    Returns
    -------
    angle_centers_deg : np.ndarray
        Centers of the angular bins.

    probability_density_per_degree : np.ndarray
        Normalized probability density P(theta).
        The integral from 0° to 180° is approximately 1.

    isotropic_density_per_degree : np.ndarray
        Expected reference density for randomly oriented bonds.

    angular_correlation : np.ndarray
        P(theta) divided by the isotropic reference.
        A value of 1 corresponds to an isotropic distribution.

    angle_histogram : np.ndarray
        Absolute number of angles per bin.

    mean_coordination_number : float
        Mean number of neighbors within the cutoff.
    """
    positions = validate_positions(positions)

    if dimension not in (2, 3):
        raise ValueError("dimension muss 2 oder 3 sein.")

    if box_length <= 0:
        raise ValueError("box_length must be greater than zero.")

    if not 0 < neighbor_cutoff <= box_length / 2.0:
        raise ValueError(
            "neighbor_cutoff must be greater than zero and no larger than L/2."
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
            # The KD-tree includes the central particle itself.
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

            # At least two neighbors are required to define an angle.
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
                f"Angles: frame {frame_index + 1} of {n_frames}"
            )

    total_angle_count = int(angle_histogram.sum())
    if total_angle_count == 0:
        raise RuntimeError(
            "No angles were found. "
            "Check the neighbor cutoff and the selected frames."
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
        # Here, the exact probability per histogram bin is
        # integrated and then divided by the bin width.
        isotropic_probability_per_bin = 0.5 * (
            np.cos(angle_edges_rad[:-1])
            - np.cos(angle_edges_rad[1:])
        )
    else:
        # In 2D, the folded relative angle on [0, pi]
        # is uniformly distributed for random directions.
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
#   R E A D   X Y Z   T R A J E C T O R Y
# ================================================================

positions, atom_names = read_xyz_trajectory(XYZ_PATH)

print(f"Complete trajectory: {positions.shape}")

if XYZ_COORDINATES_IN_ANGSTROM:
    positions = positions * 0.1

selected_positions = positions[
    START_FRAME:STOP_FRAME:FRAME_STRIDE
]

if selected_positions.shape[0] == 0:
    raise ValueError(
        "The frame selection is empty. Check START_FRAME, "
        "STOP_FRAME und FRAME_STRIDE."
    )

print(
    f"Frames used for the analysis: "
    f"{selected_positions.shape[0]}"
)
print(
    f"Particles per frame: "
    f"{selected_positions.shape[1]}"
)


# ================================================================
#   D E T E R M I N E   N E I G H B O R   C U T O F F
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
        "Automatically determined neighbor cutoff "
        f"(first RDF minimum): {neighbor_cutoff_nm:.5f} nm"
    )

    if SAVE_RDF_DIAGNOSTIC:
        plt.figure(figsize=(9, 6))
        plt.plot(r_nm, g_r, linewidth=1.0, label=r"$g(r)$")
        plt.plot(
            r_nm,
            smoothed_g_r,
            linewidth=1.5,
            label="smoothed RDF",
        )
        plt.axvline(
            r_nm[first_peak_index],
            linestyle=":",
            linewidth=1.2,
            label="first peak",
        )
        plt.axvline(
            neighbor_cutoff_nm,
            linestyle="--",
            linewidth=1.2,
            label="neighbor cutoff: first minimum",
        )
        plt.xlabel(r"distance $r$ / nm", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel(r"radial distribution function $g(r)$", fontsize=AXIS_LABEL_FONTSIZE)
        plt.title("Determination of the First Coordination Shell", fontsize=TITLE_FONTSIZE, pad=12)
        plt.grid(True)
        plt.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.tight_layout()
        plt.savefig(
            RDF_DIAGNOSTIC_PATH,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(
            "RDF diagnostic saved to: "
            f"{RDF_DIAGNOSTIC_PATH}"
        )
else:
    neighbor_cutoff_nm = float(NEIGHBOR_CUTOFF_NM)
    print(
        "Manually specified neighbor cutoff: "
        f"{neighbor_cutoff_nm:.5f} nm"
    )


# ================================================================
#   C A L C U L A T E   R E L A T I V E   A N G L E S
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

print(f"Counted relative angles: {angle_counts.sum()}")
print(
    "Mean coordination number within the cutoff: "
    f"{mean_coordination_number:.3f}"
)


# ================================================================
#   S A V E   C S V
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
        "Angle CSV saved to: "
        f"{ANGLE_CSV_PATH}"
    )


# ================================================================
#   P L O T   A N G L E   D I S T R I B U T I O N
# ================================================================

plt.figure(figsize=(9, 6))
plt.plot(
    angle_deg,
    probability_density,
    linewidth=1.5,
    label=r"measured $P(\theta)$",
)
plt.plot(
    angle_deg,
    isotropic_density,
    linestyle="--",
    linewidth=1.2,
    label=(
        r"isotropic reference $P_0(\theta)$"
        if DIMENSION == 3
        else "uniform 2D reference"
    ),
)

for reference_angle in (60.0, 90.0, 120.0, 180.0):
    plt.axvline(
        reference_angle,
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
    )

plt.xlabel(r"relative angle $\theta$ / degrees", fontsize=AXIS_LABEL_FONTSIZE)
plt.ylabel(r"probability density $P(\theta)$ / degree$^{-1}$", fontsize=AXIS_LABEL_FONTSIZE)
plt.title(
    "Relative Angle Distribution in the First Coordination Shell\n"
    f"Neighbor cutoff = {neighbor_cutoff_nm:.4f} nm",
    fontsize=TITLE_FONTSIZE,
    pad=12,
)
plt.xlim(0.0, 180.0)
plt.grid(True)
plt.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.tight_layout()

if SAVE_PLOT:
    plt.savefig(
        ANGLE_PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    print(
        "Angle-distribution plot saved to: "
        f"{ANGLE_PLOT_PATH}"
    )

plt.show()


# ================================================================
#   I S O T R O P Y - C O R R E C T E D   D I S T R I B U T I O N
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
    label="isotropic reference",
)

for reference_angle in (60.0, 90.0, 120.0, 180.0):
    plt.axvline(
        reference_angle,
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
    )

plt.xlabel(r"relative angle $\theta$ / degrees", fontsize=AXIS_LABEL_FONTSIZE)
plt.ylabel(r"angular correlation $g_\theta(\theta)$", fontsize=AXIS_LABEL_FONTSIZE)
plt.title(
    f"Isotropy-Corrected Relative Angle Distribution\n",
    fontsize=TITLE_FONTSIZE,
    pad=12,
)
plt.xlim(0.0, 180.0)
plt.grid(True)
plt.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.tight_layout()

if SAVE_PLOT:
    plt.savefig(
        ANGULAR_CORRELATION_PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    print(
        "Angular-correlation plot saved to: "
        f"{ANGULAR_CORRELATION_PLOT_PATH}"
    )

plt.show()