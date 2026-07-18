"""
Cluster count directly from XYZ trajectories
=============================================

This script reads several XYZ trajectories directly, assigns particles to
clusters in every selected frame, and plots the number of clusters as a
function of simulation time. No intermediate CSV files are created.

Cluster definition
------------------
The optimal Lennard-Jones distance is

    r_opt = 2^(1/6) * sigma

and the cluster cutoff used here is

    r_cutoff = 1.5 * r_opt.

Two particles belong to the same cluster when they are connected through
particle pairs with a periodic minimum-image distance <= r_cutoff.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


# ================================================================
# PROJECT DIRECTORY
# ================================================================

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"


# ================================================================
# INPUT DATASETS
# ================================================================

DATASETS = [
    {
        "label": "Simulation at 5 K",
        "xyz_path": RESULTS_DIR / "Long_sim_5" / "my_simulation_pos.xyz",
        "box_length_nm": 6.0,
    },
    {
        "label": "Simulation at 80 K",
        "xyz_path": RESULTS_DIR / "Long_sim" / "my_simulation_pos.xyz",
        "box_length_nm": 6.0,
    },
    {
        "label": "Simulation at 300 K",
        "xyz_path": RESULTS_DIR / "Long_Sim_300" / "my_simulation_pos.xyz",
        "box_length_nm": 6.0,
    },
]


# ================================================================
# TRAJECTORY SETTINGS
# ================================================================

# Time between two consecutive XYZ frames
FRAME_DT_PS = 0.001

# Ignore the initial equilibration phase
START_FRAME = 4000

# None means: analyze until the final frame
STOP_FRAME = None

# Analyze only every nth frame
FRAME_STRIDE = 20

# The MD code writes XYZ coordinates in angstrom
XYZ_COORDINATES_IN_ANGSTROM = True


# ================================================================
# CLUSTER SETTINGS
# ================================================================

SIGMA_NM = 0.34

OPTIMAL_DISTANCE_NM = 2.0 ** (1.0 / 6.0) * SIGMA_NM
CLUSTER_CUTOFF_NM = 1.5 * OPTIMAL_DISTANCE_NM

# Connected components smaller than this are not counted as clusters.
# Set to 2 to exclude isolated particles.
MIN_CLUSTER_SIZE = 2


# ================================================================
# OUTPUT SETTINGS
# ================================================================

SAVE_PLOT = True
PLOT_PATH = PROJECT_DIR / "cluster_count_vs_time_from_xyz.png"


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
# UNION-FIND FOR CONNECTED COMPONENTS
# ================================================================

class UnionFind:
    def __init__(self, n_items):
        self.parent = np.arange(n_items, dtype=int)
        self.size = np.ones(n_items, dtype=int)

    def find(self, item):
        root = item

        while self.parent[root] != root:
            root = self.parent[root]

        while self.parent[item] != item:
            next_item = self.parent[item]
            self.parent[item] = root
            item = next_item

        return root

    def union(self, item_a, item_b):
        root_a = self.find(item_a)
        root_b = self.find(item_b)

        if root_a == root_b:
            return

        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]


# ================================================================
# XYZ READER
# ================================================================

def iter_selected_xyz_frames(
    xyz_path,
    start_frame=0,
    stop_frame=None,
    frame_stride=1,
):
    """
    Stream selected frames from a multi-frame XYZ trajectory.

    Yields
    ------
    frame_index : int
        Original frame number.

    positions : np.ndarray
        Coordinates with shape (n_particles, 3).
    """

    xyz_path = Path(xyz_path)

    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ trajectory not found:\n{xyz_path}")

    if start_frame < 0:
        raise ValueError("START_FRAME must not be negative.")

    if stop_frame is not None and stop_frame <= start_frame:
        raise ValueError("STOP_FRAME must be greater than START_FRAME.")

    if frame_stride < 1:
        raise ValueError("FRAME_STRIDE must be at least 1.")

    frame_index = 0
    expected_particle_count = None

    with xyz_path.open("r", encoding="utf-8") as xyz_file:
        while True:
            particle_count_line = xyz_file.readline()

            if particle_count_line == "":
                break

            particle_count_line = particle_count_line.strip()

            if not particle_count_line:
                continue

            try:
                n_particles = int(particle_count_line)
            except ValueError as error:
                raise ValueError(
                    f"Invalid particle-count line before frame {frame_index} "
                    f"in:\n{xyz_path}"
                ) from error

            if expected_particle_count is None:
                expected_particle_count = n_particles
            elif n_particles != expected_particle_count:
                raise ValueError(
                    f"The particle number changes between frames in:\n{xyz_path}"
                )

            comment_line = xyz_file.readline()
            if comment_line == "":
                raise ValueError(
                    f"Incomplete comment line in frame {frame_index} of:\n{xyz_path}"
                )

            use_frame = (
                frame_index >= start_frame
                and (stop_frame is None or frame_index < stop_frame)
                and (frame_index - start_frame) % frame_stride == 0
            )

            if use_frame:
                positions = np.empty((n_particles, 3), dtype=float)

            for particle_index in range(n_particles):
                particle_line = xyz_file.readline()

                if particle_line == "":
                    raise ValueError(
                        f"Incomplete frame {frame_index} in:\n{xyz_path}"
                    )

                if use_frame:
                    columns = particle_line.split()

                    if len(columns) < 4:
                        raise ValueError(
                            f"Invalid particle line in frame {frame_index} "
                            f"of:\n{xyz_path}"
                        )

                    try:
                        positions[particle_index] = (
                            float(columns[1]),
                            float(columns[2]),
                            float(columns[3]),
                        )
                    except ValueError as error:
                        raise ValueError(
                            f"Invalid coordinates in frame {frame_index} "
                            f"of:\n{xyz_path}"
                        ) from error

            if use_frame:
                yield frame_index, positions

            frame_index += 1

            if stop_frame is not None and frame_index >= stop_frame:
                break


# ================================================================
# CLUSTER CALCULATION
# ================================================================

def calculate_cluster_sizes(positions_nm, box_length_nm, cutoff_nm):
    """
    Return the sizes of all connected components in one frame.
    """

    positions_nm = np.asarray(positions_nm, dtype=float)

    if positions_nm.ndim != 2 or positions_nm.shape[1] != 3:
        raise ValueError("positions_nm must have shape (n_particles, 3).")

    if not np.all(np.isfinite(positions_nm)):
        raise ValueError("The positions contain NaN or infinite values.")

    if box_length_nm <= 0:
        raise ValueError("box_length_nm must be greater than zero.")

    if cutoff_nm <= 0:
        raise ValueError("cutoff_nm must be greater than zero.")

    if cutoff_nm > box_length_nm / 2.0:
        raise ValueError(
            "The cluster cutoff must not exceed half the box length."
        )

    n_particles = positions_nm.shape[0]

    if n_particles == 0:
        return np.array([], dtype=int)

    wrapped_positions = np.mod(positions_nm, box_length_nm)

    tree = cKDTree(
        wrapped_positions,
        boxsize=box_length_nm,
    )

    pairs = tree.query_pairs(
        r=cutoff_nm,
        output_type="ndarray",
    )

    union_find = UnionFind(n_particles)

    for particle_a, particle_b in pairs:
        union_find.union(int(particle_a), int(particle_b))

    roots = np.fromiter(
        (union_find.find(index) for index in range(n_particles)),
        dtype=int,
        count=n_particles,
    )

    _, cluster_sizes = np.unique(roots, return_counts=True)

    return cluster_sizes


def analyze_trajectory(xyz_path, box_length_nm, label):
    """
    Calculate the number of clusters as a function of time.
    """

    time_values = []
    cluster_counts = []
    largest_cluster_sizes = []

    selected_frame_counter = 0

    for frame_index, positions in iter_selected_xyz_frames(
        xyz_path=xyz_path,
        start_frame=START_FRAME,
        stop_frame=STOP_FRAME,
        frame_stride=FRAME_STRIDE,
    ):
        if XYZ_COORDINATES_IN_ANGSTROM:
            positions_nm = positions * 0.1
        else:
            positions_nm = positions

        cluster_sizes = calculate_cluster_sizes(
            positions_nm=positions_nm,
            box_length_nm=box_length_nm,
            cutoff_nm=CLUSTER_CUTOFF_NM,
        )

        number_of_clusters = int(
            np.count_nonzero(cluster_sizes >= MIN_CLUSTER_SIZE)
        )

        largest_cluster_size = (
            int(np.max(cluster_sizes)) if cluster_sizes.size else 0
        )

        time_values.append(frame_index * FRAME_DT_PS)
        cluster_counts.append(number_of_clusters)
        largest_cluster_sizes.append(largest_cluster_size)

        selected_frame_counter += 1

        if selected_frame_counter == 1 or selected_frame_counter % 100 == 0:
            print(
                f"{label}: analyzed {selected_frame_counter} selected frames "
                f"(original frame {frame_index})"
            )

    if not time_values:
        raise ValueError(
            f"No frames were selected for {label}. "
            "Check START_FRAME, STOP_FRAME, and FRAME_STRIDE."
        )

    return (
        np.asarray(time_values, dtype=float),
        np.asarray(cluster_counts, dtype=int),
        np.asarray(largest_cluster_sizes, dtype=int),
    )


# ================================================================
# VALIDATION
# ================================================================

def validate_datasets():
    if not DATASETS:
        raise ValueError("DATASETS must contain at least one trajectory.")

    print("=" * 72)
    print("CLUSTER ANALYSIS SETTINGS")
    print("=" * 72)
    print(f"Optimal LJ distance:  {OPTIMAL_DISTANCE_NM:.6f} nm")
    print(f"Cluster cutoff:       {CLUSTER_CUTOFF_NM:.6f} nm")
    print(f"Minimum cluster size: {MIN_CLUSTER_SIZE}")
    print(f"Frame time step:      {FRAME_DT_PS:.6f} ps")
    print(f"Selected frames:      {START_FRAME}:{STOP_FRAME}:{FRAME_STRIDE}")
    print("=" * 72)

    labels = set()

    for dataset in DATASETS:
        label = dataset["label"]
        xyz_path = Path(dataset["xyz_path"])
        box_length_nm = dataset["box_length_nm"]

        if label in labels:
            raise ValueError(f"Duplicate dataset label: {label}")

        labels.add(label)

        if box_length_nm <= 0:
            raise ValueError(
                f"{label}: box_length_nm must be greater than zero."
            )

        print()
        print(f"{label}:")
        print(f"  Path:       {xyz_path}")
        print(f"  File found: {xyz_path.exists()}")
        print(f"  Box length: {box_length_nm} nm")

        if not xyz_path.exists():
            raise FileNotFoundError(
                f"XYZ trajectory not found for {label}:\n{xyz_path}"
            )


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():
    validate_datasets()

    fig, ax = plt.subplots(figsize=(12, 7))

    for dataset in DATASETS:
        label = dataset["label"]

        print()
        print("=" * 72)
        print(f"Analyzing {label}")
        print("=" * 72)

        time_ps, cluster_counts, largest_cluster_sizes = analyze_trajectory(
            xyz_path=dataset["xyz_path"],
            box_length_nm=dataset["box_length_nm"],
            label=label,
        )

        print(f"{label}: analyzed frames = {len(time_ps)}")
        print(f"{label}: mean cluster count = {np.mean(cluster_counts):.3f}")
        print(f"{label}: maximum cluster count = {np.max(cluster_counts)}")
        print(
            f"{label}: mean largest-cluster size = "
            f"{np.mean(largest_cluster_sizes):.3f}"
        )

        ax.plot(
            time_ps,
            cluster_counts,
            linewidth=1.8,
            label=label,
        )

    ax.set_xlabel("Time / ps")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Cluster Count as a Function of Time", pad=15)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()

    if SAVE_PLOT:
        fig.savefig(
            PLOT_PATH,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"\nPlot saved to:\n{PLOT_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
