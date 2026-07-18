"""
Compare radial distribution functions on a logarithmic y-axis
==============================================================

PURPOSE
-------
This script calculates and plots radial distribution functions g(r)
from several XYZ trajectories using a logarithmic y-axis.

The logarithmic scale makes weak peaks and small non-zero values easier
to inspect when strong first-neighbor peaks dominate a linear plot.

Place this file in the project root together with:

    rdf_analysis.py

Expected folder structure:

    md-project/
    ├── compare_rdf_log.py
    ├── rdf_analysis.py
    └── results/
        ├── Long_sim_5/
        │   └── my_simulation_pos.xyz
        ├── Long_sim/
        │   └── my_simulation_pos.xyz
        └── Long_sim_300/
            └── my_simulation_pos.xyz

USAGE
-----
Run:

    python compare_rdf_log.py

IMPORTANT
---------
A logarithmic axis cannot display zero or negative values. RDF values
with g(r) <= 0 are therefore replaced by NaN only for plotting. The
calculated RDF data itself is not changed.
"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rdf_analysis import analyze_xyz_trajectory


# ================================================================
# PROJECT DIRECTORY
# ================================================================

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"


# ================================================================
# DATASETS
# ================================================================

DATASETS = [
    {
        "label": "Simulation at 5 K",
        "xyz_path": RESULTS_DIR / "Long_sim_5" / "my_simulation_pos.xyz",
        "box_length_nm": 6.0,
        "start_frame": 4000,
        "stop_frame": None,
        "frame_stride": 20,
        "coordinates_in_angstrom": True,
    },
    {
        "label": "Simulation at 80 K",
        "xyz_path": RESULTS_DIR / "Long_sim" / "my_simulation_pos.xyz",
        "box_length_nm": 6.0,
        "start_frame": 4000,
        "stop_frame": None,
        "frame_stride": 20,
        "coordinates_in_angstrom": True,
    },
    {
        "label": "Simulation at 300 K",
        "xyz_path": RESULTS_DIR / "Long_sim_300" / "my_simulation_pos.xyz",
        "box_length_nm": 6.0,
        "start_frame": 4000,
        "stop_frame": None,
        "frame_stride": 20,
        "coordinates_in_angstrom": True,
    },
]


# ================================================================
# RDF SETTINGS
# ================================================================

N_BINS = 200

# Use one common maximum distance for all trajectories:
#
#     r_max = 0.5 * smallest box length
#
USE_COMMON_R_MAX = True

# Used only when USE_COMMON_R_MAX is False.
R_MAX_NM = None

# Lennard-Jones sigma parameter in nm
SIGMA_NM = 0.34

SHOW_IDEAL_GAS_REFERENCE = True
SHOW_LJ_MINIMUM = True

# Lower and upper limits of the logarithmic y-axis.
# Set either value to None for automatic scaling.
LOG_Y_MIN = 1.0e-1
LOG_Y_MAX = None

SAVE_PLOT = True
PLOT_PATH = PROJECT_DIR / "rdf_comparison_logarithmic.png"

SAVE_INDIVIDUAL_CSV_FILES = True
CSV_OUTPUT_DIR = PROJECT_DIR / "rdf_log_comparison_data"


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
# VALIDATION
# ================================================================

def validate_datasets(datasets):
    """
    Validate dataset settings and check all XYZ paths.
    """

    if not datasets:
        raise ValueError(
            "DATASETS must contain at least one trajectory."
        )

    required_keys = {
        "label",
        "xyz_path",
        "box_length_nm",
        "start_frame",
        "stop_frame",
        "frame_stride",
        "coordinates_in_angstrom",
    }

    labels = set()

    print("=" * 72)
    print("PROJECT AND INPUT PATHS")
    print("=" * 72)
    print(f"Project directory:\n{PROJECT_DIR}\n")

    for dataset_index, dataset in enumerate(datasets):
        missing_keys = required_keys - set(dataset)

        if missing_keys:
            raise KeyError(
                f"Dataset {dataset_index} is missing: "
                f"{sorted(missing_keys)}"
            )

        label = dataset["label"]
        xyz_path = Path(dataset["xyz_path"])

        if label in labels:
            raise ValueError(
                f"Duplicate dataset label: {label}"
            )

        labels.add(label)

        if dataset["box_length_nm"] <= 0:
            raise ValueError(
                f"{label}: box_length_nm must be greater than zero."
            )

        if dataset["frame_stride"] < 1:
            raise ValueError(
                f"{label}: frame_stride must be at least 1."
            )

        if dataset["start_frame"] < 0:
            raise ValueError(
                f"{label}: start_frame must not be negative."
            )

        print(f"{label}:")
        print(f"  Path:   {xyz_path}")
        print(f"  Exists: {xyz_path.exists()}\n")

        if not xyz_path.exists():
            raise FileNotFoundError(
                f"XYZ trajectory not found for {label}:\n"
                f"{xyz_path}\n\n"
                "Check the folder and file names in the results directory."
            )

    print("=" * 72)


# ================================================================
# OUTPUT HELPERS
# ================================================================

def safe_filename(text):
    """
    Convert a dataset label into a safe file-name component.
    """

    characters = []

    for character in text.lower():
        if character.isalnum():
            characters.append(character)
        else:
            characters.append("_")

    name = "".join(characters)

    while "__" in name:
        name = name.replace("__", "_")

    return name.strip("_")


def save_result_to_csv(result, label):
    """
    Save one RDF result to a CSV file.
    """

    CSV_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        CSV_OUTPUT_DIR
        / f"rdf_{safe_filename(label)}.csv"
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
        output_path,
        output_data,
        delimiter=",",
        header="r_nm,g_r,coordination_number,pair_count",
        comments="",
    )

    print(
        f"{label}: CSV saved to:\n"
        f"{output_path}"
    )


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():
    validate_datasets(DATASETS)

    if USE_COMMON_R_MAX:
        common_r_max_nm = 0.5 * min(
            dataset["box_length_nm"]
            for dataset in DATASETS
        )
    else:
        common_r_max_nm = R_MAX_NM

    results = []

    for dataset in DATASETS:
        print()
        print("=" * 72)
        print(f"Analyzing: {dataset['label']}")
        print("=" * 72)

        result = analyze_xyz_trajectory(
            xyz_path=dataset["xyz_path"],
            box_length_nm=dataset["box_length_nm"],
            start_frame=dataset["start_frame"],
            stop_frame=dataset["stop_frame"],
            frame_stride=dataset["frame_stride"],
            n_bins=N_BINS,
            r_max_nm=common_r_max_nm,
            coordinates_in_angstrom=(
                dataset["coordinates_in_angstrom"]
            ),
            progress_label=dataset["label"],
        )

        result["label"] = dataset["label"]
        results.append(result)

        if SAVE_INDIVIDUAL_CSV_FILES:
            save_result_to_csv(
                result=result,
                label=dataset["label"],
            )

    # ------------------------------------------------------------
    # Logarithmic RDF plot
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for result in results:
        g_r_for_plot = np.where(
            result["g_r"] > 0.0,
            result["g_r"],
            np.nan,
        )

        ax.plot(
            result["r_nm"],
            g_r_for_plot,
            linewidth=2.0,
            label=result["label"],
        )

    if SHOW_IDEAL_GAS_REFERENCE:
        ax.axhline(
            1.0,
            linestyle="--",
            linewidth=1.5,
            label="Uniform ideal-gas reference",
        )

    if SHOW_LJ_MINIMUM:
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

    ax.set_yscale("log")

    if LOG_Y_MIN is not None or LOG_Y_MAX is not None:
        ax.set_ylim(
            bottom=LOG_Y_MIN,
            top=LOG_Y_MAX,
        )

    ax.set_xlabel(
        "Particle distance $r$ / nm"
    )

    ax.set_ylabel(
        r"Radial distribution function $g(r)$"
        "\n(logarithmic scale)"
    )

    ax.set_title(
        "Logarithmic Comparison of Radial Distribution Functions",
        pad=15,
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_FONT_SIZE,
    )

    # Major and minor grid lines are useful on a logarithmic axis.
    ax.grid(
        True,
        which="major",
        alpha=0.45,
    )

    ax.grid(
        True,
        which="minor",
        alpha=0.20,
        linestyle=":",
    )

    ax.legend()
    fig.tight_layout()

    if SAVE_PLOT:
        fig.savefig(
            PLOT_PATH,
            dpi=300,
            bbox_inches="tight",
        )

        print()
        print(
            f"Logarithmic RDF plot saved to:\n"
            f"{PLOT_PATH}"
        )

    plt.show()


if __name__ == "__main__":
    main()
