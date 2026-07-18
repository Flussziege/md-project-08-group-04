"""
Compare radial distribution functions from multiple XYZ trajectories
====================================================================

Place this file in the project root together with rdf_analysis.py.

Expected structure:

    md-project/
    ├── compare_rdf.py
    ├── rdf_analysis.py
    └── results/
        ├── Long_sim_5/my_simulation_pos.xyz
        ├── Long_sim/my_simulation_pos.xyz
        └── Long_sim_300/my_simulation_pos.xyz
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
USE_COMMON_R_MAX = True
R_MAX_NM = None

SIGMA_NM = 0.34

SHOW_IDEAL_GAS_REFERENCE = True
SHOW_LJ_MINIMUM = True

SAVE_COMBINED_PLOT = True
COMBINED_PLOT_PATH = PROJECT_DIR / "rdf_comparison.png"

SAVE_INDIVIDUAL_CSV_FILES = True
CSV_OUTPUT_DIR = PROJECT_DIR / "rdf_comparison_data"


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
    if not datasets:
        raise ValueError("DATASETS must contain at least one trajectory.")

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
                f"Dataset {dataset_index} is missing: {sorted(missing_keys)}"
            )

        label = dataset["label"]
        xyz_path = Path(dataset["xyz_path"])

        if label in labels:
            raise ValueError(f"Duplicate dataset label: {label}")

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
                "Check the folder name and file name in the results directory."
            )

    print("=" * 72)


# ================================================================
# OUTPUT HELPERS
# ================================================================

def safe_filename(text):
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
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = CSV_OUTPUT_DIR / f"rdf_{safe_filename(label)}.csv"

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

    print(f"{label}: CSV saved to:\n{output_path}")


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():
    validate_datasets(DATASETS)

    if USE_COMMON_R_MAX:
        common_r_max_nm = 0.5 * min(
            dataset["box_length_nm"] for dataset in DATASETS
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
            coordinates_in_angstrom=dataset["coordinates_in_angstrom"],
            progress_label=dataset["label"],
        )

        result["label"] = dataset["label"]
        results.append(result)

        if SAVE_INDIVIDUAL_CSV_FILES:
            save_result_to_csv(result, dataset["label"])

    fig, ax = plt.subplots(figsize=(12, 7))

    for result in results:
        ax.plot(
            result["r_nm"],
            result["g_r"],
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
        lj_minimum_nm = 2.0 ** (1.0 / 6.0) * SIGMA_NM

        ax.axvline(
            lj_minimum_nm,
            linestyle=":",
            linewidth=1.5,
            label=r"$r_\mathrm{min}=2^{1/6}\sigma$",
        )

    ax.set_xlabel("Particle distance $r$ / nm")
    ax.set_ylabel(r"Radial distribution function $g(r)$")

    ax.set_title(
        "Comparison of Radial Distribution Functions",
        pad=15,
    )

    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()

    if SAVE_COMBINED_PLOT:
        fig.savefig(
            COMBINED_PLOT_PATH,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"\nCombined RDF plot saved to:\n{COMBINED_PLOT_PATH}"
        )

    plt.show()


if __name__ == "__main__":
    main()
