from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# INPUT
# ================================================================

# Project folder in which this Python file is located.
# This makes the paths work reliably on macOS,
# regardless of the current Terminal directory.
PROJECT_DIR = Path(__file__).resolve().parent

CSV_FILES = [
    (
        PROJECT_DIR
        / "results"
        / "Long_sim_5"
        / "my_simulation_pos_cluster_const_ids.csv"
    ),
    (
        PROJECT_DIR
        / "results"
        / "Long_sim"
        / "my_simulation_pos_cluster_const_ids.csv"
    ),
    (
        PROJECT_DIR
        / "results"
        / "Long_Sim_300"
        / "my_simulation_pos_cluster_const_ids.csv"
    ),
]

# If True, save the plot
SAVE_PLOT = True

PLOT_PATH = (
    PROJECT_DIR
    / "cluster_count_vs_time.png"
)

# Optional: save the processed time series as CSV
SAVE_SUMMARY_CSV = True


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
# HELPERS
# ================================================================

def convert_to_bool(series):
    """
    Converts a column like True/False or 'True'/'False' to boolean.
    """
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )


def calculate_cluster_count_vs_time(df):
    """
    Calculate the number of clusters as a function of time.

    Definition used here:
    - For each frame, only particles with is_clustered == True are used.
    - The cluster count is the number of unique cluster_id values
      among those clustered particles.

    Returns
    -------
    summary : pandas.DataFrame
        Columns:
        - frame
        - time_ps
        - n_clusters
        - n_clustered_particles
        - largest_cluster_size
    """

    required_columns = {
        "frame",
        "time_ps",
        "particle",
        "cluster_id",
        "cluster_size",
        "is_clustered",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"The CSV file is missing these required columns: "
            f"{sorted(missing_columns)}"
        )

    # Make sure is_clustered is really boolean
    df = df.copy()
    df["is_clustered"] = convert_to_bool(df["is_clustered"])

    # All available frames/times
    all_frames = (
        df[["frame", "time_ps"]]
        .drop_duplicates()
        .sort_values(["frame", "time_ps"])
        .reset_index(drop=True)
    )

    # Keep only clustered particles
    clustered_df = df[df["is_clustered"] == True].copy()

    # If no clustered particles exist at all
    if clustered_df.empty:
        summary = all_frames.copy()
        summary["n_clusters"] = 0
        summary["n_clustered_particles"] = 0
        summary["largest_cluster_size"] = 0
        return summary

    # Group per frame/time
    grouped = (
        clustered_df
        .groupby(["frame", "time_ps"], as_index=False)
        .agg(
            n_clusters=("cluster_id", "nunique"),
            n_clustered_particles=("particle", "count"),
            largest_cluster_size=("cluster_size", "max"),
        )
    )

    # Merge with all frames so frames without clusters get 0
    summary = all_frames.merge(
        grouped,
        on=["frame", "time_ps"],
        how="left",
    )

    summary["n_clusters"] = summary["n_clusters"].fillna(0).astype(int)
    summary["n_clustered_particles"] = (
        summary["n_clustered_particles"].fillna(0).astype(int)
    )
    summary["largest_cluster_size"] = (
        summary["largest_cluster_size"].fillna(0).astype(int)
    )

    return summary


def safe_stem(path):
    """
    Safe label for saving output files.
    """
    return path.stem.replace(" ", "_")


# ================================================================
# MAIN
# ================================================================

if not CSV_FILES:
    raise ValueError("Please provide at least one CSV file in CSV_FILES.")

plt.figure(figsize=(12, 7))

for csv_path in CSV_FILES:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{csv_path.resolve()}"
        )

    print("=" * 72)
    print(f"Reading: {csv_path}")
    print("=" * 72)

    df = pd.read_csv(csv_path)

    summary = calculate_cluster_count_vs_time(df)

    print(summary.head())
    print()
    print(
        f"Number of analyzed frames: {len(summary)}"
    )
    print(
        f"Mean number of clusters: {summary['n_clusters'].mean():.3f}"
    )
    print(
        f"Maximum number of clusters: {summary['n_clusters'].max()}"
    )
    print()

    # Save summary CSV if desired
    if SAVE_SUMMARY_CSV:
        summary_path = csv_path.with_name(
            f"{safe_stem(csv_path)}_cluster_count_vs_time.csv"
        )
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary CSV to: {summary_path}")

    # Plot cluster count vs time
    plt.plot(
        summary["time_ps"],
        summary["n_clusters"],
        linewidth=1.8,
        label=csv_path.stem,
    )

plt.xlabel("Time / ps")
plt.ylabel("Number of clusters")
plt.title("Cluster count as a function of time", pad=15)

plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()

if SAVE_PLOT:
    plt.savefig(
        PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved plot to: {PLOT_PATH.resolve()}")

plt.show()