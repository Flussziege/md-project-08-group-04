from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# EINSTELLUNGEN
# ============================================================

# Hier eine oder mehrere Dateien eintragen
FILE_PATHS = [
    Path(r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04\minimization_output\minimization_cg_armijo_a_2_0_false.csv"),
    Path(r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04\minimization_output\minimization_cg_armijo_a_2_0_true.csv"),
    # Path(r"C:\Pfad\zu\minimierung_3.txt"),
]

# Boxlänge der Simulation
# None verwenden, wenn keine periodische Korrektur gewünscht ist
BOX_LENGTH = 5.0  #nm

# Logarithmische y-Achse
USE_LOG_SCALE = True

# Ordner, in dem die Graphen gespeichert werden
OUTPUT_FOLDER = Path("distance_plots")


# ============================================================
# POSITIONSSPALTEN SORTIEREN
# ============================================================

def position_sort_key(column_name):
    """
    Sortiert beispielsweise:

    pos_0_x
    pos_0_y
    pos_0_z
    pos_1_x
    pos_1_y
    pos_1_z
    ...
    """

    match = re.fullmatch(r"pos_(\d+)_([xyz])", column_name)

    if match is None:
        raise ValueError(
            f"Ungültige Positionsspalte: {column_name}"
        )

    particle_index = int(match.group(1))
    coordinate = match.group(2)

    coordinate_order = {
        "x": 0,
        "y": 1,
        "z": 2,
    }

    return particle_index, coordinate_order[coordinate]


# ============================================================
# POSITIONEN AUS DATEI LADEN
# ============================================================

def load_positions(file_path):
    """
    Liest nur die Schrittspalte und Positionsspalten ein.

    Rückgabe:

    steps:
        Form (n_steps,)

    positions:
        Form (n_steps, n_particles, 3)
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden:\n{file_path}"
        )

    df = pd.read_csv(
        file_path,
        usecols=lambda column:
            column == "step"
            or column.startswith("pos_")
    )

    if "step" not in df.columns:
        raise ValueError(
            f"{file_path.name} enthält keine Spalte 'step'."
        )

    position_columns = [
        column
        for column in df.columns
        if re.fullmatch(r"pos_\d+_[xyz]", column)
    ]

    position_columns = sorted(
        position_columns,
        key=position_sort_key
    )

    if len(position_columns) == 0:
        raise ValueError(
            f"In {file_path.name} wurden keine Positionen gefunden."
        )

    if len(position_columns) % 3 != 0:
        raise ValueError(
            "Die Zahl der Positionsspalten ist nicht durch 3 teilbar."
        )

    number_of_steps = len(df)
    number_of_particles = len(position_columns) // 3

    positions = (
        df[position_columns]
        .to_numpy(dtype=float)
        .reshape(
            number_of_steps,
            number_of_particles,
            3
        )
    )

    steps = df["step"].to_numpy(dtype=float)

    print(
        f"{file_path.name}: "
        f"{number_of_steps} Schritte, "
        f"{number_of_particles} Teilchen"
    )

    return steps, positions


# ============================================================
# MEAN DIST UND RMS DIST BERECHNEN
# ============================================================

def calculate_pair_distances(steps, positions, box_length=None):
    """
    Berechnet für jeden gespeicherten Schritt die Abstände
    zwischen allen eindeutigen Teilchenpaaren.

    Rückgabe:
        steps
        mean_dist: mittlerer Paarabstand pro Schritt
        rms_dist:  RMS-Paarabstand pro Schritt
    """

    number_of_steps = positions.shape[0]
    number_of_particles = positions.shape[1]

    # Indizes aller eindeutigen Teilchenpaare:
    # (0,1), (0,2), ..., (1,2), ...
    particle_i, particle_j = np.triu_indices(
        number_of_particles,
        k=1
    )

    mean_dist = np.empty(number_of_steps)
    rms_dist = np.empty(number_of_steps)

    for step_index in range(number_of_steps):

        current_positions = positions[step_index]

        # Verbindungsvektor jedes eindeutigen Teilchenpaares
        delta = (
            current_positions[particle_j]
            - current_positions[particle_i]
        )

        # Minimum-Image-Konvention
        if box_length is not None:
            delta -= box_length * np.round(
                delta / box_length
            )

        # Abstand jedes Teilchenpaares
        pair_distances = np.linalg.norm(
            delta,
            axis=1
        )

        # Mittlerer Paarabstand
        mean_dist[step_index] = np.mean(
            pair_distances
        )

        # RMS-Paarabstand
        rms_dist[step_index] = np.sqrt(
            np.mean(pair_distances**2)
        )

    return steps, mean_dist, rms_dist

# ============================================================
# DATEI AUSWERTEN
# ============================================================

def analyze_file(file_path):
    steps, positions = load_positions(file_path)

    steps, mean_dist, rms_dist = calculate_pair_distances(
        steps=steps,
        positions=positions,
        box_length=BOX_LENGTH
    )

    return {
        "name": Path(file_path).stem,
        "step": steps,
        "mean_dist": mean_dist,
        "rms_dist": rms_dist,
    }

# ============================================================
# EINEN GRAPHEN ERZEUGEN
# ============================================================

def create_plot(results, metric, ylabel, filename):
    plt.figure(figsize=(10, 6))

    for result in results:

        x_values = result["step"]
        y_values = result[metric]

        # Ungültige Werte entfernen
        valid = (
            np.isfinite(x_values)
            & np.isfinite(y_values)
        )

        # Logarithmische Achse erlaubt keine Werte <= 0
        if USE_LOG_SCALE:
            valid &= y_values > 0

        plt.plot(
            x_values[valid],
            y_values[valid],
            linewidth=1.5,
            label=result["name"]
        )

    if USE_LOG_SCALE:
        plt.yscale("log")

    plt.xlabel("Minimierungsschritt")
    plt.ylabel(ylabel)

    plt.title(
        f"{ylabel} über die Minimierung"
    )

    plt.grid(
        True,
        which="both",
        alpha=0.3
    )

    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_FOLDER / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Graph gespeichert: {output_path.resolve()}")

    plt.show()
    plt.close()


# ============================================================
# HAUPTPROGRAMM
# ============================================================

sigma = 0.34

r_opt = 2**(1/6) * sigma

print(f"Optimaler LJ-Abstand: {r_opt}")


def main():

    if len(FILE_PATHS) == 0:
        raise ValueError(
            "In FILE_PATHS wurde keine Datei eingetragen."
        )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    for file_path in FILE_PATHS:
        result = analyze_file(file_path)
        results.append(result)

    create_plot(
        results=results,
        metric="mean_dist",
        ylabel="Mittlerer Teilchenpaarabstand",
        filename="mean_pair_distance.png"
    )

    create_plot(
        results=results,
        metric="rms_dist",
        ylabel="RMS-Teilchenpaarabstand",
        filename="rms_pair_distance.png"
    )


if __name__ == "__main__":
    main()