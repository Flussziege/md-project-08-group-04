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
    Path(r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04\results\2026-07-15_19-30-00\my_simulation_pos.xyz"),
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

def calculate_distances(steps, positions, box_length=None):
    """
    Berechnet die Verschiebung jedes Teilchens gegenüber
    dem unmittelbar vorherigen Minimierungsschritt.
    """

    if len(positions) < 2:
        raise ValueError(
            "Die Datei muss mindestens zwei Schritte enthalten."
        )

    # Differenz zwischen Schritt k und Schritt k-1
    delta_positions = np.diff(
        positions,
        axis=0
    )

    # Periodische Randbedingungen berücksichtigen
    if box_length is not None:

        if box_length <= 0:
            raise ValueError(
                "BOX_LENGTH muss positiv sein."
            )

        delta_positions -= box_length * np.round(
            delta_positions / box_length
        )

    # Distanz jedes Teilchens:
    # sqrt(dx² + dy² + dz²)
    particle_distances = np.linalg.norm(
        delta_positions,
        axis=2
    )

    # Mittlere Distanz über alle Teilchen
    mean_dist = np.mean(
        particle_distances,
        axis=1
    )

    # RMS-Distanz über alle Teilchen
    rms_dist = np.sqrt(
        np.mean(
            particle_distances**2,
            axis=1
        )
    )

    # Durch np.diff beginnt das Ergebnis bei Schritt 1
    result_steps = steps[1:]

    return result_steps, mean_dist, rms_dist


# ============================================================
# DATEI AUSWERTEN
# ============================================================

def analyze_file(file_path):
    steps, positions = load_positions(file_path)

    steps, mean_dist, rms_dist = calculate_distances(
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

    # Graph 1: mittlere Distanz
    create_plot(
        results=results,
        metric="mean_dist",
        ylabel="Mittlere Teilchenverschiebung",
        filename="mean_distance.png"
    )

    # Graph 2: RMS-Distanz
    create_plot(
        results=results,
        metric="rms_dist",
        ylabel="RMS-Teilchenverschiebung",
        filename="rms_distance.png"
    )


if __name__ == "__main__":
    main()