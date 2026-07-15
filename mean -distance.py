from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# EINSTELLUNGEN
# ============================================================

# Eine oder mehrere XYZ-Trajektorien eintragen
FILE_PATHS = [
    Path(
        r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04\results\2026-07-15_19-30-00\my_simulation_pos.xyz"
    ),
    # Path(r"C:\Pfad\zu\weiterer_trajektorie.xyz"),
]


# ============================================================
# EINHEITEN
# ============================================================

# Umrechnungsfaktor der XYZ-Koordinaten nach nm.
#
# 0.1:
# XYZ enthält Ångström, Umrechnung Å -> nm
#
# 1.0:
# XYZ enthält bereits nm
XYZ_TO_NM = 0.1

# Boxlänge nach der Umrechnung in nm
#
# Beispiel:
# XYZ-Koordinaten liegen zwischen 0 und 60 Å
# -> Boxlänge = 6 nm
BOX_LENGTH = 6.0


# ============================================================
# AUSWERTUNG
# ============================================================

# Nur jeden n-ten Frame auswerten.
#
# 1:
# jeden Frame
#
# 10:
# jeden zehnten Frame
FRAME_STRIDE = 1

# Unvollständigen letzten Frame ignorieren
IGNORE_INCOMPLETE_LAST_FRAME = True

# Logarithmische y-Achse für Mean- und RMS-Plot
USE_LOG_SCALE = False


# ============================================================
# HISTOGRAMM
# ============================================================

# Anzahl der Klassen
HISTOGRAM_BINS = 60

# Welcher ausgewertete Frame wird für das Histogramm verwendet?
#
# "first":
# erster ausgewerteter Frame
#
# "last":
# letzter ausgewerteter Frame
HISTOGRAM_FRAME = "last"

# Maximale dargestellte Entfernung in nm.
#
# None:
# vollständiger Bereich
#
# Beispiel:
# HISTOGRAM_MAX_DISTANCE = 2.0
HISTOGRAM_MAX_DISTANCE = None

# False:
# Anzahl der Teilchenpaare
#
# True:
# Wahrscheinlichkeitsdichte
NORMALIZE_HISTOGRAM = False


# ============================================================
# AUSGABE
# ============================================================

OUTPUT_FOLDER = Path("distance_plots")


# ============================================================
# LENNARD-JONES-PARAMETER
# ============================================================

SIGMA = 0.34  # nm

R_OPT = 2 ** (1 / 6) * SIGMA

print(
    f"Optimaler LJ-Abstand: "
    f"{R_OPT:.6f} nm"
)


# ============================================================
# SCHRITT AUS KOMMENTARZEILE LESEN
# ============================================================

def extract_step_from_comment(comment, frame_index):
    """
    Liest den Schritt aus Kommentarzeilen wie:

        step 100
        step=100
        step: 100
        frame=10 step=100

    Wird kein Schritt gefunden, wird der Frameindex verwendet.
    """

    match = re.search(
        r"\bstep\s*[=:]?\s*(-?\d+(?:\.\d+)?)",
        comment,
        flags=re.IGNORECASE
    )

    if match is None:
        return float(frame_index)

    return float(
        match.group(1)
    )


# ============================================================
# XYZ-FRAMES EINLESEN
# ============================================================

def iterate_xyz_frames(file_path):
    """
    Liest eine XYZ-Trajektorie Frame für Frame ein.

    Erwartetes Format:

        Anzahl der Teilchen
        Kommentarzeile
        Ar x y z
        Ar x y z
        ...

    Liefert für jeden vollständigen Frame:

        frame_index
        step
        positions

    positions besitzt die Form:

        (n_particles, 3)

    Die Koordinaten werden direkt in nm umgerechnet.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden:\n{file_path}"
        )

    reference_particle_count = None
    reference_atom_names = None

    frame_index = 0

    with file_path.open(
        mode="r",
        encoding="utf-8"
    ) as xyz_file:

        while True:

            # Erste nichtleere Zeile suchen
            particle_count_line = xyz_file.readline()

            while (
                particle_count_line
                and not particle_count_line.strip()
            ):
                particle_count_line = xyz_file.readline()

            # Reguläres Dateiende
            if not particle_count_line:
                break

            try:
                number_of_particles = int(
                    particle_count_line.strip()
                )

            except ValueError as error:
                raise ValueError(
                    f"Ungültige Teilchenzahl vor Frame "
                    f"{frame_index}:\n"
                    f"{particle_count_line.strip()}"
                ) from error

            if number_of_particles <= 0:
                raise ValueError(
                    f"Frame {frame_index} besitzt eine "
                    f"ungültige Teilchenzahl: "
                    f"{number_of_particles}"
                )

            # Teilchenzahl aller Frames vergleichen
            if reference_particle_count is None:
                reference_particle_count = number_of_particles

            elif number_of_particles != reference_particle_count:
                raise ValueError(
                    f"Frame {frame_index} enthält "
                    f"{number_of_particles} Teilchen.\n"
                    f"Erwartet wurden "
                    f"{reference_particle_count} Teilchen."
                )

            # Kommentarzeile lesen
            comment_line = xyz_file.readline()

            if not comment_line:

                if IGNORE_INCOMPLETE_LAST_FRAME:
                    print(
                        f"Warnung: Unvollständiger Frame "
                        f"{frame_index} wurde ignoriert."
                    )
                    break

                raise ValueError(
                    f"Kommentarzeile in Frame "
                    f"{frame_index} fehlt."
                )

            comment = comment_line.strip()

            positions = np.empty(
                shape=(number_of_particles, 3),
                dtype=float
            )

            atom_names = []
            frame_complete = True

            for particle_index in range(
                number_of_particles
            ):

                atom_line = xyz_file.readline()

                if not atom_line:
                    frame_complete = False
                    break

                columns = atom_line.split()

                if len(columns) < 4:
                    frame_complete = False
                    break

                atom_name = columns[0]

                try:
                    positions[particle_index, 0] = float(
                        columns[1]
                    )

                    positions[particle_index, 1] = float(
                        columns[2]
                    )

                    positions[particle_index, 2] = float(
                        columns[3]
                    )

                except ValueError as error:
                    raise ValueError(
                        f"Ungültige Koordinaten in Frame "
                        f"{frame_index}, Teilchen "
                        f"{particle_index}:\n"
                        f"{atom_line.strip()}"
                    ) from error

                atom_names.append(
                    atom_name
                )

            if not frame_complete:

                if IGNORE_INCOMPLETE_LAST_FRAME:
                    print(
                        f"Warnung: Unvollständiger letzter "
                        f"Frame {frame_index} wurde ignoriert."
                    )
                    break

                raise ValueError(
                    f"Frame {frame_index} ist unvollständig."
                )

            # Atomreihenfolge überprüfen
            if reference_atom_names is None:
                reference_atom_names = atom_names

            elif atom_names != reference_atom_names:
                raise ValueError(
                    f"Die Atomreihenfolge in Frame "
                    f"{frame_index} stimmt nicht mit dem "
                    f"ersten Frame überein."
                )

            if not np.all(
                np.isfinite(positions)
            ):
                raise ValueError(
                    f"Frame {frame_index} enthält NaN- "
                    f"oder unendliche Koordinaten."
                )

            # Koordinaten nach nm umrechnen
            positions *= XYZ_TO_NM

            # Schritt aus der Kommentarzeile lesen
            step = extract_step_from_comment(
                comment=comment,
                frame_index=frame_index
            )

            yield (
                frame_index,
                step,
                positions
            )

            frame_index += 1


# ============================================================
# PAARABSTÄNDE EINES FRAMES BERECHNEN
# ============================================================

def calculate_frame_pair_distances(
    positions,
    particle_i,
    particle_j,
    box_length=None
):
    """
    Berechnet für einen Frame:

        alle eindeutigen Teilchenpaarabstände
        mittleren Teilchenpaarabstand
        RMS-Teilchenpaarabstand
    """

    # Verbindungsvektoren aller eindeutigen Teilchenpaare
    delta = (
        positions[particle_j]
        - positions[particle_i]
    )

    # Minimum-Image-Konvention
    if box_length is not None:

        if box_length <= 0:
            raise ValueError(
                "BOX_LENGTH muss größer als null sein."
            )

        delta -= box_length * np.round(
            delta / box_length
        )

    # Beträge der Verbindungsvektoren
    pair_distances = np.linalg.norm(
        delta,
        axis=1
    )

    # Mittlerer Paarabstand
    mean_distance = np.mean(
        pair_distances
    )

    # RMS-Paarabstand
    rms_distance = np.sqrt(
        np.mean(
            pair_distances**2
        )
    )

    return (
        mean_distance,
        rms_distance,
        pair_distances
    )


# ============================================================
# XYZ-DATEI AUSWERTEN
# ============================================================

def analyze_file(file_path):
    """
    Liest die XYZ-Datei Frame für Frame ein und berechnet:

        mittleren Paarabstand pro Frame
        RMS-Paarabstand pro Frame
        Paarabstandsverteilung eines ausgewählten Frames
    """

    if HISTOGRAM_FRAME not in {
        "first",
        "last"
    }:
        raise ValueError(
            "HISTOGRAM_FRAME muss "
            "'first' oder 'last' sein."
        )

    steps = []
    mean_distances = []
    rms_distances = []

    histogram_pair_distances = None
    histogram_step = None

    particle_i = None
    particle_j = None

    total_frames = 0
    evaluated_frames = 0

    for (
        frame_index,
        step,
        positions
    ) in iterate_xyz_frames(file_path):

        total_frames += 1

        # Nur jeden n-ten Frame verwenden
        if frame_index % FRAME_STRIDE != 0:
            continue

        # Paarindizes nur einmal erzeugen
        if particle_i is None:

            number_of_particles = positions.shape[0]

            particle_i, particle_j = np.triu_indices(
                number_of_particles,
                k=1
            )

            number_of_pairs = len(
                particle_i
            )

            print(
                f"{Path(file_path).name}: "
                f"{number_of_particles} Teilchen, "
                f"{number_of_pairs} eindeutige Paare"
            )

        (
            mean_distance,
            rms_distance,
            pair_distances
        ) = calculate_frame_pair_distances(
            positions=positions,
            particle_i=particle_i,
            particle_j=particle_j,
            box_length=BOX_LENGTH
        )

        steps.append(
            step
        )

        mean_distances.append(
            mean_distance
        )

        rms_distances.append(
            rms_distance
        )

        # Ersten ausgewerteten Frame speichern
        if (
            HISTOGRAM_FRAME == "first"
            and histogram_pair_distances is None
        ):
            histogram_pair_distances = pair_distances.copy()
            histogram_step = step

        # Letzten ausgewerteten Frame speichern
        if HISTOGRAM_FRAME == "last":
            histogram_pair_distances = pair_distances.copy()
            histogram_step = step

        evaluated_frames += 1

    if evaluated_frames == 0:
        raise ValueError(
            f"In {Path(file_path).name} wurde kein "
            f"vollständiger Frame ausgewertet."
        )

    print(
        f"{Path(file_path).name}: "
        f"{total_frames} Frames gelesen, "
        f"{evaluated_frames} Frames ausgewertet"
    )

    print(
        f"Histogramm verwendet Schritt: "
        f"{histogram_step:g}"
    )

    return {
        "name": Path(file_path).stem,

        "step": np.asarray(
            steps,
            dtype=float
        ),

        "mean_dist": np.asarray(
            mean_distances,
            dtype=float
        ),

        "rms_dist": np.asarray(
            rms_distances,
            dtype=float
        ),

        "histogram_pair_distances": np.asarray(
            histogram_pair_distances,
            dtype=float
        ),

        "histogram_step": histogram_step,
    }


# ============================================================
# LINIENGRAPH ERZEUGEN
# ============================================================

def create_plot(
    results,
    metric,
    ylabel,
    filename
):
    """
    Erzeugt einen Graphen für mean_dist oder rms_dist.
    """

    plt.figure(
        figsize=(10, 6)
    )

    for result in results:

        x_values = result["step"]
        y_values = result[metric]

        valid = (
            np.isfinite(x_values)
            & np.isfinite(y_values)
        )

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

    plt.xlabel("Schritt")
    plt.ylabel(ylabel)

    plt.title(
        f"{ylabel} über die Trajektorie"
    )

    plt.grid(
        True,
        which="both",
        alpha=0.3
    )

    plt.legend()
    plt.tight_layout()

    output_path = (
        OUTPUT_FOLDER
        / filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"Graph gespeichert:\n"
        f"{output_path.resolve()}"
    )

    plt.show()
    plt.close()


# ============================================================
# HISTOGRAMM DER PAARABSTÄNDE
# ============================================================

def create_pair_distance_histogram(results):
    """
    Erstellt ein Histogramm der Teilchenpaarabstände.

    x-Achse:
        Teilchenpaarabstand in nm

    y-Achse:
        Anzahl der Teilchenpaare oder Wahrscheinlichkeitsdichte
    """

    plt.figure(
        figsize=(10, 6)
    )

    histogram_created = False

    for result in results:

        pair_distances = np.asarray(
            result["histogram_pair_distances"],
            dtype=float
        )

        # Ungültige Werte entfernen
        valid_distances = pair_distances[
            np.isfinite(pair_distances)
        ]

        # Optional den Abstand begrenzen
        if HISTOGRAM_MAX_DISTANCE is not None:
            valid_distances = valid_distances[
                valid_distances
                <= HISTOGRAM_MAX_DISTANCE
            ]

        if len(valid_distances) == 0:
            print(
                f"Warnung: Für {result['name']} sind keine "
                f"gültigen Paarabstände vorhanden."
            )
            continue

        step = result["histogram_step"]

        plt.hist(
            valid_distances,
            bins=HISTOGRAM_BINS,
            density=NORMALIZE_HISTOGRAM,
            alpha=0.55,
            edgecolor="black",
            label=(
                f"{result['name']}, "
                f"Schritt {step:g}"
            )
        )

        histogram_created = True

    if not histogram_created:
        plt.close()

        raise ValueError(
            "Es konnte kein Histogramm erstellt werden."
        )

    # Optimalen Lennard-Jones-Abstand markieren
    plt.axvline(
        R_OPT,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Optimaler LJ-Abstand: "
            f"{R_OPT:.3f} nm"
        )
    )

    plt.xlabel(
        "Teilchenpaarabstand / nm"
    )

    if NORMALIZE_HISTOGRAM:
        plt.ylabel(
            "Wahrscheinlichkeitsdichte"
        )
    else:
        plt.ylabel(
            "Anzahl der Teilchenpaare"
        )

    plt.title(
        "Verteilung der Teilchenpaarabstände"
    )

    plt.grid(
        True,
        axis="y",
        alpha=0.3
    )

    plt.legend()
    plt.tight_layout()

    output_path = (
        OUTPUT_FOLDER
        / "pair_distance_histogram.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"Histogramm gespeichert:\n"
        f"{output_path.resolve()}"
    )

    plt.show()
    plt.close()


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():

    if len(FILE_PATHS) == 0:
        raise ValueError(
            "In FILE_PATHS wurde keine "
            "XYZ-Datei eingetragen."
        )

    if FRAME_STRIDE < 1:
        raise ValueError(
            "FRAME_STRIDE muss mindestens 1 sein."
        )

    if HISTOGRAM_BINS < 1:
        raise ValueError(
            "HISTOGRAM_BINS muss mindestens 1 sein."
        )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    for file_path in FILE_PATHS:

        print()
        print(
            f"Werte XYZ-Datei aus:\n"
            f"{file_path}"
        )

        result = analyze_file(
            file_path
        )

        results.append(
            result
        )

    # Mittlerer Paarabstand über die Trajektorie
    create_plot(
        results=results,
        metric="mean_dist",
        ylabel="Mittlerer Teilchenpaarabstand / nm",
        filename="mean_pair_distance.png"
    )

    # RMS-Paarabstand über die Trajektorie
    create_plot(
        results=results,
        metric="rms_dist",
        ylabel="RMS-Teilchenpaarabstand / nm",
        filename="rms_pair_distance.png"
    )

    # Verteilung der Paarabstände
    create_pair_distance_histogram(
        results=results
    )


if __name__ == "__main__":
    main()