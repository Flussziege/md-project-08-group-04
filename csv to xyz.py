from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd


# ============================================================
# EINSTELLUNGEN
# ============================================================

# Eine oder mehrere Minimierungs-CSV-Dateien eintragen
CSV_FILE_PATHS = [
    Path(
        r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04\minimization_output\minimization_data_2026-07-15_19-29-17.csv"
    ),

    # Path(
    #     r"C:\...\minimization_sd.csv"
    # ),
]

# Elementsymbol in der XYZ-Datei
ATOM_SYMBOL = "Ar"

# Deine CSV-Positionen liegen offenbar in nm.
#
# 10.0:
# nm -> Ångström
#
# 1.0:
# Koordinaten unverändert übernehmen
#
# Da deine bisherige XYZ-Datei Koordinaten von ungefähr 0 bis 60
# für eine 6-nm-Box besitzt, ist 10.0 vermutlich korrekt.
XYZ_POSITION_SCALE = 10.0

# Anzahl der CSV-Zeilen, die gleichzeitig eingelesen werden.
# Dadurch können auch große Dateien verarbeitet werden.
CHUNK_SIZE = 100

# Vorhandene Dateien überschreiben?
OVERWRITE_EXISTING = False

# Präfix, das für den Ordnernamen entfernt wird
REMOVE_PREFIX = "minimization_"


# ============================================================
# POSITIONSSPALTEN ERKENNEN UND SORTIEREN
# ============================================================

def find_position_columns(column_names):
    """
    Sucht Positionsspalten wie:

        pos_0_x
        pos_0_y
        pos_0_z
        pos_1_x
        ...

    Rückgabe
    --------
    position_columns:
        Sortierte Liste der Positionsspalten.

    number_of_particles:
        Erkannte Anzahl der Teilchen.
    """

    pattern = re.compile(
        r"^pos_(\d+)_([xyz])$"
    )

    particle_columns = {}

    for column_name in column_names:

        match = pattern.fullmatch(
            column_name
        )

        if match is None:
            continue

        particle_index = int(
            match.group(1)
        )

        coordinate = match.group(2)

        if particle_index not in particle_columns:
            particle_columns[particle_index] = {}

        particle_columns[particle_index][coordinate] = column_name

    if not particle_columns:
        raise ValueError(
            "Es wurden keine Positionsspalten im Format "
            "'pos_<index>_<x/y/z>' gefunden."
        )

    particle_indices = sorted(
        particle_columns
    )

    # Erwartet werden Indizes 0, 1, 2, ..., N-1
    expected_indices = list(
        range(len(particle_indices))
    )

    if particle_indices != expected_indices:
        raise ValueError(
            "Die Teilchenindizes sind nicht vollständig.\n"
            f"Gefunden: {particle_indices[:20]}\n"
            f"Erwartet: {expected_indices[:20]}"
        )

    position_columns = []

    for particle_index in particle_indices:

        coordinates = particle_columns[
            particle_index
        ]

        missing_coordinates = [
            coordinate
            for coordinate in ("x", "y", "z")
            if coordinate not in coordinates
        ]

        if missing_coordinates:
            raise ValueError(
                f"Bei Teilchen {particle_index} fehlen "
                f"Koordinaten: {missing_coordinates}"
            )

        position_columns.extend(
            [
                coordinates["x"],
                coordinates["y"],
                coordinates["z"],
            ]
        )

    number_of_particles = len(
        particle_indices
    )

    return (
        position_columns,
        number_of_particles
    )


# ============================================================
# ORDNERNAMEN BESTIMMEN
# ============================================================

def create_output_folder_name(csv_path):
    """
    Erstellt aus

        minimization_cg_armijo_a_2_0_false.csv

    den Ordnernamen

        cg_armijo_a_2_0_false
    """

    file_stem = csv_path.stem

    if not file_stem.startswith(
        REMOVE_PREFIX
    ):
        raise ValueError(
            f"Der Dateiname '{csv_path.name}' beginnt nicht "
            f"mit dem erwarteten Präfix '{REMOVE_PREFIX}'."
        )

    folder_name = file_stem[
        len(REMOVE_PREFIX):
    ]

    if not folder_name:
        raise ValueError(
            f"Aus dem Dateinamen '{csv_path.name}' konnte "
            "kein gültiger Ordnername erzeugt werden."
        )

    return folder_name


# ============================================================
# KOMMENTARZEILE FÜR XYZ-FRAME ERZEUGEN
# ============================================================

def create_xyz_comment(row):
    """
    Erstellt die Kommentarzeile eines XYZ-Frames.

    Beispiel:
        step 250

    Der Wert stammt direkt aus der CSV-Spalte 'step'.
    """

    if "step" not in row.index:
        raise ValueError(
            "Die CSV-Datei enthält keine Spalte 'step'."
        )

    step_value = row["step"]

    if pd.isna(step_value):
        raise ValueError(
            "In der Spalte 'step' wurde ein leerer Wert gefunden."
        )

    # Ganzzahlige Schritte ohne Nachkommastelle ausgeben
    if float(step_value).is_integer():
        step_text = str(int(step_value))
    else:
        step_text = str(step_value)

    return f"step {step_text}"


# ============================================================
# CSV IN XYZ UMWANDELN
# ============================================================

def convert_csv_to_xyz(
    csv_path,
    xyz_path,
    position_columns,
    number_of_particles
):
    """
    Wandelt jede Zeile der CSV in einen XYZ-Frame um.

    Eine CSV-Zeile entspricht also einer vollständigen
    Teilchenkonfiguration.
    """

    # Metadaten, die in die XYZ-Kommentarzeile geschrieben werden
    available_columns = pd.read_csv(
        csv_path,
        nrows=0
    ).columns

    if "step" not in available_columns:
        raise ValueError(
            f"{csv_path.name} enthält keine Spalte 'step'."
        )

    metadata_columns = ["step"]

    columns_to_read = (
        metadata_columns
        + position_columns
    )

    temporary_xyz_path = xyz_path.with_name(
        xyz_path.name + ".tmp"
    )

    number_of_frames = 0

    try:

        with temporary_xyz_path.open(
            mode="w",
            encoding="utf-8",
            newline="\n"
        ) as xyz_file:

            csv_chunks = pd.read_csv(
                csv_path,
                usecols=columns_to_read,
                chunksize=CHUNK_SIZE
            )

            for chunk in csv_chunks:

                # Positionsdaten des gesamten Chunks
                position_values = (
                    chunk[position_columns]
                    .to_numpy(dtype=float)
                )

                expected_column_count = (
                    number_of_particles * 3
                )

                if (
                    position_values.shape[1]
                    != expected_column_count
                ):
                    raise ValueError(
                        "Die Zahl der eingelesenen "
                        "Positionsspalten ist falsch."
                    )

                positions = position_values.reshape(
                    len(chunk),
                    number_of_particles,
                    3
                )

                # Einheit für XYZ anpassen
                positions *= XYZ_POSITION_SCALE

                for local_frame_index in range(
                    len(chunk)
                ):

                    coordinates = positions[
                        local_frame_index
                    ]

                    if not np.all(
                        np.isfinite(coordinates)
                    ):
                        raise ValueError(
                            f"Frame {number_of_frames} enthält "
                            "NaN- oder unendliche Positionswerte."
                        )

                    row = chunk.iloc[
                        local_frame_index
                    ]

                    comment = create_xyz_comment(
                        row=row
                    )

                    # Erste Zeile: Teilchenzahl
                    xyz_file.write(
                        f"{number_of_particles}\n"
                    )

                    # Zweite Zeile: Kommentar
                    xyz_file.write(
                        f"{comment}\n"
                    )

                    # Danach: Atomname, x, y, z
                    for x, y, z in coordinates:

                        xyz_file.write(
                            f"{ATOM_SYMBOL} "
                            f"{x:.8f} "
                            f"{y:.8f} "
                            f"{z:.8f}\n"
                        )

                    number_of_frames += 1

        if number_of_frames == 0:
            raise ValueError(
                "Die CSV-Datei enthält keine Datenzeilen."
            )

        # Erst nach erfolgreichem Schreiben wird die temporäre
        # Datei zur endgültigen XYZ-Datei.
        if xyz_path.exists():

            if not OVERWRITE_EXISTING:
                raise FileExistsError(
                    f"XYZ-Datei existiert bereits:\n{xyz_path}"
                )

            xyz_path.unlink()

        temporary_xyz_path.replace(
            xyz_path
        )

    except Exception:

        # Unvollständige temporäre Datei entfernen
        if temporary_xyz_path.exists():
            temporary_xyz_path.unlink()

        raise

    return number_of_frames


# ============================================================
# EINE CSV-DATEI VERARBEITEN
# ============================================================

def process_csv_file(csv_path):
    """
    1. CSV überprüfen
    2. Zielordner erstellen
    3. XYZ schreiben
    4. CSV in Zielordner verschieben
    """

    csv_path = Path(
        csv_path
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV-Datei nicht gefunden:\n{csv_path}"
        )

    if csv_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Die Datei ist keine CSV-Datei:\n{csv_path}"
        )

    print()
    print("=" * 70)
    print(f"Verarbeite: {csv_path.name}")

    # Nur den Header lesen
    header_columns = pd.read_csv(
        csv_path,
        nrows=0
    ).columns

    position_columns, number_of_particles = (
        find_position_columns(
            header_columns
        )
    )

    folder_name = create_output_folder_name(
        csv_path
    )

    # Neuer Ordner liegt im bisherigen Ordner der CSV
    output_folder = (
        csv_path.parent
        / folder_name
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    xyz_path = (
        output_folder
        / f"{folder_name}.xyz"
    )

    moved_csv_path = (
        output_folder
        / csv_path.name
    )

    # Vorher prüfen, ob Ziel-CSV bereits existiert
    if moved_csv_path.exists():

        if not OVERWRITE_EXISTING:
            raise FileExistsError(
                f"Im Zielordner existiert bereits eine CSV:\n"
                f"{moved_csv_path}"
            )

    print(
        f"Erkannte Teilchenzahl: "
        f"{number_of_particles}"
    )

    print(
        f"Zielordner:\n{output_folder}"
    )

    # XYZ zuerst vollständig erzeugen
    number_of_frames = convert_csv_to_xyz(
        csv_path=csv_path,
        xyz_path=xyz_path,
        position_columns=position_columns,
        number_of_particles=number_of_particles
    )

    print(
        f"XYZ erstellt: "
        f"{xyz_path.name}"
    )

    print(
        f"Geschriebene Frames: "
        f"{number_of_frames}"
    )

    # Erst jetzt die ursprüngliche CSV verschieben
    if moved_csv_path.exists():
        moved_csv_path.unlink()

    shutil.move(
        str(csv_path),
        str(moved_csv_path)
    )

    print(
        f"CSV verschoben nach:\n"
        f"{moved_csv_path}"
    )

    print(
        "Verarbeitung erfolgreich abgeschlossen."
    )


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():

    if len(CSV_FILE_PATHS) == 0:
        raise ValueError(
            "In CSV_FILE_PATHS wurde keine Datei eingetragen."
        )

    successful_files = []
    failed_files = []

    for csv_path in CSV_FILE_PATHS:

        try:
            process_csv_file(
                csv_path
            )

            successful_files.append(
                Path(csv_path).name
            )

        except Exception as error:

            failed_files.append(
                (
                    Path(csv_path).name,
                    str(error)
                )
            )

            print()
            print(
                f"FEHLER bei {Path(csv_path).name}:"
            )
            print(error)

    print()
    print("=" * 70)
    print(
        f"Erfolgreich verarbeitet: "
        f"{len(successful_files)}"
    )

    print(
        f"Fehlgeschlagen: "
        f"{len(failed_files)}"
    )

    if failed_files:

        print()
        print("Fehlerübersicht:")

        for filename, error_message in failed_files:
            print(
                f"- {filename}: {error_message}"
            )


if __name__ == "__main__":
    main()