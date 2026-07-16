from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from cluster_functions import read_xyz_trajectory


# ================================================================
# EINGABEN
# ================================================================

XYZ_PATH = Path(
    r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04\results\2026-07-15_19-30-00\my_simulation_pos.xyz"
)

# Boxlänge in nm
BOX_LENGTH_NM = 5.0

# Zeit zwischen zwei tatsächlich gespeicherten XYZ-Frames.
#
# Wenn jeder Simulationsschritt gespeichert wurde:
# DT_SAVED_FRAME_PS = dt
#
# Wenn nur jeder 10. Schritt gespeichert wurde:
# DT_SAVED_FRAME_PS = 10 * dt
DT_SAVED_FRAME_PS = 0.001


# ================================================================
# GLEICHGEWICHTSBEREICH
# ================================================================

# Alle Frames davor werden nicht ausgewertet.
START_FRAME = 5000

# None bedeutet: bis zum Ende
STOP_FRAME = None


# ================================================================
# TEILCHENAUSWAHL
# ================================================================

# False:
#   MSD aller Teilchen berechnen
#
# True:
#   Nur Teilchen eines Startclusters untersuchen
USE_CLUSTER_SELECTION = False

# Verwende dazu am besten die Datei aus cluster_const.py.
CLUSTER_CSV_PATH = Path(
    "results/Long_sim/"
    "my_simulation_pos_cluster_const_ids.csv"
)

# 0 ist der größte Cluster aus Frame 0
CLUSTER_ID_TO_ANALYZE = 0

# Frame, aus dem die Teilchen des Clusters gewählt werden
CLUSTER_SELECTION_FRAME = 0


# ================================================================
# MSD-EINSTELLUNGEN
# ================================================================

# Bei einem Tropfen sehr wichtig:
#
# Entfernt die Bewegung des gesamten Tropfens durch die Box.
#
# Sonst könnte ein fester Tropfen, der sich als Ganzes bewegt,
# fälschlich wie eine Flüssigkeit erscheinen.
REMOVE_CENTER_OF_MASS_MOTION = True

# Größter betrachteter Zeitabstand
MAX_LAG_PS = 20.0

# Anzahl der Punkte der MSD-Kurve
N_LAG_POINTS = 250

# Nur jeden n-ten möglichen Startzeitpunkt verwenden.
#
# Größer:
#   schneller, aber statistisch ungenauer
#
# Kleiner:
#   langsamer, aber bessere Statistik
ORIGIN_STRIDE = 20

# Rechenblöcke zur Verringerung des Speicherbedarfs
ORIGIN_BATCH_SIZE = 200


# ================================================================
# DIFFUSIONSFIT
# ================================================================

# Bereich, in dem die MSD ungefähr linear sein soll.
#
# Diese Werte musst du nach Betrachtung des Diagramms gegebenenfalls
# anpassen.
#
# Für keinen Fit:
# FIT_START_PS = None
# FIT_END_PS = None
FIT_START_PS = 5.0
FIT_END_PS = 15.0


# ================================================================
# AUSGABE
# ================================================================

SAVE_CSV = True
SAVE_PLOT = True

CSV_PATH = XYZ_PATH.with_name(
    f"{XYZ_PATH.stem}_msd.csv"
)

PLOT_PATH = XYZ_PATH.with_name(
    f"{XYZ_PATH.stem}_msd.png"
)


# ================================================================
# TEILCHEN EINES STARTCLUSTERS EINLESEN
# ================================================================

def read_cluster_particle_indices(
    csv_path,
    frame,
    cluster_id
):
    """
    Liest die Teilchenindizes eines bestimmten Clusters aus einer
    Cluster-CSV ein.

    Für die MSD eines anfänglichen Tropfens sollte eine konstante
    Clusterdatei aus cluster_const.py verwendet werden.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cluster-CSV wurde nicht gefunden:\n{csv_path}"
        )

    particle_indices = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        required_columns = {
            "frame",
            "particle",
            "cluster_id"
        }

        if reader.fieldnames is None:
            raise ValueError(
                "Die Cluster-CSV besitzt keine Kopfzeile."
            )

        missing_columns = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "In der Cluster-CSV fehlen Spalten: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:

            current_frame = int(
                row["frame"]
            )

            current_cluster_id = int(
                row["cluster_id"]
            )

            if (
                current_frame == frame
                and current_cluster_id == cluster_id
            ):

                particle_indices.append(
                    int(row["particle"])
                )

    if not particle_indices:
        raise ValueError(
            f"Im Frame {frame} wurden keine Teilchen mit "
            f"cluster_id={cluster_id} gefunden."
        )

    return np.asarray(
        sorted(particle_indices),
        dtype=int
    )


# ================================================================
# PERIODISCHE TRAJEKTORIE ENTWICKELN
# ================================================================

def unwrap_trajectory(
    wrapped_positions,
    box_length
):
    """
    Rekonstruiert kontinuierliche Positionen aus den in die Box
    zurückgefalteten Koordinaten.

    Beispiel:
        4.99 nm -> 0.01 nm

    wird als kleine Bewegung von +0.02 nm erkannt und nicht als
    Sprung von -4.98 nm.

    Voraussetzung:
    Ein Teilchen darf sich zwischen zwei gespeicherten Frames nicht
    weiter als L/2 bewegen.
    """

    wrapped_positions = np.asarray(
        wrapped_positions,
        dtype=float
    )

    if (
        wrapped_positions.ndim != 3
        or wrapped_positions.shape[2] != 3
    ):
        raise ValueError(
            "wrapped_positions muss die Form "
            "(n_frames, n_particles, 3) besitzen."
        )

    frame_displacements = (
        wrapped_positions[1:]
        - wrapped_positions[:-1]
    )

    # Minimum-Image-Korrektur
    frame_displacements -= (
        box_length
        * np.rint(
            frame_displacements
            / box_length
        )
    )

    unwrapped_positions = np.empty_like(
        wrapped_positions
    )

    unwrapped_positions[0] = (
        wrapped_positions[0]
    )

    unwrapped_positions[1:] = (
        wrapped_positions[0]
        + np.cumsum(
            frame_displacements,
            axis=0
        )
    )

    return unwrapped_positions


# ================================================================
# SCHWERPUNKTSBEWEGUNG ENTFERNEN
# ================================================================

def remove_center_of_mass_motion(
    positions
):
    """
    Entfernt die gemeinsame Translation aller ausgewählten Teilchen.

    Die internen Bewegungen relativ zum Schwerpunkt bleiben erhalten.
    """

    center_of_mass = np.mean(
        positions,
        axis=1,
        keepdims=True
    )

    center_of_mass_displacement = (
        center_of_mass
        - center_of_mass[0]
    )

    corrected_positions = (
        positions
        - center_of_mass_displacement
    )

    return corrected_positions


# ================================================================
# MSD MIT MEHREREN ZEITURSPRÜNGEN
# ================================================================

def calculate_msd(
    positions,
    dt_ps,
    max_lag_ps,
    n_lag_points=250,
    origin_stride=1,
    batch_size=200
):
    """
    Berechnet die MSD über mehrere Zeitursprünge.

    Die Mittelung erfolgt:
    - über alle ausgewählten Teilchen
    - über viele verschiedene Startzeitpunkte

    Dadurch ist die Statistik besser als bei einer Berechnung nur
    relativ zum ersten Frame.
    """

    positions = np.asarray(
        positions,
        dtype=float
    )

    n_frames = positions.shape[0]

    if n_frames < 2:
        raise ValueError(
            "Für eine MSD werden mindestens zwei Frames benötigt."
        )

    max_lag_frames = min(
        int(max_lag_ps / dt_ps),
        n_frames - 1
    )

    if max_lag_frames < 1:
        raise ValueError(
            "MAX_LAG_PS ist kleiner als ein gespeicherter Zeitschritt."
        )

    # Gleichmäßig verteilte Zeitabstände
    lag_frames = np.unique(
        np.concatenate(
            (
                np.array(
                    [0],
                    dtype=int
                ),
                np.linspace(
                    1,
                    max_lag_frames,
                    n_lag_points - 1,
                    dtype=int
                )
            )
        )
    )

    lag_times_ps = (
        lag_frames
        * dt_ps
    )

    msd = np.zeros(
        len(lag_frames),
        dtype=float
    )

    msd_sem = np.zeros(
        len(lag_frames),
        dtype=float
    )

    number_of_origins = np.zeros(
        len(lag_frames),
        dtype=int
    )

    for lag_index, lag in enumerate(
        lag_frames
    ):

        if lag == 0:
            msd[lag_index] = 0.0
            number_of_origins[lag_index] = n_frames
            continue

        origins = np.arange(
            0,
            n_frames - lag,
            origin_stride,
            dtype=int
        )

        value_sum = 0.0
        squared_value_sum = 0.0
        value_count = 0

        for batch_start in range(
            0,
            len(origins),
            batch_size
        ):

            batch_origins = origins[
                batch_start:
                batch_start + batch_size
            ]

            displacements = (
                positions[
                    batch_origins + lag
                ]
                - positions[
                    batch_origins
                ]
            )

            # Summe über x, y, z
            squared_displacements = np.sum(
                displacements**2,
                axis=2
            )

            # Mittelwert über alle ausgewählten Teilchen
            msd_per_origin = np.mean(
                squared_displacements,
                axis=1
            )

            value_sum += np.sum(
                msd_per_origin
            )

            squared_value_sum += np.sum(
                msd_per_origin**2
            )

            value_count += len(
                msd_per_origin
            )

        mean_value = (
            value_sum
            / value_count
        )

        variance = (
            squared_value_sum
            / value_count
            - mean_value**2
        )

        variance = max(
            variance,
            0.0
        )

        msd[lag_index] = mean_value

        # Standardfehler über die Zeitursprünge
        msd_sem[lag_index] = np.sqrt(
            variance
            / value_count
        )

        number_of_origins[lag_index] = (
            value_count
        )

        if (
            lag_index % 25 == 0
            or lag_index == len(lag_frames) - 1
        ):
            print(
                f"MSD-Punkt {lag_index + 1} "
                f"von {len(lag_frames)}"
            )

    return (
        lag_times_ps,
        msd,
        msd_sem,
        number_of_origins
    )


# ================================================================
# TRAJEKTORIE EINLESEN
# ================================================================

positions, atom_names = read_xyz_trajectory(
    XYZ_PATH
)

print(
    f"Gesamte Trajektorie: {positions.shape}"
)

# Deine XYZ-Datei enthält Å.
#
# 1 Å = 0.1 nm
positions = positions * 0.1


# ================================================================
# OPTIONAL NUR EINEN STARTCLUSTER AUSWÄHLEN
# ================================================================

if USE_CLUSTER_SELECTION:

    particle_indices = (
        read_cluster_particle_indices(
            csv_path=CLUSTER_CSV_PATH,
            frame=CLUSTER_SELECTION_FRAME,
            cluster_id=CLUSTER_ID_TO_ANALYZE
        )
    )

    positions = positions[
        :,
        particle_indices,
        :
    ]

    print(
        f"Ausgewählter Startcluster: "
        f"{CLUSTER_ID_TO_ANALYZE}"
    )

    print(
        f"Ausgewählte Teilchen: "
        f"{len(particle_indices)}"
    )

else:

    print(
        f"Verwendete Teilchen: "
        f"{positions.shape[1]}"
    )


# ================================================================
# NUR GLEICHGEWICHTSBEREICH VERWENDEN
# ================================================================

positions = positions[
    START_FRAME:STOP_FRAME
]

if len(positions) < 2:
    raise ValueError(
        "Nach der Auswahl mit START_FRAME und STOP_FRAME "
        "bleiben zu wenige Frames übrig."
    )

print(
    f"Verwendete Frames: "
    f"{len(positions)}"
)


# ================================================================
# PBC ENTFERNEN
# ================================================================

unwrapped_positions = unwrap_trajectory(
    wrapped_positions=positions,
    box_length=BOX_LENGTH_NM
)


# ================================================================
# SCHWERPUNKTSBEWEGUNG ENTFERNEN
# ================================================================

if REMOVE_CENTER_OF_MASS_MOTION:

    unwrapped_positions = (
        remove_center_of_mass_motion(
            unwrapped_positions
        )
    )

    print(
        "Schwerpunktsbewegung wurde entfernt."
    )


# ================================================================
# MSD BERECHNEN
# ================================================================

(
    lag_times_ps,
    msd_nm2,
    msd_sem_nm2,
    number_of_origins
) = calculate_msd(
    positions=unwrapped_positions,
    dt_ps=DT_SAVED_FRAME_PS,
    max_lag_ps=MAX_LAG_PS,
    n_lag_points=N_LAG_POINTS,
    origin_stride=ORIGIN_STRIDE,
    batch_size=ORIGIN_BATCH_SIZE
)


# ================================================================
# DIFFUSIONSKOEFFIZIENT BESTIMMEN
# ================================================================

fit_result = None

if (
    FIT_START_PS is not None
    and FIT_END_PS is not None
):

    fit_mask = (
        (lag_times_ps >= FIT_START_PS)
        & (lag_times_ps <= FIT_END_PS)
    )

    if np.count_nonzero(fit_mask) < 2:
        raise ValueError(
            "Im gewählten Fitbereich liegen weniger "
            "als zwei MSD-Punkte."
        )

    slope, intercept = np.polyfit(
        lag_times_ps[fit_mask],
        msd_nm2[fit_mask],
        deg=1
    )

    # In drei Dimensionen:
    #
    # MSD = 6 D t
    diffusion_nm2_per_ps = (
        slope / 6.0
    )

    # 1 nm²/ps = 10^-6 m²/s
    diffusion_m2_per_s = (
        diffusion_nm2_per_ps
        * 1e-6
    )

    fit_result = {
        "slope": slope,
        "intercept": intercept,
        "D_nm2_per_ps": diffusion_nm2_per_ps,
        "D_m2_per_s": diffusion_m2_per_s
    }

    print()
    print("=" * 70)
    print("DIFFUSIONSFIT")
    print("=" * 70)

    print(
        f"Fitbereich: "
        f"{FIT_START_PS:.3f} bis "
        f"{FIT_END_PS:.3f} ps"
    )

    print(
        f"MSD-Steigung: "
        f"{slope:.6e} nm²/ps"
    )

    print(
        f"D = {diffusion_nm2_per_ps:.6e} "
        f"nm²/ps"
    )

    print(
        f"D = {diffusion_m2_per_s:.6e} "
        f"m²/s"
    )

    print("=" * 70)


# ================================================================
# CSV SPEICHERN
# ================================================================

if SAVE_CSV:

    output_data = np.column_stack(
        (
            lag_times_ps,
            msd_nm2,
            msd_sem_nm2,
            number_of_origins
        )
    )

    np.savetxt(
        CSV_PATH,
        output_data,
        delimiter=",",
        header=(
            "lag_time_ps,"
            "msd_nm2,"
            "msd_sem_nm2,"
            "number_of_time_origins"
        ),
        comments=""
    )

    print(
        f"MSD-CSV gespeichert unter: "
        f"{CSV_PATH}"
    )


# ================================================================
# MSD PLOTTEN
# ================================================================

plt.figure(figsize=(9, 6))

plt.plot(
    lag_times_ps,
    msd_nm2,
    linewidth=1.6,
    label="MSD"
)

plt.fill_between(
    lag_times_ps,
    np.maximum(
        msd_nm2 - msd_sem_nm2,
        0.0
    ),
    msd_nm2 + msd_sem_nm2,
    alpha=0.2,
    label="Standardfehler"
)

if fit_result is not None:

    fit_mask = (
        (lag_times_ps >= FIT_START_PS)
        & (lag_times_ps <= FIT_END_PS)
    )

    fit_line = (
        fit_result["slope"]
        * lag_times_ps[fit_mask]
        + fit_result["intercept"]
    )

    plt.plot(
        lag_times_ps[fit_mask],
        fit_line,
        linestyle="--",
        linewidth=1.5,
        label=(
            "linearer Fit: "
            rf"$D={fit_result['D_nm2_per_ps']:.3e}$ "
            r"nm$^2$/ps"
        )
    )

plt.xlabel("Zeitverschiebung / ps")
plt.ylabel(r"MSD / nm$^2$")

plt.title(
    "Mittlere quadratische Verschiebung"
)

plt.grid(True)
plt.legend()
plt.tight_layout()

if SAVE_PLOT:

    plt.savefig(
        PLOT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"MSD-Plot gespeichert unter: "
        f"{PLOT_PATH}"
    )

plt.show()