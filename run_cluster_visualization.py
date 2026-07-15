"""
run_cluster_visualization.py

================================================================
PRIMER: SO VERWENDEST DU DIE DREI DATEIEN
================================================================

Lege diese drei Dateien in denselben Projektordner:

    cluster_functions.py
    visualization_functions.py
    run_cluster_visualization.py

Installiere die benötigten Pakete im aktiven Python-/Conda-Environment:

    pip install numpy scipy plotly

oder mit Conda:

    conda install numpy scipy plotly

Danach bearbeitest du ausschließlich den Parameterbereich weiter unten
und startest diese Datei:

    python run_cluster_visualization.py

----------------------------------------------------------------
TYPISCHER ABLAUF
----------------------------------------------------------------

1. Trage bei XYZ_PATH den Pfad zu deiner Trajektorie ein.

2. Trage die Boxlänge aus der Simulation bei BOX_LENGTH_NM ein.

3. Setze ANALYZE_CLUSTERS = True, um eine neue Cluster-CSV zu erzeugen.

4. Setze VISUALIZE = True, um die interaktive Visualisierung zu öffnen.

5. Setze CLUSTER = True, wenn die Teilchen nach Cluster-ID eingefärbt
   werden sollen.

6. Setze CLUSTER = False, wenn du die ursprüngliche Darstellung ohne
   Clusterfarben sehen möchtest.

----------------------------------------------------------------
ERSTER SINNVOLLER CLUSTER-CUTOFF
----------------------------------------------------------------

Für Lennard-Jones-Teilchen kannst du zunächst verwenden:

    CLUSTER_CUTOFF_NM = 1.5 * SIGMA_NM

Bei sigma = 0.34 nm ergibt das:

    CLUSTER_CUTOFF_NM = 0.51 nm

Besser ist langfristig das erste Minimum der radialen
Verteilungsfunktion g(r).

----------------------------------------------------------------
WICHTIG ZU DEN FRAME-STRIDES
----------------------------------------------------------------

Am einfachsten:

    ANALYSIS_FRAME_STRIDE = 1

Dann enthält die Cluster-CSV Zuordnungen für jeden XYZ-Frame und du
kannst bei PLOT_FRAME_STRIDE beliebige Werte wie 1, 10 oder 20 wählen.

----------------------------------------------------------------
BEDEUTUNG DER CLUSTER-IDS
----------------------------------------------------------------

In jedem Frame gilt:

    cluster_id = 0   größter Cluster
    cluster_id = 1   zweitgrößter Cluster
    cluster_id = 2   drittgrößter Cluster
    cluster_id = -1  kein Cluster nach der gewählten Mindestgröße

Die ID beschreibt die Größenrangfolge im jeweiligen Frame. Cluster 0
muss daher nicht über alle Frames exakt derselbe physische Tropfen sein.
"""

from pathlib import Path

from cluster_functions import analyze_xyz_trajectory_clusters
from visualization_functions import plot_xyz_trajectory_with_slider


# ================================================================
#   H A U P T S C H A L T E R
# ================================================================

# Neue Cluster-CSV aus der XYZ-Datei erzeugen
ANALYZE_CLUSTERS = True

# Interaktive 3D-Visualisierung öffnen
VISUALIZE = True

# Cluster in der Visualisierung unterschiedlich einfärben
#
# True  = Clusterfarben verwenden
# False = ursprüngliche Darstellung ohne Clusterfarben
CLUSTER = True


# ================================================================
#   D A T E I P F A D
# ================================================================

XYZ_PATH = Path(
    r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04\results\2026-07-15_19-30-00\my_simulation_pos.xyz"
)


# ================================================================
#   S I M U L A T I O N S P A R A M E T E R
# ================================================================

# Boxlänge aus der Simulation in nm
BOX_LENGTH_NM = 6.0

# Zeitabstand zwischen zwei gespeicherten XYZ-Frames in ps.
#
# Wenn in jedem MD-Schritt ein Frame gespeichert wird, entspricht
# dies deinem Simulationszeitschritt dt.
TIME_STEP_PS = 0.001

# Lennard-Jones sigma in nm
SIGMA_NM = 0.34


# ================================================================
#   C L U S T E R P A R A M E T E R
# ================================================================

# Maximaler Abstand zweier direkter Nachbarn in nm.
CLUSTER_CUTOFF_NM = 1.5 * SIGMA_NM

# Minimale Teilchenzahl, damit eine zusammenhängende Gruppe als
# Cluster behandelt wird.
MINIMUM_CLUSTER_SIZE = 10

# Nur jeden n-ten XYZ-Frame analysieren.
#
# Empfohlen: 1
ANALYSIS_FRAME_STRIDE = 1


# ================================================================
#   A U S G A B E D A T E I E N
# ================================================================

# Bei None werden automatisch folgende Namen neben der XYZ-Datei erzeugt:
#
#   my_simulation_pos_cluster_ids.csv
#   my_simulation_pos_cluster_summary.csv
#
# Du kannst stattdessen eigene Pfade eintragen.
CLUSTER_CSV_PATH = None
SUMMARY_CSV_PATH = None


# ================================================================
#   V I S U A L I S I E R U N G
# ================================================================

# Nur jeden n-ten Frame anzeigen.
#
# Bei großen Trajektorien beispielsweise 10 oder 20 verwenden.
PLOT_FRAME_STRIDE = 30

# Größe der dargestellten Teilchen
MARKER_SIZE = 4

# HTML-Datei zusätzlich speichern
SAVE_HTML = True


# ================================================================
#   H I L F S F U N K T I O N   F Ü R   A U T O M A T I S C H E N
#   C S V - P F A D
# ================================================================

def get_default_cluster_csv_path(xyz_path: Path) -> Path:
    """
    Gibt den Standardpfad zurück, den die Clusteranalyse verwendet.
    """
    return xyz_path.with_name(
        f"{xyz_path.stem}_cluster_ids.csv"
    )


# ================================================================
#   P R O G R A M M
# ================================================================

def main() -> None:
    """
    Führt abhängig von den Hauptschaltern die Clusteranalyse und/oder
    die Visualisierung aus.
    """
    xyz_path = Path(XYZ_PATH)

    if not xyz_path.exists():
        raise FileNotFoundError(
            f"XYZ-Datei wurde nicht gefunden:\n{xyz_path.resolve()}"
        )

    # ------------------------------------------------------------
    # 1. Clusteranalyse
    # ------------------------------------------------------------

    if ANALYZE_CLUSTERS:
        generated_cluster_csv, generated_summary_csv = (
            analyze_xyz_trajectory_clusters(
                xyz_path=xyz_path,
                box_length_nm=BOX_LENGTH_NM,
                cluster_cutoff_nm=CLUSTER_CUTOFF_NM,
                minimum_cluster_size=MINIMUM_CLUSTER_SIZE,
                time_step_ps=TIME_STEP_PS,
                frame_stride=ANALYSIS_FRAME_STRIDE,
                cluster_csv_path=CLUSTER_CSV_PATH,
                summary_csv_path=SUMMARY_CSV_PATH,
            )
        )

        cluster_csv_for_plot = generated_cluster_csv

        print("\nErzeugte Dateien:")
        print(f"Cluster-IDs: {generated_cluster_csv}")
        print(f"Zusammenfassung: {generated_summary_csv}")

    else:
        if CLUSTER_CSV_PATH is None:
            cluster_csv_for_plot = get_default_cluster_csv_path(
                xyz_path
            )
        else:
            cluster_csv_for_plot = Path(CLUSTER_CSV_PATH)

    # ------------------------------------------------------------
    # 2. Visualisierung
    # ------------------------------------------------------------

    if VISUALIZE:
        if CLUSTER and not cluster_csv_for_plot.exists():
            raise FileNotFoundError(
                "CLUSTER ist True, aber die benötigte Cluster-CSV "
                "existiert nicht:\n"
                f"{cluster_csv_for_plot.resolve()}\n\n"
                "Setze ANALYZE_CLUSTERS = True oder trage bei "
                "CLUSTER_CSV_PATH eine vorhandene Datei ein."
            )

        plot_xyz_trajectory_with_slider(
            xyz_path=xyz_path,
            box_length_nm=BOX_LENGTH_NM,
            frame_stride=PLOT_FRAME_STRIDE,
            marker_size=MARKER_SIZE,
            save_html=SAVE_HTML,
            cluster=CLUSTER,
            cluster_csv_path=(
                cluster_csv_for_plot if CLUSTER else None
            ),
        )


if __name__ == "__main__":
    main()
