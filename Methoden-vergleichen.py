import subprocess
import sys
import time
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import R


# ----------------------------------------------------------------
#   E I N S T E L L U N G E N
# ----------------------------------------------------------------

project_dir = Path(r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04")
output_dir = project_dir / "minimization_output"

use_existing_files = True   # True: vorhandene Dateien benutzen
force_rerun = False         # True: Simulationen immer neu starten

line_width = 0.9            # dünnere Linien im Plot


# ----------------------------------------------------------------
#   B A S I S - P A R A M E T E R
# ----------------------------------------------------------------

base_args = {
    "n_particles": 200,
    "mass_argon": 39.95,
    "sigma_argon": 0.34,
    "epsilon_argon": 120 * R * 1e-3,

    "dt": 0.001,
    "n_steps": 1000,
    "temperature": 50,
    "box_length": 5,
    "tau_thermostat": 1,
    "rij_min": 1e-2,

    "NVT": True,
    "seed": 67,
    "max_steps": 6000,
}

methods = [
    {
        "label": "CG line search - a 2.0 - True",
        "SD": False,
        "recoursive_alpha": True,
        "alpha_method": "line_search",
        "rec_alpha_value": 2.0,
        "alpha_new_idea": True,
    },
        {
        "label": "SD",
        "SD": True,
        "recoursive_alpha": True,
        "alpha_method": "line_search",
        "rec_alpha_value": 2.0,
        "alpha_new_idea": False,
    },
]

"""
methods = [
    {
        "label": "CG Armijo - alpha-1.1",
        "SD": False,
        "recoursive_alpha": True,
        "alpha_method": "amijo",
        "rec_alpha_value": 1.1,
    },
    {
        "label": "CG Armijo - alpha-1.3",
        "SD": False,
        "recoursive_alpha": True,
        "alpha_method": "amijo",
        "rec_alpha_value": 1.3,
    },
    {
        "label": "CG Armijo - alpha-1.6",
        "SD": False,
        "recoursive_alpha": True,
        "alpha_method": "amijo",
        "rec_alpha_value": 1.6,
    },
    {
        "label": "CG Armijo - alpha-2.0",
        "SD": False,
        "recoursive_alpha": True,
        "alpha_method": "amijo",
        "rec_alpha_value": 2.0,
    },
]

    {
        "label": "CG armijo - a 2.0 - True",
        "SD": False,
        "recoursive_alpha": True,
        "alpha_method": "amijo",
        "rec_alpha_value": 2.0,
        "alpha_new_idea": True,
    },


"""


# ----------------------------------------------------------------
#   H I L F S F U N K T I O N E N
# ----------------------------------------------------------------

def label_to_filename(label):
    """
    Macht aus z.B. 'CG Armijo - alpha-1.1'
    den Dateinamen 'cg_armijo_alpha_1_1'.
    """
    name = label.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name


def get_named_csv_path(method):
    """
    Gibt den erwarteten Dateipfad für eine Methode zurück.
    """
    safe_name = label_to_filename(method["label"])
    return output_dir / f"minimization_{safe_name}.csv"


def newest_csv_since(start_time):
    """
    Findet die neueste minimization_data_*.csv,
    die seit start_time erzeugt wurde.
    """
    csv_files = list(output_dir.glob("minimization_data_*.csv"))

    candidates = [
        file for file in csv_files
        if file.stat().st_mtime >= start_time - 1
    ]

    if not candidates:
        raise FileNotFoundError("Keine neue minimization_data_*.csv gefunden.")

    return max(candidates, key=lambda file: file.stat().st_mtime)


def run_one_method(method):
    """
    Startet LJ_gas_run_MD.py einmal mit einer bestimmten Minimierungsmethode.
    Falls eine passende Datei schon existiert und use_existing_files=True ist,
    wird diese Datei direkt verwendet.
    """

    named_csv_file = get_named_csv_path(method)

    # Bestehende Datei verwenden
    if use_existing_files and named_csv_file.exists() and not force_rerun:
        print(f"Verwende bestehende Datei für {method['label']}:")
        print(named_csv_file)
        return named_csv_file

    # Sonst Simulation neu starten
    args = base_args.copy()
    args.update({
        "SD": method["SD"],
        "recoursive_alpha": method["recoursive_alpha"],
        "alpha_method": method["alpha_method"],
        "alpha_new_idea": method.get("alpha_new_idea", False),
    })

    if "rec_alpha_value" in method:
        args["alpha_factor"] = method["rec_alpha_value"]

    cmd = [
        sys.executable,
        "LJ_gas_run_MD.py",
    ]

    for key, value in args.items():
        cmd.extend([f"--{key}", str(value)])

    print("\n" + "=" * 70)
    print(f"Starte Methode: {method['label']}")
    print("=" * 70)

    start_time = time.time()

    subprocess.run(
        cmd,
        cwd=project_dir,
        check=True
    )

    csv_file = newest_csv_since(start_time)

    if named_csv_file.exists():
        named_csv_file.unlink()

    csv_file.rename(named_csv_file)

    print(f"Gespeichert als: {named_csv_file}")

    return named_csv_file


# ----------------------------------------------------------------
#   D A T E I E N   H O L E N   O D E R   S I M U L A T I O N E N   S T A R T E N
# ----------------------------------------------------------------

csv_paths = {}

for method in methods:
    csv_paths[method["label"]] = run_one_method(method)


# ----------------------------------------------------------------
#   C S V S   E I N L E S E N
# ----------------------------------------------------------------

data = {}

for label, path in csv_paths.items():
    data[label] = pd.read_csv(path)


# ----------------------------------------------------------------
#   P L O T - F U N K T I O N
# ----------------------------------------------------------------


def plot_quantity(
    column,
    ylabel,
    title,
    ylim=None,
    logy=False,
    filename=None,
    label_fontsize=15,
    title_fontsize=17,
    tick_fontsize=13,
    legend_fontsize=12
):
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, df in data.items():
        ax.plot(
            df["step"],
            df[column],
            label=label,
            linewidth=line_width
        )

    # Achsenbeschriftungen
    ax.set_xlabel(
        "Minimierungsschritt",
        fontsize=label_fontsize
    )
    ax.set_ylabel(
        ylabel,
        fontsize=label_fontsize
    )

    # Titel
    ax.set_title(
        title,
        fontsize=title_fontsize,
        pad=12
    )

    # Zahlen an den Achsen vergrößern
    ax.tick_params(
        axis="both",
        labelsize=tick_fontsize
    )

    if ylim is not None:
        ax.set_ylim(*ylim)

    if logy:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.4)

    ax.legend(
        fontsize=legend_fontsize
    )

    fig.tight_layout()

    # Graph speichern
    if filename is not None:
        fig.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
# ----------------------------------------------------------------
#   V E R G L E I C H S P L O T S
# ----------------------------------------------------------------

plot_quantity(
    column="Fmean",
    ylabel=r"mittlere Kraft $F_{\mathrm{mean}}$",
    title="Vergleich der mittleren Kraft während der Minimierung",
    ylim=(-30, 30),
    filename="vergleich_Fmean.png"
)

plot_quantity(
    column="Frms",
    ylabel=r"RMS-Kraft $F_{\mathrm{RMS}}$",
    title="Vergleich der RMS-Kraft während der Minimierung",
    ylim=(-30, 30),
    filename="vergleich_Frms.png"
)

plot_quantity(
    column="E_pot",
    ylabel=r"potentielle Energie $E_{\mathrm{pot}}$",
    title="Vergleich der potentiellen Energie während der Minimierung",
    ylim=(-2600, 30),
    filename="vergleich_Epot.png"
)