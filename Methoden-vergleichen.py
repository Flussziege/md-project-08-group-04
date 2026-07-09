import subprocess
import sys
import time
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import R


# ----------------------------------------------------------------
#   P A R A M E T E R S
# ----------------------------------------------------------------

project_dir = Path(r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04")
output_dir = project_dir / "minimization_output"

# system
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
    "seed": 42,
    "max_steps": 3000,
}


# Falls du in argparse wirklich "amijo" geschrieben hast, dann hier "amijo" lassen.
# Besser wäre langfristig: überall zu "armijo" korrigieren.
ARMİJO_NAME = "amijo"


methods = [
    {
        "label": "CG line search",
        "SD": False,
        "recoursive_alpha": False,
        "alpha_method": "line_search",
    },
    {
        "label": "CG line search recursive alpha",
        "SD": False,
        "recoursive_alpha": True,
        "alpha_method": "line_search",
    },
    {
        "label": "CG Armijo",
        "SD": False,
        "recoursive_alpha": False,
        "alpha_method": "amijo",
    },
]

def label_to_filename(label):
    """
    Macht aus 'CG line search' einen sicheren Dateinamen.
    """
    name = label.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name


def newest_csv_since(start_time):
    """
    Findet die neueste CSV-Datei, die nach start_time erzeugt wurde.
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
    Danach wird die neu erzeugte CSV sinnvoll umbenannt.
    """

    args = base_args.copy()
    args.update({
        "SD": method["SD"],
        "recoursive_alpha": method["recoursive_alpha"],
        "alpha_method": method["alpha_method"],
    })

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

    safe_name = label_to_filename(method["label"])
    new_csv_file = output_dir / f"minimization_{safe_name}.csv"

    if new_csv_file.exists():
        new_csv_file.unlink()

    csv_file.rename(new_csv_file)

    print(f"Gespeichert als: {new_csv_file}")

    return new_csv_file


# ----------------------------------------------------------------
#   A L L E   M E T H O D E N   S T A R T E N
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

def plot_quantity(column, ylabel, title, ylim=None, logy=False):
    plt.figure(figsize=(8, 5))

    for label, df in data.items():
        plt.plot(df["step"], df[column], label=label)

    plt.xlabel("Minimierungsschritt")
    plt.ylabel(ylabel)
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)

    if logy:
        plt.yscale("log")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------
#   V E R G L E I C H S P L O T S
# ----------------------------------------------------------------

plot_quantity(
    column="Fmax",
    ylabel="maximale Kraft $F_{max}$",
    title="Vergleich der maximalen Kraft während der Minimierung",
    logy=False,
    ylim=(-30,30)
)

plot_quantity(
    column="Fmean",
    ylabel="mittlere Kraft $F_{mean}$",
    title="Vergleich der mittleren Kraft während der Minimierung",
    logy=False,
    ylim=(-30,30)
)

plot_quantity(
    column="Frms",
    ylabel="RMS-Kraft $F_{RMS}$",
    title="Vergleich der RMS-Kraft während der Minimierung",
    logy=False,
    ylim=(-30,30)
)

plot_quantity(
    column="E_pot",
    ylabel="potentielle Energie $E_{pot}$",
    title="Vergleich der potentiellen Energie während der Minimierung",
    logy=False,
    ylim=(-600,30)
)