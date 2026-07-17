import subprocess
import sys
from scipy.constants import R
from pathlib import Path

#----------------------------------------------------------------
#   P A R A M E T E R S
#----------------------------------------------------------------
# system
n_particles = 500
mass_argon = 39.95              # mass in u = 1e-3 kg/mol
sigma_argon = 0.34              # sigma in nm
epsilon_argon = 200 * R * 1e-3  # epsilon in kJ/mol

# simulation
dt = 0.001
n_steps = 30000
temperature = 80
box_length = 6
tau_thermostat = 1
rij_min = 1e-2
NVT = True
seed = 101
SD = False
max_steps = 6000
recoursive_alpha = True
alpha_method = "amijo"
alpha_factor = 2.0
alpha_new_idea = True

project_dir = (
    Path.home()
    / "Documents"
    / "VSCODE"
    / "moldyn_proj"
    / "md-project"
)

run_script = project_dir / "LJ_gas_run_MD.py"

if not run_script.is_file():
    raise FileNotFoundError(
        f"LJ_gas_run_MD.py wurde nicht gefunden: {run_script}"
    )

subprocess.run(
    [
        sys.executable,
        "LJ_gas_run_MD.py",

        "--n_particles", str(n_particles),
        "--mass_argon", str(mass_argon),
        "--sigma_argon", str(sigma_argon),
        "--epsilon_argon", str(epsilon_argon),

        "--dt", str(dt),
        "--n_steps", str(n_steps),
        "--temperature", str(temperature),
        "--box_length", str(box_length),
        "--tau_thermostat", str(tau_thermostat),
        "--rij_min", str(rij_min),

        "--NVT", str(NVT),
        "--seed", str(seed),
        "--SD", str(SD),
        "--recoursive_alpha", str(recoursive_alpha),
        "--alpha_method", str(alpha_method),
        "--alpha_factor", str(alpha_factor),
        "--alpha_new_idea", str(alpha_new_idea),
        "--max_steps", str(max_steps)
    ],
    cwd=project_dir,
    check=True
)