from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2


# ================================================================
# EINGABEN
# ================================================================

ENERGY_FILE = Path(
    "results/Long_sim_120/my_simulation_ene.npy"
)

# Solltemperatur des Thermostaten
TARGET_TEMPERATURE_K = 120.0

# Anzahl der Teilchen
N_PARTICLES = 500

# Zeitschritt zwischen zwei gespeicherten Frames in ps
DT_PS = 0.001

# Konfidenzbereich:
# 0.95 entspricht einem theoretischen 95-%-Bereich
CONFIDENCE_LEVEL = 0.95

# Optional:
# Nur jeden n-ten Wert plotten, damit der Plot bei langen
# Simulationen nicht zu groß wird.
PLOT_STRIDE = 1

# Optional:
# Anfang der Simulation im Plot ausblenden.
# Die Zeitachse bleibt trotzdem korrekt.
START_FRAME = 0


# ================================================================
# DATEI EINLESEN
# ================================================================

energy_data = np.load(ENERGY_FILE)

if energy_data.ndim != 2 or energy_data.shape[1] < 4:
    raise ValueError(
        "Die Energiedatei muss mindestens vier Spalten besitzen:\n"
        "E_pot, E_kin, T, P"
    )

# Spalten:
# 0 = potentielle Energie
# 1 = kinetische Energie
# 2 = Temperatur
# 3 = Druck
temperature_K = energy_data[:, 2]

n_frames = len(temperature_K)

time_ps = np.arange(n_frames) * DT_PS


# ================================================================
# FREIHEITSGRADE
# ================================================================

# Das entspricht deiner momentanen Temperaturberechnung:
#
# T = 2 E_kin / (3 N R)
#
degrees_of_freedom = 3 * N_PARTICLES

# Falls die Schwerpunktsbewegung dauerhaft entfernt und nicht erneut
# durch das Thermostat angeregt wird, könnte man alternativ verwenden:
#
# degrees_of_freedom = 3 * N_PARTICLES - 3


# ================================================================
# THEORETISCH ERWARTETER TEMPERATURBEREICH
# ================================================================

alpha = 1.0 - CONFIDENCE_LEVEL

# Für ein kanonisches Ensemble gilt:
#
# degrees_of_freedom * T_inst / T_target
#     ~ Chi-Quadrat(degrees_of_freedom)
#
# Daraus ergeben sich die Quantile für T_inst.

lower_temperature_K = (
    TARGET_TEMPERATURE_K
    / degrees_of_freedom
    * chi2.ppf(
        alpha / 2.0,
        df=degrees_of_freedom
    )
)

upper_temperature_K = (
    TARGET_TEMPERATURE_K
    / degrees_of_freedom
    * chi2.ppf(
        1.0 - alpha / 2.0,
        df=degrees_of_freedom
    )
)

# Erwartete Standardabweichung der momentanen Temperatur
expected_temperature_std_K = (
    TARGET_TEMPERATURE_K
    * np.sqrt(
        2.0 / degrees_of_freedom
    )
)


# ================================================================
# DATEN FÜR DEN PLOT AUSWÄHLEN
# ================================================================

plot_indices = np.arange(
    START_FRAME,
    n_frames,
    PLOT_STRIDE
)

plot_time_ps = time_ps[plot_indices]
plot_temperature_K = temperature_K[plot_indices]


# ================================================================
# ERGEBNISSE AUSGEBEN
# ================================================================

print("=" * 70)
print("THEORETISCH ERWARTETER TEMPERATURBEREICH")
print("=" * 70)

print(f"Teilchenzahl:             {N_PARTICLES}")
print(f"Freiheitsgrade:          {degrees_of_freedom}")
print(f"Solltemperatur:          {TARGET_TEMPERATURE_K:.3f} K")
print(
    f"Konfidenzbereich:        "
    f"{100 * CONFIDENCE_LEVEL:.1f} %"
)
print(
    f"Untere Grenze:           "
    f"{lower_temperature_K:.3f} K"
)
print(
    f"Obere Grenze:            "
    f"{upper_temperature_K:.3f} K"
)
print(
    f"Erwartete Standardabw.:  "
    f"{expected_temperature_std_K:.3f} K"
)

measured_mean = np.mean(
    temperature_K[START_FRAME:]
)

measured_std = np.std(
    temperature_K[START_FRAME:],
    ddof=1
)

fraction_inside = np.mean(
    (
        temperature_K[START_FRAME:]
        >= lower_temperature_K
    )
    & (
        temperature_K[START_FRAME:]
        <= upper_temperature_K
    )
)

print()
print(
    f"Gemessener Mittelwert:   "
    f"{measured_mean:.3f} K"
)
print(
    f"Gemessene Standardabw.:  "
    f"{measured_std:.3f} K"
)
print(
    f"Anteil innerhalb Range:  "
    f"{100 * fraction_inside:.2f} %"
)

print("=" * 70)


# ================================================================
# PLOT
# ================================================================

plt.figure(figsize=(11, 6))

# Erwarteter Temperaturbereich
plt.fill_between(
    plot_time_ps,
    lower_temperature_K,
    upper_temperature_K,
    alpha=0.25,
    label=(
        f"theoretischer "
        f"{100 * CONFIDENCE_LEVEL:.0f}-%-Bereich"
    )
)

# Simulierte Temperatur
plt.plot(
    plot_time_ps,
    plot_temperature_K,
    linewidth=0.8,
    label="simulierte Temperatur"
)

# Solltemperatur
plt.axhline(
    TARGET_TEMPERATURE_K,
    linestyle="--",
    linewidth=1.5,
    label=(
        f"Solltemperatur "
        f"{TARGET_TEMPERATURE_K:.1f} K"
    )
)

# Grenzen zusätzlich als Linien
plt.axhline(
    lower_temperature_K,
    linestyle=":",
    linewidth=1
)

plt.axhline(
    upper_temperature_K,
    linestyle=":",
    linewidth=1
)

plt.xlabel("Zeit / ps")
plt.ylabel("Temperatur / K")

plt.title(
    "Momentane Temperatur und theoretisch erwarteter "
    "kanonischer Schwankungsbereich"
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()