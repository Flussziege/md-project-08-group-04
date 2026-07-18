"""
Temperature fluctuation analysis for an NVT molecular dynamics simulation
==========================================================================

PURPOSE
-------
This script reads the temperature trajectory from a NumPy energy file and
compares the simulated instantaneous temperature with the theoretically
expected temperature fluctuations of a canonical NVT ensemble.

The energy file is assumed to contain at least four columns:

    column 0: potential energy
    column 1: kinetic energy
    column 2: instantaneous temperature
    column 3: pressure


HOW TO USE
----------
1. Set ENERGY_FILE to the path of the file:

       my_simulation_ene.npy

2. Set TARGET_TEMPERATURE_K to the thermostat target temperature.

3. Set N_PARTICLES to the number of particles in the simulation.

4. Set DT_PS to the time interval between two stored frames.

5. If the beginning of the simulation is not equilibrated, increase
   START_FRAME. Only frames from START_FRAME onward are then used for
   calculating the measured mean and standard deviation.

6. Run the script:

       python temperature_analysis.py


THEORETICAL BACKGROUND
----------------------
The instantaneous temperature is calculated from the kinetic energy:

                2 E_kin
    T_inst = -------------
                f R

where

    E_kin = total kinetic energy,
    f     = number of kinetic degrees of freedom,
    R     = molar gas constant.

For a system with N particles in three dimensions:

    f = 3N

If the center-of-mass motion is removed permanently, one may instead use:

    f = 3N - 3


In a canonical ensemble, the kinetic energy follows a chi-square distribution.
Therefore:

    f T_inst
    --------  ~ chi-square(f)
    T_target


The theoretical confidence interval of the instantaneous temperature is:

                      T_target
    T_lower = -------------------------- chi2(alpha / 2, f)
                          f


                      T_target
    T_upper = -------------------------- chi2(1 - alpha / 2, f)
                          f

where

    alpha = 1 - confidence level.


The expected standard deviation of the instantaneous temperature is:

                                2
    sigma_T = T_target sqrt( ------- )
                                f


IMPORTANT
---------
Successive MD frames are usually correlated. Therefore, the fraction of frames
inside the theoretical interval is useful as a diagnostic, but it is not an
independent statistical hypothesis test.
"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2


# ================================================================
# INPUT PARAMETERS
# ================================================================

ENERGY_FILE = Path(
    "results/Long_sim/my_simulation_ene.npy"
)

# Thermostat target temperature in kelvin
TARGET_TEMPERATURE_K = 80.0

# Number of particles
N_PARTICLES = 500

# Time interval between two stored frames in picoseconds
DT_PS = 0.001

# Theoretical confidence interval
# 0.95 corresponds to a theoretical 95% interval
CONFIDENCE_LEVEL = 0.95

# Plot only every nth value to reduce the number of plotted points
PLOT_STRIDE = 1

# Ignore the first frames in the statistical analysis and plot
START_FRAME = 0


# ================================================================
# PLOT APPEARANCE
# ================================================================

TITLE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 17
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 13

FIGURE_WIDTH = 12
FIGURE_HEIGHT = 7

# Set general matplotlib font sizes
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
    }
)


# ================================================================
# LOAD ENERGY FILE
# ================================================================

if not ENERGY_FILE.exists():
    raise FileNotFoundError(
        f"Energy file not found:\n{ENERGY_FILE.resolve()}"
    )

energy_data = np.load(ENERGY_FILE)

if energy_data.ndim != 2 or energy_data.shape[1] < 4:
    raise ValueError(
        "The energy file must be a two-dimensional array with at least "
        "four columns:\n"
        "potential energy, kinetic energy, temperature, pressure"
    )

if N_PARTICLES <= 0:
    raise ValueError("N_PARTICLES must be greater than zero.")

if DT_PS <= 0:
    raise ValueError("DT_PS must be greater than zero.")

if not 0.0 < CONFIDENCE_LEVEL < 1.0:
    raise ValueError(
        "CONFIDENCE_LEVEL must be between 0 and 1."
    )

if PLOT_STRIDE < 1:
    raise ValueError("PLOT_STRIDE must be at least 1.")


# Columns:
# 0 = potential energy
# 1 = kinetic energy
# 2 = instantaneous temperature
# 3 = pressure
temperature_K = energy_data[:, 2]

n_frames = len(temperature_K)

if not 0 <= START_FRAME < n_frames:
    raise ValueError(
        f"START_FRAME must be between 0 and {n_frames - 1}."
    )

time_ps = np.arange(n_frames) * DT_PS


# ================================================================
# DEGREES OF FREEDOM
# ================================================================

# This corresponds to the temperature definition:
#
#                 2 E_kin
#     T_inst = ----------------
#                 3 N R
#
degrees_of_freedom = 3 * N_PARTICLES

# If the center-of-mass motion is permanently removed and is not
# reintroduced by the thermostat, use:
#
# degrees_of_freedom = 3 * N_PARTICLES - 3


# ================================================================
# THEORETICAL TEMPERATURE INTERVAL
# ================================================================

alpha = 1.0 - CONFIDENCE_LEVEL

# In a canonical ensemble:
#
#     degrees_of_freedom * T_inst / T_target
#         ~ chi-square(degrees_of_freedom)
#
# The lower and upper temperature limits are obtained from the
# corresponding chi-square quantiles.

lower_temperature_K = (
    TARGET_TEMPERATURE_K
    / degrees_of_freedom
    * chi2.ppf(
        alpha / 2.0,
        df=degrees_of_freedom,
    )
)

upper_temperature_K = (
    TARGET_TEMPERATURE_K
    / degrees_of_freedom
    * chi2.ppf(
        1.0 - alpha / 2.0,
        df=degrees_of_freedom,
    )
)

# Expected standard deviation:
#
#     sigma_T = T_target * sqrt(2 / degrees_of_freedom)
#
expected_temperature_std_K = (
    TARGET_TEMPERATURE_K
    * np.sqrt(
        2.0 / degrees_of_freedom
    )
)


# ================================================================
# SELECT DATA FOR ANALYSIS AND PLOTTING
# ================================================================

analysis_temperature_K = temperature_K[START_FRAME:]

plot_indices = np.arange(
    START_FRAME,
    n_frames,
    PLOT_STRIDE,
)

plot_time_ps = time_ps[plot_indices]
plot_temperature_K = temperature_K[plot_indices]


# ================================================================
# CALCULATE MEASURED STATISTICS
# ================================================================

measured_mean_K = np.mean(
    analysis_temperature_K
)

measured_std_K = np.std(
    analysis_temperature_K,
    ddof=1,
)

inside_interval = (
    (
        analysis_temperature_K
        >= lower_temperature_K
    )
    & (
        analysis_temperature_K
        <= upper_temperature_K
    )
)

fraction_inside = np.mean(
    inside_interval
)


# ================================================================
# PRINT RESULTS
# ================================================================

print("=" * 72)
print("THEORETICAL TEMPERATURE FLUCTUATION ANALYSIS")
print("=" * 72)

print(f"Energy file:                  {ENERGY_FILE}")
print(f"Number of particles:          {N_PARTICLES}")
print(f"Degrees of freedom:           {degrees_of_freedom}")
print(f"Thermostat target:            {TARGET_TEMPERATURE_K:.3f} K")
print(
    f"Confidence level:             "
    f"{100 * CONFIDENCE_LEVEL:.1f} %"
)
print(
    f"Theoretical lower limit:      "
    f"{lower_temperature_K:.3f} K"
)
print(
    f"Theoretical upper limit:      "
    f"{upper_temperature_K:.3f} K"
)
print(
    f"Expected standard deviation:  "
    f"{expected_temperature_std_K:.3f} K"
)

print("-" * 72)

print(
    f"Analyzed frames:              "
    f"{len(analysis_temperature_K)}"
)
print(
    f"Measured mean temperature:    "
    f"{measured_mean_K:.3f} K"
)
print(
    f"Measured standard deviation:  "
    f"{measured_std_K:.3f} K"
)
print(
    f"Frames inside interval:       "
    f"{100 * fraction_inside:.2f} %"
)

print("=" * 72)


# ================================================================
# PLOT
# ================================================================

fig, ax = plt.subplots(
    figsize=(
        FIGURE_WIDTH,
        FIGURE_HEIGHT,
    )
)

# Theoretically expected temperature interval
ax.fill_between(
    plot_time_ps,
    lower_temperature_K,
    upper_temperature_K,
    alpha=0.25,
    label=(
        f"Theoretical "
        f"{100 * CONFIDENCE_LEVEL:.0f}% interval"
    ),
)

# Simulated instantaneous temperature
ax.plot(
    plot_time_ps,
    plot_temperature_K,
    linewidth=1.0,
    label="Simulated instantaneous temperature",
)

# Thermostat target temperature
ax.axhline(
    TARGET_TEMPERATURE_K,
    linestyle="--",
    linewidth=2.0,
    label=(
        f"Target temperature: "
        f"{TARGET_TEMPERATURE_K:.1f} K"
    ),
)

# Lower and upper theoretical limits
ax.axhline(
    lower_temperature_K,
    linestyle=":",
    linewidth=1.5,
)

ax.axhline(
    upper_temperature_K,
    linestyle=":",
    linewidth=1.5,
)

ax.set_xlabel(
    "Time / ps",
    fontsize=AXIS_LABEL_FONT_SIZE,
)

ax.set_ylabel(
    "Instantaneous temperature / K",
    fontsize=AXIS_LABEL_FONT_SIZE,
)

ax.set_title(
    "Instantaneous Temperature and the Expected "
    "Canonical Fluctuation Interval",
    fontsize=TITLE_FONT_SIZE,
    pad=15,
)

ax.tick_params(
    axis="both",
    labelsize=TICK_FONT_SIZE,
)

ax.grid(
    True,
    alpha=0.4,
)

ax.legend(
    fontsize=LEGEND_FONT_SIZE,
)

fig.tight_layout()

plt.show()