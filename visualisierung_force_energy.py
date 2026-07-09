import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV einlesen
df = pd.read_csv(r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04\minimization_output\minimization_data_2026-07-08_22-20-12.csv")
dg = pd.read_csv(r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04\minimization_output\minimization_data_2026-07-08_22-21-30.csv")

#DIFF -Plots berechnen


n = 10  # erste n Werte der Differenzkurven ausblenden

diff_fmax = df["Fmax"] - dg["Fmax"]
diff_favg = df["Fmean"] - dg["Fmean"]
diff_frms = df["Frms"] - dg["Frms"]
diff_Epot = df["E_pot"] - dg["E_pot"]

diff_fmax.iloc[:n] = np.nan
diff_favg.iloc[:n] = np.nan
diff_frms.iloc[:n] = np.nan
diff_Epot.iloc[:n] = np.nan



# -----------------------------
# 1. Maximale Kraft visualisieren
# -----------------------------





plt.figure(figsize=(8, 5))

plt.plot(df["step"], df["Fmax"],  label="CG: $F_{max}$")
plt.plot(dg["step"], dg["Fmax"],  label="SD: $F_{max}$")
plt.plot(dg["step"], diff_fmax,  label="CG - SD: $F_{max}$")

plt.xlabel("Minimierungsschritt")
plt.ylabel("maximale Kraft $F_{max}$")
plt.title("Verlauf der maximalen Kraft während der Minimierung")
plt.ylim(-30, 30)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------
# 2. Mittlere Kraft visualisieren
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(df["step"], df["Fmean"], label="df: $F_{mean}$")
plt.plot(dg["step"], dg["Fmean"],  label="dg: $F_{mean}$")
plt.plot(dg["step"], diff_favg,  label="CD - SD: $F_{mean}$")


plt.xlabel("Minimierungsschritt")
plt.ylabel("mittlere Kraft $F_{mean}$")
plt.title("Verlauf der mittleren Kraft während der Minimierung")
plt.ylim(-30, 30)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------
# 3. RMS-Kraft visualisieren
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(df["step"], df["Frms"],  label="df: $F_{RMS}$")
plt.plot(dg["step"], dg["Frms"],  label="dg: $F_{RMS}$")
plt.plot(dg["step"], diff_frms,  label="CD - SD: $F_{RMS}$")


plt.xlabel("Minimierungsschritt")
plt.ylabel("RMS-Kraft $F_{RMS}$")
plt.title("Verlauf der RMS-Kraft während der Minimierung")
plt.ylim(-30, 30)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------
# 4. Energie visualisieren
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(df["step"], df["E_pot"],  label="df: $E_{pot}$")
plt.plot(dg["step"], dg["E_pot"],  label="dg: $E_{pot}$")
plt.plot(dg["step"], diff_Epot,  label="CG - SD: $E_{pot}$")


plt.xlabel("Minimierungsschritt")
plt.ylabel("potentielle Energie $E_{pot}$")
plt.title("Verlauf der potentiellen Energie während der Minimierung")
plt.ylim(-300, 30)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()