import pandas as pd
import matplotlib.pyplot as plt

# CSV einlesen
df = pd.read_csv(r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04\minimization_output\minimization_data_2026-07-08_22-20-12.csv")
dg = pd.read_csv(r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04\minimization_output\minimization_data_2026-07-08_22-21-30.csv")


# -----------------------------
# 1. Maximale Kraft visualisieren
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(df["step"], df["Fmax"], marker="o", label="df: $F_{max}$")
plt.plot(dg["step"], dg["Fmax"], marker="o", label="dg: $F_{max}$")

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

plt.plot(df["step"], df["Fmean"], marker="o", label="df: $F_{mean}$")
plt.plot(dg["step"], dg["Fmean"], marker="o", label="dg: $F_{mean}$")

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

plt.plot(df["step"], df["Frms"], marker="o", label="df: $F_{RMS}$")
plt.plot(dg["step"], dg["Frms"], marker="o", label="dg: $F_{RMS}$")

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

plt.plot(df["step"], df["E_pot"], marker="o", label="df: $E_{pot}$")
plt.plot(dg["step"], dg["E_pot"], marker="o", label="dg: $E_{pot}$")

plt.xlabel("Minimierungsschritt")
plt.ylabel("potentielle Energie $E_{pot}$")
plt.title("Verlauf der potentiellen Energie während der Minimierung")
plt.ylim(-300, 30)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()