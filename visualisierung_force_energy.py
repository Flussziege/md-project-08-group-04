import pandas as pd
import matplotlib.pyplot as plt

# CSV einlesen
df = pd.read_csv(r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04\minimization_output\minimization_data_2026-07-08_21-08-41.csv")

# -----------------------------
# 1. Kraft visualisieren
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(df["step"], df["Fmax"], marker="o")

plt.xlabel("Minimierungsschritt")
plt.ylabel("maximale Kraft $F_{max}$")
plt.title("Verlauf der maximalen Kraft während der Minimierung")

plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# 2. Energie visualisieren
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(df["step"], df["E_pot"], marker="o")

plt.xlabel("Minimierungsschritt")
plt.ylabel("potentielle Energie $E_{pot}$")
plt.title("Verlauf der potentiellen Energie während der Minimierung")

plt.grid(True)
plt.tight_layout()
plt.show()