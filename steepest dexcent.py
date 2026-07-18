"""
Steepest Descent auf einer 2D-Funktion, visualisiert als Contourplot.

Die Zielfunktion ist eine elongierte quadratische Schüssel
f(x, y) = a*x^2 + b*y^2
Durch die unterschiedliche Krümmung in x- und y-Richtung (a != b) zeigt
Steepest Descent das typische Zickzack-Verhalten sehr schön.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1) Zielfunktion und Gradient
# ---------------------------------------------------------
a, b = 1.0, 10.0  # unterschiedliche Krümmung -> Zickzack-Effekt

def f(x, y):
    return a * x**2 + b * y**2

def grad_f(x, y):
    return np.array([2 * a * x, 2 * b * y])


# ---------------------------------------------------------
# 2) Steepest Descent mit Backtracking-Linesearch (Armijo)
# ---------------------------------------------------------
def steepest_descent(start, n_steps=25, alpha0=1.0, rho=0.5, c=1e-4):
    path = [np.array(start, dtype=float)]
    x = np.array(start, dtype=float)

    for _ in range(n_steps):
        g = grad_f(*x)
        if np.linalg.norm(g) < 1e-8:
            break

        # Suchrichtung: negativer Gradient
        d = -g

        # Armijo-Backtracking-Linesearch
        alpha = alpha0
        fx = f(*x)
        while f(*(x + alpha * d)) > fx + c * alpha * np.dot(g, d):
            alpha *= rho

        x = x + alpha * d
        path.append(x.copy())

    return np.array(path)


# ---------------------------------------------------------
# 3) Pfad berechnen (wenige Schritte, damit man sie einzeln sieht)
# ---------------------------------------------------------
start_point = (4.0, 2.0)
path = steepest_descent(start_point, n_steps=5)

# ---------------------------------------------------------
# 4) Einfacher Contourplot mit Optimierungspfad
# ---------------------------------------------------------
x_range = np.linspace(-6, 6, 400)
y_range = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)

fig, ax = plt.subplots(figsize=(9, 5))

# Alle Höhenlinien in EINER Farbe -> kein Verblassen, klar als
# "Linien gleichen Funktionswerts" erkennbar
ax.contour(X, Y, Z, levels=10, colors="steelblue", linewidths=1)

# Pfad einzeichnen
ax.plot(path[:, 0], path[:, 1], "o-", color="crimson",
         markersize=7, linewidth=2, label="Weg zum Minimum (Steepest Descent)")

# Start- und Endpunkt hervorheben
ax.plot(*path[0], "o", color="black", markersize=10, label="Start")
ax.plot(*path[-1], ".", color="blue", markeredgecolor="black",
         markersize=18, label="Minimum")

# Schrittpfeile (roter Pfad = tatsächlich gegangener Weg)
black_tips = []
for i in range(len(path) - 1):
    start = path[i]
    step_vec = path[i + 1] - start
    if np.linalg.norm(step_vec) == 0:
        continue
    # Roter Pfeil: genau bis zum nächsten Punkt
    ax.annotate("", xy=path[i + 1], xytext=start,
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.8))
    # Schwarzer Pfeil: tatsächliche Schrittlänge, entgegengesetzte
    # Richtung (= Gradientenrichtung)
    tip = start - step_vec
    black_tips.append(tip)
    ax.annotate("", xy=tip, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color="black",
                                 lw=1, alpha=0.6))

# Achsenbereich: fester Grundbereich, aber erweitert falls nötig, damit
# auch die schwarzen Pfeilspitzen nicht abgeschnitten werden
all_x = np.hstack([path[:, 0], np.array(black_tips)[:, 0]]) if black_tips else path[:, 0]
all_y = np.hstack([path[:, 1], np.array(black_tips)[:, 1]]) if black_tips else path[:, 1]
pad = 0.5
xmin = min(-5, all_x.min() - pad)
xmax = max(5, all_x.max() + pad)
ymin = min(-3, all_y.min() - pad)
ymax = max(3, all_y.max() + pad)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.set_title("Steepest Descent")
ax.plot([], [], color="steelblue", label="Höhenlinien (gleicher Funktionswert)")
ax.plot([], [], color="black", alpha=0.6, label="Gradientenrichtung")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
          borderaxespad=0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(r"C:\Users\morit\_Uni-FU\Semester 4\Molekueldynamik\md-project-08-group-04\steepest_descent_contour.png", dpi=150)
plt.show()