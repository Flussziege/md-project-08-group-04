import csv
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil

import LJ_gas

def max_force_norm(F):
    return np.max(np.linalg.norm(F, axis=1))

def backtracking_line_search(
        x,
        p,
        ps,
        sim,
        E_old, 
        energy_func, 
        alpha0=1e-3
        ):
    
    alpha = alpha0

    for _ in range(50):
        x_new = np.mod(x + alpha * p, sim.box_length)
        E_new = energy_func(x_new, ps, sim)

        if E_new < E_old:
            return alpha

        alpha *= 0.5

    return 0.0   #wenn kein alpha gefunden wird, gibt es alpha = 0 zurück

def backtracking_amijo(x, p, ps, sim, E_old, F, energy_func, alpha0, c1=1e-4):
    slope = np.sum(F * p)  # Richtungsableitung, sollte > 0 sein
    if slope <= 0:
        return 0.0  # p ist keine Abstiegsrichtung

    alpha = alpha0
    for _ in range(50):
        x_new = np.mod(x + alpha * p, sim.box_length)
        E_new = energy_func(x_new, ps, sim)
        if E_new <= E_old - c1 * alpha * slope:
            return alpha
        alpha *= 0.5
    return 0.0

def energy_func(
        x_new,
        ps: LJ_gas.ParticleSystem, 
        sim: LJ_gas.SimulationParameters, 
        ):
    old_position = ps.position.copy()

    ps.position = x_new.copy()
    E_new = LJ_gas.potential_energy(ps, sim)

    ps.position = old_position
    return E_new
    

def minimise_starting_position(
        ps: LJ_gas.ParticleSystem, 
        sim: LJ_gas.SimulationParameters, 
        SD: bool = False,
        recursive_alpha: bool = False,
        alpha_method: str = "line_search",
        alpha_factor: float = 2.0,
        tolerance =1e-5,
        alpha_SD = 1e-5,
        max_steps = 1000,
        alpha_new_idea: bool = False
        ):

    #Listen um Werte zu speichern
    E_hist = []
    force_hist = []
    Fmax_hist = []
    pos_hist = []
    alpha_hist = []
    p_hist = []
    beta_hist = []

    #startkräfte und start-E_pot
    LJ_gas.calculate_force(ps, sim)
    E = LJ_gas.potential_energy(ps, sim)
    #position is ps.position  (ps.n, 3) 

    #hier ersten Werte  speichern
    #wenn man nicht copiert, wird nur die ref zum array, nicht der wert gespeichert,
    #welche später überschrieben wird
    p = ps.force.copy()
    p_hist.append(p.copy())   
    pos_hist.append(ps.position.copy())
    E_hist.append(E)
    Fmax_hist.append(max_force_norm(ps.force))
    force_hist.append(ps.force.copy()) 
    alpha = 1e-4
    alpha0 = 1e-4
    alpha_scale = 1.0
    alpha_new_idea_scale = alpha_factor if alpha_new_idea else 1.0

    #just to be safe
    step = 0


    while max_force_norm(ps.force) >= tolerance and step < max_steps: 
        # alpha bestimmen
        # position ändern
        # neue Kraft berechnen
        # neue Energie berechnen
        # neue Suchrichtung berechnen

        if alpha_new_idea and step > 0 and step % 1000 == 0:
            alpha_new_idea_scale = max(0.1, alpha_new_idea_scale - 0.2)
            print(f"alpha_new_idea_scale: {alpha_new_idea_scale}")

        if SD:
            alpha = alpha_SD
        elif alpha_method == "amijo":
            if recursive_alpha and alpha > 0:
                if alpha_new_idea:
                    alpha0 = alpha_new_idea_scale * alpha
                else:
                    alpha0 = alpha * alpha_factor
            else:
                alpha0 = alpha0

            alpha = backtracking_amijo(
                x=ps.position,
                p=p,
                ps=ps,
                sim=sim,
                E_old=E,
                F=ps.force,
                energy_func=energy_func,
                alpha0=alpha0,
                c1=1e-2
            )
        elif alpha_method == "line_search":
            if recursive_alpha and alpha > 0:
                if alpha_new_idea:
                    alpha0 = alpha_new_idea_scale * alpha
                else:
                    alpha0 = alpha * alpha_factor
            else:
                alpha0 = alpha0

            alpha = backtracking_line_search(
                x=ps.position,
                p=p,
                ps=ps,
                sim=sim,
                E_old=E,
                energy_func=energy_func,
                alpha0=alpha0
            )
        elif alpha_method == "fixed":
            alpha0 = alpha0 
            alpha = backtracking_line_search(
                x=ps.position,
                p=p,
                ps=ps,
                sim=sim,
                E_old=E,
                energy_func=energy_func,
                alpha0=alpha0
            )
        else:
            raise ValueError(
                f"Unbekannte alpha_method: {alpha_method}. Verwenden Sie 'fixed', 'line_search' oder 'amijo'."
            )
           
        if alpha == 0.0:
            print("Keine passende Schrittweite gefunden.")
            break
        

        #Schritt gehen
        ps.position = ps.position + alpha * p 

        #periodic boundary anwenden
        LJ_gas.apply_periodic_boundary(ps, sim)

        #für berechnung von beta speichern
        F_old = ps.force.copy()
        p_old = p.copy()

        #neue Kräfte berechnen
        LJ_gas.calculate_force(ps, sim)
        E = LJ_gas.potential_energy(ps, sim)

        if SD:
            p = ps.force.copy()
            beta = 0.0
        else:
            #beta berechnen mit toller formel
            beta = np.sum(ps.force * (ps.force - F_old)) / np.sum(F_old * F_old)

            #darf halt nciht negativ sein
            beta = max(beta, 0.0)

            p_new = ps.force + beta * p_old


            #wenn CG keinen fortschritt gemacht werden kann,
            #wird auf SD zurückgegriffen und beta = 0
            if np.sum(ps.force * p_new) <= 0: 
                p_new = ps.force.copy()
                beta = 0.0
            
            #Wert übernehmen
            p = p_new.copy()

        #neuen Werte speichern
        p_hist.append(p.copy())   
        pos_hist.append(ps.position.copy())
        E_hist.append(E)
        force_hist.append(ps.force.copy()) 
        Fmax_hist.append(max_force_norm(ps.force))
        alpha_hist.append(alpha)
        beta_hist.append(beta)

        step += 1
        print(step)


    print(f"n_steps: {step}")
    print(f"converged: {max_force_norm(ps.force) < tolerance}")
    print(f"final_energy: {E}")
    print(f"final_Fmax: {max_force_norm(ps.force)}")
    #print(f"avg. Force: {np.mean(np.linalg.norm(ps.force, axis=2), axis=1)}")

    return {
        "positions": ps.position.copy(),
        "E_hist": np.array(E_hist),
        "force_hist": np.array(force_hist), 
        "Fmax_hist": np.array(Fmax_hist),
        "pos_hist": np.array(pos_hist),
        "alpha_hist": np.array(alpha_hist),
        "p_hist": np.array(p_hist),
        "beta_hist": np.array(beta_hist),
    }


#the function that safes the data and creates a new text file with the data in it

def create_minimization_filename(output_dir):
    """
    Erzeugt einmalig einen Dateinamen mit Datum und Uhrzeit.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = output_dir / f"minimization_data_{timestamp}.csv"

    return filename


from pathlib import Path
import csv
import numpy as np

def write_minimization_result_to_csv(filename, result):
    """
    Schreibt die komplette Minimierungs-Historie aus minimise_starting_position()
    als CSV-Datei.
    """

    filename = Path(filename)

    required_keys = [
        "E_hist", 
        "Fmax_hist", 
        "force_hist", 
        "pos_hist", 
        "alpha_hist", 
        "p_hist", 
        "beta_hist"
    ]

    missing = [key for key in required_keys if key not in result]
    if missing:
        raise KeyError(f"result fehlt folgende Schlüssel: {missing}")

    E_hist = np.asarray(result["E_hist"], dtype=float)
    Fmax_hist = np.asarray(result["Fmax_hist"], dtype=float)
    force_hist = np.asarray(result["force_hist"], dtype=float)
    pos_hist = np.asarray(result["pos_hist"], dtype=float)
    alpha_hist = np.asarray(result["alpha_hist"], dtype=float)
    p_hist = np.asarray(result["p_hist"], dtype=float)
    beta_hist = np.asarray(result["beta_hist"], dtype=float)

    if pos_hist.ndim != 3 or pos_hist.shape[2] != 3:
        raise ValueError("pos_hist muss die Form (n_steps, n_particles, 3) haben.")

    if force_hist.shape != pos_hist.shape:
        raise ValueError("force_hist muss dieselbe Form wie pos_hist haben.")

    if p_hist.shape != pos_hist.shape:
        raise ValueError("p_hist muss dieselbe Form wie pos_hist haben.")

    if E_hist.shape[0] != pos_hist.shape[0] or Fmax_hist.shape[0] != pos_hist.shape[0]:
        raise ValueError("E_hist und Fmax_hist müssen dieselbe Anzahl an Schritten wie pos_hist haben.")

    n_steps = pos_hist.shape[0]
    n_particles = pos_hist.shape[1]

    # Kraftbeträge pro Schritt und Teilchen
    force_norms = np.linalg.norm(force_hist, axis=2)

    # mittlere Kraft und RMS-Kraft pro Schritt
    Fmean_hist = np.mean(force_norms, axis=1)
    Frms_hist = np.sqrt(np.mean(force_norms**2, axis=1))

    header = ["step", "E_pot", "Fmax", "Fmean", "Frms", "alpha", "beta"]

    for particle in range(n_particles):
        header.extend([
            f"pos_{particle}_x",
            f"pos_{particle}_y",
            f"pos_{particle}_z",
            f"force_{particle}_x",
            f"force_{particle}_y",
            f"force_{particle}_z",
            f"p_{particle}_x",
            f"p_{particle}_y",
            f"p_{particle}_z",
        ])

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for step in range(n_steps):
            row = [
                step,
                E_hist[step],
                Fmax_hist[step],
                Fmean_hist[step],
                Frms_hist[step],
                alpha_hist[step] if step < len(alpha_hist) else np.nan,
                beta_hist[step] if step < len(beta_hist) else np.nan,
            ]

            positions = pos_hist[step]
            forces = force_hist[step]
            directions = p_hist[step]

            for particle in range(n_particles):
                row.extend([
                    positions[particle, 0],
                    positions[particle, 1],
                    positions[particle, 2],
                    forces[particle, 0],
                    forces[particle, 1],
                    forces[particle, 2],
                    directions[particle, 0],
                    directions[particle, 1],
                    directions[particle, 2],
                ])

            writer.writerow(row)

    print(f"CSV-Datei erfolgreich erstellt und unter {filename} gespeichert.")

        # Zusätzlich eine Kopie in minimization_output speichern
    comparison_dir = Path("minimization_output")
    comparison_dir.mkdir(parents=True, exist_ok=True)

    comparison_file = comparison_dir / filename.name
    shutil.copy2(filename, comparison_file)

    print(f"Kopie gespeichert unter {comparison_file}")

    return filename
    
    
    
