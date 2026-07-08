import numpy as np
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
        tolerance =1e-5,
        alpha_SD = 1e-5,
        max_steps = 1000
        ):

    #Listen um Werte zu speichern
    E_hist = []
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

    #just to be safe
    step = 0


    while max_force_norm(ps.force) >= tolerance and step < max_steps: 
        # alpha bestimmen
        # position ändern
        # neue Kraft berechnen
        # neue Energie berechnen
        # neue Suchrichtung berechnen

        if SD:
            alpha = alpha_SD
        else: 
           alpha = backtracking_line_search(
            x=ps.position,
            p=p,
            ps=ps,
            sim=sim,
            E_old=E,
            energy_func=energy_func,
            alpha0=1e-4
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
        Fmax_hist.append(max_force_norm(ps.force))
        alpha_hist.append(alpha)
        beta_hist.append(beta)

        step += 1


    print(f"n_steps: {step}")
    print(f"converged: {max_force_norm(ps.force) < tolerance}")
    print(f"final_energy: {E}")
    print(f"final_Fmax: {max_force_norm(ps.force)}")

    return {
        "positions": ps.position.copy(),
        "E_hist": np.array(E_hist),
        "Fmax_hist": np.array(Fmax_hist),
        "pos_hist": np.array(pos_hist),
        "alpha_hist": np.array(alpha_hist),
        "p_hist": np.array(p_hist),
        "beta_hist": np.array(beta_hist),
    }


        

    
    
