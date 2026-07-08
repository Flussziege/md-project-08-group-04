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
        x_new = x + alpha * p
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
        alpha_SD = 0.5
        ):

    #Listen um Werte zu speichern
    E_hist = []
    Fmax_hist = []
    pos_hist = []
    alpha_hist = []
    p_hist = []

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
    Fmax_hist(max_force_norm(ps.force))


    while max_force_norm(ps.force) > tolerance: 
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
            E_old=E,
            energy_func=energy_func,
            alpha0=1e-4
        )
        
        alpha_hist.append(alpha)

        

    
    
