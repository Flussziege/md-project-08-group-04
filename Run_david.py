#----------------------------------------------------------------
#   P A R A M E T E R S
#----------------------------------------------------------------
# system
n_particles = 200
mass_argon =  39.95             # mass in u = 1e-3 kg/mol
sigma_argon = 0.34              # sigma in nm     Argon: 0.34
epsilon_argon = 120*R*1e-3      # epsilon in kJ/mol Argon: 120

# simulation
dt = 0.001             # ps
n_steps = 1000 
temperature = 50     # K
box_length = 5     # nm
tau_thermostat = 1  # thermostat coupling constant in 1/ps
rij_min = 1e-2      # nm
NVT = True          # switch to decide between NVT and NVE
seed = 67
SD = False
