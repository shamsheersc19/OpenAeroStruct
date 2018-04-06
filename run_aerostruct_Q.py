""" This is the script that was used for the automated selection paper.

Call as e.g. `python run_aerostruct_Q.py 1 30 100 75` to run an optimization
The first argument is the MDA approach index (see solver_options).
The second argument is the sweep angle in deg.
The third argument is the percentage factor to multiply the initial thickness os 2cm by.
The fourth argument if the percentage factor to miltiply the material E and G by.

"""

from __future__ import division, print_function
import sys
from time import time
import numpy as np

# Append the parent directory to the system path so we can call those Python
# files. If you have OpenAeroStruct in your PYTHONPATH, this is not necessary.
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from OpenAeroStruct import OASProblem

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":    

    # Some aircraft specs

    span = 28.42                    # wing span in m
    root_chord = 3.34               # root chord in m
    cruise_CL = 0.607               # Cruise CL estimate
    t_ov_c = 0.14                   # average t/c estimate
    MAC = 2.48                      # mean aero chord
    W_wo_fuel_wbox = 25444.         # Aicraft weight without fuel and wing struct
    SFC = 0.43/3600                 # specific fuel consumption est
    rest_CD = 0.0142                # CD of rest of aircraft estimates


    # Mesh options

    nx = 3  # number of chordwise nodal points
    ny = 15  # number of spanwise nodal points for half wing


    # User inputs

    solver_option_index = int(sys.argv[1]) # MDA solver option index. See solver_options 
    sweep_deg = int(sys.argv[2]) # Sweep angle in degrees
    thk_factor = (int(sys.argv[3])/100.) # Factor to adjust initial thickness
    stiffness_factor = (int(sys.argv[4])/100.) # Factor to adjust material stiffness
    
    # MDA approach options
    
    solver_options = ['gs_wo_aitken', 'gs_w_aitken', 'AS', 'newton_gmres', 'newton_direct', 'GS_then_Newton']
    solver_combo = solver_options[solver_option_index] 
    solver_atol = 1e-6


    # Set problem type
    prob_dict = {'type' : 'aerostruct',
                 'optimize' : 'True',
                 'with_viscous' : True,
                #  'force_fd' : True,
                 'optimizer': 'SNOPT',
                 'cg' : np.array([30., 0., 5.]),
                 'solver_combo' : solver_combo,
                 'solver_atol' : solver_atol,
                 'print_level' : 2,
                 'Re' : 12e6,             # Reynolds number
                 'reynolds_length' : MAC, # characteristic Reynolds length
                 'alpha' : 2.,            # [degrees] angle of attack
                 'M' : 0.5,               # Mach number at cruise
                 'rho' : 0.57,            # [kg/m^3] air density at 24,000 ft
                 'a' : 311.,              # [m/s] speed of sound at 24,000 ft
                 'g' : 9.80665,           # [m/s^2] acceleration due to gravity
                                          # also change the 'CT' value below
                                          # accordingly if you alter this value
                 'CT' : SFC,              # [1/s] specific fuel consumption
                 'R' : 2e6,               # [m] maximum range
                 'W0' : W_wo_fuel_wbox, # [kg] MTOW
                 }


    # Instantiate problem and add default surface
    OAS_prob = OASProblem(prob_dict)

    # Initialize the 3-D mesh object. Chordwise, spanwise, then the 3D coordinates.
    mesh = np.zeros((nx, ny, 3))

    # Start away from the symm plane and approach the plane as the array indices increase.
    mesh[:, :, 1] = np.linspace(-span/2, 0, ny)
    mesh[0, :, 0] = 0.34 * root_chord * np.linspace(1.0, 0., ny)
    mesh[nx-1, :, 0] = root_chord * (np.linspace(0.4, 1.0, ny)
                                  + 0.34 * np.linspace(1.0, 0., ny))
    for i in range(1, nx-1):
        mesh[i, :, 0] = ( mesh[nx-1, :, 0] - mesh[0, :, 0] ) / (nx-1) * i + mesh[0, :, 0]

    surf_dict = {'num_y' : ny,
                 'num_x' : 3,
                 'name' : 'wing',
                 'exact_failure_constraint' : False,
                 'mesh' : mesh,

                 # Airfoil properties for viscous drag calculation
                 'k_lam' : 0.05,         # percentage of chord with laminar
                                         # flow, used for viscous drag
                 't_over_c' : t_ov_c,    # thickness over chord ratio
                 'c_max_t' : .303,       # chordwise location of maximum (NACA0012)
                                         # thickness
                 'CL0' : 0.,             # rest of aircraft CL
                 'CD0' : rest_CD,        # rest of aircraft CD

                 'symmetry' : True,
                 'num_twist_cp' : 5,
                 'num_thickness_cp' : 5,
                 'twist_cp' : np.array([0., 0., 0., 0., 0.]),
                 'thickness_cp' : thk_factor * 2. * np.array([0.01, 0.01, 0.01, 0.01, 0.01]),
                # Material properties taken from http://www.performance-composites.com/carbonfibre/mechanicalproperties_2.asp
                # 'E' : 45.e9,
                # 'G' : 15.e9,
                # 'yield' : 350.e6 / 2.0,
                # 'mrho' : 1.6e3,
                'E' : 70.e9 * stiffness_factor, # [Pa] Young's modulus of the spar
                'G' : 30.e9 * stiffness_factor, # [Pa] shear modulus of the spar
                'yield' : 500.e6/ 2.5 / 1.5, # [Pa] yield stress divided by 2.5 for limiting case
                'mrho' : 2.8e3,          # [kg/m^3] material density
                'fem_origin' : 0.4,
                'exact_failure_constraint' : False, # if false, use KS function
                'monotonic_con' : None,
                'sweep' : 1. * sweep_deg
                }

    # Add the specified wing surface to the problem
    OAS_prob.add_surface(surf_dict)

    # Add design variables, constraint, and objective on the problem
    # OAS_prob.add_desvar('alpha', lower=-10., upper=10.)
    OAS_prob.add_constraint('L_equals_W', equals=0.)
    OAS_prob.add_objective('fuelburn', scaler=1e-5)


    # Setup problem and add design variables, constraint, and objective
    OAS_prob.add_desvar('wing.twist_cp', lower=-10., upper=10.)
    OAS_prob.add_desvar('wing.thickness_cp', lower=0.002, upper=0.2, scaler=1e2)
    OAS_prob.add_constraint('wing_perf.failure', upper=0.)
    OAS_prob.add_constraint('wing_perf.thickness_intersects', upper=0.)
    OAS_prob.setup()

    st = time()
    # Actually run the problem
    OAS_prob.run()

    print("\nFuelburn:", OAS_prob.prob['fuelburn'])
    print("Time elapsed: {} secs".format(time() - st))
    print(OAS_prob.prob['wing.thickness_cp'])
    # print(OAS_prob.prob['wing_perf.disp'])
    print(OAS_prob.prob['wing_perf.structural_weight'])
    