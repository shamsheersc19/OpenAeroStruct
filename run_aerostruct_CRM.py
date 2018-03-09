""" Example runscript to perform aerostructural optimization using CRM geometry.

Call as `python run_aerostruct_2.py 0` to run a single analysis, or
call as `python run_aerostruct_2.py 1` to perform optimization.

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

    input_arg = sys.argv[1]  
    
    # these are different MDA solver options, see run_classes.py for more.
    solver_options = ['gs_wo_aitken', 'gs_w_aitken', 'newton_gmres', 'newton_direct']
    solver_atol = 1e-6

    # Set problem type
    prob_dict = {'type' : 'aerostruct',
                 'with_viscous' : True,
                #  'force_fd' : True,
                 'optimizer': 'SNOPT',
                 'alpha' : 2.,            # [degrees] angle of attack
                 'cg' : np.array([30., 0., 5.]),
                 'solver_combo' : solver_options[0],
                 'solver_atol' : solver_atol,
                 'print_level' : 1
                 }
                 
    if input_arg.startswith('0'):  # run analysis once
        prob_dict.update({'optimize' : False})
    else:  # perform optimization
        prob_dict.update({'optimize' : True})


    # Instantiate problem and add default surface
    OAS_prob = OASProblem(prob_dict)

                 
    surf_dict = {'num_y' : 51,
                 'num_x' : 3,
                 'exact_failure_constraint' : False,
                 'wing_type' : 'CRM',
                 'span_cos_spacing' : 0,
                 'CD0' : 0.015,
                 'symmetry' : True,
                 'num_twist_cp' : 5,
                 'num_thickness_cp' : 5,
                 'twist_cp' : np.array([0., 0., 0., 0., 0.]),
                 'thickness_cp' : np.array([0.01, 0.02, 0.03, 0.03, 0.03]), # this thickness variable does not do anything, but keep it for now because run_classes expects it. This will be fixed later.
                 # The following two are thickness variables that differ from the thickness variable in the standard OAS.
                 'skinthickness_cp' : np.array([0.01, 0.02, 0.03, 0.03, 0.03]),
                 'sparthickness_cp' : np.array([0.01, 0.02, 0.03, 0.03, 0.03]),
                # Material properties taken from http://www.performance-composites.com/carbonfibre/mechanicalproperties_2.asp
                # 'E' : 45.e9,
                # 'G' : 15.e9,
                # 'yield' : 350.e6 / 2.0,
                # 'mrho' : 1.6e3,
                'E' : 70.e9,            # [Pa] Young's modulus of the spar
                'G' : 30.e9,            # [Pa] shear modulus of the spar
                'yield' : 324.e6/ 2.5 / 1.5, # [Pa] yield stress divided by 2.5 for limiting case
                'mrho' : 2.8e3,          # [kg/m^3] material density
                'strength_factor_for_upper_skin' : 1.4, # for the upper skin, the yield stress is multiplied by this factor
                'sweep' : -20.,
                 # The following are the airfoil coordinates for the wingbox section (e.g., from 15% to 65%)
                 # The chord for the corresponding airfoil should be 1
                 # The first and last x-coordinates of the upper and lower skins must be the same
                 # The datatype should be complex to work with the complex-step approximation for derivatives
                 'data_x_upper' : np.array([0.15,   0.17,   0.2,    0.22,   0.25,   0.27,   0.3,    0.33,   0.35,   0.38,   0.4,
                                             0.43,   0.45,   0.48,   0.5,    0.53,   0.55,   0.57,   0.6,    0.62,   0.65,], dtype = 'complex128'),
                 'data_x_lower' : np.array([ 0.15,   0.17,   0.2,    0.22,   0.25,   0.28,   0.3,    0.32,   0.35,   0.37,   0.4,
                                             0.42,   0.45,   0.48,   0.5,    0.53,   0.55,   0.58,   0.6,    0.63,   0.65,], dtype = 'complex128'),
                 'data_y_upper' : np.array([  0.0585,  0.0606,  0.0632,  0.0646,  0.0664,  0.0673,  0.0685,
                                             0.0692,  0.0696,  0.0698,  0.0697,  0.0695,  0.0692,  0.0684,  0.0678,  0.0666,
                                             0.0656,  0.0645,  0.0625,  0.061,   0.0585], dtype = 'complex128'),
                 'data_y_lower' : np.array([-0.0585, -0.0606, -0.0633, -0.0647, -0.0666, -0.068,  -0.0687, 
                                             -0.0692, -0.0696, -0.0696, -0.0692, -0.0688, -0.0676, -0.0657, -0.0644, -0.0614, 
                                             -0.0588, -0.0543, -0.0509, -0.0451, -0.041], dtype = 'complex128'),
                }

    # Add the specified wing surface to the problem
    OAS_prob.add_surface(surf_dict)

    # Add design variables, constraint, and objective on the problem
    # OAS_prob.add_desvar('alpha', lower=-10., upper=10.)
    OAS_prob.add_constraint('L_equals_W', equals=0.)
    OAS_prob.add_objective('fuelburn', scaler=1e-5)

    # Setup problem and add design variables, constraint, and objective
    OAS_prob.add_desvar('wing.twist_cp', lower=-10., upper=10.)
    OAS_prob.add_desvar('wing.sparthickness_cp', lower=0.002, upper=0.1, scaler=1e2)
    OAS_prob.add_desvar('wing.skinthickness_cp', lower=0.002, upper=0.1, scaler=1e2)
    OAS_prob.add_constraint('wing_perf.failure', upper=0.)
    OAS_prob.add_desvar('wing.span', lower=10., upper=100.)
    OAS_prob.add_desvar('wing.sweep', lower=-60., upper=60.)
    OAS_prob.setup()

    st = time()
    # Actually run the problem
    OAS_prob.run()

    print("\nFuelburn:", OAS_prob.prob['fuelburn'])
    print("Time elapsed: {} secs".format(time() - st))
    print(OAS_prob.prob['wing.thickness_cp'])
    print(OAS_prob.prob['wing.skinthickness_cp'])
    print(OAS_prob.prob['wing.sparthickness_cp'])
    # print(OAS_prob.prob['wing_perf.disp'])
    print(OAS_prob.prob['wing_perf.structural_weight'])
    print("Span", OAS_prob.prob['wing.span'],"m")
    print("Sweep", OAS_prob.prob['wing.sweep'])
            