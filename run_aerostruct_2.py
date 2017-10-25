""" Example runscript to perform aerostructural optimization.

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

    twist_index = int(sys.argv[1]) # Use input for twist starting point    

    for num_y in [51]:
        for index1 in [2]:
    
            solver_options = ['gs_wo_aitken', 'gs_w_aitken', 'newton_gmres', 'newton_direct']
            solver_combo = solver_options[index1] 
            # solver_combo = solver_options[2] 
            solver_atol = 1e-6

            # Set problem type
            prob_dict = {'type' : 'aerostruct',
                         'optimize' : True,
                         'with_viscous' : True,
                        #  'force_fd' : True,
                         'optimizer': 'SNOPT',
                         'alpha' : 2.,            # [degrees] angle of attack
                         'cg' : np.array([30., 0., 5.]),
                         'solver_combo' : solver_combo,
                         'solver_atol' : solver_atol,
                         'print_level' : 1
                         }


            # Instantiate problem and add default surface
            OAS_prob = OASProblem(prob_dict)

            twistz = [np.array([0., 0., 0., 0., 0.]), np.array([1., 2., 1., 7., -1.]), np.array([-2., 0., 2., 4., 6.]), np.array([6., 4., 2., 0., -2.])]
                         
            surf_dict = {'num_y' : num_y,
                         'num_x' : 3,
                         'exact_failure_constraint' : False,
                         'wing_type' : 'CRM',
                         'span_cos_spacing' : 0,
                         'CD0' : 0.015,
                         'symmetry' : True,
                         'num_twist_cp' : 5,
                         'num_thickness_cp' : 5,
                         'twist_cp' : twistz[twist_index],
                         'thickness_cp' : np.array([0.01, 0.02, 0.03, 0.03, 0.03]),
                         'skinthickness_cp' : np.array([0.01, 0.02, 0.03, 0.03, 0.03]),
                         'sparthickness_cp' : np.array([0.01, 0.02, 0.03, 0.03, 0.03]),
                        # Material properties taken from http://www.performance-composites.com/carbonfibre/mechanicalproperties_2.asp
                        # 'E' : 45.e9,
                        # 'G' : 15.e9,
                        # 'yield' : 350.e6 / 2.0,
                        # 'mrho' : 1.6e3,
                        'E' : 70.e9,            # [Pa] Young's modulus of the spar
                        'G' : 30.e9,            # [Pa] shear modulus of the spar
                        # 'yield' : 324.e6/ 2.5 / 1.5, # [Pa] yield stress divided by 2.5 for limiting case
                        'yield' : 450.e6/ 2.5 / 1.5, # [Pa] yield stress divided by 2.5 for limiting case
                        'mrho' : 2.8e3,          # [kg/m^3] material density
                        'fem_origin' : 0.4,
                        'strength_factor_for_upper_skin' : 1.0
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
            