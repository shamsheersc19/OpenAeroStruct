"""
Test with aerostructural optimization
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

import unittest


class Test(unittest.TestCase):
    
    def test_analysis(self):

        # Provide coordinates for a portion of an airfoil for the wingbox cross-section as an nparray with dtype=complex (to work with the complex-step approximation for derivatives).
        # These should be for an airfoil with the chord scaled to 1.
        # We use the 10% to 60% portion of the NASA SC2-0612 airfoil for this case
        # We use the coordinates available from airfoiltools.com. Using such a large number of coordinates is not necessary.
        # The first and last x-coordinates of the upper and lower skins must be the same

        upper_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype = 'complex128')
        lower_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype = 'complex128')
        upper_y = np.array([ 0.0447,  0.046,  0.0472,  0.0484,  0.0495,  0.0505,  0.0514,  0.0523,  0.0531,  0.0538, 0.0545,  0.0551,  0.0557, 0.0563,  0.0568, 0.0573,  0.0577,  0.0581,  0.0585,  0.0588,  0.0591,  0.0593,  0.0595,  0.0597,  0.0599,  0.06,    0.0601,  0.0602,  0.0602,  0.0602,  0.0602,  0.0602,  0.0601,  0.06,    0.0599,  0.0598,  0.0596,  0.0594,  0.0592,  0.0589,  0.0586,  0.0583,  0.058,   0.0576,  0.0572,  0.0568,  0.0563,  0.0558,  0.0553,  0.0547,  0.0541], dtype = 'complex128')
        lower_y = np.array([-0.0447, -0.046, -0.0473, -0.0485, -0.0496, -0.0506, -0.0515, -0.0524, -0.0532, -0.054, -0.0547, -0.0554, -0.056, -0.0565, -0.057, -0.0575, -0.0579, -0.0583, -0.0586, -0.0589, -0.0592, -0.0594, -0.0595, -0.0596, -0.0597, -0.0598, -0.0598, -0.0598, -0.0598, -0.0597, -0.0596, -0.0594, -0.0592, -0.0589, -0.0586, -0.0582, -0.0578, -0.0573, -0.0567, -0.0561, -0.0554, -0.0546, -0.0538, -0.0529, -0.0519, -0.0509, -0.0497, -0.0485, -0.0472, -0.0458, -0.0444], dtype = 'complex128')

        # These are the input arguments specified by the user when the script is called
        input_arg_1 = '0'
        input_arg_nx = '3'
        input_arg_ny = '21'
        input_arg_wave = '1'
        input_arg_wing_type = 'CRM'

        # These are a list of different MDA solver options, see run_classes.py for more. Compatible with OpenMDAO version 1.7.4.
        # If using OpenMDAO 1.7.3, the `gs_w_aitken` option will not work. Version 1.7.4 is recommended.
        solver_options = ['gs_wo_aitken', 'gs_w_aitken', 'newton_gmres', 'newton_direct']

        # Specify a convergence tolerance for the coupled aerostructural solver
        solver_atol = 5e-6

        # Specify the load factor for the maneuver case used for sizing
        g_factor = 2.5

        # Set problem type
        # for more defaults and options see `def get_default_prob_dict(self):` in run_classes.py
        prob_dict = {'type' : 'aerostruct',
                     'with_viscous' : True,    # use estimates for viscous drag
                     'optimizer': 'SNOPT',     # see `def setup_prob(self)` in run_classes.py for other options and tolerance settings
                     'solver_combo' : solver_options[1], # see solver options above
                     'solver_atol' : solver_atol,
                     'print_level' : 0,
                     'compute_static_margin' : False,    # static margin computation not verified, may not work right now
                     # Flow/environment properties
                     'alpha' : 0.,             # [degrees] angle of attack for cruise case
                     'reynolds_length' : 1.0,  # characteristic Reynolds length
                     'M' : 0.85,               # Mach number at cruise
                     'rho' : 0.348,            # [kg/m^3] air density at cruise (37,000 ft)
                     'a' : 295.07,             # [m/s] speed of sound at cruise (37,000 ft)
                     'Re' : 0.348*295.07*.85*1./(1.43*1e-5), # cruise Reynolds number for characteristic Re length
                     'g' : 9.80665,            # [m/s^2] acceleration due to gravity

                     # Aircraft properties
                     'CT' : 0.53/3600,         # [1/s] specific fuel consumption
                     'R' : 14.307e6,           # [m] range
                     'cg' : np.zeros((3)),     # center of gravity for the entire aircraft. Used in trim and stability calculations (not used in this example case).
                     'W0' : (143000 - 2.5*11600 + 34000), # [kg] weight without fuel and wing structure. 143000 is ~ empty W of 777-200, 11600 is 1/2 wing structural weight guess, 34000 for payload
                     'Wf_reserve' : 15000.,    # [kg] reserve fuel
                     'W_wing_factor' : 1.25,   # factor to account for extra wing structural weight
                     'beta' : 1.,              # weighting factor for mixed objective (see functionals.py, the mixed objective is not used for this case)
                     'g_factor' : g_factor,    # load factor for maneuver cases
                     'rho_maneuver' : 0.9237,  # density that gives the required dynamic pressure for the maneuver case using the cruise speed
                     }

        if input_arg_1.startswith('0'):  # run analysis once
            prob_dict.update({'optimize' : False})
        else:  # perform optimization
            prob_dict.update({'optimize' : True})


        # Instantiate problem and add default surface
        OAS_prob = OASProblem(prob_dict)

        # Provide information for the lifting surface
        # for more defaults and options see `def get_default_surf_dict(self):` in run_classes.py
        surf_dict = {'num_y' : int(input_arg_ny),        # number of spanwise nodes for complete wing (specify odd numbers)
                     'num_x' : int(input_arg_nx),        # number of chordwise nodes (specify odd numbers)
                     'exact_failure_constraint' : False, # exact or KS failure constraint
                     'wing_type' : input_arg_wing_type,  # using OAS's wings defined in geometry.py
                     'span_cos_spacing' : 0,             # uniform spacing if 0
                     'chord_cos_spacing' : 0,            # uniform spacing if 0
                     'CL0' : 0.0,                        # additional CL
                     'CD0' : 0.0078,                     # additional CD (for fuse, tails, nacelles, etc)
                     'symmetry' : True,                  # True for symmetry
                     'S_ref_type' : 'wetted',            # reference area type ('wetted' or 'projected')
                     'twist_cp' : np.array([4., 5., 8., 8., 8., 9.]), # [deg] inital values for twist distribution
                     # The following are two thickness variables that differ from the single thickness variable in the original OAS.
                     'skinthickness_cp' : np.array([0.005, 0.01, 0.015, 0.020, 0.025, 0.026]), # [m]
                     'sparthickness_cp' : np.array([0.004, 0.005, 0.005, 0.008, 0.008, 0.01]), # [m]
                     # Airfoil properties for viscous drag calculation
                     'k_lam' : 0.05,            # percentage of chord with laminar flow, used for viscous drag
                     't_over_c' : 0.12,         # original thickness to chord ratio for the airfoil specified
                     'toverc_cp' : np.array([0.08, 0.08, 0.08, 0.10, 0.10, 0.08]), # initial guess for streamwise thickness to chord cp distribution
                     'c_max_t' : .38,           # chordwise location of maximum thickness
                     # Material properties
                     'E' : 73.1e9,              # [Pa] Young's modulus
                     'G' : (73.1e9/2/1.33),     # [Pa] shear modulus (calculated using E and the Poisson's ratio here)
                     'yield' : (420.e6 / 1.5),  # [Pa] allowable yield stress
                     'mrho' : 2.78e3,           # [kg/m^3] material density
                     'strength_factor_for_upper_skin' : 1.0, # the yield stress is multiplied by this factor for the upper skin
                    #  'sweep' : -30.,
                     'monotonic_con' : None, # add monotonic constraint to the given
                                            # distributed variable. Ex. 'chord_cp'
                     # Wingbox airfoil section coordinates
                     'data_x_upper' : upper_x,
                     'data_x_lower' : lower_x,
                     'data_y_upper' : upper_y,
                     'data_y_lower' : lower_y,
                     'wave_drag' : int(input_arg_wave)
                    }

        # Add the specified wing surface to the problem
        OAS_prob.add_surface(surf_dict)

        # Add design variables, constraint, and objective on the problem

        OAS_prob.add_objective('coupled.total_perf.fuelburn.fuelburn', scaler=1e-5)

        OAS_prob.add_desvar('alpha_maneuver', lower=-15., upper=15.)
        OAS_prob.add_desvar('wing.twist_cp', lower=-15., upper=15., scaler=0.1)
        OAS_prob.add_desvar('wing.sparthickness_cp', lower=0.003, upper=0.1, scaler=1e2)
        OAS_prob.add_desvar('wing.skinthickness_cp', lower=0.003, upper=0.1, scaler=1e2)
        OAS_prob.add_desvar('wing.toverc_cp', lower=0.07, upper=0.2, scaler=10.)

        OAS_prob.add_constraint('coupled.wing_perf.CL', equals=0.5)
        OAS_prob.add_constraint('coupled_maneuver.total_perf_maneuver.L_equals_W', equals=0.)
        OAS_prob.add_constraint('failure', upper=0.)
        OAS_prob.add_constraint('coupled.wing_loads.fuel_vol_delta', lower=0.)

        # A few other possibilites for constraints and design variables...
        # OAS_prob.add_desvar('wing.span', lower=10., upper=100., scaler=2e-2)
        # OAS_prob.add_desvar('wing.sweep', lower=-60., upper=60., scaler=1e-1)
        # OAS_prob.add_constraint('coupled.wing.S_ref', lower=414., scaler=2e-3)
        # OAS_prob.add_desvar('alpha', lower=-10., upper=10.)
        # OAS_prob.add_objective('coupled_maneuver.total_perf_maneuver.fuelburn.fuelburn', scaler=1e-5)
        # OAS_prob.add_constraint('wing.twist_cp', lower=np.array([-1e99,-1e99,-1e99,-1e99,-1e99,5]), upper=np.array([1e99,1e99,1e99,1e99,1e99,5]))
        # OAS_prob.add_desvar('wing.chord_cp', lower=0.25, upper=15.)
        # OAS_prob.add_constraint('coupled.wing_perf.failure', upper=0.)

        OAS_prob.setup()

        st = time()
        # Actually run the problem
        OAS_prob.run()

        print("Time elapsed: {} secs".format(time() - st))

        print("\nFuelburn:", '%.15f' % OAS_prob.prob['coupled.total_perf.fuelburn.fuelburn'])
        print("Structural weight:", '%.15f' % OAS_prob.prob['coupled.wing_perf.structural_weight'])

        self.assertAlmostEqual(OAS_prob.prob['coupled.total_perf.fuelburn.fuelburn'], 95193.963555)
        self.assertAlmostEqual(OAS_prob.prob['coupled.wing_perf.structural_weight'], 230853.0879710879 )


    def test_opt(self):
    
        # Provide coordinates for a portion of an airfoil for the wingbox cross-section as an nparray with dtype=complex (to work with the complex-step approximation for derivatives).
        # These should be for an airfoil with the chord scaled to 1.
        # We use the 10% to 60% portion of the NASA SC2-0612 airfoil for this case
        # We use the coordinates available from airfoiltools.com. Using such a large number of coordinates is not necessary.
        # The first and last x-coordinates of the upper and lower skins must be the same
    
        upper_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype = 'complex128')
        lower_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype = 'complex128')
        upper_y = np.array([ 0.0447,  0.046,  0.0472,  0.0484,  0.0495,  0.0505,  0.0514,  0.0523,  0.0531,  0.0538, 0.0545,  0.0551,  0.0557, 0.0563,  0.0568, 0.0573,  0.0577,  0.0581,  0.0585,  0.0588,  0.0591,  0.0593,  0.0595,  0.0597,  0.0599,  0.06,    0.0601,  0.0602,  0.0602,  0.0602,  0.0602,  0.0602,  0.0601,  0.06,    0.0599,  0.0598,  0.0596,  0.0594,  0.0592,  0.0589,  0.0586,  0.0583,  0.058,   0.0576,  0.0572,  0.0568,  0.0563,  0.0558,  0.0553,  0.0547,  0.0541], dtype = 'complex128')
        lower_y = np.array([-0.0447, -0.046, -0.0473, -0.0485, -0.0496, -0.0506, -0.0515, -0.0524, -0.0532, -0.054, -0.0547, -0.0554, -0.056, -0.0565, -0.057, -0.0575, -0.0579, -0.0583, -0.0586, -0.0589, -0.0592, -0.0594, -0.0595, -0.0596, -0.0597, -0.0598, -0.0598, -0.0598, -0.0598, -0.0597, -0.0596, -0.0594, -0.0592, -0.0589, -0.0586, -0.0582, -0.0578, -0.0573, -0.0567, -0.0561, -0.0554, -0.0546, -0.0538, -0.0529, -0.0519, -0.0509, -0.0497, -0.0485, -0.0472, -0.0458, -0.0444], dtype = 'complex128')
    
        # These are the input arguments specified by the user when the script is called
        input_arg_1 = '1'
        input_arg_nx = '3'
        input_arg_ny = '15'
        input_arg_wave = '1'
        input_arg_wing_type = 'CRM'
    
        # These are a list of different MDA solver options, see run_classes.py for more. Compatible with OpenMDAO version 1.7.4.
        # If using OpenMDAO 1.7.3, the `gs_w_aitken` option will not work. Version 1.7.4 is recommended.
        solver_options = ['gs_wo_aitken', 'gs_w_aitken', 'newton_gmres', 'newton_direct']
    
        # Specify a convergence tolerance for the coupled aerostructural solver
        solver_atol = 5e-6
    
        # Specify the load factor for the maneuver case used for sizing
        g_factor = 2.5
    
        # Set problem type
        # for more defaults and options see `def get_default_prob_dict(self):` in run_classes.py
        prob_dict = {'type' : 'aerostruct',
                     'with_viscous' : True,    # use estimates for viscous drag
                     'optimizer': 'SNOPT',     # see `def setup_prob(self)` in run_classes.py for other options and tolerance settings
                     'solver_combo' : solver_options[1], # see solver options above
                     'solver_atol' : solver_atol,
                     'print_level' : 0,
                     'compute_static_margin' : False,    # static margin computation not verified, may not work right now
                     # Flow/environment properties
                     'alpha' : 0.,             # [degrees] angle of attack for cruise case
                     'reynolds_length' : 1.0,  # characteristic Reynolds length
                     'M' : 0.85,               # Mach number at cruise
                     'rho' : 0.348,            # [kg/m^3] air density at cruise (37,000 ft)
                     'a' : 295.07,             # [m/s] speed of sound at cruise (37,000 ft)
                     'Re' : 0.348*295.07*.85*1./(1.43*1e-5), # cruise Reynolds number for characteristic Re length
                     'g' : 9.80665,            # [m/s^2] acceleration due to gravity
    
                     # Aircraft properties
                     'CT' : 0.53/3600,         # [1/s] specific fuel consumption
                     'R' : 14.307e6,           # [m] range
                     'cg' : np.zeros((3)),     # center of gravity for the entire aircraft. Used in trim and stability calculations (not used in this example case).
                     'W0' : (143000 - 2.5*11600 + 34000), # [kg] weight without fuel and wing structure. 143000 is ~ empty W of 777-200, 11600 is 1/2 wing structural weight guess, 34000 for payload
                     'Wf_reserve' : 15000.,    # [kg] reserve fuel
                     'W_wing_factor' : 1.25,   # factor to account for extra wing structural weight
                     'beta' : 1.,              # weighting factor for mixed objective (see functionals.py, the mixed objective is not used for this case)
                     'g_factor' : g_factor,    # load factor for maneuver cases
                     'rho_maneuver' : 0.9237,  # density that gives the required dynamic pressure for the maneuver case using the cruise speed
                     }
    
        if input_arg_1.startswith('0'):  # run analysis once
            prob_dict.update({'optimize' : False})
        else:  # perform optimization
            prob_dict.update({'optimize' : True})
    
    
        # Instantiate problem and add default surface
        OAS_prob = OASProblem(prob_dict)
    
        # Provide information for the lifting surface
        # for more defaults and options see `def get_default_surf_dict(self):` in run_classes.py
        surf_dict = {'num_y' : int(input_arg_ny),        # number of spanwise nodes for complete wing (specify odd numbers)
                     'num_x' : int(input_arg_nx),        # number of chordwise nodes (specify odd numbers)
                     'exact_failure_constraint' : False, # exact or KS failure constraint
                     'wing_type' : input_arg_wing_type,  # using OAS's wings defined in geometry.py
                     'span_cos_spacing' : 0,             # uniform spacing if 0
                     'chord_cos_spacing' : 0,            # uniform spacing if 0
                     'CL0' : 0.0,                        # additional CL
                     'CD0' : 0.0078,                     # additional CD (for fuse, tails, nacelles, etc)
                     'symmetry' : True,                  # True for symmetry
                     'S_ref_type' : 'wetted',            # reference area type ('wetted' or 'projected')
                     'twist_cp' : np.array([4., 5., 8., 8., 8., 9.]), # [deg] inital values for twist distribution
                     # The following are two thickness variables that differ from the single thickness variable in the original OAS.
                     'skinthickness_cp' : np.array([0.005, 0.01, 0.015, 0.020, 0.025, 0.026]), # [m]
                     'sparthickness_cp' : np.array([0.004, 0.005, 0.005, 0.008, 0.008, 0.01]), # [m]
                     # Airfoil properties for viscous drag calculation
                     'k_lam' : 0.05,            # percentage of chord with laminar flow, used for viscous drag
                     't_over_c' : 0.12,         # original thickness to chord ratio for the airfoil specified
                     'toverc_cp' : np.array([0.08, 0.08, 0.08, 0.10, 0.10, 0.08]), # initial guess for streamwise thickness to chord cp distribution
                     'c_max_t' : .38,           # chordwise location of maximum thickness
                     # Material properties
                     'E' : 73.1e9,              # [Pa] Young's modulus
                     'G' : (73.1e9/2/1.33),     # [Pa] shear modulus (calculated using E and the Poisson's ratio here)
                     'yield' : (420.e6 / 1.5),  # [Pa] allowable yield stress
                     'mrho' : 2.78e3,           # [kg/m^3] material density
                     'strength_factor_for_upper_skin' : 1.0, # the yield stress is multiplied by this factor for the upper skin
                    #  'sweep' : -30.,
                     'monotonic_con' : None, # add monotonic constraint to the given
                                            # distributed variable. Ex. 'chord_cp'
                     # Wingbox airfoil section coordinates
                     'data_x_upper' : upper_x,
                     'data_x_lower' : lower_x,
                     'data_y_upper' : upper_y,
                     'data_y_lower' : lower_y,
                     'wave_drag' : int(input_arg_wave)
                    }
    
        # Add the specified wing surface to the problem
        OAS_prob.add_surface(surf_dict)
    
        # Add design variables, constraint, and objective on the problem
    
        OAS_prob.add_objective('coupled.total_perf.fuelburn.fuelburn', scaler=1e-5)
    
        OAS_prob.add_desvar('alpha_maneuver', lower=-15., upper=15.)
        OAS_prob.add_desvar('wing.twist_cp', lower=-15., upper=15., scaler=0.1)
        OAS_prob.add_desvar('wing.sparthickness_cp', lower=0.003, upper=0.1, scaler=1e2)
        OAS_prob.add_desvar('wing.skinthickness_cp', lower=0.003, upper=0.1, scaler=1e2)
        OAS_prob.add_desvar('wing.toverc_cp', lower=0.07, upper=0.2, scaler=10.)
    
        OAS_prob.add_constraint('coupled.wing_perf.CL', equals=0.5)
        OAS_prob.add_constraint('coupled_maneuver.total_perf_maneuver.L_equals_W', equals=0.)
        OAS_prob.add_constraint('failure', upper=0.)
        OAS_prob.add_constraint('coupled.wing_loads.fuel_vol_delta', lower=0.)
    
        # A few other possibilites for constraints and design variables...
        # OAS_prob.add_desvar('wing.span', lower=10., upper=100., scaler=2e-2)
        # OAS_prob.add_desvar('wing.sweep', lower=-60., upper=60., scaler=1e-1)
        # OAS_prob.add_constraint('coupled.wing.S_ref', lower=414., scaler=2e-3)
        # OAS_prob.add_desvar('alpha', lower=-10., upper=10.)
        # OAS_prob.add_objective('coupled_maneuver.total_perf_maneuver.fuelburn.fuelburn', scaler=1e-5)
        # OAS_prob.add_constraint('wing.twist_cp', lower=np.array([-1e99,-1e99,-1e99,-1e99,-1e99,5]), upper=np.array([1e99,1e99,1e99,1e99,1e99,5]))
        # OAS_prob.add_desvar('wing.chord_cp', lower=0.25, upper=15.)
        # OAS_prob.add_constraint('coupled.wing_perf.failure', upper=0.)
    
        OAS_prob.setup()
    
        st = time()
        # Actually run the problem
        OAS_prob.run()
    
        print("Time elapsed: {} secs".format(time() - st))
    
        print("\nFuelburn:", '%.15f' % OAS_prob.prob['coupled.total_perf.fuelburn.fuelburn'])
        print("Structural weight:", '%.15f' % OAS_prob.prob['coupled.wing_perf.structural_weight'])
    
        self.assertAlmostEqual(OAS_prob.prob['coupled.total_perf.fuelburn.fuelburn'], 95444.530133685242618)
        self.assertAlmostEqual(OAS_prob.prob['coupled.wing_perf.structural_weight'], 243314.226631071476731 )

if __name__ == '__main__':
    unittest.main()
