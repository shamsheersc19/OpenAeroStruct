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
    
    # NASA SC2-0612 airfoil
    # upper_x = np.array([0.1	,0.11	,0.12	,0.13	,0.14	,0.15	,0.16	,0.17	,0.18	,0.19	,0.2	,0.21	,0.22	,0.23	,0.24	,0.25	,0.26	,0.27	,0.28	,0.29	,0.3	,0.31	,0.32	,0.33	,0.34	,0.35	,0.36	,0.37	,0.38	,0.39	,0.4	,0.41	,0.42	,0.43	,0.44	,0.45	,0.46	,0.47	,0.48	,0.49	,0.5	,0.51	,0.52	,0.53	,0.54	,0.55	,0.56	,0.57	,0.58	,0.59	,0.6	,.61	,.62	,.63	,.64	,.65	,.66	,.67	,.68	,.69	,.7	,0.71	,0.72	,0.73	,0.74	,0.75	,], dtype = 'complex128')
    # lower_x = np.array([ 0.1	,0.11	,0.12	,0.13	,0.14	,0.15	,0.16	,0.17	,0.18	,0.19	,0.2	,0.21	,0.22	,0.23	,0.24	,0.25	,0.26	,0.27	,0.28	,0.29	,0.3	,0.31	,0.32	,0.33	,0.34	,0.35	,0.36	,0.37	,0.38	,0.39	,0.4	,0.41	,0.42	,0.43	,0.44	,0.45	,0.46	,0.47	,0.48	,0.49	,0.5	,0.51	,0.52	,0.53	,0.54	,0.55	,0.56	,0.57	,0.58	,0.59	,0.6	,0.61	,0.62	,0.63	,0.64	,0.65	,0.66	,0.67	,0.68	,0.69	,0.7	,0.71	,0.72	,0.73	,0.74	,0.75	,], dtype = 'complex128')
    # upper_y = np.array([  0.0447	,0.046	,0.0472	,0.0484	,0.0495	,0.0505	,0.0514	,0.0523	,0.0531	,0.0538	,0.0545	,0.0551	,0.0557	,0.0563	,0.0568	,0.0573	,0.0577	,0.0581	,0.0585	,0.0588	,0.0591	,0.0593	,0.0595	,0.0597	,0.0599	,0.06	,0.0601	,0.0602	,0.0602	,0.0602	,0.0602	,0.0602	,0.0601	,0.06	,0.0599	,0.0598	,0.0596	,0.0594	,0.0592	,0.0589	,0.0586	,0.0583	,0.058	,0.0576	,0.0572	,0.0568	,0.0563	,0.0558	,0.0553	,0.0547	,0.0541	,0.0534	,0.0527	,0.052	,0.0512	,0.0504	,0.0495	,0.0486	,0.0476	,0.0466	,0.0456	,0.0445	,0.0434	,0.0422	,0.041	,0.0397	,], dtype = 'complex128')
    # lower_y = np.array([-0.0447	,-0.046	,-0.0473	,-0.0485	,-0.0496	,-0.0506	,-0.0515	,-0.0524	,-0.0532	,-0.054	,-0.0547	,-0.0554	,-0.056	,-0.0565	,-0.057	,-0.0575	,-0.0579	,-0.0583	,-0.0586	,-0.0589	,-0.0592	,-0.0594	,-0.0595	,-0.0596	,-0.0597	,-0.0598	,-0.0598	,-0.0598	,-0.0598	,-0.0597	,-0.0596	,-0.0594	,-0.0592	,-0.0589	,-0.0586	,-0.0582	,-0.0578	,-0.0573	,-0.0567	,-0.0561	,-0.0554	,-0.0546	,-0.0538	,-0.0529	,-0.0519	,-0.0509	,-0.0497	,-0.0485	,-0.0472	,-0.0458	,-0.0444	,-0.0429	,-0.0414	,-0.0398	,-0.0382	,-0.0365	,-0.0348	,-0.033	,-0.0312	,-0.0294	,-0.0276	,-0.0258	,-0.024	,-0.0222	,-0.0204	,-0.0186	,], dtype = 'complex128')

    # NASA SC2-0612 airfoil
    upper_x = np.array([0.1	,0.11	,0.12	,0.13	,0.14	,0.15	,0.16	,0.17	,0.18	,0.19	,0.2	,0.21	,0.22	,0.23	,0.24	,0.25	,0.26	,0.27	,0.28	,0.29	,0.3	,0.31	,0.32	,0.33	,0.34	,0.35	,0.36	,0.37	,0.38	,0.39	,0.4	,0.41	,0.42	,0.43	,0.44	,0.45	,0.46	,0.47	,0.48	,0.49	,0.5	,0.51	,0.52	,0.53	,0.54	,0.55	,0.56	,0.57	,0.58	,0.59	,0.6], dtype = 'complex128')
    lower_x = np.array([ 0.1	,0.11	,0.12	,0.13	,0.14	,0.15	,0.16	,0.17	,0.18	,0.19	,0.2	,0.21	,0.22	,0.23	,0.24	,0.25	,0.26	,0.27	,0.28	,0.29	,0.3	,0.31	,0.32	,0.33	,0.34	,0.35	,0.36	,0.37	,0.38	,0.39	,0.4	,0.41	,0.42	,0.43	,0.44	,0.45	,0.46	,0.47	,0.48	,0.49	,0.5	,0.51	,0.52	,0.53	,0.54	,0.55	,0.56	,0.57	,0.58	,0.59	,0.6], dtype = 'complex128')
    upper_y = np.array([  0.0447	,0.046	,0.0472	,0.0484	,0.0495	,0.0505	,0.0514	,0.0523	,0.0531	,0.0538	,0.0545	,0.0551	,0.0557	,0.0563	,0.0568	,0.0573	,0.0577	,0.0581	,0.0585	,0.0588	,0.0591	,0.0593	,0.0595	,0.0597	,0.0599	,0.06	,0.0601	,0.0602	,0.0602	,0.0602	,0.0602	,0.0602	,0.0601	,0.06	,0.0599	,0.0598	,0.0596	,0.0594	,0.0592	,0.0589	,0.0586	,0.0583	,0.058	,0.0576	,0.0572	,0.0568	,0.0563	,0.0558	,0.0553	,0.0547	,0.0541	], dtype = 'complex128')
    lower_y = np.array([-0.0447	,-0.046	,-0.0473	,-0.0485	,-0.0496	,-0.0506	,-0.0515	,-0.0524	,-0.0532	,-0.054	,-0.0547	,-0.0554	,-0.056	,-0.0565	,-0.057	,-0.0575	,-0.0579	,-0.0583	,-0.0586	,-0.0589	,-0.0592	,-0.0594	,-0.0595	,-0.0596	,-0.0597	,-0.0598	,-0.0598	,-0.0598	,-0.0598	,-0.0597	,-0.0596	,-0.0594	,-0.0592	,-0.0589	,-0.0586	,-0.0582	,-0.0578	,-0.0573	,-0.0567	,-0.0561	,-0.0554	,-0.0546	,-0.0538	,-0.0529	,-0.0519	,-0.0509	,-0.0497	,-0.0485	,-0.0472	,-0.0458	,-0.0444], dtype = 'complex128')

    input_arg = sys.argv[1]  
    
    # these are different MDA solver options, see run_classes.py for more.
    solver_options = ['gs_wo_aitken', 'gs_w_aitken', 'newton_gmres', 'newton_direct']
    solver_atol = 5e-6
    
    g_factor = 2.5

    # Set problem type
    prob_dict = {'type' : 'aerostruct',
                 'with_viscous' : True,
                #  'force_fd' : True,
                 'optimizer': 'SNOPT',
                 'cg' : np.array([30., 0., 5.]),
                 'solver_combo' : solver_options[1],
                 'solver_atol' : solver_atol,
                 'print_level' : 0,
                 'compute_static_margin' : False,
                 # Flow/environment properties
                 'alpha' : 0.,            # [degrees] angle of attack
                 'reynolds_length' : 1.0, # characteristic Reynolds length
                 'M' : 0.85,              # Mach number at cruise
                 'rho' : 0.348,            # [kg/m^3] air density at 35,000 ft
                 'a' : 295.07,             # [m/s] speed of sound at 35,000 ft
                 'Re' : 0.348*295*.85*1./(1.45*1e-5), # Reynolds number for characteristic Re length
                 'g' : 9.80665,           # [m/s^2] acceleration due to gravity

                 # Aircraft properties
                 'CT' : 0.53/3600, # [1/s] (9.80665 N/kg * 17e-6 kg/N/s)
                                          # specific fuel consumption
                 'R' : 14.307e6,            # [m] maximum range (B777-300)
                 'cg' : np.zeros((3)), # Center of gravity for the
                                       # entire aircraft. Used in trim
                                       # and stability calculations.
                 'W0' : (143000 - 2.5*11600 + 15000 + 15000), # [kg] info from Tim: 143000 is ~empty W of 777-LR, 11600 is 1/2 wing structural weight guess, 15000 is reserves, another 15000 is 2 engines
                 'beta' : 1.,            # weighting factor for mixed objective
                 'g_factor' : g_factor,
                 'rho_25g' : 0.922,
                 }
                 
    if input_arg.startswith('0'):  # run analysis once
        prob_dict.update({'optimize' : False})
    else:  # perform optimization
        prob_dict.update({'optimize' : True})


    # Instantiate problem and add default surface
    OAS_prob = OASProblem(prob_dict)

                 
    surf_dict = {'num_y' : 31,
                 'num_x' : 3,
                 'exact_failure_constraint' : False,
                 'wing_type' : 'CRM',
                 'span_cos_spacing' : 0,
                 'chord_cos_spacing' : 0,
                 'CL0' : 0.0,
                 'CD0' : 0.010,
                 'symmetry' : True,
                 'S_ref_type' : 'wetted',
                 'num_twist_cp' : 5,
                 'num_thickness_cp' : 5,
                 'twist_cp' : np.array([4., 4., 4., 4., 4.]),
                 'thickness_cp' : np.array([0.01, 0.02, 0.03, 0.03, 0.03]), # this thickness variable does not do anything, but keep it for now because run_classes expects it. This will be fixed later.
                 # The following two are thickness variables that differ from the thickness variable in the standard OAS.
                 'skinthickness_cp' : np.array([0.01, 0.02, 0.03, 0.03, 0.03]),
                 'sparthickness_cp' : np.array([0.01, 0.01, 0.01, 0.01, 0.01]),
                # Material properties taken from http://www.performance-composites.com/carbonfibre/mechanicalproperties_2.asp
                # 'E' : 45.e9,
                # 'G' : 15.e9,
                # 'yield' : 350.e6 / 2.0,
                # 'mrho' : 1.6e3,
                # Airfoil properties for viscous drag calculation
                'k_lam' : 0.05,         # percentage of chord with laminar
                                        # flow, used for viscous drag
                't_over_c' : 0.12,      # thickness over chord ratio 
                'c_max_t' : .38,       # chordwise location of maximum 
                                        # thickness
                'E' : 73.1e9,            # [Pa] Young's modulus of the spar
                'G' : 27.e9,            # [Pa] shear modulus of the spar
                'yield' : (420.e6/ 2.5 / 1.5 * g_factor), # [Pa] yield stress divided by 2.5 for limiting case
                'mrho' : 2.78e3,          # [kg/m^3] material density
                'strength_factor_for_upper_skin' : 1.0, # for the upper skin, the yield stress is multiplied by this factor
                'sweep' : 0.,
                'monotonic_con' : None, # add monotonic constraint to the given
                                        # distributed variable. Ex. 'chord_cp'
                 # The following are the airfoil coordinates for the wingbox section (e.g., from 15% to 65%)
                 # The chord for the corresponding airfoil should be 1
                 # The first and last x-coordinates of the upper and lower skins must be the same
                 # The datatype should be complex to work with the complex-step approximation for derivatives
                 'data_x_upper' : upper_x,
                 'data_x_lower' : lower_x,
                 'data_y_upper' : upper_y,
                 'data_y_lower' : lower_y,
                 'toverc_cp' : np.array([0.12, 0.12, 0.12, 0.12, 0.12]),
                }

    # Add the specified wing surface to the problem
    OAS_prob.add_surface(surf_dict)

    # Add design variables, constraint, and objective on the problem
    # OAS_prob.add_desvar('alpha', lower=-10., upper=10.)
    OAS_prob.add_desvar('alpha_25g', lower=-10., upper=10.)
    OAS_prob.add_constraint('L_equals_W', equals=0.)
    OAS_prob.add_constraint('total_perf_25g.L_equals_W', equals=0.)
    OAS_prob.add_constraint('coupled.wing.S_ref', lower=414.1)
    OAS_prob.add_objective('fuelburn', scaler=1e-5)

    # Setup problem and add design variables, constraint, and objective
    OAS_prob.add_desvar('wing.twist_cp', lower=-10., upper=10.)
    OAS_prob.add_desvar('wing.sparthickness_cp', lower=0.002, upper=0.1, scaler=1e2)
    OAS_prob.add_desvar('wing.skinthickness_cp', lower=0.002, upper=0.1, scaler=1e2)
    OAS_prob.add_desvar('wing.toverc_cp', lower=0.06, upper=0.14, scaler=10.)
    OAS_prob.add_constraint('wing_perf.failure', upper=0.)
    OAS_prob.add_constraint('wing_perf_25g.failure', upper=0.)
    OAS_prob.add_desvar('wing.span', lower=10., upper=100.)
    OAS_prob.add_desvar('wing.sweep', lower=-60., upper=60.)
    OAS_prob.setup()

    st = time()
    # Actually run the problem
    OAS_prob.run()

    print("\nFuelburn:", OAS_prob.prob['fuelburn'])
    print("Time elapsed: {} secs".format(time() - st))
    # print(OAS_prob.prob['wing.thickness_cp'])
    print(OAS_prob.prob['wing.skinthickness_cp'])
    print(OAS_prob.prob['wing.sparthickness_cp'])
    # print(OAS_prob.prob['wing_perf.disp'])
    print(OAS_prob.prob['wing_perf.structural_weight'])
    print("Span", OAS_prob.prob['wing.span'],"m")
    print("Sweep", OAS_prob.prob['wing.sweep'])
            