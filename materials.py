from __future__ import division, print_function
import numpy as np

from openmdao.api import Component

def wingbox_props(chord, sparthickness, skinthickness, data_x_upper, data_x_lower, data_y_upper, data_y_lower, twist=0.):
    
    # Scale data points with chord 
    data_x_upper = chord * data_x_upper
    data_y_upper = chord * data_y_upper
    data_x_lower = chord * data_x_lower
    data_y_lower = chord * data_y_lower
    
    thickness_skin = skinthickness
    thickness_spar = sparthickness
    
    # Compute enclosed area for torsion constant
    # This currently does not change with twist
    A_enc = 0
    for i in range(data_x_upper.size-1):
        
        A_enc += (data_x_upper[i+1] - data_x_upper[i]) * (data_y_upper[i+1] + data_y_upper[i] - thickness_skin ) / 2 # area above 0 line
        A_enc += (data_x_lower[i+1] - data_x_lower[i]) * (-data_y_lower[i+1] - data_y_lower[i] - thickness_skin ) / 2 # area below 0 line

    A_enc -= (data_y_upper[0] - data_y_lower[0]) * thickness_spar / 2 # area of spars
    A_enc -= (data_y_upper[-1] - data_y_lower[-1]) * thickness_spar / 2 # area of spars

    # Compute perimeter to thickness ratio for torsion constant
    # This currently does not change with twist
    p_by_t = 0
    for i in range(data_x_upper.size-1):
        p_by_t += ((data_x_upper[i+1] - data_x_upper[i])**2 + (data_y_upper[i+1] - data_y_upper[i])**2)**0.5 / thickness_skin # length / thickness of caps
        p_by_t += ((data_x_lower[i+1] - data_x_lower[i])**2 + (data_y_lower[i+1] - data_y_lower[i])**2)**0.5 / thickness_skin # length / thickness of caps
        
    p_by_t += (data_y_upper[0] - data_y_lower[0] - thickness_skin) / thickness_spar # length / thickness of spars
    p_by_t += (data_y_upper[-1] - data_y_lower[-1] - thickness_skin) / thickness_spar # length / thickness of spars

    # Torsion constant
    J = 4 * A_enc**2 / p_by_t

    # Rotate the wingbox
    theta = twist

    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

    data_x_upper_2 = data_x_upper.copy()
    data_y_upper_2 = data_y_upper.copy()
    data_x_lower_2 = data_x_lower.copy()
    data_y_lower_2 = data_y_lower.copy()
        
    for i in range(data_x_upper.size):
        
        data_x_upper_2[i] = rot_mat[0,0] * data_x_upper[i] + rot_mat[0,1] * data_y_upper[i]
        data_y_upper_2[i] = rot_mat[1,0] * data_x_upper[i] + rot_mat[1,1] * data_y_upper[i]

        data_x_lower_2[i] = rot_mat[0,0] * data_x_lower[i] + rot_mat[0,1] * data_y_lower[i]
        data_y_lower_2[i] = rot_mat[1,0] * data_x_lower[i] + rot_mat[1,1] * data_y_lower[i]
        
    data_x_upper = data_x_upper_2.copy()
    data_y_upper = data_y_upper_2.copy()
    data_x_lower = data_x_lower_2.copy()
    data_y_lower = data_y_lower_2.copy()
    
    # Compute area moment of inertia about x axis
    # First compute centroid and area
    first_moment_area_upper = 0
    upper_area = 0
    first_moment_area_lower = 0
    lower_area = 0
    for i in range(data_x_upper.size-1):
        first_moment_area_upper += ((data_y_upper[i+1] + data_y_upper[i]) / 2 - (thickness_skin/2) ) * thickness_skin * (data_x_upper[i+1] - data_x_upper[i])
        upper_area += thickness_skin * (data_x_upper[i+1] - data_x_upper[i])
        
        first_moment_area_lower += ((data_y_lower[i+1] + data_y_lower[i]) / 2 + (thickness_skin/2) ) * thickness_skin * (data_x_lower[i+1] - data_x_lower[i])
        lower_area += thickness_skin * (data_x_lower[i+1] - data_x_lower[i])

    area = upper_area + lower_area
    centroid = (first_moment_area_upper + first_moment_area_lower) / area
    
    # Then compute area moment of inertia for upward bending
    # This is calculated using derived analytical expression assuming linear interpolation between airfoil data points
    I_horiz = 0
    for i in range(data_x_upper.size-1): # upper surface
        a = (data_y_upper[i] - data_y_upper[i+1]) / (data_x_upper[i] - data_x_upper[i+1])
        b = (data_y_upper[i+1] - data_y_upper[i] + thickness_skin) / 2
        x2 = data_x_upper[i+1] - data_x_upper[i]
        
        I_horiz += 2 * ((1./12. * a**3 * x2**4 + 1./3. * a**2 * x2**3 * b + 1./2. * a * x2**2 * b**2 + 1./3. * b**3 * x2))
        
        I_horiz += x2  * thickness_skin * ((data_y_upper[i] + data_y_upper[i+1])/2 - thickness_skin/2 - centroid)**2

    
    # Compute area moment of inertia about y axis
    for i in range(data_x_lower.size-1): # lower surface
        a = -(data_y_lower[i] - data_y_lower[i+1]) / (data_x_lower[i] - data_x_lower[i+1])
        b = (-data_y_lower[i+1] + data_y_lower[i] + thickness_skin) / 2
        x2 = data_x_lower[i+1] - data_x_lower[i]
        
        I_horiz += 2 * ((1./12. * a**3 * x2**4 + 1./3. * a**2 * x2**3 * b + 1./2. * a * x2**2 * b**2 + 1./3. * b**3 * x2))
        
        I_horiz += x2 * thickness_skin * ((-data_y_lower[i] - data_y_lower[i+1])/2 - thickness_skin/2 + centroid)**2
        

    # Compute area moment of inertia for backward bending
    I_vert = 0
    first_moment_area_left = (data_y_upper[0] - data_y_lower[0]) * thickness_spar * (data_x_upper[0] + thickness_spar / 2)
    first_moment_area_right = (data_y_upper[-1] - data_y_lower[-1]) * thickness_spar * (data_x_upper[-1] - thickness_spar / 2)
    centroid_Ivert = (first_moment_area_left + first_moment_area_right) / \
                    ( ((data_y_upper[0] - data_y_lower[0]) + (data_y_upper[-1] - data_y_lower[-1])) * thickness_spar)

    I_vert += 1./12. * (data_y_upper[0] - data_y_lower[0]) * thickness_spar**3 + (data_y_upper[0] - data_y_lower[0]) * thickness_spar * (centroid_Ivert - (data_x_upper[0] + thickness_spar/2))**2
    I_vert += 1./12. * (data_y_upper[-1] - data_y_lower[-1]) * thickness_spar**3 + (data_y_upper[-1] - data_y_lower[-1]) * thickness_spar * (data_x_upper[-1] - thickness_spar/2 - centroid_Ivert)**2
    
    # Add contribution of skins
    I_vert += 2 * ( 1./12. * thickness_skin * (data_x_upper[-1] - data_x_upper[0] - 2 * thickness_spar)**3 + thickness_skin * (data_x_upper[-1] - data_x_upper[0] - 2 * thickness_spar) * (centroid_Ivert - (data_x_upper[-1] + data_x_upper[0]) / 2)**2 )

    area_spar = ((data_y_upper[0] - data_y_lower[0] - 2 * thickness_skin) + (data_y_upper[-1] - data_y_lower[-1] - 2 * thickness_skin)) * thickness_spar 
    area += area_spar
    
    # Distances for calculating max bending stresses (KS function used)
    ks_rho = 500. # Hard coded, see Martins and Poon 2005 for more
    fmax_upper = np.max(data_y_upper)
    htop = fmax_upper + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (data_y_upper - fmax_upper)))) - centroid
    
    fmax_lower = np.max(-data_y_lower)
    hbottom = fmax_lower + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (-data_y_lower - fmax_lower)))) + centroid
    
    hleft =  centroid_Ivert - data_x_upper[0]
    hright = data_x_upper[-1] - centroid_Ivert
    
    return I_horiz, I_vert, J, area, A_enc, htop, hbottom, hleft, hright, area_spar

class MaterialsTube(Component):
    """
    Compute geometric properties for a tube element.
    The thicknesses are added to the interior of the element, so the
    'radius' value is the outer radius of the tube.

    Parameters
    ----------
    radius : numpy array
        Outer radii for each FEM element.
    thickness : numpy array
        Tube thickness for each FEM element.

    Returns
    -------
    A : numpy array
        Cross-sectional area for each FEM element.
    Iy : numpy array
        Area moment of inertia around the y-axis for each FEM element.
    Iz : numpy array
        Area moment of inertia around the z-axis for each FEM element.
    J : numpy array
        Polar moment of inertia for each FEM element.
    """

    def __init__(self, surface):
        super(MaterialsTube, self).__init__()

        self.surface = surface

        self.ny = surface['num_y']
        self.mesh = surface['mesh']
        name = surface['name']
        
        self.data_x_upper = surface['data_x_upper']
        self.data_x_lower = surface['data_x_lower']
        self.data_y_upper = surface['data_y_upper']
        self.data_y_lower = surface['data_y_lower']

        # self.add_param('radius', val=np.ones((self.ny - 1)))
        self.add_param('chords_fem', val=np.ones((self.ny - 1), dtype = complex))
        self.add_param('twist_fem', val=np.ones((self.ny - 1),  dtype = complex))
        self.add_param('thickness', val=np.ones((self.ny - 1), dtype = complex))
        
        self.add_param('sparthickness', val=np.ones((self.ny - 1), dtype = complex))
        self.add_param('skinthickness', val=np.ones((self.ny - 1),  dtype = complex))
        
        self.add_output('A', val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('A_enc', val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('A_spar', val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('Iy', val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('Iz', val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('J', val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('htop', val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('hbottom', val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('hleft', val=np.ones((self.ny - 1),  dtype = complex))
        self.add_output('hright', val=np.ones((self.ny - 1),  dtype = complex))

        self.arange = np.arange((self.ny - 1))
        
        self.deriv_options['type'] = 'cs'
        # self.deriv_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        
        for i in range(self.ny - 1):
            
            unknowns['Iz'][i], unknowns['Iy'][i], unknowns['J'][i], unknowns['A'][i], unknowns['A_enc'][i],\
            unknowns['htop'][i], unknowns['hbottom'][i], unknowns['hleft'][i], unknowns['hright'][i], unknowns['A_spar'][i]  = \
            wingbox_props(params['chords_fem'][i], params['sparthickness'][i], params['skinthickness'][i], self.data_x_upper, \
            self.data_x_lower, self.data_y_upper, self.data_y_lower, -params['twist_fem'][i])
