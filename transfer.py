"""
Define the transfer components to couple aero and struct analyses.
"""

from __future__ import division, print_function
import numpy as np

try:
    import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex

from openmdao.api import Component

def norm(vec):
    return np.sqrt(np.sum(vec**2))

class TransferDisplacements(Component):
    """
    Perform displacement transfer.

    Apply the computed displacements on the original mesh to obtain
    the deformed mesh.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Flattened array defining the lifting surfaces.
    disp[ny, 6] : numpy array
        Flattened array containing displacements on the FEM component.
        Contains displacements for all six degrees of freedom, including
        displacements in the x, y, and z directions, and rotations about the
        x, y, and z axes.

    Returns
    -------
    def_mesh[nx, ny, 3] : numpy array
        Flattened array defining the lifting surfaces after deformation.
    """

    def __init__(self, surface):
        super(TransferDisplacements, self).__init__()

        self.surface = surface

        self.ny = surface['num_y']
        self.nx = surface['num_x']
        self.fem_origin = (surface['data_x_upper'][0] *(surface['data_y_upper'][0]-surface['data_y_lower'][0]) + \
        surface['data_x_upper'][-1]*(surface['data_y_upper'][-1]-surface['data_y_lower'][-1])) / \
        ( (surface['data_y_upper'][0]-surface['data_y_lower'][0]) + (surface['data_y_upper'][-1]-surface['data_y_lower'][-1]))

        self.add_param('mesh', val=np.zeros((self.nx, self.ny, 3), dtype=data_type))
        self.add_param('disp', val=np.zeros((self.ny, 6), dtype=data_type))
        self.add_output('def_mesh', val=np.zeros((self.nx, self.ny, 3), dtype=data_type))

        if not fortran_flag:
            self.deriv_options['type'] = 'fd'

    def solve_nonlinear(self, params, unknowns, resids):
        mesh = params['mesh']
        disp = params['disp']

        # Get the location of the spar within the wing and save as w
        w = self.fem_origin

        # Run Fortran if possible
        if fortran_flag:
            def_mesh = OAS_API.oas_api.transferdisplacements(mesh, disp, w)

        else:

            # Get the location of the spar
            ref_curve = (1-w) * mesh[0, :, :] + w * mesh[-1, :, :]

            # Compute the distance from each mesh point to the nodal spar points
            Smesh = np.zeros(mesh.shape, dtype=data_type)
            for ind in range(self.nx):
                Smesh[ind, :, :] = mesh[ind, :, :] - ref_curve

            # Set up the mesh displacements array
            mesh_disp = np.zeros(mesh.shape, dtype=data_type)
            cos, sin = np.cos, np.sin

            # Loop through each spanwise FEM element
            for ind in range(self.ny):
                dx, dy, dz, rx, ry, rz = disp[ind, :]

                # 1 eye from the axis rotation matrices
                # -3 eye from subtracting Smesh three times
                T = -2 * np.eye(3, dtype=data_type)
                T[ 1:,  1:] += [[cos(rx), -sin(rx)], [ sin(rx), cos(rx)]]
                T[::2, ::2] += [[cos(ry),  sin(ry)], [-sin(ry), cos(ry)]]
                T[ :2,  :2] += [[cos(rz), -sin(rz)], [ sin(rz), cos(rz)]]

                # Obtain the displacements on the mesh based on the spar response
                mesh_disp[:, ind, :] += np.dot(T, Smesh[:, ind, :].T).T
                mesh_disp[:, ind, 0] += dx
                mesh_disp[:, ind, 1] += dy
                mesh_disp[:, ind, 2] += dz

            # Apply the displacements to the mesh
            def_mesh = mesh + mesh_disp

        unknowns['def_mesh'] = def_mesh

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        mesh = params['mesh']
        disp = params['disp']

        # w = self.surface['fem_origin']
        # w = (self.surface['data_x_upper'][0] + self.surface['data_x_upper'][-1]) / 2.
        surface = self.surface
        w = (surface['data_x_upper'][0] *(surface['data_y_upper'][0]-surface['data_y_lower'][0]) + \
        surface['data_x_upper'][-1]*(surface['data_y_upper'][-1]-surface['data_y_lower'][-1])) / \
        ( (surface['data_y_upper'][0]-surface['data_y_lower'][0]) + (surface['data_y_upper'][-1]-surface['data_y_lower'][-1]))

        if mode == 'fwd':
            a, b = OAS_API.oas_api.transferdisplacements_d(mesh, dparams['mesh'], disp, dparams['disp'], w)
            dresids['def_mesh'] += b.real

        if mode == 'rev':
            a, b = OAS_API.oas_api.transferdisplacements_b(mesh, disp, w, unknowns['def_mesh'], dresids['def_mesh'])
            dparams['mesh'] += a.real
            dparams['disp'] += b.real

class TransferLoads(Component):
    """
    Perform aerodynamic load transfer.

    Apply the computed sectional forces on the aerodynamic surfaces to
    obtain the deformed mesh FEM loads.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : numpy array
        Flattened array defining the lifting surfaces after deformation.
    sec_forces[nx-1, ny-1, 3] : numpy array
        Flattened array containing the sectional forces acting on each panel.
        Stored in Fortran order (only relevant when more than one chordwise
        panel).

    Returns
    -------
    loads[ny, 6] : numpy array
        Flattened array containing the loads applied on the FEM component,
        computed from the sectional forces.
    """

    def __init__(self, surface, prob_dict, g_factor):
        super(TransferLoads, self).__init__()

        self.surface = surface
        self.prob_dict = prob_dict
        self.g_factor = g_factor

        self.ny = surface['num_y']
        self.nx = surface['num_x']
        self.fem_origin = (surface['data_x_upper'][0] *(surface['data_y_upper'][0]-surface['data_y_lower'][0]) + \
        surface['data_x_upper'][-1]*(surface['data_y_upper'][-1]-surface['data_y_lower'][-1])) / \
        ( (surface['data_y_upper'][0]-surface['data_y_lower'][0]) + (surface['data_y_upper'][-1]-surface['data_y_lower'][-1]))

        self.add_param('def_mesh', val=np.zeros((self.nx, self.ny, 3), dtype=complex))
        self.add_param('sec_forces', val=np.zeros((self.nx-1, self.ny-1, 3),
                       dtype=complex))
        self.add_param('fuelburn', val=1.)
        self.add_param('A', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_param('A_int', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_output('loads', val=np.zeros((self.ny, 6),
                        dtype=complex))
        self.add_output('fuel_vol_delta', val=1.)

        self.deriv_options['type'] = 'cs'
        self.deriv_options['check_type'] = 'fd'
        self.deriv_options['check_form'] = 'central'
        
        self.element_lengths = np.ones(self.ny - 1, complex)

    def solve_nonlinear(self, params, unknowns, resids):
        mesh = params['def_mesh']

        sec_forces = params['sec_forces']

        # Compute the aerodynamic centers at the quarter-chord point of each panel
        w = 0.25
        a_pts = 0.5 * (1-w) * mesh[:-1, :-1, :] + \
                0.5 *   w   * mesh[1:, :-1, :] + \
                0.5 * (1-w) * mesh[:-1,  1:, :] + \
                0.5 *   w   * mesh[1:,  1:, :]

        # Compute the structural midpoints based on the fem_origin location
        w = self.fem_origin
        s_pts = 0.5 * (1-w) * mesh[0, :-1, :] + \
                0.5 *   w   * mesh[-1, :-1, :] + \
                0.5 * (1-w) * mesh[0,  1:, :] + \
                0.5 *   w   * mesh[-1,  1:, :]

        # Find the moment arm between the aerodynamic centers of each panel
        # and the FEM elmeents
        diff = a_pts - s_pts
        moment = np.zeros((self.ny - 1, 3), dtype=complex)
        for ind in range(self.nx-1):
            moment += np.cross(diff[ind, :, :], sec_forces[ind, :, :], axis=1)

        # Compute the loads based on the xyz forces and the computed moments
        loads = np.zeros((self.ny, 6), dtype=complex)
        sec_forces_sum = np.sum(sec_forces, axis=0)
        loads[:-1, :3] += 0.5 * sec_forces_sum[:, :]
        loads[ 1:, :3] += 0.5 * sec_forces_sum[:, :]
        loads[:-1, 3:] += 0.5 * moment
        loads[ 1:, 3:] += 0.5 * moment

        #=======================================================================
        # For fuel- and structural-weight loads
        #=======================================================================
        
        # Fuel weight
        fuel_weight = (params['fuelburn']/2. + self.prob_dict['Wf_reserve']/2.) * self.prob_dict['g'] * self.g_factor
        
        # First we need element lengths
        nodes = (1-w) * mesh[0, :, :] + w * mesh[-1, :, :]
        for i in range(self.ny - 1):
            self.element_lengths[i] = norm(nodes[i+1] - nodes[i])
        
        # And we also need the deltas between consecutive nodes
        deltas = nodes[1:, :] - nodes[:-1, :]
                
        # Next we multiply the element lengths with the A_int for the internal volumes of the wingobox segments
        vols = self.element_lengths * params['A_int']
        
        sum_vols = np.sum(vols)
        
        # Now we need the fuel weight per segment
        # Assume it's divided evenly based on vols
        z_weights = vols * fuel_weight / sum_vols
        
        # Adding wing structural weights
        struct_weights = self.element_lengths * params['A'] * self.surface['mrho'] * self.prob_dict['g'] * self.g_factor * self.prob_dict['W_wing_factor']
        z_weights += struct_weights
        
        # Assume weight coincides with the elastic axis
        z_forces_for_each = z_weights / 2.
        z_moments_for_each = z_weights * self.element_lengths / 12. * (deltas[:, 0]**2 + deltas[:,1]**2)**0.5 / self.element_lengths
        
        # Loads in z-direction
        loads[:-1, 2] += -z_forces_for_each
        loads[1:, 2] += -z_forces_for_each
        
        # Bending moments for consistency
        loads[:-1, 3] += -z_moments_for_each * deltas[: , 1] / self.element_lengths
        loads[1:, 3] += z_moments_for_each * deltas[: , 1] / self.element_lengths
        
        loads[:-1, 4] += -z_moments_for_each * deltas[: , 0] / self.element_lengths
        loads[1:, 4] += z_moments_for_each * deltas[: , 0] / self.element_lengths
        
        unknowns['loads'] = loads
        
        # This is used for the fuel-volume constraint. It should be positive for fuel to fit. 
        # Fuel density is assumed to be 803 kg/m^3
        unknowns['fuel_vol_delta'] = sum_vols - (params['fuelburn']/2. + self.prob_dict['Wf_reserve']/2.) / 803
