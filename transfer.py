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
        # self.fem_origin = surface['fem_origin']
        # self.fem_origin = (surface['data_x_upper'][0] + surface['data_x_upper'][-1]) / 2.
        self.fem_origin = (surface['data_x_upper'][0] *(surface['data_y_upper'][0]-surface['data_y_lower'][0]) + \
        surface['data_x_upper'][-1]*(surface['data_y_upper'][-1]-surface['data_y_lower'][-1])) / \
        ( (surface['data_y_upper'][0]-surface['data_y_lower'][0]) + (surface['data_y_upper'][-1]-surface['data_y_lower'][-1]))

        self.add_param('mesh', val=np.zeros((self.nx, self.ny, 3), dtype=data_type))
        self.add_param('disp', val=np.zeros((self.ny, 6), dtype=data_type))
        self.add_output('def_mesh', val=np.zeros((self.nx, self.ny, 3), dtype=data_type))

        if not fortran_flag:
            self.deriv_options['type'] = 'cs'

    def solve_nonlinear(self, params, unknowns, resids):
        mesh = params['mesh']
        disp = params['disp']

        # Get the location of the spar within the wing and save as w
        # w = self.surface['fem_origin']
        # w = (self.surface['data_x_upper'][0] + self.surface['data_x_upper'][-1]) / 2.
        surface = self.surface
        w = (surface['data_x_upper'][0] *(surface['data_y_upper'][0]-surface['data_y_lower'][0]) + \
        surface['data_x_upper'][-1]*(surface['data_y_upper'][-1]-surface['data_y_lower'][-1])) / \
        ( (surface['data_y_upper'][0]-surface['data_y_lower'][0]) + (surface['data_y_upper'][-1]-surface['data_y_lower'][-1]))

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
                mesh_disp[:, ind, :] += Smesh[:, ind, :].dot(T)
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

    def __init__(self, surface):
        super(TransferLoads, self).__init__()

        self.surface = surface

        self.ny = surface['num_y']
        self.nx = surface['num_x']
        # self.fem_origin = surface['fem_origin']
        # self.fem_origin = (surface['data_x_upper'][0] + surface['data_x_upper'][-1]) / 2.
        self.fem_origin = (surface['data_x_upper'][0] *(surface['data_y_upper'][0]-surface['data_y_lower'][0]) + \
        surface['data_x_upper'][-1]*(surface['data_y_upper'][-1]-surface['data_y_lower'][-1])) / \
        ( (surface['data_y_upper'][0]-surface['data_y_lower'][0]) + (surface['data_y_upper'][-1]-surface['data_y_lower'][-1]))

        self.add_param('def_mesh', val=np.zeros((self.nx, self.ny, 3), dtype=complex))
        self.add_param('sec_forces', val=np.zeros((self.nx-1, self.ny-1, 3),
                       dtype=complex))
        self.add_output('loads', val=np.zeros((self.ny, 6),
                        dtype=complex))

        self.deriv_options['type'] = 'cs'
        self.deriv_options['check_type'] = 'fd'
        self.deriv_options['check_form'] = 'central'

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

        unknowns['loads'] = loads
