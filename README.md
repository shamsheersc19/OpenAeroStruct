# OpenAeroStruct + wingbox

OpenAeroStruct is a Python tool that performs aerostructural optimization of lifting surfaces using OpenMDAO. It uses a vortex lattice method (VLM) for the aerodynamics and a finite element method (FEM) spatial beam model with 6-DOF-per-node elements for the structures.
This repository contains a modified version of OpenAeroStruct that uses effective section properties of a wingbox.
See the [conference paper](http://mdolab.engin.umich.edu/sites/default/files/EngOpt_preprint_0.pdf) for more on the wingbox model.

![Optimized CRM-type wing example](/example.png?raw=true "Example Optimization Result and Visualization")

Check out some videos:

[Video with span and sweep design variables](https://youtu.be/c5qO-deZZgs)

[Video with span, sweep, and chord design variables](https://youtu.be/Q36UYrk4H64)

Documentation of the original OpenAeroStruct [version](https://github.com/mdolab/OpenAeroStruct) is available [here](http://openaerostruct.readthedocs.io/en/latest/).
We recommend gaining familiarity with the original [version](https://github.com/mdolab/OpenAeroStruct) first.

Please see the [SMO journal paper](https://link.springer.com/article/10.1007%2Fs00158-018-1912-8) for more information and please cite it, along with the conference paper linked above, if you use this version of OpenAeroStruct in your research. Here's an open-access read-only copy of the journal paper: http://rdcu.be/Gtl1


## Installation

To use this, you must first install OpenMDAO 1.7.4 by following the instructions here: https://github.com/openmdao/openmdao1. If you are unfamiliar with OpenMDAO and wish to modify the internals of OpenAeroStruct, you should examine the OpenMDAO documentation at http://openmdao.readthedocs.io/en/1.7.3/. Note that OpenMDAO 1.7.4 is the most recent version that has been tested and confirmed working with this code.

Next, clone this repository:

    git clone -b mpt_wingbox https://github.com/shamsheersc19/OpenAeroStruct.git

Lastly, from within the OpenAeroStruct folder, compile the Fortran library:

    make

Note that the code will run without compiling the Fortran library, but it will run significantly faster when using Fortran.

## Usage

`run_aerostruct_comp_CRM.py` contains an example script for aerostructural analysis and optimization using the wingbox model.
It also contains instructions on how to run it.

For each case, you can view the optimization results using `plot_all.py`.

An example workflow would be:

    python run_aerostruct_comp_CRM.py 1 3 31 1 uCRM_based
    python plot_all.py aerostruct.db

The first command performs aerostructural optimization and the second visualizes the optimization history.

The keywords used for each file are explained in their respective docstrings at the top of the file.

If you wish to examine the code in more depth, see `run_classes.py` and the methods it calls. These methods interface directly with OpenMDAO.

If using this code, please check regularly for updates and please direct any questions or concerns to sschau@umich.edu.

## Notes

This code may not currently work for multiple lifting surfaces, and for aerodynamics-only and structures-only analyses and optimizations. Further developments are planned.