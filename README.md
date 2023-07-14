# SZ_2D_thermal_structure

This code was developed by Gabrielle Hobson and Dave May and is distributed under the "LICENSE". It is part of the Megathrust Modeling Framework ([MTMOD](https://sites.utexas.edu/mtmod/), supported by NSF FRES grant EAR-2121568. 

### Install and environment 

Starting from the command line, this github repository can be cloned like so: 

`git clone https://github.com/gabriellemhobson/SZ_2D_thermal_structure`

Once you have navigated to the main directory `SZ_2D_thermal_structure`, use the `environment.yaml` file to create a conda environment using the following line of code. 

`conda env create -f environment.yml`

This will create a conda environment named `NAME HERE`. This code requires FEniCS 2019.1.0 which can be added to the environment like so:

`conda install -c conda-forge fenics`

This code has been tested for compatibility with PETSc versions 3.12, 3.14 and 3.17. There are differences between these versions that are handled in the code. Other versions of PETSc have not been tested and may not be compatible. 

