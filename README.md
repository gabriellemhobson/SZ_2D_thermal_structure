# SZ_2D_thermal_structure 

This code was developed by Gabrielle Hobson and Dave May and is distributed under the GNU GPL v3 license. It is part of the Megathrust Modeling Framework ([MTMOD](https://sites.utexas.edu/mtmod/)), supported by NSF FRES grant EAR-2121568. 

If you use this code, please cite it using this citation:

Hobson, G. M., & May, D. A. (2024). Sensitivity analysis of the thermal structure within subduction zones using reduced-order modeling (v1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.12765667

## Install and environment 

Starting from the command line, this github repository can be cloned like so: 

`git clone https://github.com/gabriellemhobson/SZ_2D_thermal_structure`

Once you have navigated to the main directory `SZ_2D_thermal_structure`, use the `environment.yaml` file to create a conda environment using the following line of code. 

`conda env create -f environment.yml`

This will create a conda environment named `SZ_2D_thermal_structure`.

This code has been tested for compatibility with PETSc versions 3.12, 3.14 and 3.17. There are differences between these versions that are handled in the code. Other versions of PETSc have not been tested and may not be compatible. 

This code also depends on [ParametricModelUtils](https://github.com/hpc4geo/ParametricModelUtils), a "tool for defining and running parametric computational models". It should be added to the `SZ_2D_thermal_structure` directory. 

`git clone https://github.com/hpc4geo/ParametricModelUtils` 

Finally, the mesh generation process requires the freely available software [GMSH](https://gmsh.info/), version 4.10. Make sure that you can run gmsh commands on the command line - this may require adding the GMSH app to your PATH variable like so: 

`export PATH=:$PATH:/path/to/Gmsh.app`

You can check that GMSH commands work on the command line by entering this line, for example. 

`gmsh --info`

## Usage

### Generating meshes

The code `driver_generate_mesh_generic.py` takes as input start and end points for 2D profiles of slab data, in latitude and longitude coordinates, stored in a .csv file. It creates a .geo file for the geometry based on that profile. The argument `--write_msh` is by default set to `False`, but if it is included in the command line arguments, it automatically meshes the .geo file to create a .msh file with the computational mesh and several .msh files necessary for visualization purposes.

The code `convert_msh_to_fenics_files.py` converts .msh files to .xml files as required for fenics usage. It also writes other files with information about the mesh which are required for post-processing. 

```
cd generate_meshes 

# to just write the .geo files
python3 driver_generate_mesh_generic.py --profile_csv "cascadia_start_end_points.csv" --slab_name "cascadia" --corner_depth -35.0 --output_path "output"

# OR, to write .geo and .msh files
python3 driver_generate_mesh_generic.py --profile_csv "cascadia_start_end_points.csv" --slab_name "cascadia" --corner_depth -35.0 --output_path "output" --write_msh

cd ..

python3 convert_msh_to_fenics_files.py --mesh_dir 'generate_meshes/output/cascadia_profile_B' --profile_name 'cascadia_profile_B' 

```

#### Manually exporting meshes

If you did not include the --write_msh flag, there will just be .geo files written and not .msh files. To manually generate the correct msh files using the GMSH GUI:

- Open geo file, let’s say it is called filename.geo
- In mesh section of side menu, hit Mesh 2D
- File > Export, select .msh file type. The file name should be filename.msh. Hit Save
- MSH Options box will pop up, choose Version 2 ASCII

- Under Mesh, click Refine by Splitting (only once for P2, twice for P4 etc). 
- File > Export, change the filename to be filename_viz.msh. Then hit Save
- MSH Options box will pop up, choose Version 2 ASCII

- File > Export, change the filename to be filename_viz_v4_ascii.msh. Then hit Save
- MSH Options box will pop up, choose Version 4 ASCII

Both viz files should be placed in a subfolder called viz located in the same directory as the computation .geo and .msh files. 

### Parameter handling

The file `define_parameters.py` creates a class with the base case values for the parameters stored as attributes. An instance of this class is passed to the forward model solver. It handles the required nondimensionalization and also contains the function `set_param()` which is used to update or set existing parameter attributes while handling the nondimensionalization. 

The user should decide which parameters they wish to vary when running the forward model and set the ranges in the file `input_param.csv`. The file is space-delimited and should contain the parameter name, the min value, the max value, and the units. For example, to vary the slab velocity and coefficient of friction, the `input_param.csv` file should look like this:

```
# parameter name, min value, max value, unit
slab_vel 4.0 5.0 cm/yr
mu 0.02 0.04 ~
```

Here are some suggested ranges for parameters to vary. 

| Parameter name    | Description                | Reasonable range | Units |
| ---------------- | ---- | --- | --- |
| slab_vel | Slab velocity  | [3, 5] | cm/yr |
| ddc | Depth of decoupling | [70, 80] | km |
| deg_pc | Degree of partial coupling | [0, 0.1] | |
| mu | Coefficient of friction | [0, 0.1] | |
| A_diff | Diffusion creep pre-exp factor | [2.5 x $10^7$, 2.4 x $10^{10}$] | Pa s |
| E_diff | Diffusion creep activation energy | [300 x $10^3$, 450 x $10^3$] | J/mol |
| A_disl | Dislocation creep pre-exp factor  | [1 x $10^4$, 5 x $10^4$] | Pa s^(1/n) |
| E_disl | Dislocation creep activation energy | [480 x $10^3$, 560 x $10^3$] | J/mol |
| n | Power law exponent | [0, 3.5] | |
| Tb | Mantle inflow temperature | [1550, 1750] | °K |
| slab_age | Slab age | [8, 10] | Myr |
| z_bc | Depth of geotherm | [10, 60] | km |

### Running forward models

Running forward models is handled by `schedule_script.py`. The user passes in input arguments, which are specified below. The code in `schedule_script.py` creates an instance of the class contained in `forward_model.py`, given the input arguments. This class uses `ParametricModelUtils` so that forward models are batched, run, and their statuses tracked. In general usage, `forward_model.py` does not need to be edited or run. It is just called by `schedule_script.py`. 

**Required arguments for `schedule_script.py`:**

| Argument name    | type | Description                                                 | Options |
| ---------------- | ---- | ---                                                         | --- |
| --profile_name   | str  | Which slice to use                                          | |
| --mesh_dir       | str  | Path to directory with mesh                                 | |
| --output_path    | str  | Output directory                                            | | 
| --sample_method  | str  | Which method to use to sample parameter space.              | 'halton' or 'latinhypercube'|
| --n1             | int  | Number of samples to draw initially to advance the sequence | |
| --n2             | int  | Number of forward models to run                             | |
| --seed           | int  | Seed for the sequence of samples drawn from parameter space | |
| --jobs_csv       | str  | Name of csv file to tracks jobs run.                        | |
| --viscosity_type | str  | Type of viscosity flow law                                  | 'isoviscous', 'diffcreep', 'disccreep', or 'mixed' |

**Optional arguments for `schedule_script.py`:**

| Argument name    | type | Description                                                 | Default |
| ---------------- | ---- | ---                                                         | --- |
| --solver  | str | Type of forward model solver, must be 'ss' or 'time_dep'. | 'ss' |
| --tol  | float | Residuals for pde solutions in forward model solve must be below this tolerance. | 1e-5|
| --n_picard_it | int | Maximum number of picard iterations. | 10 |
| --n_iters | int | Maximum number of iterations. | 10 | 
| --diff_tol | float | Tolerance for reaching a converged solution. | 1e-1 |
| --T_CG_order | int | Order of CG elements for temperature. | 2 | 

### Post-processing and plotting

The script `post_process.py` handles reordering the model output, creating a plot of the thermal structure, computing the locations of intersections between specific isotherms and the slab interface, and creating a plot of those isotherms. 

`python3 post_process.py --jobs_csv "cascadia_profile_B_example_log.csv" --mesh_path "generate_meshes/output/cascadia_profile_B" --profile_name "cascadia_profile_B" --include "halton"`

### Start-to-finish usage

- Activate the conda environment 

    `conda activate SZ_2D_thermal_structure`

- Check the repo is up to date and paths are correct

    `source setup.sh`

- Enter the generate_mesh subdirectory 

    `cd generate_meshes`

- Check that the slab profile info in the csv input file is correct. 

- Generate the geometry and mesh files:

    `python3 driver_generate_mesh_generic.py --profile_csv "cascadia_start_end_points.csv" --slab_name "cascadia" --corner_depth -35.0 --output_path "output" --write_msh`

- Create the required mesh files for fenics usage and post-processing steps. 

    `cd ..`

    `python3 convert_msh_to_fenics_files.py --mesh_dir 'generate_meshes/output/cascadia_profile_B' --profile_name 'cascadia_profile_B' `

- Set the desired ranges of input parameters in `input_param.csv`. 

- Set the forward model running. 

    `python3 schedule_script.py --profile_name "cascadia_profile_B" --mesh_dir "generate_meshes/output/cascadia_profile_B" --output_path "output/cascadia_profile_B/example" --sample_method "halton" --n1 1 --n2 1 --seed 92014 --jobs_csv "cascadia_profile_B_example_log.csv" --viscosity_type "isoviscous" `

- Once the forward model is done, perform the post-processing steps to create plots and compute isotherm-slab interface intersection locations. 

    `python3 post_process.py --jobs_csv "cascadia_profile_B_example_log.csv" --mesh_path "generate_meshes/output/cascadia_profile_B" --profile_name "cascadia_profile_B" --include "halton"`
    
- Look at your plots and be proud that you've run this code!
