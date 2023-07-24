'''
Processing a mesh. Requires the user to define the compute mesh, the visualization mesh, 
and a v4 ascii version of the visualization mesh. Also requires the user to define the T_CG_order to use. 
Uses gmshparser to get the tags of nodes on the slab interface and writes the X,Y,T info to file. 
Then computes the transfer matrix between the T_CG_order FunctionSpace associated with the compute mesh 
and forward model solutions and the CG 1 order FunctionSpace associated with the visualization mesh. 
This transfer matrix is written to file for later use. 
'''
import os
import subprocess
import argparse
from utils import process_mesh

class CMDA: # cmdline_args
    pass

args = CMDA()
parser = argparse.ArgumentParser()
parser.add_argument('--mesh_dir', type=str, required=True, help="Path where mesh files are stored.")
parser.add_argument('--profile_name', type=str, required=True, help="Name of specific profile to convert")
parser.parse_known_args(namespace=args)

v4_vizfile_name = os.path.join(args.mesh_dir,os.path.join('viz', args.profile_name+'_viz_v4_ascii.msh'))
meshfile_name = os.path.join(args.mesh_dir,args.profile_name)
vizfile_name = os.path.join(args.mesh_dir,os.path.join('viz', args.profile_name+'_viz'))
T_CG_order = 2

command = ["dolfin-convert", meshfile_name+".msh", meshfile_name+".xml"]
subprocess.run(command)

command = ["dolfin-convert", vizfile_name+".msh", vizfile_name+".xml"]
subprocess.run(command)

# get the tags associated with the slab interface
pm = process_mesh.Process_Mesh(T_CG_order,meshfile_name,vizfile_name,v4_vizfile_name,args.mesh_dir)
# Write to file 
X,Y,TAGS = pm.get_slab_interface_tags()

# write the transfer matrix that goes from a CG 4 FunctionSpace defined on the mesh to a CG 1 FunctionSpace defined on the viz mesh
matrix_filename = os.path.join(args.mesh_dir, "M.dat")
pm.write_transfer_matrix(matrix_filename)

# write the x y coord and cell data to file for later plotting
pm.write_plotting_data()

# compute the distance fields
pm.compute_distance_fields(plot_or_not=False)
