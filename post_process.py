import numpy as np
import argparse
import pandas as pd
import pickle as pkl
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import os
from cmcrameri import cm
from utils import load_write
from utils import process_mesh
from utils import find_isotherms
from utils import plotting

lw = load_write.Load_Write()

class CMDA: # cmdline_args
    pass

def load_fields(path_list,data_fname):
    # get size of array
    fname = os.path.join(path_list[0],data_fname)
    fp = open(fname, "rb")
    vector = pkl.load(fp)
    fp.close()
    narr = np.zeros((np.shape(vector)[0],len(path_list)))
    for k in range(len(path_list)):
        fname = os.path.join(path_list[k],data_fname)
        fp = open(fname, "rb")
        narr[:,k] = pkl.load(fp)
        fp.close()
    return narr

args = CMDA()
parser = argparse.ArgumentParser()
parser.add_argument('--jobs_csv', type=str, default="jobs_log.csv", required=True, help="Name of csv file containing paths to input data.")
parser.add_argument('--data_fname', type=str, default='temperature.pkl', required=False, help="The name of the datafile to be loaded from each directory.")
parser.add_argument('--mesh_path', type=str, default='input', required=True, help="Path where mesh files are located.")
parser.add_argument('--profile_name', type=str, default=None, required=True, help="Which profile to use.")
parser.add_argument('--reorder', type=bool, default=True, required=False, help="Whether or not to reorder the fields.")
parser.add_argument('--output_path', type=str, default='./', required=False, help="Path where generated output will be written.")
parser.add_argument('--include', type=str, nargs='+', required=True, help="Which sampling methods to include.")
parser.add_argument('--pre_dir', type=str, default='./', required=False, help="Path to handle absolute vs relative data paths in logs.")
parser.parse_known_args(namespace=args)

found = os.path.exists(args.output_path)
if found:
    print('Output path', args.output_path, 'exists')
else:
    os.makedirs(args.output_path, exist_ok=True)
    print('Output path', args.output_path, 'created')

jobs_log = pd.read_csv(args.jobs_csv,comment='#')
for j in range(len(jobs_log)): # if necessary, join to get the absolute path
    jobs_log.loc[j,"path"] = os.path.join(args.pre_dir,jobs_log.loc[j,"path"])

for s in args.include:
    jobs_log = jobs_log[jobs_log.path.str.contains(s)]

# mesh info
X = lw.load(os.path.join(args.mesh_path,'X.pkl'))
Y = lw.load(os.path.join(args.mesh_path,'Y.pkl'))
TAGS = lw.load(os.path.join(args.mesh_path,'TAGS.pkl'))

# load plotting data
coords_x = lw.load(os.path.join(args.mesh_path,"coords_x.pkl"))
coords_y = lw.load(os.path.join(args.mesh_path,"coords_y.pkl"))
mesh_cells = lw.load(os.path.join(args.mesh_path,"mesh_cells.pkl"))

v4_vizfile_name = os.path.join(args.mesh_path,os.path.join("viz",args.profile_name+"_viz_v4_ascii.msh"))
meshfile_name = os.path.join(args.mesh_path,args.profile_name)
vizfile_name = os.path.join(args.mesh_path,os.path.join("viz",args.profile_name+"_viz"))
T_CG_order = 2

M = lw.load_transfer_matrix(os.path.join(args.mesh_path,"M.dat"))
pm = process_mesh.Process_Mesh(T_CG_order,meshfile_name,vizfile_name,v4_vizfile_name,args.mesh_path)

# create an instance of the Find_Isotherms class
iso = [423, 623, 723, 873]
find_iso = find_isotherms.Find_Isotherms(iso)
# create a file to write the iso_info to
iso_info_csv = os.path.join(args.output_path,"iso_info.csv")
print('Writing isotherm-slab interface intersection info to ',iso_info_csv)
fp = open(iso_info_csv,'w')
fp.write('Experiment,')
for k in range(len(iso)):
    fp.write("X,Y,T,D,")
fp.write('\n')
fp.close()

result_arr = []
for dir in jobs_log["path"]:
    if args.reorder == True:
        reorder_field = pm.reorder_data(os.path.join(dir,args.data_fname),M)
        lw.write(reorder_field,os.path.join(dir,"temperature_reordered.pkl"))

    # get T on the slab interface
    result = lw.load(os.path.join(dir, "temperature_reordered.pkl"))
    result_arr.append(result)
    T = result[TAGS]

    lw.write(T, os.path.join(dir,"T.pkl"))

    # print(T)
    # find iso info and write it
    iso_info = find_iso.locate_isotherm_intersection(X,Y,T,int(1e6),1e-6)
    print('iso_info', iso_info)

    D = find_iso.along_slab_distance(X,Y,T,iso_info)
    # print('iso_info',iso_info)
    print('D',D)
    f = open(iso_info_csv,'a')
    f.write(str(dir)); f.write(',')
    for k in range(len(iso_info)):
        f.write(str(iso_info[k][0])); f.write(',')
        f.write(str(iso_info[k][1])); f.write(',')
        f.write(str(iso_info[k][2])); f.write(',')
        f.write(str(D[k])); f.write(',')
    f.write('\n')
    f.close()

    title = r"$T_{FOM}$"
    png_name = os.path.join(dir,"temperature_C.png")
    # level_vals = (423,623,800,1000,1200,1400)
    # level_vals = (150,350,450,600,800,1000,1200)
    level_vals = (100,200,300,400,500,600,700,800,900,1000,1100,1200)
    # my_plotting_tool = plotting.Plotting(title,png_name,level_vals,iso_info)
    my_plotting_tool = plotting.Plotting(title,png_name,level_vals)
    tri_mesh = my_plotting_tool.plot_result(coords_x,coords_y,mesh_cells,result-273)

    # save the tri_mesh so it can be used elsewhere
    tri_mesh._cpp_triangulation = None
    lw.write(tri_mesh,os.path.join(dir,"tri_mesh.pkl"))

    font_size = 18
    fig = plt.figure(figsize=(20,12))
    ax1 = fig.add_subplot(211)
    ax1.set_aspect('equal')
    ax1.scatter(T, Y)
    ax1.tick_params(axis='both', which='major', labelsize=font_size)
    ax1.set_xlabel('T (K)', fontsize=font_size)
    ax1.set_ylabel('y (km)', fontsize=font_size)
    
    ax2 = fig.add_subplot(212)
    ax2.set_aspect('equal')
    ax2.scatter(T, X)
    ax2.tick_params(axis='both', which='major', labelsize=font_size)
    ax2.set_xlabel('T (K)', fontsize=font_size)
    ax2.set_ylabel('x (km)', fontsize=font_size)
    ax2.invert_yaxis()
    plt.savefig("T_vs_X_and_Y.png")

# 
II = np.argmin(np.abs(Y+210.0))
# T_slab_norm = np.sqrt( np.sum(T[0:II]**2)/ T[0:II].shape[0] )
T_slab_norm = np.sqrt( np.mean(T[0:II]**2) )
# print('norm of slab interface T to 210 km depth', np.linalg.norm(T[0:II]))
print('T_slab_norm (K)',T_slab_norm)
print('T_slab_norm (C)',T_slab_norm-273.0)

exit()

# isotherm variation plot
font_size = 26
alpha = 1.0
png_name = os.path.join(args.output_path,"isotherm_variation.pdf")
print('Plotting isotherm variation inset.')

color_arr = cm.lajolla(np.linspace(0, 1, 5))
c_arr_2 = cm.hawaii(np.linspace(0,1,10))

fig = plt.figure(figsize=(20,12))
ax1 = fig.add_subplot(111)
ax1.set_aspect('equal')
tri_mesh = tri.Triangulation(coords_x, coords_y, mesh_cells)
for k in range(len(result_arr)):
    ax1.tricontour(tri_mesh, result_arr[k], levels=[423], colors='lightskyblue', alpha=alpha)
for k in range(len(result_arr)):
    ax1.tricontour(tri_mesh, result_arr[k], levels=[623], colors=mplc.to_hex(color_arr[1,:]), alpha=alpha)
for k in range(len(result_arr)):
    ax1.tricontour(tri_mesh, result_arr[k], levels=[723], colors=mplc.to_hex(color_arr[2,:]), alpha=alpha)
for k in range(len(result_arr)):
    ax1.tricontour(tri_mesh, result_arr[k], levels=[873], colors=mplc.to_hex(c_arr_2[0,:]), alpha=alpha)
fig = plt.plot(X,Y,'k-',linewidth=4)
fig = plt.xticks(fontsize=font_size);
fig = plt.yticks(fontsize=font_size);
fig = plt.xlabel('x (km)', fontsize=font_size)
fig = plt.ylabel('y (km)', fontsize=font_size)
# fig = plt.title(title, fontsize=font_size)
# plt.minorticks_on()
# plt.grid(visible=True, which='both')
plt.savefig(png_name)

png_name = os.path.join(args.output_path,"isotherm_variation_inset.pdf")
plt.xlim(0.0,350.0)
plt.ylim(-160.0,0.0)
plt.savefig(png_name)

plt.close("all")
