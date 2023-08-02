# Script to generate a mesh for a specific subduction zone

import generate_mesh
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import json as json
import subprocess

def slice_generic(profile_fname, fname_slab, data_path, output_path, slab_id, args):

    # Input specs which seemingly don't relate to the geometry - remove from Generate_Mesh?
    start_line = 2878 # These appear to be set as attributes then over-ridden
    end_line = 2931

    fname_trench = os.path.join(data_path, "PlateBoundaries/", "PB2002_boundaries.dig.txt")

    beginning_strings, geo_info, constrain = generate_input_options(args)
    print(beginning_strings)

    # record options
    record = dict()
    record["slab2_file"] = fname_slab
    record["trench_file"] = fname_trench
    record["beginning_strings"] = beginning_strings
    record["geo_info"] = geo_info
    record["constrain"] = constrain
    record["args"] = args.__dict__
    with open(os.path.join(output_path, "config.json"), "w") as fp:
      json.dump(record, fp, indent=2)

    df = pd.read_csv(profile_fname,comment="#")

    labels = list(df["Label"])
    start_points_arr = [[df["lon_start"][k],df["lat_start"][k]] for k in range(len(df))]
    end_points_arr = [[df["lon_end"][k],df["lat_end"][k]] for k in range(len(df))]

    gm = generate_mesh.Generate_Mesh(fname_slab,fname_trench,constrain,start_line,end_line)

    profiles = []
    profiles_lat_lon_arr = []
    for k in range(len(labels)):
        label = labels[k] # If we changed this to k and then `Labal` doesn't need to be defined in the csv file.
        start_point_lon_lat = start_points_arr[k]
        end_point_lon_lat = end_points_arr[k]

        output_subfolder = os.path.join(output_path,slab_id + '_profile_{}'.format(label))
        os.makedirs(output_subfolder, exist_ok=True)

        geo_filename = slab_id + '_profile_{}.geo'.format(label)
        
        profile, profile_lat_lon = gm.run_generate_mesh(geo_filename,geo_info,start_point_lon_lat,end_point_lon_lat,plot_verbose=False,write_msh=args.write_msh)
        
        fname_profile = slab_id + '_profile_{}.xy'.format(label) # Avoid using `z` as last character otherwise numpy will zip the file
        fname_profile = os.path.join(output_path, fname_profile)
        np.savetxt(fname_profile, profile)
        
        profiles.append(profile)
        profiles_lat_lon_arr.append(profile_lat_lon)

        if args.write_msh:
          command = ["gmsh", geo_filename, "-parse_and_exit"]
          CP = subprocess.run(command)
          if CP.returncode != 0:
            raise RuntimeError('Error running command',str(command), ', subprocess returncode is ',CP.returncode)

          # move files to appropriate subdirectories
          os.rename(geo_filename, os.path.join(output_subfolder, geo_filename))
          os.rename(geo_filename[:-4]+'.msh', os.path.join(output_subfolder, geo_filename[:-4]+'.msh'))
          viz_subfolder = os.path.join(output_subfolder,"viz")
          os.makedirs(viz_subfolder,exist_ok=True)
          os.rename(geo_filename[:-4]+'_viz.msh', os.path.join(viz_subfolder, geo_filename[:-4]+'_viz.msh'))
          os.rename(geo_filename[:-4]+'_viz_v4_ascii.msh', os.path.join(viz_subfolder, geo_filename[:-4]+'_viz_v4_ascii.msh'))

    fig_name = slab_id + '_slices.pdf'
    fig_name = os.path.join(output_path, fig_name)
    label_offset_dict = {"cascadia":[-0.25,0],"hikurangi":[0.25,-1.0],"nankai":[0,-0.5]}
    if slab_id in label_offset_dict:
      label_offset = label_offset_dict[slab_id]
    else:
      label_offset = [0,0]
    print('label_offset',label_offset)
    gm.plotting_slices_map(start_points_arr,end_points_arr,profiles_lat_lon_arr,labels,label_offset,fig_name)

    fig_name = slab_id + '_profiles.pdf'
    fig_name = os.path.join(output_path, fig_name)
    buffer = 50
    # xlim_max = np.max(np.array([np.max(p[:,0]) for p in profiles])) + buffer
    # ylim_min = np.min(np.array([np.min(p[:,1]) for p in profiles])) - buffer
    xlim_max = 750.0
    ylim_min = -450.0
    fig = plt.figure(figsize=(12,12))
    font_size=12
    for k in range(len(labels)):
        ax = fig.add_subplot(4,3,k+1)
        ax.set_aspect('equal')
        ax.plot(profiles[k][:,0],profiles[k][:,1],c='k')
        ax.set_xlabel('x (along-slab km)', fontsize=font_size)
        ax.set_ylabel('y (km)', fontsize=font_size)
        ax.set_xlim(0,xlim_max)
        ax.set_ylim(ylim_min,0)
        ax.text(xlim_max-100,-100,str(labels[k]),fontsize=font_size)
        plt.minorticks_on()
        plt.grid(visible=True, which='both')
    plt.savefig(fig_name)
    plt.show()

    fig_name = slab_id + '_superimposed_profiles.pdf'
    fig_name = os.path.join(output_path, fig_name)
    # xlim_max = np.max(np.array([np.max(p[:,0]) for p in profiles])) + buffer
    # ylim_min = np.min(np.array([np.min(p[:,1]) for p in profiles])) - buffer
    xlim_max = 700.0
    ylim_min = -450.0
    fig = plt.figure(figsize=(10.6,6.75))
    font_size=18
    ax = fig.add_subplot(1,1,1)
    # ax.set_aspect('equal')
    for k in range(len(labels)):
      ax.plot(profiles[k][:,0],profiles[k][:,1],label=labels[k],lw=3)
    ax.set_xlabel('x (km)', fontsize=font_size)
    ax.set_ylabel('y (km)', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.set_xlim(0,xlim_max)
    ax.set_ylim(ylim_min,0)
    # ax.text(xlim_max-100,-100,str(labels[k]),fontsize=font_size)
    plt.legend(fontsize=font_size)
    # plt.minorticks_on()
    plt.grid(visible=True, which='major')
    plt.title(slab_id + ' profiles',fontsize=font_size)
    plt.savefig(fig_name)
    plt.show()


class CMDA: # cmdline_args
  pass


def parse_geo_info(p):
  p.add_argument('--corner_depth', type=float, default=-38, required=False, help="--- (km)")


def parse_constraints(p):
  p.add_argument('--constrain_TF', type=bool, default=False, required=False)
  p.add_argument('--cut_at_lat', type=float, required=False, help="Latitutde value to define crop")
  p.add_argument('--less_or_greater', type=str, required=False, help="Crop action")


def parse_beginning_strings(p):
  p.add_argument('--slab_thickness',  type=float, default = 50.0, required=False, help="Thickness of the slab (km)")
  p.add_argument('--overplate_notch', type=float, default = 0.5, required=False, help="---- (km)")
  p.add_argument('--domain_width_x',  type=float, default = 660.0, required=False, help="Width of the domain (km)")
  p.add_argument('--extension_x',     type=float, default = 50.0, required=False, help="--- (km)")
  p.add_argument('--z_in_out',        type=float, default = 150.0, required=False, help="Depth (km) of inflow/outflow transition on wedge side")
  p.add_argument('--h_fine',          type=float, default = 0.5, required=False, help="Default value for fine resolution (km)")
  p.add_argument('--h_med',           type=float, default = 8.0, required=False, help="Default value for medium resolution (km)")

def generate_input_options(a):
  
  beginning_strings = ['SetFactory("OpenCASCADE");']
  
  beginning_strings.append(
      'slab_thickness = DefineNumber[ {}, Name "Parameters/slab_thickness" ];'.format(a.slab_thickness) )
  beginning_strings.append(
      'overplate_notch = DefineNumber[ {}, Name "Parameters/overplate_notch" ];'.format(a.overplate_notch) )
  beginning_strings.append(
      'domain_width_x = DefineNumber[ {}, Name "Parameters/domain_width_x" ];'.format(a.domain_width_x) )
  beginning_strings.append(
      'extension_x = DefineNumber[ {}, Name "Parameters/extension_x" ];'.format(a.extension_x) )
  beginning_strings.append(
      'z_in_out = DefineNumber[ {}, Name "Parameters/z_in_out" ];'.format(a.z_in_out) )
  beginning_strings.append(
      'h_fine = DefineNumber[ {}, Name "Parameters/h_fine" ];'.format(a.h_fine) )
  beginning_strings.append(
      'h_med = DefineNumber[ {}, Name "Parameters/h_med" ];'.format(a.h_med) )
   
  geo_info = dict()
  geo_info['beginning_strings'] = beginning_strings
  geo_info['corner_depth'] = a.corner_depth

  constrain = dict()
  constrain['constrain_TF'] = a.constrain_TF
  if a.constrain_TF:
    constrain['cut_at_lat'] = a.cut_at_lat
    constrain['less_or_greater'] = a.less_or_greater

  return beginning_strings, geo_info, constrain


def determine_slab_data(profiles, data_path):
  df = pd.read_csv(profiles, comment="#")
  lat_s = np.array(df["lat_start"])
  lat_e = np.array(df["lat_end"])
  lon_s = np.array(df["lon_start"])
  lon_e = np.array(df["lon_end"])
  target_box_la, target_box_lo = [None, None], [None, None]
  target_box_lo[0] = np.min(lon_s)
  target_box_lo[1] = np.max(lon_e)
  target_box_la[0] = np.min(lat_s)
  target_box_la[1] = np.max(lat_e)
  print('target', 'lat', target_box_la, 'lon', target_box_lo)
  
  spath = os.path.join(data_path, "Slab2/")
  
  flist = [
           "alu_slab2_dep_02.23.18.xyz",	"hin_slab2_dep_02.24.18.xyz",	"png_slab2_dep_02.26.18.xyz",
           "cal_slab2_dep_02.24.18.xyz",	"izu_slab2_dep_02.24.18.xyz",	"puy_slab2_dep_02.26.18.xyz",
           "cam_slab2_dep_02.24.18.xyz",	"ker_slab2_dep_02.24.18.xyz",	"ryu_slab2_dep_02.26.18.xyz",
           "car_slab2_dep_02.24.18.xyz",	"kur_slab2_dep_02.24.18.xyz",	"sam_slab2_dep_02.23.18.xyz",
           "cas_slab2_dep_02.24.18.xyz",	"mak_slab2_dep_02.24.18.xyz",	"sco_slab2_dep_02.23.18.xyz",
           "cot_slab2_dep_02.24.18.xyz",	"man_slab2_dep_02.24.18.xyz",	"sol_slab2_dep_02.23.18.xyz",
           "hal_slab2_dep_02.23.18.xyz",	"mue_slab2_dep_02.24.18.xyz",	"sul_slab2_dep_02.23.18.xyz",
           "hel_slab2_dep_02.24.18.xyz",	"pam_slab2_dep_02.26.18.xyz", "sum_slab2_dep_02.23.18.xyz",
           "him_slab2_dep_02.24.18.xyz",	"phi_slab2_dep_02.26.18.xyz", "van_slab2_dep_02.23.18.xyz",
          ]

  matches = 0
  matched_slab = list()
  for sfile in flist:
    fname = os.path.join(spath, sfile)
    a = np.loadtxt(fname, delimiter=',')
    box_la, box_lo = [None, None], [None, None]
    box_lo[0] = np.min(a[:, 0])
    box_lo[1] = np.max(a[:, 0])
    box_la[0] = np.min(a[:, 1])
    box_la[1] = np.max(a[:, 1])
    print(sfile, '--> lat', box_la, 'lon', box_lo)

    if target_box_lo[0] >= box_lo[0] and target_box_lo[1] <= box_lo[1]:
      if target_box_la[0] >= box_la[0] and target_box_la[1] <= box_la[1]:
        matches += 1
        matched_slab.append(sfile)

  if matches == 0:
    print("The bounding box for the profiles are not contained with any slab defined within the Slab2 data given by:\n", flist)
    raise RuntimeError("No match between profile bounding box and slab data.")
  elif matches > 1:
    print("The bounding box for the profiles are contained in more than one slab defined within the Slab2 data.\n")
    print("The following slabs contain the profiles:\n", matched_slab)
    raise RuntimeError("Profile bounding box contained in multiple slab data files.")
  else:
    print("Profiles bounding box contained within:", matched_slab[0])

  return os.path.join(spath, matched_slab[0])


if __name__ == '__main__':
    args = CMDA()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data/", required=False, help="Path where Slab2/ is located")
    parser.add_argument('--profile_csv', type=str, required=True, help="CSV file defining the profiles")
    parser.add_argument('--slab_name', type=str, required=True, help="Textual identifier you want to associate with the output generated")
    parser.add_argument('--output_path', type=str, default='./', required=False, help="Path where generated output will be written")
    # parser.add_argument('--write_msh', type=bool, required=True, help="Bool for whether or not to write .msh file.")
    parser.add_argument('--write_msh', action='store_true')

    # These options may also be readily parsed from an input file if
    # the command line argument approach becomes unmanageable.
    parse_geo_info(parser)
    parse_constraints(parser)
    parse_beginning_strings(parser)
    
    parser.parse_known_args(namespace=args)
    
    fname_slab = determine_slab_data(args.profile_csv, args.data_path)
    # fname_slab = "data/Slab2/cas_slab2_dep_02.24.18.xyz"
    # fname_slab = "data/Slab2/ker_slab2_dep_02.24.18.xyz"
    # fname_slab = "data/Slab2/ryu_slab2_dep_02.26.18.xyz"
    
    os.makedirs(args.output_path, exist_ok=True)

    slice_generic(args.profile_csv, fname_slab, args.data_path, args.output_path, args.slab_name, args)

