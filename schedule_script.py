import os
from forward_model import Subduction
import sys
import argparse
# sys.path.append(os.pardir) # this needs to be fixed
from ParametricModelUtils.parametric_model_utils import parametric_model as pm
from ParametricModelUtils.parametric_model_utils import parametric_scheduler as ps
from ParametricModelUtils.parametric_model_utils import ExecutionStatus as status
from utils import create_samples

class CMDA: # cmdline_args
  pass

args = CMDA()
parser = argparse.ArgumentParser()
parser.add_argument('--profile_name', type=str, default="", required=True, help="Which slice to use")
parser.add_argument('--profile_dir', type=str, default="", required=True, help="Path to directory with mesh.")
parser.add_argument('--pre_dir', type=str, default="set", required=False, help="Output directory, for example 'training_1' or 'testing_1'.")
parser.add_argument('--sample_method', type=str, default="halton", required=True, help="Which method to use to sample parameter space.")
parser.add_argument('--n1', type=int, required=True, help="Number of samples to draw initially to advance the sequence.")
parser.add_argument('--n2', type=int, required=True, help="Number of forward models to run.")
parser.add_argument('--seed', type=int, required=True, help="Seed for the sequence of samples drawn from parameter space.")
parser.add_argument('--jobs_csv', type=str, default="jobs_log.csv", required=False, help="Name of csv file to tracks jobs run.")
parser.parse_known_args(namespace=args)

mesh_dir = os.path.join(os.getcwd(),args.profile_dir)
meshfile_name = os.path.join(mesh_dir,args.profile_name)
v4_vizfile_name = os.path.join(mesh_dir,os.path.join("viz",args.profile_name+"_viz_v4_ascii.msh"))
vizfile_name = os.path.join(mesh_dir,os.path.join("viz",args.profile_name+"_viz"))

dfield_fname = os.path.join(mesh_dir,'distance_field.pkl')
slab_d_fname = os.path.join(mesh_dir,'slab_d_field.pkl')
indices_fname = os.path.join(mesh_dir,'distance_field_indices.pkl')

input_dict = {'mesh_dir':mesh_dir,'v4_vizfile_name':v4_vizfile_name,'meshfile_name':meshfile_name,'vizfile_name':vizfile_name,\
    'dfield_fname':dfield_fname,'slab_d_fname':slab_d_fname,'indices_fname':indices_fname}

M = Subduction("input_param.csv",input_dict)

sch = ps.ParametricScheduler(args.pre_dir)
sch.output_path_prefix = args.sample_method
sch.set_model(M)

l_bounds = [M.P.param_bounds[k][0] for k in M.P.param_bounds.keys()]
u_bounds = [M.P.param_bounds[k][1] for k in M.P.param_bounds.keys()]
cpc = create_samples.CreateSamples(args.seed,args.sample_method,M.P.n_inputs,l_bounds,u_bounds)
discard = cpc.generate_samples(args.n1)
p_vals = cpc.generate_samples(args.n2)
cpc.write_csv(args.n2,args.pre_dir)

run, ignore = sch.batched_schedule(p_vals, max_jobs=94, wait_time=5.0)
nscans = sch.wait_all(1.0)

print('run', run)
print('ignore', ignore)

sch.cache_generate_log(args.jobs_csv)
