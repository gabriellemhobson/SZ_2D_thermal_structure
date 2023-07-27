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
# required arguments
parser.add_argument('--profile_name', type=str, default="", required=True, help="Which slice to use")
parser.add_argument('--mesh_dir', type=str, default="", required=True, help="Path to directory with mesh.")
parser.add_argument('--output_path', type=str, required=True, help="Output directory, for example 'training_1' or 'testing_1'.")
parser.add_argument('--sample_method', type=str, default="halton", required=True, help="Which method to use to sample parameter space.")
parser.add_argument('--n1', type=int, required=True, help="Number of samples to draw initially to advance the sequence.")
parser.add_argument('--n2', type=int, required=True, help="Number of forward models to run.")
parser.add_argument('--seed', type=int, required=True, help="Seed for the sequence of samples drawn from parameter space.")
parser.add_argument('--jobs_csv', type=str, required=True, help="Name of csv file to tracks jobs run.")
parser.add_argument('--viscosity_type', type=str, required=True, help="Type of viscosity flow law. Options are 'isoviscous', 'diffcreep', 'disccreep', or 'mixed'.")
# optional arguments
parser.add_argument('--solver', type=str, default="ss", required=False, help="Type of solver, must be 'ss' or 'time_dep'. ")
parser.add_argument('--tol', type=float, default=1e-5, required=False, help="Residuals for pde solutions in forward model solve must be below this tolerance.")
parser.add_argument('--n_picard_it', type=int, default=10, required=False, help="Maximum number of picard iterations.")
parser.add_argument('--n_iters', type=int, default=10, required=False, help="Maximum number of iterations.")
parser.add_argument('--diff_tol', type=float, default=1e-1, required=False, help="Tolerance for reaching a converged solution.")
parser.add_argument('--T_CG_order', type=int, default=2, required=False, help="Order of CG elements for temperature.")
parser.parse_known_args(namespace=args)

mesh_dir = os.path.join(os.getcwd(),args.mesh_dir)
meshfile_name = os.path.join(mesh_dir,args.profile_name)
v4_vizfile_name = os.path.join(mesh_dir,os.path.join("viz",args.profile_name+"_viz_v4_ascii.msh"))
vizfile_name = os.path.join(mesh_dir,os.path.join("viz",args.profile_name+"_viz"))

dfield_fname = os.path.join(mesh_dir,'distance_field.pkl')
slab_d_fname = os.path.join(mesh_dir,'slab_d_field.pkl')
indices_fname = os.path.join(mesh_dir,'distance_field_indices.pkl')

input_dict = {'mesh_dir':mesh_dir,\
              'v4_vizfile_name':v4_vizfile_name,\
              'meshfile_name':meshfile_name,\
              'vizfile_name':vizfile_name,\
              'dfield_fname':dfield_fname,\
              'slab_d_fname':slab_d_fname,\
              'indices_fname':indices_fname,\
              'viscosity_type':args.viscosity_type,\
              'solver':args.solver,\
              'tol':args.tol,\
              'n_picard_it':args.n_picard_it,\
              'n_iters':args.n_iters,\
              'diff_tol':args.diff_tol,\
              'T_CG_order':args.T_CG_order}

M = Subduction("input_param.csv",input_dict)

sch = ps.ParametricScheduler(args.output_path)
sch.output_path_prefix = args.sample_method
sch.set_model(M)

l_bounds = [M.P.param_bounds[k][0] for k in M.P.param_bounds.keys()]
u_bounds = [M.P.param_bounds[k][1] for k in M.P.param_bounds.keys()]
cpc = create_samples.CreateSamples(args.seed,args.sample_method,M.P.n_inputs,l_bounds,u_bounds)
discard = cpc.generate_samples(args.n1)
p_vals = cpc.generate_samples(args.n2)
cpc.write_csv(args.n2,args.output_path)

run, ignore = sch.batched_schedule(p_vals, max_jobs=4, wait_time=5.0)
nscans = sch.wait_all(1.0)

print('run', run)
print('ignore', ignore)

sch.cache_generate_log(args.jobs_csv)
