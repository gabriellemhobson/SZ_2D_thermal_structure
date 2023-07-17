import os
import numpy as np
import sys
sys.path.append(os.pardir)
from ParametricModelUtils.parametric_model_utils import parametric_model as pm
from ParametricModelUtils.parametric_model_utils import parametric_scheduler as ps
from ParametricModelUtils.parametric_model_utils import ExecutionStatus as status

class Subduction(pm.ParametricModel):
    def __init__(self, input_def, input_dict, **kwargs):
        super().__init__(input_def, **kwargs)
        self.input_dict = input_dict

    def create_param_combos(self,method,n_runs=1):
        import itertools

        range_vals = []
        if method=='linspace':
            for k in range((self.P.n_inputs)):
                range_vals.append(np.linspace(self.P.param_bounds[self.P.param_names[k]][0],\
                    self.P.param_bounds[self.P.param_names[k]][1],n_runs))
        else:
            raise ValueError('This method is not supported. Only linspace is currently supported.')

        # loop through and create cartesian product list
        cart = [list(i) for i in itertools.product(*range_vals)]
        cart_nd = np.zeros((len(cart),self.P.n_inputs))
        for k in range(len(cart)):
            cart_nd[k,:] = cart[k]
        return cart_nd

    def evaluate(self, params): # required
        import subprocess

        # base_dir = self.output_path
        identifier = self.P.get_identifier(params)
        # base_dir = os.path.join(self.output_path, identifier)
        base_dir = self.output_path
        print('Subduction.evaluate() base_dir', base_dir)
        # log the params going in
        p = self.P._convert(params)
        vals = list(p.values())
        viscosity_type = 'mixed'
        tol = 1e-5
        n_picard_it = 1
        n_iters = 40
        diff_tol = 1e-1
        T_CG_order = 2
        print('p',p)
        fp = open(os.path.join(base_dir, 'forward_subduction_detached.py'), "w")
        fp.write('import sys \n')
        fp.write('import os \n')
        fp.write('from collections import OrderedDict \n')
        fp.write('sys.path.append("'+os.pardir+'") \n')
        fp.write('from forward_model_solvers import pde_solver_time_dep as pde_solver \n')
        fp.write('import define_parameters \n')
        fp.write('solver = pde_solver.PDE_Solver("' + self.input_dict['meshfile_name']+'","'+self.input_dict['vizfile_name']+'","'+self.input_dict['dfield_fname']+'","'+self.input_dict['slab_d_fname']+'","'+self.input_dict['indices_fname']+'") \n')
        fp.write('solver.param = define_parameters.Parameters('+str(T_CG_order)+',"'+str(viscosity_type)+'",'+str(tol)+','+str(n_picard_it)+','+str(n_iters)+','+str(diff_tol)+') \n')
        fp.write('solver.param.set_param(**'+ str(p) + ') \n')
        fp.write('solver.output_dir = "' + str(base_dir) + '" \n')
        fp.write('print("solver.output_dir",solver.output_dir) \n')
        fp.write('print("os.getcwd()",os.getcwd()) \n')
        fp.write('fs = open(os.path.join("' + base_dir + '","fenics_start.txt"),"w") \n')
        fp.write('fs.write("Starting the forward model solve.") \n')
        fp.write('fs.close() \n')
        fp.write('solver.run_solver() \n')
        fp.write('fe = open(os.path.join("' + base_dir + '","fenics_end.txt"),"w") \n')
        fp.write('fe.write("Finished the forward model solve.") \n')
        fp.write('fe.close() \n')
        fp.write('solver.param = None')
        fp.close()

        p = subprocess.Popen( ["python3", os.path.join(base_dir, 'forward_subduction_detached.py')],stdin=None, stdout=None, stderr=None)

    def exec_status(self): # required
        base_dir = self.output_path
        found_start = os.path.exists( os.path.join(base_dir, "fenics_start.txt"))
        found_executable = os.path.exists( os.path.join(base_dir, "forward_subduction_detached.py"))
        found_data = os.path.exists( os.path.join(base_dir, "temperature.pkl"))
        found_end = os.path.exists( os.path.join(base_dir, "fenics_end.txt"))
        if found_data:
            return pm.ExecutionStatus.SUCCESS
        elif not found_end:
            return pm.ExecutionStatus.UNDEFINED
        elif not found_start and not found_executable:
            return pm.ExecutionStatus.ERROR
        else:
            return pm.ExecutionStatus.ERROR
            # return pm.ExecutionStatus.UNDEFINED
