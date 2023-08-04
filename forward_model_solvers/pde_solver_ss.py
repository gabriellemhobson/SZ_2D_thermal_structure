from dolfin import * 
import numpy as np
import pickle as pkl
import os
import ufl

# Test for PETSc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_krylov_solver_preconditioner("amg"):
    info("Sorry, this code is only available when DOLFIN is compiled with AMG "
         "preconditioner, Hypre or ML.")
    exit()

# Set backend to PETSC
parameters["linear_algebra_backend"] = "PETSc"

# Check that the correct PETSc version is being used
import petsc4py
ver = petsc4py.__version__.split('.')
major_minor, patch = ver[0] + '.' + ver[1], ver[2]
if (major_minor == "3.12"): # docker version
    # print('Using PETSc version', str(major_minor),'.')
    pass
elif (major_minor == "3.14") or (major_minor == "3.17"): # conda env version
    # print('Using PETSc version', str(major_minor),'.')
    PETScOptions.set("log_view", "")
    PETScOptions.set("info","true")
    PETScOptions.set("pc_type","lu")
    PETScOptions.set("pc_factor_mat_solver_type","umfpack")
    PETScOptions.set("pc_factor_mat_ordering_type","external")
    PETScOptions.set("ksp_type","preonly")
else:
    print("This code is not compatible with versions of PETSc other than 3.12, 3.14, and 3.17. Exiting.")
    exit()

class PDE_Solver():
    def __init__(self,meshfile_name,vizfile_name,dfield_fname,slab_d_fname,indices_fname,**kwargs):
        self.meshfile_name = meshfile_name
        self.vizfile_name = vizfile_name
        self.dfield_fname = dfield_fname
        self.slab_d_fname = slab_d_fname
        self.indices_fname = indices_fname

        if 'output_dir' in kwargs:
            self.output_dir = kwargs['output_dir']
        else:
            self.output_dir = ""
        if 'param' in kwargs:
            self.param = kwargs['param']
        else:
            self.param = None

        # load meshes and associated 
        self.mesh = Mesh("%s.xml" %(self.meshfile_name))
        self.mesh_viz = Mesh("%s.xml" %(self.vizfile_name))
        self.subdomains = MeshFunction("size_t", self.mesh,"%s_physical_region.xml" %(self.meshfile_name))
        self.boundaries = MeshFunction("size_t", self.mesh,"%s_facet_region.xml" %(self.meshfile_name))
        self.boundaries_viz = MeshFunction("size_t", self.mesh_viz,"%s_facet_region.xml" %(self.vizfile_name))

        self.dx = Measure("dx", domain=self.mesh, subdomain_data=self.subdomains)
        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)
        self.dS = Measure("dS", domain=self.mesh, subdomain_data=self.boundaries) 

        # slab_sub = SubMesh(self.mesh, self.subdomains, 17)
        # wedge_sub = SubMesh(self.mesh, self.subdomains, 18)
        # overplate_sub = SubMesh(self.mesh, self.subdomains, 19)
        
        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(28,16))
        # plot(slab_sub,color='tab:blue')
        # plot(wedge_sub,color='tab:orange')
        # plot(overplate_sub,color='tab:green')
        # plt.xlabel("x (km)",fontsize=36)
        # plt.xticks(fontsize=26)
        # plt.ylabel("y (km)",fontsize=36)
        # plt.yticks(fontsize=26)
        # plt.savefig(self.output_dir + "mesh_colored.png")

        # boundaries have numbers associated, from gmsh
        self.surface = 20
        self.overplate_right = 21
        self.overplate_base = 22
        self.overplate_left = 23
        self.slab_left = 24
        self.slab_overplate_int = 25
        self.slab_wedge_int = 27
        self.slab_right = 28
        self.slab_base = 29
        self.wedge_base = 30
        self.outflow_wedge = 31
        self.inflow_wedge = 32

    def write(self,vector,filename):
        file = open(filename, "wb")
        pkl.dump(vector, file)
        file.close()

    def load(self,filename):
        file = open(filename, "rb")
        vector = pkl.load(file)
        file.close()
        return vector

    def project_gmh(self, v, V, bcs=None, mesh=None,function=None):
        if (major_minor == "3.12"):
            function = project(v,V)
        elif (major_minor == "3.14") or (major_minor == "3.17"):
            import ufl
            __all__ = ['project']
            # Ensure we have a mesh and attach to measure
            if mesh is None:
                mesh = V.mesh()
            dx = ufl.dx(mesh)

            # Define variational problem for projection
            w = TestFunction(V)
            Pv = TrialFunction(V)
            a = ufl.inner(w, Pv) * dx
            L = ufl.inner(w, v) * dx
            # Solve linear system for projection
            if function is None:
                function = Function(V)
            A = assemble(a)
            b = assemble(L)
            if bcs is not None:
                for bc in bcs:
                    bc.apply(A)
                    bc.apply(b)
            solver = PETScKrylovSolver()
            # solver.get_options_prefix()
            solver.set_operator(A)
            solver.set_from_options()
            # print('Starting solve in project_gmh()')
            solver.solve(function.vector(),b)
            # print('Finishing solve in project_gmh()')
        return function

    def solve_slab_flow(self,eta_slab,eta_wedge):
        '''
        Solves for flow in the slab. Masks out the other subdomains 
        but does not further modify them, or adjust the boundary of the slab. 
        '''
        # define function space for velocity, pressure
        P_u = VectorElement("CG", self.mesh.ufl_cell(), 2) # P2
        P_p = FiniteElement("CG", self.mesh.ufl_cell(), 1) # P1
        self.W_u_p = FunctionSpace(self.mesh, P_u * P_p)
        (self.v_1, self.v_2) = TestFunctions(self.W_u_p)
        (self.phi_1, self.phi_2) = TrialFunctions(self.W_u_p)

        U = Function(self.W_u_p) # used for iteration, residual
        
        beta = Constant(20)
        h = MaxCellEdgeLength(self.mesh) # CellSize(mesh)
        n = FacetNormal(self.mesh)

        def symgrad(uu):
            return 0.5*(grad(uu) + (grad(uu)).T)

        def sig(uu,pp):
            return 2.0 * eta_slab * symgrad(uu) - pp * Identity(2)

        def nsn(uu,pp,nn):
            sn = dot(sig(uu,pp), nn)
            return dot(nn, sn)

        self.f = Constant((0, 0))

        # slab 
        a = (inner(2*eta_slab*sym(grad(self.phi_1)), grad(self.v_1)) - div(self.v_1)*self.phi_2 - self.v_2*div(self.phi_1))*self.dx(17)
        L = inner(self.f, self.v_1)*self.dx(17)
        
        # mask out the wedge and overlying plate domains 
        a += 1e-8 * dot(self.v_1, self.phi_1) * self.dx(18) +  1e-8 * dot(self.v_1, self.phi_1) * self.dx(19)
        a += 1e-8 * self.v_2 * self.phi_2 * self.dx(18) +  1e-8 * self.v_2 * self.phi_2 * self.dx(19)
        
        sgn = "-"

        sigma_n_up = dot(sig(self.phi_1(sgn),self.phi_2(sgn)), n(sgn)) # traction
        sigma_n_vq = dot(sig(self.v_1(sgn),self.v_2(sgn)), n(sgn)) # traction

        a_nitsche = (
                    - inner(dot(sigma_n_up, n(sgn)),  dot(self.v_1(sgn), n(sgn)))
                    - inner(dot(sigma_n_vq, n(sgn)),  dot(self.phi_1(sgn), n(sgn)))
                    + beta/h(sgn)*inner(dot(self.phi_1(sgn), n(sgn)), dot(self.v_1(sgn), n(sgn))) )

        # upper boundary
        # a += a_nitsche * self.dS(25) + a_nitsche * self.dS(26) + a_nitsche * self.dS(27) 
        a += a_nitsche * self.dS(25) + a_nitsche * self.dS(27) 

        a_nitsche_exterior_boundary = (
                    - inner( nsn(self.phi_1,self.phi_2,n),  dot(self.v_1, n))
                    - inner( nsn(self.v_1,self.v_2,n),  dot(self.phi_1, n))
                    + beta/h*(dot(self.phi_1, n) * dot(self.v_1, n)) )

        # lower boundary
        a += a_nitsche_exterior_boundary * self.ds(29) # slab base

        # inflow
        a += a_nitsche_exterior_boundary * self.ds(24)
        g_dot_n = Expression("-vel",vel=self.param.slab_vel,degree=1) 
        L += beta/h*(g_dot_n * dot(self.v_1, n)) * self.ds(24)
        L += - ( nsn(self.v_1,self.v_2,n) *  g_dot_n) * self.ds(24)

        # outflow
        a += a_nitsche_exterior_boundary * self.ds(28)
        g_dot_n = Expression("vel",vel=self.param.slab_vel,degree=1)
        L += beta/h*(g_dot_n * dot(self.v_1, n)) * self.ds(28)
        L += - ( nsn(self.v_1,self.v_2,n) *  g_dot_n) * self.ds(28)

        if (major_minor == "3.12"):
            # print('Starting slab Stokes solve')
            solve(a == L, U)
            # print('Finishing slab Stokes solve')
        elif (major_minor == "3.14") or (major_minor == "3.17"):
            A = assemble(a)
            b = assemble(L)
            solver = PETScKrylovSolver()
            # solver.get_options_prefix()
            solver.set_operator(A)
            solver.set_from_options()

            # print('Starting slab Stokes solve')
            solver.solve(U.vector(),b)
            # print('Finishing slab Stokes solve')
        u_n, p = U.split()
        return u_n

    def apply_pc_and_trans(self,u_input,slab_d_field):
        '''
        Starting from the slab flow solution, apply partial coupling and the transition
        along the slab-wedge interface. Set and collect the wedge boundary conditions, 
        since we only need to do this once.  
        '''
        u_n,p_n_1 = Function(self.W_u_p).split()
        u_n.vector()[:] = u_input.vector()[:]

        # manually set slab-overplate interface to zero
        slab_overplate_bc = DirichletBC(self.W_u_p.sub(0), Expression(("0.0", "0.0"), degree=1), self.boundaries, self.slab_overplate_int)
        slab_overplate_bc.apply(u_n.vector())

        # iterate through cells and set overlying plate cells dofs to have zero velocity exactly
        self.dm_W_u_p = self.W_u_p.dofmap()
        for cell in cells(self.mesh):
            subdomain_index = self.subdomains.array()[cell.index()]
            cell_dofs = self.dm_W_u_p.cell_dofs(cell.index())
            if subdomain_index == 19:
                for k in cell_dofs:
                    u_n.vector()[k] = Constant(0.0)

        slab_x_coords = []
        slab_y_coords = []
        for facet in facets(self.mesh):
            if (self.boundaries.array()[facet.index()] == self.slab_wedge_int) \
                or (self.boundaries.array()[facet.index()] == self.slab_overplate_int):
                for v in vertices(facet):
                    slab_x_coords.append(v.point().x())
                    slab_y_coords.append(v.point().y())

        ddc_I = np.argmin(np.abs(np.array(slab_y_coords)+self.param.ddc))
        ddc_along_slab = np.array([0.0])
        slab_d_field.eval(ddc_along_slab, np.array([slab_x_coords[ddc_I], slab_y_coords[ddc_I]]))
        
        # apply transition
        exp_pc_trans = Expression(("0.5*u_x*((deg_pc+1) + (1-deg_pc)*tanh( (d-c)/(L/4) ))", \
            "0.5*u_y*((deg_pc+1) + (1-deg_pc)*tanh( (d-c)/(L/4) ))"), \
            degree=2, u_x=u_n.sub(0), u_y=u_n.sub(1), deg_pc=self.param.deg_pc, \
            d = slab_d_field, c=ddc_along_slab[0], L=self.param.L_trans)
        pc_trans_bc = DirichletBC(self.W_u_p.sub(0), exp_pc_trans, self.boundaries, self.slab_wedge_int)
        pc_trans_bc.apply(u_n.vector())

        # set velocity along wedge base to be same as slab at corner
        slab_corner = np.array([np.max(slab_x_coords), np.min(slab_y_coords)],dtype=float)
        slab_corner_vel = np.array([0.0,0.0])
        u_n.eval(slab_corner_vel, slab_corner)

        # use interface values as DirichletBCs for wedge problem
        interface_exp = Expression(("val_x", "val_y"), degree=1, val_x=u_n.sub(0),val_y=u_n.sub(1))
        collect_bc = [DirichletBC(self.W_u_p.sub(0), interface_exp, self.boundaries, self.slab_wedge_int), \
            DirichletBC(self.W_u_p.sub(0), Expression(("0.0", "0.0"), degree=1), self.boundaries, self.overplate_base), \
            DirichletBC(self.W_u_p.sub(0), slab_corner_vel, self.boundaries, self.wedge_base)]

        for bc in collect_bc:
            bc.apply(u_n.vector())

        return u_n, collect_bc

    def compute_jump(self,u_n_slab,u_n,I_Field):
        '''
        Compute the difference between the modified and original slab interface boundaries, 
        to represent the jump across the interface. 
        '''
        diff,p_diff = Function(self.W_u_p).split()
        diff.vector()[:] = u_n.vector() - u_n_slab.vector()

        P = FiniteElement("CG", self.mesh.ufl_cell(), 1)
        W = FunctionSpace(self.mesh, P)

        J = Function(W)
        f_y = Function(W)
        diff.vector()[:] *= diff.vector()
        J,f_y = diff.split()
        J.vector()[:] += f_y.vector()
        J.vector().set_local(np.sqrt(J.vector().get_local()))
        J.vector().apply('')

        P_T = FiniteElement("CG", self.mesh.ufl_cell(), self.param.T_CG_order)
        # W_T = FunctionSpace(self.mesh, P_T)
        J_viz = project(J,self.W_T,solver_type='mumps')

        P_idx = FiniteElement("CG", self.mesh_viz.ufl_cell(), 1)
        W_idx = FunctionSpace(self.mesh_viz, P_idx)
        v2d_idx = vertex_to_dof_map(W_idx)
        J_idx = Function(W_idx)

        # get jump everywhere
        for v in vertices(self.mesh_viz):
            point_to_v = int(I_Field.vector()[v2d_idx[v.index()]])
            xy = np.array([Vertex(self.mesh_viz,point_to_v).point().x(), Vertex(self.mesh_viz,point_to_v).point().y()],dtype=float)
            eval_j = np.array([0.0])
            J_viz.eval(eval_j,xy)
            dof_idx = v2d_idx[v.index()]
            J_idx.vector()[dof_idx] = eval_j[0]

        return J_idx

    def update_inflow_boundary(self,u):
        '''
        Here, we iterate through each cell in the mesh, get the dofs of that cell, then iterate through the facets
        of the cell and reassign the boundary if necessary, based on the u0_x values of the cell. 
        This section isn't handling mixed cases well yet. 
        '''
        dm = self.W_u_p.dofmap()
        for cell in cells(self.mesh):
            cell_dofs = dm.cell_dofs(cell.index())
            if np.all(u.vector()[cell_dofs[0:4]] > DOLFIN_EPS):
                for facet in facets(cell):
                    facet_label = self.boundaries.array()[facet.index()]
                    if (facet_label == self.inflow_wedge): # wedge inflow
                        # print('wedge inflow reassigned to outflow')
                        self.boundaries.array()[facet.index()] = self.outflow_wedge
            elif np.all(u.vector()[cell_dofs[0:4]] < DOLFIN_EPS):
                for facet in facets(cell):
                    facet_label = self.boundaries.array()[facet.index()]
                    if (facet_label == self.outflow_wedge): # wedge outflow
                        # print('wedge outflow reassigned to inflow')
                        self.boundaries.array()[facet.index()] = self.inflow_wedge
            elif np.any(u.vector()[cell_dofs[0:4]] <= DOLFIN_EPS) and np.any(u.vector()[cell_dofs[0:4]] > DOLFIN_EPS):
                for facet in facets(cell):
                    facet_label = self.boundaries.array()[facet.index()]
                    if (facet_label == self.inflow_wedge): # wedge inflow
                        # print('mixed case, wedge inflow')
                        pass
                    elif (facet_label == self.outflow_wedge): # wedge outflow
                        # print('mixed case, wedge outflow')
                        pass

    def solver_stokes(self,eta_wedge,u_n,collect_bc,I_Field):
        '''
        Starting from the modified slab flow solution, having set and collected the 
        boundary conditions for the wedge subdomain, compute the flow in the wedge. 
        '''
        # solve flow problem in wedge
        a_wedge = (inner(2*eta_wedge*sym(grad(self.phi_1)), grad(self.v_1)) - div(self.v_1)*self.phi_2 - self.v_2*div(self.phi_1))*self.dx(18)
        L_wedge = inner(self.f, self.v_1)*self.dx(18)
        U_new = Function(self.W_u_p)

        if (major_minor == "3.12"):
            # print('Starting wedge Stokes solve.')
            solve(a_wedge == L_wedge, U_new, collect_bc)
            # print('Finished wedge Stokes solve.')
        elif (major_minor == "3.14") or (major_minor == "3.17"):
            A = assemble(a_wedge)
            b = assemble(L_wedge)
            for bc in collect_bc:
                bc.apply(A)
                bc.apply(b)
            # A,b = assemble_system(a_wedge,L_wedge,collect_bc)
            solver = PETScKrylovSolver()
            solver.set_operator(A)
            solver.set_from_options()
            # print('Solving with KrylovSolver')
            # solver.solve(U_new.vector(), bb)
            # print('Starting wedge Stokes solve.')
            solver.solve(U_new.vector(),b)
            # print('Finished wedge Stokes solve.')
        u_new, p_new = U_new.split()

        for cell in cells(self.mesh):
            subdomain_index = self.subdomains.array()[cell.index()]
            cell_dofs = self.dm_W_u_p.cell_dofs(cell.index())
            if subdomain_index == 18:
                u_n.vector()[cell_dofs[0:5]] = u_new.vector()[cell_dofs[0:5]]
                u_n.vector()[cell_dofs[5:10]] = u_new.vector()[cell_dofs[5:10]]
                u_n.vector()[cell_dofs[10:]] = u_new.vector()[cell_dofs[10:]]
        
        # update inflow/outflow bc
        self.update_inflow_boundary(u_n)
        return u_n
    
    def get_depths(self):
        # get depth of overlying plate, for linear right-hand bc
        overplate_right_coords_y = []
        for facet in facets(self.mesh):
            if (self.boundaries.array()[facet.index()] == self.overplate_right):
                for v in vertices(facet):
                    overplate_right_coords_y.append(v.point().y())
        self.z_base = np.min(overplate_right_coords_y)
        self.z_top = np.max(overplate_right_coords_y)
        return 


    def get_H_sh(self,J,D_Field):
        # this should give the shear heating field and only needs to be done once
        class Pressure(UserExpression):
            def __init__(self, param, z_base, z_top, **kwargs):
                super(Pressure, self).__init__(**kwargs)
                self.param = param
                self.z_base = z_base
                self.z_top = z_top
            def eval(self, values, x):
                if x[1] >= self.z_base:
                    values[0] = self.param.rho_crust*self.param.g*np.abs(x[1] - self.z_top)
                if x[1] < self.z_base:
                    values[0] = (self.param.rho_crust*self.param.g*np.abs(self.z_base - self.z_top) \
                        + self.param.rho_mantle*self.param.g*np.abs(x[1] - self.z_base))
            def value_shape(self):
                return (1,)

        pressure = Pressure(self.param, self.z_base, self.z_top)
        H_sh = Expression(("(mu*jump*p)/(abs(sig)*sqrt(2*pi))*exp(-pow(d,2)/(2*pow(sig,2)))"), \
            degree=1, mu=self.param.mu, d=D_Field,sig=self.param.sigma,jump=J,p=pressure)
        H_sh_field = self.project_gmh(H_sh,self.W_T)
        H_sh_file = File(os.path.join(self.output_dir,"H_sh.pvd"))
        H_sh_file << H_sh_field
        return H_sh_field


    def solver_adv_diff(self,u_n,H_sh_field):
        T_n = Function(self.W_T)

        # Tbcs
        Tleft = Expression("Ts + (T0-Ts)*erf(abs(x[1])/(2*sqrt(kappa*slab_age)))", T0 = self.param.Tb, Ts = self.param.Ts, kappa = self.param.kappa_slab, slab_age=self.param.slab_age, degree = self.param.T_CG_order)  # half-space cooling on slab_left
        # Tplate = Expression("Ts + (T0-Ts)*abs(x[1] - z_top)/abs(z_base - z_top)", T0 = self.param.Tb, Ts = self.param.Ts, z_top=z_top, z_base = z_base, degree = self.param.T_CG_order) # linear for overplate_right

        class TPlate(UserExpression):
            def __init__(self, param, z_base, z_top, **kwargs):
                super(TPlate, self).__init__(**kwargs)
                self.param = param
                self.z_base = z_base
                self.z_top = z_top
            def eval(self, values, x):
                if self.param.z_bc is not None:
                    if x[1] >= - self.param.z_bc:
                        values[0] = self.param.Ts + (self.param.Tb-self.param.Ts)*abs(x[1] - self.z_top)/abs(- self.param.z_bc - self.z_top)
                    elif x[1] < - self.param.z_bc:
                        values[0] = self.param.Tb
                else:
                    values[0] = self.param.Ts + (self.param.Tb-self.param.Ts)*abs(x[1] - self.z_top)/abs(self.z_base - self.z_top)
            def value_shape(self):
                return (1,)
            
        class TInflow(UserExpression):
            def __init__(self, param, z_base, z_top, **kwargs):
                super(TInflow, self).__init__(**kwargs)
                self.param = param
                self.z_base = z_base
                self.z_top = z_top
            def eval(self, values, x):
                if self.param.z_bc is not None:
                    if x[1] >= - self.param.z_bc:
                        values[0] = self.param.Ts + (self.param.Tb-self.param.Ts)*abs(x[1] - self.z_top)/abs(- self.param.z_bc - self.z_top)
                    elif x[1] < - self.param.z_bc:
                        values[0] = self.param.Tb
                else:
                    values[0] = self.param.Tb
            def value_shape(self):
                return (1,)

        TP = TPlate(self.param,self.z_base,self.z_top)
        Tplate = Expression("TP", TP=TP, degree = self.param.T_CG_order)

        TIn = TInflow(self.param,self.z_base,self.z_top)
        T_inflow = Expression("TIn", TIn=TIn, degree = self.param.T_CG_order)

        # Initialise temperature
        Tinit = Expression("Tinit", Tinit = self.param.Ts, degree=self.param.T_CG_order) 
        T_n.interpolate(Tinit)

        Tbcs = [DirichletBC(self.W_T, Tleft, self.boundaries, self.slab_left)]
        for bc in Tbcs:
            bc.apply(T_n.vector())
        T_base_val = np.max(T_n.vector()[:])
        T_base_exp = Expression("T_base", T_base=T_base_val, degree=self.param.T_CG_order)
        Tbcs.append(DirichletBC(self.W_T, T_base_exp, self.boundaries, self.slab_base))
        Tbcs.append(DirichletBC(self.W_T, Tplate, self.boundaries, self.overplate_right))
        # Tbcs.append(DirichletBC(self.W_T, Constant(self.param.Tb), self.boundaries, self.inflow_wedge))
        Tbcs.append(DirichletBC(self.W_T, T_inflow, self.boundaries, self.inflow_wedge))
        Tbcs.append(DirichletBC(self.W_T, Constant(self.param.Ts), self.boundaries, self.surface))

        Tbcs_res = [bc for bc in Tbcs]

        for bc in Tbcs:
            bc.apply(T_n.vector())

        F_temp = inner(grad(self.v_T), Constant(self.param.k_slab)*grad(self.phi_T))*self.dx(17)
        F_temp += inner(grad(self.v_T), Constant(self.param.k_mantle)*grad(self.phi_T))*self.dx(18)
        F_temp += inner(grad(self.v_T), Constant(self.param.k_crust)*grad(self.phi_T))*self.dx(19)
        F_temp += Constant(self.param.rho_slab)*Constant(self.param.cp)*self.v_T*inner(u_n, grad(self.phi_T))*self.dx(17)
        F_temp += Constant(self.param.rho_mantle)*Constant(self.param.cp)*self.v_T*inner(u_n, grad(self.phi_T))*self.dx(18)
        F_temp += Constant(self.param.rho_crust)*Constant(self.param.cp)*self.v_T*inner(u_n, grad(self.phi_T))*self.dx(19)

        # # add in shear heating
        
        F_temp += - inner(H_sh_field,self.v_T)*dx # minus sign is here so that lhs() and rhs() works out
        a_T,L_T = lhs(F_temp),rhs(F_temp)
        T_i = Function(self.W_T)
        if (major_minor == "3.12"):
            # print('Starting energy equation solve.')
            solve( a_T == L_T, T_i, Tbcs)
            # print('Finished with energy equation solve.')
        elif (major_minor == "3.14") or (major_minor == "3.17"):
            # A,b = assemble_system(a_T,L_T,Tbcs) # DOES NOT WORK
            A = assemble(a_T)
            b = assemble(L_T)
            for bc in Tbcs:
                bc.apply(A)
                bc.apply(b)
            solver = PETScKrylovSolver()
            solver.set_operator(A)
            solver.set_from_options()
            # print('Starting energy equation solve.')
            solver.solve(T_i.vector(),b)
            # print('Finished with energy equation solve.')
        res_T = self.compute_residual(F_temp,T_i,self.W_T,Tbcs_res)
        T_n.assign(T_i)
        return T_n, res_T.norm('l2')

    def compute_residual(self,F_s,w,W,bcs):
        w_ = Function(W)
        w_.assign(w) # Assign w_ to be the solution obtained from solve()
        F = action(F_s, w_) # Contracts trial function space basis with coefficients
        # Note this modifies the boundary condition object!
        for bc in bcs:
            bc.homogenize() # Inserts zeros into Dirichlet slots
        b = assemble(F) # Assembles linear form for the residual of the variational problem
        for bc in bcs:
            bc.apply(b) # Inserts zeros into residual vector
        return b

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Functions for different rheologies.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def isoviscous(self):
        eta_slab = Constant(1)
        eta_wedge = Constant(1)
        return eta_slab,eta_wedge
    def diffcreep(self,T):
        eta_diff = (self.param.A_diff * exp(self.param.E_diff / (self.param.R*T))) / self.param.Eta_star
        eta_eff = 1.0/(1.0/eta_diff + 1.0/self.param.eta_max)
        return eta_eff
    def disccreep(self,u,T):
        def symgrad(u):
                return 0.5*(grad(u)+grad(u).T) - (1/3)*Identity(2)*div(u)
        e_dev = symgrad(u)
        e_II_term = sqrt((1/2) * inner(e_dev, e_dev))  / self.param.Tstar
        eta_disl = ( self.param.A_disl * exp(self.param.E_disl/(self.param.n*self.param.R*T)) * pow(e_II_term,((1-self.param.n)/self.param.n)) ) / self.param.Eta_star
        eta_eff = 1.0/((1/eta_disl) + (1/self.param.eta_max))
        return eta_eff
    def mixed(self,u,T):
        def symgrad(u):
                return 0.5*(grad(u)+grad(u).T) - (1/3)*Identity(2)*div(u)
        
        e_dev = symgrad(u)

        eta_diff = ( self.param.A_diff * exp(self.param.E_diff / (self.param.R*T)) ) / self.param.Eta_star
        e_II_term = sqrt((1/2) * inner(e_dev, e_dev))  / self.param.Tstar
        eta_disl = ( self.param.A_disl * exp(self.param.E_disl/(self.param.n*self.param.R*T)) * pow(e_II_term,((1-self.param.n)/self.param.n)) ) / self.param.Eta_star

        # nondimensional
        eta_eff = 1.0/((1/eta_diff) + (1/eta_disl) + (1/self.param.eta_max))

        # eta = pow((pow(eta_min,M)+pow(eta_eff,M)),(1/M))
        return eta_eff

    def distance_fields(self):
        P = FiniteElement("CG", self.mesh_viz.ufl_cell(), 1)
        W = FunctionSpace(self.mesh_viz, P)
        D = Function(W)
        d2v = dof_to_vertex_map(W)
        
        D.vector()[:] = self.load(self.dfield_fname)
        D.vector()[:] = D.vector().get_local()[d2v]

        mid_field = Function(W)
        slab_d_field = Function(W)
        mid_field.vector()[:] = self.load(self.slab_d_fname)
        slab_d_field.vector()[:] = mid_field.vector().get_local()[d2v]

        I_Field = Function(W)
        I_Field.vector()[:] = self.load(self.indices_fname)
        I_Field.vector()[:] = I_Field.vector().get_local()[d2v]

        return D,slab_d_field,I_Field

    def error_checking(self):
        # ----------------------------------------------
        # Is the ddc higher than the base of the wedge?
        wedge_base_y = []
        for facet in facets(self.mesh):
            if (self.boundaries.array()[facet.index()] == self.wedge_base):
                for v in vertices(facet):
                    wedge_base_y.append(v.point().y())
                break

        if self.param.ddc > np.max(wedge_base_y):
            pass
        else:
            ValueError('The ddc parameter is deeper than the wedge base. The transition between partial and full coupling \
                must happen within the domain. Please decrease the ddc.')

        # ----------------------------------------------
        # Is the degree of partial coupling a reasonable value, ie in (0,1)?
        if (self.param.deg_pc > 0.0) and (self.param.deg_pc < 1.0):
            pass
        else: 
            ValueError('The param deg_pc (degree of partial coupling) must be between 0 and 1.')

        # ----------------------------------------------
        # Is the slab velocity greater than zero?
        if (self.param.slab_vel > 0.0):
            pass
        else: 
            ValueError('The param slab_vel must be greater than 0.')

    def run_solver(self):
        self.error_checking()
        if self.param == None:
            AttributeError('The param object has not been set. Please set PDE_Solver.param ')

        # define function space for temperature
        P_T = FiniteElement("CG", self.mesh.ufl_cell(), self.param.T_CG_order)
        self.W_T = FunctionSpace(self.mesh, P_T)
        self.v_T = TestFunction(self.W_T)
        self.phi_T = TrialFunction(self.W_T)

        D_Field,slab_d_field,I_Field = self.distance_fields()

        tfile_pvd = File(os.path.join(self.output_dir,"temperature.pvd"))
        ufile_pvd = File(os.path.join(self.output_dir,"velocity.pvd"))

        # first solve isoviscous case in slab
        eta_slab,eta_wedge = self.isoviscous()
        u_n_slab = self.solve_slab_flow(eta_slab,eta_wedge)

        # slab_sub = SubMesh(self.mesh, self.subdomains, 17)
        # wedge_sub = SubMesh(self.mesh, self.subdomains, 18)
        # overplate_sub = SubMesh(self.mesh, self.subdomains, 19)

        # slab_sub_pvd = File(os.path.join(self.output_dir,"slab.pvd"))
        # slab_sub_pvd << slab_sub
        # wedge_sub_pvd = File(os.path.join(self.output_dir,"wedge.pvd"))
        # wedge_sub_pvd << wedge_sub
        # oplate_sub_pvd = File(os.path.join(self.output_dir,"oplate.pvd"))
        # oplate_sub_pvd << overplate_sub

        u_n,collect_bc = self.apply_pc_and_trans(u_n_slab,slab_d_field)

        J_idx = self.compute_jump(u_n_slab,u_n,I_Field)

        self.get_depths() # this can be anywhere after meshes are loaded

        # compute shear heating
        H_sh_field = self.get_H_sh(J_idx,D_Field)

        # first solve isoviscous case
        i = 0
        while i<self.param.n_picard_it:
            u_n = self.solver_stokes(eta_wedge,u_n,collect_bc,I_Field)
            # ufile_pvd << u_n
            T_n, res_T = self.solver_adv_diff(u_n,H_sh_field)
            # tfile_pvd << T_n
            print("Picard iteration: ", i)
            # print('residual u: {0}'.format(res_u))
            print('residual T: {0}'.format(res_T))
            # if  (res_u < tol) and (res_T < tol):
            if (res_T < self.param.tol):
                # print('Both residuals are below tolerance of {0}'.format(tol))
                print('Residual for T below tolerance of {0}'.format(self.param.tol))
                break
            i += 1

        # check Gaussian is scaled correctly
        # G_file = File(self.output_dir + 'gaussian.pvd')
        G = Expression(("1/(abs(sig)*sqrt(2*pi))*exp(-pow(d,2)/(2*pow(sig,2)))"), \
            degree=1, d=D_Field,sig=self.param.sigma)
        G_field = self.project_gmh(G,self.W_T)
        # G_file << G_field

        int_G = assemble(G_field*self.dx)
        print('Integral of Gaussian, compute mesh',int_G)

        eta_pvd = File(os.path.join(self.output_dir,"viscosity.pvd"))
        if self.param.viscosity_type != 'isoviscous':
            T_old = Function(self.W_T)
            T_old.assign(T_n)
            for k_it in range((self.param.n_iters)):
                print('solve', k_it)
                i = 0
                while i<self.param.n_picard_it:
                    if self.param.viscosity_type == 'diffcreep':
                        eta_wedge = self.diffcreep(T_n)
                    elif self.param.viscosity_type == 'disccreep':
                        eta_wedge = self.disccreep(u_n,T_n)
                    elif self.param.viscosity_type == 'mixed':
                        eta_wedge = self.mixed(u_n,T_n)
                    else:
                        raise ValueError("The options for viscosity must be 'isoviscous', 'diffcreep', 'disccreep', or 'mixed'.")
                    
                    P_eta = FiniteElement("DG", self.mesh.ufl_cell(), 0)
                    W_eta = FunctionSpace(self.mesh, P_eta)
                    eta_field = self.project_gmh(eta_wedge,W_eta)

                    dm = W_eta.dofmap()
                    for cell in cells(self.mesh):
                        subdomain_index = self.subdomains.array()[cell.index()]
                        cell_dofs = dm.cell_dofs(cell.index())
                        if subdomain_index == 17:
                            eta_field.vector()[cell_dofs] = Constant(1e26/self.param.Eta_star)
                        if subdomain_index == 19:
                            eta_field.vector()[cell_dofs] = Constant(1e26/self.param.Eta_star)
                
                    # u_n = self.solver_stokes(eta_wedge,u_n,collect_bc,I_Field)
                    u_n = self.solver_stokes(eta_field,u_n,collect_bc,I_Field)
                    T_n, res_T = self.solver_adv_diff(u_n,H_sh_field)

                    print("Picard iteration: ", i)
                    # print('residual u: {0}'.format(res_u))
                    print('residual T: {0}'.format(res_T))
                    # if  (res_u < tol) and (res_T < tol):
                    if (res_T < self.param.tol):
                        # print('Both residuals are below tolerance of {0}'.format(tol))
                        print('Residual for T below tolerance of {0}'.format(self.param.tol))
                        break
                    i += 1
                print('diff:', (T_old.vector() - T_n.vector()).norm('l2'))
                if (T_old.vector() - T_n.vector()).norm('l2') < self.param.diff_tol:
                    break
                T_old.assign(T_n)
            eta_store = self.project_gmh(eta_field,FunctionSpace(self.mesh, FiniteElement("CG", self.mesh.ufl_cell(), 1)))
            eta_pvd << eta_store
        tfile_pvd << T_n
        ufile_pvd << u_n
        

        self.write(np.array(T_n.vector()),os.path.join(self.output_dir, "temperature.pkl"))
        if self.param.viscosity_type != 'isoviscous':
            self.write(np.array(eta_store.vector()),os.path.join(self.output_dir, "viscosity.pkl"))
