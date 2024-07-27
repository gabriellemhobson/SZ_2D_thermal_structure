''' 
Create class containing the parameters for the problem.
'''
class Parameters():
    def __init__(self,T_CG_order,viscosity_type,tol,n_picard_it,n_iters,diff_tol):
        # variables for nondimensionalization
        self.cmyr_to_ms = 1/1e2/(365*24*3600) # cm/yr to m/s
        self.Lstar = 1000 # m
        self.Vstar = 1*self.cmyr_to_ms # 1 cm/yr in m/s
        self.Tstar = self.Lstar/self.Vstar # s
        self.Eta_star = 1e21 # Pa s, characteristic viscosity
        
        # set default values for physical parameters
        self.slab_vel = 5.0*self.cmyr_to_ms/self.Vstar
        self.Ts = 273 # K, temp at surface
        self.Tb = 1573 # K, temp at inflow boundary of wedge
        # self.Tb = 1623 # 274 # K, verification
        self.z_bc = None # km, depth of base of linear geotherm on overplate_right bc
        # 100 Myr for Hik, 20 Myr for Nankai, 8 Myr for cascadia
        # 10 Ma for verification
        self.slab_age = 8.0*(1e6*365.0*24.0*60.0*60.0)/self.Tstar # Myr in seconds, for erf() heat flow function, nondim
        # kappa_SI = 0.7272e-6 # m^2 s^-1
        # self.kappa = (kappa_SI / (self.Lstar**2)) * self.Tstar # nondimensional
        self.eta_max = 1e26 / self.Eta_star # Pa s then nondimensionalized
        self.A_diff = 1.32043e9 # Pa s
        self.E_diff = 335 * 1e3 # J/mol
        self.A_disl = 28968.6 # Pa s 
        self.E_disl = 540e3 # J/mol
        self.R = 8.3145 # J/mol K
        self.n = 3.5

        self.ddc = 80 # km, depth of decoupling
        self.deg_pc = 0.05 # degree of partial coupling
        self.L_trans = 10 # km, width of transition

        self.char_kg = self.Eta_star * self.Lstar * self.Tstar # derived from Eta_star
        self.mu = 0.03 # coefficient of friction
        self.rho_mantle = 3300 / self.char_kg * (self.Lstar**3) # kg m^-3 -> nondimensionalized, density of the mantle
        self.rho_crust = 2700 / self.char_kg * (self.Lstar**3) # kg m^-3 density of crust
        self.rho_slab = 3300 / self.char_kg * (self.Lstar**3) # kg m^-3 density of slab
        # self.rho_mantle = 3300 / self.char_kg * (self.Lstar**3) # verification
        # self.rho_crust = 3300 / self.char_kg * (self.Lstar**3) # verification
        # self.rho_slab = 3300 / self.char_kg * (self.Lstar**3) # verification
        # # self.rho = 3300 / char_kg * (self.Lstar**3) # kg m^-3 -> nondimensionalized, density of the mantle
        self.g = 9.8 / self.Lstar * (self.Tstar**2)# m s^-2, gravitational acceleration
        self.k_mantle = 3.1 / self.char_kg / self.Lstar * (self.Tstar**3) # W / m K -> units of K
        self.k_crust = 2.5 / self.char_kg / self.Lstar * (self.Tstar**3) # W / m K -> units of K
        self.k_slab = 3.1 / self.char_kg / self.Lstar * (self.Tstar**3) # W / m K -> units of K
        self.cp = 1250 / (self.Lstar**2) * (self.Tstar **2) # J / kg K -> units of K
        # self.k_mantle = 3.0 / self.char_kg / self.Lstar * (self.Tstar**3) # verification
        # self.k_crust = 3.0 / self.char_kg / self.Lstar * (self.Tstar**3) # verification
        # self.k_slab = 3.0 / self.char_kg / self.Lstar * (self.Tstar**3) # verification
        # self.cp = 1250 / (self.Lstar**2) * (self.Tstar **2) # J / kg K -> units of K
        self.kappa_slab = self.k_slab / (self.rho_slab * self.cp)
        self.sigma = 0.2 # 0.2

        # set numerical parameters
        self.T_CG_order = T_CG_order
        self.viscosity_type = viscosity_type
        self.tol = tol
        self.n_picard_it = n_picard_it
        self.n_iters = n_iters
        self.diff_tol = diff_tol
        self.dt = 100000 * (365*24*60*60) / self.Tstar # 100,000 yrs nondimensionalized, only used when pde_solver_time_dep.py is used 
        self.n_ts = 180

    def order_param_dict(self):
        from collections import OrderedDict
        param_ord = OrderedDict()

        keylist = list(self.__dict__.keys())
        ord = sorted(keylist)
        for key in ord:
            param_ord[key] = self.__dict__[key]
        return param_ord

    def set_param(self, **kwargs):
        ks = list(kwargs.keys())
        check_updated = {k:False for k in ks}
        if 'slab_vel' in ks:
            self.slab_vel = kwargs['slab_vel']*self.cmyr_to_ms/self.Vstar
            check_updated['slab_vel'] = True
        if 'Ts' in ks:
            self.Ts = kwargs['Ts']
            check_updated['Ts'] = True
        if 'Tb' in ks:
            self.Tb = kwargs['Tb']
            check_updated['Tb'] = True
        if 'z_bc' in ks:
            self.z_bc = kwargs['z_bc']
            check_updated['z_bc'] = True
        if 'slab_age' in ks:
            print('updating slab age')
            self.slab_age = kwargs['slab_age']*(1e6*365.0*24.0*60.0*60.0)/self.Tstar
            check_updated['slab_age'] = True
        if 'A_diff' in ks:
            self.A_diff = kwargs['A_diff']
            check_updated['A_diff'] = True
        if 'E_diff' in ks:
            self.E_diff = kwargs['E_diff']
            check_updated['E_diff'] = True
        if 'A_disl' in ks:
            self.A_disl = kwargs['A_disl']
            check_updated['A_disl'] = True
        if 'E_disl' in ks:
            self.E_disl = kwargs['E_disl']
            check_updated['E_disl'] = True
        if 'n' in ks:
            self.n = kwargs['n']
            check_updated['n'] = True
        if 'rho_mantle' in ks:
            self.rho_mantle = kwargs['rho_mantle'] / self.char_kg * (self.Lstar**3)
            check_updated['rho_mantle'] = True
        if 'rho_crust' in ks:
            self.rho_crust = kwargs['rho_crust'] / self.char_kg * (self.Lstar**3)
            check_updated['rho_crust'] = True
        if 'rho_slab' in ks:
            self.rho_slab = kwargs['rho_slab'] / self.char_kg * (self.Lstar**3)
            check_updated['rho_slab'] = True
        if 'g' in ks:
            self.g = kwargs['g'] / self.Lstar * (self.Tstar**2)
            check_updated['g'] = True
        if 'k_mantle' in ks:
            self.k = kwargs['k_mantle'] / self.char_kg / self.Lstar * (self.Tstar**3)
            check_updated['k_mantle'] = True
        if 'k_crust' in ks:
            self.k_crust = kwargs['k_crust'] / self.char_kg / self.Lstar * (self.Tstar**3)
            check_updated['k_crust'] = True
        if 'k_slab' in ks:
            self.k_slab = kwargs['k_slab'] / self.char_kg / self.Lstar * (self.Tstar**3)
            check_updated['k_slab'] = True
        if 'cp' in ks:
            self.cp = kwargs['cp'] / (self.Lstar**2) * (self.Tstar **2)
            check_updated['cp'] = True
        if ('rho_slab' in ks) or ('k' in ks) or ('cp' in ks):
            self.kappa_slab = self.k / (self.rho_slab * self.cp)
        if 'ddc' in ks:
            self.ddc = kwargs['ddc']
            check_updated['ddc'] = True
        if 'deg_pc' in ks:
            self.deg_pc = kwargs['deg_pc']
            check_updated['deg_pc'] = True
        if 'L_trans' in ks:
            self.L_trans = kwargs['L_trans']
            check_updated['L_trans'] = True
        if 'mu' in ks:
            self.mu = kwargs['mu']
            check_updated['mu'] = True
        if all(check_updated.values()):
            print('All parameter values have been updated.')
        else:
            raise RuntimeError('Not all parameter values have been properly updated.')
