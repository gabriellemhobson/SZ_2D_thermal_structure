import numpy as np
from copy import deepcopy
import os
from scipy.stats import qmc

class CreateSamples:
    def __init__(self,seed,method,n_inputs,l_bounds,u_bounds):
        self.seed = seed
        self.method = method
        self.n_inputs = n_inputs
        self.l_bounds = l_bounds
        self.u_bounds = u_bounds
        method_valid = ["cartesian_product", "halton","latinhypercube","corner"]
        if self.method not in method_valid:
            raise RuntimeError("Method must be one of " + str(method_valid))
        if self.method == "cartesian_product":
            self.sampler = self.cartesian_product
        elif self.method == "halton":
            self.sampler = qmc.Halton(self.n_inputs,seed=self.seed)
        elif self.method == "latinhypercube":
            self.sampler = qmc.LatinHypercube(self.n_inputs,seed=self.seed)
        elif self.method == "corner":
            self.sampler = self.corner_samples

    def cartesian_product(self,n):
        '''
        Creates arrays of values between the min and max param values using linspace, 
        then takes the cartesian product of the arrays, returning an np array.
        '''
        import itertools
        range_vals = []
        for k in range(self.n_inputs):
            range_vals.append(np.linspace(self.l_bounds[k],self.u_bounds[k],n))
        # loop through and create cartesian product list
        cart = [list(i) for i in itertools.product(*range_vals)]
        cart_nd = np.zeros((len(cart),self.n_inputs))
        for k in range(len(cart)):
            cart_nd[k,:] = cart[k]
        return cart_nd

    def corner_samples(self):
        sample = self.cartesian_product(2)
        print(sample)
        return sample

    def generate_samples(self,n):
        if self.method == "cartesian_product":
            sample = self.sampler(n)
        elif self.method == "corner":
            sample = self.sampler()
        elif self.method == "halton" or self.method == "latinhypercube":
            sample = self.sampler.random(n)
            sample = qmc.scale(sample,self.l_bounds,self.u_bounds)
        return sample

    def write_csv(self,n,dir):
        fp = open(os.path.join(dir,"sampling_info.csv"),'a')
        fp.write('Sampling method,'); fp.write(str(self.method)); fp.write("\n")
        fp.write('Seed,'); fp.write(str(self.seed)); fp.write("\n")
        fp.write('Number of samples,'); fp.write(str(n)); fp.write("\n")
        fp.close()
