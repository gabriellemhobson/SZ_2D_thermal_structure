'''
Analyzing isotherms and their intersection with the slab interface. 
'''
import numpy as np
# from scipy.interpolate import interp2d

class Find_Isotherms():
    def __init__(self,iso_vals):
        self.iso_vals = iso_vals

    def locate_isotherm_intersection_vertex(self,X,Y,T):
        lst = []
        for t in self.iso_vals:
            I = np.argmin(np.abs(np.array(T) - t))
            lst.append([X[I], Y[I], T[I]])
        if len(self.iso_vals) == 1:
            lst = [lst]
        return lst

    def locate_isotherm_intersection(self,X,Y,T,ref,tol):
        lst = []
        for t in self.iso_vals:
            for k in range(len(T)-1):
                if (t >= T[k]) and (t < T[k+1]):
                    xs = np.linspace(X[k],X[k+1],ref)
                    ys = np.linspace(Y[k],Y[k+1],ref)
                    Ts = np.linspace(T[k],T[k+1],ref)
                    # f = interp2d(xs,ys,Ts,kind='linear')
                    if np.min(np.abs(np.array(Ts) - t)) < tol:
                        I = np.argmin(np.abs(np.array(Ts) - t))
                        lst.append([xs[I], ys[I], Ts[I]])
                    else:
                        print('Cannot find isotherm to given tol with that refinement level. Please increase ref value or drop tol value.')
        if len(self.iso_vals) == 1:
            lst = [lst]
        return lst

    def along_slab_distance(self,X,Y,T,iso_info):
        # X = np.array(X); Y = np.array(Y)
        D = []
        for inter in iso_info:
            X_cut = X[X<=inter[0]]
            Y_cut = Y[Y>=inter[1]]
            diff_X = np.append(X_cut[1:] - X_cut[0:-1],np.abs(inter[0] - X_cut[-1]))
            diff_Y = np.append(Y_cut[1:] - Y_cut[0:-1],np.abs(inter[1] - Y_cut[-1]))
            D.append(np.sum(np.sqrt(diff_X**2 - diff_Y**2)))
        return D

