from utils import load_write
import numpy as np
from dolfin import *
from petsc4py import PETSc
from scipy.spatial import KDTree
import os

class Process_Mesh():
    '''
    Process_Mesh(). 
        Initialize with:
            T_CG_order. Must match the T_CG_order that will be used for all forward model runs, as well as the order of the visualization mesh.
            meshfile_name: refers to the compute mesh.
            vizfile_name: refers to the visualization mesh, should be the compute mesh refined by splitting appropriately considering T_CG_order. 
            v4_vizfile_name: the visualization mesh, but saved as a version 4 ascii msh file rather than version 2. 
        get_slab_interface_tags()
    '''
    def __init__(self,T_CG_order,meshfile_name,vizfile_name,v4_vizfile_name,mesh_dir):
        self.T_CG_order = T_CG_order
        self.meshfile_name = meshfile_name
        self.vizfile_name = vizfile_name
        self.v4_vizfile_name = v4_vizfile_name
        self.mesh_dir = mesh_dir

        self.mesh = Mesh("./%s.xml" %(self.meshfile_name))
        self.subdomains = MeshFunction("size_t", self.mesh,"%s_physical_region.xml" %(self.meshfile_name))
        self.mesh_viz = Mesh("./%s.xml" %(self.vizfile_name))
        P_T = FiniteElement("CG", self.mesh.ufl_cell(), self.T_CG_order)
        self.W_T = FunctionSpace(self.mesh, P_T)
        self.T_n = Function(self.W_T)
        self.W_VZ = FunctionSpace(self.mesh_viz,"CG",1)
        self.T_VZ = Function(self.W_VZ)
        self.v2d = vertex_to_dof_map(self.W_VZ)
        
        self.lw = load_write.Load_Write()

    def get_slab_interface_tags(self,writefile_or_not=True):
        import gmshparser
        mesh = gmshparser.parse(self.v4_vizfile_name)

        X = []; Y = []; TAGS = [];
        for entity in mesh.get_node_entities():
            # if (entity.get_tag() == 9) or (entity.get_tag() == 10) or (entity.get_tag() == 11):
            if (entity.get_tag() == 9) or (entity.get_tag() == 10):
                for node in entity.get_nodes():
                    nid = int(node.get_tag() - 1) # subtract by 1 since xml files are 0-indexed by dolfin-convert
                    ncoords = node.get_coordinates()
                    X.append(ncoords[0])
                    Y.append(ncoords[1])
                    TAGS.append(nid)
            for node in entity.get_nodes():
                if node.get_tag() == int(1) or node.get_tag() == int(3) or node.get_tag() == int(2): # startpoint, endpoint, and cornerpoint
                        ncoords = node.get_coordinates()
                        X.append(ncoords[0])
                        Y.append(ncoords[1])
                        TAGS.append(int(node.get_tag() - 1)) # subtract by 1 since xml files are 0-indexed by dolfin-convert

        # reorder X,Y,TAGS
        ind = np.argsort(X)
        X = np.array(X)[ind]
        Y = np.array(Y)[ind]
        TAGS = np.array(TAGS)[ind]

        # write info to file
        if writefile_or_not == True:
            lw = load_write.Load_Write()
            lw.write(X,os.path.join(self.mesh_dir,'X.pkl'))
            lw.write(Y,os.path.join(self.mesh_dir,'Y.pkl'))
            lw.write(TAGS,os.path.join(self.mesh_dir,'TAGS.pkl'))
        return X,Y,TAGS
    
    def write_transfer_matrix(self,matrix_filename):
        M = PETScDMCollection.create_transfer_matrix(self.W_T,self.W_VZ)
        viewer = PETSc.Viewer().createBinary(matrix_filename, 'w')
        viewer.view(M.mat())
    
    def write_plotting_data(self):
        lw = load_write.Load_Write()
        x = self.mesh_viz.coordinates()[:,0]
        y = self.mesh_viz.coordinates()[:,1]
        mesh_cells = self.mesh_viz.cells()
        lw.write(x,os.path.join(self.mesh_dir,"coords_x.pkl"))
        lw.write(y,os.path.join(self.mesh_dir,"coords_y.pkl"))
        lw.write(mesh_cells,os.path.join(self.mesh_dir,"mesh_cells.pkl"))

    def reorder_data(self,datafile,M):
        lw = load_write.Load_Write()
        self.T_n.vector()[:] = lw.load(datafile)
        self.T_VZ.vector()[:] = M*self.T_n.vector()

        # result = np.asarray(self.T_VZ.vector())
        result = self.T_VZ.vector().get_local()
        result = result[self.v2d]
        return result

    def compute_distance_fields(self,plot_or_not=False):
        import gmshparser
        # load slab interface info and transfer matrix files
        X = self.lw.load(os.path.join(self.mesh_dir,'X.pkl'))
        Y = self.lw.load(os.path.join(self.mesh_dir,'Y.pkl'))
        TAGS = self.lw.load(os.path.join(self.mesh_dir,'TAGS.pkl'))
        matrix_filename = os.path.join(self.mesh_dir,"M.dat")
        M = self.lw.load_transfer_matrix(matrix_filename)

        # load plotting data
        coords_x = self.lw.load(os.path.join(self.mesh_dir,"coords_x.pkl"))
        coords_y = self.lw.load(os.path.join(self.mesh_dir,"coords_y.pkl"))
        mesh_cells = self.lw.load(os.path.join(self.mesh_dir,"mesh_cells.pkl"))

        mesh = gmshparser.parse(self.v4_vizfile_name)
        # get x,y coords of entire mesh
        full_XY = [];
        nids = []
        for entity in mesh.get_node_entities():
            for node in entity.get_nodes():
                nids.append(node.get_tag())
                ncoords = node.get_coordinates()
                full_XY.append([ncoords[0],ncoords[1]])
        # for each point in mesh, compute distance to nearest slab neighbor
        slab_XY = [[X[k],Y[k]] for k in range(len(X))]
        kdt = KDTree(slab_XY)
        d,i = kdt.query(full_XY)
        inds = [TAGS[j] for j in i]
        self.lw.write(d,os.path.join(self.mesh_dir,"distance_field.pkl"))
        self.lw.write(inds,os.path.join(self.mesh_dir,"distance_field_indices.pkl"))

        # ~~~~~~~~~~~~~ Compute along-slab distance ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        slab_d = [0]
        dval = 0
        slab_d_field = np.ones(len(full_XY))
        slab_d_field[TAGS[0]] = dval
        for k in range(len(X)-1):
            dval += np.sqrt( (X[k+1]-X[k])**2 + (Y[k+1]-Y[k])**2)
            slab_d.append(dval)
            slab_d_field[TAGS[k+1]] = dval

        self.lw.write(slab_d,os.path.join(self.mesh_dir,"slab_d.pkl"))
        self.lw.write(slab_d_field,os.path.join(self.mesh_dir,"slab_d_field.pkl"))

        if plot_or_not == True:
            import matplotlib.pyplot as plt
            font_size = 18
            fig = plt.figure(figsize=(18,12))
            ax1 = fig.add_subplot(121)
            # ax1.set_aspect('equal')
            ax1.plot(X,slab_d)
            ax1.set_xlabel('x (km)', fontsize=font_size)
            ax1.set_ylabel('slab_d (km)', fontsize=font_size)
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            ax2 = fig.add_subplot(122)
            ax2.plot(Y,slab_d)
            ax2.set_xlabel('y (km)', fontsize=font_size)
            ax2.set_ylabel('slab_d (km)', fontsize=font_size)
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.savefig(os.path.join(self.mesh_dir,'slab_d.png'))
            plt.close("all")
