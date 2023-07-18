import numpy as np
import pickle as pkl

class Load_Write:
    '''
    Load_write does not need to be initialized with any arguments. 
    Functions:
        load(filename) 
            Loads and returns the contents of a .pkl file.
            Arguments: filename, a string containing the name of a .pkl file. 
            Returns: vector, the contents of the file.
        write(vector,filename) 
            Writes data to a .pkl file.
            Arguments:  vector, some data to be written to a file
                        filename, the name of the file to write
            Returns: none, just writes the file. 
        load_transfer_matrix(matrix_filename)
            Loads the matrix written by Process_Mesh.write_transfer_matrix(), 
            which defines the transformation from a FunctionSpace defined on the compute mesh 
            to a FunctionSpace defined on the visualization mesh. 
            Arguments: matrix_filename, the name of the file to load
            Returns: M_mat, the transformation matrix. 

    '''
    def __init__(self,verbose=True):
        self.verbose=verbose

    def load(self,filename):
        file = open(filename, "rb")
        vector = pkl.load(file)
        file.close()
        return vector

    def write(self,vector,filename):
        file = open(filename, "wb")
        pkl.dump(vector, file)
        file.close()

    def load_transfer_matrix(self,matrix_filename):
        from petsc4py import PETSc
        from dolfin import PETScMatrix

        viewer = PETSc.Viewer().createBinary(matrix_filename, 'r')
        M = PETSc.Mat().load(viewer)
        M_mat = PETScMatrix(M)
        return M_mat