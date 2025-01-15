'''
Plotting results. 
'''
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as mcl

class Plotting():
    def __init__(self,title,png_name,level_vals,iso_info=None):
        self.title = title 
        self.png_name = png_name 
        self.level_vals = level_vals
        self.iso_info = iso_info

    def plot_result(self,x,y,mesh_cells,result,cmap_name='bwr'):
        '''
        Creates a color plot with black contours on top at the levels specified in level_vals. 
        Input level_vals must be a tuple. 
        '''
        tri_mesh = tri.Triangulation(x, y, mesh_cells)

        font_size = 24
        fig = plt.figure(figsize=(18,12))
        ax1 = fig.add_subplot(111)
        ax1.set_aspect('equal')
        cfill = ax1.tricontourf(tri_mesh, result, levels=200, cmap=cmap_name)
        cb = plt.colorbar(cfill, ax=ax1, fraction=0.025)
        cb.set_label(label='T (K)', fontsize=font_size)
        # cfill.set_clim(0,0.0025)
        cb.ax.tick_params(labelsize=font_size)
        contours = ax1.tricontour(tri_mesh, result, levels=self.level_vals, colors='k')
        fmt = {}
        for l, s in zip(contours.levels, self.level_vals):
            fmt[l] = str(round(s,2))
        # ax1.clabel(contours, contours.levels, fmt=fmt, inline=True, fontsize=font_size)
        fig = plt.xlabel('x (km)', fontsize=font_size)
        fig = plt.ylabel('y (km)', fontsize=font_size)
        fig = plt.xticks(fontsize=font_size)
        fig = plt.yticks(fontsize=font_size)
        fig = plt.title(self.title, fontsize=font_size)

        # colors = list(mcl.TABLEAU_COLORS.keys())
        colors = ['tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue', 'tab:red']
        if self.iso_info is not None:
            for k in range(len(self.iso_info)):
                if round(self.iso_info[k][2],2) > 600.0:
                    label_str = 'Downdip limit: x = '
                else: 
                    label_str = 'Updip limit: x = '
                plt.scatter(self.iso_info[k][0], self.iso_info[k][1],c=colors[k],marker='*', s=600, zorder=2, \
                    label=label_str+str(round(self.iso_info[k][0],2))+' km, y = '\
                        +str(round(self.iso_info[k][1],2))+' km, T = '\
                        +str(round(self.iso_info[k][2],2))+' K')
            # plt.scatter(self.iso_info[1][0], self.iso_info[1][1],c='tab:orange',marker='*', s=100, zorder=3, \
            #     label='Downdip limit: x = '+str(round(self.iso_info[1][0],2))+' km, y = '\
            #         +str(round(self.iso_info[1][1],2))+' km, T = '\
            #         +str(round(self.iso_info[1][2],2))+' K')
        # plt.legend(loc='lower left',fontsize=font_size)
        plt.xticks(fontsize=font_size);
        plt.yticks(fontsize=font_size);
        # plt.xlim([0,350])
        # plt.ylim([-150,0])
        # plt.rcParams["font.family"] = "Times"
        plt.savefig(self.png_name)
        plt.close("all")

        return tri_mesh
        
    def plot_isotherm_variation(self,x,y,mesh_cells,X,Y,X_eval):
        print('Plotting isotherm variation')
        tri_mesh = tri.Triangulation(x, y, mesh_cells)
        font_size = 18
        fig = plt.figure(figsize=(18,12))
        ax1 = fig.add_subplot(111)
        ax1.set_aspect('equal')
        fig = plt.plot(X,Y,'k-')
        for k in range(X_eval.shape[1]):
            ax1.tricontour(tri_mesh, X_eval[:,k], levels=[423], colors='lightskyblue')
            ax1.tricontour(tri_mesh, X_eval[:,k], levels=[623], colors='0.8')
        fig = plt.xticks(fontsize=font_size);
        fig = plt.yticks(fontsize=font_size);
        fig = plt.xlabel('x (km)', fontsize=font_size)
        fig = plt.ylabel('y (km)', fontsize=font_size)
        fig = plt.title(self.title, fontsize=font_size)
        plt.savefig(self.png_name)
        plt.close("all")
