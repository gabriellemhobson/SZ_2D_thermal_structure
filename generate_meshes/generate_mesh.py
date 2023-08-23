import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
import scipy as sp
import copy as copy

class Generate_Mesh:
    '''
    __init__() loads slab 2.0 and trench data and adds the attributes self.af, self.trench_lat_lon,
               self.trench_norm, and self.slab_norm.  
    '''
    def __init__(self,fname_slab,fname_trench,constrain,start_line,end_line):
        print('Loading file', fname_slab)
        a = np.loadtxt(fname_slab, delimiter=',')
        af = a[~np.isnan(a).any(axis=1), :] # lon lat depth

        # constrain slab data to near-trench region
        if constrain["constrain_TF"] == True:
            if constrain["less_or_greater"]=='less':
                af_const = af[af[:,1]<constrain["cut_at_lat"]]
            elif constrain["less_or_greater"]=='greater':
                af_const = af[af[:,1]>constrain["cut_at_lat"]]
            else: 
                raise ValueError('Must pass in "less" or "greater" to the "less_or_greater" key in constrain.' )
            af = af_const 
        self.af = af
        self.trench_lat_lon, trench_xy = self.load_trench_data(fname_trench,start_line,end_line) 

        self.slab_norm = self.normalize_lat_lon_data(self.af)

        # normalize trench data also
        self.trench_norm = np.zeros(np.shape(self.trench_lat_lon))
        for k in range((len(self.trench_lat_lon))):
            self.trench_norm[k] = self.normalize_point(self.af,self.trench_lat_lon[k,:])

    def normalize_lat_lon_data(self,af):
        data = copy.deepcopy(af) # this should be unnecessary
        # Determine bounding box
        minL1, maxL1 = np.min(data[:,0]), np.max(data[:,0])
        minL2, maxL2 = np.min(data[:,1]), np.max(data[:,1])
        minD, maxD   = np.min(data[:,2]), np.max(data[:,2])

        # Determine intervals
        dL1, dL2, dD = np.abs(maxL1 - minL1), np.abs(maxL2 - minL2), np.abs(maxD - minD)

        # Normalize lat/lon/z data
        data[:,0] -= minL1
        data[:,0] /= dL1
        data[:,1] -= minL2
        data[:,1] /= dL2
        if np.sign(minD) > 0:
            data[:,2] -= minD
        elif np.sign(minD) < 0:
            data[:,2] += np.abs(minD)
        data[:,2] /= dD
        return data

    def normalize_point(self,data_in,data_point_in):
        # normalize them
        data = copy.deepcopy(data_in) # this should be unnecessary
        data_point = copy.deepcopy(data_point_in)
        # Determine bounding box
        minL1, maxL1 = np.min(data[:,0]), np.max(data[:,0])
        minL2, maxL2 = np.min(data[:,1]), np.max(data[:,1])
        # Determine intervals
        dL1, dL2 = np.abs(maxL1 - minL1), np.abs(maxL2 - minL2)
        # Normalize lat/lon/z data
        if np.sign(minL1) > 0:
            data_point[0] -= minL1
        elif np.sign(minL1) < 0:
            data_point[0] += np.abs(minL1)
        data_point[0] = data_point[0]/dL1
        if np.sign(minL2) > 0:
            data_point[1] -= minL2
        elif np.sign(minL2) < 0:
            data_point[1] += np.abs(minL2)
        # data_point[1] -= minL2
        data_point[1] = data_point[1]/dL2

        if np.shape(data_point_in)[0] > 2:
            print('Point has dim 3, rescaling z axis')
            minD, maxD = np.min(data[:,2]), np.max(data[:,2])
            dD = np.abs(maxD - minD)
            if np.sign(minD) > 0:
                data_point[:,2] -= minD
            elif np.sign(minD) < 0:
                data_point[:,2] += np.abs(minD)
            # data[:,2] /= dD
            data_point[2] = data_point[2]/dD

        return data_point

    def create_RBF(self,data_in,l1s,l2s,coarse):
        data = copy.deepcopy(data_in)
        # Build RBF - for normalized data as depth = F(lat/lon), not z = F(x,y)
        coords = np.vstack((data[::coarse,0],data[::coarse,1])).T # COARSE
        slab_surface = RBFInterpolator(coords,data[::coarse,2])
        ds = slab_surface(np.vstack((l1s, l2s)).T)
        return ds

    def profile_rescale_xyz(self,data,l1s,l2s,ds):
        minL1, maxL1 = np.min(data[:,0]), np.max(data[:,0])
        minL2, maxL2 = np.min(data[:,1]), np.max(data[:,1])
        minD, maxD   = np.min(data[:,2]), np.max(data[:,2])
        dL1, dL2, dD = np.abs(maxL1 - minL1), np.abs(maxL2 - minL2), np.abs(maxD - minD)
        # Rescale ds, l1s, l2s back to lat-lon-depth coords
        l1s *= dL1
        l1s += minL1
        l2s *= dL2
        l2s += minL2
        ds *= dD
        if np.sign(minD) > 0:
            ds += minD
        elif np.sign(minD) < 0:
            ds -= np.abs(minD)

        # # Convert to xyz
        # lon = l1s * np.pi/180.0
        # lat = l2s * np.pi/180.0
        # #R = 6371.0 + ds
        # R = 6371.0

        # x = R * np.cos(lat) * np.cos(lon)
        # y = R * np.cos(lat) * np.sin(lon)
        # # z = R * np.sin(lat)
        # z = ds
        # xyz = np.vstack([x,y,z]).T
        xyz = np.vstack([l1s,l2s,ds]).T
        return xyz

    def profile_rescale_lat_lon(self,data,l1s_in,l2s_in,ds_in):
        l1s_cp,l2s_cp,ds_cp = copy.deepcopy(l1s_in), copy.deepcopy(l2s_in), copy.deepcopy(ds_in)
        minL1, maxL1 = np.min(data[:,0]), np.max(data[:,0])
        minL2, maxL2 = np.min(data[:,1]), np.max(data[:,1])
        minD, maxD   = np.min(data[:,2]), np.max(data[:,2])
        dL1, dL2, dD = np.abs(maxL1 - minL1), np.abs(maxL2 - minL2), np.abs(maxD - minD)
        # Rescale ds, l1s, l2s back to lat-lon-depth coords
        l1s_cp *= dL1
        l1s_cp += minL1
        l2s_cp *= dL2
        l2s_cp += minL2
        ds_cp *= dD
        if np.sign(minD) > 0:
            ds_cp += minD
        elif np.sign(minD) < 0:
            ds_cp -= np.abs(minD)
        xyz = np.vstack([l1s_cp,l2s_cp,ds_cp]).T
        return xyz

    def convert_slab2_to_xyz(self,af):
        '''
        Converts lat lon coordinates to x y z coordinates and returns an array 
        with data points in the row direction and x y z in the column direction. 
        '''
        af_in = copy.deepcopy(af) # not strictly necessary
        lon = af_in[:,0] * np.pi/180.0
        lat = af_in[:,1] * np.pi/180.0
        # R = 6371.0 + af_in[:,2] 
        R = 6371.0

        x = R * np.cos(lat) * np.cos(lon)
        y = R * np.cos(lat) * np.sin(lon)
        # z = R * np.sin(lat)
        z = af[:,2]
        # z = R

        xyz = np.vstack([x,y,z]).T
        print('dimension', xyz.shape)
        return xyz 

    def plot_xyz(self,slab_xyz,trench_xyz,point,profile_xyz):
        fig = plt.figure(figsize=(12,8))
        font_size=14
        ax = fig.add_subplot(111)
        fig1 = ax.scatter(slab_xyz[:,0],slab_xyz[:,1],c=slab_xyz[:,2],cmap='viridis')
        cb = plt.colorbar(fig1)
        cb.set_label(label='depth (normalized)', fontsize=font_size)
        # cfill.set_clim(0,0.0025)
        cb.ax.tick_params(labelsize=font_size)
        # fig2 = ax.scatter(profile_xyz[:,0],profile_xyz[:,1],c=profile_xyz[:,2],cmap='magma')
        fig1 = ax.scatter(trench_xyz[:,0],trench_xyz[:,1],c='g')
        fig1 = ax.scatter(trench_xyz[point,0],trench_xyz[point,1],c='m')
        fig1= ax.scatter(profile_xyz[:,0],profile_xyz[:,1],c='k')
        ax.set_xlabel('x (km)', fontsize=font_size)
        ax.set_ylabel('y (km)', fontsize=font_size)
        # ax.set_zlabel('z (km)', fontsize=font_size)
        ax.legend(['Slab data','Profile','Trench'])
        # ax.set_xlim([-3000,-1500])
        # ax.set_ylim([-4250,-2750])
        # ax.set_zlim([-3800,5000])
        # ax.view_init(azim=0, elev=90)
        plt.savefig("xyz_map_view.png")
        plt.close('all')
        # plt.show()

    def plot_profile(self,profile_1,title):
        font_size = 18
        fig = plt.figure(figsize=(10,6))
        if np.shape(profile_1)[1] > 2:
            plt.scatter(profile_1[:,0],profile_1[:,2])
        elif np.shape(profile_1)[1] <= 2:
            plt.scatter(profile_1[:,0],profile_1[:,1])
        else:
            print('Error: profile to be plotted is not the correct dimension.')
        plt.xlabel('x (km)', fontsize=font_size)
        plt.ylabel('z (km)', fontsize=font_size)
        plt.minorticks_on()
        plt.grid(visible=True, which='both')
        plt.title(title)
        plt.savefig('profile_slice_xyz.png')
        plt.close('all')
        # plt.show()

    def euclidean_distance(self,p1,p2):
        if np.shape(p1)[0] == 2:
            h = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        elif np.shape(p1)[0] == 3:
            h = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)
        else:
            print('Error: problem in size of points being passed into euclidean distance fcn.')
        return h

    def neighbour_distance(self,profile_xyz):
        h = np.zeros(np.shape(profile_xyz)[0])
        h[0] = self.euclidean_distance(profile_xyz[0],profile_xyz[1])
        for k in range(1,np.shape(profile_xyz)[0]-1):
            h[k] = np.min((self.euclidean_distance(profile_xyz[k],profile_xyz[k+1]),self.euclidean_distance(profile_xyz[k],profile_xyz[k-1])))
        h[-1] = self.euclidean_distance(profile_xyz[-2],profile_xyz[-1])
        return h

    def smooth_slice(self,profile,h,N):
        # d_flat = 150 # km
        # control_point_1 = profile[0,:] + [-d_flat,0,d_flat/40]
        # control_point_2 = profile[0,:] + [-d_flat/2,0,d_flat/40]
        # profile = np.vstack((control_point_1,control_point_2,profile))
        cs = sp.interpolate.CubicSpline(profile[:,0],profile[:,2])
        xs = np.linspace(profile[0,0],profile[-1,0],N)
        profile_smooth = np.vstack((xs,cs(xs))).T

        h_smooth = np.ones(np.shape(xs)[0])*(profile[-1,0]-profile[0,0])/N
        if np.max(h_smooth) > np.min(h):
            print('Error: interpolated profile points are coarser than data spacing.')

        return profile_smooth,h_smooth

    def x_along_profile_coords(self,profile):
        profile_trans = np.zeros((np.shape(profile)))
        profile_trans[:,1] = profile[:,1]
        profile_trans[:,2] = profile[:,2]

        d_along_profile = 0.0
        for k in range(1,np.shape(profile)[0]):
            h_k = self.euclidean_distance(profile[k,:],profile[k-1,:])
            d_along_profile += h_k
            profile_trans[k,0] = d_along_profile

        profile_trans[:,2] -= profile_trans[0,2]
        return profile_trans

    def flip_profile_lr(self,profile_in):
        profile = copy.deepcopy(profile_in)
        min_z_I,max_z_I = np.argmin(profile_in[:,1]), np.argmax(profile_in[:,1])
        if profile[max_z_I,0] < profile[min_z_I,0]:
            print('Profile does not need to be flipped, already descends left to right.')
        elif profile[max_z_I,0] > profile[min_z_I,0]:
            print('Profile must be flipped so it descends left to right.')
            profile[:,0] *= -1
        return profile

    def write_slice_to_geo(self,profile,h,N,filename,geo_info,write_msh):
        print('profile',np.shape(profile))
        print('Writing to file: ',filename)

        # find or create corner point
        corner_depth = geo_info["corner_depth"]
        corner_pt = np.argmin(np.abs(profile[:,1]-corner_depth))
        print('corner_pt',corner_pt)

        with open(filename, 'w') as f:
            for ln in geo_info['beginning_strings']:
                f.write(ln); f.write('\n')
            for k in range((np.shape(profile)[0])):
                line = 'Point(' + str(int(k)) + ') = {' + str(profile[k,0]) + ', ' + str(profile[k,1]) + ', ' + str(0.0) + ', ' + str(h[k]) + '};'
                f.write(line); f.write('\n')
            # write BSpline(1)
            spline_line = 'BSpline(1) = ' + str({*np.arange(0,np.shape(profile)[0])}) + ';'
            f.write(spline_line); f.write('\n')

            line = 'Point(' + str(int(k+1)) + ') = {' + str(profile[0,0]) + ', ' + str(profile[0,1]) + '-slab_thickness, ' + str(0.0) + ', ' + str(h[0]) + '};'
            f.write(line); f.write('\n')
            f.write('BSpline(2) = {' + str(k+1) + ',0};'); f.write('\n')
            f.write('Wire(1) = {1};'); f.write('\n')
            f.write('Extrude { Line{2}; } Using Wire {1}'); f.write('\n')
            f.write('pt_num = newp;'); f.write('\n')
            f.write('c[] = Point{pt_num-3};'); f.write('\n')
            # Point(pt_num) = {-150.0, -1.0391570839971678+overplate_notch, 0.0, h_fine};
            # Point(pt_num+1) = {530.1749673106902+extension_x, -1.0391570839971678+overplate_notch, 0.0, h_med};
            f.write('Point(pt_num) = {' + str(profile[0,0]) + ', ' + str(profile[0,1]) + '+overplate_notch, ' + str(0.0) + ', ' + str(h[0]) + '};'); f.write('\n')
            f.write('Point(pt_num+1) = {' + str(profile[-1,0]) + '+extension_x, ' + str(profile[0,1]) + '+overplate_notch, ' + str(0.0) + ', ' + str(h[0]) + '};'); f.write('\n')
            f.write('Delete {Surface{1}; }'); f.write('\n')
            f.write('Delete {Curve{6}; }'); f.write('\n')
            f.write('Line(6) = {pt_num+1, pt_num};'); f.write('\n')
            f.write('Line(7) = {pt_num, 0};'); f.write('\n')
            f.write('Line(8) = {pt_num-3,' + str(np.shape(profile)[0]-1) + '};'); f.write('\n')
            f.write('Delete {Curve{1}; Curve{5}; }'); f.write('\n')
            f.write('corner_pt = '+ str(corner_pt) +';'); f.write('\n')
            f.write('BSpline(newc) = ' + str({*np.arange(0,corner_pt+1)}) + ';'); f.write('\n')
            f.write('BSpline(newc) = ' + str({*np.arange(corner_pt,np.shape(profile)[0])}) + ';'); f.write('\n')
            f.write('c2[] = Point{corner_pt};'); f.write('\n')
            f.write('pt_num_2 = newp;'); f.write('\n')
            f.write('Point(pt_num_2) = {' + str(profile[-1,0]) + '+extension_x, c2[1], 0, h_med};'); f.write('\n')
            f.write('c3[] = Point{' + str(np.shape(profile)[0]-1) + '};'); f.write('\n')
            f.write('Point(pt_num_2+1) = {' + str(profile[-1,0]) + '+extension_x, c3[1], 0, h_med};'); f.write('\n')
            f.write('Point(pt_num_2+2) = {' + str(profile[-1,0]) + '+extension_x,' + str(profile[-1,1]) + '+z_in_out, 0, h_med};'); f.write('\n')
            f.write('Line(11) = {corner_pt,pt_num_2};'); f.write('\n')
            f.write('Line(12) = {pt_num_2-1,pt_num_2};'); f.write('\n')
            f.write('Line(13) = {pt_num_2,pt_num_2+2};'); f.write('\n')
            f.write('Line(14) = {pt_num_2+2, pt_num_2+1};'); f.write('\n')
            f.write('Line(15) = {'+ str(np.shape(profile)[0]-1) +', pt_num_2+1};'); f.write('\n')        
            f.write('Delete {Curve{2}; Curve{4}; }'); f.write('\n')
            f.write('Line(16) = {'+ str(np.shape(profile)[0]+1) +',0};'); f.write('\n')
            
            f.write('Curve Loop(3) = {6, 7, 9, 11, -12};'); f.write('\n')
            f.write('Curve Loop(4) = {10,15,-14,-13,-11};'); f.write('\n')
            f.write('Curve Loop(5) = {9, 10, -8, -3, 16};'); f.write('\n')
            f.write('Plane Surface(1) = {3};'); f.write('\n')
            f.write('Plane Surface(2) = {4};'); f.write('\n')
            f.write('Plane Surface(3) = {5};'); f.write('\n')
            f.write('Physical Surface("slab_interior", 17) = {3};'); f.write('\n')
            f.write('Physical Surface("wedge_interior", 18) = {2};'); f.write('\n')
            f.write('Physical Surface("overplate_interior", 19) = {1};'); f.write('\n')
        
            f.write('Physical Curve("surface", 20) = {6};'); f.write('\n')
            f.write('Physical Curve("overplate_right", 21) = {12};'); f.write('\n')
            f.write('Physical Curve("overplate_base", 22) = {11};'); f.write('\n')
            f.write('Physical Curve("overplate_left", 23) = {7};'); f.write('\n')
            f.write('Physical Curve("slab_left", 24) = {16};'); f.write('\n')
            f.write('Physical Curve("slab_overplate_int", 25) = {9};'); f.write('\n')
            f.write('Physical Curve("slab_wedge_int", 27) = {10};'); f.write('\n')
            f.write('Physical Curve("slab_right", 28) = {8};'); f.write('\n')
            f.write('Physical Curve("slab_base", 29) = {3};'); f.write('\n')
            f.write('Physical Curve("wedge_base", 30) = {15};'); f.write('\n')
            f.write('Physical Curve("outflow_wedge", 31) = {14};'); f.write('\n')
            f.write('Physical Curve("inflow_wedge", 32) = {13};'); f.write('\n')

            f.write('MeshSize' + str({*np.arange(0,np.shape(profile)[0])}) + ' = h_fine;'); f.write('\n')
            f.write('MeshSize{' + str(np.shape(profile)[0]+1) + ', ' + str(np.shape(profile)[0]+2) + '} = h_med;'); f.write('\n')

            if write_msh:
                f.write('Mesh 2;'); f.write('\n')
                f.write('Mesh.MshFileVersion = 2;'); f.write('\n')
                f.write('Save "'+ filename[:-4]+'.msh' +'";'); f.write('\n')
                f.write('RefineMesh;'); f.write('\n')
                f.write('Save "'+filename[:-4]+'_viz.msh'+'";'); f.write('\n')
                f.write('Mesh.MshFileVersion = 4.1;'); f.write('\n')
                f.write('Save "'+filename[:-4]+'_viz_v4_ascii.msh'+'";'); f.write('\n')

            f.close()
            return 

    def direction_of_subduction(self,point,plate_vel_vec,scaling,trench_xy):
        theta_p = np.radians(plate_vel_vec[0]) # deg N
        phi_p = np.radians(plate_vel_vec[1]) # deg E
        omega_p = plate_vel_vec[2] / 1e6 * (np.pi/180) # deg Myr-1 converted to rad / yr

        omega_x = np.cos(theta_p) * np.cos(phi_p)
        omega_y = np.cos(theta_p) * np.sin(phi_p)
        # omega_z = np.sin(theta_p)
        # omega_vec = omega_p * np.array([omega_x, omega_y, omega_z])
        omega_vec = omega_p * np.array([omega_x, omega_y])

        vel_vec = np.cross(omega_vec,trench_xy[point,:])
        print('vel_vec',vel_vec)
        # vel_vec_norm = normalize_point(slab_xyz,vel_vec)
        end_point_xy = trench_xy[point,:] + scaling*vel_vec # in x y z coords
        # end_point = trench_norm[point,:] + scaling*vel_vec_norm[0:1]
        return end_point_xy

    def trench_normal(self,point,scaling,trench_data):
        trench_tangent = trench_data[point+1,:] - trench_data[point-1,:]
        vel_vec = np.array([trench_tangent[1],-trench_tangent[0]])
        # if len(trench_data[point,:]) > 2:
        #     end_point = trench_data[point,:] + scaling*np.append(vel_vec,0.0)
        if len(trench_data[point,:]) == 2:
            end_point = trench_data[point,:] + scaling*vel_vec
        else:
            raise ValueError('Mismatch in length of trench_data[point,:] and vel_vec.')
        return end_point

    def load_trench_data(self,fname,start_line,end_line):
        print('Loading file', fname)
        f = open(fname,'r')
        boundaries_all = f.readlines()
        boundaries_all = np.array(boundaries_all[start_line:end_line])
        boundaries = np.zeros((np.shape(boundaries_all)[0],2))
        for k in range((np.shape(boundaries_all)[0])):
            boundaries[k,:] = boundaries_all[k].split(sep=',')
        boundaries = boundaries.astype(float)

        # get lon into same ref as slab data 
        if np.any(boundaries[:,0] < 0):
            print('Adding 360 to longitudes so they are in same ref frame as slab data.')
            boundaries[:,0] += 360

        lon = boundaries[:,0] * np.pi/180.0
        lat = boundaries[:,1] * np.pi/180.0
        R = 6371.0

        x = R * np.cos(lat) * np.cos(lon)
        y = R * np.cos(lat) * np.sin(lon)
        # z = R * np.sin(lat)

        xy = np.vstack([x,y]).T
        print('dimension', xy.shape)
        return boundaries, xy

    def run_generate_mesh(self,geo_filename,geo_info,start_point_latlon,end_point_latlon,plot_verbose=False,write_msh=False):

        start_point = self.normalize_point(self.af,start_point_latlon)
        # end_point_xy = trench_normal(point,scaling,trench_xy)
        # end_point_xy = direction_of_subduction(point,plate_vel_vec,scaling,trench_xy)

        end_point = self.normalize_point(self.af,end_point_latlon)

        nsample = 100
        l1s = np.linspace(start_point[0], end_point[0], nsample)
        l2s = np.linspace(start_point[1], end_point[1], nsample)
        ds = self.create_RBF(self.slab_norm,l1s,l2s,coarse=10)

        profile_lat_lon = self.profile_rescale_lat_lon(self.af,l1s,l2s,ds)  

        profile_xyz = self.convert_slab2_to_xyz(profile_lat_lon)
        self.plot_profile(profile_xyz,'Profile in x,z coords') # here profile is x,y,z
        h = self.neighbour_distance(profile_xyz)
        
        # do the conversion here
        profile_trans = self.x_along_profile_coords(profile_xyz)
        # self.plot_profile(profile_trans,'Profile transformed into along-slab coords')

        N = 1000
        profile_smooth,h_smooth = self.smooth_slice(profile_trans,h,N)
        # self.plot_profile(profile_smooth,'Profile interpolated to add N = {} points'.format(N))

        profile_flipped = self.flip_profile_lr(profile_smooth)
        self.plot_profile(profile_flipped,'Final profile')
        
        self.write_slice_to_geo(profile_flipped,h_smooth,N,geo_filename,geo_info,write_msh=write_msh)

        if plot_verbose==True:

            fig = plt.figure(figsize=(11,8))
            font_size=14
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            fig1 = ax.scatter(self.af[:,0],self.af[:,1],c=self.af[:,2],cmap='viridis')
            cb = plt.colorbar(fig1)
            cb.set_label(label='depth (normalized)', fontsize=font_size)
            cb.ax.tick_params(labelsize=font_size)
            fig1 = ax.scatter(self.trench_lat_lon[:,0],self.trench_lat_lon[:,1],c='g')
            ax.set_xlabel('lon', fontsize=font_size)
            ax.set_ylabel('lat', fontsize=font_size)
            plt.minorticks_on()
            plt.grid(visible=True,which='both')
            plt.savefig("latlon_map_view.pdf")
            plt.show()
            plt.close('all')

            fig = plt.figure(figsize=(12,8))
            font_size=14
            ax = fig.add_subplot(111,projection='3d')
            ax.scatter3D(self.slab_norm[:,0],self.slab_norm[:,1],self.slab_norm[:,2],c=self.slab_norm[:,2],cmap='viridis',alpha=0.04)
            ax.scatter3D(start_point[0],start_point[1])
            ax.scatter3D(end_point[0],end_point[1])
            ax.scatter3D(l1s,l2s,ds)
            ax.set_xlabel('x', fontsize=font_size)
            ax.set_ylabel('y', fontsize=font_size)
            # ax.view_init(azim=0, elev=90)
            plt.title('Normalized slab data with profile')
            plt.show()
            plt.close('all')

            fig = plt.figure(figsize=(11,8))
            font_size=14
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            fig1 = ax.scatter(self.af[:,0],self.af[:,1],c=self.af[:,2],cmap='viridis')
            cb = plt.colorbar(fig1)
            cb.set_label(label='depth (normalized)', fontsize=font_size)
            cb.ax.tick_params(labelsize=font_size)
            fig1 = ax.scatter(self.trench_lat_lon[:,0],self.trench_lat_lon[:,1],c='g')
            fig1 = ax.scatter(start_point_latlon[0],start_point_latlon[1],c='b')
            fig1 = ax.scatter(profile_lat_lon[:,0],profile_lat_lon[:,1],c='k')
            fig1 = ax.scatter(end_point_latlon[0],end_point_latlon[1],c='orange')
            # fig1 = ax.scatter3D(trench_lat_lon[point,0],trench_lat_lon[point,1],c='m')
            ax.set_xlabel('lon', fontsize=font_size)
            ax.set_ylabel('lat', fontsize=font_size)
            #ax.set_xlim(164.0,175.0)
            # ax.set_ylim(-45.0,-34.0)
            # ax.set_zlabel('depth', fontsize=font_size)
            # ax.view_init(azim=0, elev=90)
            plt.minorticks_on()
            plt.grid(visible=True,which='both')
            plt.savefig("latlon_map_view_with_profile.pdf")
            plt.show()
            plt.close('all')

        return profile_flipped,profile_lat_lon

    def plotting_slices_map(self,start_points_lon_lat_arr,end_points_lon_lat_arr,profiles_lon_lat,labels,label_offset,fig_name):
        
        fig = plt.figure(figsize=(12,8))
        font_size=14
        ax = fig.add_subplot(111)
        fig1 = ax.scatter(self.af[:,0],self.af[:,1],c=self.af[:,2],cmap='viridis',alpha=1)
        cb = plt.colorbar(fig1)
        cb.draw_all()
        cb.set_alpha(1)
        cb.set_label(label='depth', fontsize=font_size)
        cb.ax.tick_params(labelsize=font_size)
        # fig1 = ax.scatter(self.trench_lat_lon[:,0],self.trench_lat_lon[:,1],c='g')
        for k in range(len(labels)):
            fig1 = ax.scatter(start_points_lon_lat_arr[k][0],start_points_lon_lat_arr[k][1],c='b')
            fig1 = ax.scatter(end_points_lon_lat_arr[k][0],end_points_lon_lat_arr[k][1],c='orange')
            fig1 = ax.plot(profiles_lon_lat[k][:,0],profiles_lon_lat[k][:,1],c='k')
            fig1 = ax.text(start_points_lon_lat_arr[k][0]+label_offset[0],start_points_lon_lat_arr[k][1]+label_offset[1],labels[k])
        ax.set_xlabel('lon', fontsize=font_size)
        ax.set_ylabel('lat', fontsize=font_size)
        ax.legend(['Slab data','Start point','End point','Profile'])
        
        plt.savefig(fig_name)
        plt.show()