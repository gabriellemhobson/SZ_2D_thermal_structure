import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
import scipy as sp
from scipy.spatial import KDTree
import copy as copy

np.set_printoptions(legacy='1.25')

class Generate_Mesh:
    '''
    __init__() loads slab 2.0 and trench data and adds the attributes self.af and self.slab_norm.
                It also sets the attribute abbrev_dist, which sets how far a profile can extrapolate 
                outside of Slab2 data before being abbreviated. 
    '''
    def __init__(self,fname_slab,constrain):
        self.abbrev_dist = 10.0 # km
        
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
        
        self.slab_norm = self.normalize_lat_lon_data(self.af)

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
            # h_k = self.euclidean_distance(profile[k,:],profile[k-1,:])
            h_k = self.euclidean_distance(profile[k,0:2],profile[k-1,0:2])
            d_along_profile += h_k
            profile_trans[k,0] = d_along_profile

        print('Shifting profile up by : ', profile_trans[0,2])
        self.vertical_shift = profile_trans[0,2]
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
    
    def get_point_normal(self,profile,slab_thickness):
        df = (profile[1,:] - profile[0,:])
        dn = df[::-1]*np.array([1,-1])
        pn = (slab_thickness/np.linalg.norm(dn))*dn
        return pn

    def write_slice_to_geo(self,profile,h,N,pn,filename,geo_info,write_msh,adjust_depths):
        print('profile shape',np.shape(profile))
        print('Writing to file: ',filename)

        # find or create corner point
        if adjust_depths:
            corner_depth = - geo_info["overplate_thickness"] - self.vertical_shift
            print('With vertical_shift of', self.vertical_shift, \
                  "and overplate_thickness of", geo_info["overplate_thickness"], \
                    "the corner_depth is", corner_depth)
        else:
            corner_depth = - geo_info["overplate_thickness"]
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

            # line = 'Point(' + str(int(k+1)) + ') = {' + str(profile[0,0]) + ', ' + str(profile[0,1]) + '-slab_thickness, ' + str(0.0) + ', ' + str(h[0]) + '};'
            line = 'Point(' + str(int(k+1)) + ') = {' + str(pn[0]) + ', ' + str(pn[1]) + ', ' + str(0.0) + ', ' + str(h[0]) + '};'
            f.write(line); f.write('\n')
            f.write('BSpline(2) = {' + str(k+1) + ',0};'); f.write('\n')
            f.write('Wire(1) = {1};'); f.write('\n')
            f.write('Extrude { Line{2}; } Using Wire {1}'); f.write('\n')
            f.write('pt_num = newp;'); f.write('\n')
            f.write('c[] = Point{pt_num-3};'); f.write('\n')

            # f.write('Point(pt_num) = {' + str(profile[0,0]) + ', ' + str(profile[0,1]) + '+overplate_notch, ' + str(0.0) + ', ' + str(h[0]) + '};'); f.write('\n')
            # f.write('Point(pt_num+1) = {' + str(profile[-1,0]) + '+extension_x, ' + str(profile[0,1]) + '+overplate_notch, ' + str(0.0) + ', ' + str(h[0]) + '};'); f.write('\n')
            
            # f.write('Point(pt_num) = {' + str(profile[0,0]) + ', ' + str(profile[0,1]) + '+overplate_notch, ' + str(0.0) + ', ' + str(h[0]) + '};'); f.write('\n')
            f.write('Point(pt_num+1) = {' + str(profile[-1,0]) + '+extension_x, ' + str(profile[0,1]) + ', ' + str(0.0) + ', h_med};'); f.write('\n')
            

            f.write('Delete {Surface{1}; }'); f.write('\n')
            f.write('Delete {Curve{6}; }'); f.write('\n')
            f.write('Line(6) = {pt_num+1, 0};'); f.write('\n')
            # f.write('Line(7) = {pt_num, 0};'); f.write('\n')
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
            
            # f.write('Curve Loop(3) = {6, 7, 9, 11, -12};'); f.write('\n')
            f.write('Curve Loop(3) = {6, 9, 11, -12};'); f.write('\n')
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
            # f.write('Physical Curve("overplate_left", 23) = {7};'); f.write('\n')
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

    def run_generate_mesh(self,geo_filename,geo_info,start_point_latlon,end_point_latlon,slab_thickness,plot_verbose,write_msh=False,adjust_depths=False):

        start_point = self.normalize_point(self.af,start_point_latlon)
        # end_point_xy = trench_normal(point,scaling,trench_xy)
        # end_point_xy = direction_of_subduction(point,plate_vel_vec,scaling,trench_xy)

        end_point = self.normalize_point(self.af,end_point_latlon)

        nsample = 100
        l1s = np.linspace(start_point[0], end_point[0], nsample)
        l2s = np.linspace(start_point[1], end_point[1], nsample)
        
        # redo start point so it is nearest to trench
        # this is useful if the start point lies outside of the slab2 data
        dist_arr = np.ones(self.slab_norm.shape[0])
        for k in range((l1s.shape[0])):
            dist_in = np.ones(self.slab_norm.shape[0])
            for j in range((self.slab_norm.shape[0])):
                dist_in[j] = np.linalg.norm(np.array([l1s[k],l2s[k]]) - self.slab_norm[j,0:2])
            dist_arr[k] = np.min(dist_in)
            if dist_arr[k] < 0.01:
                start_point = np.array([l1s[k], l2s[k]])
                break

        l1s = np.linspace(start_point[0], end_point[0], nsample)
        l2s = np.linspace(start_point[1], end_point[1], nsample)


        # ``````````````````````````````````````
        ds = self.create_RBF(self.slab_norm,l1s,l2s,coarse=10)


        profile_lat_lon = self.profile_rescale_lat_lon(self.af,l1s,l2s,ds)  

        profile_xyz_ext = self.convert_slab2_to_xyz(profile_lat_lon)

        # abbreviate profile where it goes beyond Slab2 data
        af_xyz = self.convert_slab2_to_xyz(self.af)
        tree = KDTree(af_xyz)
        dist, I = tree.query(profile_xyz_ext)
        profile_xyz = profile_xyz_ext[dist < self.abbrev_dist]
        profile_lat_lon = profile_lat_lon[dist < self.abbrev_dist]

        self.profile_xyz = profile_xyz
        self.plot_profile(profile_xyz,'Profile in x,z coords') # here profile is x,y,z
        h = self.neighbour_distance(profile_xyz)
        
        self.profile_xyz_compare = profile_xyz
        # do the conversion here
        profile_trans = self.x_along_profile_coords(profile_xyz)
        self.plot_profile(profile_trans,'Profile transformed into along-slab coords')

        N = 1000
        profile_smooth,h_smooth = self.smooth_slice(profile_trans,h,N)
        self.plot_profile(profile_smooth,'Profile interpolated to add N = {} points'.format(N))

        profile_flipped = self.flip_profile_lr(profile_smooth)
        self.plot_profile(profile_flipped,'Final profile')
        
        # check if they are all monotonically decreasing
        if np.argmax(profile_flipped[:,1]) != 0:
            print('Profile is not monotonically decreasing.')
            exit()
        else:
            print('Profile is monotonically decreasing.')

        # compute the point normal to the end of the slab interface
        pn = self.get_point_normal(profile_flipped,slab_thickness)

        self.write_slice_to_geo(profile_flipped,h_smooth,N,pn,geo_filename,geo_info,write_msh=write_msh,adjust_depths=adjust_depths)

        if plot_verbose:
            fig = plt.figure(figsize=(12,8))
            font_size=14
            ax = fig.add_subplot(111,projection='3d')
            ax.scatter3D(af_xyz[:,0],af_xyz[:,1],af_xyz[:,2],c=af_xyz[:,2],cmap='viridis',alpha=0.04)
            # ax.scatter3D(start_point[0],start_point[1])
            # ax.scatter3D(end_point[0],end_point[1])
            ax.scatter3D(profile_xyz[:,0],profile_xyz[:,1],profile_xyz[:,2])
            ax.set_xlabel('x', fontsize=font_size)
            ax.set_ylabel('y', fontsize=font_size)
            # ax.view_init(azim=0, elev=90)
            plt.title('Slab data with profile in x, y, depth')
            plt.show()
            plt.close('all')

            fig = plt.figure(figsize=(11,8))
            font_size=14
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            fig1 = ax.scatter(self.af[:,0],self.af[:,1],c=self.af[:,2],cmap='viridis')
            cb = plt.colorbar(fig1)
            cb.set_label(label='depth', fontsize=font_size)
            cb.ax.tick_params(labelsize=font_size)
            fig1 = ax.scatter(start_point_latlon[0],start_point_latlon[1],c='b')
            fig1 = ax.scatter(profile_lat_lon[:,0],profile_lat_lon[:,1],c='k')
            fig1 = ax.scatter(end_point_latlon[0],end_point_latlon[1],c='orange')
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
        fig1 = ax.scatter(self.af[:,0],self.af[:,1],c=self.af[:,2],cmap='cividis',alpha=1)
        cb = plt.colorbar(fig1)
        cb.solids.set(alpha=1)
        cb.set_alpha(1)
        cb.set_label(label='depth', fontsize=font_size)
        cb.ax.tick_params(labelsize=font_size)
        for k in range(len(labels)):
            fig1 = ax.scatter(start_points_lon_lat_arr[k][0],start_points_lon_lat_arr[k][1],c='b')
            fig1 = ax.scatter(end_points_lon_lat_arr[k][0],end_points_lon_lat_arr[k][1],c='orange')
            fig1 = ax.plot(profiles_lon_lat[k][:,0],profiles_lon_lat[k][:,1],c='k')
            fig1 = ax.text(start_points_lon_lat_arr[k][0]+label_offset[0],start_points_lon_lat_arr[k][1]+label_offset[1],labels[k])
        ax.set_xlabel('lon', fontsize=font_size)
        ax.set_ylabel('lat', fontsize=font_size)
        ax.legend(['Slab data','Start point','End point','Profile'])
        # ax.set_xlim([170.0,180])
        # ax.set_ylim([-45.0,-37.0])
        plt.savefig(fig_name)
        plt.show()


    def plotting_slices_map_cartopy(self,start_points_lon_lat_arr,end_points_lon_lat_arr,profiles_lon_lat,labels,label_offset,fig_name):
        
        import cartopy
        import cartopy.crs as ccrs
        from cartopy import config
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        from cartopy import feature as cfeature
        from cartopy.feature import NaturalEarthFeature, LAND, COASTLINE, OCEAN, LAKES, BORDERS
        import matplotlib.ticker as mticker

        fs = 12
        clon = (np.min(self.af[:,0]) + np.max(self.af[:,0]))/2
        fig,ax = plt.subplots(figsize=(10,10),frameon=True,subplot_kw={'projection':ccrs.PlateCarree(central_longitude=clon)})
        # ax = plt.axes(projection=ccrs.Mercator(central_longitude=135.0, min_latitude=31.0, max_latitude=39.0, globe=None))
        # ax = plt.axes(projection=ccrs.Mercator())
        ax.set_aspect('equal')
        g1 = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, linestyle='dashed', draw_labels=True, alpha=0.7)
        g1.ylocator = mticker.FixedLocator(np.arange(-90.0, 90.0, 1.0))
        g1.xlocator = mticker.FixedLocator(np.arange(-180.0, 180.0, 1))
        g1.xformatter = LONGITUDE_FORMATTER
        g1.yformatter = LATITUDE_FORMATTER

        g1.top_labels = False 
        g1.right_labels = False 

        g1.xlabel_style = {'size': fs, 'color': 'k'}
        g1.ylabel_style = {'size': fs, 'color': 'k'}

        print(np.min(self.af[:,0]), np.max(self.af[:,0]))
        print(np.min(self.af[:,1]), np.max(self.af[:,1]))
        fig1 = ax.scatter(self.af[:,0],self.af[:,1],c=[self.af[:,2]],cmap='cividis',alpha=1, transform=ccrs.PlateCarree(),vmin=-400,vmax=0)
        
        # nankai
        # ax.set_extent([131.0, 138.0, 31.0, 38.0],crs=ccrs.PlateCarree())

        # cascadia
        # ax.set_extent([232.0, 242.0, 40.0, 50.0],crs=ccrs.PlateCarree())

        # hikurangi
        # ax.set_extent([173.0, 179.0, -43.0, -37.0],crs=ccrs.PlateCarree())

        ax.set_extent([np.min(self.af[:,0]), np.max(self.af[:,0]), np.min(self.af[:,1]), np.max(self.af[:,1])],crs=ccrs.PlateCarree())

        cb = plt.colorbar(fig1,shrink=0.6)
        cb.solids.set(alpha=1)
        cb.set_alpha(1)
        # cb.set_clim([-400.0,0.0])
        cb.set_label(label='Depth (km)', fontsize=fs)
        cb.ax.tick_params(labelsize=fs)
        # cb.ax.set_aspect(30)

        for k in range(len(labels)):
            fig1 = ax.plot(profiles_lon_lat[k][:,0],profiles_lon_lat[k][:,1],c='k', transform=ccrs.PlateCarree(),zorder=1)
            # fig1 = ax.scatter(start_points_lon_lat_arr[k][0],start_points_lon_lat_arr[k][1],c='b', transform=ccrs.PlateCarree(),zorder=3)
            # fig1 = ax.scatter(end_points_lon_lat_arr[k][0],end_points_lon_lat_arr[k][1],c='orange', transform=ccrs.PlateCarree(),zorder=3)
            fig1 = ax.text(start_points_lon_lat_arr[k][0]+label_offset[0],start_points_lon_lat_arr[k][1]+label_offset[1],labels[k], transform=ccrs.PlateCarree(),zorder=3)

        ax.coastlines();
        # ax.gridlines(draw_labels=True)
        # ax.set_aspect('equal')
        # plt.gca().set_aspect('equal')
        ax.set_aspect('equal')
        plt.savefig(fig_name)
        plt.show()

    def convert_profile_back_lat_lon(self, xy):
        """
        Function to convert points from the model-coordinate space back to the lat lon space 
        from which the profile was originally drawn. 
        """
        # undo vertical shift
        print('self.vertical_shift', self.vertical_shift)
        
        pt = xy[:,:]
        pt[:,-1] += self.vertical_shift

        # convert along profile x y z to standard x y z 
        ptb = pt[:,:]
        
        A = self.profile_xyz[0,:]
        B = self.profile_xyz[-1,:]

        print('A', A)
        print('B', B)
        rise = np.abs(B[1] - A[1])
        run = np.abs(B[0] - A[0])
        print('rise', rise)
        print('run', run)
        theta = np.arctan(run/rise)
        print(np.rad2deg(theta))

        xp = A[0] + ptb[:,0]*np.sin(theta) 
        yp = A[1] - ptb[:,0]*np.cos(theta)

        font_size = 18
        fig = plt.figure(figsize=(10,6))
        plt.scatter(xp,yp, c='red', marker='o')
        plt.scatter(self.profile_xyz[:,0],self.profile_xyz[:,1], c='blue', marker='+')

        plt.xlabel('x (km)', fontsize=font_size)
        plt.ylabel('y (km)', fontsize=font_size)
        plt.minorticks_on()
        plt.grid(visible=True, which='both')
        plt.axis('equal')
        plt.show()

        # convert x y z to lat lon 
    
        LL = np.zeros(xy.shape)
        
        R = 6371.0
        lon = np.arctan2(yp,xp)
        lat = np.arccos( yp / (R*np.sin(lon) ))
        
        lon /= np.pi/180.0
        lat /= np.pi/180.0

        LL[:,0] = lon + 360.0
        LL[:,1] = lat
        LL[:,2] = ptb[:,-1]

        return LL