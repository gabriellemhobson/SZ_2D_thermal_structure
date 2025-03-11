import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.spatial import KDTree
import copy as copy
import pyproj

np.set_printoptions(legacy='1.25')

class Generate_Mesh:
    '''
    __init__() loads slab 2.0 and trench data and adds the attributes self.af and self.slab_norm.
                It also sets the attribute abbrev_dist, which sets how far a profile can extrapolate 
                outside of Slab2 data before being abbreviated. 
                It also sets the attribute R, radius of the earth in km. 
    '''
    def __init__(self, fname_slab, constrain):
        self.abbrev_dist = 10.0 # km
        self.R = 6371.0
        
        print('Loading file', fname_slab)
        a = np.loadtxt(fname_slab, delimiter=',')
        af = a[~np.isnan(a).any(axis=1), :] # lon lat depth

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
    
    def normalize_in_given_box(self, af, arr_in):
        data = copy.deepcopy(af) # this should be unnecessary
        arr = copy.deepcopy(arr_in)
        # Determine bounding box
        minL1, maxL1 = np.min(data[:,0]), np.max(data[:,0])
        minL2, maxL2 = np.min(data[:,1]), np.max(data[:,1])
        minD, maxD   = np.min(data[:,2]), np.max(data[:,2])

        # Determine intervals
        dL1, dL2, dD = np.abs(maxL1 - minL1), np.abs(maxL2 - minL2), np.abs(maxD - minD)

        # Normalize lat/lon/z data
        arr[:,0] -= minL1
        arr[:,0] /= dL1
        arr[:,1] -= minL2
        arr[:,1] /= dL2
        if arr.shape[1] > 2:
            if np.sign(minD) > 0:
                arr[:,2] -= minD
            elif np.sign(minD) < 0:
                arr[:,2] += np.abs(minD)
            arr[:,2] /= dD
        return arr
    
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
    
    def great_circle(self, lat_1, lon_1, lat_2, lon_2):
        """
        Calculate the great circle distance between two points on a globe.
        
        Parameters:
        -----------
            lat_1 : float
                Latitude of first point.
            lon_1 : float
                Longitude of first point.
            lat_2 : float
                Latitude of second point.
            lon_2 : float
                Longitude of second point.
                
        Returns:
        --------
            a : float
                Great circle distance in degrees.
        """
        
        # First we have to convert the latitudes to colatitudes:
        colat_1, colat_2 = 90. - lat_1, 90. - lat_2
        # and alpha is the difference betwee the two longitudes
        alpha = lon_2 - lon_1
        # Then lets make life easy on us and convert degrees to radians
        colat_1, colat_2, alpha = np.radians(colat_1),\
                np.radians(colat_2), np.radians(alpha)# continued line from above
        # From spherical trig we know that:
        cosa = np.cos(colat_1) * np.cos(colat_2) + np.sin(colat_1) * np.sin(colat_2) * np.cos(alpha)
        # Solve for a.
        a = np.arccos(cosa) # Take the arc cosine of cosa.
        # Remember to convert back to degrees!  
        return 111*np.degrees(a) # Return the great circle distance in degrees.  
    
    def great_circle_navpaths(self, lat_1, lon_1, lat_2, lon_2):
        '''
        Parameters:
        -----------
            lat_1 : float
                Latitude of first point.
            lon_1 : float
                Longitude of first point.
            lat_2 : float
                Latitude of second point.
            lon_2 : float
                Longitude of second point.
        '''
        lat_1, lon_1, lat_2, lon_2 = np.radians(lat_1), np.radians(lon_1), np.radians(lat_2), np.radians(lon_2)

        e_u1 = np.array([ np.cos(lat_1)* np.cos(lon_1), np.cos(lat_1)* np.sin(lon_1) , np.sin(lat_1)])
        e_u2 = np.array([ np.cos(lat_2)* np.cos(lon_2), np.cos(lat_2)* np.sin(lon_2) , np.sin(lat_2)])

        cos_theta = np.dot(e_u1, e_u2)
        theta = np.arccos(cos_theta)
        dist = theta*self.R

        return dist
    
    def get_point_normal(self,profile,slab_thickness):
        df = (profile[1,:] - profile[0,:])
        dn = df[::-1]*np.array([1,-1])
        pn = (slab_thickness/np.linalg.norm(dn))*dn
        return pn
    
    def write_slice_to_geo(self,profile,pn,filename,geo_info,write_msh,adjust_depths):
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
                line = 'Point(' + str(int(k)) + ') = {' + str(profile[k,0]) + ', ' + str(profile[k,1]) + ', ' + str(0.0) + ', h_fine};'
                f.write(line); f.write('\n')
            # write BSpline(1)
            spline_line = 'BSpline(1) = ' + str({*np.arange(0,np.shape(profile)[0])}) + ';'
            f.write(spline_line); f.write('\n')

            # line = 'Point(' + str(int(k+1)) + ') = {' + str(profile[0,0]) + ', ' + str(profile[0,1]) + '-slab_thickness, ' + str(0.0) + ', ' + str(h[0]) + '};'
            line = 'Point(' + str(int(k+1)) + ') = {' + str(pn[0]) + ', ' + str(pn[1]) + ', ' + str(0.0) + ', h_fine};'
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
            f.write('BSpline(newc) = {' + str(sorted({*np.arange(0,corner_pt+1)}))[1:-1] + '};'); f.write('\n')
            # breakpoint()


            f.write('BSpline(newc) = {' + str(sorted({*np.arange(corner_pt,np.shape(profile)[0])}))[1:-1] + '};'); f.write('\n')
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
        
    def plotting_slices_map_cartopy(self,start_points_lon_lat_arr,end_points_lon_lat_arr,profiles_lon_lat,labels,label_offset,fig_name):
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
        ax.set_extent([np.min(self.af[:,0]), np.max(self.af[:,0]), np.min(self.af[:,1]), np.max(self.af[:,1])],crs=ccrs.PlateCarree())

        cb = plt.colorbar(fig1,shrink=0.6)
        cb.solids.set(alpha=1)
        cb.set_alpha(1)
        cb.set_label(label='Depth (km)', fontsize=fs)
        cb.ax.tick_params(labelsize=fs)

        for k in range(len(labels)):
            fig1 = ax.plot(profiles_lon_lat[k][:,0],profiles_lon_lat[k][:,1],c='k', transform=ccrs.PlateCarree(),zorder=1)
            fig1 = ax.text(start_points_lon_lat_arr[k][0]+label_offset[0],start_points_lon_lat_arr[k][1]+label_offset[1],labels[k], transform=ccrs.PlateCarree(),zorder=3)

        ax.coastlines();
        ax.set_aspect('equal')
        plt.savefig(fig_name)
        plt.show()

    def convert_geographic_to_cartesian(self, geo_in):
        '''
        geo_in: (n, 3) array of lon, lat, depth 
                e.g. from Slab2 
        '''
        
        geo = copy.deepcopy(geo_in)
        lon = geo[:,0] * np.pi/180.0
        lat = geo[:,1] * np.pi/180.0
        r = self.R + geo[:,2]

        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)

        xyz = np.vstack([x,y,z]).T
        print('dimension', xyz.shape)
        return xyz 

    def compute_gc_normal(self, lon_1, lat_1, lon_2, lat_2):
        lat_1, lon_1, lat_2, lon_2 = np.radians(lat_1), np.radians(lon_1), np.radians(lat_2), np.radians(lon_2)

        e_u1 = np.array([ np.cos(lat_1)* np.cos(lon_1), np.cos(lat_1)* np.sin(lon_1) , np.sin(lat_1)])
        e_u2 = np.array([ np.cos(lat_2)* np.cos(lon_2), np.cos(lat_2)* np.sin(lon_2) , np.sin(lat_2)])

        normal = np.cross(e_u1, e_u2)
        normal = normal/np.linalg.norm(normal)
        return normal, e_u1, e_u2

    def first_rotation(self,A,B):
        G = np.array([ [np.dot(A,B), -np.linalg.norm(np.cross(A,B)), 0 ], \
                       [np.linalg.norm(np.cross(A,B)), np.dot(A,B), 0], \
                       [0, 0, 1]])
        F = np.linalg.inv(np.array([ A, (B - (np.dot(A,B)*A))/np.linalg.norm(B - (np.dot(A,B)*A)), np.cross(A, B) ]).T)
        U = np.matmul(np.matmul(np.linalg.inv(F), G), F)
        return U

    def sample_along_greatcircle(self, sp, ep):
        g = pyproj.Geod(ellps='sphere')
        (az12, az21, dist) = g.inv(sp[0], sp[1], ep[0], ep[1])
        lonlats = g.npts(sp[0], sp[1], ep[0], ep[1], 1 + int(dist / 1000))
        profile_lat_lon = np.array(lonlats)

        if np.any(profile_lat_lon[:,0] < 0):
            profile_lat_lon[:,0] += 360.0

        profile_lat_lon = np.vstack([sp, profile_lat_lon, ep])
        return profile_lat_lon

    def run_generate_mesh(self,geo_filename,geo_info,start_point_latlon,end_point_latlon,slab_thickness,plot_verbose,write_msh=False,adjust_depths=False,choose_vaxis='plane'):

        profile_lat_lon = self.sample_along_greatcircle(start_point_latlon, end_point_latlon)

        # normalize those points
        norm_profile = self.normalize_in_given_box(self.af, profile_lat_lon)

        # evaluate RBF along those points
        ds = self.create_RBF(self.slab_norm,norm_profile[:,0],norm_profile[:,1],coarse=10)

        profile_geog = self.profile_rescale_lat_lon(self.af,norm_profile[:,0],norm_profile[:,1],ds) 

        # abbreviate profile where it goes beyond Slab2 data
        tree = KDTree(self.af)
        dist, I = tree.query(profile_geog)
        profile_geog = profile_geog[dist < self.abbrev_dist]
        
        # convert to xyz
        profile_xyz = self.convert_geographic_to_cartesian(profile_geog)

        # compute great circle normal
        gc_norm, e_u1, e_u2 = self.compute_gc_normal(start_point_latlon[0], start_point_latlon[1], end_point_latlon[0], end_point_latlon[1])

        # align xy plane with the plane of the great circle
        A = np.array([0,0,1])

        U = self.first_rotation(gc_norm, A)

        profile_prime = np.nan*np.ones(profile_xyz.shape)
        for k in range(profile_prime.shape[0]):
            profile_prime[k, :] = np.matmul(U, profile_xyz[k,:])

        vec_trench = profile_prime[0,:]/np.linalg.norm(profile_prime[0,:])
        B = np.array([1.0, 0.0, 0.0])
        U2 = self.first_rotation(vec_trench, B)
        profile_prpr = np.nan*np.ones(profile_prime.shape)
        for k in range(profile_prpr.shape[0]):
            profile_prpr[k, :] = np.matmul(U2, profile_prime[k,:])

        shifted = profile_prpr[:,0:2]
        shifted = shifted[:,::-1]
        shifted[:,0] = np.abs(shifted[:,0])
        if choose_vaxis == 'plane':
            self.vertical_shift = 0
        shifted[:,1] -= shifted[0,1]

        # if using plane y axis
        profile_plane = shifted
        
        # if using slab2 depths as vertical axis
        profile_slab2 = np.vstack([shifted[:,0], profile_geog[:,-1]]).T
        if choose_vaxis == 'slab2':
            self.vertical_shift = profile_slab2[0,1]
        profile_slab2[:,1] -= profile_slab2[0,1]
        

        surface, surface_ll = self.run_surface(start_point_latlon, end_point_latlon, self.af)
        print('Max depth of surface', np.min(surface[:,1]))

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.plot(profile_plane[:,0], profile_plane[:,1], label='Plane', lw=2)
        ax.plot(profile_slab2[:,0], profile_slab2[:,-1], label='Slab2', lw=2, linestyle='dashed')
        ax.axhline(0.0, color='lightgray')
        ax.plot(surface[:,0], surface[:,1], color='green', label='Surface')
        ax.set_aspect('equal')
        plt.legend(fontsize=18)
        plt.grid(visible=True, which='major')
        plt.xlabel("x' (km)")
        plt.ylabel("y' (km)")
        plt.savefig("profile_with_surface.png", dpi=600)
        plt.show()

        # choose which coordinates to use
        if choose_vaxis == 'slab2':
            profile = profile_slab2
        elif choose_vaxis == 'plane':
            profile = profile_plane
        else:
            raise ValueError('Argument choose_vaxis must be "slab2" or "plane". ')

        pn = self.get_point_normal(profile,slab_thickness)
        self.write_slice_to_geo(profile,pn,geo_filename,geo_info,write_msh=write_msh,adjust_depths=adjust_depths)

        return profile, profile_lat_lon
    
    def run_surface(self, start_point_latlon, end_point_latlon, box):

        profile_lat_lon = self.sample_along_greatcircle(start_point_latlon, end_point_latlon)

        # normalize those points
        norm_profile = self.normalize_in_given_box(box, profile_lat_lon)

        ds = np.zeros(profile_lat_lon.shape[0])

        profile_geog = self.profile_rescale_lat_lon(box,norm_profile[:,0],norm_profile[:,1],ds) 
        # profile_geog = np.vstack([profile_lat_lon.T, ds]).T
        
        # convert to xyz
        profile_xyz = self.convert_geographic_to_cartesian(profile_geog)

        # compute great circle normal
        gc_norm, e_u1, e_u2 = self.compute_gc_normal(start_point_latlon[0], start_point_latlon[1], end_point_latlon[0], end_point_latlon[1])

        # align xy plane with the plane of the great circle
        A = np.array([0,0,1])

        U = self.first_rotation(gc_norm, A)

        profile_prime = np.nan*np.ones(profile_xyz.shape)
        for k in range(profile_prime.shape[0]):
            profile_prime[k, :] = np.matmul(U, profile_xyz[k,:])

        vec_trench = profile_prime[0,:]/np.linalg.norm(profile_prime[0,:])
        B = np.array([1.0, 0.0, 0.0])
        U2 = self.first_rotation(vec_trench, B)
        profile_prpr = np.nan*np.ones(profile_prime.shape)
        for k in range(profile_prpr.shape[0]):
            profile_prpr[k, :] = np.matmul(U2, profile_prime[k,:])

        shifted = profile_prpr[:,0:2]
        shifted = shifted[:,::-1]
        shifted[:,0] = np.abs(shifted[:,0])
        shifted[:,1] -= shifted[0,1]

        # if using plane y axis
        profile = shifted

        return profile, profile_lat_lon
    
    def run_testA(self):
        '''
        Test for a profile with points running along a single meridian, longitude 90, 
        with latitude between 20 and 30.
        The points are assumed to lie on the surface, depth = 0. 
        '''
        start_point_latlon = np.array([90.0, 25.0])
        end_point_latlon = np.array([90.0, 30.0])
        box = np.array([[80.0, 20.0, 0.0],[100.0, 35.0, 0.0]])

        profile_lat_lon = self.sample_along_greatcircle(start_point_latlon, end_point_latlon)

        plt.figure()
        plt.plot(profile_lat_lon[:,0], profile_lat_lon[:,1])
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title("Test A input")
        plt.show()

        ds = np.zeros(profile_lat_lon.shape[0])
        profile_geog = np.vstack([profile_lat_lon.T, ds]).T
        
        # convert to xyz
        profile_xyz = self.convert_geographic_to_cartesian(profile_geog)

        print('Summed Euclidean distance between xyz points')
        distA = 0
        for k in range(profile_xyz.shape[0]-1):
            distA += np.linalg.norm(profile_xyz[k+1,:] - profile_xyz[k,:])
        print(distA)

        # compute great circle normal
        gc_norm, e_u1, e_u2 = self.compute_gc_normal(start_point_latlon[0], start_point_latlon[1], end_point_latlon[0], end_point_latlon[1])

        # align xy plane with the plane of the great circle
        A = np.array([0,0,1])

        U = self.first_rotation(gc_norm, A)

        profile_prime = np.nan*np.ones(profile_xyz.shape)
        for k in range(profile_prime.shape[0]):
            profile_prime[k, :] = np.matmul(U, profile_xyz[k,:])

        vec_trench = profile_prime[0,:]/np.linalg.norm(profile_prime[0,:])
        B = np.array([1.0, 0.0, 0.0])
        U2 = self.first_rotation(vec_trench, B)
        profile_prpr = np.nan*np.ones(profile_prime.shape)
        for k in range(profile_prpr.shape[0]):
            profile_prpr[k, :] = np.matmul(U2, profile_prime[k,:])

        shifted = profile_prpr[:,0:2]
        shifted = shifted[:,::-1]
        shifted[:,0] = np.abs(shifted[:,0])
        shifted[:,1] -= shifted[0,1]

        # if using plane y axis
        profile_plane = shifted
        
        surface, surface_ll = self.run_surface(start_point_latlon, end_point_latlon, box)

        plt.figure()
        plt.plot(profile_plane[:,0], profile_plane[:,1], label='profile')
        plt.plot(surface[:,0], surface[:,1], linestyle='dashed', label='surface')
        plt.xlabel("x' (km)")
        plt.ylabel("y' (km)")
        plt.title("Test A output")
        plt.show()

        print('Summed Euclidean distance between plane points')
        distB = 0
        for k in range(profile_plane.shape[0]-1):
            distB += np.linalg.norm(profile_plane[k+1,:] - profile_plane[k,:])
        print(distB)

        self.vertical_shift = 0.0

        if np.abs(distA - distB) < 1:
            check = True
        else:
            check = False

        return check, profile_plane, profile_lat_lon
    
    def run_testB(self):
        '''
        Test for points in x,y,z space that run from the surface at (R*cos(45 deg), 0, 0 ) to the origin. 
        '''

        profile_xyz = np.vstack([np.linspace(self.R*np.cos(np.deg2rad(45)), 0, 100), np.zeros(100), np.zeros(100)]).T

        plt.figure()
        plt.plot(profile_xyz[:,0], profile_xyz[:,2])
        plt.xlabel('x (km)')
        plt.ylabel('z (km)')
        plt.title("Test B input")
        plt.show()

        print('Summed Euclidean distance between xyz points')
        distA = 0
        for k in range(profile_xyz.shape[0]-1):
            distA += np.linalg.norm(profile_xyz[k+1,:] - profile_xyz[k,:])
        print(distA)

        # compute great circle normal
        # gc_norm, e_u1, e_u2 = self.compute_gc_normal(start_point_latlon[0], start_point_latlon[1], end_point_latlon[0], end_point_latlon[1])
        gc_norm = np.array([0,1,0])
        # align xy plane with the plane of the great circle
        A = np.array([0,0,1])
        U = self.first_rotation(gc_norm, A)

        profile_prime = np.nan*np.ones(profile_xyz.shape)
        for k in range(profile_prime.shape[0]):
            profile_prime[k, :] = np.matmul(U, profile_xyz[k,:])

        vec_trench = profile_prime[0,:]/np.linalg.norm(profile_prime[0,:])
        B = np.array([1.0, 0.0, 0.0])

        if np.abs(np.dot(vec_trench,B)) - 1 > 1e-12:
            U2 = self.first_rotation(vec_trench, B)
            profile_prpr = np.nan*np.ones(profile_prime.shape)
            for k in range(profile_prpr.shape[0]):
                profile_prpr[k, :] = np.matmul(U2, profile_prime[k,:])
        else:
            profile_prpr = profile_prime

        shifted = profile_prpr[:,0:2]
        shifted = shifted[:,::-1]
        shifted[:,0] = np.abs(shifted[:,0])
        shifted[:,1] -= shifted[0,1]

        # if using plane y axis
        profile_plane = shifted
        
        print('Summed Euclidean distance between plane points')
        distB = 0
        for k in range(profile_plane.shape[0]-1):
            distB += np.linalg.norm(profile_plane[k+1,:] - profile_plane[k,:])
        print(distB)

        plt.figure()
        plt.plot(profile_plane[:,0], profile_plane[:,1])
        plt.xlim([np.min(profile_plane[:,0]) - 1.0, np.max(profile_plane[:,0])+1.0])
        plt.xlabel("x' (km)")
        plt.ylabel("y' (km)")
        plt.title("Test B output")
        plt.show()

        if np.abs(distA - distB) < 1:
            check = True
        else:
            check = False

        profile_lat_lon = np.zeros((profile_plane.shape[0], 2))
        self.vertical_shift = 0

        return check, profile_plane, profile_lat_lon
    
    def run_testC(self):
        '''
        Test for points in x,y,z space that run from the surface at (R*cos(45 deg), 0, R*cos(45 deg) ) to the origin. 
        '''

        profile_xyz = np.vstack([np.linspace(self.R*np.cos(np.deg2rad(45)), 0, 100), np.zeros(100), np.linspace(self.R*np.cos(np.deg2rad(45)), 0, 100)]).T

        plt.figure()
        plt.plot(profile_xyz[:,0], profile_xyz[:,2])
        plt.xlabel('x (km)')
        plt.ylabel('z (km)')
        plt.title("Test C input")
        plt.show()


        print('Summed Euclidean distance between xyz points')
        distA = 0
        for k in range(profile_xyz.shape[0]-1):
            distA += np.linalg.norm(profile_xyz[k+1,:] - profile_xyz[k,:])
        print(distA)

        # compute great circle normal
        # gc_norm, e_u1, e_u2 = self.compute_gc_normal(start_point_latlon[0], start_point_latlon[1], end_point_latlon[0], end_point_latlon[1])
        gc_norm = np.array([0,1,0])
        # align xy plane with the plane of the great circle
        A = np.array([0,0,1])

        U = self.first_rotation(gc_norm, A)

        profile_prime = np.nan*np.ones(profile_xyz.shape)
        for k in range(profile_prime.shape[0]):
            profile_prime[k, :] = np.matmul(U, profile_xyz[k,:])

        vec_trench = profile_prime[0,:]/np.linalg.norm(profile_prime[0,:])
        B = np.array([1.0, 0.0, 0.0])
        U2 = self.first_rotation(vec_trench, B)
        profile_prpr = np.nan*np.ones(profile_prime.shape)
        for k in range(profile_prpr.shape[0]):
            profile_prpr[k, :] = np.matmul(U2, profile_prime[k,:])

        shifted = profile_prpr[:,0:2]
        shifted = shifted[:,::-1]
        shifted[:,0] = np.abs(shifted[:,0])
        shifted[:,1] -= shifted[0,1]

        # if using plane y axis
        profile_plane = shifted
        
        print('Summed Euclidean distance between plane points')
        distB = 0
        for k in range(profile_plane.shape[0]-1):
            distB += np.linalg.norm(profile_plane[k+1,:] - profile_plane[k,:])
        print(distB)

        plt.figure()
        plt.plot(profile_plane[:,0], profile_plane[:,1])
        plt.xlim([np.min(profile_plane[:,0]) - 1.0, np.max(profile_plane[:,0])+1.0])
        plt.xlabel("x' (km)")
        plt.ylabel("y' (km)")
        plt.title("Test C output")
        plt.show()

        if np.abs(distA - distB) < 1:
            check = True
        else:
            check = False

        profile_lat_lon = np.zeros((profile_plane.shape[0], 2))
        self.vertical_shift = 0

        return check, profile_plane, profile_lat_lon
    

    def run_testD(self):
        '''
        Test for points in x,y,z space that run from the surface at (R*cos(45 deg), 0, R*cos(45 deg) ) to (R*cos(45 deg), 0, 0 )
        '''
        profile_xyz = np.vstack([np.linspace(self.R*np.cos(np.deg2rad(45)), self.R*np.cos(np.deg2rad(45)), 100), np.zeros(100), np.linspace(self.R*np.cos(np.deg2rad(45)), 0, 100)]).T

        plt.figure()
        plt.plot(profile_xyz[:,0], profile_xyz[:,2])
        # plt.xlim([np.min(profile_plane[:,0]) - 1.0, np.max(profile_plane[:,0])+1.0])
        plt.xlabel('x (km)')
        plt.ylabel('z (km)')
        plt.title("Test D input")
        # plt.savefig("testD_profile_input.png")
        plt.show()

        print('Summed Euclidean distance between xyz points')
        distA = 0
        for k in range(profile_xyz.shape[0]-1):
            distA += np.linalg.norm(profile_xyz[k+1,:] - profile_xyz[k,:])
        print(distA)

        # compute great circle normal
        # gc_norm, e_u1, e_u2 = self.compute_gc_normal(start_point_latlon[0], start_point_latlon[1], end_point_latlon[0], end_point_latlon[1])
        gc_norm = np.array([0,1,0])
        # e_u1 = np.array([1,0,0])
        # e_u2 = np.array([0,0,1])
        # align xy plane with the plane of the great circle
        # breakpoint()
        A = np.array([0,0,1])

        U = self.first_rotation(gc_norm, A)
        # U = np.vstack([e_u1/np.linalg.norm(e_u1), \
        #                np.cross(e_u1, gc_norm)/np.linalg.norm(np.cross(e_u1, gc_norm)), \
        #                 gc_norm])

        # breakpoint()
        profile_prime = np.nan*np.ones(profile_xyz.shape)
        for k in range(profile_prime.shape[0]):
            profile_prime[k, :] = np.matmul(U, profile_xyz[k,:])
        # print(profile_prime)

        # breakpoint()
        vec_trench = profile_prime[0,:]/np.linalg.norm(profile_prime[0,:])
        B = np.array([1.0, 0.0, 0.0])
        U2 = self.first_rotation(vec_trench, B)
        profile_prpr = np.nan*np.ones(profile_prime.shape)
        for k in range(profile_prpr.shape[0]):
            profile_prpr[k, :] = np.matmul(U2, profile_prime[k,:])
        # profile_prpr = profile_prime
        
        # breakpoint()
        shifted = profile_prpr[:,0:2]
        shifted = shifted[:,::-1]
        shifted[:,0] = np.abs(shifted[:,0])
        shifted[:,1] -= shifted[0,1]

        # if using plane y axis
        profile_plane = shifted
        
        print('Summed Euclidean distance between plane points')
        distB = 0
        for k in range(profile_plane.shape[0]-1):
            distB += np.linalg.norm(profile_plane[k+1,:] - profile_plane[k,:])
        print(distB)

        plt.figure()
        plt.plot(profile_plane[:,0], profile_plane[:,1])
        plt.xlim([np.min(profile_plane[:,0]) - 1.0, np.max(profile_plane[:,0])+1.0])
        plt.xlabel("x' (km)")
        plt.ylabel("y' (km)")
        plt.title("Test D output")
        # plt.savefig("testD_profile_result.png")
        plt.show()

        if np.abs(distA - distB) < 1:
            check = True
        else:
            check = False

        profile_lat_lon = np.zeros((profile_plane.shape[0], 2))
        self.vertical_shift = 0

        return check, profile_plane, profile_lat_lon
    
    def run_testE(self):
        '''
        Testing coordinate transformations are correct.
        '''

        start_point_latlon = np.array([0.0,0.0])
        end_point_latlon = np.array([0.0, 90.0])
        # define points along the great circle between start_point_latlon and end_point_latlon
        # g = pyproj.Geod(ellps='sphere')
        # (az12, az21, dist) = g.inv(start_point_latlon[0], start_point_latlon[1], end_point_latlon[0], end_point_latlon[1])
        # lonlats = g.npts(start_point_latlon[0], start_point_latlon[1], end_point_latlon[0], end_point_latlon[1], 10)
        # lonlats.insert(0, (start_point_latlon[0], start_point_latlon[1]))
        # lonlats.append((end_point_latlon[0], end_point_latlon[1]))
        # profile_lat_lon = np.array(lonlats)

        profile_lat_lon = self.sample_along_greatcircle(start_point_latlon, end_point_latlon)

        print('Start point:', start_point_latlon)
        print('End point:', end_point_latlon)
        # print('Profile points between them:', profile_lat_lon)
        print('With zero depth, start point in xyz is:', self.convert_geographic_to_cartesian(np.atleast_2d(np.hstack([start_point_latlon, 0.0]))))
        print('With zero depth, end point in xyz is:', self.convert_geographic_to_cartesian(np.atleast_2d(np.hstack([end_point_latlon, 0.0]))))
        print('Check calculation of great circle plane normal is correct')
        gc_norm, e_u1, e_u2 = self.compute_gc_normal(start_point_latlon[0], start_point_latlon[1], end_point_latlon[0], end_point_latlon[1])
        print('gc_norm', gc_norm)

        print('-------------')
        print('Checking another pair of points:')
        start_point_latlon = np.array([90.0,-45.0])
        end_point_latlon = np.array([90.0, 45.0])
        # define points along the great circle between start_point_latlon and end_point_latlon
        g = pyproj.Geod(ellps='sphere')
        (az12, az21, dist) = g.inv(start_point_latlon[0], start_point_latlon[1], end_point_latlon[0], end_point_latlon[1])
        lonlats = g.npts(start_point_latlon[0], start_point_latlon[1], end_point_latlon[0], end_point_latlon[1], 10)
        lonlats.insert(0, (start_point_latlon[0], start_point_latlon[1]))
        lonlats.append((end_point_latlon[0], end_point_latlon[1]))
        profile_lat_lon = np.array(lonlats)

        print('Start point:', start_point_latlon)
        print('End point:', end_point_latlon)
        # print('Profile points between them:', profile_lat_lon)
        print('With zero depth, start point in xyz is:', self.convert_geographic_to_cartesian(np.atleast_2d(np.hstack([start_point_latlon, 0.0]))))
        print('With zero depth, end point in xyz is:', self.convert_geographic_to_cartesian(np.atleast_2d(np.hstack([end_point_latlon, 0.0]))))

        print('Check calculation of great circle plane normal is correct')
        gc_norm, e_u1, e_u2 = self.compute_gc_normal(start_point_latlon[0], start_point_latlon[1], end_point_latlon[0], end_point_latlon[1])
        print('gc_norm', gc_norm)

        return

    def run_testF(self):
        '''
        Test normalization and re-scaling process. 
        '''
        start_point_latlon = np.array([90.0, 25.0])
        end_point_latlon = np.array([90.0, 30.0])
        box = np.array([[80.0, 20.0, 0.0],[100.0, 35.0, 0.0]])

        profile_lat_lon = self.sample_along_greatcircle(start_point_latlon, end_point_latlon)
        
        norm_profile = self.normalize_in_given_box(box, profile_lat_lon)

        ds = np.zeros(profile_lat_lon.shape[0])

        profile_geog = self.profile_rescale_lat_lon(box,norm_profile[:,0],norm_profile[:,1],ds) 

        if (np.max(np.abs(profile_lat_lon - profile_geog[:,0:2])) < 1e-12):
            check = True
        else:
            check = False

        return check