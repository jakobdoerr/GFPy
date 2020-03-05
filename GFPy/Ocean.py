# -*- coding: utf-8 -*-
"""
This module contains functions that read and plot CTD data.
"""
 
from seabird.cnv import fCNV
import gsw
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import interp1d,griddata
from scipy.io import loadmat
from matplotlib.dates import date2num
import cmocean
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

############################################################################
#MISCELLANEOUS FUNCTIONS
############################################################################
def create_latlon_text(lat,lon):
    '''
    Creates two strings which contain a text for latitude and longitude

    Parameters
    ----------
    lat : scalar
        latitude.
    lon : scalar
        longitude.

    Returns
    -------
    latstring : str
        the string for the latitude.
    lonstring : str
        the string for the longitude.

    '''
    lat_minutes = str(np.round((np.abs(lat - int(lat)))*60,5))
    if lat < 0:
        lat_letter = 'S'
    else:
        lat_letter = 'N'
    latstring = str(int(np.abs(lat)))+ ' ' + lat_minutes + ' ' + lat_letter
    
    lon_minutes = str(np.round((np.abs(lon - int(lon)))*60,5))
    if lon < 0:
        lon_letter = 'W'
    else:
        lon_letter = 'E'
    lonstring = str(int(np.abs(lon)))+ ' ' + lon_minutes + ' ' + lon_letter
    
    return latstring,lonstring

def CTD_to_grid(CTD,stations=None,interp_opt= 1,x_type='distance'):
    '''
    This function accepts a CTD dict of dicts, finds out the maximum 
    length of the depth vectors for the given stations, and fills all
    fields to that maximum length, using np.nan values. 

    Parameters
    ----------
    CTD : dict of dicts
        CTD data. Is created by `read_CTD`
    stations : array_like, optional
        list of stations to select from `CTD`.
    interp_opt : int, optional
        flag how to interpolate over X (optional). 
                     0: no interpolation,
                     1: linear interpolation, fine grid (default),
                     2: linear interpolation, coarse grid. The default is 1.
    x_type : str, optional
        whether X is 'time' or 'distance'. The default is 'distance'.

    Returns
    -------
    fCTD : dict
        dict with the gridded CTD data.
    Z : array_like
        common depth vector.
    X : array_like
        common X vector.
    station_locs : array_like
        locations of the stations as X units.

    '''

    # if no stations are given, take all stations available
    if stations is None:
        stations = list(CTD.keys())
        
    # construct the Z-vector from the max and min depth of the given stations
    maxdepth = np.nanmax([np.nanmax(-CTD[i]['z']) for i in stations])
    mindepth = np.nanmin([np.nanmin(-CTD[i]['z']) for i in stations])
    Z = np.linspace(mindepth,maxdepth,int(maxdepth-mindepth)+1)
    
    # construct the X-vector, either distance or time
    if x_type == 'distance':
        LAT = np.asarray([d['LAT'] for d in CTD.values()])
        LON = np.asarray([d['LON'] for d in CTD.values()])
        X = np.insert(np.cumsum(gsw.distance(LON,LAT)/1000),0,0)
    elif x_type == 'time':
        X = np.array([date2num(d['datetime']) for d in CTD.values()])
        X = (X - X[0])*24
        
    # this X vector is where the stations are located, so save that
    station_locs = X
    fields = set([field for field in CTD[stations[0]] 
                        if np.size(CTD[stations[0]][field]) > 1])
    
    fCTD = {}
    if interp_opt == 0: # only grid over depth
        for field in fields:
            fCTD[field] = np.array([interp1d(-value['z'],value[field],
                                             bounds_error=False)(Z)
                            for value in CTD.values()]).transpose()
    
    elif interp_opt == 1: # grid over depth and x (time or distance)
        X_fine = np.linspace(np.min(X),np.max(X),len(X)*20) # create fine X grid
        # original grids
        X_orig,Z_orig = [f.ravel() for f in np.meshgrid(X,Z)] 
        for field in fields:
            # grid in Z
            temp_array = np.array([interp1d(-value['z'],value[field],
                                            bounds_error=False)(Z)
                            for value in CTD.values()]).transpose().ravel()
            mask = np.where(~np.isnan(temp_array)) # NaN mask

            # grid in X
            fCTD[field] = griddata((X_orig[mask],Z_orig[mask]), # old grid
                                    temp_array[mask], # data
                                    tuple(np.meshgrid(X_fine,Z))) # new grid
        X = X_fine
        
    elif interp_opt == 2: # grid over depth and x, use coarse resolution
        X_coarse = np.linspace(np.min(X),np.max(X),20) # create coarse X grid
        Z_coarse = np.linspace(mindepth,maxdepth,50)
        # original grids
        X_orig,Z_orig = [f.ravel() for f in np.meshgrid(X,Z)] 
        for field in fields:
            # grid in Z
            temp_array = np.array([interp1d(-value['z'],value[field],
                                            bounds_error=False)(Z)
                            for value in CTD.values()]).transpose().ravel()
            mask = np.where(~np.isnan(temp_array)) # NaN mask

            # grid in X
            fCTD[field] = griddata((X_orig[mask],Z_orig[mask]), # old grid
                                    temp_array[mask], # data
                                    tuple(np.meshgrid(X_coarse,Z_coarse))) # new grid
        X,Z = X_coarse,Z_coarse

        
    return fCTD,Z,X,station_locs
  
def calc_freshwater_content(salinity,depth,ref_salinity=34.8):
    '''
    

    Parameters
    ----------
    salinity : array-like
        The salinity vector.
    depth : TYPE
        The depth vector.
    ref_salinity : float, optional
        The reference salinity. The default is 34.8.

    Returns
    -------
    float
        The freshwater content for the profile, in meters

    '''
    salinity = np.mean([salinity[1:],salinity[:-1]])
    
    dz = np.diff(depth)
    
    return np.sum((salinity-ref_salinity)/ref_salinity *dz)
    
############################################################################
#READING FUNCTIONS
############################################################################
def read_CTD(inpath,cruise_name,outpath=None,stations=None,corr=(1.,0.)):
    '''
    This function reads in the CTD data from cnv files in `inpath`
    for the stations `stations` and returns a list of dicts containing
    the data. Conductivity correction (if any) can be specified in `corr`

    Parameters
    ----------
    inpath : str
        input path where the cnv files are stored.
    cruise_name : str
        name of the cruise.
    outpath : str, optional
        path where to store the output. The default is None.
    stations : array_like, optional
        list of stations to read in (optional). If not given, 
        the function will read all stations in `inpath`. The default is None.
    corr : tuple, optional
        tuple with 2 values containing (slope,intersect) of
                      linear correction model. The default is (1.,0.).

    Returns
    -------
    CTD_dict : dict
        a dict of dicts contaning the data for
                    all the relevant station data.

    '''
    # create a dict that converts the variable names in the cnv files to
    # the variable names used by us:
    var_names = {'PRES':'P','temperature':'T','t168C':'T2','CNDC':'C',
                 'CNDC2':'C2','PSAL':'S','PSAL2':'S2','oxygen_ml_L':'OX',
                 'flC':'flC','par':'PAR','altM':'Z','timeJ':'time',
                 'timeK':'time','timeS':'elapsedtime','longitude':'lon',
                 'latitude':'lat'}
    # get all CTD station files in inpath
    files = glob.glob(inpath+'*.cnv')
    #If stations are provided, select the ones that exist
    if stations is not None:
        use_files = [i for i in files for j in stations if str(j) in i]
        assert len(use_files) > 0, 'None of your provided stations exists!'
        if len(use_files) < len(stations): 
            print('Warning: Some stations you provided do not exist!')
        files = use_files
        
    files = sorted(files)
    
    # Read in the data, file by file
    CTD_dict = {}
    for file in files:
        # get all the fields, construct a dict with the fields
        profile = fCNV(file)
        p = {var_names[name]:profile[name] 
            for name in profile.keys() if name in var_names}
        
        # get the interesting header fields and append it to the dict
        p.update(profile.attrs)
            
        # rename the most important ones to the same convention used in MATLAB, 
        # add other important ones
        p['LAT'] = p.pop('LATITUDE')
        p['LON'] = p.pop('LONGITUDE')
        p['z'] = gsw.z_from_p(p['P'],p['LAT'])
        p['BottomDepth'] = np.round(np.nanmax(np.abs(p['z']))+8)
        p['C'][p['C']<1] = np.nan
        p['C'] = corr[0]*p['C'] + corr[1] # apply correction
        p['T'][p['T']<-2] = np.nan
        p['S'] = gsw.SP_from_C(p['C']*10,p['T'],p['P'])
        p['S'][p['S']<20] = np.nan
        p['C'][p['S']<20] = np.nan
        p['SA'] = gsw.SA_from_SP(p['S'],p['P'],p['LON'],p['LAT'])
        p['CT'] = gsw.CT_from_t(p['SA'],p['T'],p['P'])
        p['SIGTH'] = gsw.sigma0(p['SA'],p['CT'])
        p['st'] = int(p['filename'].split('.')[0][-4::])
        
        CTD_dict[p['st']]= p

    # save data if outpath was given    
    if outpath is not None:
        np.save(outpath+cruise_name+'_CTD',CTD_dict)
        
    return CTD_dict

def read_CTD_from_mat(matfile):
    '''
    Reads CTD data from matfile

    Parameters
    ----------
    matfile : str
        The full path to the .mat file. This should contain a struct with the
        name CTD. This is the common output style of the cruise matlab scripts.

    Returns
    -------
    CTD : dict
        The dictionary with the CTD Data.

    '''
    # read the raw data using scipy.io.loadmat
    raw_data = loadmat(matfile, squeeze_me=True, struct_as_record=False)['CTD']
    # convert to dictionary
    CTD = {}
    for record in raw_data:
        station = record.__dict__['st']
        CTD[station] = record.__dict__
        CTD[station].pop('_fieldnames',None)
        
    if 'note' in CTD[next(iter(CTD))]:
        print('Note: This CTD data is already calibrated.')
        
    return CTD
    
def read_mooring_from_mat(matfile):
    '''
    Read mooring data prepared in a .mat file.

    Parameters
    ----------
    matfile : str
        Full path to the .mat file.

    Returns
    -------
    raw_data : dict
        Dictionary with the mooring data.

    '''
    # read raw data using scipy.io.loadmat
    raw_data = loadmat(matfile, squeeze_me=True, struct_as_record=False)
    # find the name of the MATLAB struct
    variable_name = list(raw_data.keys())[-1]
    # convert to dictionary, remove _fieldnames (internal)
    raw_data = raw_data[variable_name].__dict__
    raw_data.pop('_fieldnames',None)
    # extract the units dictionary
    raw_data['units'] = raw_data['units'].__dict__
    # extract the raw data for the instruments
    raw_data['ins'] = [value.__dict__ for value in raw_data['ins']]
    # extract processed data
    raw_data['mat'] = raw_data['mat'].__dict__
    
    return raw_data
############################################################################
#PLOTTING FUNCTIONS
############################################################################
def contour_section(X,Y,Z,Z2=None,ax=None,station_pos=None,cmap='jet',Z2_contours=None,
                    clabel='',bottom_depth=None,clevels=20,station_text=''):
    '''    
    Plots a filled contour plot of *Z*, with contourf of *Z2* on top to 
    the axes *ax*. It also displays the position of stations, if given in
    *station_pos*, adds labels to the contours of Z2, given in 
    *Z2_contours*. If no labels are given, it assumes Z2 is density (sigma0) 
    and adds its own labels. It adds bottom topography if given in *bottom_depth*.
    
    Parameters
    ----------
    X : (N,K) array_like
        X-values.
    Y : (N,K) array_like
        Y-values.
    Z : (N,K) array_like
        the filled contour field.
    Z2 : (N,K) array_like, optional
        the contour field on top. The default is None.
    ax : plot axes, optional
        axes object to plot on. The default is the current axes.
    station_pos : (S,) array_like, optional
        the station positions. The default is None (all stations are plotted).
    cmap : str or array_like, optional
        the colormap for the filled contours. The default is 'jet'.
    Z2_contours : array_like, optional
        the contour label positions for `Z2`str. The default is None.
    clabel : str, optional
        label to put on the colorbar. The default is ''.
    bottom_depth : (S,) array_like, optional
        list with bottom depth. The default is None.
    clevels : array_like or number, optional
        list of color levels, or number of levels to use for `Z`. 
        The default is 20.
    station_text : str, optional
        Name to label the station locations. Can be the Section Name for 
        instance. The default is ''.

    Returns
    -------
    ax : plot axes
        The axes of the plot.

    '''
    # open new figure and get current axes, if none is provided
    if ax is None:
        ax = plt.gca()
        
    # get the labels for the Z2 contours
    if Z2 is not None and Z2_contours is None:
        Z2_contours = np.concatenate([list(range(21,26)),np.arange(25.5,29,0.2)])
        Z2_contours = [i for i in Z2_contours 
                        if np.nanmin(Z2) < i < np.nanmax(Z2)]
    
    # get the Y-axis limits 
    y_limits = (0,np.nanmax(Y))
        
    cT = ax.contourf(X,Y,Z,cmap=cmap,levels=clevels) # draw Z
    if Z2 is not None:
        cSIG = ax.contour(X,Y,Z2,levels = Z2_contours,
                           colors='k',linewidths=[1],alpha=0.6) # draw Z2
        clabels = plt.clabel(cSIG, Z2_contours,fontsize=8,fmt = '%1.1f') # add contour labels
        [txt.set_bbox(dict(facecolor='white', edgecolor='none',
                           pad=0,alpha=0.6)) for txt in clabels]
    plt.colorbar(cT,ax = ax,label=clabel,pad=0.01) # add colorbar
    ax.set_ylim(y_limits)
    ax.invert_yaxis()
    
    # add bathymetry
    if bottom_depth is not None:
        # make sure bottom_depth is an np.array
        bottom_depth = np.asarray(bottom_depth)
        
        ax.fill_between(station_pos,bottom_depth*0+y_limits[1]+10,bottom_depth,
                     zorder=999,color='gray')
       
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    
    # add station ticks
    if station_pos is not None:
        for i,pos in enumerate(station_pos):
            ax.text(pos,0,'v',ha='center',fontweight='bold')
            if station_text != '':
                ax.annotate(station_text+str(i+1),(pos,0),xytext=(0,10),
                        textcoords='offset points',ha='center')
    
def plot_CTD_section(CTD,stations,section_name='',cruise_name = '',
                     x_type='distance'):
    '''
    This function plots a CTD section of Temperature and Salinity,
    given CTD data either directly (through `CTD`) or via a file (through)
    `infile`.
    
    Parameters
    ----------
    CTD : str or dict
        Either a dict of dicts containing the CTD data, which can be made with
              the function read_CTD. Or a str with a file where the dict is stored
    stations : array_like
        stations to plot (station numbers have to be found inside the CTD data!).
    section_name : str, optional
        name of the Section, will appear in the plot title. The default is ''.
    cruise_name : str, optional
        name of the Cruise, will also appear in the title. The default is ''.
    x_type : str, optional
        Wheter to use 'distance' or 'time' as the x-axis. The default is 'distance'.

    Returns
    -------
    None.

    '''
    # Check if the function has data to work with
    assert type(CTD) in [dict,str], 'Parameter *CTD*: You must provide either\n'\
            ' a) a data dict or \n b) a npy file string with the data !'
    
    # read in the data (only needed if no CTD-dict, but a file was given)
    if type(CTD) is str:
        print('reading file...')
        CTD = np.load(CTD,allow_pickle=True).item()
        
    # Check if all stations given are found in the data
    assert min([np.isin(st,list(CTD.keys())) for st in stations]), 'Not all '\
            'of the provided stations were found in the CTD data! \n'\
            'The following stations were not found in the data: '\
            +''.join([str(st)+' ' for st in stations if ~np.isin(st,list(CTD.keys()))])
    # Check if x_type is either distance or time
    assert x_type in ['distance','time'], 'x_type must be eigher distance or '\
            'time!'
            
    
    
    # select only the given stations in the data
    CTD = {key:CTD[key] for key in stations}
    
    # extract Bottom Depth    
    BDEPTH = np.asarray([d['BottomDepth'] for d in CTD.values()])

    # put the fields (the vector data) on a regular, common pressure and X grid
    # by interpolating. 
    fCTD,Z,X,station_locs = CTD_to_grid(CTD,x_type=x_type)
    
    # plot the figure
    fig,[axT,axS] = plt.subplots(2,1,figsize=(8,9))
  
    # Temperature
    contour_section(X,Z,fCTD['T'],fCTD['SIGTH'],ax = axT,
                          station_pos=station_locs,cmap=cmocean.cm.thermal,
                          clabel='Temperature [˚C]',bottom_depth=BDEPTH,
                          station_text=section_name)
    # Salinity
    contour_section(X,Z,fCTD['S'],fCTD['SIGTH'],ax=axS,
                          station_pos=station_locs,cmap=cmocean.cm.haline,
                          clabel='Salinity [g kg$^{-1}$]',bottom_depth=BDEPTH)    
    # Add x and y labels
    axT.set_ylabel('Depth [m]')
    axS.set_ylabel('Depth [m]')
    if x_type == 'distance':
        axS.set_xlabel('Distance [km]')
    else:
        axS.set_xlabel('Time [h]')
        
    # add title
    fig.suptitle(cruise_name+' Section '+section_name,fontweight='bold')
    
    # tight_layout
    fig.tight_layout(h_pad=0.1,rect=[0,0,1,0.95])
    
def plot_CTD_single_section(CTD,stations,section_name='',cruise_name = '',
                     x_type='distance',parameter='T',clabel='Temperature [˚C]',
                     cmap=cmocean.cm.thermal):
    '''
    This function plots a CTD section of a chosen variable,
    given CTD data either directly (through `CTD`) or via a file (through)
    `infile`.

    Parameters
    ----------
    CTD : str or dict
        Either a dict of dicts containing the CTD data, which can be made with
              the function read_CTD. Or a str with a file where the dict is stored
    stations : array_like
        stations to plot (station numbers have to be found inside the CTD data!).
    section_name : str, optional
        name of the Section, will appear in the plot title. The default is ''.
    cruise_name : str, optional
        name of the Cruise, will also appear in the title. The default is ''.
    x_type : str, optional
        Wheter to use 'distance' or 'time' as the x-axis. The default is 'distance'.
    parameter : str, optional
        Which parameter to plot. Check what parameters are available
        in `CTD`. The default is 'T'.
    clabel : str, optional
        The label on the colorbar axis. The default is 'Temperature [˚C]'.
    cmap : array-like or str, optional
        The colormap to be used. The default is cmocean.cm.thermal.

    Returns
    -------
    None.

    '''
    # Check if the function has data to work with
    assert type(CTD) in [dict,str], 'Parameter *CTD*: You must provide either\n'\
            ' a) a data dict or \n b) a npy file string with the data !'
    
    # read in the data (only needed if no CTD-dict, but a file was given)
    if type(CTD) is str:
        print('reading file...')
        CTD = np.load(CTD,allow_pickle=True).item()
        
    # Check if all stations given are found in the data
    assert min([np.isin(st,list(CTD.keys())) for st in stations]), 'Not all '\
            'of the provided stations were found in the CTD data! \n'\
            'The following stations were not found in the data: '\
            +''.join([str(st)+' ' for st in stations if ~np.isin(st,list(CTD.keys()))])
    # Check if x_type is either distance or time
    assert x_type in ['distance','time'], 'x_type must be eigher distance or '\
            'time!'
            
    # select only the given stations in the data
    CTD = {key:CTD[key] for key in stations}
    
    # extract Bottom Depth    
    BDEPTH = np.asarray([d['BottomDepth'] for d in CTD.values()])

    # put the fields (the vector data) on a regular, common pressure and X grid
    # by interpolating. 
    fCTD,Z,X,station_locs = CTD_to_grid(CTD,x_type=x_type)
    
    # plot the figure
    fig,ax = plt.subplots(1,1,figsize=(8,5))
  
    # Temperature
    contour_section(X,Z,fCTD[parameter],fCTD['SIGTH'],ax = ax,
                          station_pos=station_locs,cmap=cmap,
                          clabel=clabel,bottom_depth=BDEPTH,
                          station_text=section_name)
    # Add x and y labels
    ax.set_ylabel('Depth [m]')
    if x_type == 'distance':
        ax.set_xlabel('Distance [km]')
    else:
        ax.set_xlabel('Time [h]')
        
    # add title
    fig.suptitle(cruise_name+' Section '+section_name,fontweight='bold')
    
    # tight_layout
    fig.tight_layout(h_pad=0.1,rect=[0,0,1,0.95])
    
def plot_CTD_station(CTD,station,add = False):
    '''
    Plots the temperature and salinity profile of a single station.

    Parameters
    ----------
    CTD : str or dict
        Either a dict of dicts containing the CTD data, which can be made with
              the function read_CTD. Or a str with a file where the dict is stored
    station : number
        Number which station to plot (must be in the CTD data!).
    add : bool, optional
        Switch whether to add the plot to a figure (True), or to create a 
        new figure for the plot (False). The default is False.

    Returns
    -------
    None.
    '''
    # Check if the function has data to work with
    assert type(CTD) in [dict,str], 'Parameter *CTD*: You must provide either\n'\
            ' a) a data dict or \n b) a npy file string with the data !'
    
    # read in the data (only needed if no CTD-dict, but a file was given)
    if type(CTD) is str:
        print('reading file...')
        CTD = np.load(CTD,allow_pickle=True).item()
        
    # Check if all stations given are found in the data
    assert np.isin(station,list(CTD.keys())), 'The station was not found in '\
            'the CTD data! \n The following stations are in the data: '\
            +''.join([str(st) +' ' for st in CTD.keys()])
    
    # end of checks.
            
    # select station
    CTD = CTD[station]
    
    if add:
        ax = plt.gca()
    else:
        fig,ax = plt.subplots(1,1,figsize=(5,5))
    
    # plot
    ax.plot(CTD['CT'],-CTD['z'],'r')
    ax.set_xlabel('Conservative temperature [˚C]',color='r')
    ax.set_ylabel('Depth [m]')
    ax.spines['bottom'].set_color('r')
    ax.tick_params(axis='x', colors='r')
    ax.invert_yaxis()
    ax2 = ax.twiny()
    ax2.plot(CTD['SA'],-CTD['z'],'b')
    ax2.set_xlabel('Absolute salinity [g / kg]',color='b')
    ax2.tick_params(axis='x', colors='b')
    plt.tight_layout()
    
def plot_CTD_map(CTD,stations=None,topofile=None,extent=None,
                 depth_contours=[10,50,100,150,200,300,400,500,1000,2000,
                          3000,4000,5000]):
    '''
    Function which plots a very basic map of selected CTD stations.

    Parameters
    ----------
    CTD : dict
        Dictionary containing the CTD data. TODO: Should also take a list of pos!
    stations : array_like, optional
        The positions to put on the map. The default is all stations.
    topofile : str, optional
        MAT file with bathymetry data. The default is None, then no bathymetry
        will be plotted.
    extent : (4,) array_like, optional
        List of map extent. Must be given as [lon0,lon1,lat0,lat1].
        The default is None.
    depth_contours : array_like, optional
        A list containing contour levels for the bathymetry. The default is 
        [10,50,100,150,200,300,400,500,1000,2000,3000,4000,5000].

    Returns
    -------
    None.

    '''
    
    # if no stations are provided, just plot all stations
    if stations is None:
        stations = CTD.keys()
        
    # select only stations
    CTD = {key:value for key,value in CTD.items() if key in stations}
    lat = [value['LAT'] for value in CTD.values()]
    lon = [value['LON'] for value in CTD.values()]
    std_lat,std_lon = np.std(lat),np.std(lon) 
    lon_range = [min(lon)-std_lon,max(lon)+std_lon]
    lat_range = [min(lat)-std_lat,max(lat)+std_lat]
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    if extent is None:
        extent = [lon_range[0],lon_range[1],lat_range[0],lat_range[1]]
    ax.set_extent(extent)
    
    if topofile is not None: # if topography is provided
        topo = loadmat(topofile)
        topo_lat,topo_lon,topo_z = topo['lat'],topo['lon'],topo['D']

        BC = ax.contour(topo_lon,topo_lat,topo_z,colors='lightblue',
                   levels=depth_contours,linewidths=0.3,
                    transform=ccrs.PlateCarree())
        clabels = ax.clabel(BC, depth_contours,fontsize=4,fmt = '%i')
        [txt.set_bbox(dict(facecolor='none', edgecolor='none',
                           pad=0,alpha=0.)) for txt in clabels]
        ax.contour(topo_lon,topo_lat,topo_z,levels=[0],colors='k',linewidths=0.5)
        ax.contourf(topo_lon,topo_lat,topo_z,levels=[-1,1],
                    colors=['lightgray','white'])
    else: # if no topography is provided
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='auto',
                                                    facecolor='lightgray',
                                                    linewidth=0.5))
        
    # add the points, and add labels
    ax.plot(lon,lat,'xr',transform=ccrs.PlateCarree())
    for i,station in enumerate(stations):
        ax.text(lon[i],lat[i],str(station),horizontalalignment='center',
                verticalalignment='bottom')
       
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    # make sure aspect ration of the axes is not too extreme
    ax.set_aspect('auto')
    plt.gcf().canvas.draw()
    plt.tight_layout()

def plot_CTD_ts(CTD,stations=None,pref = 0):
    '''
    Plots a TS diagram of selected stations from a CTD dataset. 

    Parameters
    ----------
    CTD : dict
        Dictionary containing the CTD data.
    stations : array-like, optional
        The desired stations. The default is all stations in CTD.
    pref : TYPE, optional
        Which reference pressure to use. The following options exist:\n
        0:    0 dbar\n
        1: 1000 dbar\n
        2: 2000 dbar\n
        3: 3000 dbar\n
        4: 4000 dbar\n
        The default is 0.

    Returns
    -------
    None.

    '''
    # select only input stations
    if stations is not None:
        CTD = {key:value for key,value in CTD.items() if key in stations}
        
    max_S = max([np.nanmax(value['SA']) for value in CTD.values()]) + 0.1
    min_S = min([np.nanmin(value['SA']) for value in CTD.values()]) - 0.1
    
    max_T = max([np.nanmax(value['CT']) for value in CTD.values()]) + 0.5
    min_T = min([np.nanmin(value['CT']) for value in CTD.values()]) - 0.5
    
    
    create_empty_ts((min_T,max_T),(min_S,max_S),p_ref=pref)
    
    # Plot the data in the empty TS-diagram
    for station in CTD.values():
        plt.plot(station['SA'],station['CT'],linestyle='none',marker='.',
                 label=station['st'])
        
    if len(CTD.keys()) > 1:
        plt.legend(ncol=2,framealpha=1,columnspacing=0.7,handletextpad=0.4)
    
def create_empty_ts(T_extent,S_extent,p_ref = 0):
    '''
    Creates an empty TS-diagram to plot data into. 

    Parameters
    ----------
    T_extent : (2,) array_like
        The minimum and maximum conservative temperature.
    S_extent : (2,) array_like
        The minimum and maximum absolute salinity.
    p_ref : int, optional
        Which reference pressure to use. The following options exist:\n
        0:    0 dbar\n
        1: 1000 dbar\n
        2: 2000 dbar\n
        3: 3000 dbar\n
        4: 4000 dbar\n
        The default is 0.

    Returns
    -------
    None.

    '''
    
    sigma_functions = [gsw.sigma0,gsw.sigma1,gsw.sigma2,gsw.sigma3,gsw.sigma4]
    T = np.linspace(T_extent[0],T_extent[1],100)
    S = np.linspace(S_extent[0],S_extent[1],100)
    
    T,S = np.meshgrid(T,S)
    
    SIGMA = sigma_functions[p_ref](S,T)
    
    cs = plt.contour(S,T,SIGMA,colors='k',linestyles='--')
    plt.clabel(cs,fmt = '%1.1f')
    
    plt.ylabel('Conservative Temperature [°C]')
    plt.xlabel('Absolute Salinity [g kg$^{-1}$]')
    plt.title('$\Theta$ - $S_A$ Diagram')
    if p_ref > 0:
        plt.title('Density: $\sigma_{'+str(p_ref)+'}$',loc='left',fontsize=10)
    
    