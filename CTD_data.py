# -*- coding: utf-8 -*-
"""
This module contains functions that read and plot CTD data.
"""
 
from seabird.cnv import fCNV
import gsw
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import interp1d,griddata
from matplotlib.dates import date2num
import cmocean

def create_latlon_text(lat,lon):
    '''Creates two strings which contain a text for latitude and longitude
    Inputs:
        lat - latitude
        lon - longitude
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

def readCTD(inpath,cruise_name,outpath=None,stations=None):
    '''This function reads in the CTD data from cnv files in *inpath*
    for the stations *stations* and returns a list of dicts containing
    the data. 
    
    inputs:
        inpath(string) - input path where the cnv files are stored
        cruise_name(string) - name of the cruise
        outpath(string) - path where to store the output (optional)
        stations(list) - list of stations to read in (optional). If not given, 
                   the function will read all stations in *inpath*
    returns:
        CTD(dict<dict>) - a dict of dicts contaning the data for
                    all the relevant station data
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
        np.save(cruise_name+'_CTD',CTD_dict)
        
    return CTD_dict

def CTD_to_grid(CTD,stations,interp_opt= 1,x_type='distance'):
    '''This functions accepts a CTD dict of dicts, finds out the maximum 
    length of the depth vectors for the given stations, and fills all
    fields to that maximum length, using np.nan values. 
    inputs:
        CTD - dict of dicts with the CTD data
        stations - list of stations to select
        interp_opt - flag how to interpolate over X (optional). 1: no interpolation,
                     1: linear interpolation, fine grid (default),
                     2: linear interpolation, coarse grid
        x_type - whether X is time or distance (default). Optional
        
    outputs:
        fCTD - dict of dicts with the filled CTD data of the selected stations
        Z - common depth vector
        X - common Z vector (either time or distance)
        station_locs - locations of the stations as X units (time or distance)
    '''

    # construct the Z-vector from the max and min depth of the given stations
    maxdepth = np.nanmax([np.nanmax(-CTD[i]['z']) for i in CTD.keys()])
    mindepth = np.nanmin([np.nanmin(-CTD[i]['z']) for i in CTD.keys()])
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
            fCTD[field] = np.array([interp1d(-value['z'],value[field],bounds_error=False)(Z)
                            for value in CTD.values()]).transpose()
    
    elif interp_opt == 1: # grid over depth and x (time or distance)
        X_fine = np.linspace(np.min(X),np.max(X),len(X)*20) # create fine X grid
        # original grids
        X_orig,Z_orig = [f.ravel() for f in np.meshgrid(X,Z)] 
        for field in fields:
            # grid in Z
            temp_array = np.array([interp1d(-value['z'],value[field],bounds_error=False)(Z)
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
            temp_array = np.array([interp1d(-value['z'],value[field],bounds_error=False)(Z)
                            for value in CTD.values()]).transpose().ravel()
            mask = np.where(~np.isnan(temp_array)) # NaN mask

            # grid in X
            fCTD[field] = griddata((X_orig[mask],Z_orig[mask]), # old grid
                                    temp_array[mask], # data
                                    tuple(np.meshgrid(X_coarse,Z_coarse))) # new grid
        X,Z = X_coarse,Z_coarse

        
    return fCTD,Z,X,station_locs
    
     
def contour_section(ax,X,Y,Z,Z2,station_pos=None,cmap='jet',Z2_contours=None,
                    clabel='',bottom_depth=None,clevels=None,station_text=''):
    '''Plots a filled contour plot of *Z*, with contourf of *Z2* on top to 
    the axes *ax*. It also displays the position of stations, if given in
    *station_pos*, adds labels to the contours of Z2, given in 
    *Z2_contours*. If no labels are given, it assumes Z2 is density (sigma0) 
    and adds its own labels. It adds bottom topography if given in *bottom_depth*.
    Inputs:
        ax - axes object to plot on
        X(array) - X-values
        Y(array) - Y-values
        Z(array) - the filled contour field
        Z2(array) - the contour field on top
        station_pos(list) - the station positions (optional)
        cmap - the colormap for the filled contours (optional)
        Z2_contours(list) - the contour label positions for *Z2* (optional)
        clabel(string) - label to put on the colorbar (optional)
        bottom_depth(list) - list with bottom depth
        clevels(list) - list of color levels to use for *Z* (optional)
        
    Outputs:
        ax - the axes object
    '''
    # get the labels for the Z2 contours
    if Z2_contours is None:
        Z2_contours = np.concatenate([list(range(21,26)),np.arange(25.5,29,0.2)])
        Z2_contours = [i for i in Z2_contours 
                        if np.nanmin(Z2) < i < np.nanmax(Z2)]
    
    # get the Y-axis limits 
    y_limits = (0,np.nanmax(Y)+10)
    
    # get the color levels
    if clevels is None:
        clevels = 20
        
    cT = ax.contourf(X,Y,Z,cmap=cmap,levels=clevels) # draw Z
    cSIG = ax.contour(X,Y,Z2,levels = Z2_contours,
                       colors='k',linewidths=[1],alpha=0.6) # draw Z2
    clabels = plt.clabel(cSIG, Z2_contours,fontsize=8,fmt = '%1.1f') # add contour labels
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0,alpha=0.6)) for txt in clabels]
    plt.colorbar(cT,ax = ax,label=clabel,pad=0.01) # add colorbar
    ax.set_ylim(y_limits)
    ax.invert_yaxis()
    
    # add bathymetry
    if bottom_depth is not None:
        ax.fill_between(station_pos,bottom_depth*0+y_limits[1],bottom_depth,
                     zorder=999,color='gray')
       
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    # add station ticks
    if station_pos is not None:
        for i,pos in enumerate(station_pos):
            ax.text(pos,0,'v',ha='center',fontweight='bold')
            if station_text is not '':
                ax.annotate(station_text+str(i+1),(pos,0),xytext=(0,10),
                        textcoords='offset points',ha='center')
    return ax
    
def plot_CTD_section(stations,CTD=None,infile=None,
                     section_name='',cruise_name = ''):
    '''This function plots a CTD section of Temperature and Salinity,
    given CTD data either directly (through *CTD*) or via a file (through)
    *infile*.
    
    inputs: 
        stations - a list of stations to plot (station numbers have to be 
                   found inside the CTD data!)
        CTD - a dict of dicts containing the CTD data. Can be made with
              the function readCTD
        infile - string with file (and path) to a numpy file of the CTD data
                 Can also be made with the function readCTD
        section_name - name of the Section, will appear in the plot title
        cruise_name - name of the Cruis, will also appear in the title
    outputs:
        fig - a handle to the figure
    '''
    # Check if the function has data to work with
    assert CTD is not None or infile is not None, 'You must provide either \n'\
            ' a) a data list (parameter CTD) or \n b) a file with'\
            ' the data (parameter infile)!'
    # Check if all stations given are found in the data
    assert min([np.isin(st,list(CTD.keys())) for st in stations]), 'Not all '\
            'of the provided stations were found in the CTD data! \n'\
            'The following stations were not found in the data: '\
            +''.join([str(st)+' ' for st in stations if ~np.isin(st,list(CTD.keys()))])
    
    # read in the data (only needed if no CTD-dict, but a file was given)
    if CTD is None:
        CTD = np.load(infile,allow_pickle=True)
    
    # select only the given stations in the data
    CTD = {key:CTD[key] for key in CTD.keys() if key in stations}
    

    # extract Bottom Depth    
    BDEPTH = np.asarray([d['BottomDepth'] for d in CTD.values()])

    # put the fields (the vector data) on a regular, common pressure grid
    # by interpolating. 
    fCTD,Z,X,station_locs = CTD_to_grid(CTD,stations,x_type='distance')
    
    # plot the figure!
    fig,[axT,axS] = plt.subplots(2,1,figsize=(8,9))
    
    # Temperature
    axT = contour_section(axT,X,Z,fCTD['T'],fCTD['SIGTH'],
                          station_pos=station_locs,cmap=cmocean.cm.thermal,
                          clabel='Temperature [˚C]',bottom_depth=BDEPTH,
                          station_text=section_name)
    # Salinity
    axS = contour_section(axS,X,Z,fCTD['S'],fCTD['SIGTH'],
                          station_pos=station_locs,cmap=cmocean.cm.haline,
                          clabel='Salinity [g kg$^{-1}$]',bottom_depth=BDEPTH)
    
    # FIXME: Axis labels!
    # add title
    fig.suptitle(cruise_name+' Section '+section_name)
    
 
# Only for testing purposes:
if __name__ == '__main__':
    CTD = readCTD('/Users/jakobdorr/Documents/Phd/Teaching/MATLAB_TO_PYTHON_CRUISE2020/2019_Masfjorden/Data/GS2018/','JAKOB') 
    plot_CTD_section(range(401,410),CTD = CTD,cruise_name='JAKOB',section_name='A')