# -*- coding: utf-8 -*-
"""
This module contains functions that read and plot CTD data.
"""
 
from seabird.cnv import fCNV
import gsw
import numpy as np
import matplotlib.pyplot as plt
import glob

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
    CTD_list = []
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
        
        CTD_list.append(p)

    # save data if outpath was given    
    if outpath is not None:
        np.save(cruise_name+'_CTD',CTD_list)
        
    return CTD_list