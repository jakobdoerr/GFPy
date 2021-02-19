#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:08:52 2020

@author: Christiane Duscha (cdu022)

Collection of functions to read and plot meteorological data relevant to GFI field work campaigns
"""
import numpy as np
import pandas as pd
import os
import requests
from netCDF4 import Dataset
from netCDF4 import num2date, date2num 
from datetime import datetime, timedelta 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# General reading function
# =============================================================================
def read_netcdf(file):
    """
    Read netcdf datasets (with maximum of 2 dimension variables)

    Parameters
    ==========
    file: str
        path and name of file to be read
        
    Returns
    =======
    buffer: dictionary
        dictionary, variables can be inspected using buffer.keys()

    """
    # Define the file
    data = Dataset(file)

    # Get the time vector
    try:
        time = num2date(data['time'][:].data, units='gregorian',
                            calendar=data['time'].calendar)
    # In case the netcdf stars in year 0000 (not possible in netcdf4 module)
    except ValueError:
        # to unidata about this
        tunits = data['time'].units
        since_yr_idx = tunits.index('since ') + 6
        year = int(tunits[since_yr_idx:since_yr_idx+4])
        year_diff = year - 1

        new_units = tunits[:since_yr_idx] + '0001-01-01 00:00:00'
        time = num2date(data['time'][:].data, new_units, calendar='gregorian')
        time = [datetime(d.year + year_diff, d.month, d.day, 
                  d.hour, d.minute, d.second) for d in time]    

    # Get the variable names and dimensions
    var_name = []
    var_dim = []
        
    buffer = {}    
    for var in data.variables:
        var_name = np.append(var_name, var)
        var_dim = np.append(var_dim, len(data[var].shape))
        # Save the variable
        if var == 'level':
            buffer[var] = data[var][:].data
        elif len(data[var].shape) == 1:
            buffer[var] = pd.DataFrame(data[var][:].data, index = time, columns = [var])
        elif len(data[var].shape) == 2:
            buffer[var] = pd.DataFrame(data[var][:,:].data, index = time, columns = data['level'][:].data)
        
    data.close()
    return buffer

# =============================================================================
# Instrument specific Reading 
# =============================================================================

# =============================================================================
# Automated Weather station   
# =============================================================================
def read_single_AWS(filename):
    """
    Read meteorological data from a single Autometed weather station (AWS). Depending on the data format
    
    Parameters
    ==========
    filename: str
        path and name of file to be read
        
    Returns
    =======
    station: pandas dataframe
        dataframe containing all variables observed by AWS
    variables: dictionary
        dictionary containing all met variable shortuts and their long names
    
    """
    data_format = filename.split('.')[-1]
    
    if data_format == 'dat':
        station = read_dat_AWS(filename)
            # Define variable long names and units
        variables = {'P': 'Pressure [hPa]',
                    'TA': r'Air Temperature [$^\circ$C]',
                    'UU': 'Relative Humidity [%]',
                    'FF': 'Wind Speed [m/s]',
                    'DD': r'Wind Direction [$^\circ$]',
                    'RR_01': 'Rain Rate [mm]',
                    'QSI_01': r'Shortwave Irradiation [W/m$^2$]'}
    elif data_format == 'nc':
        station = read_netcdf_AWS(filename)
        variables = {'P': 'Pressure [hPa]',
                     'T': r'Air Temperature [$^\circ$C]',
                     'RH': 'Relative Humidity [%]',
                     'FF': 'Wind Speed [m/s]',
                     'DD': r'Wind Direction [$^\circ$]',
                     'RR': 'Rain Rate [mm]',
                     'RAD': r'Shortwave Irradiation [W/m$^2$]'}
    else:
        print('The data format '+str(data_format)+' is not supported by read_single_AWS')
        station   = pd.DataFrame()        
        variables = pd.DataFrame()     
    
    return station, variables
    
def read_netcdf_AWS(filename):
    """
    Read the raw data of AWS station from file in netcdf format (.nc)

    Parameters
    ==========
    filename: str
        path and name of file to be read
        
    Returns
    =======
    station: pandas dataframe
        dataframe containing all variables observed by AWS
     
    """
    
    # open the netcdf dataset
    dataset = Dataset(filename)

    # get all variable names from the dataset
    variables = [i for i in dataset.variables]
    
    # Get the dimension name of the dataset
    dim = [i for i in dataset.dimensions]

    # Get the time vector of the dataset 
    try:
        time = num2date(dataset[dim[0]][:].data, units=dataset[dim[0]].units,
                            calendar=dataset[dim[0]].calendar)
    # In case the netcdf stars in year 0000 (not possible in netcdf4 module)
    except ValueError:
        # to unidata about this
        tunits = dataset[dim[0]].units
        since_yr_idx = tunits.index('since ') + 6
        year = int(tunits[since_yr_idx:since_yr_idx+4])
        year_diff = year - 1

        new_units = tunits[:since_yr_idx] + '0001-01-01 00:00:00'
        time = num2date(dataset[dim[0]][:].data, new_units, calendar=dataset[dim[0]].calendar)
        time = [datetime(d.year + year_diff, d.month, d.day, 
                  d.hour, d.minute, d.second) for d in time]    
    
    # write the data to padas dataframe    
    # set the time vector as index    
    station = pd.DataFrame(index = time)
    for var in variables:
        # write the variables to the dataframe
        station[var] = dataset[var][:].data
    
    return station
    
def read_dat_AWS(filename):
    """
    Read the raw data of AWS station from file in ascii format (.dat)

    Parameters
    ==========
    
    filename: str
        path and name of file to be read
    
    Returns
    =======
    
    station: pandas dataframe
        dataframe containing all variables observed by AWS
    
    """
    # Read the dat file using pandas
    station = pd.read_csv(filename, header = 1, skiprows = [2,3])
    
    # Set the timestamp as data index
    station.index = pd.to_datetime(station['TIMESTAMP'].tolist(),
                                   format='%Y-%m-%d %H:%M:%S')
    # delete the time stamp column (As it is conserved in the index)
    del station['TIMESTAMP']

    return station

# =============================================================================
# Ship Log (Kristine Bonnevie)
# =============================================================================
def read_ship_log(filenames):
    """
    Read the data logged by Kristine Bonnevie

    Parameters
    ==========
    
    filenames str
        path and name of file to be read
    
    Returns
    =======
    
    log: pandas dataframe
        dataframe containing all variables observed by Kristine Bonnevie
    
    """
    ship_log = pd.DataFrame()              
    for filename in filenames:
        # Read the data 
        log = pd.read_csv(filename, skiprows=2, encoding = "ISO-8859-1", engine='python')
    
        # Get the date
        day = filename[-14:-12]
        month = filename[-11:-9]
        year = filename[-8:-4]
    
        log.Time = pd.to_datetime(year+'-'+month+'-'+day+' '+log.Time.copy())
        log.Time.iloc[-1] = log.Time.iloc[-1]+timedelta(days=1)
    
        log.index = log.Time.copy()
    
        # Get the longitude and latitude information
        lon = np.zeros(len(log))
        lat = np.zeros(len(log))
        for i in range(len(log)):
            try:
                lon[i] = float(log['Longitude'].iloc[i][:3]) + float((log['Longitude'].str.split(' ').iloc[i][0])[3:])/60.
                lat[i] = float(log['Latitude'].iloc[i][:2]) +  float((log['Latitude'].str.split(' ').iloc[i][0])[2:])/60.
            except TypeError:
                lon[i] = np.nan
                lat[i] = np.nan
            
        log['Longitude'] = lon
        log['Latitude']  = lat
    
        log[log==-999] = np.nan
        ship_log = pd.concat([ship_log,log], axis=0)
    # sort the index
    ship_log = ship_log.sort_index()    
    
    return ship_log

# =============================================================================
# HOBO Raingauges 
# =============================================================================
def read_HOBO(filenames):
    """
    Function to read the raw HOBO data for all deployed HOBO stations
    example input (GEOF232-2020): filenames = [path+'HOBO2020_20547.txt', path+'HOBO2020_20853.txt', 
                                               path+'HOBO2020_21075.txt', path+'HOBO2020_21079.txt',
                                               path+'HOBO2020_21081.txt',path+'HOBO2020_21083.txt',
                                               path+'HOBO2020_21084.txt',path+'HOBO2020_21085.txt']
    Must define a path first
    
    Parameters
    ==========
    filenames: list
        list of path and name of files to be read
        
    Returns
    =======
    station_ID: list
        list of station ID names (str) read
    
    station: dict
        Dictionary containing pandas DataFrames corresponding to Raw data of each station
    hourly: dict
        Dictionary containing pandas DataFrames corresponding to 1 hour rain rate data of each station
    """
    # Define dictionary to save all stations
    station = {}
    hourly = {}
    # Read data
    # Iterate over stations --> save station ID
    station_ID = []
    for file in filenames:
        ID = file[-9:-4]
        station_ID.append(ID)
        station[file[-9:-4]] = pd.read_csv(file, header = 1, names = ['Timestamp', 'data'])
        station[file[-9:-4]]['Event'], station[file[-9:-4]]['Count'] = station[file[-9:-4]].data.str.split('\t',1).str
        #del station[file[-18:-13]].data
        station[file[-9:-4]]['Event'] = station[file[-9:-4]]['Event'].astype(int)
        station[file[-9:-4]]['Count'] = station[file[-9:-4]]['Count'].astype(int)
        # Get timestamp
        station[file[-9:-4]].index = pd.to_datetime(station[file[-9:-4]]['Timestamp'].tolist(),
               format='%m/%d/%y %H:%M:%S')
        # Reduce dataset to necessary variables
        station[file[-9:-4]] = station[file[-9:-4]][['Count','Event']]
        # set the Rain Rate per count (0.2mm)
        station[file[-9:-4]]['RR'] = 0.2           
        
        # Get the hourly rain rate (mm/h)
        hourly[file[-9:-4]] = station[file[-9:-4]]['RR'].resample('1h').sum()
        
    return station_ID, station, hourly

# =============================================================================
# OPUS Logger
# =============================================================================
def read_OPUS(filenames):
    """
    Function to read the raw OPUS Data
    example input (GEOF232-2020): filenames = [path+'OPUS20_2020_Nr010_20200317_160955.csv',
                                               path+'OPUS20_2020_Nr012_20200317_160549.csv']
    
    Must define a path first
    
    Parameters
    ==========
    filenames: list
        list of path and name of files to be read
        
    Returns
    =======
    station: dict
        Dictionary containing pandas DataFrames corresponding to Raw data of each station, 
        Station ID names can be acced by station.keys()
    
    """
    # Define dictionary to save all stations
    station = {}
    # Read data
    # Iterate over stations --> save station ID
    # Define column names
    cols = ['date and time', 'temperature','relative humidity','pressure']
    for file in filenames:
        ID = file[-25:-20]
        station[str(ID)] = pd.read_csv(file, header = None,names =cols, skiprows = 10, delimiter=';')
        timestamp = station[str(ID)]['date and time']
        station[str(ID)].index = pd.to_datetime(timestamp.tolist(), format='%d.%m.%Y %H.%M.%S')
        del station[str(ID)]['date and time']
    return station

# =============================================================================
# EasyLog
# =============================================================================
    
def read_EasyLog(filenames):
    """
    Function to read the raw EasyLog Data
    example input (GEOF232-2020): filenames = [path+'OPUS20_2020_Nr010_20200317_160955.csv',
                                               path+'OPUS20_2020_Nr012_20200317_160549.csv']
    
    Must define a path first
    
    Parameters
    ==========
    filenames: list
        list of path and name of files to be read
        
    Returns
    =======
    station: dict
        Dictionary containing pandas DataFrames corresponding to Raw data of each station, 
        Station ID names can be acced by station.keys()
    
    """
    # Define dictionary to save all stations
    station = {}

    # Read data
    # Iterate over stations --> save station ID
    for file in filenames:
        ID = file[-5:-4]
        station[str(ID)] = pd.read_csv(file, skiprows=1, 
                                       usecols=[1,2], 
                                       names = ['time','temperature'])
        station[str(ID)].index = pd.to_datetime(station[str(ID)]['time'].tolist(), format='%d/%m/%Y %H:%M:%S')
        del station[str(ID)]['time']
    return station

# =============================================================================
# TinyTag
# =============================================================================
def read_TinyTag(filenames):
    """
    function to read the Tinytags
    
    Parameters
    ==========
    filenames: list
        list of path and filenames to read 
    
    Results
    =======
    stations: dictionary
        dictionary containing DataFrames of different TinyTags
    """
    stations = {}
    for file in filenames:
        ID = pd.read_csv(filenames[0], encoding = "ISO-8859-1", nrows=3, sep='\t')['1'][1]
        # read the file
        data = pd.read_csv(filenames[0], skiprows=[0,1,2,3], sep='\t', encoding = "ISO-8859-1")
        # Get the values (seperate from the units)
        TA = [float(i[0]) for i in data.Temperature.str.split(' ')]  # temperature
        RH = [float(i[0]) for i in data.Humidity.str.split(' ')]     # humidity
        TD = [float(i[0]) for i in data['Dew Point'].str.split(' ')] # dew point
        # the month string is in norwegian, hence must be replaced by the international format
        time = [pd.to_datetime(i.replace('mai','May').replace('okt','Oct').replace('des','Dec'), 
                               format = '%d %b %Y %H.%M.%S') for i in data['Unnamed: 1']]
        # Write to dataframe 
        buffer = pd.DataFrame({'Temperature':TA,'Humidity':RH,'Dew Point':TD}, index = time)
        stations[ID] = buffer
        
    return stations

# ================================================================================
# Get information and data about the MET Norway (Meteorologisk Institutt) stations 
# ================================================================================

def get_MET_station_info(filename):
    '''
    function to extract the MET norway station ID and corresponding location for a certain norwegian municipality

    Parameters:
    ===========
    filename : str
        path and name of file to be read
    
    Returns:
    ========   
    met_stations : pandas DataFrame
        DataFrame containing the MET norway station ID and corresponding location
    '''

    # open the file 
    file = open(filename,'r')
    lines = file.readlines()

    # remove the info about the external IDs
    lines = pd.Series(lines)[~pd.Series(lines).str.contains('external ID')]

    ID_loc = lines[lines.str.contains('ID:')].index

    # Get the important station info
    ID  = []
    lon = []
    lat = []
    for i in range(len(lines[lines.str.contains('ID:')])):
        ID  = np.append(ID, lines[ID_loc[i]].split(':')[1][:-1])
        lon = np.append(lon, float(lines[ID_loc[i]+2].split(':')[1][:-1]))
        lat = np.append(lat, float(lines[ID_loc[i]+3].split(':')[1][:-1]))
        #Voss_stations[ID] = {}
        #print(lines.iloc[ID_loc[ID]])
    
    # Make a dataframe from the information
    met_stations = pd.DataFrame({'lon':lon, 'lat':lat}, index=ID)
    
    return met_stations

def get_frost_variables(client_ID,station_ID, ref_time = '2020-01-01'):
    """
    Function to extract the variable names of a certain MET Norway station referring to a certain point in time.
    
    Parameters:
    ===========
    client_ID : str
        For accessing observations from frost.met.no you need to have a client ID.
        Visit https://frost.met.no/howto.html for getting a client ID, and for more information.
    
    station_ID : str
        the station ID, as defined by MET Norway
    
    ref_time : str
        availability of the variables at a certain reference point in time. format: YYYY-MM-DD
    
    Returns:
    ========
    variables : list
        list containing the available variable names of a station for a certain point in time    
    """
    
    url = 'https://frost.met.no/observations/availableTimeSeries/v0.jsonld'
    params = {
              'sources': station_ID,
              'referencetime': ref_time,
             }

    # Issue an HTTP GET request
    r = requests.get(url, params, auth=(client_ID,''))
    # Extract JSON data
    json = r.json()
    if r.status_code == 200:
        data = json['data']
        print('Data retrieved from frost.met.no!')
    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % json['error']['message'])
        print('Reason: %s' % json['error']['reason'])
        raise RuntimeError(station_ID, json['error']['reason'])
    df = pd.DataFrame(data)

    try:
        variables = df.elementId[df.elementId.apply(lambda e: '(' not in e)].values
    except RuntimeError as re:
        variables = []
        print('Could not load variables for station:', s)
        
    return variables

def get_frost_data(client_ID,station_ID, variable, period):
    """
    Access and export observation of a certain MET Norway station.
    
    Parameters:
    ===========
    client_ID : str
        For accessing observations from frost.met.no you need to have a client ID.
        Visit https://frost.met.no/howto.html for getting a client ID, and for more information.
    
    station_ID : str
        the station ID, as defined by MET Norway
    
    variable : str
        the variable to be read. Check the available variables with:
    
    period : str
        period of interest to read. The format (start/end) is: YYYY-MM-DD/YYYY-MM-DD 
    
    Returns:
    ========
    df.value : pandas DataFrame 
        DataFrame containing the time series of the variable of interest
    
    """
    # Define endpoint and parameters
    # Check the available variables e.g.:
    # https://frost.met.no/observations/availableTimeSeries/v0.jsonld?sources=SN51610
        
    endpoint = 'https://frost.met.no/observations/v0.jsonld'
    parameters = {
                  'sources': station_ID,
                  'elements': variable,
                  'referencetime': period,
                 }
    # Issue an HTTP GET request
    r = requests.get(endpoint, parameters, auth=(client_ID,''))
    # Extract JSON data
    json = r.json()
    if r.status_code == 200:
        data = json['data']
        print('Data retrieved from frost.met.no!')
    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % json['error']['message'])
        print('Reason: %s' % json['error']['reason'])
        raise RuntimeError(station_ID, json['error']['message'])

        
    df = pd.json_normalize(data, record_path='observations')
    dff = pd.json_normalize(data).drop('observations', axis=1)
        
    df = pd.concat([df, dff], axis=1)
    df['time'] = pd.to_datetime(df.referenceTime)
    df.index = df['time'].copy()
    
        
    return df.value
# =============================================================================
# Read the high resolutioned topographic dataset
# =============================================================================

def read_topo(filename):
    """
    Read the high resolutioned topographic dataset (added 2021-02-19). 
    The topography data can be downloaded from:
    http://www.viewfinderpanoramas.org/Coverage%20map%20viewfinderpanoramas_org1.htm
    
    Parameters:
    ===========
    filename : str
        path and name of file to be read
    
    Returns:
    ========
    alt : masked array
        masked array containing the altitude information on a lat-lon grid. 
        ocean grid points (altitude<0m) are masked. The unmasked altitude array can be accessed via alt.data 
    lat : array
        array containing the latitude vector
    lon : array
        array containing the longitude vector
    """
    
    siz = os.path.getsize(filename)
    dim = int(np.sqrt(siz/2))
    
    # Check if the dimensions match
    assert dim*dim*2 == siz, 'Invalid file size'

    # get the altitude data
    alt = np.fromfile(filename, np.dtype('>i2'), dim*dim).reshape((dim, dim))

    # --------------------------------------
    # Get the latitude and longitude vectors
    #---------------------------------------
    # Get the lat and lon start points
    lat_min = float(filename[-10:-8])
    lon_min = float(filename[-7:-4])
    # create the corresponding vectors
    lat = np.arange(lat_min,lat_min+1.0,1/dim)[::-1]
    lon = np.arange(lon_min,lon_min+1.0,1/dim)
    
    alt = np.ma.masked_where(alt == 0, alt)
    
    return alt, lat, lon

# =============================================================================
# Ploting section
# =============================================================================

def plot_AWS(filename):
    """
    Plot the meteorological data from a single Autometed weather station (AWS). 
    
    Parameters
    ==========
    filename: str
        path and name of file to be read
        
    Returns
    =======
    figure: matplotlib.pyplot.figure
        figure of all variables connected to AWS observations
            
    """
    
    # Read the AWS file 
    station, variables = read_single_AWS(filename)
    
    # Plot the data
    
    fig, ax = plt.subplots(nrows = len(variables), figsize = [14,len(variables)*3], sharex=True)
    i = 0
    for var in variables:
        ax[i].plot(station[var])
        ax[i].set_ylabel(variables[var])
        i = i+1
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[0].set_xlim([station.index[0],station.index[-1]])


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    Function to use only a fraction of a matplotlib colormap
    from: https://stackoverflow.com/a/18926541
    
    Parameters: 
    ===========
    cmap : str
        name of chosen colormap (matplotlib.colors.LinearSegmentedColormap)
    minval : float
       lower edge of colormap (minimum: 0.0)
    maxval : float
       upper edge of colormap (maximum: 1.0)
    n : integer
       number of color segments
       
    Returns:
    ========
    new_cmap : LinearSegmentedColormap
        fraction [minval,maxval] of a matplotlib colormap
    
    '''
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap    