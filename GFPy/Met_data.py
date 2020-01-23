#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:08:52 2020

@author: Christiane Duscha (cdu022)

Collection of functions to read and plot meteorological data relevant to GFI field work campaigns
"""
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from netCDF4 import num2date, date2num 
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# Reading section
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


def read_ship_log(filename):
    """
    Read the data logged by Kristine Bonnevie

    Parameters
    ==========
    
    filename: str
        path and name of file to be read
    
    Returns
    =======
    
    log: pandas dataframe
        dataframe containing all variables observed by Kristine Bonnevie
    
    """
    
    # Read the data 
    log = pd.read_csv(filename)
    
    # set the index to time in Datetime format
    log.index = pd.to_datetime(log['Timestamp'].tolist(),
                               format='%Y-%m-%d %H:%M:%S')
    
    # delete the time stamp column (As it is conserved in the index)    
    del log['Timestamp']
    
    return log


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
    
    