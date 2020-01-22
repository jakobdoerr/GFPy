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
    elif data_format == 'nc':
        station = read_netcdf_AWS(filename)
    else:
        print('The data format '+str(data_format)+' is not supported by read_single_AWS')
        station = pd.DataFrame()        
    
    # Define variable long names and units
    variables ={'P': 'Pressure [hPa]',
                'TA': r'Air Temperature [$^\circ$C]',
                'UU': 'Relative Humidity [%]',
                'FF': 'Wind Speed [m/s]',
                'DD': r'Wind Direction [$^\circ$]',
                'RR_01': 'Rain Rate [mm]',
                'QSI_01': r'Shortwave Irradiation [W/m$^2$]'}
    
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