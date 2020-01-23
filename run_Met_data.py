#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:08:52 2020

@author: Christiane Duscha (cdu022)

Executable for functions in GFPy
"""
import sys
sys.path.append('GFPy')
from GFPy.Met_data import *
import glob


# =============================================================================
# Read an example AWS file 
# =============================================================================

# Document Data location (absolute or relative to current location)
data_loc = '/home/cdu022/Documents/PhD/Lidar_campaigns/GEOF-232_2019/Example_data/AWS/'

# -----------------
# Read ascii format
#------------------ 
data_format ='*.dat'

# Get all files of sought format
filenames = sorted(glob.glob(data_loc+data_format))

# Read the first ascii file
AWS_dat = read_dat_AWS(filenames[0])

plot_AWS(filenames[0])
# ------------------
# read netcdf format
# ------------------

data_format ='*.nc'

# Get all files of sought format
filenames = sorted(glob.glob(data_loc+data_format))

# Read the first netcdf file
AWS_nc = read_netcdf_AWS(filenames[0])


# ---------------------------
# read ascii or netcdf format
# ---------------------------

AWS, variables = read_single_AWS(filenames[0])

# =============================================================================
# Plot the data
# =============================================================================

# Plot the AWS
plot_AWS(filenames[0])

# =============================================================================
# Read the dataset recorded by the ship log of Kristine Bonnevie
# =============================================================================

# Document Data location (absolute or relative to current location)
data_loc = '/home/cdu022/Documents/PhD/Lidar_campaigns/GEOF-232_2019/Example_data/Ship_log/'

# define the data format
data_format ='*.csv'

# Get all files of sought format
filenames = sorted(glob.glob(data_loc+data_format))

# read the file
ship_log = read_ship_log(filenames[0])


