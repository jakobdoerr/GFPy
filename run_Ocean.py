#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:08:52 2020

@author: Christiane Duscha (cdu022)

Executable for functions in GFPy.Met_data
"""

from GFPy.Ocean import readCTD,plot_CTD_section,plot_CTD_station
import matplotlib.pyplot as plt

# =============================================================================
# read in some CTD data
# =============================================================================

# Document Data location (absolute or relative to current location)
data_loc = '/Users/jakobdorr/Documents/Phd/Teaching/MATLAB_TO_PYTHON_CRUISE2020/'\
            '2019_Masfjorden/Data/GS2018/'

# Read all station files in specified folder, and save result in a .npy file
CTD_all = readCTD(data_loc,'test_cruise',outpath='./')
 
# Read specific stations in the folder
stations = range(400,410)
CTD_part = readCTD(data_loc,'test_cruise',stations=stations) 

# =============================================================================
# Plot a CTD section
# =============================================================================

#define stations included in the CTD section
stations = range(401,410)

# plot the section, given the variable CTD_all from above
plot_CTD_section(CTD_all,stations,cruise_name='test_cruise',section_name='A')

# plot the section, given the path to the .npy file given above
plot_CTD_section('./test_cruise_CTD.npy',stations,cruise_name='test_cruise',section_name='A')

# save the current figure
plt.savefig('./test_image.pdf')
# =============================================================================
# Plot a CTD profile
# =============================================================================
# define the station for which to plot a profile
station = 402

# plot a single profile
plot_CTD_station(CTD_all, station)