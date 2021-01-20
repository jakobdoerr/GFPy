#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:08:52 2020

@author: Jakob Dörr (jdo043)

Example script for functions in GFPy.Ocean
"""

import GFPy.Ocean as Oc
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')
# =============================================================================
# read in some CTD data
# =============================================================================

# Document Data location (absolute or relative to current location)
data_loc = './testdata/CTD/'

# Read all station files in specified folder, and save result in a .npy file
CTD_all = Oc.read_CTD(data_loc,'test_cruise',outpath='./')
 
# Read specific stations in the folder
stations = range(401,410)
CTD_part = Oc.read_CTD(data_loc,'test_cruise',stations=stations) 

# =============================================================================
# Plot a CTD section
# =============================================================================

#define stations included in the CTD section
stations = range(401,410)

# plot the section, given the variable CTD_all from above
Oc.plot_CTD_section(CTD_all,stations,
                 cruise_name='test_cruise',section_name='A')

axx = plt.gcf().axes
axx[0].set_ylim([200,0])
axx[0].set_clim(-4,4)
# %%
# plot the section, given the path to the .npy file given above
Oc.plot_CTD_section('./test_cruise_CTD.npy',stations,
                 cruise_name='test_cruise',section_name='A')
# save the current figure
plt.savefig('./test_image.pdf')

# plot a section of a single variable of your choice (for example Oxygen)
Oc.plot_CTD_single_section(CTD_all,stations,parameter='OX',
                           clabel='Oxygen [mg/l]',cmap='cmo.oxy')

# =============================================================================
# Plot a CTD profile
# =============================================================================
# define the station for which to plot a profile
station = 402

# plot a single profile
Oc.plot_CTD_station(CTD_all, station)
plt.savefig('./test_profile.pdf')

# plot several single profiles in one plot with subplots
plt.figure()
plt.subplot(121)
Oc.plot_CTD_station(CTD_all, station,add = True)
plt.subplot(122)
Oc.plot_CTD_station(CTD_all, station+1,add = True)
# if you want to manipulate the figure afterwards, you have to get the
# axes using plt.gcf().axes. There are 4 axes in this example, 2 for each
# subplot, because we have two x-axes in each subplot. To, i.e. change the
# temperaure range and xlabel in the first sublplot, do:
axx = plt.gcf().axes
axx[0].set_xlim(0,15)
axx[0].set_xlabel('Liberal temperature [˚C]')

# =============================================================================
# Plot a map of CTD stations
# =============================================================================
# define the stations to plot on the map
stations = range(401,410)
bathy_file = '/Users/jakobdorr/Documents/PhD/teaching/MATLAB_TO_PYTHON_CRUISE2020'\
            '/2019_Masfjorden/Data/Bathymetry/Masfjorden_bathy.mat'
 
plt.figure()
Oc.plot_CTD_map(CTD_all,stations,topofile=bathy_file)
plt.savefig('./test_map.pdf')
plt.figure()
Oc.plot_CTD_map(CTD_all) # (you should not plot all stations in one map...)


# =============================================================================
# Plot a TS diagram with CTD data
# =============================================================================
plt.figure()
Oc.plot_CTD_ts(CTD_all,[401,402],pref=0)

# you can also create an empty TS-Diagram, and plot your data in it:
plt.figure()
Oc.create_empty_ts((-2,35),(0,40),0)

# Calculate freshwater content
print(Oc.calc_freshwater_content(CTD_all[401]['S'],CTD_all[401]['z']))

# =============================================================================
# Read in and plot some ADCP data (in nc format for now!)
# =============================================================================
# %%
CTD = Oc.read_CTD_from_mat('/Users/jakobdorr/Documents/Phd/Teaching/MATLAB_TO_PYTHON_CRUISE2020/2019_Masfjorden/Data/KB2019602/ctd/KB2019602_cal_CTD.mat')
ADCP = Oc.read_ADCP('/Users/jakobdorr/Documents/Phd/Teaching/MATLAB_TO_PYTHON_CRUISE2020/2019_Masfjorden/Data/KB2019602/ADCP/os150nb.nc')
section = [169,167,168]
Oc.plot_CTD_map(CTD,section)
Oc.plot_ADCP_CTD_section(ADCP, CTD, section)#,levels = np.linspace(-0.06,0.06,13))

#%%
ADCP = Oc.read_ADCP('/Users/jakobdorr/Desktop/contour/os150nb.nc')
CTD = Oc.read_CTD_from_mat('/Users/jakobdorr/Documents/Phd/Teaching/GEOF337_Fjord_oceanography/CTDdataKB2020603_CTD.mat')
section = list(range(258,264))
section.remove(260)
Oc.plot_CTD_map(CTD,section)
Oc.plot_ADCP_CTD_section(ADCP, CTD, section,levels = np.linspace(-0.2,0.2,11),
                         geostr=True)
Oc.plot_CTD_single_section(CTD,section,parameter='OX',
                           clabel='Oxygen [mg/l]',cmap='cmo.oxy',
                           clevels=np.linspace(2,8,13))