#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:05:58 2020

@author: yuchen
"""

import flopy
import flopy.utils.binaryfile as bf
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import pandas as pd


def Get_Conc(Times, units):
    ucnobj = bf.UcnFile('MT3D001.UCN')
    times = ucnobj.get_times()
    concList = []
    for i in Times:
        concList.append('conc_%dd'%(i))
        
    conc_dict = {}
    for index, item in enumerate(concList):
        conc_dict[item] = ucnobj.get_data(totim=Times[index])
        
    density = 1500
    
    if units == 'g':
        cF = (1000*prsity)/density  # conversion factor to convert from kg m**3 to mug L
    else:
        cF = 10**6
        
    for k in conc_dict.keys():
        conc_dict[k] = [cF*x for x in conc_dict[k]]
        
        
    return conc_dict

def Concentration_Plan_View(conc_dict, units):
    
    fig = plt.figure(figsize=(14,10))
    ax1 = fig.add_subplot(2, 2, 1)
    # flopy plot object
    pmv = flopy.plot.PlotMapView(model=mf, layer=0)
    # plot grid
    lc = pmv.plot_grid() # grid
    
    # plot concentration
    cmap = plt.get_cmap('jet')
    if units == 'g':
        cmin = 0
        cmax = 0.1
    else:
        cmin = 0
        cmax = 20000
    
    cs = pmv.plot_array(conc_dict['conc_%dd'%(Times[0])],
                        cmap=cmap,vmin=cmin,vmax=cmax) # concentration colourmap
    cbar = plt.colorbar(cs,format='%.0e') # colour bar
    if units == 'g':
        cbar.ax.set_ylabel('C [$\mu$g g$^-$$^1$]')
    else:
        cbar.ax.set_ylabel('C [$\mu$g L$^-$$^1$]')
        
    # plot the spill site location 
    ax1.plot(wel_most_loc_xcoord, wel_most_loc_ycoord,'o',c='r',label='Spill site',
             markersize=mS)
    
    # setting labels
    ax1.set_xlabel('X-coordinate [m]')
    ax1.set_ylabel('Y-coordinate [m]')
    ax1.set_xlim(wel_most_loc_xcoord-50,wel_most_loc_xcoord+100)
    ax1.set_ylim(wel_most_loc_ycoord-80,wel_most_loc_ycoord+60) 
    tday = Times[0]
    titleText = "Concentration at t = %i" % tday+' days'
    ax1.set_title(titleText,loc='left')
    
    
    #fig = plt.figure(figsize=(14,10))
    ax2 = fig.add_subplot(2, 2, 2)
    # flopy plot object
    pmv = flopy.plot.PlotMapView(model=mf, layer=0)
    # plot grid
    lc = pmv.plot_grid() # grid
    
    # plot concentration
    cmap = plt.get_cmap('jet')
    if units == 'g':
        cmin = 0
        cmax = 0.1
    else:
        cmin = 0
        cmax = 20000
    
    cs = pmv.plot_array(conc_dict['conc_%dd'%(Times[1])],
                        cmap=cmap,vmin=cmin,vmax=cmax) # concentration colourmap
    cbar = plt.colorbar(cs,format='%.0e') # colour bar
    if units == 'g':
        cbar.ax.set_ylabel('C [$\mu$g g$^-$$^1$]')
    else:
        cbar.ax.set_ylabel('C [$\mu$g L$^-$$^1$]')
        
    # plot the spill site location
    ax2.plot(wel_most_loc_xcoord, wel_most_loc_ycoord,'o',c='r',label='Spill site',
             markersize=mS)
    
    # setting labels
    ax2.set_xlabel('X-coordinate [m]')
    ax2.set_ylabel('Y-coordinate [m]')
    ax2.set_xlim(wel_most_loc_xcoord-50,wel_most_loc_xcoord+100)
    ax2.set_ylim(wel_most_loc_ycoord-80,wel_most_loc_ycoord+60) 
    tday = Times[1]
    titleText = "Concentration at t = %i" % tday+' days'
    ax2.set_title(titleText,loc='right')
    
    
    #fig = plt.figure(figsize=(14,10))
    ax3 = fig.add_subplot(2, 2, 3)
    # flopy plot object
    pmv = flopy.plot.PlotMapView(model=mf, layer=0)
    # plot grid
    lc = pmv.plot_grid() # grid
    
    # plot concentration
    cmap = plt.get_cmap('jet')
    if units == 'g':
        cmin = 0
        cmax = 0.1
    else:
        cmin = 0
        cmax = 20000
    
    cs = pmv.plot_array(conc_dict['conc_%dd'%(Times[2])],
                        cmap=cmap,vmin=cmin,vmax=cmax) # concentration colourmap
    cbar = plt.colorbar(cs,format='%.0e') # colour bar
    if units == 'g':
        cbar.ax.set_ylabel('C [$\mu$g g$^-$$^1$]')
    else:
        cbar.ax.set_ylabel('C [$\mu$g L$^-$$^1$]')
        
    # plot the spill site location
    ax3.plot(wel_most_loc_xcoord, wel_most_loc_ycoord,'o',c='r',label='Spill site',
             markersize=mS)
    
    # setting labels
    ax3.set_xlabel('X-coordinate [m]')
    ax3.set_ylabel('Y-coordinate [m]')
    ax3.set_xlim(wel_most_loc_xcoord-50,wel_most_loc_xcoord+100)
    ax3.set_ylim(wel_most_loc_ycoord-85,wel_most_loc_ycoord+60) 
    tday = Times[2]
    titleText = "Concentration at t = %i" % tday+' days'
    ax3.set_title(titleText,loc='left')    
    
    
    #fig = plt.figure(figsize=(14,10))
    ax4 = fig.add_subplot(2, 2, 4)
    # flopy plot object
    pmv = flopy.plot.PlotMapView(model=mf, layer=0)
    # plot grid
    lc = pmv.plot_grid() # grid
    
    # plot concentration
    cmap = plt.get_cmap('jet')
    if units == 'g':
        cmin = 0
        cmax = 0.1
    else:
        cmin = 0
        cmax = 20000
    
    cs = pmv.plot_array(conc_dict['conc_%dd'%(Times[3])],
                        cmap=cmap,vmin=cmin,vmax=cmax) # concentration colourmap
    cbar = plt.colorbar(cs,format='%.0e') # colour bar
    if units == 'g':
        cbar.ax.set_ylabel('C [$\mu$g g$^-$$^1$]')
    else:
        cbar.ax.set_ylabel('C [$\mu$g L$^-$$^1$]')
        
    # plot the spill site location
    ax4.plot(wel_most_loc_xcoord, wel_most_loc_ycoord,'o',c='r',label='Spill site',
             markersize=mS)
    
    # setting labels
    ax4.set_xlabel('X-coordinate [m]')
    ax4.set_ylabel('Y-coordinate [m]')
    ax4.set_xlim(wel_most_loc_xcoord-50,wel_most_loc_xcoord+100)
    ax4.set_ylim(wel_most_loc_ycoord-85,wel_most_loc_ycoord+60) 
    tday = Times[3]
    titleText = "Concentration at t = %i" % tday+' days'
    ax4.set_title(titleText,loc='right')
    
    plt.tight_layout()
    
    return None

def Concentration_Cross_View(units):  
    
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(2, 2, 1)
    hds = bf.HeadFile(modelname + '.hds')
    times = hds.get_times()
    head = hds.get_data(totim=times[-1])
    
    modelxsect = flopy.plot.PlotCrossSection(model=mf, line={'Row': wel_most_loc_row})
    cmap = plt.get_cmap('jet')
    if units == 'g':
        cmin = 0
        cmax = 0.1
    else:
        cmin = 0
        cmax = 50
    hvc = modelxsect.plot_array(np.array(conc_dict['conc_%dd'%(Times[0])]),
                                cmap=cmap,vmin=cmin,vmax=cmax,head=head)
    cbar = plt.colorbar(hvc) # colour bar
    if units == 'g':
        cbar.ax.set_ylabel('C [$\mu$g g$^-$$^1$]')
    else:
        cbar.ax.set_ylabel('C [$\mu$g L$^-$$^1$]')
    
    ax1.grid()
    #ax3.set_xlim(-10,375)
    ax1.set_ylim(-8.8,3.2)
    ax1.set_xlabel('X-coordinate [m]')
    ax1.set_ylabel('Elevation [m]')
    tday = Times[0]
    titleText = "Vertical section: Concentration at t = %i" % tday+' days'
    ax1.set_title(titleText,loc='left')
    

    ax2 = fig.add_subplot(2, 2, 2)
    hds = bf.HeadFile(modelname + '.hds')
    times = hds.get_times()
    head = hds.get_data(totim=times[-1])
    
    modelxsect = flopy.plot.PlotCrossSection(model=mf, line={'Row': wel_most_loc_row})
    cmap = plt.get_cmap('jet')

    hvc = modelxsect.plot_array(np.array(conc_dict['conc_%dd'%(Times[1])]),
                                cmap=cmap,vmin=cmin,vmax=cmax,head=head)
    cbar = plt.colorbar(hvc) # colour bar
    if units == 'g':
        cbar.ax.set_ylabel('C [$\mu$g g$^-$$^1$]')
    else:
        cbar.ax.set_ylabel('C [$\mu$g L$^-$$^1$]')
    
    ax2.grid()
    #ax4.set_xlim(-10,375)
    ax2.set_ylim(-8.8,3.2)
    ax2.set_xlabel('X-coordinate [m]')
    ax2.set_ylabel('Elevation [m]')
    tday = Times[1]
    titleText = "Vertical section: Concentration at t = %i" % tday+' days'
    ax2.set_title(titleText,loc='right')
    
    
    ax3 = fig.add_subplot(2, 2, 3)
    hds = bf.HeadFile(modelname + '.hds')
    times = hds.get_times()
    head = hds.get_data(totim=times[-1])
    
    modelxsect = flopy.plot.PlotCrossSection(model=mf, line={'Row': wel_most_loc_row})
    cmap = plt.get_cmap('jet')

    hvc = modelxsect.plot_array(np.array(conc_dict['conc_%dd'%(Times[2])]),
                                cmap=cmap,vmin=cmin,vmax=cmax,head=head)
    cbar = plt.colorbar(hvc) # colour bar
    if units == 'g':
        cbar.ax.set_ylabel('C [$\mu$g g$^-$$^1$]')
    else:
        cbar.ax.set_ylabel('C [$\mu$g L$^-$$^1$]')
    
    ax3.grid()
    #ax4.set_xlim(-10,375)
    ax3.set_ylim(-8.8,3.2)
    ax3.set_xlabel('X-coordinate [m]')
    ax3.set_ylabel('Elevation [m]')
    tday = Times[2]
    titleText = "Vertical section: Concentration at t = %i" % tday+' days'
    ax3.set_title(titleText,loc='left')
        
    
    ax4 = fig.add_subplot(2, 2, 4)
    hds = bf.HeadFile(modelname + '.hds')
    times = hds.get_times()
    head = hds.get_data(totim=times[-1])
    
    modelxsect = flopy.plot.PlotCrossSection(model=mf, line={'Row': wel_most_loc_row})
    cmap = plt.get_cmap('jet')

    hvc = modelxsect.plot_array(np.array(conc_dict['conc_%dd'%(Times[3])]),
                                cmap=cmap,vmin=cmin,vmax=cmax,head=head)
    cbar = plt.colorbar(hvc) # colour bar
    if units == 'g':
        cbar.ax.set_ylabel('C [$\mu$g g$^-$$^1$]')
    else:
        cbar.ax.set_ylabel('C [$\mu$g L$^-$$^1$]')
    
    ax4.grid()
    #ax4.set_xlim(-10,375)
    ax4.set_ylim(-8.8,3.2)
    ax4.set_xlabel('X-coordinate [m]')
    ax4.set_ylabel('Elevation [m]')
    tday = Times[3]
    titleText = "Vertical section: Concentration at t = %i" % tday+' days'
    ax4.set_title(titleText,loc='right')
    
    plt.tight_layout()
    
    return None

def Plot_Map():
    
    # Plan view
    fig = plt.figure(figsize=(10,12))
    ax = fig.add_subplot(1, 1, 1)
    
    hds = bf.HeadFile(modelname+'.hds')
    times = hds.get_times()
    head = hds.get_data(totim=times[-1])
    levels = np.linspace(-1, 0.3, 10)
    #levels = np.linspace(-1, 3, 16)
    
    cbb = bf.CellBudgetFile(modelname+'.cbc') # read budget file
    kstpkper_list = cbb.get_kstpkper()
    # cbb.textlist to get a list of data texts
    frf = cbb.get_data(text='FLOW RIGHT FACE', totim=times[-1])[0]
    fff = cbb.get_data(text='FLOW FRONT FACE', totim=times[-1])[0]
    
    # flopy plot object
    pmv = flopy.plot.PlotMapView(model=mf, layer=0)
    
    # plot grid
    lc = pmv.plot_grid() # grid
    
    # plot contour
    cs = pmv.contour_array(head, levels=levels) # head contour
    plt.clabel(cs, fontsize=fS, fmt='%1.3f') # contour label
    
    # plot discharge quiver
    quiver = pmv.plot_discharge(frf, fff, head=head)
    
    # plot ibound
    qm = pmv.plot_ibound()
    
    #plot well locations
    plt.plot(wel_most_loc_xcoord,
              wel_most_loc_ycoord,'o',markersize=mS,label='WEL')

    ## plot high rechare zone
    #r3_pt = ((r3_col+0.5)*delr, Ly - ((r3_row + 0.5)#*delc))
    #plt.plot(r3_xcoord,r3_ycoord,'cs',markersize=mS,#label='High recharge')
    
    # Specifying and plotting observation points
    
    #obs2_xcoord = 50.
    #obs2_ycoord = 120.
    #obs2_col = int(np.round(obs2_xcoord/delc))
    #obs2_row = int(np.round((Ly-obs2_ycoord)/delr))
    
    plt.plot(obs2_xcoord,obs2_ycoord,'kx',markersize=mS,marker='v',label='Obs2')
    
    plt.plot(obs1_xcoord,obs1_ycoord,'cs',markersize=mS,marker='x',label='Obs1')

    plt.plot(obs3_xcoord,obs3_ycoord,'cs',markersize=mS,marker='*',label='Obs3')

    #plt.plot(obs4_xcoord,obs4_ycoord,'cs',markersize=mS,marker='v',label='Obs3')        

    
    # setting labels
    ax.set_xlabel('X-coordinate [m]')
    ax.set_ylabel('Y-coordinate [m]')
    ax.set_title('Plan view - Flow field',loc='left')
    ax.legend(loc='upper left')
    plt.show()
    
    return None

def Time_Series_Plot():
    
    fig = plt.figure(figsize=(50,10))
    ax1 = fig.add_subplot(3,1,1)
    hds = bf.HeadFile(modelname + '.hds')
    times = hds.get_times()
    head = hds.get_data(totim=times[-1])
    head_all = hds.get_alldata()
    lyr = 0 # Defines layer to use when plotting
    
    h_obs1 = head_all[:,lyr,obs1_row,obs1_col]
    h_obs3 = head_all[:,lyr,obs3_row,obs3_col]
    h_obs2 = head_all[:,lyr,obs2_row,obs2_col]
    h_obs4 = head_all[:,lyr,obs4_row,obs4_col]
    
    # TODO: the location of the wrow, wcol
    h_Spill = head_all[:,lyr,wel_most_loc_row,wel_most_loc_col]
    
    ax1.plot(times,h_obs1,'-',c='c',label='Obs 1',
            markersize=mS, linewidth=lW)
    ax1.plot(times,h_obs2,'-',c='y',label='Obs 2',
            markersize=mS, linewidth=lW)
    ax1.plot(times,h_obs3,'-',c='r',label='Obs 3',
            markersize=mS, linewidth=lW)
    ax1.plot(times,h_Spill,'-',c='b',label='Spill site',
            markersize=mS, linewidth=lW)
    # ax1.plot(times,h_obs4,'-',c='k',label='Obs 4',
    #         markersize=mS, linewidth=lW)
    
    ax1.grid()
    
    
    #TODO: change
    ax1.set_xlim(0,12000)
    #ax1.set_ylim(0,0.06)
    
    #ax1.set_ylim(0.00001,1000)
    #ax1.set_yscale('log')
    ax1.set_xlabel('Elapsed time [days]')
    ax1.set_ylabel('Hydraulic head [m]')
    titleText = 'Hydraulic heads: Layer %i' % lyr
    ax1.set_title(titleText,loc='left')
    ax1.legend(loc='upper right')
    
    
    ax2 = fig.add_subplot(3,1,2)
    hds = bf.HeadFile(modelname + '.hds')
    times = hds.get_times()
    head = hds.get_data(totim=times[-1])
    head_all = hds.get_alldata()
    
    lyr = 1 # Defines layer to use when plotting
    
    # h_obs1 = head_all[:,lyr,obs1_row,obs1_col]
    # h_obs2 = head_all[:,lyr,obs2_row,obs2_col]
    # h_obs3 = head_all[:,lyr,obs3_row,obs3_col]
    #h_Spill = head_all[:,lyr,obsSpill_row,obsSpill_col]
    
    # TODO: the location of the wrow, wcol
    h_Spill = head_all[:,lyr,wel_most_loc_row,wel_most_loc_col]
    
    
    #h_obs4 = head_all[:,lyr,obs4_row,obs4_col]
    
    # ax2.plot(times,h_obs1,'-',c='c',label='Obs 1',
    #         markersize=mS, linewidth=lW)
    # ax2.plot(times,h_obs2,'-',c='y',label='Obs 2',
    #         markersize=mS, linewidth=lW)
    # ax2.plot(times,h_obs3,'-',c='r',label='Obs 3',
    #         markersize=mS, linewidth=lW)
    ax2.plot(times,h_Spill,'-',c='b',label='Spill site',
            markersize=mS, linewidth=lW)
    # ax2.plot(times,h_obs4,'-',c='k',label='Obs 4',
    #         markersize=mS, linewidth=lW)
    
    ax2.grid()
    
    #TODO: change
    #ax2.set_xlim(0,nper)
    #ax2.set_ylim(0,0.06)
    
    
    #ax2.set_ylim(0.00001,1000)
    #ax2.set_yscale('log')
    ax2.set_xlabel('Elapsed time [days]')
    ax2.set_ylabel('Hydraulic head [m]')
    titleText = 'Hydraulic heads: Layer %i' % lyr
    ax2.set_title(titleText,loc='left')
    ax2.legend(loc='upper right')
    
    
    ax3 = fig.add_subplot(3,1,3)
    
    hds = bf.HeadFile(modelname + '.hds')
    times = hds.get_times()
    head = hds.get_data(totim=times[-1])
    head_all = hds.get_alldata()
    
    lyr = 2 # Defines layer to use when plotting
    
    # h_obs1 = head_all[:,lyr,obs1_row,obs1_col]
    # h_obs2 = head_all[:,lyr,obs2_row,obs2_col]
    # h_obs3 = head_all[:,lyr,obs3_row,obs3_col]
    #h_Spill = head_all[:,lyr,obsSpill_row,obsSpill_col]
    
    # TODO: the location of the wrow, wcol
    h_Spill = head_all[:,lyr,wel_most_loc_row,wel_most_loc_col]
    
    
    
    
    #h_obs4 = head_all[:,lyr,obs4_row,obs4_col]
    
    # ax3.plot(times,h_obs1,'-',c='c',label='Obs 1',
    #         markersize=mS, linewidth=lW)
    # ax3.plot(times,h_obs2,'-',c='y',label='Obs 2',
    #         markersize=mS, linewidth=lW)
    # ax3.plot(times,h_obs3,'-',c='r',label='Obs 3',
    #         markersize=mS, linewidth=lW)
    ax3.plot(times,h_Spill,'-',c='b',label='Spill site',
            markersize=mS, linewidth=lW)
    # ax3.plot(times,h_obs4,'-',c='k',label='Obs 4',
    #         markersize=mS, linewidth=lW)
    
    ax3.grid()
    
    #TODO: change
    #ax3.set_xlim(0,nper)
    #ax3.set_ylim(0,0.06)
    
    #ax3.set_ylim(0.00001,1000)
    #ax3.set_yscale('log')
    ax3.set_xlabel('Elapsed time [days]')
    ax3.set_ylabel('Hydraulic head [m]')
    titleText = 'Hydraulic heads: Layer %i' % lyr
    ax3.set_title(titleText,loc='left')
    ax3.legend(loc='upper right')
    # preventing overlap
    plt.tight_layout()
    return None


def Plot_Conc_Break_Down():
    ucnobj = flopy.utils.UcnFile('MT3D001.UCN')
    times = ucnobj.get_times()
    conc = ucnobj.get_alldata()
    
    obs1_data = conc[:,:,obs1_row,obs1_col][:,0]    # Observation at high recharge cell
    obs2_data = conc[:,:,obs2_row,obs2_col][:,0] # Observation well 2
    obs3_data = conc[:,:,obs3_row,obs3_col][:,0]
    obs4_data = conc[:,:,wel_most_loc_row,wel_most_loc_col][:,0] # Observation at extraction well

    fig = plt.figure(figsize = (50,10))
    ax = fig.add_subplot(1,1,1)
    
    # Panel 1 (Concentration time series)
    ax.plot(times,obs1_data,'-^',c='r',label='obs1',
            markersize=mS, linewidth=lW)
    ax.plot(times,obs2_data,'-s',c='k',label='Obs2',
            markersize=mS, linewidth=lW)
    ax.plot(times,obs3_data,'-o',c='b',label='Obs3',
            markersize=mS, linewidth=lW)
    ax.plot(times,obs4_data,'-o',c='g',label='spill point',
            markersize=mS, linewidth=lW)

    ax.grid()
    ax.set_xlim(-100,sum(perlen))
    ax.set_ylim(0.0001,10**5)
    ax.set_yscale('log')
    ax.set_xlabel('Elapsed time [days]')
    ax.set_ylabel('Concentration [g L$^-$$^1$]')
    ax.set_title('Contaminant concentration',loc='left')
    ax.legend(loc='upper right')
    
    return None


# The file path
Merged = '/Users/yuchenli/Documents/UQ/CIVL4140/pymake-master/examples/MT3DMS_model/Test_flopy/resoluted.tif'
SpillSite = '/Users/yuchenli/Documents/UQ/CIVL4140/pymake-master/examples/MT3DMS_model/Test_flopy/resolutedspillsite.tif'
MF2005_exeLocation = '/Users/yuchenli/Documents/UQ/CIVL4140/pymake-master/examples/mf2005'
MT3DMS_exeLocation = '/Users/yuchenli/Documents/UQ/CIVL4140/pymake-master/examples/mt3dms'

modelname = 'Final_Project'
mf = flopy.modflow.Modflow(
    modelname,
    exe_name=MF2005_exeLocation
)

# Get the data from tiff files
demDs = gdal.Open(Merged)
spillSite = gdal.Open(SpillSite)
geot = demDs.GetGeoTransform()

demData = demDs.GetRasterBand(1).ReadAsArray()
spillData = spillSite.GetRasterBand(1).ReadAsArray()
demNd = demDs.GetRasterBand(1).GetNoDataValue()


# Modle Domain

nlay = 3
nrow = demDs.RasterYSize
ncol = demDs.RasterXSize
ztop = np.zeros((nrow, ncol), dtype = np.float32)+3
zbotm = np.zeros((nlay, nrow, ncol), dtype = np.float32)
zbotm[0,:,:] = -1.5 # upper layer, sand
zbotm[1,:,:] = -1.5-0.3 # middle layer, clay
zbotm[2,:,:] = -7 # lower layer, silt and clay
delr = geot[1]
delc = geot[1]
itmuni = 4 # Days
lenuni = 2 # M



# # =============Get the ibound way 1 ===============

# ibound = np.zeros(demData.shape, dtype=np.int32)
# ibound[demData == 255] = -1 #constant head
# ibound[demData == 0] = 1 #active


## ==============Get the ibound way 2 ===============
#import pandas as pd
ibound=pd.read_csv('/Users/yuchenli/Documents/UQ/CIVL4140/pymake-master/examples/MT3DMS_model/Test_flopy/ibound.csv'
                   ,header=None,sep=',').values


#np.savetxt("/Users/yuchen/Desktop/ibound.csv", ibound, delimiter=',')


# Get the constant head

# 2: drainage, 3: coast 4: link
# strt = np.zeros(demData.shape, dtype=np.int32)
# strt[demData == 255] = 0 # constant head = 0

strt = pd.read_csv('/Users/yuchenli/Documents/UQ/CIVL4140/pymake-master/examples/MT3DMS_model/Test_flopy/StrtMask.csv'
                   ,header=None,sep=',').values
strt[strt == 2] = 0.5
strt[strt == 3] = 0.5
strt[strt == 4] = 0.5


# Simulation time, stress period

nper = 30 # Number of stress periods 385: 2020 01 30
# 10196 days = Nov 2015

# =============== rainfall data and number of days ================
RainfallAVG = pd.read_csv('/Users/yuchenli/Documents/UQ/CIVL4140/pymake-master/examples/MT3DMS_model/Test_flopy/RainfallAVG.csv')

Mdata = RainfallAVG['Date'].values
#Mdata = Mdata.apply(lambda x:x[:2]).tolist() 
#Mdata = pd.Series(Mdata, dtype='int').to_numpy()


perlen = Mdata[:nper] 
nstp = Mdata[:nper]
periodType = RainfallAVG['Type'].values[:nper]
periodType[0] = True

# Creat the discretization object
dis = flopy.modflow.ModflowDis(
    mf, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper,
    perlen=perlen, nstp=nstp, steady=periodType,
    delr=delr, delc=delc, top=ztop, botm=zbotm,
    itmuni=itmuni, lenuni=lenuni
)

# Create BAS object
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

# Hydraulic conducticity of each layer

K_horiz = np.zeros((nlay, nrow, ncol), dtype=np.float32)
K_vert = np.zeros((nlay,nrow,ncol), dtype=np.float32)

K_horiz[0,:,:] = 0.01*29.7*24
K_horiz[1,:,:] = 0.01*0.2*24
K_horiz[2,:,:] = 0.01*0.02*24


K_vert[0,:,:] = 0.01*29.7*24
K_vert[1,:,:] = 0.01*0.2*24
K_vert[2,:,:] = 0.01*0.02*24

sy = np.zeros((nlay,nrow,ncol),dtype=np.float32)+0.1
ss = np.zeros((nlay,nrow,ncol),dtype=np.float32)+1e-4
scoefbool = False

laytyp = [1,1,1] # Sets layer type: 1=uncofnined; 0=confined
laywet = [1,1,1] # 0 = wetting inactive, not 0 = active wetting
layvka = [0, 0, 0] # 0 = Kv values, >0 or <0 = ratio Kh to Kv

# Create LPF object
lpf = flopy.modflow.ModflowLpf(mf, hk=K_horiz, vka=K_vert,
                               laytyp=laytyp, layvka=layvka, laywet=laywet,
                               sy=sy, ss=ss, ipakcb=53)


nrchop=3
rain_Data = RainfallAVG['Rain'] #rainfall for each month
rchF = 0.12 # Recharge factor - % rainfall that becomes recharge

n = nper
recharge = [[] for _ in range(n)]


for i in range (nper):
    recharge[i] = np.ones((nrow,ncol),dtype=np.float32)*rchF*(
            rain_Data[i]/1000) #/Mdata[i]
    
rech_StressPeriod_dat = {}
for i in range(nper):
    rech_StressPeriod_dat[i] = [recharge[i]]
    


# Create recharge object
rch = flopy.modflow.ModflowRch(mf, nrchop=nrchop, ipakcb=None,
                               rech=rech_StressPeriod_dat, irch=0,
                               extension='rch', unitnumber=None,
                               filenames=None)

# Well stress period
temp = np.where(spillData == 255) # possible location top->botm
Loc_index = np.dstack((temp[0], temp[1])).reshape(-1,2) #row, col row is top -> botm

cord = ibound.shape
w_pumping_rate = 0.00275/30 #m^3/day

# Get the stress period data for well package and source/sink package
stress_period_data = {} #For well
ssm_data = {} # For source/sink package
itype = flopy.mt3d.Mt3dSsm.itype_dict()
C_in = 100 # density kg/m3
welpt = [] # wel location
# y_distance = cord[0]*delr# how much cells (high) the map is
for i in range(nper):
    index = np.random.randint(Loc_index.shape[0])
    welpt.append(Loc_index[index])
    ssm_data[i] = [(0, Loc_index[index][0], Loc_index[index][1], C_in, itype['WEL'])]
    stress_period_data[i] = [0, Loc_index[index][0], Loc_index[index][1], w_pumping_rate]

welindex = []
for i in welpt:
    welindex.append(i[0])

welpt_most_frequent = np.argmax(np.bincount(welindex))

wel_most_loc_index = welindex.index(welpt_most_frequent)
wel_most_loc = welpt[wel_most_loc_index] #row, col row is top->bom
wel_most_loc_row = wel_most_loc[0] # row top->botm
wel_most_loc_col = wel_most_loc[1] # col botm->top
wel_most_loc_xcoord = wel_most_loc_col*delr # left -> right
wel_most_loc_ycoord = (cord[0]-wel_most_loc_row)*delr #botm -> top




# Well package
wel = flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)

# OC package
sp_data = {}
for kper in range(nper):
    for kstp in range(nstp[kper]):
        sp_data[(kper, kstp)] = ['save head','save drawdown','save budget',
                'print head','print budget']

# Create Output Control (OC) package
oc = flopy.modflow.ModflowOc(mf, stress_period_data=sp_data, compact=True)

# Solver
pcg = flopy.modflow.mfpcg.ModflowPcg(mf, mxiter=1000, iter1=500, npcond=1,
                                      hclose=1e-05, rclose=1e-05, relax=1.0,
                                      nbpol=0, iprpcg=0, mutpcg=3, damp=1.0,
                                      dampt=1.0, ihcofadd=0, extension='pcg',
                                      unitnumber=None, filenames=None)

# Link to the MT3DMS LMT package
lmt = flopy.modflow.ModflowLmt(mf, output_file_name='mt3d_link.ftl')

# Check model for basic errors
chk = mf.check(verbose=True,level=1)

# Write the MODFLOW model input files
mf.write_input()

# Run the MODFLOW model
success, buff = mf.run_model(report=True)

if not success:
    raise Exception('MODFLOW did not terminate normally.')

# Creat MT3DMS object
mt = flopy.mt3d.Mt3dms(modflowmodel=mf, 
                       modelname=modelname, 
                       exe_name=MT3DMS_exeLocation,
                       ftlfilename='mt3d_link.ftl')


prsity=0.39


# Boundary types: 0=inactive, <0=constant conc, >0=active cell | same as ibound
#icbund_array = np.ones((nlay, nrow, ncol), dtype=np.float32)
# icbund_array = np.zeros(demData.shape, dtype=np.int32)
# icbund_array[demData == 255] = -1 #constant head
# icbund_array[demData == 0] = 1 #active

icbund_array = ibound




# Starting concentration
sconc_array = np.zeros((nlay, nrow, ncol), dtype=np.float32)

# Specifying initial concentration at spill location
# obsSpill_xcoord = 144
# obsSpill_ycoord = 264
# obsSpill_col = int(np.round(obsSpill_xcoord/delc))
# obsSpill_row = int(np.round((cord[0]*delr-obsSpill_ycoord)/delr))

sconc_array[0, wel_most_loc[0], wel_most_loc[1]] = 1056.2 #initial C upper layer

# Set up BTN package
nper=nper #Number of stress periods
perlen=perlen #length of stress period
nstp=nstp #Number of timesteps in each stress period
tsmult=1.0 # Time step multiplier
dt0=0 #Transport step size
ttsmax=0 #maximum transport step size
nprs=-1 #options: >0 save at "timprs", 0=only end step saved, <0 save at transport step

btn = flopy.mt3d.Mt3dBtn(mt, prsity=0.3, icbund=icbund_array,
                         sconc=sconc_array, ncomp=1, perlen=perlen, nper=nper,
                         nstp=nstp, tsmult=tsmult, nprs=nprs,
                         nprobs = 10, cinact = -1, chkmas=True)

# advection package
adv = flopy.mt3d.Mt3dAdv(mt, mixelm=-1, percel=0.75)


alpha = 10 # Longitudinal dispersivity [l] (alpha)
ratio_ht = 0.1 # Ratio of horizontal to transverse dipsersivity 
ratio_hv = 0.1 # Ratio of horizontal to vertical dipsersivity 
DmCoeff = 2.5e-5 # Molecular diffusion coefficient [L2/T]

dsp = flopy.mt3d.Mt3dDsp(mt, al=alpha, trpt=ratio_ht, trpv=ratio_ht,
                         dmcoef=DmCoeff)

# source/sink package
ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=ssm_data)

# ===== Matrix solver package =============================================
gcg = flopy.mt3d.Mt3dGcg(mt, cclose=1e-6)

# ===== Write MT3DMS input ================================================
mt.write_input()

# ===== Run MT3DMS ========================================================
mt.run_model()





# Plotting section ========================================================
mS = 12 # Used to set marker size
lW = 3 # Used to set linewidth
fS = 28 # Used to set font size
plt.rcParams['font.family'] = 'Times New Roman' # Globally sets the font type
plt.rc('font',size=fS)
plt.rc('axes',titlesize=fS)
plt.rc('axes',labelsize=fS)
plt.rc('xtick',labelsize=fS)
plt.rc('ytick',labelsize=fS)
plt.rc('legend',fontsize=fS)
plt.rc('figure',titlesize=fS)


# Specifying and plotting observation points
obs1_xcoord = 150
obs1_ycoord = 250
obs1_col = int(np.round(obs1_xcoord/delc))
obs1_row = int(np.round((cord[0]*5-obs1_ycoord)/delr))

obs2_xcoord = 160
obs2_ycoord = 240
obs2_col = int(np.round(obs2_xcoord/delc))
obs2_row = int(np.round((cord[0]*5-obs2_ycoord)/delr))

obs3_xcoord = 180
obs3_ycoord = 220
obs3_col = int(np.round(obs3_xcoord/delc))
obs3_row = int(np.round((cord[0]*5-obs3_ycoord)/delr))

# obs4_xcoord = 125
# obs4_ycoord = 130
# obs4_col = int(np.round(obs4_xcoord/delc))
# obs4_row = int(np.round((cord[0]*5-obs4_ycoord)/delr))



units = 'L'
Times = [30, 60, 300, sum(perlen)]
#Times = [30,100,1000,1500]
conc_dict = Get_Conc(Times, units)
#Plot_Map()
#Time_Series_Plot()  
Concentration_Plan_View(conc_dict, units)
#Concentration_Cross_View(units)






# a = conc_dict['conc_%dd'%(sum(perlen))][0]

    
# np.savetxt("conc_1_L.csv", a, delimiter=',')

    
# b = conc_dict['conc_%dd'%(10195)][0] 

# np.savetxt('conc_1_L_2015NOV.csv',b, delimiter=',')
        










