# -*- coding: utf-8 -*-
# import libs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import flopy
import flopy.utils.binaryfile as bf

# Setting up simulation 
modelName = 'Project' # Modle name used when creating output files
MF2005_exeLocation = '/Users/yuchenli/Documents/UQ/CIVL4140/pymake-master/examples/mf2005'
MT3DMS_exeLocation = '/Users/yuchenli/Documents/UQ/CIVL4140/pymake-master/examples/mt3dms'

# Creat MODFLOW model object
mf = flopy.modflow.Modflow(
    modelName,
    exe_name=MF2005_exeLocation
)

# Model domain and grid definition
Lx = 170.
Ly = 170.
nlay = 3
nrow = 50
ncol = 50
delr = Lx/ncol
delc = Ly/nrow

# boundary condition
def sector_boundary(nlay, nrow, ncol, radius, drop = False):
    ibound = np.ones((nlay, nrow, ncol), dtype = np.int32)*-1
    y,x = np.ogrid[0:nrow, 0:ncol]
    mask = x**2+y**2 <= radius**2
#    for i , element in enumerate(mask):
#        index = []
#        print(i,element)
#        for j in range(ncol):
#            if mask[i,j] == False:
#                index.append(i)
#                index.append(j)
#

    mask = np.vstack((mask, mask, mask)).reshape(nlay, nrow, ncol)
    ibound[mask] = 1

    return ibound, mask

ibound, mask = sector_boundary(nlay, nrow, ncol, ncol, drop = False)
#spill_loc_x = round((ncol/Lx)*50)
#spill_loc_y = round((nrow/Ly)*50)
#ibound[:,spill_loc_y,spill_loc_x] = -1
#bound_loc_x = 40
#bound_loc_y = 120
#obs2_col = int(np.round(bound_loc_x/delc))
#obs2_row = int(np.round((Ly-bound_loc_y)/delr))

#ibound[:, obs2_row:obs2_col, 0] = -1
# ibound[:,0,:] = 0
# ibound[:,:,0] = 0
# TODO: Clay layer = 0.3m thick, add 0.3?
# TODO: The ibound of the spill site in which layer? all layer or just top two?
strt = np.zeros((nlay,nrow,ncol), dtype=np.float32) +1.46 # (HAT)
strt[mask] = 0 # constant head right hand
#strt[:,0,:] = 0 #(Constant head) 
strt[:,:,0] = 0
#strt[:,spill_loc_y,spill_loc_x] = 0.5

# Setting up 3D grid
# TODO: plus 3 correct? zbotm correct?
ztop = np.zeros((nrow, ncol), dtype = np.float32)+3
zbotm = np.zeros((nlay, nrow, ncol), dtype = np.float32)
zbotm[0,:,:] = -1.5 # upper layer, sand
zbotm[1,:,:] = -1.5-0.3 # middle layer, clay
zbotm[2,:,:] = -7 # lower layer, silt and clay

# Time and length units
itmuni = 4
lenuni = 2

# TODO: first simulate 1 year, assume slug release, not steady
nper=13 # Number of stress periods, Default=1
perlen = [1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] #1st steady state
nstp = [1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
periodType = [True, False, False, False, False, False, False, False, False, False,
          False, False, False]

# Creat the discretization object
dis = flopy.modflow.ModflowDis(
    mf, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper,
    perlen=perlen, nstp=nstp, steady=periodType,
    delr=delr, delc=delc, top=ztop, botm=zbotm,
    itmuni=itmuni, lenuni=lenuni
)

# Create BAS object
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

#check grid
modelmap = flopy.plot.PlotMapView(model = mf, layer = 0)
grid = modelmap.plot_grid()
ib = modelmap.plot_ibound()
#spill_plot = modelmap.plot_bc(ftype='WEL')
#chd_plot = modelmap.plot_bc(ftype='GHB')

plt.show(modelmap)

# add LPF package
K_horiz = np.zeros((nlay, nrow, ncol), dtype=np.float32)
K_vert = np.zeros((nlay,nrow,ncol), dtype=np.float32)

# TODO: hydraulic conductivity of each layer??? 
K_horiz[0,:,:] = 0.01*29.7*24
K_horiz[1,:,:] = 0.01*0.2*24
K_horiz[2,:,:] = 0.01*0.02*24


K_vert[0,:,:] = 0.01*29.7*24
K_vert[1,:,:] = 0.01*0.2*24
K_vert[2,:,:] = 0.01*0.02*24

# Sy and Ss
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

# TODO: recharge package this script is using data from 3D TransientTest
nrchop=3

rain_Data = [158,175,139,90,99,71,63,43,35,94,97,126,1048.9] #Month + Ann Av
rchF = 0.5 # Recharge factor - % rainfall that becomes recharge

recharge_AnnualAvg = np.ones((nrow,ncol),
                             dtype=np.float32)*rchF*(rain_Data[-1]/1000/365)
recharge_Jan = np.ones((nrow,ncol),
                             dtype=np.float32)*rchF*(rain_Data[0]/1000/31)
recharge_Feb = np.ones((nrow,ncol),
                             dtype=np.float32)*rchF*(rain_Data[1]/1000/28)
recharge_Mar = np.ones((nrow,ncol),
                             dtype=np.float32)*rchF*(rain_Data[2]/1000/31)
recharge_Apr = np.ones((nrow,ncol),
                             dtype=np.float32)*rchF*(rain_Data[3]/1000/30)
recharge_May = np.ones((nrow,ncol),
                             dtype=np.float32)*rchF*(rain_Data[4]/1000/31)
recharge_Jun = np.ones((nrow,ncol),
                             dtype=np.float32)*rchF*(rain_Data[5]/1000/30)
recharge_Jul = np.ones((nrow,ncol),
                             dtype=np.float32)*rchF*(rain_Data[6]/1000/31)
recharge_Aug = np.ones((nrow,ncol),
                             dtype=np.float32)*rchF*(rain_Data[7]/1000/31)
recharge_Sep = np.ones((nrow,ncol),
                             dtype=np.float32)*rchF*(rain_Data[8]/1000/30)
recharge_Oct = np.ones((nrow,ncol),
                             dtype=np.float32)*rchF*(rain_Data[9]/1000/31)
recharge_Nov = np.ones((nrow,ncol),
                             dtype=np.float32)*rchF*(rain_Data[10]/1000/30)
recharge_Dec = np.ones((nrow,ncol),
                             dtype=np.float32)*rchF*(rain_Data[11]/1000/31)

rech_StressPeriod_dat = {0: recharge_AnnualAvg, 1: recharge_Jan, 2: recharge_Feb,
                      3: recharge_Mar, 4: recharge_Apr, 5: recharge_May,
                      6: recharge_Jun, 7: recharge_Jul, 8: recharge_Aug, 
                      9: recharge_Sep, 10: recharge_Oct, 11: recharge_Nov, 
                      12: recharge_Dec}

# Create recharge object
rch = flopy.modflow.ModflowRch(mf, nrchop=nrchop, ipakcb=None,
                               rech=rech_StressPeriod_dat, irch=0,
                               extension='rch', unitnumber=None,
                               filenames=None)

# WELL package
w_pumping_rate = 0.00275/30
spill_loc_x = 50
spill_loc_y = 120

w_col = int(np.round(spill_loc_x/delc))
w_row = int(np.round((Ly-spill_loc_y)/delr))

# stress period
w_sp1 = [0, w_row, w_col, w_pumping_rate]
stress_period_data = {0: w_sp1}

# add well package
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
                       modelname=modelName, 
                       exe_name=MT3DMS_exeLocation,
                       ftlfilename='mt3d_link.ftl')

# TODO: porosity is what??
prsity=0.39


# Boundary types: 0=inactive, <0=constant conc, >0=active cell | same as ibound
icbund_array = np.ones((nlay, nrow, ncol), dtype=np.float32)

# Starting concentration
sconc_array = np.zeros((nlay, nrow, ncol), dtype=np.float32)

# Specifying initial concentration at spill location
obsSpill_xcoord = 50
obsSpill_ycoord = 120
obsSpill_col = int(np.round(obsSpill_xcoord/delc))
obsSpill_row = int(np.round((Ly-obsSpill_ycoord)/delr))

sconc_array[0, obsSpill_row, obsSpill_col] = 750 # Sets inital c in upper layer

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




# set up btn
#btn = flopy.mt3d.Mt3dBtn(mt, prsity = prsity, ncomp = 1, perlen = perlen, nstp = nstp, tsmult = 1, nprs = -1, nprobs = 10, cinact = -1, chkmas = True)

# advection package
adv = flopy.mt3d.Mt3dAdv(mt, mixelm=-1, percel=0.75)

# TODO: where can i find these variables??
alpha = 10 # Longitudinal dispersivity [l] (alpha)
ratio_ht = 0.1 # Ratio of horizontal to transverse dipsersivity 
ratio_hv = 0.1 # Ratio of horizontal to vertical dipsersivity 
DmCoeff = 2.5e-5 # Molecular diffusion coefficient [L2/T]

dsp = flopy.mt3d.Mt3dDsp(mt, al=alpha, trpt=ratio_ht, trpv=ratio_ht,
                         dmcoef=DmCoeff)

# TODO: source/sink package
ssm_data = {}
itype = flopy.mt3d.Mt3dSsm.itype_dict()
# Format is [layer, row column, concentration, input type 'WEL' = well package]
C_in = 1
ssm_data[0] = [(0, spill_loc_y, spill_loc_x, C_in, itype['WEL'])] 
ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=ssm_data)

# ===== Matrix solver package =============================================
gcg = flopy.mt3d.Mt3dGcg(mt, cclose=1e-6)

# ===== Write MT3DMS input ================================================
mt.write_input()

# ===== Run MT3DMS ========================================================
mt.run_model()

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
obs1_xcoord = 55
obs1_ycoord = 110
obs1_col = int(np.round(obs1_xcoord/delc))
obs1_row = int(np.round((Ly-obs1_ycoord)/delr))

obsSpill_xcoord = 50
obsSpill_ycoord = 120
obsSpill_col = int(np.round(obsSpill_xcoord/delc))
obsSpill_row = int(np.round((Ly-obsSpill_ycoord)/delr))

obs3_xcoord = 80
obs3_ycoord = 100
obs3_col = int(np.round(obs3_xcoord/delc))
obs3_row = int(np.round((Ly-obs3_ycoord)/delr))

obs4_xcoord = 120
obs4_ycoord = 80
obs4_col = int(np.round(obs4_xcoord/delc))
obs4_row = int(np.round((Ly-obs4_ycoord)/delr))


# Plan view
fig = plt.figure(figsize=(10,12))
ax = fig.add_subplot(1, 1, 1)

hds = bf.HeadFile(modelName + '.hds')
times = hds.get_times()
head = hds.get_data(totim=times[-1])
levels = np.linspace(-1, 2.1, 6)

cbb = bf.CellBudgetFile(modelName+'.cbc') # read budget file
kstpkper_list = cbb.get_kstpkper()
# cbb.textlist to get a list of data texts
frf = cbb.get_data(text='FLOW RIGHT FACE', totim=times[-1])[0]
fff = cbb.get_data(text='FLOW FRONT FACE', totim=times[-1])[0]

# flopy plot object
pmv = flopy.plot.PlotMapView(model=mf, layer=0)

# plot grid
lc = pmv.plot_grid() # grid

# plot contour
cs = pmv.contour_array(head, levels=np.linspace(-1, 3, 16)) # head contour
plt.clabel(cs, fontsize=fS, fmt='%1.1f') # contour label

# plot discharge quiver
quiver = pmv.plot_discharge(frf, fff, head=head)

# plot ibound
qm = pmv.plot_ibound()

# plot well locations
#w1_pt = [((w_col+0.5)*delr, Ly - ((w_row + 0.5)*delc))]
w1_pt = [50,120]
plt.plot(w1_pt[0],w1_pt[1],'yo',markersize=mS,label='Well')

## plot high rechare zone
#r3_pt = ((r3_col+0.5)*delr, Ly - ((r3_row + 0.5)#*delc))
plt.plot(obs1_xcoord,obs1_ycoord,'cs',markersize=mS,marker='x',label='Obs1')

plt.plot(obs3_xcoord,obs3_ycoord,'cs',markersize=mS,marker='*',label='Obs2')

plt.plot(obs4_xcoord,obs4_ycoord,'cs',markersize=mS,marker='v',label='Obs3')        

         
# Specifying and plotting observation points

#obs2_xcoord = 50.
#obs2_ycoord = 120.
#obs2_col = int(np.round(obs2_xcoord/delc))
#obs2_row = int(np.round((Ly-obs2_ycoord)/delr))

#plt.plot(obs2_xcoord,obs2_ycoord,'kx',markersize=mS,l3bel='Obs')


# setting labels
ax.set_xlabel('X-coordinate [m]')
ax.set_ylabel('Y-coordinate [m]')
ax.set_title('Plan view - Flow field',loc='left')
ax.legend(loc='upper left')
plt.show()

#==============================================================================
# ===== POST-PROCESSING MODEL RESULTS =========================================
#==============================================================================

# 4.1 ===== Setting plot formatting (global settings - applies to all plots) ==
mS = 12 # Used to set marker size
lW = 3 # Used to set linewidth
fS = 18 # Used to set font size
plt.rcParams['font.family'] = 'Times New Roman' # Globally sets the font type
plt.rc('font',size=fS)
plt.rc('axes',titlesize=fS)
plt.rc('axes',labelsize=fS)
plt.rc('xtick',labelsize=fS)
plt.rc('ytick',labelsize=fS)
plt.rc('legend',fontsize=fS)
plt.rc('figure',titlesize=fS)





# 4.2 ===== Time series plots of water table at spill location ===============
fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(1,3,1)

hds = bf.HeadFile(modelName + '.hds')
times = hds.get_times()
head = hds.get_data(totim=times[-1])
head_all = hds.get_alldata()

lyr = 0 # Defines layer to use when plotting

h_obs1 = head_all[:,lyr,obs1_row,obs1_col]
h_obs3 = head_all[:,lyr,obs3_row,obs3_col]
h_Spill = head_all[:,lyr,obsSpill_row,obsSpill_col]
h_obs4 = head_all[:,lyr,obs4_row,obs4_col]

ax1.plot(times,h_obs1,'-',c='c',label='Obs 1',
        markersize=mS, linewidth=lW)
ax1.plot(times,h_obs3,'-',c='r',label='Obs 2',
        markersize=mS, linewidth=lW)
ax1.plot(times,h_Spill,'-',c='b',label='Spill site',
        markersize=mS, linewidth=lW)
ax1.plot(times,h_obs4,'-',c='k',label='Obs 3',
        markersize=mS, linewidth=lW)
ax1.grid()
ax1.set_xlim(-10,375)
#ax1.set_ylim(1.45,2)
#ax1.set_ylim(0.00001,1000)
#ax1.set_yscale('log')
ax1.set_xlabel('Elapsed time [days]')
ax1.set_ylabel('Hydraulic head [m]')
titleText = 'Hydraulic heads: Layer %i' % lyr
ax1.set_title(titleText,loc='left')
ax1.legend(loc='upper right')


ax2 = fig.add_subplot(1,3,2)

hds = bf.HeadFile(modelName + '.hds')
times = hds.get_times()
head = hds.get_data(totim=times[-1])
head_all = hds.get_alldata()

lyr = 1 # Defines layer to use when plotting

h_obs1 = head_all[:,lyr,obs1_row,obs1_col]
h_obs3 = head_all[:,lyr,obs3_row,obs3_col]
h_Spill = head_all[:,lyr,obsSpill_row,obsSpill_col]
h_obs4 = head_all[:,lyr,obs4_row,obs4_col]

ax2.plot(times,h_obs1,'-',c='c',label='Obs 1',
        markersize=mS, linewidth=lW)
ax2.plot(times,h_obs3,'-',c='r',label='Obs 2',
        markersize=mS, linewidth=lW)
ax2.plot(times,h_Spill,'-',c='b',label='Spill site',
        markersize=mS, linewidth=lW)
ax2.plot(times,h_obs4,'-',c='k',label='Obs 3',
        markersize=mS, linewidth=lW)
ax2.grid()
ax2.set_xlim(-10,375)
#ax2.set_ylim(1.45,2)
#ax2.set_ylim(0.00001,1000)
#ax2.set_yscale('log')
ax2.set_xlabel('Elapsed time [days]')
ax2.set_ylabel('Hydraulic head [m]')
titleText = 'Hydraulic heads: Layer %i' % lyr
ax2.set_title(titleText,loc='left')
ax2.legend(loc='upper right')


ax3 = fig.add_subplot(1,3,3)

hds = bf.HeadFile(modelName + '.hds')
times = hds.get_times()
head = hds.get_data(totim=times[-1])
head_all = hds.get_alldata()

lyr = 2 # Defines layer to use when plotting

h_obs1 = head_all[:,lyr,obs1_row,obs1_col]
h_obs3 = head_all[:,lyr,obs3_row,obs3_col]
h_Spill = head_all[:,lyr,obsSpill_row,obsSpill_col]
h_obs4 = head_all[:,lyr,obs4_row,obs4_col]

ax3.plot(times,h_obs1,'-',c='c',label='Obs 1',
        markersize=mS, linewidth=lW)
ax3.plot(times,h_obs3,'-',c='r',label='Obs 2',
        markersize=mS, linewidth=lW)
ax3.plot(times,h_Spill,'-',c='b',label='Spill site',
        markersize=mS, linewidth=lW)
ax3.plot(times,h_obs4,'-',c='k',label='Obs 3',
        markersize=mS, linewidth=lW)
ax3.grid()
ax3.set_xlim(-10,375)
#ax3.set_ylim(1.45,2)
#ax3.set_ylim(0.00001,1000)
#ax3.set_yscale('log')
ax3.set_xlabel('Elapsed time [days]')
ax3.set_ylabel('Hydraulic head [m]')
titleText = 'Hydraulic heads: Layer %i' % lyr
ax3.set_title(titleText,loc='left')
ax3.legend(loc='upper right')



# ===== Basic concentration conditons =====================================
fig = plt.figure(figsize=(14,10))

# Getting concentration data
ucnobj = bf.UcnFile('MT3D001.UCN')

#print(ucnobj.list_records()) # get values
times = ucnobj.get_times() # simulation time
times_30d = times[29]
times_60d = times[59]
times_100d = times[99]
times_180d = times[179]
times_365d = times[364]

conc_30d = ucnobj.get_data(totim=times_30d)
conc_60d = ucnobj.get_data(totim=times_60d)
conc_100d = ucnobj.get_data(totim=times_100d)
conc_180d = ucnobj.get_data(totim=times_180d)
conc_365d = ucnobj.get_data(totim=times_365d)

#converting from g/L to micrograms/L
cF = 10**6 # conversion factor to convert from kg m**3 to mug L
conc_30d = conc_30d*cF
conc_60d = conc_60d*cF
conc_100d = conc_100d*cF
conc_180d = conc_180d*cF
conc_365d = conc_365d*cF

#===== Plan view of concentration - Left panel
ax1 = fig.add_subplot(2, 2, 1)

# flopy plot object
pmv = flopy.plot.PlotMapView(model=mf, layer=0)

# plot grid
lc = pmv.plot_grid() # grid

# plot concentration
cmap = plt.get_cmap('jet')
cmin = 0
cmax = 10**6
cs = pmv.plot_array(conc_30d,cmap=cmap,vmin=cmin,vmax=cmax) # concentration colourmap
cbar = plt.colorbar(cs,format='%.0e') # colour bar
cbar.ax.set_ylabel('C [$\mu$g L$^-$$^1$]')

# plot spill site
ax1.plot(obsSpill_xcoord, obsSpill_ycoord,'o',c='r',label='Spill site',
        markersize=mS)

# setting labels
ax1.set_xlabel('X-coordinate [m]')
ax1.set_ylabel('Y-coordinate [m]')
tday = round(times_30d)
titleText = "Plan view: Concentration at t = %i" % tday+' days'
ax1.set_title(titleText,loc='left')



#===== Plan view of concentration - Right panel
ax2 = fig.add_subplot(2, 2, 2)

# flopy plot object
pmv = flopy.plot.PlotMapView(model=mf, layer=0)

# plot grid
lc = pmv.plot_grid() # grid

# plot concentration
cmap = plt.get_cmap('jet')
cmin = 0
cmax = 10**6
cs = pmv.plot_array(conc_365d,cmap=cmap,vmin=cmin,vmax=cmax) # concentration colourmap
cbar = plt.colorbar(cs,format='%.0e') # colour bar
cbar.ax.set_ylabel('C [$\mu$g L$^-$$^1$]')

# plot spill site
ax2.plot(obsSpill_xcoord, obsSpill_ycoord,'o',c='r',label='Spill site',
        markersize=mS)

# setting labels
ax2.set_xlabel('X-coordinate [m]')
ax2.set_ylabel('Y-coordinate [m]')
tday = round(times_365d)
titleText = "Plan view: Concentration at t = %i" % tday+' days'
ax2.set_title(titleText,loc='left')

#===== Vertial section of concentration - left panel
ax3 = fig.add_subplot(2, 2, 3)
hds = bf.HeadFile(modelName + '.hds')
times = hds.get_times()
head = hds.get_data(totim=times[-1])

modelxsect = flopy.plot.PlotCrossSection(model=mf, line={'Row': 15})
cmap = plt.get_cmap('jet')
cmin = 0
cmax = 10**6
hvc = modelxsect.plot_array(conc_30d,cmap=cmap,vmin=cmin,vmax=cmax,head=head)
cbar = plt.colorbar(hvc,format='%.0e') # colour bar
cbar.ax.set_ylabel('C [$\mu$g L$^-$$^1$]')
ax3.grid()
#ax3.set_xlim(-10,375)
ax3.set_ylim(-2,3.2)
ax3.set_xlabel('X-coordinate [m]')
ax3.set_ylabel('Elevation [m]')
tday = round(times_30d)
titleText = "Vertical section: Concentration at t = %i" % tday+' days'
ax3.set_title(titleText,loc='left')

#===== Vertial section of concentration - right panel
ax4 = fig.add_subplot(2, 2, 4)
hds = bf.HeadFile(modelName + '.hds')
times = hds.get_times()
head = hds.get_data(totim=times[-1])

modelxsect = flopy.plot.PlotCrossSection(model=mf, line={'Row': 15})
cmap = plt.get_cmap('jet')
cmin = 0
cmax = 10**6
hvc = modelxsect.plot_array(conc_365d,cmap=cmap,vmin=cmin,vmax=cmax,head=head)
cbar = plt.colorbar(hvc,format='%.0e') # colour bar
cbar.ax.set_ylabel('C [$\mu$g L$^-$$^1$]')
ax4.grid()
#ax4.set_xlim(-10,375)
ax4.set_ylim(-2,3.2)
ax4.set_xlabel('X-coordinate [m]')
ax4.set_ylabel('Elevation [m]')
tday = round(times_365d)
titleText = "Vertical section: Concentration at t = %i" % tday+' days'
ax4.set_title(titleText,loc='left')


plt.tight_layout()



# Extracting time and concentration data
ucnobj = flopy.utils.UcnFile('MT3D001.UCN')
times = ucnobj.get_times()
conc = ucnobj.get_alldata()

obs1_data = conc[:,:,obs1_row,obs1_col][:,0]    # Observation at high recharge cell
obs2_data = conc[:,:,obs3_row,obs3_col][:,0] # Observation well 2
obs3_data = conc[:,:,obs4_row,obs4_col][:,0]
obs4_data = conc[:,:,int(obsSpill_row),int(obsSpill_col)][:,0] # Observation at extraction well


# Plotting time series data
fig = plt.figure(figsize=(14,12))
ax = fig.add_subplot(1, 1, 1)

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
ax.set_xlim(0,sum(perlen))
ax.set_ylim(0.00005,1100)
ax.set_yscale('log')
ax.set_xlabel('Elapsed time [days]')
ax.set_ylabel('Concentration [g L$^-$$^1$]')
ax.set_title('Contaminant concentration',loc='left')
ax.legend(loc='upper right')











