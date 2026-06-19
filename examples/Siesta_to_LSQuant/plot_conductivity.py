import numpy as np

from scipy.constants import h,hbar, e as q, Boltzmann, pi

from scipy.signal import convolve

import scipy.optimize

from scipy.integrate import cumulative_trapezoid as cumtrapz

import f90nml

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.legend_handler import HandlerTuple




# Gradient routine using central differences
# Step size assumed to be uniform
def gradient_central(y,dx):

  dydx = np.zeros(len(y))

  # Do gradient at each point, accounting for edges
  for i in range(len(y)) :
    if i == 0 :
      dydx[i] = (y[i+1]-y[i]) / dx
    elif i == len(y)-1 :
      dydx[i] = (y[i]-y[i-1]) / dx
    else :
      dydx[i] = (y[i+1]-y[i-1]) / (2*dx)

  return dydx



# Gradient routine using N=11 low-noise differentiation
# Step size assumed to be uniform
# From: http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/
def gradient_lownoise_N11(y,dx):

  if len(y) < 2:
    return 0

  else:
    dydx = np.zeros(len(y))

    # Do gradient at each point, accounting for edges
    for i in range(len(y)) :
      if i == 0 :
        dydx[i] = (y[i+1]-y[i]) / dx
      elif i == len(y)-1 :
        dydx[i] = (y[i]-y[i-1]) / dx
      elif i == 1 or i == len(y)-2 :
        dydx[i] = (y[i+1]-y[i-1]) / (2*dx)
      elif i == 2 or i == len(y)-3 :
        dydx[i] = ( 2*(y[i+1]-y[i-1]) + (y[i+2]-y[i-2]) ) / (8*dx)
      elif i == 3 or i == len(y)-4 :
        dydx[i] = ( 5*(y[i+1]-y[i-1]) + 4*(y[i+2]-y[i-2]) + (y[i+3]-y[i-3]) ) / (32*dx)
      elif i == 4 or i == len(y)-5 :
        dydx[i] = ( 14*(y[i+1]-y[i-1]) + 14*(y[i+2]-y[i-2]) + 6*(y[i+3]-y[i-3]) + (y[i+4]-y[i-4]) ) / (128*dx)
      else :
        dydx[i] = ( 42*(y[i+1]-y[i-1]) + 48*(y[i+2]-y[i-2]) + 27*(y[i+3]-y[i-3]) + 8*(y[i+4]-y[i-4]) + (y[i+5]-y[i-5]) ) / (512*dx)

    return dydx



# Fitting function for mean square displacement (MSD)
def msd_fit(t, mfp,tp):
  return 2*mfp**2 * (t/tp - 1 + np.exp(-t/tp) )

# Fitting function for diffusivity = d(MSD)/dt / 4
def d_fit(t, mfp,tp):
  return mfp**2/tp/2 * (1 - np.exp(-t/tp) )



# Convert hbar to eV-s
hbar = hbar/q


# Quantum of conductance
G0 = 2*q**2/h


# Read in sample file and get desired parameters
fsamp = open('unitcell.txt','r')

line = fsamp.readline()
A1x = list(map(float, line.split()))[0]   # [Angstrom]
A1y = list(map(float, line.split()))[1]   # [Angstrom]
L1 = np.sqrt(A1x**2 + A1y**2)

line = fsamp.readline()
A2x = list(map(float, line.split()))[0]   # [Angstrom]
A2y = list(map(float, line.split()))[1]   # [Angstrom]
L2 = np.sqrt(A2x**2 + A2y**2)

line = fsamp.readline()
acc = float(line)                        # [Angstrom]

line = fsamp.readline()
line = fsamp.readline()
natoms = int(line)
fsamp.close()


# Read in params.txt into namelist object
#  and get desired parameters
nml   = f90nml.read('params.txt')
tstep = nml['params']['tstep']         # Time step [fs]
nT    = nml['params']['nT']            # Number of time steps
time  = np.linspace(0,nT*tstep,nT+1)   # Time vector

t     = nml['params']['teV']           # Hopping [eV]
#t     = t * 13.78*np.exp(-1.85*acc)    # Account for Simon model
vf    = 3/2 * (acc*1e-10)*t/hbar       # Graphene Fermi velocity

Emin  = nml['params']['Emin']          # Minimum energy to examine [eV]
Emax  = nml['params']['Emax']          # Maximum energy [eV]
dE    = nml['params']['dE']            # Energy step [eV]
nE = int((Emax-Emin)/dE) + 1

nrow = nml['params']['nrow']
ncol = nml['params']['ncol']
Lmax = min(ncol*L1,nrow*L2)   # Sample dimension [Angstrom]


# Area per atom [1/m^2]
A = (A1x*A2y - A1y*A2x) * 1e-20 / natoms



# Get DOS, convert to [1/J/m^2]
data = np.loadtxt('dos.txt')
E   = data[:,0]
dos = data[:,1]
dos = 2*dos/A / q   # The 2 is for spin


# Get fine-energy-grid DOS [1/J/m^2]
data = np.loadtxt('dos.txt')
Efine   = data[:,0]
dosfine = data[:,1]
dosfine = 2*dosfine/A / q   # The 2 is for spin

# Get index of charge neutrality point
# (chosen here to be the minimum of the DOS)
iCNP = np.argmin(dosfine)

# Get carrier density from full spectrum DOS
n2Dfine = cumtrapz(dosfine*A*q,Efine, initial=0)
n2Dfine = n2Dfine - n2Dfine[iCNP]
n2Dfine = n2Dfine / A
print()
print('iCNP:          ',iCNP)
print('Efine[iCNP]:   ',Efine[iCNP])
print('n2Dfine[iCNP]: ',n2Dfine[iCNP])
print()

# Interpolate carrier density onto zoomed energy grid
n2D = np.interp(E, Efine,n2Dfine)



# Allocate space
E   = np.linspace(Emin,Emax,nE)
dL2 = np.zeros((nE,nT+1))


# Read in all mean-square displacements, units are [Angstrom^2]
# Here we calculate total mean-square displacement as dX²+dY²

data_dX2 = np.loadtxt('dX2.txt')
data_dY2 = np.loadtxt('dY2.txt')

for it in range(nT):
  #print(istr)
  dX2  = data_dX2[:,it+1]
  dY2  = data_dY2[:,it+1]

  #dX2[dX2<0] = 0   # Convert negative values of MSD to 0
  #dY2[dY2<0] = 0   # (sometimes happens below the spectral edge)
  dL2[:,it+1]  = dX2+dY2


# Get time-dependent conductivity at each energy
#  as well as the semiclassical conductivity
D        = np.zeros((nE,nT+1))
sigma    = np.zeros((nE,nT+1))
D_sc     = np.zeros(nE)
sigma_sc = np.zeros(nE)
mfp      = np.zeros(nE)
tp       = np.zeros(nE)
guess = (1000,100)  # mfp in Angstroms, tp in fs
for iE in range(nE):

  # Diffusivity = d(dL2)/dt / 4 (convert to units of [m^2/s])
  # (divide by 4 because this is 2D -- dL2 = dX2+dY2)
  #D[iE,:] = dL2[iE,:]/time/4 / (1e10)**2 * 1e15
  #D[iE,:] = gradient_central(dL2[iE,:],tstep)/4 / (1e10)**2 * 1e15
  D[iE,:] = gradient_lownoise_N11(dL2[iE,:],tstep)/4 / (1e10)**2 * 1e15
  D[iE,0] = 0

  # Conductivity vs. time = e^2 * dos * D [1/Ohm]
  sigma[iE,:] = q**2 * dos[iE] * D[iE,:]

  # Semiclassical values from max(D)
  D_sc[iE]     = max(D[iE,:])
  sigma_sc[iE] = max(sigma[iE,:])
  tp[iE] = 2*D_sc[iE] / vf**2 * 1e15   # [fs]

  # Semiclassical values from fitting to MSD
  #ind = np.sqrt(dL2[iE,:]) <= Lmax   # Only fit for spreading less than sample size
  #params,pcov = scipy.optimize.curve_fit(msd_fit, time[ind], dL2[iE,ind], guess, maxfev=10000)
  #mfp[iE],tp[iE] = params   # [Angstrom, fs]
  #guess = params
  #D_sc[iE] = (mfp[iE]*1e-10)**2 / (tp[iE]*1e-15) / 2
  #sigma_sc[iE] = q**2 * dos[iE] * D_sc[iE]

  # Semiclassical values from fitting to diffusivity
  #params,pcov = scipy.optimize.curve_fit(d_fit, time, D[iE,:] * (1e10)**2 / 1e15, guess)
  #mfp[iE],tp[iE] = params   # [Angstrom, fs]
  #guess = params
  #D_sc[iE] = (mfp[iE]*1e-10)**2 / (tp[iE]*1e-15) / 2
  #sigma_sc[iE] = q**2 * dos[iE] * D_sc[iE]



# Get mobility from generalized Einstein relation [cm^2/V-s]
mu   = np.zeros(nE)
dndE = dos
mu   = D_sc * dndE/abs(n2D) * q * 1e4



############
# Plot DOS #
############
# Analytical DOS of graphene
# (the part at the end accounts for hopping renormalization due to Simon's model)
dos_anal = 2*abs(E) / pi / (hbar*vf)**2 / 1e18   # [1/eV/nm^2]

fig,ax = plt.subplots()
ax.plot(E,dos*q/1e18 , lw=1.5 , label='Numerical')
ax.plot(E,dos_anal   , lw=1 , c='black', ls=(0,(3,3)) , label='Analytical')

ax.set_box_aspect(1)

ax.set_xlim(min(E),max(E))
ax.set_ylim(0)

# Axis labels
ax.set_xlabel(r'Energy (eV)'               , fontsize=12)
ax.set_ylabel(r'DOS (eV$^{-1}$ nm$^{-2}$)' , fontsize=12)

# Formatting of plot
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('$%g$'))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('$%g$'))
ax.tick_params(direction='in', right='True', top='True', labelsize=10)

# Legend
leg = ax.legend()
for line, text in zip(leg.get_lines(), leg.get_texts()):
  text.set_color(line.get_color())

# Save
fig.savefig('dos.png',dpi=300,bbox_inches='tight')



##############################
# Plot MSD(t) at each energy #
##############################
fig,ax = plt.subplots()

# Set up color cycle for the set of lines
colors = plt.cm.coolwarm(np.linspace(0,1,nE))
ax.set_prop_cycle('color', colors)

# Horizontal line at MSD = Lmax^2
ax.axhline((Lmax/1e4)**2,0,1)

# Plot sigma(t) at each energy
for iE in range(nE):
  ax.plot(time,dL2[iE,:] / (1e4)**2, lw=1)

# Plot fit at each energy
for iE in range(nE):
  ax.plot(time,msd_fit(time,mfp[iE],tp[iE]) / (1e4)**2 , c='black',lw=0.5,ls=(0,(3,3)))


ax.set_xlim(min(time),max(time))
ax.set_ylim(0,(dL2/(1e4)**2).max())

# Axis labels
ax.set_xlabel(r'Time (fs)'                 , fontsize=14)
ax.set_ylabel(r'MSD ($\mathrm{\mu}$m$^2$)' , fontsize=14)

# Formatting of plot
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('$%g$'))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('$%g$'))
ax.tick_params(direction='in', right='True', top='True')

# Save
fig.savefig('MSD_vs_t.png',dpi=300,bbox_inches='tight')



############################
# Plot D(t) at each energy #
############################
fig,ax = plt.subplots()

# Set up color cycle for the set of lines
colors = plt.cm.coolwarm(np.linspace(0,1,nE))
ax.set_prop_cycle('color', colors)

# Plot D(t) at each energy
for iE in range(nE):
  ax.plot(time,D[iE,:]*1e4, lw=1)

# Plot fit at each energy
for iE in range(nE):
  ax.plot(time,d_fit(time,mfp[iE],tp[iE]) / (1e10)**2 * 1e15 * 1e4 , c='black',lw=0.5,ls=(0,(3,3)))

ax.set_xlim(min(time),max(time))
ax.set_ylim(0)

# Axis labels
ax.set_xlabel(r'Time (fs)'      , fontsize=14)
ax.set_ylabel(r'$D$ (cm$^2$/s)' , fontsize=14)

# Formatting of plot
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('$%g$'))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('$%g$'))
ax.tick_params(direction='in', right='True', top='True')

# Save
fig.savefig('D_vs_t.png',dpi=300,bbox_inches='tight')



################################
# Plot sigma(t) at each energy #
################################
fig,ax = plt.subplots()

# Set up color cycle for the set of lines
colors = plt.cm.coolwarm(np.linspace(0,1,nE))
ax.set_prop_cycle('color', colors)

# Plot sigma(t) at each energy
for iE in range(nE):
  ax.plot(time,sigma[iE,:]/G0, lw=1)

ax.set_xlim(min(time),max(time))
ax.set_ylim(0)

# Axis labels
ax.set_xlabel(r'Time (fs)'           , fontsize=14)
ax.set_ylabel(r'$\sigma$ ($2e^2/h$)' , fontsize=14)

# Formatting of plot
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('$%g$'))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('$%g$'))
ax.tick_params(direction='in', right='True', top='True')

# Save
fig.savefig('sigma_vs_t.png',dpi=300,bbox_inches='tight')



##############################################
# Plot semiclassical conductivity vs. energy #
##############################################
fig,ax = plt.subplots()
ax.plot(E,sigma_sc/G0, lw=1.5)

ax.set_xlim(min(E),max(E))
ax.set_ylim(0)

# Axis labels
ax.set_xlabel(r'Energy (eV)'                     , fontsize=14)
ax.set_ylabel(r'$\sigma_\mathrm{sc}$ ($2e^2/h$)' , fontsize=14)

# Formatting of plot
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('$%g$'))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('$%g$'))
ax.tick_params(direction='in', right='True', top='True')

# Save
fig.savefig('sigma_sc.png',dpi=300,bbox_inches='tight')



############################################
# Plot momentum relaxation time vs. energy #
############################################
fig,ax = plt.subplots()
ax.plot(E,tp, lw=1.5, marker='o',ms=6,mfc='w')

ax.set_xlim(min(E),max(E))
ax.set_ylim(0)
#ax.set_yscale('log')

# Axis labels
ax.set_xlabel(r'Energy (eV)'   , fontsize=14)
ax.set_ylabel(r'$\tau_p$ (fs)' , fontsize=14)

# Formatting of plot
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('$%g$'))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('$%g$'))
ax.tick_params(direction='in', right='True', top='True')

# Save
fig.savefig('tp.png',dpi=300,bbox_inches='tight')

# Also save tp to a file
np.savetxt('tp.txt',np.transpose([E,tp*1e15]))



############################
# Plot mobility vs. energy #
############################
fig,ax = plt.subplots()
ax.plot(E,mu, lw=1.5, marker='o',ms=6,mfc='w')

ax.set_xlim(min(E),max(E))
ax.set_ylim(0,20000)
#ax.set_yscale('log')

# Axis labels
ax.set_xlabel(r'Energy (eV)'          , fontsize=14)
ax.set_ylabel(r'$\mu$ (cm$^2$ / V-s)' , fontsize=14)

# Formatting of plot
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('$%g$'))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('$%g$'))
ax.tick_params(which='both', direction='in', right='True', top='True')
ax.grid(which='both',axis='y')

# Save
fig.savefig('mu.png',dpi=300,bbox_inches='tight')

# Also save mobility to a file
np.savetxt('mu.txt',np.transpose([E,n2D/1e4,mu]))
