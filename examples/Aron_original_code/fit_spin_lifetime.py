import numpy as np
import scipy.optimize
from scipy.constants import pi,e,hbar
import f90nml
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter



# Convert hbar to eV-s
hbar = hbar/e



""" Define fitting function for spin lifetime """
def exp_decay(t, A,ts):
    return A * np.exp(-t/ts)

def exp_cos(t, A,ts,tw):
    A = 1
    return A * np.exp(-t/ts) * np.cos(2*pi*t/tw)



""" Choose type of function to fit """
fit_type = 'exp'
#fit_type = 'exp_cos'
#fit_type = 'cos'



""" Read in params.txt into namelist object
     and get desired parameters """
nml = f90nml.read('params.txt')
lR  = nml['params']['VR']   # Rashba SOC [eV]



""" Read in time scale, energy, and spin polarization """
t  = np.loadtxt('t.txt')
E  = np.loadtxt('E.txt')
sz = np.loadtxt('szout.txt')

""" Create fitting arrays and their StDevs"""
A      = np.zeros(len(E))
ts     = np.zeros(len(E))
tw     = np.zeros(len(E))
A_std  = np.zeros(len(E))
ts_std = np.zeros(len(E))
tw_std = np.zeros(len(E))

""" Loop over all energies, and fit spin lifetime at each """
if fit_type == 'exp':
    guess = (1,20)
elif fit_type == 'exp_cos':
    guess = (1,0.1,0.3)

Ef = 0.5
iEf = (np.abs(E-Ef)).argmin()

for i in range(len(E)):

    # Do the fit based on the chosen fitting function
    if fit_type == 'exp':
        params,pcov = scipy.optimize.curve_fit(exp_decay, t, sz[:,i], guess)
        A[i],ts[i] = params
        A_std[i],ts_std[i] = np.sqrt(np.diag(pcov))
    elif fit_type == 'exp_cos':
        params,pcov = scipy.optimize.curve_fit(exp_cos, t, sz[:,i], guess)
        A[i],ts[i],tw[i] = params
        A_std[i],ts_std[i],tw_std[i] = np.sqrt(np.diag(pcov))

    # Update the fitting guess
    guess = params

    # Plot data and fit a chosen Ef
    #if True:
    if i == iEf:
        fig1,ax1 = plt.subplots()
        ax1.plot(t,sz[:,i], lw=0,marker='o',ms=6,mfc='w')
        if fit_type == 'exp':
            ax1.plot(t,exp_decay(t,A[i],ts[i]), c='r')
        if fit_type == 'exp_cos':
            ax1.plot(t,exp_cos(t,A[i],ts[i],tw[i]), c='r')
        ax1.set_xlim(0)
        ax1.set_title(r'$E$ = %g eV'%E[i])
        ax1.set_ylabel(r'$S_z$')
        ax1.set_xlabel(r'Time (ps)')
        ax1.tick_params(direction='in', right='True', top='True')
        ax1.xaxis.set_major_formatter(FormatStrFormatter('$%g$'))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('$%g$'))
        if i == iEf:
            fig1.savefig('fit_spin.png',dpi=300 , bbox_inches='tight')
        plt.show()


""" Calculate theoretical spin lifetimes  """
# Get momentum relaxation time
data = np.loadtxt('tp.txt')
Etp = data[:,0]
tp  = data[:,1]*1e-15

# Calculate DP spin relaxation
tsDP = 1 / (tp * (2*lR/hbar)**2) * 1e12

# Calculate dephasing relaxation
tsOSC = pi*tp * 1e12

""" Plot fitted spin lifetimes """
fig2,ax2 = plt.subplots()
#ax2.errorbar(E,ts,ts_std, marker='o',ms=6,mfc='w', label='Numerical')
ax2.plot(E,ts/1000, lw=0, marker='o',ms=6,mfc='w', label='Numerical')
ax2.plot(Etp,tsDP/1000 ,c='tab:green'  , label='Dyakonov-Perel')
ax2.plot(Etp,tsOSC/1000,c='tab:purple' , label='Dephasing')
ax2.set_ylabel(r'$\tau_s$ (ns)')
ax2.set_xlabel(r'Energy (eV)')
ax2.set_xlim(min(E),max(E))
ax2.set_ylim(0,180)
ax2.tick_params(direction='in', right='True', top='True')
ax2.xaxis.set_major_formatter(FormatStrFormatter('$%g$'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('$%g$'))
leg = ax2.legend(loc='upper left')
for line, text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
plt.show()
fig2.savefig('ts.png',dpi=300,bbox_inches='tight')
