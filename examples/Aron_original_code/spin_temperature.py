import numpy as np
import scipy.optimize
from scipy.constants import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import container



""" Define fitting functions for spin lifetime """
def exp_decay(t, A,ts):
    #return A * np.exp(-t/ts)
    return np.exp(-t/ts)

def exp_cos(t, A,ts,tw):
    return A * np.exp(-t/ts) * np.cos(2*pi*t/tw)

def cos(t, A,tw):
    return A * np.cos(2*pi*t/tw)



""" Choose type of function to fit """
fit_type = 'exp'
#fit_type = 'exp_cos'
#fit_type = 'cos'


""" Consider spin at all temperatures """
T = [100,150,200,250,300,350,400]

fig2x,ax2x = plt.subplots()
fig2z,ax2z = plt.subplots()
fig3, ax3  = plt.subplots()

# Set up color cycle for the set of lines
colors = plt.cm.coolwarm(np.linspace(0,1,len(T)))
ax2x.set_prop_cycle('color', colors)
ax2z.set_prop_cycle('color', colors)
ax3.set_prop_cycle('color', colors)

for iT in T:

  """ Read in time scale, energy, and spin polarization """
  rp = [1]
  i = 0
  for irp in rp:
    dirx = str(iT)+'K'+'/sx_7x7_rp'+str(irp)+'/'
    dirz = str(iT)+'K'+'/sz_7x7_rp'+str(irp)+'/'
    txi  = np.loadtxt(dirx+'t.txt')
    Exi  = np.loadtxt(dirx+'E.txt')
    sxi  = np.loadtxt(dirx+'sxout.txt')
    tzi  = np.loadtxt(dirz+'t.txt')
    Ezi  = np.loadtxt(dirz+'E.txt')
    szi  = np.loadtxt(dirz+'szout.txt')
    if i == 0:
      tx = txi
      Ex = Exi
      sx = sxi
      tz = txi
      Ez = Exi
      sz = szi
    else:
      sx = sx + sxi
      sz = sz + szi
    i = i+1
  sx = sx / len(rp)
  sz = sz / len(rp)



  """ Create fitting arrays and their StDevs"""
  Ax      = np.zeros(len(Ex))
  tsx     = np.zeros(len(Ex))
  twx     = np.zeros(len(Ex))
  Ax_std  = np.zeros(len(Ex))
  tsx_std = np.zeros(len(Ex))
  twx_std = np.zeros(len(Ex))
  Az      = np.zeros(len(Ez))
  tsz     = np.zeros(len(Ez))
  twz     = np.zeros(len(Ez))
  Az_std  = np.zeros(len(Ez))
  tsz_std = np.zeros(len(Ez))
  twz_std = np.zeros(len(Ez))



  """ Loop over all energies, and fit spin lifetime at each """
  if fit_type == 'exp':
    guess = (1,10)
  elif fit_type == 'exp_cos':
    guess = (1,10,10)
  elif fit_type == 'cos':
    guess = (1,10)

  E = Ex   # ASSUME that sx and sz are on the same energy scale (needed for calculating anisotropy)
  Ef =  0.05
  iEf = (np.abs(E-Ef)).argmin()
  for i in range(len(E)):

    # Do the fit, save the fitted parameters and their standard deviations
    if fit_type == 'exp':
      params,pcov = scipy.optimize.curve_fit(exp_decay, tx, sx[:,i], guess)
      Ax[i],tsx[i] = params
      Ax_std[i],tsx_std[i] = np.sqrt(np.diag(pcov))
      params,pcov = scipy.optimize.curve_fit(exp_decay, tz, sz[:,i], guess)
      Az[i],tsz[i] = params
      Az_std[i],tsz_std[i] = np.sqrt(np.diag(pcov))
    elif fit_type == 'exp_cos':
      params,pcov = scipy.optimize.curve_fit(exp_cos, tx, sx[:,i], guess)
      Ax[i],tsx[i],twx[i] = params
      Ax_std[i],tsx_std[i],twx_std[i] = np.sqrt(np.diag(pcov))
      params,pcov = scipy.optimize.curve_fit(exp_cos, tz, sz[:,i], guess)
      Az[i],tsz[i],twz[i] = params
      Az_std[i],tsz_std[i],twz_std[i] = np.sqrt(np.diag(pcov))
    elif fit_type == 'cos':
      params,pcov = scipy.optimize.curve_fit(cos, tx, sx[:,i], guess)
      Ax[i],twx[i] = params
      Ax_std[i],twx_std[i] = np.sqrt(np.diag(pcov))
      params,pcov = scipy.optimize.curve_fit(cos, tz, sz[:,i], guess)
      Az[i],twz[i] = params
      Az_std[i],twz_std[i] = np.sqrt(np.diag(pcov))

    # Print some information to the screen
    print('T = '+str(iT)+'K')
    print('  E = '+str(round(E[i],2))+' eV')
    print('    Ax  = '+str(round(Ax[i],2)) + ' -- Az  = '+str(round(Az[i],2)))
    if fit_type == 'exp' or fit_type == 'exp_cos':
      print('    tsx = '+str(round(tsx[i]/1000,2))+' ns' + ' -- tsz = '+str(round(tsz[i]/1000,2))+' ns')
    if fit_type == 'exp_cos' or fit_type == 'cos':
      print('    twx = '+str(round(twx[i]/1000,2))+' ns' + ' -- twz = '+str(round(twz[i]/1000,2))+' ns')

    # Update the fitting guess
    #guess = params

    # Plot data and fit for a chosen Ef
    #if True:
    if i == iEf:
      fig1,ax1 = plt.subplots()
      ax1.plot(tx,sx[:,i], marker='o',ms=6,mfc='w', label=r'$S_x$')
      ax1.plot(tz,sz[:,i], marker='o',ms=6,mfc='w', label=r'$S_z$')

      if fit_type == 'exp':
        ax1.plot(tx,exp_decay(tx,Ax[i],tsx[i]), c='r', label='fit')
        ax1.plot(tz,exp_decay(tz,Az[i],tsz[i]), c='r')
      elif fit_type == 'exp_cos':
        ax1.plot(tx,exp_cos(tx,Ax[i],tsx[i],twx[i]), c='r', label='fit')
        ax1.plot(tz,exp_cos(tz,Az[i],tsz[i],twz[i]), c='r')
      elif fit_type == 'cos':
        ax1.plot(tx,cos(tx,Ax[i],twx[i]), c='r', label='fit')
        ax1.plot(tz,cos(tz,Az[i],twz[i]), c='r')

      ax1.set_title('T = '+str(iT)+'K, '+r'$E_F$ = '+str(round(E[i],2))+' eV')

      ax1.set_xlabel(r'Time (ps)', fontsize=12)
      ax1.set_ylabel(r'$S_{x,z}$', fontsize=12)

      ax1.set_xlim(0)

      ax1.xaxis.set_major_formatter(FormatStrFormatter('$%g$'))
      ax1.yaxis.set_major_formatter(FormatStrFormatter('$%g$'))

      ax1.tick_params(which='major', direction='in')
      ax1.tick_params(which='minor', direction='in')

      leg1 = ax1.legend()
      for line, text in zip(leg1.get_lines(), leg1.get_texts()):
        text.set_color(line.get_color())


      plt.show()
      if i == iEf:
        fig1.savefig('fit_spin_'+str(iT)+'K'+'.png',dpi=300 , bbox_inches='tight')



  """ Plot fitted spin lifetimes """
  if fit_type == 'cos':
    ax2x.errorbar(E,twx,twx_std, marker='o',ms=6,mfc='w', label='T = '+str(iT)+'K')
    ax2z.errorbar(E,twz,twz_std, marker='o',ms=6,mfc='w', label='T = '+str(iT)+'K')
    ax2x.set_ylabel(r'$\tau_{\omega x}$ (ps)', fontsize=12)
    ax2z.set_ylabel(r'$\tau_{\omega z}$ (ps)', fontsize=12)
  else:
    ax2x.errorbar(E,tsx/1e3,tsx_std/1e3, marker='o',ms=6,mfc='w', label='T = '+str(iT)+'K')
    ax2z.errorbar(E,tsz/1e3,tsz_std/1e3, marker='o',ms=6,mfc='w', label='T = '+str(iT)+'K')
    ax2x.set_ylabel(r'$\tau_{sx}$ (ns)', fontsize=12)
    ax2z.set_ylabel(r'$\tau_{sz}$ (ns)', fontsize=12)


  """ Plot spin lifetime anisotropy """
  if fit_type == 'cos':
    print()
    print('No anisotropy for cosine fitting')
    print()

  else:
    ax3.plot(E,tsz/tsx, marker='o',ms=6,mfc='w', label='T = '+str(iT)+'K')



""" Format and save spin lifetime plot """
ax2x.set_title(r'80nm x 80nm, suspended, NPT')
ax2z.set_title(r'80nm x 80nm, suspended, NPT')

ax2x.set_xlabel(r'Energy (eV)', fontsize=12)
ax2z.set_xlabel(r'Energy (eV)', fontsize=12)

ax2x.set_yscale('log')
ax2x.set_ylim([0.1,10])
ax2z.set_yscale('log')
ax2z.set_ylim([0.1,10])

ax2x.xaxis.set_major_formatter(FormatStrFormatter('$%g$'))
ax2x.yaxis.set_major_formatter(FormatStrFormatter('$%g$'))
ax2z.xaxis.set_major_formatter(FormatStrFormatter('$%g$'))
ax2z.yaxis.set_major_formatter(FormatStrFormatter('$%g$'))

ax2x.tick_params(which='major', direction='in')
ax2x.tick_params(which='minor', direction='in')
ax2z.tick_params(which='major', direction='in')
ax2z.tick_params(which='minor', direction='in')

# Remove the errorbars from the legend
handles, labels = ax2x.get_legend_handles_labels()
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
handles, labels = ax2z.get_legend_handles_labels()
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]

# Then make the legend and color the text
leg2 = ax2x.legend(handles,labels)
for line, text in zip(leg2.get_lines(), leg2.get_texts()):
  text.set_color(line.get_color())
leg2 = ax2z.legend(handles,labels)
for line, text in zip(leg2.get_lines(), leg2.get_texts()):
  text.set_color(line.get_color())

fig2x.savefig('tsx.png',dpi=300 , bbox_inches='tight')
fig2z.savefig('tsz.png',dpi=300 , bbox_inches='tight')



""" Format and save the spin lifetime anisotropy plot """
ax3.set_ylim(0,1)
ax3.axhline(0.5,0,1, c='black',lw=1,linestyle=(0,(3,3)), zorder=0)
ax3.axhline(1  ,0,1, c='black',lw=1,linestyle=(0,(3,3)), zorder=0)

ax3.set_title(r'80nm x 80nm, suspended, NPT')

ax3.set_xlabel(r'Energy (eV)', fontsize=12)
ax3.set_ylabel(r'$\tau_{sz}$ / $\tau_{sx}$', fontsize=12)

ax3.xaxis.set_major_formatter(FormatStrFormatter('$%g$'))
ax3.yaxis.set_major_formatter(FormatStrFormatter('$%g$'))

ax3.tick_params(which='major', direction='in')
ax3.tick_params(which='minor', direction='in')

leg3 = ax3.legend(handles,labels)
for line, text in zip(leg3.get_lines(), leg3.get_texts()):
  text.set_color(line.get_color())

fig3.savefig('spin_anisotropy.png',dpi=300 , bbox_inches='tight')
