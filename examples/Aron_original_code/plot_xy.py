import numpy as np

import os.path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter




# Read in xy coordinates
if(os.path.isfile('xy.txt')):
  data = np.loadtxt('xy.txt')
  x = data[:,0]
  y = data[:,1]
  e = data[:,2]


# Read in xy coordinates at atoms without 3 neighbors
if(os.path.isfile('xy1.txt')):
  data = np.loadtxt('xy1.txt')
  x1 = data[:,0]
  y1 = data[:,1]

if(os.path.isfile('xy2.txt')):
  data = np.loadtxt('xy2.txt')
  x2 = data[:,0]
  y2 = data[:,1]



# Read in xy coordinates of 'special' atoms
if(os.path.isfile('xy0.txt')):
  data = np.loadtxt('xy0.txt')
  x0 = data[:,0]
  y0 = data[:,1]



# Plot atomic coordinates
fig,ax = plt.subplots()

if(os.path.isfile('xy.txt')):
  ax.scatter(x,y,c=e, s=10,linewidths=0)
if(os.path.isfile('xy1.txt')):
  ax.scatter(x1,y1,c='tab:red', s=10,linewidths=0)
if(os.path.isfile('xy2.txt')):
  ax.scatter(x2,y2,c='tab:green', s=10,linewidths=0)
if(os.path.isfile('xy0.txt')):
  ax.scatter(x0,y0,c='yellow', s=10,linewidths=0)
ax.set_aspect('equal')

ax.set_xlabel(r'$x$ ($\mathrm{\AA}$)' , fontsize=12)
ax.set_ylabel(r'$y$ ($\mathrm{\AA}$)' , fontsize=12)

plt.gca().xaxis.set_major_formatter(FormatStrFormatter('$%g$'))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('$%g$'))
ax.tick_params(direction='in', right='True', top='True', labelsize=10)


# Save image
fig.savefig('xy.png',dpi=600,bbox_inches='tight')
