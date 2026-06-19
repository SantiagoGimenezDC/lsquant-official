import numpy as np
import pylab as plt

data = np.loadtxt('cn.txt')
fig,ax = plt.subplots()
ax.plot(data[:,0],data[:,1])
ax.set_yscale('log')
fig.savefig('cn.png',dpi=300,bbox_inches='tight')
