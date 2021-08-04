# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 11:27:03 2021

@author: dusti
"""
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:41:18 2021

@author: dusti
"""
from pylab import *
import schemdraw as schem
import schemdraw.elements as e
import smith as smi## requires schemdraw module !!
import mwave as mw
import matplotlib.pyplot  as plt
import numpy as np


#te

f = 5e9
filename = 'smithcharts/match_g_s_lumped_no_stabi.png'
print("All values are normalized to 50 Ohm!")
fig, ax = plt.subplots(figsize=(12,12))

Z0 = 50
mysmith = smi.Smith(ax,'both', Z0)

Z1=mysmith.addstart(50)
GS = -0.322-0.445j


ZL = (1+GS)/(1-GS)
print("ZL = "+str(ZL))
YL = 1/ZL
print("YL = "+str(YL))

Y_step = 1 + 1j*imag(YL)
Z_step = 1/Y_step
print("Z_step = "+str(Z_step))
print("Y_step = "+str(Y_step))

YCpar = 1j*np.imag(Y_step)
XCpar = 1/YCpar
XLser = 1j*np.imag(ZL-Z_step)
print("YCpar = "+str(YCpar))
print("XCpar = "+str(XCpar))
print("XLser = "+str(XLser))

Cpar = 1/(2*np.pi*f*abs(50*XCpar))
Lser = 50*abs(XLser)/(2*np.pi*f)
print("Cpar = "+str(Cpar*1e12)+" pF")
print("Lpar = "+str(Lser*1e9)+" nH")


mysmith.addarrow(GS)

#Z2 = mysmith.addseries(Z1, 30j)
Z2 = mysmith.addpara(Z1, 50*XCpar)

mysmith.addpoint(Z2, '$Z_{2}$', 'SE')

Z3 = mysmith.addseries(Z2, 50*XLser)
#Z3 = mysmith.addpara(Z2, -35j)
mysmith.addpoint(Z3, '$Z_{3}$', 'NW')

plt.tight_layout(pad=4.0)
savefig(filename)
plt.show()
