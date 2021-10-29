#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 09:22:31 2021
Demo version of the two-compartment hodgkin huxley model implementation in 
python. The simulation package brian2 was used to simulate a passive soma 
compartment that is connected to an active spike initiation zone (SIZ). This
script calculates the model with a standard parameter set that generates a 
T cell like response regarding to resting potential, input resistance and spike
count. Interesting to observe: When depolarizing current is injected in the 
soma, you can really see how the passive response in the soma preceedes the 
passive response of the SIZ. Later, the dynamic of the voltage dependent 
channels make the potential of the SIZ rise much faster and the Soma follows.
This really amazes me :D

FYI: brian requires you to handle units. This can be really tricky if you are 
not used to it. I highly recomend reading the brian documentation.
@author: Kevin Sandbote
"""

#%% import packages
import brian2 as brian
from matplotlib import pyplot as plt
from analysis import ephys_analysis
# import units
from brian2 import second, ms, um, mV, msiemens, cm, uF, ohm, amp, nA
#%% create morphology
# first we need to define the sampling rate. The sampling rate can be decreased
# to decrease runtime, however indices for analysis (see bottom) have to be
# revised then.
brian.defaultclock.dt = 0.01*ms

# Morphology
somaSurface = 5000 * um**2
somaDiameter = brian.sqrt(somaSurface / (brian.pi*2))

#SIZ is a cylinder without base surfaces, therefore its surface is defined by 
#pi*d*h. We can chose either d or h. Lets say d is 2 um. 
SIZSurface = 500 * um**2
SIZDiameter = 2 * um
SIZLength = SIZSurface/(brian.pi*SIZDiameter)

# actually build morphology
morpho = brian.Soma(diameter=somaDiameter)
morpho.SIZ = brian.Cylinder(diameter=SIZDiameter, length=SIZLength,n=1)

#%% defining parameters
# reversal potentials
ENa = 40 * mV
EK = -70 * mV
El1 = -15 * mV
El2 = -15 * mV

# conductance denstities
gNa_var = 680 * msiemens/cm**2
gK_var = 20 * msiemens/cm**2
gM_var = 12 * msiemens/cm**2
gl1 = 0.4 * msiemens/cm**2
gl2 = 0.4 * msiemens/cm**2

# time constants
tm = 0.75 * ms # time constant for m
th = 7.5 * ms # time constant for h
tn = 4.0 * ms # time constant for n
tz = 350 * ms # time constant for z

#%% HH equations
eqs = '''
    # ion currents
    IN = gNa * m**4 * h * (ENa-v) : amp/meter**2
    IK = gK * n**2 * (EK-v) : amp/meter**2
    IL = gl * (El-v) : amp/meter**2
    IM = gM * z**2 * (EK-v) : amp/meter**2

    # transmembrane current
    Im = IN + IK + IL + IM : amp/meter**2

    # applied current
    I : amp (point current)

    # derivatives
    #dV/dt = Im / Cm : volt
    dm/dt = (infm - m) / taum  : 1
    dn/dt = (infn - n) / taun  : 1
    dh/dt = (infh - h) / tauh  : 1
    dz/dt = (infz - z) / tauz  : 1


    #activation/inaktivation functions
    infm = 1 / (1 + exp(-(v+20.0*mV) /(8*mV))) : 1 # no unit
    taum = tm * (0.1 + 2 / (exp(-(v+20.0*mV)/(16*mV)) + exp((v+20.0*mV) / (16.0*mV)))) : second # ms

    infh = 1 / (1 + exp( (v+36.0*mV)/(5*mV))) :  1
    tauh = th * (0.1 + 2 / ( exp(-(v+36.0*mV)/(10*mV)) + exp((v+36.0*mV)/(10.0*mV)))) : second

    infn = 1/ (1 + exp(-(v+20.0*mV)/(8*mV)) ) : 1
    taun = tn * (0.1 + 2 / ( exp(-(v+20.0*mV)/(16*mV)) + exp((v+20.0*mV)/(16.0*mV)))) : second

    infz = 1 / ( 1 + exp(-(v+37*mV)/(4.0*mV)) ) : 1
    tauz = tz + tz * ( 2 / ( exp(-(v+35.0*mV)/(6*mV)) + exp((v+35.0*mV)/(6.0*mV)))) : second

    gNa : siemens/meter**2
    gK : siemens/meter**2
    gl : siemens/meter**2
    El : volt
    gM :siemens/meter**2
    '''

#%% define neuron as a spatial neuron object
neuron = brian.SpatialNeuron(morphology=morpho, model=eqs,
                       Cm=1*uF/cm**2, Ri=11*ohm*cm, method='exponential_euler')
# initial potential
neuron[0].v = -33*mV
# initial input current
neuron[0].I = 0*amp
#initial conducances and reversal potentials for the soma
neuron[0].gNa = 0
neuron[0].gK = gK_var/4
neuron[0].gl = gl1
neuron[0].El = El1
neuron[0].gM = gM_var/4

# and for the SIZ
neuron[1].gNa = gNa_var
neuron[1].gK = gK_var
neuron[1].gl = gl2
neuron[1].El = El2
neuron[1].gM = gM_var

#%% Monitors
# the monitor method determines which variables are saved for the whole
# simulation period
mon=brian.StateMonitor(neuron, 'v', record=True)

#%% run the model
# one seconds pre stim period
brian.run(1 * second) 
# half a second hyperpolarization
neuron[0].I[0] = -1 *nA
brian.run(0.5 * second)
# half a second no stim period
neuron[0].I[0] = 0 * nA
brian.run(0.5 * second)
# half a second depolarization
neuron[0].I[0] = 1 * nA
brian.run(0.5 * second)
# half a second no stim period
neuron[0].I[0] = 0 * nA
brian.run(0.5 * second)

#%% plot results
plt.figure('simulated voltage trace')
plt.plot(mon.t/ms, mon.v[0]/mV, label='Soma')
plt.plot(mon.t/ms, mon.v[1]/mV, label='SIZ')
plt.legend()
#%% analyse data
# extract voltage trace of soma without unit
vS = mon.v[0]/mV
# define idx of rois for sub threshold feature analysis
idx_sub_thr = {
    "restPotidx" : [50000 , 100000],
    "hyperidx" : [100000, 150000]
}
# define idx of rois for spike feature analysis
idxStim = {
    "stimidx" : [200000, 250000]
}

# run analysis script for both feature groups
restPot, inpRes, sag, spontAkt = ephys_analysis.subthrAnalysis(vS, **idx_sub_thr)
nPeaks, amp3rd, ampDiff = ephys_analysis.spikeAnalysis(vS, **idxStim)