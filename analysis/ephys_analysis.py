'''
functions for a basic analysis of an intracellular recording
'''
# import packages
from math import nan
from scipy.signal import find_peaks
import statistics
import numpy as np
#%%

# define functions
def subthrAnalysis(v, **kwargs):
    ''' 
    analyses subthreshold responsefeatures of an intracellular recording.
    Arguments are:
        v              voltage trace of a recording / simulation

    Keywordarguments are:
        restPotidx     indices for resting potential
        hyperidx       indices for hyperpolarized signal part

    Analyzed responsefeatures are: Resting potential, input resistance, 
    voltage sag during hyperpolarization and spont. activity.
    '''

    restPotidx = kwargs["restPotidx"]
    hyperidx = kwargs["hyperidx"]
    #%%
    noStim = v[restPotidx[0] : restPotidx[1]]
    hyperPol = v[hyperidx[0] : hyperidx[1]]

    restPot = statistics.median(noStim)
    medHyp = statistics.median(hyperPol)
    inpRes = (medHyp - restPot) * (-1)

    minHyp = hyperPol.min()
    sag = medHyp - minHyp

    peakSpont, properties = find_peaks(noStim,prominence=(4, None))
    if peakSpont.size > 0:
        spontAkt = 1

    else:
        spontAkt = 0
        
    subthrFeat = np.array([restPot, inpRes, sag, spontAkt])
    
    #%%
    return subthrFeat


def spikeAnalysis(v, **kwargs):
    """Funcion to analyze spike related response features of a simulated 
    voltage trials.

    Args:
        v (np array [mV]): simulated voltage trace

    kwargs:
        stimidx (tuple): indices of start and stop of the stimulation period

    Returns:
        nPeaks (integer): number of spikes counted
        amp3rd (float [mV]): amplitude of the 3rd spike
        ampDiff (float [mV]): difference of the amplitude between the 2nd and 
        the 4th spike 
    """    
    stimIDX = kwargs["stimidx"]

    stimPeriod = v[stimIDX[0] : stimIDX[1]]

    # counting spikes
    peaks, properties = find_peaks(
        stimPeriod,
        prominence=(4, None),
        distance=100
        )
    
    nPeaks = peaks.size

    # calculate height of the 3rd spike and the difference between the 2nd and
    # 4th spike.
    
    if nPeaks >= 4:
        negPeaks,properties = find_peaks(
            -stimPeriod,
            prominence=(4, None),
            distance=100
            )
        
        if negPeaks.size >= 4:
            amp2nd = (stimPeriod[peaks[1]] - stimPeriod[negPeaks[1]]) # convert to mV
            amp3rd = (stimPeriod[peaks[2]] - stimPeriod[negPeaks[2]])
            amp4th = (stimPeriod[peaks[3]] - stimPeriod[negPeaks[3]])
            ampDiff = amp2nd - amp4th

        elif negPeaks.size == 3:
            amp3rd = (stimPeriod[peaks[2]] - stimPeriod[negPeaks[2]])
            ampDiff = nan

        else:
            amp3rd = nan
            ampDiff = nan

    elif nPeaks == 3:
        negPeaks,properties = find_peaks(
            -stimPeriod,
            prominence=(4, None),
            distance=100
            )
        
        if negPeaks.size >= 3:
            amp3rd = (stimPeriod[peaks[2]] - stimPeriod[negPeaks[2]])
            ampDiff = nan

        else:
            amp3rd = nan
            ampDiff = nan

    else:
        amp3rd = nan
        ampDiff = nan

    spfeat = np.array([nPeaks, amp3rd, ampDiff])
    return spfeat
