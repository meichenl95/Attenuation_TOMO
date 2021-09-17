#!/home/meichen/anaconda3/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import obspy
import glob
import os
from scipy.sparse.linalg import lsqr
from scipy.sparse import coo_matrix
from mtspec import mtspec
from obspy.taup import TauPyModel
from scipy.signal import hilbert
from scipy.interpolate import interp1d

def rsm(x,y,f):
    fun = interp1d(x,y)
    yn = fun(f)
    return yn

def smooth(y,W):
    yy = np.zeros(y.shape)
    for i in np.arange(W):
        yy[i]  = np.mean(y[0:i+W])
        yy[-i-1] = np.mean(y[-i-W::])
    for i in np.arange(W,len(y)-W):
        yy[i]  = np.mean(y[i-W:i+W])
 
    return yy

def spec(tr,leftlen,rightlen,f,envNS):
    
    data    = tr.data
    cut     = data[envNS-leftlen:envNS+rightlen]
    cut     = cut/np.max(cut)

    ## ---- fourier transform
    nfft    = int(2**np.ceil(np.log2(cut.shape[0]*10)))
    amp,frequency  =  mtspec(data=cut,delta=tr.stats.delta,time_bandwidth=1.5,number_of_tapers=2,nfft=nfft,adaptive=False)
    amp     = np.sqrt(amp)
    amp     = smooth(amp,10)

    ## ---- resample to f
    amp     = rsm(frequency,amp,f)

    return amp

def s2n(tr,envNS):

    data     = tr.data
    envelope = np.abs(hilbert(data))
    flag     = 1

    ## ---- find S wave duration
    # left length
    i = 1
    while (envelope[envNS-i]>0.5*envelope[envNS]):
        i = i + 1
        if envNS-i<0:
            flag = 0
            break
    leftlen = 3*i
    # right length
    i = 1
    while (envelope[envNS+i]>0.5*envelope[envNS]):
        i = i + 1
        if envNS+i>=tr.stats.npts:
            flag = 0
            break
    rightlen = 3*i

    ## ---- compute signal noise ratio
    dur = leftlen+rightlen
    if (envNS-5*dur>0):
        ratiomax  = envelope[envNS]/np.max(envelope[envNS-5*dur:envNS-2*dur])
        ratiomean = envelope[envNS]/np.mean(envelope[envNS-5*dur:envNS-2*dur])
    else:
        ratiomax  = 0
        ratiomean = 0
        flag      = 0

    return ratiomean,ratiomax,leftlen,rightlen, flag

def envelopemax(tr,model):

    data     = tr.data
    envelope = np.abs(hilbert(data))
    flag     = 1

    ## ---- expected S arrival times from PREM
    arrivals = model.get_travel_times(source_depth_in_km=tr.stats.sac.evdp,distance_in_degree=tr.stats.sac.gcarc,phase_list=['S'])
    tS       = arrivals[0].time

    ## ---- find the envelope peak of S wave
    NS       = int((tS-tr.stats.sac.b)/tr.stats.delta)
    # find S
    Nwin     = int(20/tr.stats.delta)
    depmax   = envelope[NS]
    envNS    = NS
    for i in np.arange(Nwin):
        if(envelope[NS-i]>depmax and envelope[NS-i]>envelope[NS-i-1] and envelope[NS-i]>envelope[NS-i+1]):
            depmax    = envelope[NS-i]
            envNS     = NS-i
        if(envelope[NS+i]>depmax and envelope[NS+i]>envelope[NS+i-1] and envelope[NS+i]>envelope[NS+i+1]):
            depmax    = envelope[NS+i]
            envNS     = NS+i

    if (envNS == NS):
        flag = 0

    return envNS, flag

def main():

    ## ---- path of seismograms
    path    = "/home/meichen/work1/Attn_TOMO/seismograms"
    model   = obspy.taup.TauPyModel(model='prem')
    ## ---- number of sampled frequency
    f       = np.array([0.05,0.1,0.2,0.4,0.6,0.8])
    num_frq = len(f)

    ## ---- change current directory
    os.chdir(path)

    ## ---- numbers of stations and events
    set_evt = set()
    set_stn = set()
    for sacfile in glob.glob("*.BHT.SAC.vel"):
        eventid, nw,stn,loc,_,_,_ = sacfile.split('.')
        set_evt.add(eventid)
        station = '.'.join([nw,stn,loc])
        set_stn.add(station)
    num_evt = len(set_evt)
    num_stn = len(set_stn)

    ## ---- initialize matrix
    num_sei = len(glob.glob("*.BHT.SAC.vel"))
    row     = []
    col     = []
    A       = np.zeros((num_sei,int(num_frq*(num_evt+num_stn+1))))
    rhs     = []
    
    ## ---- create matrix
    line    = 0
    for i,eventid in enumerate(set_evt):
        for j,stnid in enumerate(set_stn):
            fln = glob.glob("{}.{}.BHT.SAC.vel".format(eventid,stnid))
            if(fln == []):
                continue
            else:
                tr = obspy.read(fln[0])[0]
                tr.filter('bandpass',freqmin=0.01,freqmax=2,zerophase=True)
                ## ---- locate S wave
                envNS,flag = envelopemax(tr,model)
                if (flag==0):
                    continue
                ## ---- comput signal-noise ratio
                ratiomean,ratiomax,leftlen,rightlen,flag = s2n(tr,envNS)
                if (ratiomean<6 or ratiomax<3 or flag == 0):
                    continue
                ## ---- compute amplitude
                amp = spec(tr,leftlen,rightlen,f,envNS)
                for k,freq in enumerate(f):
                    row.append(line)
                    col.append(k)
                    row.append(line)
                    col.append(num_frq+i*num_frq+k)
                    row.append(line)
                    col.append(num_frq+num_evt*num_frq+j*num_frq+k)
                    rhs.append(amp[k])
                    line = int(line + 1)

    ## ---- solve least-squares problem
    A = coo_matrix((np.ones(len(row)),(row,col)),shape=(len(rhs),num_frq+num_evt*num_frq+num_stn*num_frq))
    x,istop,itn,_,_,_,_,_,_,_ = lsqr(A,rhs)
    print(A.shape)
    print(np.linalg.norm(rhs))
    print(np.linalg.norm((A*x)))
    print(np.linalg.norm((A*x-rhs)))

    ## ---- plot figure
    fig,ax = plt.subplots(1,3,figsize=[9,3])
    ax[0].plot(f,x[0:num_frq])
    ax[0].set_xlabel('Frequency (Hz)',size=12)
    ax[0].set_ylabel('T(f)',size=12)
    for i in np.arange(num_evt):
        ax[1].plot(f,x[num_frq+i*num_frq:num_frq+(i+1)*num_frq])
    for i in np.arange(num_stn):
        ax[2].plot(f,x[num_frq*(num_evt+1)+i*num_frq:num_frq*(num_evt+1)+(i+1)*num_frq])
    ax[1].set_xlabel('Frequency (Hz)',size=12)
    ax[1].set_ylabel('S(f)',size=12)
    ax[2].set_xlabel('Frequency (Hz)',size=12)
    ax[2].set_ylabel('R(f)',size=12)
    fig.tight_layout()
    plt.savefig('/home/meichen/Research/Attn_TOMO/results.pdf')
    plt.show()

main()
