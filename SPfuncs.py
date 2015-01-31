"""Useful functions for processing the SP data. Hopefully some are generic enough to be used elsewhere too.

"""
from __future__ import division
from pylab import *
import matplotlib
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
from scipy import stats
from copy import deepcopy
import pdb
from collections import namedtuple
from scipy.optimize import curve_fit
from scipy.ndimage import convolve1d
import os
from itertools import count
import re
import MT, MF
from MT import nonnan, cached_property, nancurve_fit, figDontReplace, running_median_insort, vprint
from scipy import random
from scipy.signal import windows
from statsmodels import api as sm
#from SPDataSet import SPDataSet

bTest=False

def secs2sd(secs):
    return secs/23.9344696/3600;
def sd2secs(sd):
    return sd*23.9344696*3600;

Window =namedtuple('Window', ['duration', 'offset'])
RotationSpec=namedtuple('RotationSpec', ['startingAngle', 'delay', 'interval', 'rotAngles', 'extraRotAngle', 'numRotations']);
RawData=namedtuple('RawData', ['t', 'sig'])
PointData=namedtuple('PointData', ['t', 'sig', 'err', 'chi2', 'theta' ])
CorrelationData=namedtuple('CorrelationData', ['t', 'sig', 'err', 'chi2', 'theta'])
SiderealFitData=namedtuple('SiderealFitData', ['t', 'sig', 'err', 'chi2', 'sidTheta','labTheta', 'meta'])
LabSetData=namedtuple('LabSetData', ['t', 'sig', 'err', 'chi2', 'labTheta'])



def getRotationsFromRSpec(rSpec, numReps=1):
    if not iterable(rSpec.rotAngles):
        rotAngles=[rSpec.rotAngles]
    else:
        rotAngles=rSpec.rotAngles
    rotationSequence=array( [rSpec.extraRotAngle]+ [rotAngles[k] for k in arange(rSpec.numRotations-1, dtype='i4')%len(rotAngles)])
    allRotations=tile(rotationSequence, numReps)
    return allRotations
def getAnglesFromRSpec(rSpec, numReps=1):
    """ Return angles for the sequence defined by @rSpec, repeated @numReps times.
    Angles are that /before/ each rotation, including before the extraRotation

    """
    #Logic here is a little screwy. We start with extraRotAngle then subtract it from everything
    # Also remember, following the logic on spin-doctor we only do the main rotation N-1 times, the last one being the 'extraRotation'
    
    allRotations=getRotationsFromRSpec(rSpec, numReps)
    allAngles= rSpec.startingAngle-rSpec.extraRotAngle + cumsum(allRotations)
    allAngles=allAngles.reshape((numReps, rSpec.numRotations))
    #allAngles=allAngles[:,:-1]
    #allAngles=tile(angles, (numReps,1) )+arange(numReps)[:,newaxis]*rSpec.extraRotAngle
    return allAngles

def makeTriggers(sd,rSpec, seqStartTimes=None, bIncludeExtraRot=True, gapBetween=False):
    """ Construct a set of triggers by analysing the timing for the
    signal.
    """
    sdDiff=np.diff(sd); 
    meanDt=np.mean(sdDiff) # Mean difference between 1 point and the next (should be 0.02s ish)
    meanDt=np.median(sdDiff) # (Maybe using the median will be more robust?)
    
    if gapBetween:
        vprint("You probably shouldn't be using the gapBetween argument to makeTriggers")
        gapBetween=secs2sd(gapBetween)
        dataTime=secs2sd(rSpec.numRotations*rSpec.interval+rSpec.delay)
        seqStartTimes=arange(sd[0],sd[-1]-dataTime, (dataTime +gapBetween) )

    if seqStartTimes is None:
        mask=np.hstack([[True], sdDiff>8*meanDt]) #Make a mask array for where the difference is much greater than the average
        #The above should perhaps not start with at true, but a false? I imagine we start recording data BEFORE we send the first trigger, not the other-way around?

        seqStartTimes=sd[mask]; #We'll assume those points are where each sequence of rotations starts
        if len(seqStartTimes)==1:
            vprint("Maybe there's no gaps in the sequence? makeTriggers is only returning one set of triggers")
    
    Ntrigs = rSpec.numRotations-1
    if bIncludeExtraRot:
        Ntrigs+=1
    group=secs2sd(np.arange(Ntrigs)*rSpec.interval+rSpec.delay); #A group is a group of triggers for 1 sequence

    trigTimes=[group+seqStartTimes[0]]; #Trig will be the list of all triggers which we slowly build up.
    for i in range(1, seqStartTimes.size): #(this could be easily vectorized, as the size of trigTimes is known in advance)
        if trigTimes[-1][-1]<seqStartTimes[i]:
            trigTimes.append(group+seqStartTimes[i])

    #angles=getAnglesFromRSpec(rSpec, seqStartTimes.size)#, bIncludeExtraRot=bIncludeExtraRot)
    return np.array(trigTimes)#, np.array(angles)

def sliceSorted(t, t_0, t_end=None, delta_t=None):
        """ Cut up the times @t into sections that start at t_0 and and at t_end OR t_0 + delta_t if they're all the same.

        Returns a list of index arrays with length t_0.size
        """
        startIs=t.searchsorted(t_0.ravel()); #ravel here because x0 can be multi-d
        if t_end is not None:
            endIs = t.searchsorted(t_end.ravel())
            if delta_t is not None:
                raise Exception("should have t_end OR delta_t, not both!")
        elif delta_t:
            endIs = t.searchsorted(t_0.ravel()+delta_t)

        startIs=startIs[endIs<t.size] #Throw out cuts where the end index is out of range
        endIs=endIs[:startIs.size]

        if t_0.ndim>1:
            #If we lose some part so we have an incomplete sequence, we'll dump them.
            Norphans=startIs.size%t_0.shape[-1] #this is only general for 2d- need rethinking for more dimensions
            if Norphans>0:
                startIs=startIs[:-Norphans]
                endIs=endIs[:-Norphans]

            newShape=[int(startIs.size/t_0.shape[-1]), t_0.shape[-1]]
        else: 
            newShape=[endIs.size]

        indxL=empty(newShape, dtype='O')
        indxL.ravel()[:]=[np.r_[sI:eI] for sI,eI in zip(startIs, endIs)]

        return indxL

def sliceReduceData(x, y, x0, delta_x=None,endTimes=None, functionL=[np.size, np.mean, lambda vals, **kwargs: np.std(vals, **kwargs)/sqrt(np.size(vals))],  bJustGiveCuts=False ):
        """Cut the given x data at points starting from x0, and going to x0+delta_x, and perform the functions in functionL on each cut section. 

        Returns meanXs. finalL, indxL
        """
        
        if endTimes is None:
            endTimes = x0 + delta_x
        indxL=sliceSorted(x, x0, t_end=endTimes);

        if y.ndim==1:
            y=y[newaxis,:]
        tPtsL=[x[inds] for inds in indxL.flat]

        finalL=[]
        for yax in y:

            yPtsL=[yax[inds] for inds in indxL.flat]
            resL=[]
            for f in functionL: #Apply each function in the list to the cuts
                    vals=[f(ys, axis=0) for ys in yPtsL]
                    vals=array(vals).reshape(indxL.shape)
                    resL.append(vals)
            processedL=np.array(resL)
            finalL.append(processedL)
        
        meanXs= (x0+endTimes)/2.# delta_x/2. # The mean time for each cut
        meanXs=meanXs.ravel()[:indxL.size].reshape(indxL.shape)

        if len(finalL)==1:
            finalL=finalL[0]
        return meanXs, finalL, indxL    

def stringSimple3pt(x, err=None):
    coeffs=array([0.25, -0.5, 0.25])
    coeffs/=abs(coeffs).sum()
    cnv=convolve1d(x, coeffs, axis=0, mode='constant')[1:-1]
    if len(cnv.shape)>1:
        res=cnv.mean(axis=1) # Or median?
        if err is not None:
            raise ValueError("if x is 2d, this function will calculate errors so errors shouldn't be included")
        err3pt = std(cnv, axis=1)/sqrt(x.shape[0])
    else: 
        res=cnv
        err3pt= sqrt(convolve(err**2, coeffs**2,mode='valid'))
    st3pt=res* (arange(res.size)%2-0.5)*-2 #Multiply by [1,-1,1,-1]....


    return st3pt, err3pt

def stringSimple5pt(x, err=None, rotDir=1.):
    """Calculate 5pt strings and errors
    """
    #coeffs=array([0.25, -0.5, 0.25]) #Copy coefficients from matlab code
    coeffs=array([
       [ 0.25 ,  0., -0.5,  0. ,  0.25],
        [-0.25  ,  0.5   ,  0.    , -0.5   ,  0.25],
       [ 0.125     , -0.25      ,  0.25      , -0.25      ,  0.125     ]])
    #coeffs=array([[1.,0.,-2.,0.,1.], #1h0
    #            [-1,2.,0.,-2.,1.], #1h90
    #            [1,-2, 2, -2, 1]], #2h0
    #            )
    #coeffs.T[:]/=abs(coeffs).sum(axis=1)
    N=len(x)-4
    resA=zeros([N, 4])
    errA=zeros([N, 4])
    for cf, r, e in zip(coeffs, resA.T, errA.T):
        cnv=convolve1d(x, cf, axis=0, mode='constant')[2:-2]
        #cnv=convolve1d(x, coeffs, axis=0, mode='constant')[2:-2]
        if len(cnv.shape)>1: #If x is 2d, we'll assume that each column represents repeated measurements of the same value
            thisRes=cnv.mean(axis=1) # (Maybe median would be better here?)
            if err is not None:
                raise ValueError("if x is 2d, this function will calculate errors so errors shouldn't be included")
            thisErr = std(cnv, axis=1)/sqrt(x.shape[0])
        else: 
            res=cnv
            thisErr= sqrt(convolve(err**2, cf**2,mode='valid')[2:-2])
        r[:]=thisRes
        e[:]=thisErr
    resA[:,-1]*=nan
    res1h=(resA[:,0] + 1j*resA[:,1])*exp(-1j*pi/2*arange(N))#%2-0.5)*-2
    err1h=(errA[:,0] + 1j*errA[:,1])*exp(+1j*pi/2*arange(N)) #(arange(N)%2-0.5)*-2
    errA[:,0],errA[:,1]=abs(err1h.real), abs(err1h.imag)

    resA[:,0]=res1h.real
    resA[:,1]=res1h.imag
    resA[:,2]=resA[:,2]*(arange(N)%2-0.5)*-2

    #st0=res[::2]
    #st90=res[1::2]
    #err0=err[::2]
    #err90=err[1::2]
    #st0=st0* (arange(st0.size)%2-0.5)*-2 #Multiply by [1,-1,1,-1]....
    #st90=st90* (arange(st90.size)%2-0.5)*-2 #Multiply by [1,-1,1,-1]....


    return resA.reshape([N,2,2]), errA.reshape([N,2,2])

def getHarmonicsFit( theta_in, x_in, errs_in=None, harmonics=[1], pars0=[1.,1.]):
    """Calculate the harmonics present in the data x corresponding to phases theta.

    Here we'll fit to sin waves.
    Inputs theta_in, x_in, and errs_in may be 2d

    Returns:
        thL, ampL, covL (thetas, fitted a#mplitude, covariance- except currently only the diagonal components, i.e. variance)
    """
    bOnlyOneHarmonic=False
    if not hasattr(harmonics, '__len__'): #If we haven't been given a list of harmonics, we'll assume it's a number
        bOnlyOneHarmonic=True

    theta_in=theta_in.ravel()
    x_in=x_in.ravel()
    #errs_in=errs_in[
    errs_in=errs_in.ravel()
    #if errs_in==None:
    #errs_in=ones(x.shape, dtype=x.dtype)
    #d_theta=theta[1]-theta[0]
    #theta=arange(0, d_theta*N, d_theta)
    N=theta_in.size
    thL=[]
    ampL=[]
    covL=[]
    chi2L=[]
    delt=0.1
    THx=theta_in[0]
    THy=THx+pi/2. #This is the sin phase, so it's before cos 
    theta = theta_in-THx
    p_mn0=nanmean(x_in)
    #x=x_in-nanmean(x_in)
    x=x_in

    bSinQuad=True;
    bCosQuad=True;

    bTest=False
    if bTest:
        fig, [ax, ax2]=subplots(2,1)
        #ax=gca()
        ax.plot(theta_in, x)
        #ax.plot(theta_in, x_in, 'ro')
        ax.plot(theta_in, x, 'bo')
    for n in harmonics:
        if sum(abs(sin(theta[~isnan(x)])))/N <0.1: # Means there's not much data for the sin quadrature
            bSinQuad=False;
        if bSinQuad and bCosQuad:
            #[Ax, Ay, B], cov=nancurve_fit(lambda theta, ax,ay,b: ax*cos(n*theta)+ ay*sin(n*theta) + b, theta, x,p0=[0.5e-1,0.5e-1,1.],sigma=errs_in)# [0][0] #sin(n*theta)*x;
            f=lambda theta, ax,ay, mn: ax*sin(n*theta)+ ay*cos(n*theta) + mn
            [Ax, Ay, Xmn], cov=nancurve_fit(f, theta, x,p0=pars0+[p_mn0],sigma=errs_in**2, )#, absolute_sigma=True)# [0][0] #sin(n*theta)*x;
            if bTest:
                ax2.plot(theta_in, f(theta, Ax, Ay, Xmn))
            chi2= nansum((f(theta, Ax, Ay, Xmn)-x)**2*1./errs_in**2)/(sum(~isnan(x).ravel())-2)
        elif bCosQuad:
            #[Ax, B],cov=nancurve_fit(lambda theta, ax,b: ax*cos(n*theta) + b, theta, x,p0=[0.5e-1,1.],sigma=errs_in)
            f=lambda theta, ay, mn: ay*cos(n*theta) + mn
            [Ay, Xmn],cov=nancurve_fit(f, theta, x,p0=[pars0[0], p_mn0],sigma=errs_in)#, absolute_sigma=True)
            if bTest:
                ax2.plot(theta_in, f(theta, Ay, Xmn))
            Ax=nan
            THx=nan
            cov=array([[nan, nan], [nan, cov[0,0]]]) # Let's pretend we've got realy covariance
            chi2= nansum((f(theta, Ay, Xmn)-x)**2*1./errs_in**2)/(sum(~isnan(x).ravel())-1)
        elif bSinQuad: 
            f=lambda theta, ax, mn: ax*sin(n*theta) + mn
            [Ax, Xmn],cov=nancurve_fit(f , theta, x,p0=[pars0[1], p_mn0],sigma=errs_in)#, absolute_sigma=True)
            if bTest:
                ax2.plot(theta_in, f(theta, Ax, Xmn))
            Ay=nan
            cov=array([[cov[0,0], nan], [nan, nan]]) # Cheating
            chi2= nansum((f(theta, Ax, Xmn)-x)**2*1./errs_in**2)/(sum(~isnan(x).ravel())-1)
        else:
            raise Exception('Error in getHarmonicsFit: input angles are probably unexpected?')
        
        if bTest:
            show()
            raw_input('Continue?')

        if bOnlyOneHarmonic:
            covL=cov
            ampL=(Ax,Ay)
            thL=(THx, THy)
        else:
            covL.append(cov)
            ampL.append((Ax,Ay))
            thL.append((THx, THy))
            chi2L.append(chi2)
        #covL.append(sqrt(diag(cov)))
        #covL.append((sqrt(cov_x[0][0]), sqrt(cov_y[0][0])))
    return thL, ampL, covL, chi2L

def mgStringHarmonics(theta, x, err=None, harmonics=[1], chi2Filter=inf):
    """ Input measurements x taken at angles theta0+ n* dtheta (assumed to be evenly spaced) and return harmonic components

    Returns: phi, amp, err, chi2
    where they each have the shape
    [ (val1, val2), (val1, val2)... ], where each tuple corresponds to a different harmonic and the individual components correspond to the different quadratures.
    incalculable values are replaced with "nan"

    That is, return the coefficients A1, A2 that fit the data for A1*sin(n*theta) + A2*cos(n*theta) for harmonic n, as well as standard deviations and chi^2
    """
    # We'll assume that theta is evenly spaced for the moment
    assert(theta.shape[0]==x.shape[0])
    theta0=theta[0]
    dtheta=theta[1]-theta[0]
    if any(abs(diff(diff(theta)))>0.1): # Is the _change_ in step-size changing? (Why do we care??)
        if any(abs(abs(diff(theta))-pi) >0.1): #Maybe we're going back and forth by 180, in which case we can still operate
            raise ValueError("Theta values are apparently not evenly spaced. stringHarmonics can't handle this (yet)")
    if (2*pi-0.05)%abs(dtheta) < 0.1:
        raise ValueError("string harmonics only implemented for dtheta that fits evenly into 2*pi")

    #drnL=
    for H in harmonics:#range(4)+1:  #count():
        rem=(pi+0.1)%(H*abs(dtheta))
        if abs(rem)<0.2: # Check that the angles are suitable, i.e. they're all at near 0, pi

            div=(pi)/(H*dtheta) #How many steps to get to a pi change in the signal
            nptString= round(2*abs(div)+1) #Use this to calculate the number of string points to use

            #Throw away outliers, without considering correlations
            if x.ndim>1:
                pterr=nanstd(x, axis=-1)
                pts=nanmean(x, axis=-1)
            else:
                pts=x
            badMask=isnan(pts)
            while 1:
                inv_variance=1./pterr**2
                inv_variance_sum=nansum(inv_variance[~badMask])
                sampleMean=nansum(pts[~badMask]*inv_variance[~badMask])/inv_variance_sum
                zsq=(pts-sampleMean)**2*inv_variance/inv_variance_sum
                zsq[badMask]=nan
                biggestDev=nanmax(zsq)
                if biggestDev>20 and 0:
                    I=nanargmax(zsq)
                    badMask[I]=True
                    if sum(badMask)>5:
                        badMask[:]=True
                        vprint("Unrecoverable seq (too many points deviate)")
                        break
                else:
                    break
            x[badMask]=nan    
            
            #result = [[ nan,  nan ]], [[ nan, nan ]], [[[ nan, nan ],[ nan,nan ]]], [[ nan, nan ]]
            minNumPts=2*nptString+1
            if nptString==3:
                st, stErr = stringSimple3pt(x, err)
                seq1, seqErr1, seqChi21=  MT.fancy_column_stats(st, stErr, cf=3, chi2Max=chi2Filter, Nmin=5)
                result = array([[theta0, nan]]), array([seq1]), array([seqErr1]), array([seqChi21])
            elif nptString==5:
                st, stErr= stringSimple5pt(x, err, rotDir=sign(dtheta))
                seq1, seqErr1, seqChi21=  MT.fancy_column_stats(st[:,0], stErr[:,0], cf=5, chi2Max=chi2Filter, Nmin=5)
                seq2, seqErr2, seqChi22=  MT.fancy_column_stats(st[:,1], stErr[:,1], cf=5, chi2Max=chi2Filter, Nmin=5)
                result = array([[theta0, theta0-dtheta], [theta0, nan+theta0+abs(dtheta)/2]]), array([seq1, seq2]), array([seqErr1, seqErr2]), array([seqChi21, seqChi22])
                vprint("th:{}, seq2:{}".format(theta0%(2*pi), seq2))
            else:
                print("nptString is {0}".format(nptString))


            # CHI2 ELIMINATION
            ## While the set of string points doesn't fit a chi2 model, we eliminate the largest deviating points

            if 0:
                while 1:
                    inv_variance=1./stErr**2; # Variance from individual error bars
                    inv_variance_sum=nansum(inv_variance)
                    ampFinal=nansum(st*inv_variance)/inv_variance_sum

                    N=sum(~isnan(st))
                    cf=3.*((N-1)/N)**2# 'Correlation factor': Used to be 2.54. This stuff is (ftm) worked out emprically
                    degOfF=N/cf 
                    devSq=(st-ampFinal)**2*inv_variance
                    chi2=nansum(devSq)/degOfF; #: = var(st)/varfromerrorbars Needs checking too.
                    if chi2>chi2Filter:
                        I=nanargmax(devSq)#/nanvar(st)
                        #I=nanargmax(dev)
                        st[I]=nan
                        stErr[I]=nan
                        if sum(~isnan(st))<minNumPts: #if we have too few points left
                            vprint("too many bad string points (by chi2 limit) "),
                            st[:]=nan
                            stErr[:]=nan
                            ampFinal=nan
                            break;
                    else:
                        break

                #Decide if there were too many points thrown out:
                N=sum(~isnan(st))
                if N < minNumPts and 1: #If there's less than a few independent string points remaining, we'll give up
                    #vprint("Unrecoverable sequence")
                    return [(theta0, nan)], [(nan, nan)], [[[nan,nan],[nan,nan]]], [(nan, nan)]
                #adjDegOfF=stats.chi2.ppf(.5, N -2.549); #Adjusted degrees of freedom for calculating chi2
                cf=3.*((N-1)/N)**2# Used to be 2.54. This stuff is (atm) worked out emprically
                degOfF=N/cf 
                chi2=nansum(((st-ampFinal)**2)*inv_variance)/degOfF; #: = var(st)/varfromerrorbars Needs checking too.

                errFinal=sqrt(nansum((st-ampFinal)**2*inv_variance/inv_variance_sum)/degOfF*sqrt(2))#/sqrt(1-stats.chi2.cdf(chi2, degOfF));

                #Assemble the results
                result= array([[nan, nan]]), array([[nan, nan]]), array([[[nan, nan], [nan, nan]]]), array([[nan, nan]])

                #result[0][0]=[theta0,nan]
                #result[1][0]=[ampFinal,nan]
                #result[2][0]=[[[errFinal,nan],[nan,nan]]]
                #result[3][0]=[[chi2,nan]]


            #return [(theta0, nan)], [(ampFinal, nan)], [[[errFinal,nan],[nan,nan]]], [(chi2, nan)]




        else:
            print("abs(rem)!<0.2 for dtheta = {0}, H = {1}. Can't do string analysis on these".format(dtheta, H))
            #This should probably raise an exception

            result[0][0]=[nan,nan]
            result[1][0]=[nan,nan]
            result[2][0]=[[[nan,nan],[nan,nan]]]
            result[3][0]=[[nan,nan]]
            break;
    return result
    
def renameFieldMaybe(recArray, oldName, newName):
        names=list(recArray.dtype.names)
        try:
            I=names.index(oldName)
            names[I]=newName
            recArray.dtype.names=names
        except ValueError:
            pass;
        return recArray

def addFakeData(rawAmp, apparatusAngles=None,trigs=None, rotationRate=None, amp=1., sigma=0., phi=0, amp2=0, phi2=0, zeroingTime=100, fracDev=0, sizeDev=0.1, fastAddF=None, bBlankOriginalData=False):
    """ Add fake signal/noise to an existing signal

    """
    sd=rawAmp.t
    sig=rawAmp.sig.copy()
    if bBlankOriginalData:
        sig*=0

    if apparatusAngles is not None:
        theta= 2*pi*sd +phi+ apparatusAngles[:,:,newaxis] # Angle of the apparatus with respect to the cosmo   
        # Now to cut the data into 'sequences'
        #seqStartTimes=arange(0,totTime, secs2sd(rSpec.delay + rSpec.interval*rSpec.numRotations + zeroingTime))+startTime


        #thPtsL=[theta[inds] for inds in indxL.flat]
        #apparatusAngles=getAnglesFromRSpec(rSpec, numReps=len(pretrigs))/180.*pi
        #theta=hstack(array(thPtsL) +apparatusAngles.ravel()[:len(thPtsL)])
        #flat_ind=indxL.ravel() #OR something else??

    elif rotationRate is not None:
        theta= 2*pi*sd + phi + 2*pi*rotationRate*sd2secs(sd)
    else:
        raise ValueError("Need to include either apparatusAngles or rotationRate")

    fast_noiseF=lambda t: random.normal(size=t.size, scale=sigma) if sigma>0 else 0

    sig+= amp*sin(theta) + amp2*sin(2*theta+phi2) #add base signal

    #Random outlier additions:
    sig+= where(rand(*sig.shape)>fracDev, 0, sizeDev)
    #hstack([zeros(inds.shape) if rand()>fracDev else sizeDev*ones(inds.shape) for inds in indxL.flat ])

    #Uniform noise
    #sd=hstack(tPtsL)
    sig+= fast_noiseF(sd)
    
    #Repetitive noise
    if fastAddF is not None:
        addN=hstack([fastAddF(sd2secs(tPts-tPts[0])) for tPts in tPtsL ])
        sig+=addN
            #sig[ind]+=fastAddF(sd[ind]-sd[ind[0]])

    if bTest:
        figure('TestAddFakeData')
        ax=subplot(311)
        plot(sd, sig,',')
        subplot(312, sharex=ax)
        plot(trigs.flatten(), apparatusAngles.flatten(),'-o')
        subplot(313)
        plot(trigs.flatten(),'o')

    return RawData(sd, sig)#SPDataSet('test', preloadDict={'fastD':[sd, sig, ['sig']]}, rSpec=rSpec)

    #return SPDataSet(fastD=[sd, sig, ['sig']])


    return cutAmpAdded

def makeTestData(rSpec, amp=1., amp2=0.0, sigma=5., N=100000, sampleRate=10, phi=0, phi2=0, zeroingTime=100, fracDev=0, sizeDev=0.1, fastAddF=None, startTime=0):
    """ Make up some data that the experiment _might_ have measured

    Plan: 1. Make a pure signal, and add some noise. 2. Calculate the component that would be measured in each orientation. 3. Add white noise.. 
    """

    S1cosmos=amp*exp(1j*phi)
    S2cosmos=amp2*exp(1j*2*phi2)

    totTime=secs2sd(N/sampleRate) #Total time of the set in sidereal days
    #zeroingTime=100 #How long to wait between sequences
    #phi=pi/3 #Signal phase
    sd= linspace(0,totTime, N) + startTime
    theta= 2*pi*sd  # Angle of the room with respect to the cosmos
    fast_noiseF=lambda t: random.normal(size=t.size, scale=sigma)

   
    # Now to cut the data into 'sequences'
    seqStartTimes=arange(0,totTime, secs2sd(rSpec.delay + rSpec.interval*rSpec.numRotations + zeroingTime))+startTime

    trigs=makeTriggers(sd, rSpec, seqStartTimes=seqStartTimes, bIncludeExtraRot=True);
    #theta=theta/180*pi

    pretrigs=trigs[:,0]-secs2sd(rSpec.delay)
    startTimes=hstack([pretrigs[:,newaxis],trigs[:,:-1]])
    endTimes=trigs


    #tPtsL,thPtsL,indxL= sliceReduceData(sd, theta, x0=startTimes.ravel(), endTimes=endTimes.ravel(), bJustGiveCuts=True)
    #Get the indices divided up according to the triggers
    indxL= sliceSorted(sd, t_0=startTimes.ravel(), t_end=endTimes.ravel()) #indices of the points between each rotation
    tPtsL=[sd[inds] for inds in indxL.flat]
    thPtsL=[theta[inds] for inds in indxL.flat]

    apparatusAngles=getAnglesFromRSpec(rSpec, numReps=len(seqStartTimes+1))/180.*pi
    theta=hstack(array(thPtsL) +apparatusAngles.ravel()[:len(thPtsL)]) #angle of apparatus with respect to the cosmos
    
    sig=real(S1cosmos*exp(-1j*theta) + S2cosmos*exp(-1j*2*theta))
    #sig= amp*sin(theta + phi) + amp2*sin(2*theta + phi2)#Base signal

    #Random outlier additions:
    sig+=hstack([zeros(inds.shape) if rand()>fracDev else sizeDev*ones(inds.shape) for inds in indxL.flat ])

    #Uniform noise
    sd=hstack(tPtsL)
    sig+= fast_noiseF(sd)
    
    #Repetitive noise
    if fastAddF is not None:
        addN=hstack([fastAddF(sd2secs(tPts-tPts[0])) for tPts in tPtsL ])
        sig+=addN
            #sig[ind]+=fastAddF(sd[ind]-sd[ind[0]])

    if bTest:
        figure('TestMakeTestData')
        ax=subplot(311)
        plot(sd, sig,',')
        subplot(312, sharex=ax)
        plot(trigs.flatten(), apparatusAngles.flatten(),'-o')
        subplot(313)
        plot(trigs.flatten(),'o')

    return sd, sig#SPDataSet('test', preloadDict={'fastD':[sd, sig, ['sig']]}, rSpec=rSpec)
    #return SPDataSet(fastD=[sd, sig, ['sig']])

def group_by_axis(thetaIn):
    theta=thetaIn[~isnan(thetaIn)]
    absC=cos(theta)**2
    absS=sin(theta)**2
    sQ=mean(absS)
    cQ=mean(absC)
    #if any(theta%(2*pi)
    if any( abs(theta%(pi) -pi/2) <0.05 ) and abs(0.5-sQ <0.05) and abs(0.5 -cQ) <0.05: # If we have a near-uniform sampling of angles...
            thetaRef=0;

    else: # Otherwise we'll try to pick a good angle (This is not perfect by any  means)
        thetaRef=arctan2(sQ, cQ);
        print("Data doesn't seem to be purely NS->EW. Setting main angle as {0} deg".format(thetaRef/pi*180))
        #pdb.set_trace()
    absC=abs(cos(thetaIn-thetaRef))
    absS=abs(sin(thetaIn-thetaRef))

    xI=where(absC>0.99)[0] # Points 
    yI=where(absS>0.99)[0] # P
    if len(yI)<4:
        vprint("Not enough data points for orthogonal quadrature. Set yI =[]")
        yI=[]
    if MT.nearby_angles(thetaRef, pi/2, 0.1):
        xI,yI=yI,xI
        thetaRef=0
    return thetaRef, xI, yI



def rotate_sequence_amps(seqAmp, rotateTo=0):
    """
    >>> N=100
    >>> th=linspace(0,20,N)%(2*pi)
    >>> sa= CorrelationData(linspace(0,1,N), sin(2*th), ones(N))

    """
    rotAmp=deepcopy(seqAmp)
    for h in range(rotAmp.sig.shape[1]):
        #newThetas, newSig, newErr=zip(*[th, s, er for th, s, er in zip(sig.theta[:,h], sig.sig[:,h], sig.err[:,h])])
        zipped=zip(seqAmp.theta[:,h], seqAmp.sig[:,h], seqAmp.err[:,h]) 
        rotAmp.theta[:,h], rotAmp.sig[:,h], rotAmp.err[:,h]=zip(*[MT.rotate_quadrature_sample(th*(h+1), s, er,
        rotateTo=rotateTo, debug=True) if not all(isnan(s)) else (th, s, er) for (th, s, er) in zipped])
        rotAmp.theta[:,h]/=(h+1)
    return rotAmp

def process_sequences_multifit(sig, sigSensL, sigSensVaryPhaseIs=[], subtractHarmsL=[], minFreq=None, maxFreq=None, bPlot=False, harmonic=1):
    """ 
    """
    h=harmonic-1
    varyPhaseIndL=[]
    #sig=CorrelationData(*[c[:,h::h+1] for c in s]) for s in sig]
    #sigSensL=[CorrelationData(*[c[:,h:] for c in s]) for s in sigSensL]
    t=sig.t[:,h,0]
    #_, sig1h, cov1h=MT.rotate_quadrature_sample(sig.theta[:,0,0], sig.sig[:,0], sig.err[:,0])
    thetaRef, xI, yI=group_by_axis(sig.theta[:,0])        

    vprint("Main angle (for sidereal fit): {0}".format(thetaRef))
    sigRot=rotate_sequence_amps(sig, rotateTo=0)
    
    sigSensL=[rotate_sequence_amps(sigSens, rotateTo=thetaRef) for sigSens in
    sigSensL]
    yL=[sigSens.sig[:,h] for sigSens in sigSensL]
    if sigSensVaryPhaseIs:
        varyPhaseIndL.extend(sigSensVaryPhaseIs)
        print("Allowing phase for sensors at {0} to vary".format(sigSensVaryPhaseIs))

    gdQuads=[]
    for k,th, s in zip(arange(sigRot.theta.shape[2]), sigRot.theta[:,h,:].T, sigRot.sig[:,h,:].T):
        notNan=th[~isnan(s)]
        if notNan.size:
            gdQuads.append(k)
            if not all(MT.nearby_angles(notNan, notNan.min(), 0.1)):
                vprint("Not all angles the same: maybe they're orthogonal?")

    #yL.append(vstack([sin(2*pi*t+sig.theta[:,0]), cos(2*pi*t + sig.theta[:,0])]).T)
    #Lab frame signal
    dc=ones(t.size, dtype='f8')
    yL.append(vstack([dc, dc*0]).T)
    varyPhaseIndL.append(len(yL)-1)
    # Sidereal
    refThetas=sigRot.theta.copy()
    if len(gdQuads)==1:
        if 0 in gdQuads:
            refThetas[:,h,1]=refThetas[:,h,0]+pi/2/harmonic
        else:
            refThetas[:,h,0]=refThetas[:,h,1]-pi/2/harmonic

    yL.append(1*cos(2*harmonic*pi*t[:,newaxis]+harmonic*refThetas[:,h]))
    varyPhaseIndL.append(len(yL)-1)

    #Add the 3rd harmonic ot the list(if it's there), and let it vary phase
    for harm in subtractHarmsL:
        if sigRot.sig.shape[1]>=harm:
            #_, sigH, covH=MT.rotate_quadrature_sample(sig.theta, sig.sig[:,h-1], sig.err[:,h-1])
            yL.append(sigRot.sig[:,harm-1])
            varyPhaseIndL.append(len(yL)-1)
        else:
            print("{}th harmonic doesn't exist, can't subtract it")

    #nanIs=isnan(sigRot.sig[:,0])
    sig2fit=sig
    fitparams, cov, adjchi2, res_obj=MF.orthogonal_multifit(t, sigRot.sig[:,h], yL,varyPhaseIndL, var=sigRot.err[:,h],  minFreq=minFreq, maxFreq=maxFreq, bPlot=bPlot, bReturnFitObj=True)
    Nsensors=len(sigSensL)

    sensCoeffs=array(fitparams[:Nsensors])
    labCoeffs=array(fitparams[Nsensors:Nsensors+2])#-thetaRef
    labCov=cov[Nsensors:Nsensors+2, Nsensors:Nsensors+2]
    sidCoeffs=array(fitparams[Nsensors+2:Nsensors+4])#-thetaRef
    sidCov=cov[Nsensors+2:Nsensors+4, Nsensors+2:Nsensors+4]
    sidAmp=SiderealFitData(t=nanmean(t), sig=sidCoeffs, err=sidCov, 
            chi2=adjchi2, 
            labTheta=array([0,pi/2]),#0*nanmean(-sigRot.theta[:,0], axis=0) ,
            sidTheta=None,
            meta=None)
    labAmp=LabSetData(t=nanmean(t), sig=labCoeffs, err=labCov, chi2=None, labTheta=array([0.,pi/2]))
    #print("subtract fit params: {0}".format(res_obj.params))


    # Convert data types down here
    #sigSubtracted=sig._replace(sig=(sig.sig.ravel()-sensCoeffs*array(sigSensL)).reshape(sig.sig.shape))

    return labAmp, sidAmp, None, res_obj

def subtract_correlations(sig, sigSensL, dontSubtract=[], scaleFact=None, minFreq=0.00005, maxFreq=None, bPlot=False):
    """ 
    """
    # Interpolate sigSensor to be the same as sig (if necessary))

    sigSensL=[interp(sig.t.ravel(), sigSens.t.ravel(), sigSens.sig.ravel()) for sigSens in sigSensL]

    sigSig=MT.naninterp(sig.sig.ravel())
    sigSensL=[MT.naninterp(sigSens) for sigSens in sigSensL]

    if minFreq is not None or maxFreq is not None: #Filter
        fs=1/median(diff(MT.nonnan(sig.t.ravel())))/(3600*24)
        if minFreq is not None and maxFreq is not None:
            sigFilt=MF.butter_bandpass_filter(sigSig, [minFreq, maxFreq], fs)
            sigSensL=[MF.butter_bandpass_filter(sigSens, [minFreq, maxFreq], fs) for sigSens in sigSensL]
        elif maxFreq is not None:
            sigFilt=MF.butter_lowpass_filter(sigSig, maxFreq, fs)
            sigSensL=[MF.butter_lowpass_filter(sigSens, maxFreq, fs) for sigSens in sigSensL]
        elif minFreq is not None:
            sigFilt=MF.butter_highpass_filter(sigSig, minFreq, fs)
            sigSensL=[MF.butter_highpass_filter(sigSens, minFreq, fs) for sigSens in sigSensL]
    else:
        sigFilt=sigSig
        #sigSensL=sigSensL

    # if @scaleFact is None, try to do a least-squares fit (or maybe something else in the longer-term)
    if scaleFact is None:
        t=MT.naninterp(sig.t.ravel())
        if hasattr(sig,'theta'):
            theta=sig.theta
        elif hasattr(sig, 'labTheta'):
            theta=sig.labTheta
        else:
            theta=None
        if theta is not None:
            theta=MT.naninterp(theta)
        dontSubtractD={ #Make these lambda functions so we won't evaluate them unless necessary
                'lab_sinusoid': lambda: array((sin(theta), cos(theta))),
                'sid_sinusoid': lambda: array((sin(2*pi*t+theta), cos(2*pi*t+theta)))
            }
        exog=array(sigSensL).T
        for st in dontSubtract: #Elements should either be functions (of t) or strings 
            arr=dontSubtractD[st]()
            exog=hstack([exog,arr.T])
        endog=sigFilt
        exog=sm.add_constant(exog,prepend=False)
        model=sm.WLS(endog, exog, missing='drop') #(?)
        res=model.fit()
        scaleFact=array(res.params[:len(sigSensL)])
        print("subtract fit params: {0}".format(res.params))


    sigSubtracted=sig._replace(sig=(sig.sig.ravel()-scaleFact*array(sigSensL)).reshape(sig.sig.shape))

    if bPlot or 1:
        figure()
        plot(t, sigFilt, '.')
        plot(t, exog, '.')
        plot(t, (scaleFact*array(sigSensL)).T)
    return sigSubtracted

def preprocess_raw(rawAmp, trigTimes, sigWindow):
    #vprint("using {0}".format(sigWindow))
    indxL= sliceSorted(rawAmp.t, t_0=trigTimes + secs2sd(sigWindow.offset),  delta_t=secs2sd(sigWindow.duration)) #indices of the points between each rotation
    #Now we 'regularize' it, making sure there's the same number of points in each slice (needed to make things faster later)
    NptsL=[ind.size for ind in indxL.ravel() if ind.size>1] #2 pts may be the minimum we can work with?

    minPtsI=argmin(NptsL)
    maxPts=max(NptsL)
    k=0;
    while NptsL[minPtsI]< 0.9*maxPts:
        vprint("A cut was too short: {0} vs {1}".format(NptsL[minPtsI], maxPts)) 
        NptsL.pop(minPtsI)
        minPtsI=argmin(NptsL)
        k+=1;
        if k>=20:
            print("Too mny short cuts. There's probably a segment of bad data")
            #raise ValueError("Too mny short cuts. There's probably a segment of bad data")
    minPts=NptsL[minPtsI]
    

    indxLReg=array([ind[:minPts] if ind.size>=minPts else -1*ones(minPts,dtype='i8') for ind in indxL.ravel()])
    
    indxLReg=indxLReg.reshape(indxL.shape[0], indxL.shape[1], minPts)

    rawAmpOut=RawData(t= rawAmp.t[indxLReg], sig=rawAmp.sig[indxLReg])
    rawAmpOut.sig[indxLReg==-1]=nan
    rawAmpOut.t[indxLReg==-1]=nan
    return rawAmpOut
    
def process_raw(rawAmp):
    def window_mean(arr, axis=0):
        win=windows.kaiser(arr.shape[axis], 10)
        win/=win.sum()
        #This isn't right
        return sum(arr*win[newaxis,newaxis,:],axis=axis)
        
    funcs_on_cutsL=[np.mean, lambda vals, **kwargs: np.std(vals, **kwargs)/sqrt(np.size(vals))]
    T=rawAmp.t.mean(axis=-1)
    Y=window_mean(rawAmp.sig, axis=-1) 
    Err=np.std(rawAmp.sig,axis=-1)
    #Y, Err= processedL[0:]

    #rawSig=array([y[ind[:minPts]] if ind.size>=minPts else zeros(minPts)*nan for ind in indxL.ravel()])
    #rawSig=rawSig.reshape(indxL.shape[0], indxL.shape[1], minPts)
    #rawSigF=fft(rawSig)
    return  PointData(t=T, sig=Y, err=Err,chi2=None,theta=None)

def process_points(ptLabAngle, pointAmp, cutAmp, stPtChi2Lim=50):
    """ Turn groups of points into sequences- except currently we go directly from a a group of 'cuts' (i.e. the cut-out raw data) to one value for each sequences.
    """
    vprint("chi2 limit on string points: {}".format(stPtChi2Lim))
    pointAmp=None
    ptT=mean(cutAmp.t, axis=-1)
    ptErr=std(cutAmp.sig, axis=-1)
    if ptLabAngle.shape[0] > cutAmp.sig.shape[0]:
        if ptLabAngle.shape[0]==cutAmp.sig.shape[0]+1:
            ptLabAngle=ptLabAngle[:-1]
        else:
            raise ValueError("angles shouldn't be more than 1 sequence bigger than the dataset (can happen if only part of a sequence is included). But here, ptLabAngle.shape[0] is {0} and cutAmp.shape[0] is {1}".format(ptLabAngle.shape[0], cutAmp.sig.shape[0]))
            
    # Process each group of sequences
    outL=zip(*[mgStringHarmonics(th, seq, None, chi2Filter=inf) for th,seq, err in zip(ptLabAngle, cutAmp.sig, ptErr)])
    if stPtChi2Lim is not inf:
        #origSigA=array(outL[1])
        chi2temp=nanmean(array(outL[3]), axis =-1).squeeze()
        smoothedChi2=MT.running_nanmedian(chi2temp, 100)
        outL=zip(*[mgStringHarmonics(th, seq, None, chi2Filter=stPtChi2Lim*smChi2) for th,seq, err, smChi2 in zip(ptLabAngle, cutAmp.sig, ptErr, smoothedChi2)])
    labAngle, seqSig, seqErr, seqChi2 = [array(p) for p in outL] #Array-ify the results

    #Old stuff for doing harmonicsFit
    #th0s,h1,cov1=zip(*[getHarmonicsFit(th, seq, err) for th,seq, err in zip(theta[:-1], ptSig, ptErr )])
    #h1=array(h1)
    #cov1=array(cov1)
    #covAs=cov1[:,0].squeeze()
    #th0s=array(th0s)
    #chi2=None  

    seqT=nanmean(ptT, axis=1)
    seqT=seqT[:,newaxis, newaxis]*ones(seqSig.shape) #make T's the same shape as the rest
    return CorrelationData(t=seqT, sig=seqSig, err=seqErr, chi2=seqChi2, theta=labAngle)

def filterBySensors(cutAmp, sensDataL=[], bView=True):
    #bView=True
    if not sensDataL:
        vprint("No sensor filtering")
        return cutAmp

    Npts=cutAmp.sig.size/(cutAmp.sig.shape[-1])
    badM=zeros(Npts, dtype=bool) #Mask for 'bad' data
    if bView:
        fig, axL=subplots(len(sensDataL)+1, 1, sharex=True)
        #vprint("Filtering out bad points using sensors {0}".format(sensorNameL))
    for k, sD in enumerate(sensDataL):
        #sDTs, [_, sDCutSig, sDCutErr], sDCutIndxL = sliceReduceData( sD[0], sD[1], trigTimes + secs2sd(sigWindow.offset), secs2sd(sigWindow.duration) ) 
        badSDM, xSM=MT.findOutliers(sD.sig.ravel(), windowLen=70, sigmas=3, bReturnMeanLine=True)
        badM= badM | badSDM

        if bView:
            axL[k+1].plot(sD.t.ravel(), sD.sig.ravel(),'.')
            axL[k+1].plot(sD.t.ravel()[badSDM], sD.sig.ravel()[badSDM],'o')
            #axL[k+1].plot(sDTs.ravel()[badSDM], sDCutSig.ravel()[badSDM],'o')
    if bView:
        sigA=cutAmp.sig.mean(axis=-1)
        axL[0].plot(sD.t.ravel(), sigA.ravel(),'.')
        axL[0].plot(sD.t.ravel()[badM], sigA.ravel()[badM],'.') 


    
    filtCutAmp=cutAmp._replace(sig=cutAmp.sig.copy())
    filtCutAmp.sig.reshape(-1,filtCutAmp.sig.shape[-1])[badM]=nan
    return filtCutAmp
#def process_sequences(sT, sSig, sErr, sLabAngles):
def process_sequences(corrAmp):
    '''Process sequence values, sSig, into final numbers
    '''
    sT,sSig, sErr, sChi2, sLabAngles=[s[:,0,:] for s in corrAmp]
    if sSig.size==0:
        return None
    #sErr*sqrt(sChi2)
    #winI=20
    #chi2=nanmedian(corrAmp.chi2[I]*(corrAmp.Neff[I]./stats.chi2.ppf(0.5, corrAmp.Neff)), axis=0) 
    #chisq1=nanmedian(v1(w,3,:).*(v1(w,4,:)./chi2inv(.5,v1(w,4,:))),1);
    #v1(j,2,:)=v1(j,2,:).*sqrt(chisq1);
    
    if not iterable(sLabAngles) or len(sLabAngles)==1 or unwrap(sLabAngles%pi).sum()<pi:#(maaaaaybeee)
        vprint("no lab angles given for sequence data in 'process_sequences'. Doing something dodgy that probably isn't right to make up for it")
        if sSig:
            
            labSig= average(sSig, weights=sErr**2)
        else:
            labSig=[];labErr=[];labAngles=[]
        labAmp=LabSetData(t=mean(sT), sig=labSig, err=None, chi2=None, labTheta=squeeze(labAngles))
    else:
        if sErr.ndim>2:
            sErr=vstack([sErr[:,0,0], sErr[:,1,1]]).T
        [labAngles, labSignal, labErr, labChi2]=zip(*getHarmonicsFit(sLabAngles, sSig, sErr))[0] #This returns two angles, and we'll limit ourselves to the first harmonic
        labAmp=LabSetData(t=mean(sT)*ones(len(labAngles)),
                sig=array(labSignal), 
                err=sqrt(diag(labErr)),
                chi2=labChi2,
                labTheta=array(labAngles))
        #labAmp=[array(labAmpTemp[0]), array(labAmpTemp[1]), sqrt(diag(labAmpTemp[2]))]
    ######SIDEREAL-FRAME FITTING##########
    #LV0=zip(*getHarmonicsFit(sT*2*pi, sSig, sErr))[0] #First harmonic only
    if sLabAngles is None:
        print("Not checked if it works when you don't give lab angles for sequences... we'll presume all measuremetns are taken in the same lab-orientation, but beware!")
        [sidAngles, sidSignal, sidErr, sidChi2]=zip(*getHarmonicsFit(sT*2*pi, sSig, sErr))[0]
    else:
        [sidAngles, sidSignal, sidErr, sidChi2]=zip(*getHarmonicsFit(sT*2*pi + sLabAngles, sSig, sErr))[0]

    sidFitDat=SiderealFitData(t=mean(sT)*ones(len(labAngles)),
                    #sig=array(sidSignal), err=sqrt(diag(sidErr)),
                    sig=array(sidSignal), err=sidErr,
                    labTheta=array(sidAngles), chi2=sidChi2, sidTheta=None, meta=None)
    #Repeat this for lab angles, and other sideral angles
    #sidAmp=[array(sidAmpTemp[0]), array(sidAmpTemp[1]), sqrt(diag(sidAmpTemp[2]))]
    return labAmp, sidFitDat

def process_continuous_raw(sigRaw, phaseRef, rotationRate, bPlot=False, th0FGz=2.87):
    #split up data

    #iterate through data
    #if phaseRef is a rawSig tuple (e.g. flux-gate), then fit it
    #else we'll assume it's time-theta pairs, in which case we'll also just interpolate 

    #Fit the raw signal to a sin/cos pair plus some polynomial terms
    tOffs=sigRaw.t[0]
    #K=3600*24
    tSplit, sSplit = MT.splitAtGaps(sd2secs(sigRaw.t-tOffs), sigRaw.sig)
    tMedSplit, fgSplit = MT.splitAtGaps(sd2secs(phaseRef.t-tOffs), phaseRef.sig)
    gdInds=[~isnan(s) for s in sSplit]
    tSplit=[t[gdI] for t, gdI in zip(tSplit, gdInds)]
    sSplit=[s[gdI] for s, gdI in zip(sSplit, gdInds)]
    #plot(sig.t, MT.smooth(sig.sig,500, window='hanning'))

    fp4L=[]
    err4L=[]
    chi2L=[]
    tL=[]
    fgFitL=[]
    phaseL=[]
    thL=[]
    k=0
    for t, s, tfg, sfg in zip(tSplit, sSplit, tMedSplit, fgSplit):
        if t.size<2:
            continue
        midT=t[t.size/2]
        cutOffSecs=0
        t-=midT
        I=t.searchsorted(t[-1]-cutOffSecs)
        t=t[:I]
        s=s[:I]

        tfg-=midT
        I=tfg.searchsorted(tfg[-1]-cutOffSecs)
        tfg=tfg[:I]
        sfg=sfg[:I]

        try:
            p0=[ rotationRate,  0.03262361,  0.03052155,  5.41955202]
            gdI=~isnan(sfg)
            fFG=lambda t, f, a1, a2, offs: a1*sin(2*pi*f*t) + a2*cos(2*pi*f*t) + offs
            fpFG, err=curve_fit(fFG, tfg[gdI], sfg[gdI],p0=p0)
            fgFitL.append(fpFG)

            #figure()
            #plot(tfg,sfg)
            #plot(tfg, fFG(tfg, *fpFG))
            #figure()
            #plot(tfg, MT.smooth(sfg-fFG(tfg, *fpFG), 10))


            th0=arctan2(fpFG[1], fpFG[2]) # The angle where the fluxgate shows a minimum deviationF
            phaseL.append(th0)

            thfg=2*pi*fpFG[0]*tfg-th0%(2*pi)-2*pi
            #thfg=2*pi*(fpFG[0]-0.005)*tfg-th0%(2*pi)-2*pi
            th=interp(t, tfg,thfg)

            f4=lambda th, a1x, a1y, a2x, a2y, a3x,a3y, p1, p2, p3, p4, p5, offs: a1x*sin(th) + a1y*cos(th) + a2x*sin(2*th) + a2y*cos(2*th)+ a3x*sin(3*th) + a3y*cos(3*th)+ p1*th +p2*th**2 + p3*th**3 + p4*th**4 + p5*th**5 + offs

            p0=[0.1, 0.1, 1e-5, 1e-7, 1e-9, 0.08]

            sampleRate=1./median(diff(t))
            #print("Sample rate is {}".format(sampleRate))
            Nds=round(sampleRate)

            p0=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1e-5, 1e-7, 1e-9, 1e-9, 1e-9, 0.08]
            th_ds, s_ds=MT.downsample_npts(th,Nds), MT.downsample_npts(s, Nds)
            if th_ds.size <= len(p0) +1:
                raise RuntimeError("not enough data here")
            fp4, err4=curve_fit(f4, th_ds, s_ds, p0=p0)

            chi2= sum((f4(th, *fp4)-s)**2)/(th.size-len(p0))
            #fp, err=curve_fit(f, th,s,p0=p0)
            #fp2, err2=curve_fit(f2, th, s,p0=p0)
            #figure()
            #plot(t,MT.smooth(s,Nds, window='hanning'))
            #plot(t, f(th, *fp))
            #plot(t, f2(th, *fp2))
            #print("fp4: {0}".format(fp4))
            fp4L.append(fp4)
            chi2L.append(chi2)
            #err4L.append(sqrt(diag(err4)))
            err4L.append(err4)
            tL.append(midT)
            thL.append(th)

        except RuntimeError:
            print("No fit at {}".format(midT))


    fp4A=array(fp4L)
    fpA=fp4A[:,:2]
    fp2A=fp4A[:,2:4]
    fp3A=fp4A[:,4:6]
    err4A=array(err4L)
    errA=err4A[:,:2,:2]
    err2A=err4A[:,2:4,2:4]
    err3A=err4A[:,4:6,4:6]

    chi2A=array(chi2L)


    tA=array(tL)
    if bPlot:
        figure()
        errorbar(tA, fpA[:,0], errA[:,0],fmt='.')
        errorbar(tA, fpA[:,1], errA[:,1],fmt='.')
        title("1st harmonic")
        figure()
        errorbar(tA, fp2A[:,0], err2A[:,0],fmt='.')
        errorbar(tA, fp2A[:,1], err2A[:,1],fmt='.')
        title('2nd Harmonic')

        figure()
        plot(tA, [p[0] for p in fgFitL])
        figure()
        plot(tA, [p[1] for p in fgFitL])
        plot(tA, [p[2] for p in fgFitL])
        figure()
        plot(tA, phaseL)
    N=tA.size
    tSeq=tile(secs2sd(tA)+tOffs, [2,2,1]).T
    sigSeq=hstack( [fpA[:,newaxis, :2], fp2A[:,newaxis,:2], fp3A[:,newaxis,:2]] )
    errSeq=hstack( [errA[:,newaxis, :2], err2A[:,newaxis,:2], err3A[:, newaxis,:2]] )
    #chi2Seq=hstack( [chi2_1A[:,newaxis], chi2_2A[:,newaxis]] )
    chi2Seq=hstack( [chi2A[:,newaxis], chi2A[:,newaxis], chi2A[:,newaxis]] )
    chi2Seq=dstack([chi2Seq, chi2Seq])
    sgn=-1 if rotationRate<0 else 1
    thetaSeq1=vstack( [zeros(N), zeros(N), zeros(N)])
    #thetaSeq2=vstack( [sgn*ones(N)*pi/2, sgn*ones(N)*pi/4, sgn*ones(N)*pi/6] )
    thetaSeq2=vstack( [sgn*ones(N)*pi/2, sgn*ones(N)*pi/2, sgn*ones(N)*pi/2] )
    #thetaSeq1=vstack( [zeros(N), ones(N)*pi/2])
    #thetaSeq2=vstack( [zeros(N), ones(N)*pi/4])

    #thetaSeq2=vstack( [ones(N)*pi/2, ones(N)*pi/4])
    thetaSeq=dstack([thetaSeq1.T, thetaSeq2.T])
    seqAmp=CorrelationData(
            t=tSeq.copy(),
            sig=sigSeq.copy(),
            err=errSeq.copy(),
            chi2=chi2Seq.copy(),
            theta=thetaSeq.copy() + th0FGz, #th0FGz is the orientation of the FGz correlation
            )
    return seqAmp


def split_and_process_sequences(seqAmp):
    seqT, seqSig, seqErr, seqChi2, labAngle=seqAmp
    def seqFiltAngle(seqAmp, th):
        mask=nearby_angles(seqAmp.theta[:,0,0], th, 0.1)
        return CorrelationData(*[s[mask] for s in seqAmp])
    seqAmp1=CorrelationData(seqAmp.t, *[d[:,0,0].squeeze() for d in seqAmp[1:]])
    seqAmp2=CorrelationData(seqAmp.t, *[d[:,0,1].squeeze() for d in seqAmp[1:]])



    thetaRef, xI, yI=group_by_axis(seqAmp1.theta)        
    
    seqAmp1_0=CorrelationData(*[d[xI] for d in seqAmp1])
    seqAmp1_90=CorrelationData(*[d[yI] for d in seqAmp1])

    #seqAmp1_0=seqAmp1_0._replace(theta= seqAmp1_0.theta*sin(seqAmp1.theta[xI]-thetaRef))
    #seqAmp1_90=seqAmp1_90._replace(theta=seqAmp1_90.theta*cos(seqAmp1.theta[yI]-thetaRef))

    #seqSig1_0=seqSig1[xI]*sin(labAngle1[xI]-thetaRef)
    #seqErr1_0=seqErr1[xI]
    #seqSig1_90=seqSig1[yI]*cos(labAngle1[yI]-thetaRef)
    #seqErr1_90=seqErr1[yI]
    #seqT_0=seqT[xI]
    #seqT_90=seqT[yI]

    #seqAmp_0=CorrelationData([d[xI] for d in 
    #seqAmp_0CorrelationData(t=seqT_0, sig=seqSig1_0, err=seqErr1_0, chi2=seqChi21[xI], labTheta=labAngle1[xI], None)

    seqAmpL=(seqAmp1_0, seqAmp1_90, seqAmp1)
    ampL=[ process_sequences(sAmp) for sAmp in seqAmpL   ]

    #labAmp1_0, sidAmp1_0=process_sequences(seqAmp1_0)
    #labAmp1_90, sidAmp1_90=process_sequences(seqAmp1_90)
    #labAmp, sidAmp=process_sequences(seqAmp1)
    #labAmp0, sidAmp0=process_sequences(seqT_0, seqSig1_0, seqErr1_0, thetaRef)
    #labAmp90, sidAmp90=process_sequences(seqT_90, seqSig1_90, seqErr1_90, thetaRef+pi/2)
    #labAmp, sidAmp=process_sequences(seqT, seqSig1, seqErr1, labAngle1)

    #labAmpAlt= [labAmp0, labAmp90]
    #NS, EW=average([sidAmp0[1], sidAmp90[1][::-1]], axis=0, weights=[sidAmp[2]**2, sidAmp90[2][::-1]])
    #NS_err, EW_err = sqrt(sidAmp0[2]**2 + sidAmp90[2][::-1]**2)/2.
    #sidAmpAlt=[ (NS, EW), (NS_err, EW_err)] 
    
    return zip(*ampL) + [seqAmpL] # --> (labAmps, sidAmps, seqAmps)
    #return [ (labAmp, labAmpAlt), (sidAmp, sidAmpAlt)]

def view_raw(rawAmp0, cutAmpL, trigTimes, cutAmpLNames=None, figName=None):
    if figName is not None:
        figDontReplace(figName + ' -raw')
    else:
        figDontReplace()
    gs=GridSpec(2,3)
    ax1=subplot(gs[0,:])
    ax2=subplot(gs[1,0])
    ax3=subplot(gs[1,1])
    ax4=subplot(gs[1,2])

    ax1.plot(rawAmp0.t, rawAmp0.sig)
    for cutAmp, name in zip(cutAmpL, cutAmpLNames):
        ax1.plot(cutAmp.t.ravel(), cutAmp.sig.ravel(), label=name, markersize=5)

    #ax1.vlines(trigTimes, rawAmp0.sig.min(), rawAmp0.sig.max(), linestyles='dotted')
    ax1.set_title('raw signal')
    ax1.set_xlabel('sd')
    ax1.legend()
    #cutAmp.sigF=fft(cutAmp.sig- nanmean(cutAmp.sig, axis=-1)[:,:,newaxis])
    cutAmpSigF, fax=MF.rPSD(cutAmp.sig- nanmean(cutAmp.sig, axis=-1)[:,:,newaxis], t=cutAmp.t[0][0])
    ax2.plot(fax/3600/24, nanmean(nanmean(abs(cutAmpSigF), axis=0), axis=0))
    ax2.set_title('PSD')
    ax2.set_xlabel('Hz')

    if 0:
        ax3.hist(nonnan(nanstd(cutAmp.sig, axis=-1).ravel()), bins=30)
        ax3.set_title('std')

        ttemp=cutAmp.t[0][0]
        slopes=[polyfit(ttemp, y, 1)[0] for y in cutAmp.sig.reshape(-1, cutAmp.sig.shape[-1]) if not all(isnan(y))]
        ax4.hist(nonnan(slopes), bins=30)
        ax4.set_title('slopes')

def view_correlation_filtering(pointAmpInit, correlationAmpInit, pointAmpFilt, correlationAmpFilt, figName=None, sharexAx=None):
    if figName is None:
        figDontReplace()
    else:
        figDontReplace(figName+' -Corr diff')
    gs=GridSpec(3,1)

    pointAxs=MT.plot_errorbar_and_hist(pointAmpInit.t.ravel(), pointAmpInit.sig.ravel(), pointAmpInit.err.ravel(), subplotSpec=gs[0], sharexs=[sharexAx,None])
    pointAxs=MT.plot_errorbar_and_hist(pointAmpFilt.t.ravel(), pointAmpFilt.sig.ravel(), 
            pointAmpFilt.err.ravel(), *pointAxs
            )

    thetaRef, xI, yI=group_by_axis(correlationAmpInit.theta[:,0,:].ravel())        
    corAxs=MT.plot_errorbar_and_hist(correlationAmpInit.t.ravel()[xI], 
            correlationAmpInit.sig.ravel()[xI]*sin(correlationAmpInit.theta.ravel()[xI]),
            correlationAmpInit.err.ravel()[xI], subplotSpec=gs[1], sharexs=[pointAxs[0], None]
            )
    MT.plot_errorbar_and_hist(correlationAmpInit.t.ravel()[yI], correlationAmpInit.sig.ravel()[yI]*cos(correlationAmpInit.theta.ravel()[yI]), correlationAmpInit.err.ravel()[yI], *corAxs)

    thetaRef, xI, yI=group_by_axis(correlationAmpFilt.theta[:,0,:].ravel())        
    corAxs=MT.plot_errorbar_and_hist(correlationAmpFilt.t.ravel()[xI], 
            correlationAmpFilt.sig.ravel()[xI]*sin(correlationAmpFilt.theta.ravel()[xI]),
            correlationAmpFilt.err.ravel()[xI], *corAxs
            )
    MT.plot_errorbar_and_hist(correlationAmpFilt.t.ravel()[yI], correlationAmpFilt.sig.ravel()[yI]*cos(correlationAmpFilt.theta.ravel()[yI]), correlationAmpFilt.err.ravel()[yI], *corAxs)

    chi2Axs=MT.plot_scatter_and_hist(correlationAmpInit.t.ravel(), correlationAmpInit.chi2.ravel(), subplotSpec=gs[2], sharexs=[pointAxs[0],None], scatD={'c':'b'})
    chi2Axs=MT.plot_scatter_and_hist(correlationAmpFilt.t.ravel(), correlationAmpFilt.chi2.ravel(), *chi2Axs, scatD={'c':'g'})

def view_correlations(pointAmp, correlationAmp, figName=None, sharexAx=None, correlationSigComp=None ):
    if figName is None:
        figDontReplace()
    else:
        figDontReplace(figName+' -Corr')
    gs=GridSpec(3,1)
    gs=GridSpec(5,1)

    if correlationSigComp is not None and 1:
        #refNonNanM=~isnan(correlationSigComp)

        diffI=where( (correlationSigComp[:,0].ravel()!=correlationAmp.sig[:,0].ravel()) & (~isnan(correlationSigComp[:,0].ravel())))[0]
        #diffMask=correlationSigComp[refNonNanM]!=correlationAmp.sig[refNonNanM]

        #diffI=arange(correlationSigComp.size)[refNonNanM][diffMask.ravel()]
        #diffAmp[diffAmp==0]=nan
        #diffMask=~isnan(diffAmp.ravel())
        #pdb.set_trace()
    
    pointAxs=MT.plot_errorbar_and_hist(pointAmp.t.ravel(), pointAmp.sig.ravel(), pointAmp.err.ravel(), subplotSpec=gs[0], sharexs=[sharexAx,None])

    correlationAmp=rotate_sequence_amps(correlationAmp)
    thetaRef, xI, yI=group_by_axis(correlationAmp.theta[:,0,:].ravel())        
    err1h=sqrt(vstack([correlationAmp.err[:,0,0,0],correlationAmp.err[:,0,1,1]]).T)

    corAxs=MT.plot_errorbar_and_hist(correlationAmp.t[:,0].ravel()[xI], 
        correlationAmp.sig[:,0].ravel()[xI]*cos(correlationAmp.theta[:,0].ravel()[xI]), 
        err1h.ravel()[xI], 
        subplotSpec=gs[1], sharexs=[pointAxs[0], None])
    MT.plot_errorbar_and_hist(correlationAmp.t[:,0].ravel()[yI], correlationAmp.sig[:,0].ravel()[yI]*sin(correlationAmp.theta[:,0].ravel()[yI]), err1h.ravel()[yI], *corAxs)

    #pdb.set_trace()
    if correlationSigComp is not None:
        #corAxs[0].plot(correlationAmp.t.ravel()[diffMask==xI], diffAmp.ravel()[diffMask==xI], 'rx')
        xdI=intersect1d(xI, diffI)
        ydI=intersect1d(yI, diffI)
        if xdI.size:
            corAxs[0].plot(correlationAmp.t[:,0].ravel()[xdI], correlationSigComp[:,0].ravel()[xdI]*cos(correlationAmp.theta[:,0].ravel()[xdI]), 'rx')
        #corAxs[0].plot(correlationAmp.t.ravel()[diffMask==yI], diffAmp.ravel()[diffMask==yI], 'yx')
        if ydI.size:
            corAxs[0].plot(correlationAmp.t[:,0].ravel()[ydI], correlationSigComp[:,0].ravel()[ydI]*sin(correlationAmp.theta[:,0].ravel()[ydI]), 'yx')


    ax=subplot(gs[2])
    #ax.plot(correlationAmp.t[:,0].ravel()[yI], correlationAmp.chi2[:,0].ravel()[yI],'o')
    #ax.plot(correlationAmp.t[:,0].ravel()[xI], correlationAmp.chi2[:,0].ravel()[xI],'o')
    chi21hAxs=MT.plot_scatter_and_hist(correlationAmp.t[:,0].ravel()[xI], correlationAmp.chi2[:,0].ravel()[xI], subplotSpec=gs[2])
    MT.plot_scatter_and_hist(correlationAmp.t[:,0].ravel()[yI], correlationAmp.chi2[:,0].ravel()[yI], *chi21hAxs)
    if correlationAmp.t.shape[1]>1: #2H
        err2h=sqrt(vstack([correlationAmp.err[:,1,0,0],correlationAmp.err[:,1,1,1]]).T)
        thetaRef, xI2, yI2=group_by_axis(2*correlationAmp.theta[:,1,:].ravel())        
        cor2hAxs=MT.plot_errorbar_and_hist(correlationAmp.t[:,1].ravel()[xI2], correlationAmp.sig[:,1].ravel()[xI2]*cos(2*correlationAmp.theta[:,1].ravel()[xI2]), err2h.ravel()[xI2], subplotSpec=gs[3], sharexs=[pointAxs[0], None])

        MT.plot_errorbar_and_hist(correlationAmp.t[:,1].ravel()[yI2], correlationAmp.sig[:,1].ravel()[yI2]*sin(2*correlationAmp.theta[:,1].ravel()[yI2]), err2h.ravel()[yI2], *cor2hAxs)
        subplot(gs[4]).plot(correlationAmp.t[:,1].ravel(), correlationAmp.chi2[:,1].ravel(),'o')

    #chi2Axs=MT.plot_scatter_and_hist(correlationAmp.t[:,0].ravel(), correlationAmp.chi2[:,0].ravel(), subplotSpec=gs[2], sharexs=[pointAxs[0],None])

    if not MT.nearby_angles(thetaRef, 0, 0.1):
        corAxs[0].set_title('Reference angle: {0}'.format(thetaRef/pi*180))
    #tight_layout()
        
def view_sidereal(seqTup, sidFitTup, labFrameTup=None):
    """Not functional anymore
    """
    figure()
    fx=lambda th, ax: ax*sin(th)
    fy=lambda th, ay: ay*cos(th)
    f=lambda th, ax, ay: ax*sin(th) + ay*cos(th)

    gs=GridSpec(2,1)
    sigAxs=MT.plot_errorbar_and_hist(seqTup.t.ravel(), seqTup.sig.ravel(), seqTup.err.ravel(), subplotSpec=gs[0])
    if seqTup.chi2 is not None:
        chiAxs=MT.plot_scatter_and_hist(seqTup.t, seqTup.chi2, gs[1])
        chiAxs[0].set_title(r'$\chi^2$')


    sigRot=MT.rotMat2d(-sidFitTup.labTheta[0]).dot(sidFitTup.sig)
    errRot=abs(MT.rotMat2d(-sidFitTup.labTheta[0]).dot(sidFitTup.err)) #This is very fragile- should do full covariance in here.
    sigAxs[0].plot(seqTup.t, f(2*pi*seqTup.t + 0*sidFitTup.labTheta[0], *sidFitTup.sig),
    label="$X$:{:.3} $\pm$ {:.3}, $Y$:{:.3} $\pm$ {:.3}".format(
    sigRot[0,0], errRot[0,0], sigRot[0,1], errRot[0,1])
    )
    sigAxs[0].legend()
    sigAxs[0].set_title('Sidereal Signal (lab: $S_{0:.3}={1:.3} \pm {2:.3}$, $S_{3:.3}= {4:.3} \pm {5:.3}$'.format(
                labFrameTup.labTheta[0]/pi*180, labFrameTup.sig[0], labFrameTup.err[0], 
                labFrameTup.labTheta[1]/pi*180, labFrameTup.sig[1], labFrameTup.err[1] ))
    
    #Could also do another here if we're fitting both
    #bothAx1.legend()

    if 0:
        for T, A, LV in ([seqT_0, seqSig1_0, sidAmp0], [seqT_90, seqSig1_90, sidAmp90]):
            if len(A)>3:
                axLV.plot(T, f(2*pi*T+ LV[0][0] -0*thetaRef, *LV[1]  ), label="$_X$:{:.3} $\pm$ {:.3}, $_Y$:{:.3} $\pm$ {:.3}".format(LV[1][0], sqrt(LV[2][0]), LV[1][1], sqrt(LV[2][1])))
            else:
                vprint("Need at least 3 sequence points")
        axLV.text( 0.5, 0.05, r'$\theta_0 = {:.3}$'.format(thetaRef/pi*180), bbox=dict(facecolor='grey', alpha=0.1), transform=axLV.transAxes, fontsize=16)
        axLV.legend(loc=0)

def combine_angle_amps(ampL, plot=False):

        ampL=[amp for amp in ampL if amp is not None]
        ts = [ sAmp.t for sAmp in ampL]
        if ampL:
            #Might not rotate in this case, but just in case
            (ths, ys, covs)= zip(*[ MT.rotate_quadrature_sample(sAmp.labTheta, sAmp.sig, sAmp.err) for sAmp in ampL if sAmp is not None])
            ys=array(ys)
            if covs[0].ndim>1:
                errs=array([sqrt(cov.diagonal()) for cov in covs]) #cheat and throw away off-diagonals
            else:
                errs=array(covs)


            ts=array(ts)
            if ts.ndim < ys.ndim:
                ts=tile(ts[:,newaxis],2)
            mn, wEr, apparentUnc, chi2= MT.weightedStats(ys, 1./(errs**2), axis=0, bReturnChi2=True)
            varSclFact=where( (chi2>1.) & (~isinf(chi2)), chi2, 1.)
            #calcedUnc=sqrt(nansum(errs**2,axis=0)/sum(~isnan(errs))**2)
            calcedUnc=1/sqrt(nansum(1/errs**2,axis=0))
            combAmp=SiderealFitData(mean(ts), mn, diag(calcedUnc**2*varSclFact), chi2, [0,pi/2], [0,pi/2], None) 
        else:
            ts=ys=errs=array([])
            combAmp=None #SiderealFitData(*(7*[None]))
        if plot:
            if isinstance(plot, Axes):
                axL=[plot, plot]
            elif hasattr(plot, "__getitem__"):
                axL=plot #assume it's a list with two axes
            else:
                figure()
                axL=[gca(), gca()]

            for t, y, e, ax in zip(ts.T, ys.T, errs.T, axL):
                ax.errorbar( t, y, e, fmt='.') 
            axL[0].set_title('x')
            axL[1].set_title('y')

        #gdInd=~(isnan(ys) | isinf(ys) | isinf(errs))
        #ys=ys[gdInd]
        #errs=errs[gdInd]
        #if sum(~isnan(errs)) <1:
            #return None
        #sm.WLS(
        
        return combAmp
        #return mn, calcedUnc, apparentUnc, wEr, chi2

def spSearchSet(timestamp, rSpec, sigWindow, bgWindow, startTime=-inf, endTime=inf, bPlotAll=True ): 
        '''
        Todos:
            - Filter out sequence points by chi squared, error bar, or even deviation from the mean
                - Low pass filter first, then look at slope or remaining RMS for weighting value
                - Histogram the string points and try to find any common signature to the outliers
            - Do a multiple regression for the correlation data vs the signal data
            - Output extra details like 'cosmic' angle, and mean signal in the lab-frame.
        '''
        ####################LOAD DATA##########################
        if hasattr(timestamp, 'fastD'): #If we've actually been handed a dataset, not a timestamp
            D=timestamp
        else: #Load one
            D=SPDataSet(timestamp, startTime=startTime, endTime=endTime)

        ###################CUT AND PROCESS INTO POINTS###################
        rawAmp0=RawData(t=D.fastD[0], sig=D.fastD[1])
        trigTimes=makeTriggers(D.fastD[0], rSpec, bIncludeExtraRot=True);
        cutAmp=preprocess_raw(rawAmp0, trigTimes, sigWindow)
        pointAmp=process_raw(cutAmp)
        if bPlotAll:
            view_raw(rawAmp0, cutAmp, trigTimes)

        #################### FILTER OUT BAD POINTS#######################
        if 0:
            filterSensorNameL= ['Tilt Sensor 1'],
            rawSig, badM= filterBySensors(cutAmp,
                        sensDataL=getSensors(D, filterSensorNameL),
                        bView= bPlotAll)
         
        ##################PROCESS BY SEQUENCE (CORRELATION)####################
        seqAmp=process_points(D.apparatusAngles, pointAmp, cutAmp )
        #view_correlations(seqAmp)

        ################PROCESS ALL SEQUENCES, I.E. WHOLE SET####################
        ## Divide the sequences into groups (e.g. EW and NS) and process them, returning also the lab-fram numbers
        labAmps, sidAmps, seqAmps= split_and_process_sequences(seqAmp)

        view_sidereal(seqAmps[0], sidAmps[0] )
        view_sidereal(seqAmps[1], sidAmps[1] )
        view_sidereal(seqAmps[2], sidAmps[2] )


        return labAmps, sidAmps#, sidAmpAlt

if __name__=="__main__":
    from pylab import *
    from scipy import random


    if 0:
        # Check string analysis:
        N=10000
        t=linspace(0,10,N)
        dat=(arange(N)%2 -0.5)*1
        dat+=random.normal(0, size=N)

        plot(t,dat)
        vprint(stringAnalysis(t, dat, ones(N)*0.5))
    
    if 0: #Check MakeTestData
        bTest=True
        dat=MakeTestData(RotationSpec(startingAngle=0, delay=9, interval=30, numRotations=20, rotAngles=[180], extraRotAngle=90), amp=1, sigma=0.01, zeroingTime=500)

    if 0:
        D=loadSet('5348.82', startTime=5348.86)
        nameL=['Tilt Sensor 1']
        rSpec=RotationSpec(startingAngle=0, delay=9, interval=30, numRotations=20, rotAngles=[-180], extraRotAngle=-90)
        [ts1]=getSensors(D, nameL)
        #out=sliceReduceData( ts1[0], ts1[1], 
                            #trigTimes + secs2sd(sigWindow.offset), secs2sd(sigWindow.duration) )
        window=Window(offset=-5.0, duration=4)
        trigTimes=makeTriggers(ts1[0], rSpec, bIncludeExtraRot=True);
        ts1Ts, [_, ts1CutSig, ts1CutErr], ts1CutIndxL = sliceReduceData( ts1[0], ts1[1], trigTimes + secs2sd(window.offset), secs2sd(window.duration) ) 
        #M=nanmean(ts1CutSig.ravel())
        badM, ts1SM=MT.findOutliers(ts1CutSig.ravel(), windowLen=50, sigmas=3, bReturnMeanLine=True)
        figure()
        #as
        plot(ts1Ts.ravel(), ts1CutSig.ravel(),'.')
        plot(ts1Ts.ravel()[badM], ts1CutSig.ravel()[badM],'.')
        plot(ts1Ts.ravel(), ts1SM, lw=2)
    if 0: #Check analysis
        timestamps=[
            '5227.40',
            #'5310.41',
            #'5318.41',
            #'5348.86',
        ]
        rSpec=RotationSpec(startingAngle=0, delay=9, interval=10, numRotations=20, rotAngles=[-180], extraRotAngle=-90)
        res=spSearchSet(
            MakeTestData(rSpec, amp=10.05, sigma=2, N=800000, phi=-0.15*pi, zeroingTime=30, fracDev=0.00, sizeDev=50.0),
            rSpec,
            Window(offset=-3, duration=2), Window(offset=-1,duration=1),
            bPlotAll=True,
            )
    if 0:
        from SPDataSet import SPDataSet
        rSpec=RotationSpec(startingAngle=0, delay=9, interval=10, numRotations=20, rotAngles=[-180], extraRotAngle=-90)
        sd,sig=makeTestData(
        rSpec, amp=10.05, sigma=2, N=800000, phi=-0.15*pi, zeroingTime=30, fracDev=0.00, sizeDev=50.0)
        ds=SPDataSet('test', preloadDict={'fastD': [sd, sig, 'sig']}, rSpec=rSpec )
