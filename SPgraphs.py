"""
TODO:
    Need to actually calculate c, b coefficients and plot them. Initially this can just involve multiplying or not multiplying by the polarisation in the final result
"""
import matplotlib as mpl
mpl.rcParams['figure.figsize']=[13,11]
from pylab import *
import SPDataSet as sd
import SPfuncs as spf
import pdb
from MT import vprint
from functools import partial
figdir='figs2'

#2013 Data
timestampL=[
'4803.31',
'4812.51',
'4818.33',	
'4827.26',
'4833.40',
'4834.39',
'4838.50',
'4840.65',
'4850.54',
'4857.47',
'4859.39',
'4864.65',
'4865.75',
'4867.48',
'4868.41',
'4869.74',
'4876.97',
'4881.71',
'4885.53',
'4888.50',
'4889.50',
'4894.57',
'4898.68',
'4901.78',
'4902.85',
'4903.58',
'4906.58',
'4911.09',
'4914.07',
'4929.88',
'4938.05',
'4945.09',
'4953.70',
'4959.91',
'4966.95',
]
#timestampL=timestampL[:-6]

timestampL=[
    #'5209.26',
    #'5240.33',
    '5249.72',
    '5273.74',
    '5282.23',
    #'5292.71',
    '5294.31',
    #'5296.33',
    '5310.41',
    #'5318.41',
    '5348.82',
    '5369.87',
    '5386.02',
    '5390.74',
    '5410.04',
    '5410.87',
    '5411.92',
    '5415.95',
    '5420.00',
    #'5421.02',
    #'5422.00',
    '5422.80',
    '5424.94',
]
dsL=[sd.SPDataSet(ts) for ts in timestampL]

def edit_infos(dsLin=None):
    """Walk through the datasets and display their metadata for editing/viewing
    """
    if dsLin is None:
        dsLin =dsL

    for ds in dsLin:
        ds.edit_set_info()
        raw_input("Enter for next")

import copy
def combinedD(d1, d2):
    out=copy.copy(d1); out.update(d2)
    return out
Ndiv=10
def analyseBy(datSetL=None, sidAmpDict={}, dsModifyF=None, generalF=None, plot=False):
    sidAmpL=[]
    if datSetL is None:
        datSetL=dsL
    for ds in datSetL:
        try:
            if dsModifyF is not None:
                ds, D=dsModifyF(ds, sidAmpDict)
            else:
                D=sidAmpDict
            #D= sidAmpDict(ds) if callable(sidAmpDict) else sidAmpDict

            if generalF is not None:
                amp=generalF(ds, D)
            else:
                amp=ds.sidAmp(**D)
            if amp is not None and not hasattr(amp, 't'):
                sidAmpL.extend(amp)
            else:
                sidAmpL.append(amp)

        except (ZeroDivisionError, ValueError):
            sys.exc_clear()
            #pass #Failed probably because there was no data, so we won't add it to the list
    #figure('Pol dn (split)')
    if not hasattr(plot, 'draw') and plot==True:
        figure()
        plot=gca()
    return spf.combine_angle_amps(sidAmpL, plot), sidAmpL

def polUp(plot=False, **kwargs):
    sidAmpL=[]
    #for ts in timestampL:
    for ds in dsL:
        #ds=sd.SPDataSet(ts)
        sidAmpL.append(ds.sidAmp(genSeqFiltF=sd.genSeqFiltPol(1),**kwargs))
    figure('Pol up (by set)')
    ax=gca()
    return spf.combine_angle_amps(sidAmpL,plot)

def polDn(plot=False, **kwargs):
    sidAmpL=[]
    for ds in dsL:#timestampL:
        #ds=sd.SPDataSet(ts)
        sidAmpL.append(ds.sidAmp(genSeqFiltF=sd.genSeqFiltPol(-1),**kwargs))
    figure('Pol dn (by set)')
    ax=gca()
    return spf.combine_angle_amps(sidAmpL,plot)

def polUpSplit(plot=False, **kwargs):
    sidAmpL=[]
    #for ts in timestampL:
    for ds in dsL:
        try:
            mn, calcedUnc, apparentUnc, wEr=spf.combine_angle_amps(
                    ds.sidAmp(sd.seqFiltFuncInterleave(Ndiv), genSeqFiltF=sd.genSeqFiltPol(1), **kwargs)
                    )
            sidAmpFull=ds.sidAmp()
            sidAmpL.append(spf.SiderealFitData(t=sidAmpFull.t, sig=mn, err=diag(apparentUnc)**2, chi2=None, sidTheta=sidAmpFull.sidTheta, labTheta=sidAmpFull.labTheta))

        except (ZeroDivisionError, ValueError):
            sys.exc_clear()
            #pass #Failed probably because there was no data, so we won't add it to the list
    #figure('Pol up (split)')
    return spf.combine_angle_amps(sidAmpL, gca())

def polDnSplit(plot=False, **kwargs):
    sidAmpL=[]
    #for ts in timestampL:
    for ds in dsL:
        try:
            mn, calcedUnc, apparentUnc, wEr=spf.combine_angle_amps(
                    ds.sidAmp(sd.seqFiltFuncInterleave(Ndiv), genSeqFiltF=sd.genSeqFiltPol(-1), **kwargs)
                    )
            sidAmpFull=ds.sidAmp()
            sidAmpL.append(spf.SiderealFitData(t=sidAmpFull.t, sig=mn, err=diag(apparentUnc)**2, chi2=None, sidTheta=sidAmpFull.sidTheta, labTheta=sidAmpFull.labTheta))

        except (ZeroDivisionError, ValueError):
            sys.exc_clear()
            #pass #Failed probably because there was no data, so we won't add it to the list
    #figure('Pol dn (split)')
    return spf.combine_angle_amps(sidAmpL, gca())

def byDay(plot=False, **kwargs):
    #By day
    sidAmpL=[]
    for ds in dsL:
        #ds=sd.SPDataSet(ts)
        print("timestamp: {}".format(ds.timestamp))
        sdL=ds.sidAmp(sd.seqFiltFuncTime(), **kwargs)
        if sdL is not None:
            if hasattr(sdL,'t'):
                sidAmpL.append(sdL)
            else:
                sidAmpL.extend(sdL)
        #if len(sidAmpL)==15:
        #    pdb.set_trace()
        ds._cache.clear()
        #wtMn, calcedUnc, apparentUnc, wtErr= spf.combine_angle_amps(sidAmpL)
        #(ths, ys, covs)= zip(*[ MT.rotate_quadrature_sample(sAmp.labTheta, sAmp.sig, sAmp.err) for sAmp in ampL])

        #ys=array(ys)
        #ts=array(ts)
        #errs=array([sqrt(cov.diagonal()) for cov in covs]) #cheat and throw away covariance

        #figure()
        #for t, y, e in zip(ts.T, ys.T, errs.T):
        #    errorbar( t, y, e, fmt='.') #Probaby not quite right
    #figure('By day')
    ax=gca()
    #mn, calcedUnc, apparentUnc, wEr=spf.combine_angle_amps(sidAmpl)
    return spf.combine_angle_amps(sidAmpL,plot)

def bySetEarly(plot=False, **kwargs):
    sidAmpL=[]
    for ds in dsL:
        #ds=sd.SPDataSet(ts)
        win=ds.set_info['windows']['sig']
        
        win['duration']=win['duration']/2
        sidAmpL.append(ds.sidAmp(sigWindow=spf.Window(**win)))
    #figure('Early')
    ax=gca()

    return spf.combine_angle_amps(sidAmpL,plot)
def bySetLate(plot=False, **kwargs):
    sidAmpL=[]
    for ds in dsL:
        #ds=sd.SPDataSet(ts)
        win=ds.set_info['windows']['sig']
        #win=win._replace(offset=win.offset+win.duration/2)
        win['offset']=win['offset']+win['duration']/2
        #win=win._replace(duration=win.duration/2)
        win['duration']=win['duration']/2
        sidAmpL.append(ds.sidAmp(sigWindow=spf.Window(**win)))
    #figure('Late')
    ax=gca()
    return spf.combine_angle_amps(sidAmpL,plot)

def bySetSplit(plot=False, ):
    sidAmpL=[]
    for ds in dsL:
        #ds=sd.SPDataSet(ts)
        mn, calcedUnc, apparentUnc, wEr=spf.combine_angle_amps(ds.sidAmp(sd.seqFiltFuncInterleave(Ndiv)))
        sidAmpFull=ds.sidAmp()
        sidAmpL.append(spf.SiderealFitData(t=sidAmpFull.t, sig=mn, err=diag(apparentUnc)**2, chi2=None, sidTheta=sidAmpFull.sidTheta, labTheta=sidAmpFull.labTheta))
    #figure('By set, split-up')
    ax=gca()
    return spf.combine_angle_amps(sidAmpL,plot)

def bgSub(plot=False, ):
    sidAmpL=[]
    for ds in dsL:
        #ds=sd.SPDataSet(ts)
        sidAmpL.append(ds.sidAmp(subtractWindow='bg'))
    #figure('BG subtracted')
    ax=gca()
    return spf.combine_angle_amps(sidAmpL,plot)

def bySet(plot=False, **kwargs):
    sidAmpL=[]
    for ds in dsL:
        #ds=sd.SPDataSet(ts)
        print("timestamp: {}".format(ds.timestamp))
        sAmp=ds.sidAmp(**kwargs)
        if hasattr(sAmp, 't'):
            sidAmpL.append()
        else: 
            sidAmpL.extend(sAmp)
        ds._cache.clear()
    #figure('By set')
    ax=gca()
    return spf.combine_angle_amps(sidAmpL,plot), sidAmpL

def dividedUp(plot=False, ):
    sidAmpL=[]
    for ds in dsL:
        #ds=sd.SPDataSet(ts)
        sidAmpL.extend(ds.sidAmp(sd.seqFiltFuncInterleave(Ndiv)))
    #figure('split up')
    ax=gca()
    return spf.combine_angle_amps(sidAmpL,plot)

def NS(plot=False, **kwargs):
    sidAmpL=[]
    for ds in dsL:
        #ds=sd.SPDataSet(ts)
        sidAmpL.append(ds.sidAmp(sd.seqFiltFuncAxis(0), label='NS Only', **kwargs))
    #figure('NS (by set)')
    ax=gca()
    return spf.combine_angle_amps(sidAmpL,plot)

def EW(plot=False, **kwargs):
    sidAmpL=[]
    for ds in dsL:
        #ds=sd.SPDataSet(ts)
        sidAmpL.append(ds.sidAmp(sd.seqFiltFuncAxis(pi/2), label='EW only', **kwargs))
    #figure('EW (by set)')
    ax=gca()
    return spf.combine_angle_amps(sidAmpL,plot)

def EWSplit(plot=False, **kwargs):
    sidAmpL=[]
    def filt(seqAmp):
        return sd.seqFiltFuncInterleave(Ndiv)(sd.seqFiltFuncAxis(pi/2)(seqAmp))
    for ds in dsL:
        try:
            mn, calcedUnc, apparentUnc, wEr=spf.combine_angle_amps(
                    ds.sidAmp(filtF=filt, **kwargs)
                    )
            sidAmpFull=ds.sidAmp()
            sidAmpL.append(spf.SiderealFitData(t=sidAmpFull.t, sig=mn, err=diag(apparentUnc)**2, chi2=None, sidTheta=sidAmpFull.sidTheta, labTheta=sidAmpFull.labTheta))
        except (ZeroDivisionError, ValueError):
            sys.exc_clear()
            #pass #Failed probably because there was no data, so we won't add it to the list
    #figure('EW (split)')
    return spf.combine_angle_amps(sidAmpL, gca())

def NSSplit(plot=False, **kwargs):
    sidAmpL=[]
    def filt(seqAmp):
        return sd.seqFiltFuncInterleave(Ndiv)(sd.seqFiltFuncAxis(0)(seqAmp))
    for ds in dsL:
        try:
            mn, calcedUnc, apparentUnc, wEr=spf.combine_angle_amps(
                    ds.sidAmp(filtF=filt, **kwargs)
                    )
            sidAmpFull=ds.sidAmp()
            sidAmpL.append(spf.SiderealFitData(t=sidAmpFull.t, sig=mn, err=diag(apparentUnc)**2, chi2=None, sidTheta=sidAmpFull.sidTheta, labTheta=sidAmpFull.labTheta))
        except (ZeroDivisionError, ValueError):
            sys.exc_clear()
            #pass #Failed probably because there was no data, so we won't add it to the list
    #figure('NS (split)')
    return spf.combine_angle_amps(sidAmpL, gca())

def NSEW(plot=False,):
    sidAmpL=[]
    for ds in dsL:
        #vprint(ts)
        #ds=sd.SPDataSet(ts)
        sidAmpL.append(ds.sidAmp(sd.seqFiltFunc2Axes(0)))
    #figure('NSEW (by set)')
    ax=gca()
    return spf.combine_angle_amps(sidAmpL,plot)

def NSEWSplit(plot=False, **kwargs):
    sidAmpL=[]
    def filt(seqAmp):
        return sd.seqFiltFuncInterleave(Ndiv)(sd.seqFiltFunc2Axes(0)(seqAmp))
    for ds in dsL:
        try:
            mn, calcedUnc, apparentUnc, wEr=spf.combine_angle_amps(
                    ds.sidAmp(filtF=filt, **kwargs)
                    )
            sidAmpFull=ds.sidAmp()
            sidAmpL.append(spf.SiderealFitData(t=sidAmpFull.t, sig=mn, err=diag(apparentUnc)**2, chi2=None, sidTheta=sidAmpFull.sidTheta, labTheta=sidAmpFull.labTheta))
        except (ZeroDivisionError, ValueError):
            sys.exc_clear()
            #pass #Failed probably because there was no data, so we won't add it to the list
    #figure('NSEW (split)')
    return spf.combine_angle_amps(sidAmpL, gca())

def Rot45Split(plot=False, **kwargs):
    sidAmpL=[]
    def filt(seqAmp):
        return sd.seqFiltFuncInterleave(Ndiv)(sd.seqFiltFunc2Axes(pi/4)(seqAmp))
    for ds in dsL:
        try:
            mn, calcedUnc, apparentUnc, wEr=spf.combine_angle_amps(
                    ds.sidAmp(filtF=filt, **kwargs)
                    )
            sidAmpFull=ds.sidAmp()
            sidAmpL.append(spf.SiderealFitData(t=sidAmpFull.t, sig=mn, err=diag(apparentUnc)**2, chi2=None, sidTheta=sidAmpFull.sidTheta, labTheta=sidAmpFull.labTheta))
        except (ValueError, ZeroDivisionError):
            sys.exc_clear()
            #pass #Failed probably because there was no data, so we won't add it to the list
    #figure('Rot45 (split)')
    return spf.combine_angle_amps(sidAmpL, gca())

def Rot45(plot=False, ):
    sidAmpL=[]
    for ds in dsL:#timestampL:
        sidAmpL.append(ds.sidAmp(sd.seqFiltFunc2Axes(pi/4)))
    #figure('rot45 (by set)')
    #ax=gca()
    return spf.combine_angle_amps(sidAmpL, plot)

systL=[
    ('set', bySet),
    ('setVertSub', partial(bySet, coFitL='med:Vert Pos Det 2')),
    ('day', byDay),
    ('dayVertSub', partial(byDay, coFitL='med:Vert Pos Det 2')),
    ('sets_S', bySetSplit),
    ('alldivided', dividedUp),
    ('bgSub', bgSub),
    ]
portionL=[
    ('axes0', NSEW),
    ('axes45', Rot45),
    ('EW', EW),
    ('NS', NS),
    ('polUp', polUp),
    ('polDn', polDn),
    ('setEarly', bySetEarly),
    ('setLate', bySetLate),
    ]
portionSplitL=[
    ('axes0_S', NSEWSplit),
    ('axes45_S', Rot45Split),
    ('EW_S', EWSplit),
    ('NS_S', NSSplit),
    ('polUp_S', polUpSplit),
    ('polDn_S', polDnSplit),
    #('setEarly', bySetEarlySplit),
    #('setLate', bySetLateSplit),
    ]

def directCompare():
    L=portionL
    for k in range(len(L)/2):
        name1, f1=L[2*k]
        name2, f2=L[2*k+1]
        fig=figure(name1 + ' vs ' + name2)
        ax1=subplot(211)
        ax2=subplot(212)
        f1(plot=[ax1,ax2])
        f2(plot=[ax1,ax2])
        fig.savefig(figdir+'/{} vs {}.pdf'.format(name1,name2))

def systComparison(L, axL=None):
    labels, dat=zip(*[(label, f()) for label, f in L])
    mnL, trustUncL, apparentUncL, wErL = zip(*dat)
    mnL=array(mnL);
    trustUncL=array(trustUncL);
    apparentUncL=array(apparentUncL)
    if axL==None:
        figure()
        ax1=gca()
        figure()
        ax2=gca()
        axL=[ax1, ax2]
    else:
        ax1,ax2=axL

    xpos=arange(len(mnL))
    k=0
    for mn, appUnc, trustUnc, ax in zip(mnL.T, apparentUncL.T, trustUncL.T, axL):
        ax.errorbar(xpos, mn, appUnc, fmt='o', elinewidth=2, capthick=2) 
        ax.errorbar(xpos, mn, trustUnc, fmt=None, elinewidth=2, capthick=2) 
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.set_xlim([-0.5, len(mnL)+0.5])
        ax.set_ylabel('fT')
    ax1.set_title('x-quad')
    ax2.set_title('y-quad')
    ax1.grid(True)
    ax2.grid(True)
    return labels, dat

def plotAllSyst():
    k=1
    for L in [systL, portionL, portionSplitL]: 
        fig=figure()
        ax1=subplot(211)
        ax2=subplot(212)
        systComparison(L,[ax1,ax2])
        fig.savefig(figdir+'/systematic {}.pdf'.format(k))
        k+=1

def plotAllSeq():
    for ds in dsL:
        ds.viewSeq()
        tight_layout()
        savefig(figdir+"/{0}:corr.png".format(ds.timestamp))

        ds.viewSeq(sigWindow='bg')
        tight_layout()
        savefig(figdir+"/{0}:corr_bg.png".format(ds.timestamp))

        ds.viewCorrelationFilt()
        tight_layout()
        savefig(figdir+"/{0}:corr_filt.png".format(ds.timestamp))
        ds.clearRaw()

def test():
    ds=sd.SPDataSet('5369.87')
    sAmp1=ds.sequenceAmp(sigWindow=spf.Window(2.5,-5.5))
    sAmp2=ds.sequenceAmp(sigWindow=spf.Window(2.5,-3))
    pAmp1=ds.pointAmp(sigWindow=spf.Window(2.5,-5.5))
    pAmp2=ds.pointAmp(sigWindow=spf.Window(2.5,-3))

    #sAmp1=ds.sequenceAmp(sigWindow=spf.Window(3.25,-6.5))
    #sAmp2=ds.sequenceAmp(sigWindow=spf.Window(3.25,-3.25))
    #pAmp1=ds.pointAmp(sigWindow=spf.Window(3.25,-6.5))
    #pAmp2=ds.pointAmp(sigWindow=spf.Window(3.25,-3.25))

    sig1=sAmp1.sig.ravel()
    sig2=sAmp2.sig.ravel()
    err1=sAmp1.err.ravel()
    err2=sAmp2.err.ravel()
    t=sAmp1.t.ravel()
    figure()
    fit, (ax1,ax2)=subplots(2,1, sharex=True)
    ax1.plot(pAmp1.t.ravel(), pAmp1.sig.ravel())
    ax1.plot(pAmp2.t.ravel(), pAmp2.sig.ravel())
    ax2.errorbar(t, sig1, err1, fmt='.')
    ax2.errorbar(t, sig2, err2, fmt='.')

