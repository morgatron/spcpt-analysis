"""

"""
import pdb
from collections import namedtuple
from functools import partial
from pylab import *
import SPfuncs as sp
from MT import nonnan, cached_property, memoize_inst_meth, nearby_angles, combine_angle_pairs, rotate_quadrature_sample, weightedStats, vprint
import MT, os
import loadSPFiles
from SPfuncs import getAnglesFromRSpec, renameFieldMaybe, getHarmonicsFit, mgStringHarmonics, sliceReduceData, sliceSorted, makeTriggers, Window, RotationSpec, secs2sd, sd2secs, makeTestData, spSearchSet
from SPfuncs import CorrelationData, SiderealFitData, PointData, RawData
import SPfuncs as spf
import loadSPFiles
from scipy import random
import inspect
import yaml
from copy import copy
from pprint import pprint

#This works with memoize_inst_methd methods only (and only with Morgan's modified version)
def divide_args_memo(kwargs, func_to_follow):
    names_to_keep=list(set(kwargs.keys()) & set(func_to_follow.orig_args[1:]))
    kept_args=dict( [(key, kwargs.pop(key)) for key in names_to_keep] )
    return kwargs, kept_args

def divide_args(cur_kwargs, work_func):
    """Helpful for calling work_func(following_func(**remaining_args), **work_args))
    work_args are the entries of cur_kwargs that are in the argument list of work_func. Remaining_args are those left over
    Currently just returns the divided argument list
    """
    names_to_keep=list(set(cur_kwargs.keys()) & set(inspect.getargspec(work_func).args))
    work_args=dict( [(key, cur_kwargs.pop(key)) for key in names_to_keep] )
    #return work_func(following_func(**cur_kwargs), **work_args)
    return cur_kwargs, work_args

ttl=300
class SPDataSet(object):
    setNameD={
              'med':'medD',
              'medium':'medD',
              'rel':'relD',
              'relaxed':'relD',
              'fast':'fastD',
              '':'fastD',
              'motor':'motorD',
              'mot':'motorD',
            }
    def __init__(self, timestamp, preloadDict={}, startTime=-inf, endTime=inf, **kwargs):
        self.windowD={}
        self.timestamp=timestamp    
        self.startTime=startTime
        self.endTime=endTime
        self.base_dir, self.base_name=loadSPFiles.getSpinningDataNames(timestamp)
        #Add to the class whatever is in preloadDict (these will overide the cached_property functions)
        self.__dict__.update(preloadDict)


        #Convenience parameter loading from set_info.yaml:
        if kwargs:
            # Just in case you didn't mean to do this, e.g. a typo
            vprint("Warning: updating dataset according to kwargs: {0}".format(kwargs))
            self.__dict__.update(kwargs)
        if self.set_info.has_key('windows'):
            winD=self.set_info['windows']
            
            for winName, winVals in winD.iteritems():
               self.windowD[winName]=Window(**winVals)
        if self.set_info.has_key('badTimes'):
            self.badTimes=self.set_info['badTimes']
        else:
            self.badTimes=[]
        if self.set_info.has_key('skipSeqIs'):
            self.skipSeqIs=self.set_info['skipSeqIs']
            if not hasattr(self.skipSeqIs, '__iter__'):
                self.skipSeqIs=[self.skipSeqIs]
        else:
            self.skipSeqIs=[]
        if self.set_info.has_key('dropTriggerIs'):
            self.dropTriggerIs=self.set_info['dropTriggerIs']
            if not hasattr(self.dropTriggerIs, '__iter__'):
                self.dropTriggerIs=[self.dropTriggerIs]
        else:
            self.dropTriggerIs=[]

        if self.set_info.has_key('calOverride'):
            self.calOverride=self.set_info['calOverride']
        else:
            self.calOverride=None
            
    def clearRaw(self):
        for varname in ['fastD', 'medD', 'relD', 'motorD']:
            self._cache.pop(varname, None)
        
    @property
    def rSpec(self):
        return (self.__dict__['rSpec'] if 
                self.__dict__.has_key('rSpec') else 
                RotationSpec(**self.set_info['rotationSpec'])
                )
        #return RotationSpec(**self.set_info['rotationSpec'])
    @cached_property(ttl=0)
    def set_info(self):
        """Property doc string!!"""
        if self.timestamp=='test':
            return {}
        return loadSPFiles.loadSetInfo(self.base_dir) 

    def update_set_info(self, newD):
        cur_info=copy(self.set_info)
        new_keys=newD.keys()
        old_keys=cur_info.keys()
        if not all([key in old_keys for key in new_keys]):
            raise ValueError("Some keys don't already exist. Edit the info file instead")
        cur_info.update(newD)
        loadSPFiles.writeSetInfo(self.base_dir, newD)

    def tRange(self):
        return [self.fastD[0].min(), self.fastD[0].max()]

    #Base level loading
    @staticmethod
    def splitRecArray(rA):
        t=rA['sd']
        names=list(rA.dtype.names)
        names.remove('sd')
        y=rA[names].view(rA.dtype[-1]).reshape(rA.size,-1)
        return [t.copy(), y.T.copy(), names]

    @cached_property(ttl=ttl)
    def medD(self):
        rec=loadSPFiles.loadBins('{0}{1} {2}'.format(self.base_dir, self.base_name, 'Medium'), useMetaFile=True) 
        if rec is not None:
            renameFieldMaybe(rec, 'Sidereal Days', 'sd')
            dat=self.splitRecArray(rec)
            startI, endI=dat[0].searchsorted([self.startTime, self.endTime])
            dat[0]=dat[0][startI:endI]
            dat[1]=dat[1][:,startI:endI]
        else:
            return None;
        return dat

    @cached_property(ttl=ttl)
    def motorPositions(self):
        base_dir,base_name=loadSPFiles.getSpinningDataNames(self.timestamp, bMotorData=True)
        rec=loadSPFiles.loadBins('{0}{1} {2}'.format(base_dir, base_name, 'positions'), useMetaFile=True) 
        if rec is not None:
            renameFieldMaybe(rec, 'Sidereal Days', 'sd')
            dat=self.splitRecArray(rec)
            startI, endI=dat[0].searchsorted([self.startTime, self.endTime])
            dat[0]=dat[0][startI:endI]
            dat[1]=dat[1][:,startI:endI]
        else:
            vprint("No motor position data")
            return None;
        return dat

    @cached_property(ttl=ttl)
    def motorSensors(self):
        base_dir,base_name=loadSPFiles.getSpinningDataNames(self.timestamp, bMotorData=True)
        rec=loadSPFiles.loadBins('{0}{1} {2}'.format(base_dir, base_name, 'sensors'), useMetaFile=True) 
        if rec is not None:
            renameFieldMaybe(rec, 'Sidereal Days', 'sd')
            dat=self.splitRecArray(rec)
            startI, endI=dat[0].searchsorted([self.startTime, self.endTime])
            dat[0]=dat[0][startI:endI]
            dat[1]=dat[1][:,startI:endI]
        else:
            vprint("No motor sensor data")
            return None;
        return dat

    @cached_property(ttl=ttl)
    def relD(self):
        rec=loadSPFiles.loadBins('{0}{1} {2}'.format(self.base_dir, self.base_name, 'Relaxed'), useMetaFile=True) 
        if rec is not None:
            renameFieldMaybe(rec, 'Sidereal Days', 'sd')
            dat=self.splitRecArray(rec)
            startI, endI=dat[0].searchsorted([self.startTime, self.endTime])
            dat[0]=dat[0][startI:endI]
            dat[1]=dat[1][:,startI:endI]
        else:
            return None;
        return dat

    @cached_property(ttl=ttl)
    def fastD(self):
        rec=loadSPFiles.loadBins('{0}{1} {2}'.format(self.base_dir, self.base_name, 'Fast'), totalDataType=[('sd','>f8'), ('s1', '>f4')]) 
        if rec is not None:
            dat=self.splitRecArray(rec)
            dat[1]=dat[1].squeeze()
            if self.calOverride is None:
                calArr=self.calArr
                dat[1]=dat[1]*MT.const_interp(dat[0], calArr.t, calArr.val)
            else:
                dat[1]=dat[1]*self.calOverride
            #dat[1]=self.cal*dat[1]
            startI, endI=dat[0].searchsorted([self.startTime, self.endTime])
            dat[0]=dat[0][startI:endI]
            dat[1]=dat[1][startI:endI]
        else:
            return None
        return dat

    @cached_property(ttl=ttl)
    def trigTimes(self):
        if self.fastD: #prefer to make triggers with the fast data if it's available
            t=self.rawData().t#fastD[0]
        elif self.medD:
            t=self.medD[0]
        elif self.relD:
            t=self.relD[0]
        else:
            return None
        trigs= makeTriggers(t, self.rSpec, bIncludeExtraRot=True);
        #if self.set_info.has_key('skipFirstTrigger') and self.set_info['skipFirstTrigger']==True:
        #    trigs=trigs[:,1:]
        gdTrigMask=ones(self.rSpec.numRotations, dtype=bool) 
        gdTrigMask[self.dropTriggerIs]=0
        gdSeqMask=ones(trigs.shape[0], dtype=bool); gdSeqMask[self.skipSeqIs]=0
        trigs=trigs[gdSeqMask]
        trigs=trigs[:,gdTrigMask]
        return trigs

    @cached_property(ttl=ttl)
    def apparatusRotations(self):
        gdTrigMask=ones(self.rSpec.numRotations, dtype=bool) 
        gdTrigMask[self.dropTriggerIs]=0
        return getRotationsFromRSpec(self.rSpec, self.trigTimes.shape[0])[:,gdTrigMask]/180.*pi
    @cached_property(ttl=0)
    def apparatusAngles(self):
        gdTrigMask=ones(self.rSpec.numRotations, dtype=bool) 
        gdTrigMask[self.dropTriggerIs]=0
        return getAnglesFromRSpec(self.rSpec, self.trigTimes.shape[0])[:,gdTrigMask]/180.*pi

    @cached_property(ttl=0)
    def zeroingD(self):
        zeroD= loadSPFiles.loadZeroFile('{0}{1} Zeroing List.asc'.format(self.base_dir, self.base_name))
        return zeroD
    
    def cBzCal(self):
        dat=loadtxt('{0}{1} Zeroing z1 Calibrations.asc'.format(self.base_dir, self.base_name)).T
        typ= namedtuple('z1CalData', ['t', 'val'])
        return typ(t= dat[0], val=dat[1])
        
    @property
    def calArr(self, process=None):
        calArr=self.zeroingD['Cal']
        if calArr.t.size>1 and process=='average':
            cal=(diff(calArr.t)*calArr.val[:-1]).sum()/diff(calArr.t).sum() 
            #cal=(diff(calArr.t)*calArr.val[:-1]).sum()/diff(calArr.t).sum() # A rough time-average. diff(t)*sum(vals)/sum(diff(t))
        elif process is None:
            if calArr.t.size>1:
                cal=calArr
            elif calArr.t.size==1:
                #cal=calArr.val[0]
                print("only one Bx callibration, maybe that's a problem")
                cal=calArr
            else:
                print("no Bx callibration found for timestamp {0}? Using coarse-bz calibration instead".format(self.timestamp))
                cal = self.cBzCal()
        #vprint("Callibration is {0}".format(cal))
        return cal

    def pol(self, t=None):
        polTup=self.zeroingD['Pol']
        tmeas=polTup.t
        if t is None:
            return tmeas, polTup.val
        else:
            pol=interp(t, tmeas, polTup.val)
            return pol

    def getSensorData(self, nameL):
        return [getRawByName(name) for name in nameL]
    def getRawByName(self, name):
        ''' Names should be of the form "med:Total Position Sensor 1" etc
        '''
        if name=='sig':
            return self.rawData()
        setSt, nameSt=name.split(':')
        data_set=self.__getattribute__(self.setNameD[setSt])
        if data_set is None:
            vprint("No data for {0}".format(name))
            return None
        nameL=data_set[2]

        if nameSt not in nameL:
            raise ValueError("Requested name '{0}' is not in the list of '{1}' sensors".format(nameSt, setSt))
        I=nameL.index(nameSt)
        return RawData(t=data_set[0], sig=data_set[1][I])
        
    def rawData(self, rawData=None, **kwargs):
        if isinstance(rawData, RawData):
            rawData=rawData
        elif rawData==None:
            rawData=RawData(self.fastD[0], self.fastD[1])
        elif isinstance(rawData, str): #Load a data set according to the string
            rawData=self.getRawByName(rawData)
        if rawData is None:
            return

        goodMask=ones(rawData.t.size, dtype=bool)
        if self.badTimes and not hasattr(self.badTimes[0], '__iter__'):
            self.badTimes=[self.badTimes]
        for interval in self.badTimes: 
            if interval[1]==-1:
                interval[1]=inf
            Istart, Istop=rawData.t.searchsorted(interval)
            goodMask[Istart:Istop]=False

        rawData= RawData(*[p[goodMask] for p in rawData])
        #.sig[~goodMask]
        self.clearRaw()
        return rawData
        #return RawData(t=rawData.t[goodMask], sig=rawData.sig[goodMask])

    #Processing 0
    @memoize_inst_meth
    def cutAmp(self, sigWindow=None, cutSensSubL=[], addFakeD=None, **kwargs):
        if sigWindow==None:
            if not self.windowD.has_key('sig'):
                raise ValueError("No signal window was given, and there's no default for this set (check set_info file)")
            sigWindow=self.windowD['sig'] # if no sigWindow is given, assume there's a default one for this dataset
        elif isinstance(sigWindow, basestring):
            sigWindow=self.windowD[sigWindow]
        vprint("Calculating cut amplitudes")
        cutAmp= spf.preprocess_raw(self.rawData(**kwargs), self.trigTimes, sigWindow)
        if cutSensSubL:
            sensCutAmpL=[self.cutAmp(rawData=name) for name in cutSensSubL]
        if addFakeD is not None:
            cutAmp=spf.addFakeData(cutAmp, self.apparatusAngles, self.trigTimes, **addFakeD)
        return cutAmp
        #return RawData(rawT, rawSig)

    @memoize_inst_meth
    def filteredCutAmp(self, sensNameL=None, **kwargs):
        if sensNameL is None:
            if self.set_info.has_key('sensorFilt'):
                sensNameL=self.set_info['sensorFilt']
            else:
                sensNameL=[]

        rem_args, work_args=divide_args(kwargs, spf.filterBySensors)
        D=copy(kwargs)
        if D.has_key('rawData'):
            D.pop('rawData')
        if D.has_key('sensNameL'):
            D.pop('sensNameL')
        cutSensorAmps=[self.pointAmp(rawData=name, **D) for name in sensNameL]
        return spf.filterBySensors(self.cutAmp(**rem_args), sensDataL=cutSensorAmps, **work_args)
    
    @memoize_inst_meth
    def seqAmpFitSplit(self, addFakeD=None, **kwargs):
        """Assume it's a continuously rotating sample.
        """
        rem_args, work_args=divide_args(kwargs, spf.process_continuous_raw)
        rawSig=self.rawData(**rem_args)
        if addFakeD is not None:
            rawSig=spf.addFakeData(rawSig, rotationRate=self.set_info['rotationRate'], **addFakeD)

        rawRef=self.rawData('med:Fluxgate Z')
        if self.set_info.has_key('rotationRate'):
            rotationRate=float(self.set_info['rotationRate'])
        else:
            print("WARNING:No ration rate in set_info. We'll guess it's -0.03, but if it's not then the answers will be wrong!!")
            rotationRate=-0.03
        seqAmp=spf.process_continuous_raw(rawSig, rawRef, rotationRate) 
        return seqAmp

    #Processing 1
    @memoize_inst_meth
    def pointAmp(self, **kwargs):
        vprint("Calculating point amplitudes")
        if self.set_info.has_key('continuousRotation') and self.set_info['continuousRotation']==True:
            Nds=250
            rAmp=self.rawData(**kwargs)
            ptVals, ptErr= MT.downsample_npts(rAmp.sig, Nds, bRetErr=True)
            ptT=MT.downsample_npts(rAmp.t, Nds)
            return PointData(t=ptT, sig=ptVals,err=ptErr/sqrt(Nds), theta=None, chi2=None)
            #return PointData(t=zeros(0), sig=zeros(0),err=zeros(0), theta=zeros(0), chi2=zeros(0))
        rem_args, work_args=divide_args(kwargs, spf.process_raw)
        #pointAmp= spf.process_raw(self.filteredCutAmp(**rem_args), **work_args)
        pointAmp= spf.process_raw(self.cutAmp(**rem_args), **work_args)
        pointAmp=pointAmp._replace(theta=self.apparatusAngles)
        return pointAmp

    #Processing 2
    @memoize_inst_meth
    def sequenceAmp(self, bRemoveZerothRotation=True, subtractWindow=None,genSeqFiltF=None, seqSensSubL=[], **kwargs):
        vprint("Calculating sequence amplitudes")
        if self.set_info.has_key('continuousRotation') and self.set_info['continuousRotation']==True:
            seqAmp=self.seqAmpFitSplit(**kwargs)
        else: #stop-start manner
            rem_args, work_args=divide_args(kwargs, spf.process_points)
            pointAmp=self.pointAmp(**rem_args)
            cutAmp=self.filteredCutAmp(**rem_args)

            if bRemoveZerothRotation:
                pointAmp=PointData(*[arr.take(r_[1:arr.shape[-1]], axis=-1) if arr is not None else None for arr in pointAmp])
                cutAmp=RawData(*[arr.take(r_[1:arr.shape[-1]], axis=-1) if arr is not None else None for arr in cutAmp])

            seqAmp=spf.process_points(self.apparatusAngles, pointAmp, cutAmp, **work_args)
            if subtractWindow:
                vprint('subtracting window "{}"'.format(subtractWindow))
                D=copy(rem_args)
                D.update({'sigWindow':subtractWindow})
                refAmp=self.sequenceAmp(**D)
                seqAmp.sig[:]-=refAmp.sig

        if seqSensSubL:
            sensSeqAmpL=[self.sequenceAmp(rawData=name) for name in seqSensSubL]
            sensSeqAmpL=[CorrelationData(*[par[:,0,1] for par in sAmp]) for sAmp in sensSeqAmpL]
            sub1HAmp=spf.subtract_correlations(CorrelationData(*[par[:,0,1] for par in seqAmp]), sensSeqAmpL)
            seqAmp.sig[:,0,1]=sub1HAmp.sig

        if genSeqFiltF: #General filter on the sequences
            seqAmp=genSeqFiltF(self, seqAmp)


        return seqAmp

    #Processing 3
    @memoize_inst_meth
    def labAmp(self, filtF=None, **kwargs):
        vprint("Calculating lab-frame amplitudes")
        rem_args, work_args=divide_args(kwargs, spf.process_sequences)
        seqAmp=self.sequenceAmp(**rem_args)
        if filtF:
            seqAmp=filtF(seqAmp)
        labAmp, sidAmp= spf.process_sequences(seqAmp)
        return labAmp
        #return spf.split_and_process_sequences(seqAmp, **work_args)

    #Processing 3
    @memoize_inst_meth
    def sidAmp(self, filtF=None, coFitL=[], coFitPhaseVaryIs=[], subtractHarmsL=[], harmonic=1, **kwargs):
        h=harmonic-1
        vprint("Calculating sidereal amplitudes")
        rem_args, work_args=divide_args(kwargs, spf.process_sequences_multifit)
        seqAmp=self.sequenceAmp(**rem_args)

        #This will be used for the actual processing
        #seqAmp, sensSeqAmpL,
        sensSeqAmpL=[self.sequenceAmp(rawData=name, **rem_args) for name in coFitL]

        #Apply the filtering function
        if filtF: #Use some sub-set of the sequences
            seqAmp=filtF(seqAmp)
            if not seqAmp:
                return None
            sensSeqAmpL=zip(*[filtF(sensAmp) for sensAmp in sensSeqAmpL])
            if not sensSeqAmpL:
                sensSeqAmpL=len(seqAmp)*[[]] #Make it a list of empty lists

        # Now seqAmp and sensSeqAmpL may be lists (or lists of lists)
        # But if they're not, we'll make sure they are
        if hasattr(seqAmp, 't'):
            seqAmp=[seqAmp]
            if sensSeqAmpL==[]:
                sensSeqAmpL=[sensSeqAmpL]
            
            

        if kwargs.has_key('bFitOldWay'):
            work_func=partial(spf.process_sequences)
            labAmp, sidAmp= zip(*[work_func(seqA) for seqA in seqAmp if seqA is not None])
        elif seqAmp:
            work_func=partial(spf.process_sequences_multifit,  
                    sigSensVaryPhaseIs=coFitPhaseVaryIs, 
                    subtractHarmsL=subtractHarmsL,harmonic=harmonic, 
                    **work_args)
            labAmp, sidAmp, _,fObj= zip(*[work_func(seqA, sensAL)[:4] 
                    for seqA, sensAL in zip(seqAmp, sensSeqAmpL) 
                    if seqA is not None])
        else:
            print("Nothing at timestamp {0}".format(self.timestamp))
            return None
        if len(sidAmp)==1:
            labAmp=labAmp[0]
            sidAmp=sidAmp[0]
            fObj=fObj[0]

        #return labAmp, sidAmp#, fObj
        return sidAmp#, sidAmp
        
        #return spf.split_and_process_sequences(seqAmp, **work_args)

    def viewSensors(self, sensorNames, bSharex=False, bPlotStChi2Comp=True, **kwargs):
        if sensorNames=='med':
            sensorNames=['med:'+ name for name in loadSPFiles.mediumNames]
        elif sensorNames=='rel':
            sensorNames=['rel:'+ name for name in loadSPFiles.relaxedNames]
        elif sensorNames=='all':
            sensorNames=['rel:'+ name for name in loadSPFiles.relaxedNames]
            sensorNames.extend(['med:'+ name for name in loadSPFiles.mediumNames])
        self.viewSeq(rawData=sensorNames[0], figName=sensorNames[0], stPtChi2Lim=5, **kwargs)
        ax=gca()
        figL=[gcf()]
        if bSharex==True:
            sharex= ax 
        else:        
            sharex=None;
        vprint("sharex: {}".format(sharex))
        for name in sensorNames[1:]:
            self.viewSeq(rawData=name, figName=name, sharexAx=sharex, **kwargs)
            figL.append(gcf())
        return figL

    def viewRaw(self, figName="", **kwargs):
        if kwargs.has_key('sigWindow'):
            cutAmpL=[self.cutAmp(**kwargs)]
            windowNames=['sig']
        else:
            windowNames, cutAmpL=zip(*[(key, self.cutAmp(sigWindow= win, **kwargs)) for key, win in self.windowD.items()])
        spf.view_raw(self.rawData(**kwargs), cutAmpL, self.trigTimes, cutAmpLNames=windowNames, figName=self.timestamp+'- '+figName) 

    def viewSeq(self, figName="", sharexAx=None, bPlotStChi2Comp=True, **kwargs):
        if kwargs.has_key('sigWindow'):# and isinstance(kwargs['sigWindow'], basestring):
            figName +='- ({})'.format(kwargs['sigWindow'])
        correlationSigComp=None
        if bPlotStChi2Comp is not None:
            d=copy(kwargs); d['stPtChi2Lim']=inf;
            correlationSigComp=self.sequenceAmp(**d)
        spf.view_correlations(self.pointAmp(**kwargs), self.sequenceAmp(**kwargs), self.timestamp+'- '+figName, sharexAx=sharexAx, correlationSigComp=correlationSigComp.sig)

    def viewSid(self, **kwargs):
        #spf.view_sidereal(self.sequenceAmp(**kwargs), self.sidAmp(**kwargs))
        sidAmp=self.sidAmp(**kwargs)
        seqAmp=self.sequenceAmp(**kwargs)
        labAmp=self.labAmp(**kwargs)
        spf.view_sidereal(seqAmp, sidAmp, labAmp)

    def viewCorrelationFilt(self, **kwargs):
       D=copy(kwargs)
       D.update(dict(sensNameL=[], stPtChi2Lim=inf))
       pointAmpBase=self.pointAmp(**D)
       seqAmpBase=self.sequenceAmp(**D)

       pointAmpFilt=self.pointAmp(**kwargs)
       seqAmpFilt=self.sequenceAmp(**kwargs)
       spf.view_correlation_filtering(pointAmpBase, seqAmpBase,
            pointAmpFilt, seqAmpFilt)

    def edit_set_info(self):
        MT.open_file_external(os.path.join(self.base_dir, 'set_info.yaml'))
    @memoize_inst_meth
    def test_div(self, **kwargs):
        kwargs, kept_args=divide_args(kwargs, self.pointAmp)
        return kwargs, kept_args

    def checkDirections(self, bPlot=False, **kwargs):
        """Check that the calculated apparatus directions fit with the available data
        
        To do this we'll check the North position sensor, the motor trigger-log, and the Z-axis flux-gate, if available
        """
        #Flux-gates
        expTh, expSig = [0, pi/2], [ 0.018, -0.045]
        if self.set_info.has_key('bFieldZeroed') and self.set_info['bFieldZeroed'] is False:
            expTh, expSig = [0, pi/2], [ 0.02, -0.08]
        zCor=self.sequenceAmp(rawData='med:Fluxgate Z', **kwargs)
        actSig=zCor.sig[:,0,:]
        actTh=zCor.theta[:,0,:]
        expRot=array([MT.rotate_quadrature_sample(expTh, expSig, cov=None, rotateTo=-th[0])[1] for th in actTh])
        I=~isnan(actSig)
        gSig=actSig[I]
        gExp=expRot[I]

        bProblem=False
        if any(abs(gExp-gSig)>0.01):
            vprint("Fluxgate Z show's problems! (timestamp: {0})".format(self.timestamp))
            bProblem=True

        if bProblem or bPlot:
            figure(self.timestamp+'- direction check')
            gT=zCor.t[:,0,:][I]
            plot(gT, gSig, '.', label='Measured')
            plot(gT, gExp,'.', label='Expected')
            plot(gT, gExp-gSig,'.', label='difference')
            xlabel('Sid. Days')
            ylabel('V')
            legend(fontsize=10)


        # North-positon sensor
        try:
            if self.rawData().t[0] < 5200: #(Or whenever the sidereal day the sensor was actually switched?)

                npsDat=self.rawData(rawData='med:North Position Sensor') #(or whatever it actually was)
            else:
                npsDat=self.rawData(rawData='motor:North Position Sensor') #(or whatever it actually was)
        except (OSError, ValueError):
            npsDat=None

        if npsDat is not None:
            tNorthSens=npsDat.t[npsDat.sig<4]

            ang=(self.apparatusAngles()-pi)%(2*pi) + pi #Move everything to the range (-pi, +pi)
            tRot=self.trigTimes()
            tBound=c_[tRot[:-1], tRot[:-1]+self.rSpec.interval].ravel()
            angBound=c_[ang[:-1], ang[1:]].ravel()

            apAngleIntp=interp(npsDat.t, tBound, angBound) #Interpolate to the same times as the north-sensor is had

            figure('Check North-position sensor:')
            plot(npsDat.t, apAngleIntp)
            plot(npsDat.t, npsDat.sig)
        else:
            vprint("No north-position sensor data for checking")
        
        
        # Motor trigger-log

        return
#------------------------------------------------------------------------------

## Processing functions
def genSeqFiltPol(sgn): 
    def genSeqFilt(ds, seqAmp):
        ts=seqAmp.t[:,0,0]
        calArr=ds.calArr
        cal=MT.const_interp(ts, calArr.t, calArr.val)
        seqAmp.sig[sign(cal)!=sgn]*=nan
        return seqAmp
    return genSeqFilt


def seqFiltFuncAxisInterleave(axisAngle, Ndiv):
    def filt(seqAmp):
        return seqFiltFuncInterleave(Ndiv)((seqFiltFuncAxis(sgn)(seqAmp)))
    return filt
#genSeqFiltPol(1)(seqAmp)

def seqFiltFuncAxis(axisAngle):
    def filt(seqAmp):
        mask= (nearby_angles(seqAmp.theta[:,0,0], axisAngle, 0.1) |
              nearby_angles(seqAmp.theta[:,0,0], axisAngle+pi,0.1) 
              )
        return CorrelationData(*[s[mask] for s in seqAmp])
    return filt

def seqFiltFunc2Axes(axisAngle):
    def filt(seqAmp):
        mask= (nearby_angles(seqAmp.theta[:,0,0], axisAngle, 0.1) | 
              nearby_angles(seqAmp.theta[:,0,0], axisAngle+pi,0.1) |
              nearby_angles(seqAmp.theta[:,0,0], axisAngle+pi/2,0.1) |
              nearby_angles(seqAmp.theta[:,0,0], axisAngle+3*pi/2,0.1) 
              )
        return CorrelationData(*[s[mask] for s in seqAmp])
    return filt
def seqFiltFuncAngle(angle):
    def filt(seqAmp):
        mask= nearby_angles(seqAmp.theta[:,0,0], angle, 0.1)
        return CorrelationData(*[s[mask] for s in seqAmp])
    return filt

def seqFiltFuncSlc(slc):
    def filt(seqAmp):
        return CorrelationData(*[s[slc] for s in seqAmp])
    return filt


def seqFiltFuncInterleave(Ndiv):
    #slcL=[slice(k,None, Ndiv) for k in range(Ndiv)]
    def filt(seqAmp):
        Npts=seqAmp.t.shape[0]
        if Npts < 5*Ndiv:
            N = int(seqAmp.t.shape[0]/5)
        else:
            N=Ndiv
        Npts=floor(Npts/Ndiv)*Ndiv
        return [CorrelationData(*[s[k-Npts::N] for s in seqAmp]) for k in range(N)]
    return filt

def seqFiltFuncInterleaveGenerator(otherFiltF, Ndiv=10):
    #slcL=[slice(k,None, Ndiv) for k in range(Ndiv)]
    def filt(seqAmp, k):
        Npts=seqAmp.t.shape[0]
        if Npts < 5*Ndiv:
            N = int(seqAmp.t.shape[0]/5)
        else:
            N=Ndiv
        Npts=floor(Npts/Ndiv)*Ndiv
        if k>=N:
            return None
        return otherFiltF(CorrelationData(*[s[k-Npts::N] for s in seqAmp]))

    for k in range(Ndiv):
        #f=partial(filt, k=N)
        yield partial(filt, k=k)

def seqFiltFuncTime(Tdiv=1): #Divid it up into small chunks of time and process each
    def filt(seqAmp):
        totalTime=seqAmp.t[-1,0,0]-seqAmp.t[0,0,0]
        t0=seqAmp.t[0,0,0]
        N=floor(totalTime/Tdiv)
        startTimes=arange(N)*Tdiv + t0
        indxL=sliceSorted(seqAmp.t[:,0,0], startTimes, delta_t=Tdiv)
        return [CorrelationData(*[s[indx] for s in seqAmp]) for indx in indxL]
    return filt

def checkDSUnc(ds, Ndiv=10, **kwargs):
        #Process every Nth sequence:
        #fL=[seqFiltFuncSlc(slice(k,None, Ndiv)) for k in range(Ndiv)]
        #sidAmpL=[ds.sidAmp(filtF=f) for f in fL]
        sidAmpL=ds.sidAmp(seqFiltFuncInterleave(Ndiv), **kwargs)
        #sidAmpL=ds.sidAmp(filtF=seqFiltFuncTime(Tdiv=0.3))
        #wtMn, calcedUnc, apparentUnc, wtErr= spf.combine_angle_amps(sidAmpL, plot=True)
        combinedSid= spf.combine_angle_amps(sidAmpL, plot=True)

        #Probably don't need to rotate in this case, but just in case

        sidAmpFull=ds.sidAmp(**kwargs)
        print("Number of sets: {0}".format(len(sidAmpL)))
        pprint(combinedSid)
        pprint(sidAmpFull)
        #print("Subset deviation: {}, combined Mean:{}\n, combined unc (trust/don't trust): {}, {},\n    full mean:{}, full uncert {}".format(wtErr, wtMn,
                #calcedUnc, apparentUnc,  
                #sidAmpFull.sig, sqrt(sidAmpFull.err.diagonal())) )

        show()

def testDS(rSpec=None, rotFact=1, **kwargs):
    defaultD=dict(amp=0.05, sigma=2, amp2=0.00, 
        N=10000000, phi=0*pi, 
        zeroingTime=30, fracDev=0.00, 
        sizeDev=50.0, startTime=0)
    defaultD.update(kwargs)
    rsBase=RotationSpec(startingAngle=0, delay=9, interval=10, numRotations=20, rotAngles=[rotFact*180], extraRotAngle=rotFact*90)
    rSpec= rSpec if rSpec is not None else rsBase
    sd,sig=makeTestData(
        rSpec, **defaultD)
    ds=SPDataSet('test', preloadDict={'fastD': [sd, sig, 'sig']}, rSpec=rSpec, windowD={'sig': Window(5, -6)} )
    return ds

def testDS2(rSpec=None, rotDir=1, **kwargs):
    defaultD=dict(amp=0.5, sigma=1.0, amp2=0.0, 
        N=1000000, phi=0.25*pi, 
        zeroingTime=30, fracDev=0.00, 
        sizeDev=00.0, startTime=0)
    defaultD.update(kwargs)
    rsBase=RotationSpec(startingAngle=0, delay=9, interval=10, numRotations=8, rotAngles=[rotDir*90], extraRotAngle=rotDir*45)
    rSpec= rSpec if rSpec is not None else rsBase
    sd,sig=makeTestData(
        rSpec, **defaultD)
    ds=SPDataSet('test', preloadDict={'fastD': [sd, sig, 'sig']}, rSpec=rSpec, windowD={'sig': Window(5, -6)} )
    return ds

if __name__=="__main__":
    d=dict(sigWindow=Window(4,-5))
    if 0:
        sidAmpL=[]
        if 0: # Check theoretical
            rsBase=RotationSpec(startingAngle=0, delay=9, interval=10, numRotations=20, rotAngles=[-180], extraRotAngle=-90)
            rsL=[rsBase._replace(startingAngle=th) for th in hstack([zeros(15), -ones(0)*90, ones(0)*45])]
            for rs, startTime in zip(rsL, 0.2*arange(len(rsL))):
                sd,sig=makeTestData(
                    rs, amp=1.00, sigma=2, N=400000, phi=0*pi, zeroingTime=30, fracDev=0.05, sizeDev=50.0, startTime=startTime*1)

                ds=SPDataSet('test', preloadDict={'fastD': [sd, sig, 'sig']}, rSpec=rs )
                sAmp=ds.sidAmp(sigWindow=Window(4,-5))
                vprint(sAmp)
                sidAmpL.append(sAmp)
        elif 0: #Check slicing
            ds=testDS()
            checkDSUnc(ds, 30)
            #sAmp=ds.sidAmp(sigWindow=Window(4,-5))
            #vprint sAmp

             
            #EW-NS
            #sidAmpAngsL=[ds.sidAmp(sigWindow=Window(4,-5), filtF=seqFiltFuncAxis(ang)) for ang in (0, pi/2)]
            #sidAmpL=sidAmpSlcsL


        



        else: #Check actual data
            timestampL=[
                    #'5209.26',
                    #'5240.33',
                    '5249.72',
                    '5273.74',
                    '5282.23',
                    '5292.71',
                    '5294.31',
                    #'5296.33',
                    '5310.41',
                    '5318.41',
                    '5348.82',
                    '5369.87',
                    '5386.02',
                    '5390.74',
                ]
            for ts in timestampL:
                #try:
                    ds=SPDataSet(ts)#, preloadDict={'fastD': [sd, sig, 'sig']}, rSpec=rSpec )
                    ds.checkDirections()
                    show()
                    sAmp=ds.sidAmp()
                    vprint(sAmp)
                    sidAmpL.append(sAmp)
                #except Exception as (errno, strerror):
                #    vprint "Exception({0}): {1}".format(errno, strerror)
                #    vprint("For timestamp: {0}".format(ts))
                    #pass; #Print the error and say which set it was, but keep going.
                
        ts = [ sAmp.t for sAmp in sidAmpL]
        (ths, ys, covs)= zip(*[ rotate_quadrature_sample(sAmp.labTheta, sAmp.sig, sAmp.err) for sAmp in sidAmpL])
        #ys, covs= [ ]

        ys=array(ys)
        errs=array([sqrt(cov.diagonal()) for cov in covs])
        ts=array(ts)
        #errs=[diag(cov) for cov in covs]
        figure()
        for t, y, e in zip(ts.T, ys.T, errs.T):
            errorbar( t, y, e, fmt='.') #Probaby not quite right

        mn, wEr, unc= weightedStats(ys, 1./errs**2, axis=0)
        vprint("Subset deviation: {0}, (average errorbar):{1},\nmean: {2},, (from subsets): {3},  final uncert: (full){4}".format(er, errs.mean(axis=0), mn, unc, None) )
        show()

    
        
        #seqAxNSFilt= seqFiltFuncAxis(0);
        #seqAxEWFilt= seqFiltFuncAxis(pi/2);
    if 0: #Do idiot checks on experiment data
        timestampL=[
                #'5310.41',
                '5318.42',
                #'5348.82',
                #'5369.87',
                #'5386.02',
                #'5390.74',
                #'5393.04',
            ]
        
        for ts in timestampL:
            ds=SPDataSet(ts)
            ds.checkDirections(bPlot=True)
            show()
            ds.edit_set_info()
            raw_input('enter to continue')
        #ds.checkDirections(**d)
        #SPDataSet('5386.02').checkDirections(**d)
        #SPDataSet('5310.41').checkDirections(**d)
        #SPDataSet('5348.82').checkDirections(**d)
        #SPDataSet('5369.87').checkDirections(**d)
    if 0:
        for ts in loadSPFiles.getAllTimeStamps():
            if float(ts)>5310:
                ds=SPDataSet(ts)
                if ds.set_info['bUseful']=='unknown' or 1:
                    st=''
                    while st!='c':
                        vprint("Current timestamp is: {0}".format(ts))
                        ds.edit_set_info()
                        #ds.viewRaw()
                        ds.viewSeq()
                        ds.viewSeq(rawData='rel:Tilt Sensor 1')
                        ds.checkDirections()
                        #ds.viewSeq()
                        #try:
                        #    ds.checkDirections()
                        #except Exception as ex:
                        #    vprint ("Exception occured: {0}".format(ex))
                        show()
                        st=raw_input('Enter for again, c+enter to continue to next')

