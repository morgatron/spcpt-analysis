"""Functions for loading data files from the SP experiment

"""
import numpy as np
from os import path
import os
import re
from collections import namedtuple
import yaml
import MT
import convert

relaxedNames="""Tot Pos Det
Tot Pos Det 2
Tot Pos Det 3
Cell Temp
Ferrite Temp
Oven Temp
Temperature Controller Output
Breadboard Temp
Room Temp
G10 Lid Temp
Tilt Sensor 1
Tilt Sensor 2
WavelengthErr
Pump Current Mon
Pump Temp Mon
Probe Current Mon
Probe Temp Mon
Water Flow
Apparatus Position
""".splitlines()
mediumNames="""Accelerometer
Position Purple
Position Green
Fluxgate X
Fluxgate Y
Fluxgate Z
Tiltmeter X
Tiltmeter Y
Vert Pos Det
Horiz Pos Det
Vert Pos Det 2
Horiz Pos Det 2
Vert Pos Det 3
Horiz Pos Det 3
""".splitlines()

def getSpinningDataNames(timestamp, bMotorData=False):
    if not bMotorData:
        base_name='SP_aparatus {0}'.format(timestamp)
    else:
        base_name='SP_motor {0}'.format(timestamp)
    base_dir=os.path.join(os.getenv('SP_DATA_DIR'), base_name + ' Bundle/')
    return base_dir, base_name

def getAllTimeStamps(bApparatus=True, bMotor=False):
    data_dir=os.path.join(os.getenv('SP_DATA_DIR'))
    ap_dir_regex=re.compile('SP_aparatus (\d+(?:\.\d*)?|\.\d+) Bundle$')
    mo_dir_regex=re.compile('SP_motor (\d+(?:\.\d*)?|\.\d+) Bundle$')

    regExL=[]
    if bApparatus:
        regExL.append(ap_dir_regex)
    if bMotor:
        regExL.append(mo_dir_regex)
    outL=[]
    for reg in regExL:
        match_list=[ap_dir_regex.match(line) for line in os.listdir(data_dir)]
        stamps=[m.groups()[0] for m in match_list if m is not None]
        stamps.sort()
        outL.append(stamps)
    if len(outL)==1:
        outL=outL[0]
    return outL

#old
def loadSet(timestamp, cal=None, subL=["Fast", "Medium", "Relaxed"], startTime=-np.inf, endTime=np.inf):
    """OLD Load all the data associated with @timestamp
    """
    if cal is None:
        print("Haven't implemented loading of calibration data yet, Your data will stay in Volts")
        cal=1
    base_dir, base_name=getSpinningDataNames(timestamp)

    def splitRecArray(rA):
        t=rA['sd']
        names=list(rA.dtype.names)
        names.remove('sd')
        y=rA[names].view(rA.dtype[-1]).reshape(rA.size,-1)
        return [t, y.T, names]

    #startI=0
    fastD=None;
    medD=None; 
    relD=None;
    if "Fast" in subL:
        fastRec=loadSPBinary.loadBins('{0}{1} {2}'.format(base_dir, base_name, 'Fast'), totalDataType=[('sd','>f8'), ('s1', '>f4')]) 
        if fastRec is not None:
            fastD=splitRecArray(fastRec)
            fastD[1]=fastD[1].squeeze()
            fastD[1]*=cal

            startI, endI=fastD[0].searchsorted([startTime, endTime])
            fastD[0]=fastD[0][startI:endI]
            fastD[1]=fastD[1][startI:endI]

    if "Medium" in subL:
        medRec=loadSPBinary.loadBins('{0}{1} {2}'.format(base_dir, base_name, 'Medium'), useMetaFile=True) 
        if medRec is not None:
            renameFieldMaybe(medRec, 'Sidereal Days', 'sd')
            medD=splitRecArray(medRec)
            startI, endI=medD[0].searchsorted([startTime, endTime])
            medD[0]=medD[0][startI:endI]
            medD[1]=medD[1][:,startI:endI]



    if "Relaxed" in subL:
        relRec=loadSPBinary.loadBins('{0}{1} {2}'.format(base_dir, base_name, 'Relaxed'), useMetaFile=True) 
        if relRec is not None:
            renameFieldMaybe(relRec, 'Sidereal Days', 'sd')
            relD=splitRecArray(relRec)
            startI, endI=relD[0].searchsorted([startTime, endTime])
            relD[0]=relD[0][startI:endI]
            relD[1]=relD[1][:,startI:endI]
    if "Slow" in subL:
        slRec=loadSPBinary.loadBins('{0}{1} {2}'.format(base_dir, base_name, 'Slow'), useMetaFile=True) 
        if slRec is not None:
            renameFieldMaybe(slRec, 'Sidereal Days', 'sd')
            slD=splitRecArray(slRec)

    return SPDataSet(fastD, medD=medD, relD=relD)

set_info_defaults={
            'bUseful': 'unknown',
            'bDataConfirmed':False,
            'acquisition':{'sampleRate': 50, 'timeConstant': 0.003},
            #'rotationSpec':{'delay': 9, 
            #                'interval':30, 
            #                'numRotations':20,
            #                'rotAngles': [-180],
            #                'extraRotAngle': -90.,
            #                'startingAngle': 0,
            #                },
            'timeStamp':'unknown',
            'notes': 'Default',
            #'windows':{}
            'windows':{
                'sig':{
                    'duration': 4,
                    'offset': -6.6,
                    },
                'bg':{
                    'duration': 1,
                    'offset': 0,
                    },
                    }
            }
def loadSetInfo(dir_name):
    ts=dir_name.split()[-2]
    bWriteNew=False
    filepath=os.path.join(dir_name, 'set_info.yaml')
    try:
        f=open(filepath)
        infoD=yaml.load(f)
        f.close()
    except IOError:
        print("No set_info.yaml exists in {}, making defaults".format(dir_name))
        bWriteNew=True
        infoD= {}

    loadedKeys=infoD.keys()
    if 'bandwidth' in loadedKeys: #It's a matlab generated yaml file
        print("updating matlab generated yaml")
        #newD=copy(set_info_defaults)
        newD={
            'bUseful':infoD['bUseful'],
            'bDataConfirmed':infoD['bDataConfirmed'],
            'acquisition':{'sampleRate': infoD['bandwidth'][1], 
                            'timeConst': infoD['bandwidth'][0]
                         },
            'rotationSpec':{'delay': infoD['delay'], 'interval':infoD['interval'],
                            'numRotations':infoD['numRotations'], 
                            'rotAngles': [infoD['rotationAngles'][0]], 
                            'extraRotAngle':infoD['rotationAngles'][1],
                            'startingAngle':MT.cardinalToDeg(infoD['startPosition']),
                            },
            'timeStamp':infoD['timestamp'],
            'notes':infoD['notes']+'\nConverted from matlab',
            }
        infoD=newD
        writeSetInfo(dir_name, infoD)
    
    if infoD.has_key('rotationSpec') and infoD['rotationSpec'].has_key('startPosition'):
        infoD['rotationSpec']['startingAngle']= MT.cardinalToDeg(infoD['rotationSpec']['startPosition'])
        infoD['rotationSpec'].pop('startPosition')

    newLoadedKeys=infoD.keys()
    for key in set_info_defaults.keys():
        if key not in newLoadedKeys:
            infoD[key]=set_info_defaults[key]
            print("Loading {0} from defaults, it may not be right. Go fix this in set_info.yaml!".format(key))

    if bWriteNew:
        infoD['timeStamp']=ts
        writeSetInfo(dir_name, infoD)
    return infoD

def writeSetInfo(dir_name, D):
        filepath=os.path.join(dir_name, 'set_info.yaml')
        print("writing new set_info.yaml file...")
        fout=open(filepath,'w')
        fout.write(yaml.dump(D))
        fout.close()
        print("{0} was written.".format(filepath))

def loadZeroFile(fname):
    proc_dict={
            1: 'cBz',
            2: 'By',
            3: 'Bx',
            35: 'Cal',
            29: 'Pol',
    }
    dat=np.loadtxt(fname).T;
    t=dat[0]
    proc_num=dat[1]
    val=dat[2]
    slope=dat[3]
    sigma=dat[4]
    ZeroProc=namedtuple('ZeroProc', ['t', 'val', 'slope', 'sigma'])
    zD={}
    for num, name in proc_dict.iteritems():
        I=np.where(proc_num==float(num))[0]
        zD[name]=ZeroProc(t=t[I], val=val[I], slope=slope[I], sigma=sigma[I])
        #DataFrame =namedtuple('DataFrame', ['sd', 'y'])
    return zD


def getAllSPSensor(names, slc=None, timeRange=None):
    data_dir=os.path.join(os.getenv('SP_DATA_DIR'))
    ap_dir_regex=re.compile('SP_aparatus (\d+(?:\.\d*)?|\.\d+) Bundle$')
    mo_dir_regex=re.compile('SP_motor (\d+(?:\.\d*)?|\.\d+) Bundle$')

    ap_match_list=[ap_dir_regex.match(line) for line in os.listdir(data_dir)]
    ap_stamps=[m.groups()[0] for m in ap_match_list if m is not None]
    mo_match_list=[mo_dir_regex.match(line) for line in os.listdir(data_dir)]
    mo_stamps=[m.groups()[0] for m in mo_match_list if m is not None]

    mo_stamps.sort()
    ap_stamps.sort()
    ioff()
    sdL=[]
    if slc is None:
        [firstI, lastI]=searchsorted(ap_stamps, timeRange)
        slc=slice(firstI, lastI)
    #slc=slice(-20,-15)
    relIdxs=[relaxedNames.index(name) for name in names if name in relaxedNames]
    medIdxs=[mediumNames.index(name) for name in names if name in mediumNames]
    #slIdxs=[slowNames.index(name) for name in names if name is in slowNames]
    #medList=[idx for name in names if name is in mediumNames]
    #slList=[idx for name in names if name is in slNames]

    allMedL=[]
    allRelL=[]
    subSetL=[]
    if medIdxs:
        subSetL.append("Medium")
    if relIdxs:
        subSetL.append("Relaxed")
        
    for stamp in ap_stamps[slc]:
        thisTsYL=[]
        dat=loadSet(stamp, subL=subSetL)
        relL=[]
        medL=[]
        #tsL=[]
        for idxs, D, l in [(relIdxs, dat.relD, relL), (medIdxs, dat.medD, medL)]:
            if idxs and D:
                l.append([D[0]]+[D[1][idx] for idx in idxs])
                #tsL.append(D[0])
            #else:
                #tsL.append(None)
                #l.append(None)
            
        if medL:
            allMedL.append(medL)
        if relL:
            allRelL.append(relL)

        if 0:
            try:
                dat=loadSet(stamp, subL=subSet)
                if subSet=='Relaxed':
                    D=dat.relD
                elif subSet=='Medium':
                    D=dat.medD
                elif subSet=='Slow':
                    D=dat.slowD
                if D:
                    sd=D[0].copy()
                    y=D[1][D[2].index(name)].copy()
                    del dat
                    sdL.append(sd)
                    yL.append(y)
                    
            except ValueError:
                print("\nTimestamp {0} didn't load, maybe no '{1}' data? \n \n".format(stamp, subSet))
        if allRelL:
            allRelL=dstack(allRelL).squeeze()
        if allMedL:
            allMedL=dstack(allMedL).squeeze()
    return allMedL, allRelL

def loadBins(root_path, useMetaFile=True, nVars=1, endian='', dType=None, totalDataType=None):
    """Load all the binary files in that match the root path.

    They match the root path with regex 'root_path( \d+)?.bin',
    that is anything that starts with root_path, ends with .bin, and optionally has a number after a space. If there are numbers after the space, the files are loaded and the data appended in the correct order.
    """
    dirname=path.dirname(root_path) 
    rootname=path.basename(root_path)   
    r=re.compile('{0}( \d+)?\.bin'.format(rootname))
    dlist=os.listdir(dirname);
    dlist=[st for st in dlist if r.match(st)]
    if not dlist:
        print("No data for {0}".format(root_path))
        return None
    if len(dlist)>1:
        dlist.sort( key=lambda st: int(r.match(st).groups()[0]) ) # dlist is now sorted
    
    dat_list=[loadABin(path.join(dirname, fname), useMetaFile=useMetaFile, nVars=nVars, endian=endian, totalDataType=totalDataType) for fname in dlist]
    # Combine these into a single file...

    dat=np.hstack(dat_list)
    #datnames=dat_list[0].keys()
    return dat

def loadABin(filepath, nVars=1, useMetaFile=True, endian='', totalDataType=None):
    """Load a single binary file at filepath
    """
    #This is all just establishing how the data is stored in the file
    if totalDataType is None: #If a data type wasn't given, we'll try to work it out:
        if useMetaFile: #Usually the data type is stored in a meta file: 
            fname_components=path.splitext(filepath)[0].split(' ')
            if fname_components[-1].isdigit():
                fname_components.pop()
            meta_file_path=' '.join(fname_components) + '.npmeta'
            npmeta_path=' '.join(fname_components) + '.npmeta'
            if path.exists(npmeta_path):
                mfile=open(npmeta_path)
                lines=mfile.readlines()
                pairs=[line.strip().split(',') for line in lines if not line.isspace() and not line.startswith('#') ]# should be tuples of e.g. ('signal', 'f8');
                defaultDtype='>f4'
                totalDataType=[]
                for p in pairs:
                    name=p[0].strip()
                    if len(p)>1:
                        dt=p[1].strip()
                        defaultDtype=dt;
                    else:
                        dt=defaultDtype;
                    totalDataType.append((name, dt));
            else: #If there was no appropriate unambiguous meta-file, look for the old style:
                meta_file_path=' '.join(fname_components) + '.meta'
                if not path.exists(meta_file_path): #Sometimes they have the extension .meta.txt for no good reason, so check for that too
                    meta_file_path=meta_file_path+'.txt'
                if path.exists(meta_file_path):
                    convert.spmeta2npmeta(meta_file_path) #This convert the old style to the new (Corrects for some changes, but not all)
                    print("Found an old meta-file only. Will attempt to make something useful out of it (old style is ambiguous)")
                    return loadABin(filepath, nVars, useMetaFile) #We'll run the function again with the new meta file exists
                    #mfile=open(meta_file_path)
                else:
                    raise IOError("Looking for a meta-file (for {0}) but none exists".format(' '.join(fname_components)))
                sigNames=mfile.readlines()
                sigNames=[name.strip() for name in sigNames]
                if sigNames[0]=='Sidereal Days':
                    sigNames=sigNames[1:]
                nVars=len(sigNames)
                sigDataType=[(name, endian+'f8') for name in sigNames]
                timeDataType=[('sd', endian+'f8')]
                totalDataType=timeDataType+sigDataType

        else:
            timeDataType=[('sd', endian+'f8')]
            sigDataType=[('s{0}'.format(n), endian+'f8') for n in range(1,nVars+1)]
            totalDataType=timeDataType+sigDataType

    # This is where we actually read it
    dat=np.fromfile(filepath, totalDataType)

    return dat;

if __name__=="__main__":
    #ddir='/media/morgan/6a0066c2-6c13-49f4-9322-0b818dda2fe8/Princeton Data/South Pole/SP_motor 5253.67 Bundle/'
    #fpath=ddir+'SP_motor 5253.67 sensors 8.bin'
    ddir='/media/morgan/6a0066c2-6c13-49f4-9322-0b818dda2fe8/Princeton Data/South Pole/SP_aparatus 5256.38 Bundle/'
    #ddir='/media/morgan/6a0066c2-6c13-49f4-9322-0b818dda2fe8/Princeton Data/South Pole/SP_motor 5256.38 Bundle/'
    fname='SP_aparatus 5256.38 Medium 6.bin'
    fname_f='SP_aparatus 5256.38 Fast'
    ddir2='/media/morgan/6a0066c2-6c13-49f4-9322-0b818dda2fe8/Princeton Data/South Pole/SP_motor 5253.67 Bundle/'
    fname2='SP_motor 5253.67 sensors'
    fpath=ddir+fname
    fpath_f=ddir+fname_f

    dat=loadABin(fpath, useMetaFile=True);
    dat_f=loadBins(fpath_f, useMetaFile=False, totalDataType=[('sd','>f8'), ('s1', '>f4')]);
    dat2=loadBins(ddir+'SP_aparatus 5256.38 Medium')
    dat3=loadBins(ddir2+fname2, useMetaFile=True)
