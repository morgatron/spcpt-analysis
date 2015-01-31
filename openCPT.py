import numpy as np
from os import path
import os
import re

base_dir=os.environ['SP_DATA_DIR']
ddir= "/media/morgan/6a0066c2-6c13-49f4-9322-0b818dda2fe8/Princeton Data/South Pole/SP_aparatus 5240.33 Bundle/"
filepath=ddir+"SP_aparatus 5240.33 Medium 2.bin"


template_regex='SP_aparatus (\d+(?:\.\d*)?|\.\d+) {0}( \d+)?\.bin'

def loadApparatusFiles(root_path, useMetaFile=True):
    dirname=path.dirname(root_path) 
    rootname=path.basename(root_path)   
    r=re.compile('{0}( \d+)?\.bin'.format(rootname))
    dlist=os.listdir(dirname);
    dlist=[st for st in dlist if r.match(st)]
    dlist.sort( key=lambda st: int(r.match(st).groups()[0]) ) # dlist is now sorted
    
    dat_list=[openFile(path.join(dirname, fname), useMetaFile=useMetaFile) for fname in dlist]
    # Combine these into a single file...

    dat=np.hstack(dat_list)
    #datnames=dat_list[0].keys()
    return dat

def openFile(filepath, nVars=1, useMetaFile=False, endian='>'):
    timeDataType=[('sd', endian+'f8')]
    if useMetaFile:
        fname_components=path.splitext(filepath)[0].split(' ')
        if fname_components[-1].isdigit():
            fname_components.pop()
        meta_file_path=' '.join(fname_components) + '.meta'
        if path.exists(meta_file_path):
            mfile=open(meta_file_path)
        elif path.exists(meta_file_path+'.txt'):
            mfile=open(meta_file_path+'.txt')
        else:
            raise IOError
        sigNames=mfile.readlines()
        sigNames=[name.strip() for name in sigNames]
        if sigNames[0]=='Sidereal Days':
            sigNames=sigNames[1:]
        nVars=len(sigNames)
        sigDataType=[(name, endian+'f4') for name in sigNames]

    else:
        sigDataType=[('s{0}'.format(n), endian+'f4') for n in range(1,nVars+1)]



    totalDataType=timeDataType+sigDataType
    print("total dtype: {0}".format(totalDataType))
    dat=np.fromfile(filepath, totalDataType)

    return dat;

if __name__=="__main__":

    ddir= "/media/morgan/6a0066c2-6c13-49f4-9322-0b818dda2fe8/Princeton Data/South Pole/SP_aparatus 5240.33 Bundle/"
    filepath=ddir+"SP_aparatus 5240.33 Medium 2.bin"
    dat=loadApparatusFiles(ddir+'SP_aparatus 5240.33 Medium')



