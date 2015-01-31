import numpy as np
from os import path
import os
import re


def loadBins(root_path, useMetaFile=True):
	dirname=path.dirname(root_path) 
	rootname=path.basename(root_path)	
	r=re.compile('{0}( \d+)?\.bin'.format(rootname))
	dlist=os.listdir(dirname);
	dlist=[st for st in dlist if r.match(st)]
	dlist.sort( key=lambda st: int(r.match(st).groups()[0]) ) # dlist is now sorted
	
	dat_list=[loadAbin(path.join(dirname, fname), useMetaFile=useMetaFile) for fname in dlist]
	# Combine these into a single file...

	dat=np.hstack(dat_list)
	#datnames=dat_list[0].keys()
	return dat



def loadAbin(filepath, nVars=1, useMetaFile=True, endian=''):
	if useMetaFile:
		fname_components=path.splitext(filepath)[0].split(' ')
		if fname_components[-1].isdigit():
			fname_components.pop()
		meta_file_path=' '.join(fname_components) + '.npmeta'
		npmeta_path=' '.join(fname_components) + '.npmeta'
		if path.exists(npmeta_path):
			mfile=open(npmeta_path)
			lines=mfile.readlines()
			#totalDataType=[line.replace(',', ' ').split() for line in lines if not line.isspace() and not line.startswith('#') ]# should be tuples of e.g. ('signal', 'f8');
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
					
				
			#print("totalDataType: {0}".format(totalDataType))
		else:
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
			sigDataType=[(name, endian+'f8') for name in sigNames]
			timeDataType=[('sd', endian+'f8')]
			totalDataType=timeDataType+sigDataType

	else:
		timeDataType=[('sd', endian+'f8')]
		sigDataType=[('s{0}'.format(n), endian+'f8') for n in range(1,nVars+1)]
		totalDataType=timeDataType+sigDataType



	print("total dtype: {0}".format(totalDataType))
	dat=np.fromfile(filepath, totalDataType)

	return dat;

if __name__=="__main__":
	#ddir='/media/morgan/6a0066c2-6c13-49f4-9322-0b818dda2fe8/Princeton Data/South Pole/SP_motor 5253.67 Bundle/'
	#fpath=ddir+'SP_motor 5253.67 sensors 8.bin'
	ddir='/media/morgan/6a0066c2-6c13-49f4-9322-0b818dda2fe8/Princeton Data/South Pole/SP_aparatus 5256.38 Bundle/'
	#ddir='/media/morgan/6a0066c2-6c13-49f4-9322-0b818dda2fe8/Princeton Data/South Pole/SP_motor 5256.38 Bundle/'
	fname='SP_aparatus 5256.38 Medium 6.bin'
	ddir2='/media/morgan/6a0066c2-6c13-49f4-9322-0b818dda2fe8/Princeton Data/South Pole/SP_motor 5253.67 Bundle/'
	fname2='SP_motor 5253.67 sensors'
	fpath=ddir+fname

	dat=loadAbin(fpath, useMetaFile=True);
	dat2=loadBins(ddir+'SP_aparatus 5256.38 Medium')
	dat3=loadBins(ddir2+fname2, useMetaFile=True)
