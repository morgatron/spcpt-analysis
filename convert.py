import os
import re

def allDirsSpmeta2Npmeta(base_dir=os.getenv('SP_DATA_DIR')):
    ap_dir_regex=re.compile('.*SP_aparatus (\d+(?:\.\d*)?|\.\d+) Bundle.*')
    mo_dir_regex=re.compile('.*SP_motor (\d+(?:\.\d*)?|\.\d+) Bundle.*')

    it=os.walk(base_dir)
    for D in it:
        files=D[2]
        metaFiles=[fn for fn in files if fn.endswith(".meta") or fn.endswith('.meta.txt')]
        for metaFile in metaFiles:
            fpath=os.path.join(D[0], metaFile)
            if ap_dir_regex.match(D[0]):
                spmeta2npmeta(fpath, 'apparatus')
            #if mo_dir_regex.match(D[0]):
        #       spmeta2npmeta(fpath, 'motor')

def spmeta2npmeta(filePath, style=None):
    """Style can be 'apparatus' or 'motor'
    """
    if style is None:
        motorI=filePath.rfind('motor')
        apparatusI=filePath.rfind('aparatus')
        if motorI==apparatusI:
            raise ValueError("Don't know what kind of meta file {0} is.".format(filePath))
        else:
            style = 'motor' if motorI>apparatusI else 'apparatus'

    lines=open(filePath).readlines()
    if style=='apparatus':
        endian='>'
    elif style=='motor':
        endian='<'
    else:
        raise NotImplementedError("Don't know what style '{}' is".format(style))
    output= ['{}, {}f8\r\n'.format(lines.pop(0).strip(), endian)]
    output+= ['{}, {}f4\r\n'.format(line.strip(), endian) for line in lines]


    ind=filePath.rfind('.meta')
    outPath=filePath[:ind] + '.npmeta'

    open(outPath, 'w').writelines(output)
