from os import path
from os import environ

def getSPData(timestamp, rotations, interval, delay):

    base_data_dir=environ['SP_DATA_DIR'];

    ddir=path.join(base_data_dir, 'SP_aparatus {0} Bundle'.format(timestamp))

    [sdF,sigF,sdM,sigM,sdRe,sigRe,sdS,sigS,medNames,reNames,sloNames]=loadCPTFiles(timestamp);

% Get triggers either by making them if parameters are given or loading
% them from the triggers file if available
if nargin>1
    trig=makeTrigM(sdF,rotations,interval,delay);
    if trig(end)>sdF(end)
        trig=trig(1:end-rotations);
    end
else
    trig=[];
    rotations=find(diff(trig)>(trig(2)-trig(1))*1.1,1);
end
mtrig=[];
sdP=[]; pos=[]; stat=[];
sensor=[]; sensorNames=[];
if isdir( fullfile(data_dir, ['SP_motor ',timestamp,' Bundle']) )
    cd( fullfile(data_dir, ['SP_motor ',timestamp,' Bundle']) );
    trigfiles=dir('*triggers*.txt');
    posfiles=dir('*positions*.bin');
    sensfiles=dir('*sensors*.bin');
    for i=1:length(trigfiles)
        mtrig=[mtrig;load(trigfiles(i).name)];
    end
    for i=1:length(posfiles)
        fid=fopen(posfiles(i).name);
        sdP=[sdP;fread(fid,'float64',8)];
        fseek(fid,0,'bof');
        data=fread(fid,'int32=>int32');
        data2=reshape(data,4,length(data)/4)';
        pos=[pos;data2(:,3)];
        stat=[stat;data2(:,4)];
        fclose(fid);
    end
    for i=1:length(sensfiles)
    end
end
if isempty(trig)
    trig=mtrig;
end
if rem(length(trig),rotations)
        disp(['There are ',num2str(rem(length(trig),rotations)),' extra triggers!'])
end

% Check North positions using Apparatus Position in Relaxed data 
if ~isempty(sdRe) & (mean(sigRe(:,19))>1)
    checkNorth(sdRe,sigRe,trig(1:find(trig<max(sdRe),1,'last')));
end
    
zerodata=GetZeroDataSP(timestamp);
cd(start_dir)
end

