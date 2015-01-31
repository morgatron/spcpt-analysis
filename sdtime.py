from __future__ import division
from math import floor
import time
#y2000=time.struct_time([2000,1,0,0,0,0,0,0,0]);
y2000secs= 946684800# time.mktime(y2000);
secsPerSidDay= 86164.0905#( 23.9344696*3600);
FF=0.277715#0.7222841#*0.2776;

def secs2sd(t):
    return (t-y2000secs)/secsPerSidDay + FF #Fudge factor for some reason
def sd2secs(sd):
    return (sd-FF)*secsPerSidDay+y2000secs
def sd2gm(sd):
    return time.gmtime(sd2secs(sd))
def sd2local(sd):
    return time.localtime(sd2secs(sd))
def sdNow():
    return secs2sd(time.time());
def loc2sd(ltime_str, fmt=None):
    #raise NotImplementedError
    if fmt:
        ltime=time.strptime(ltime_str, fmt)
    elif fmt is None:
        fmt='%m %d %Y'
        try:
            ltime=time.strptime(ltime_str, fmt)
        except ValueError:
            fmt='%c'
            ltime=time.strptime(ltime_str,fmt)
    return secs2sd(time.mktime(ltime))
    


def solarDaysSinceY2000(unixTime=None):
    if unixTime==None:
        unixTime=time.time();
    return (unixTime + 3029572800.)/86400.

def siderealTimeKornack(unixTime=None):
    return (280.46061837 + 360.98564736629 * solarDaysSinceY2000(unixTime))/360.;

def siderealTimeMeesus(T=None):
    if T==None:
        T=time.gmtime()
    return 367*T.tm_year-floor(7*(T.tm_year+floor((T.tm_mon+9)/12))/4)+floor(275*T.tm_mon/9) +T.tm_mday+(T.tm_hour+T.tm_min/60+T.tm_sec/3600)/24-730531.5   

def siderealTimeMe(T=None):
    if T==None:
        T=time.time();
    T-=946684800
    return T/secsPerSidDay#/secsPerSidDay#30*366.242199
