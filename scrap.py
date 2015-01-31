import loadSPFiles as lspf
import SPDataSet as sd
import SPfuncs as spf
import MT



def test():
    ds=sd.SPDataSet('5310.41')
    sAmp1=ds.sequenceAmp(sigWindow=spf.Window(3,-6))
    sAmp2=ds.sequenceAmp(sigWindow=spf.Window(3,-3))

    sig1=sAmp1.sig.ravel()
    sig2=sAmp2.sig.ravel()
    err1=sAmp1.err.ravel()
    err2=sAmp2.err.ravel()
    t=sAmp1.t.ravel()
    figure()
    errorbar(t, sig1, err1)
    errorbar(t, sig2, err2)

