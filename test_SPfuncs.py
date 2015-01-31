import SPfuncs as spf
import SPDataSet as sd
import pytest
from matplotlib import *


@pytest.fixture
def fake_data_stop_start_4dirs():
    rSpec=spf.RotationSpec(startingAngle=0, delay=9, interval=30, numRotations=20, rotAngles=[180], extraRotAngle=90)
    return sd.testDS(rSpec)

@pytest.fixture
def fake_data_stop_start_2dirs():
    rSpec= spf.RotationSpec(startingAngle=0, delay=9, interval=30, numRotations=20, rotAngles=[180], extraRotAngle=0)
    return sd.testDS(rSpec)

@pytest.fixture
def fake_data_continuous():
    ds=sd.SPDataSet('5410.04')
    rAmp=ds.rawData()
    rAmpAdded=spf.addFakeData(rAmp, rotationRate=ds.set_info['rotationRate'], amp=20, bBlankOriginalData=True, phi=pi/5)
    ds.fastD=rAmpAdded
    return ds
    #ds=sd.SPDataSet('test_test', preloadDict={'fastD': [s, sig, 'sig']})
    #need to put something ehre

def test_continuous_fitting(fake_data_continuous):
    contData=fake_data_continuous
    print(contData.sidAmp(subtractHarmsL=[]))

def test_cutAmp(fake_data_stop_start_4dirs):
    madeData=fake_data_stop_start_4dirs
    cA=madeData.cutAmp()
    print(cA.sig.shape)


