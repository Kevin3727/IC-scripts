#!/usr/bin/env python

from optparse import OptionParser
from os.path import expandvars
import os
import sys
import glob

parser = OptionParser(
    usage='%prog GCDfile I3Files -o outputFile -t dstype [OPTIONS]', description='nu Selection')

parser.add_option("-o", "--outputFile", dest="outputFile",
                  default=None, type=str, help="output .i3(.bz2) file")
parser.add_option("-s", "--sim", dest="sim", default=1, type=int,
                  help="0 for data, 1 for simulation, default=1")

options, args = parser.parse_args()

if options.outputFile is None:
    raise Exception("output file is not defined")


"""
Example millipede muon energy loss fits.  The first fits the
loss pattern as stochastic losses (e.g. from a high-energy
muon), and the second as continuous losses.
input: offline reconstructed .i3 file(s)
"""

from definitions2_new import *
from I3Tray import *
import sys
from icecube import icetray, dataio, dataclasses, photonics_service
load('millipede')
load("RandomStuff")
from calibration_extras_new import *
import math
import numpy as np
from icecube.millipede import HighEnergyExclusions


from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


table_base = os.path.expandvars('$I3_DATA/photon-tables/splines/emu_%s.fits')
muon_service = photonics_service.I3PhotoSplineService(
    table_base % 'abs', table_base % 'prob', 0)

iceModelDir = '/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/'
iceModelBasenames = {"SpiceMie": "ems_mie_z20_a10",
                     "Spice1": "ems_spice1_z20_a10"}
iceModelBasename = iceModelBasenames["SpiceMie"]

cascade_service = photonics_service.I3PhotoSplineService(
    iceModelDir + iceModelBasename + '.abs.fits', iceModelDir + iceModelBasename + '.prob.fits', 0)


muon_service = photonics_service.I3PhotoSplineService('/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/InfBareMu_mie_abs_z20a10_V2.fits',
                                                      '/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/InfBareMu_mie_prob_z20a10_V2.fits', 0, 400.0)
cascade_service = photonics_service.I3PhotoSplineService('/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/LowEnergyCorrectedCascades_z20_a10_HE.abs.fits',
                                                         '/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/LowEnergyCorrectedCascades_z20_a10_HE.prob.fits', 0.0, 400.0)

global clf4
clf4 = joblib.load(
    '/data/user/kevin/sterile/startingTrack/DataSelection/clf_AdaBoost_90percent_10vars/2012_clf4_AdaBoost_90percent_10vars_v1.pkl')


def selector(omkey, index, pulse):
    return omkey.om < 61


class IceTopRemover(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if('CutL4' not in frame.keys()):
            frame["TTPulses_recombined_noIT"] = dataclasses.I3RecoPulseSeriesMapMask(
                frame, "TTPulses_recombined", selector)
        self.PushFrame(frame)


class PrePulseCuts(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if('CutL3' in frame.keys()):
            frame['CutL4'] = icetray.I3Bool(True)
        else:
            if('CutL4' not in frame.keys()):
                frame['CutL4'] = icetray.I3Bool(True)
                if('TTPulses_recombined' in frame.keys()):
                    frame.Delete('CutL4')
        self.PushFrame(frame)


class FixTimeWindow(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if('CutL4' not in frame.keys()):
            # print(frame['I3EventHeader'].event_id)
            trigs = dataclasses.I3TriggerHierarchy.from_frame(
                frame, 'DSTTriggers')
            #print('trig time = ', trigs[0].time)
            twindow = frame[pulses + 'TimeRange']
            # print(twindow.start,twindow.stop)
            # if(trigs[0].time >twindow.stop):
            if(options.sim):
                off_pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(
                    frame, 'InIcePulses')
                all_pulses = [p for i, j in off_pulses for p in j]
                pulse_time = np.array([p.time for p in all_pulses])
                del frame[pulses + 'TimeRange']
                t = min(pulse_time) - 306  # (trigs[0].time - 10000)
                #print('t= ',t)
                frame[pulses + 'TimeRange'] = dataclasses.I3TimeWindow(
                    t, t + twindow.stop - twindow.start)
                #frame[pulses+'TimeRange'] = dataclasses.I3TimeWindow(twindow.start + t, twindow.stop + t)
        self.PushFrame(frame)


'''
        if trigs[0].time != 0:
            twindow = frame[pulses+'TimeRange']
            #del frame[pulses+'TimeRange']
            print(twindow.start,twindow.stop)
            #frame[pulses+'TimeRange'] = dataclasses.I3TimeWindow(twindow.start + trigs[0].time, twindow.stop + trigs[0].time)
        self.PushFrame(frame)
'''


class PreAdaBoostCuts(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if('CutL4' not in frame.keys()):
            frame['CutL4'] = icetray.I3Bool(True)
            if(frame['LDir'].value > 200):
                if(frame['SPEFit8Status'].value):
                    if(frame['SPEFit8'].dir.zenith > np.pi / 2):
                        if(np.isinf(frame['LDir'].value) == False):
                            if(np.isnan(frame['LDir'].value) == False):
                                if(np.isinf(frame['NDir'].value) == False):
                                    if(np.isnan(frame['NDir'].value) == False):
                                        if(np.isinf(frame['SDir'].value) == False):
                                            if(np.isnan(frame['SDir'].value) == False):
                                                frame.Delete('CutL4')
        self.PushFrame(frame)


spline = np.genfromtxt(
    '/data/user/kevin/sterile/startingTrack/paraboloidCorrectionSpline.dat')


def correctionSpline(n):
    nl = np.log10(n)
    if(nl < spline[0][0]):
        ns = nl - spline[0][0]
        return ((ns * spline[0][3]) + spline[0][4])
    if(nl > spline[8][0]):
        ns = nl - spline[8][0]
        return ((ns * spline[8][1]) + spline[8][2])
    for i in range(8):
        if(nl > spline[i][0]):
            if(nl < spline[i + 1][0]):
                ns = nl - spline[i][0]
                return ((np.power(ns, 3) * spline[i][1]) + (np.power(ns, 2) * spline[i][2]) + (ns * spline[i][3]) + spline[i][4])


class ParabSigmaCorrCalc(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if('CutL4' not in frame.keys()):
            parabSigma = np.sqrt(np.divide(np.add(np.multiply(frame['ParaboloidErr1'].value, frame['ParaboloidErr1'].value),
                                                  np.multiply(frame['ParaboloidErr2'].value, frame['ParaboloidErr2'].value)), 2))
            frame['parabSigmaCorr'] = dataclasses.I3Double(
                parabSigma * correctionSpline(frame['NCh'].value))
        self.PushFrame(frame)


class AdaBoostCalc(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if('CutL4' not in frame.keys()):
            frame['CutL4'] = icetray.I3Bool(True)
            df_input = [np.cos(frame['SPEFit8'].dir.zenith),
                        frame['LDir'].value,
                        frame['NDir'].value,
                        frame['SDir'].value,
                        frame['spef_logl'].value,
                        frame['spef_rlogl'].value,
                        frame['parabSigmaCorr'].value,
                        frame['totQ'].value,
                        frame['NCh'].value,
                        frame['SCh'].value]
            df4 = clf4.decision_function(df_input)[0]
            if(df4 > -0.05):
                frame['decisionFunction_2012_10var_v1'] = dataclasses.I3Double(
                    df4)
                frame.Delete('CutL4')
        self.PushFrame(frame)


class ExtraWriter(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if('CutL4' not in frame.keys()):
            frame['SPEFitParaboloid2Status'] = icetray.I3Bool(
                frame['SPEFitParaboloid2'].fit_status == 0)
        self.PushFrame(frame)


class MillipedeWriter(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if('CutL4' not in frame.keys()):
            f = frame['MillipedeHighEnergy']
            isStarting = -1
            for i in range(len(f)):
                if(f[i].energy > 0):
                    if(((f[i].pos.x * f[i].pos.x) + (f[i].pos.y * f[i].pos.y)) < 250000):
                        if((f[i].pos.z > -500) & (f[i].pos.z < 500)):
                            j_0 = i
                            isStarting = 1
                            break
                    j_0 = i
                    isStarting = 0
                    break
            frame['cascadeFirstHit_millipede'] = dataclasses.I3Particle(f[j_0])
            if(isStarting == 1):
                if(len(f) > j_0 + 2):
                    cascadeEnergy_millipede = f[j_0].energy + \
                        f[j_0 + 1].energy + f[j_0 + 2].energy
                else:
                    if(len(f) > j_0 + 1):
                        cascadeEnergy_millipede = f[j_0].energy + \
                            f[j_0 + 1].energy
                    else:
                        cascadeEnergy_millipede = f[j_0].energy
            else:
                cascadeEnergy_millipede = -1
            frame['cascadeEnergy_millipede'] = dataclasses.I3Double(
                cascadeEnergy_millipede)
            # Muon Energy
            muonDeltaEnergy_millipede = 0
            muonStartPos = f[j_0 + 3].pos
            muonEndPos = f[j_0 + 3].pos
            for j in range(j_0 + 3, len(f)):
                if(((f[j].pos.x * f[j].pos.x) + (f[j].pos.y * f[j].pos.y)) < 250000):
                    if((f[j].pos.z > -500) & (f[j].pos.z < 500)):
                        muonDeltaEnergy_millipede += f[j].energy
                        muonEndPos = f[j].pos
            muonDeltax_millipede = np.linalg.norm(
                muonEndPos - muonStartPos) + 10.  # 10 meters
            muondEdx = muonDeltaEnergy_millipede / muonDeltax_millipede
            frame['Deltax_millipede'] = dataclasses.I3Double(
                muonDeltax_millipede)
            frame['DeltaE_millipede'] = dataclasses.I3Double(
                muonDeltaEnergy_millipede)
            frame['dEdx_millipede'] = dataclasses.I3Double(muondEdx)
        self.PushFrame(frame)


import pickle
MuonEnergySpline = pickle.load(open(
    "/data/user/kevin/sterile/startingTrack/DataSelection/spline_dEdx2Emu.p", "rb"))


class EnergyWriter(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if('CutL4' not in frame.keys()):
            if(frame['cascadeEnergy_millipede'].value != -1):
                muonEn = np.power(10, MuonEnergySpline(
                    np.log10(frame['dEdx_millipede'].value)))
                # print(frame['dEdx_millipede'].value,type(frame['dEdx_millipede'].value))
                # print(muonEn,type(muonEn))
                frame['MuonEnergy_splined_millipede'] = dataclasses.I3Double(
                    muonEn)
                frame['NeutrinoEnergy_splined_millipede'] = dataclasses.I3Double(
                    muonEn + frame['cascadeEnergy_millipede'].value)
        self.PushFrame(frame)


class StupidCut(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if('CutL4' not in frame.keys()):
            self.PushFrame(frame)


if(options.sim):
    pulses = 'TTPulses_recombined'
else:
    pulses = 'TTPulses_recombined_noIT'

tray = I3Tray()
tray.AddModule('I3Reader', 'reader', FilenameList=args)

if(options.sim == 0):
    tray.Add(IceTopRemover, 'iceTopRemover')

tray.Add(PrePulseCuts, 'prePulseCuts')

tray.Add(PreAdaBoostCuts, 'preAdaBoostCuts')
tray.Add(ParabSigmaCorrCalc, 'parabSigmaCorrCalc')

tray.Add(AdaBoostCalc, 'adaBoostCalc')

tray.Add(StupidCut, 'stupidCut')


from icecube.wavedeform import AddMissingTimeWindow
tray.AddModule(AddMissingTimeWindow, pulses + 'TimeRange', Pulses=pulses)

tray.Add(CalibrationExtras, 'calibration_extras', mc=True, cePulse=pulses)

tray.Add(FixTimeWindow, 'fixTimeWindow')


from icecube.CascadeL3_IC79.level3.reco import AxialCscdLlhFitter
tray.Add(AxialCscdLlhFitter, "L4CascadeLlh", Pulses=pulses, Seed='SPEFit8')

exclusionsHE = tray.AddSegment(
    HighEnergyExclusions, "excludes_high_energies",
    Pulses=pulses,
    ExcludeDeepCore="DeepCoreDOMs",
    BadDomsList="BadDomsList")
exclusionsHE.append(pulses + "ExcludedTimeRange")

tray.AddModule('MuMillipede', 'millipede_highenergy',
               MuonPhotonicsService=muon_service, CascadePhotonicsService=cascade_service,
               PhotonsPerBin=2, MuonRegularization=0, ShowerRegularization=0,
               MuonSpacing=0, ShowerSpacing=10, SeedTrack='SPEFit8',
               ExcludedDOMs=exclusionsHE,
               Output='MillipedeHighEnergy', Pulses=pulses)


tray.AddModule("PacketCutter", "Cutter")(
    ("CutStream", "TTrigger"),
    ("CutObject", "CutL4")
)

tray.Add(ExtraWriter, 'extraWriter')
tray.Add(MillipedeWriter, 'millipedeWriter')
tray.Add(EnergyWriter, 'energyWriter')

if(options.sim):
    tray.AddModule('I3Writer', 'writer', filename=options.outputFile,
                   DropOrphanStreams=map(icetray.I3Frame.Stream, 'Q'),
                   Streams=map(icetray.I3Frame.Stream, 'IQP'))
else:
    tray.AddModule('I3Writer', 'writer', filename=options.outputFile,
                   DropOrphanStreams=map(icetray.I3Frame.Stream, 'GCDQ'),
                   Streams=map(icetray.I3Frame.Stream, 'GCDIQP'))


tray.AddModule('TrashCan', 'can')
tray.Execute()
tray.Finish()
