#!/usr/bin/env python

from optparse import OptionParser
from os.path import expandvars
import os
import sys
import glob

parser = OptionParser(
    usage='%prog GCDfile I3Files -o outputFile -t dstype [OPTIONS]',
    description='nu Selection')

parser.add_option(
    "-o",
    "--outputFile",
    dest="outputFile",
    default=None,
    type=str,
    help="output .h(df)5 file")
parser.add_option(
    "-t",
    "--dstype",
    dest="dstype",
    default=None,
    type=str,
    help="data, nugen, nufsgen or corsika")

options, args = parser.parse_args()

if options.outputFile is None:
    raise Exception("output file is not defined")


from definitions2_new import *
from I3Tray import *
from icecube import icetray, dataio, dataclasses, photonics_service
import math
import numpy as np
import tables
from icecube.hdfwriter import I3HDFWriter
#from icecube import millipede
from icecube import MuonGun, simclasses
#from icecube import myexamples
from icecube import MuonInjector
from icecube import TopologicalSplitter
from icecube import lilliput, improvedLinefit
from icecube import gulliver, gulliver_modules, improvedLinefit, cscd_llh
from icecube.cscd_llh import I3CscdLlhFitParams


sys.path.append('/data/user/kevin/softwares/IceCubeMinDist')
import ICMinDist

load("libMuonInjector")
load("RandomStuff")
load("libdataio")

load("icetray")
load("dataclasses")
load("dataio")

load("SeededRTCleaning")
load("libphys-services")
load("linefit")
load("lilliput")
load("gulliver")
load("gulliver-modules")

load("libmue")


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.externals import joblib


class MCPrimaryFixer(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if(options.dstype == 'nufsgen'):
            if('MCPrimary' in frame.keys()):
                frame.Delete('MCPrimary')
            frame['MCPrimary'] = dataclasses.I3Particle(
                frame['MCMaxInIceTrack'])
            if('MCPrimary' in frame.keys()):
                frame.Delete('MCCascade')
            frame['MCCascade'] = dataclasses.I3Particle(
                dataclasses.get_most_energetic_cascade(frame['I3MCTree']))
        self.PushFrame(frame)


class EnergyDefs(object):
    def __init__(self):
        pass

    def get_millipede_energy_data_short(self, frame, starting_cut=False):
        if('MillipedeHighEnergy' not in frame.keys()):
            return [], [], 0, 1000, False
        Millipede_ = frame['MillipedeHighEnergy']
        Time_ = []
        Energy_ = []
        AccumulatedEnergy_ = []
        AccumulatedEnergy_Percentile_ = []
        CascadeEnergy_ = 0.0
        j_0 = 0  # begining of sequence
        j_1 = 0  # end of sequence
        # find begining of track
        for i in range(len(Millipede_)):
            if(Millipede_[i].energy > 0.0):
                if(ICMinDist.minDistDet(Millipede_[i].pos.x,
                                        Millipede_[i].pos.y,
                                        Millipede_[i].pos.z) <= 0):
                    j_0 = i
                    IsStarting_ = True
                    mc_ = ICMinDist.minDistDet(Millipede_[i].pos.x,
                                               Millipede_[i].pos.y,
                                               Millipede_[i].pos.z)
                    break
                j_0 = i
                IsStarting_ = False
                mc_ = 1000
                break
        # find end of track
        for i in range(j_0, len(Millipede_)):
            if(ICMinDist.minDistDet(Millipede_[i].pos.x,
                                    Millipede_[i].pos.y,
                                    Millipede_[i].pos.z) >= 0):
                j_1 = i
                break
        # get cascade energy
        if(IsStarting_):
            for i in range(j_0, min(j_0 + 3, j_1)):
                CascadeEnergy_ += Millipede_[i].energy
        # will remove the cascade part (3x10m or define distance)
        if(starting_cut):
            if(IsStarting_):
                j_0 += 3
        # if track length is zero
        if(j_0 >= j_1):
            return [], [], CascadeEnergy_, mc_, False
        for i in range(j_0, j_1):
            Time_.append(Millipede_[i].time)
            Energy_.append(Millipede_[i].energy)
            AccumulatedEnergy_.append(sum(Energy_))
        for i in range(100):
            AccumulatedEnergy_Percentile_.append(
                np.percentile(AccumulatedEnergy_, i))
        return Time_, AccumulatedEnergy_Percentile_, CascadeEnergy_, mc_, True


regr = joblib.load(
    '/data/user/kevin/sterile/startingTrack/nuEn/regr/regr_12560_weighted_30.pkl')

import pickle
MuonEnergySpline = pickle.load(
    open(
        "/data/user/kevin/sterile/startingTrack/DataSelection/spline_dEdx2Emu.p",
        "rb"))


class EnergyWriterFixer(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        self.defs = EnergyDefs()

    def Physics(self, frame):
        if('NeutrinoEnergy_adaboost' in frame.keys()):
            frame.Delete('NeutrinoEnergy_adaboost')
        if('cascadeEnergy_millipede' in frame.keys()):
            frame.Delete('cascadeEnergy_millipede')
        if('Deltax_millipede' in frame.keys()):
            frame.Delete('Deltax_millipede')
        if('DeltaE_millipede' in frame.keys()):
            frame.Delete('DeltaE_millipede')
        if('dEdx_millipede' in frame.keys()):
            frame.Delete('dEdx_millipede')
        if('NeutrinoEnergy_splined_millipede' in frame.keys()):
            frame.Delete('NeutrinoEnergy_splined_millipede')
        Time, AccumulatedEnergy_Percentile, CascadeEnergy, mc_, Status = self.defs.get_millipede_energy_data_short(
            frame, starting_cut=True)
        frame['Cascade_MinDist'] = dataclasses.I3Double(mc_)
        # Neutrino Energy Estimator
        if((mc_ < 0) & Status):
            # Cascade Energies
            frame['cascadeEnergy_millipede'] = dataclasses.I3Double(
                CascadeEnergy)
            # Track Energies
            if(len(Time) > 0):
                muonDeltax_millipede = (Time[-1] - Time[0]) * 0.299792458
            else:
                muonDeltax_millipede = 0.0
            if(len(AccumulatedEnergy_Percentile) > 0):
                muonDeltaEnergy_millipede = AccumulatedEnergy_Percentile[-1]
            else:
                muonDeltaEnergy_millipede = 0.0
            if(muonDeltax_millipede > 0):
                muondEdx = muonDeltaEnergy_millipede * 1.0 / muonDeltax_millipede
            else:
                muondEdx = 0.0
            frame['Deltax_millipede'] = dataclasses.I3Double(
                muonDeltax_millipede)
            frame['DeltaE_millipede'] = dataclasses.I3Double(
                muonDeltaEnergy_millipede)
            frame['dEdx_millipede'] = dataclasses.I3Double(muondEdx)
            X_ = AccumulatedEnergy_Percentile
            X_.append(Time[-1] - Time[0])
            X_.append(CascadeEnergy)
            Nu_Energy_predict_ = regr.predict(X_)[0]
            frame['NeutrinoEnergy_adaboost'] = dataclasses.I3Double(
                Nu_Energy_predict_)
            # Neutrino energy by unfolding
            if(frame['dEdx_millipede'].value == 0):
                muonEn = 0.0
            else:
                muonEn = np.power(
                    10, MuonEnergySpline(
                        np.log10(
                            frame['dEdx_millipede'].value)))
            frame['NeutrinoEnergy_splined_millipede'] = dataclasses.I3Double(
                muonEn + frame['cascadeEnergy_millipede'].value)
        else:
            frame['cascadeEnergy_millipede'] = dataclasses.I3Double(-1)
        self.PushFrame(frame)


class DecisionFunctionCut(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if(frame['decisionFunction_2012_10var_v1'].value > 0.0343):
            self.PushFrame(frame)


tray = I3Tray()
tray.AddModule('I3Reader', 'reader', FilenameList=args)


tray.Add(DecisionFunctionCut, 'decisionFunctionCut')


tray.Add(MCPrimaryFixer, 'mcPrimaryFixer')


tray.Add(EnergyWriterFixer, 'energyWriterFixer')


tray.AddModule("muex", "muex")(
    # MuEx does not handle noise, so it should run on cleaned pulses
    ("pulses", "TTPulses_recombined"),
    ("rectrk", "SPEFit8"),
    ("result", "MuEx"),
    ("lcspan", 0),
    ("repeat", 0),
    ("rectyp", True),
    ("usempe", False),
    ("detail", False),
    ("energy", True),
    ("icedir", expandvars("$I3_BUILD/mue/resources/ice/mie")),
    ("badoms", "BadDomsList")
    #("If",finalSample)
)


if(options.dstype != 'data'):
    tray.AddModule('I3Writer', 'writer', filename=options.outputFile,
                   DropOrphanStreams=map(icetray.I3Frame.Stream, 'Q'),
                   Streams=map(icetray.I3Frame.Stream, 'IQP'))
elif(options.dstype == 'data'):
    tray.AddModule('I3Writer', 'writer', filename=options.outputFile,
                   DropOrphanStreams=map(icetray.I3Frame.Stream, 'GCDQ'),
                   Streams=map(icetray.I3Frame.Stream, 'GCDIQP'))


tray.AddModule('TrashCan', 'can')
tray.Execute()
tray.Finish()
