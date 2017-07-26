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
parser.add_option(
    "-f",
    "--format",
    dest="format",
    default='i3',
    type=str,
    help="i3 or h5"
)

options, args = parser.parse_args()

if options.outputFile is None:
    raise Exception("Output file is not defined!")
if options.format not in ['i3', 'h5']:
    raise Exception("Format can only be either 'i3' or 'h5'!")

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
from icecube.hdfwriter import I3HDFWriter


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


clf = joblib.load(
    '/data/user/kevin/sterile/startingTrack/DataSelection/Starting_2012_clf_AdaBoost_3vars/Starting_2012_clf_AdaBoost_3vars_v1.pkl')


class Classifier(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if frame['Cascade_MinDist'].value < 0 and\
           frame['cascadeEnergy_millipede'].value != -1:
            __cascadeEnergy_millipede = frame['cascadeEnergy_millipede'].value
            __dEdx_millipede = frame['dEdx_millipede'].value
            __Cascade_MinDist = frame['Cascade_MinDist'].value
            __DecisionFunction = clf.decision_function([[__cascadeEnergy_millipede,
                                                         __dEdx_millipede,
                                                         __Cascade_MinDist]])[0]
            if __DecisionFunction > 0.08:
                self.PushFrame(frame)


class RelevantEnergyWriter(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if 'RelevantEnergy' in frame.keys():
            frame.Delete('RelevantEnergy')
        if 'NeutrinoEnergy_splined_millipede' in frame.keys():
            frame['RelevantEnergy'] = frame['NeutrinoEnergy_splined_millipede']
            self.PushFrame(frame)


tray = I3Tray()
tray.AddModule('I3Reader', 'reader', FilenameList=args)


tray.Add(Classifier, 'classifier')


tray.Add(RelevantEnergyWriter,'relevantEnergyWriter')


if(options.format == 'i3'):
    if(options.dstype != 'data'):
        tray.AddModule('I3Writer', 'writer', filename=options.outputFile,
                       DropOrphanStreams=map(icetray.I3Frame.Stream, 'IQ'),
                       Streams=map(icetray.I3Frame.Stream, 'IQP'))
    elif(options.dstype == 'data'):
        tray.AddModule('I3Writer', 'writer', filename=options.outputFile,
                       DropOrphanStreams=map(icetray.I3Frame.Stream, 'GCDIQ'),
                       Streams=map(icetray.I3Frame.Stream, 'GCDIQP'))
elif(options.format == 'h5'):
    tray.AddSegment(
        I3HDFWriter,
        keys=[
            'CascadeLlhVertexFitStatus',
            'Cascade_MinDist',
            'CorsikaWeightMap',
            'CutL3',
            'CutL4',
            'DeltaE_millipede',
            'Deltax_millipede',
            'FilterMask',
            'GaisserH3aWeight',
            'GenerationSpec',
            'HondaWeight',
            'I3MCWeightDict',
            'IsSingleTrack',
            'L4MonopodFit',
            'LDir',
            'MCCascade',
            'MCMaxInIceTrack',
            'MCMu',
            'MCNu',
            'MCPrimary',
            'MCTrue',
            'MonopodFit',
            'MuEx',
            'MuonEnergy_splined_millipede',
            'NCh',
            'NCh_noIT',
            'NDir',
            'NeutrinoEnergy_adaboost',
            'NeutrinoEnergy_splined_millipede',
            'ParaboloidErr1',
            'ParaboloidErr2',
            'RelevantEnergy',
            'SCh',
            'SDir',
            'SPEFit8',
            'SPEFit8Status',
            'SPEFitParaboloid2Status',
            'atmNumber',
            'cascadeEnergy_millipede',
            'cascadeFirstHit_millipede',
            'cascadeTime',
            'cascade_logl',
            'cascade_rlogl',
            'dEdx_millipede',
            'decisionFunction_2012_10var_v1',
            'firstVeto',
            'muonLength',
            'parabSigmaCorr',
            'spef_logl',
            'spef_rlogl',
            'totQ',
            'totQ_filtered'],
        SubEventStreams=['TTrigger'],
        Output=options.outputFile)


tray.AddModule('TrashCan', 'can')
tray.Execute()
tray.Finish()
