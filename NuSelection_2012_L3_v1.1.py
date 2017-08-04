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
parser.add_option("-t", "--dstype", dest="dstype", default=None,
                  type=str, help="data, nugen, nufsgen or corsika")

options, args = parser.parse_args()

if options.outputFile is None:
    raise Exception("output file is not defined")
if options.dstype is None:
    raise Exception("dstype is not defined")
if options.dstype not in ["data", "nugen", "nufsgen", "corsika"]:
    raise Exception("dstype does not exist")
if len(args) == 0:
    raise Exception("no input file is given")

from definitions import *
import icecube
from I3Tray import *
from icecube import icetray, dataio, dataclasses, photonics_service, DomTools, weighting
from icecube import tableio, hdfwriter
import copy
import math
import numpy as np
import tables
from calibration_extras_new import *
from icecube.hdfwriter import I3HDFWriter
from icecube import MuonGun, simclasses
from icecube import MuonInjector
from icecube import TopologicalSplitter
from icecube import lilliput, improvedLinefit
from icecube import gulliver, gulliver_modules, improvedLinefit, paraboloid, cscd_llh
from icecube.icetray import I3Units
from icecube.cscd_llh import I3CscdLlhFitParams
from icecube.common_variables import direct_hits
dh_defs = [direct_hits.I3DirectHitsDefinition(
    "Chris", -10 * I3Units.ns, 150 * I3Units.ns)]

from cascade_reconstruction import OfflineCascadeReco, MuonReco
from icecube import phys_services


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


tabledir = os.path.expandvars("$I3_DATA/photon-tables/splines")
abs_table = os.path.join(tabledir, 'ems_mie_z20_a10.abs.fits')
prob_table = os.path.join(tabledir, 'ems_mie_z20_a10.prob.fits')
cascade_service = photonics_service.I3PhotoSplineService(
    abs_table, prob_table, 0)

stringB86 = [1, 2, 3, 4, 5, 6, 7, 13, 14, 21, 22, 30, 31, 40,
             41, 50, 51, 59, 60, 67, 68, 72, 73, 74, 75, 76, 77, 78]
stringB79 = [2, 3, 4, 5, 6, 7, 8, 13, 15, 21, 23, 30, 32, 40,
             41, 50, 51, 59, 60, 67, 68, 72, 73, 74, 75, 76, 77, 78]
stringB79_2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 20, 21, 23, 24, 29, 30, 32, 33, 39, 40,
               41, 42, 49, 50, 51, 52, 58, 59, 60, 61, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78]
stringB86_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 29, 30, 31, 32, 39,
               40, 41, 42, 49, 50, 51, 52, 58, 59, 60, 61, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78]


goodFramesList = ['I3SuperDST', 'I3EventHeader', 'LineFit', 'MCPrimary', 'QFilterMask', 'LineFit_TT', 'MPEFit_TT', 'MonopodFit', 'cascadeTime', 'muonLength',
                  'firstVeto', 'totQ', 'totQ_filtered', 'MuEx', 'MuEx_k', 'MuEx_SRV', 'L4MonopodFit', 'CutL3', 'CutLk',
                  'GenerationSpec', 'MCMaxInIceTrack', 'MCCascade', 'MCTrue', 'MCMu',
                  'TrackFitParaboloidFitParams', 'TrackFitCuts', 'TrackFitCuts_NoDC',
                  'mpef_rlogl', 'mpef_logl', 'NCh', 'MPEFitParaboloid2FitParams', 'SCh',
                  'I3MCWeightDict', 'HondaWeight', 'GaisserH3aWeight', 'ParaboloidErr1', 'ParaboloidErr2', 'I3TriggerHierarchy',
                  'BundleGen', 'I3SuperDST', 'I3MCTree', 'I3MCTree_preMuonProp', 'TTPulses_recombined', 'TTPulses',
                  'SRTOfflinePulses', 'TWSRTOfflinePulses', 'Multiplicity', 'MMCTrackList', 'MPEFitMuEX', 'CorsikaWeightMap', 'MCNu']

goodFilterList = ['CascadeFilter_12', 'DeepCoreFilter_12', 'DeepCoreFilter_TwoLayerExp_12',
                  'FSSFilter_12', 'GCFilter_12', 'ICOnlineL2Filter_12', 'LowUp_12', 'MuonFilter_12',
                  'SlopFilterTrig_12', 'VEFFilter_12']

totChargeCut = 100
cascadeVetoRange = 100.0
surface = MuonGun.Cylinder(1000, 500)

I3GeoFile = dataio.I3File(args[0])
I3GeoFile.pop_frame()
geoFrame = I3GeoFile.pop_frame()
geoframe = geoFrame['I3Geometry']
geoFrameKeys = geoFrame['I3Geometry'].omgeo.keys()

recoName = "SPEFit8"
cePulse = "TTPulses_recombined"
# cePulse="OfflinePulsesHLC"
offlinePulsesHLC = "OfflinePulsesHLC"  # "OfflinePulses"
offlinePulses = "InIcePulses"


@icetray.traysegment
def CascadeHitCleaning(tray, name, Pulses='SplitInIcePulses',
                       TWOfflinePulsesHLC='TWOfflinePulsesHLC',
                       If=lambda f: True,
                       SubEventStreams=[''],
                       ):

    tray.AddModule('I3LCPulseCleaning', name + '_LCCleaning',
                   Input=Pulses,
                   OutputHLC='OfflinePulsesHLC',  # ! Name of HLC-only DOMLaunches
                   OutputSLC='OfflinePulsesSLC',  # ! Name of the SLC-only DOMLaunches
                   If=lambda f: 'OfflinePulsesHLC' not in f,
                   )

    tray.AddModule('I3TimeWindowCleaning<I3RecoPulse>', name + 'TWC_HLC',
                   InputResponse=Pulses,  # ! Use pulse series
                   OutputResponse='TWOfflinePulsesHLC',  # ! Name of cleaned pulse series
                   TimeWindow=6000 * I3Units.ns,  # ! 6 usec time window
                   If=lambda f: 'TWOfflinePulsesHLC' not in f,
                   )


def make_conj(f1, f2):
    def conj(frame):
        c1 = f1(frame)
        if(not c1):
            return(False)
        return(f2(frame))
    return(conj)


def make_disj(f1, f2):
    def disj(frame):
        c1 = f1(frame)
        if(c1):
            return(True)
        return(f2(frame))
    return(disj)


class selector(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, frame):
        return(self.func(frame))

    def __and__(self, other):
        return(selector(make_conj(self.func, other.func)))

    def __or__(self, other):
        return(selector(make_disj(self.func, other.func)))

    def __invert__(self):
        return(selector(lambda frame: not self.func(frame)))


class SplitCounter(icetray.I3PacketModule):
    def __init__(self, ctx):
        icetray.I3PacketModule.__init__(self, ctx, icetray.I3Frame.DAQ)
        self.AddOutBox("OutBox")

    def Configure(self):
        pass

    def FramePacket(self, frames):
        countP = 0
        countAP = 0
        for frame in frames:
            if(frame["I3EventHeader"].sub_event_stream == 'TTrigger'):
                countP += 1
                if('IsAfterPulse' in frame):
                    countAP += 1
        for frame in frames:
            if(frame.Stop == icetray.I3Frame.DAQ):
                frame['countP'] = icetray.I3Int(countP)
                frame['countAP'] = icetray.I3Int(countAP)
                if(countP == 1 and countAP == 0):
                    frame['IsSingleTrack'] = icetray.I3Bool(True)
                else:
                    frame['IsSingleTrack'] = icetray.I3Bool(False)
            self.PushFrame(frame)


class StartingTrackDef(object):
    def __init__(self):
        pass

    def Initializing(self, frame):
        global off_pulses, all_pulses, stringNo, omNo, pulse_charge, pulse_time
        off_pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(
            frame, offlinePulsesHLC)
        all_pulses = [p for i, j in off_pulses for p in j]
        stringNo = np.array([i.string for i, j in off_pulses for p in j])
        omNo = np.array([i.om for i, j in off_pulses for p in j])
        pulse_charge = np.array([p.charge for p in all_pulses])
        pulse_time = np.array([p.time for p in all_pulses])
        # noDC
        pulse_time = pulse_time[stringNo < 79]
        pulse_charge = pulse_charge[stringNo < 79]
        omNo = omNo[stringNo < 79]
        stringNo = stringNo[stringNo < 79]
        # noIT
        pulse_time = pulse_time[omNo < 61]
        pulse_charge = pulse_charge[omNo < 61]
        omNo = omNo[omNo < 61]
        stringNo = stringNo[omNo < 61]
    # the time difference between hit time calculated based on the
    # reconstructed track ('SPEFit2') and the actual hit on the DOM

    def InitializingFilters(self, frame):
        global stringNo_filtered, omNo_filtered, pulse_charge_filtered, pulse_time_filtered, filter_b1, filter_b2, filter_time_b, filter_time
        filter_b1 = (omNo == 1) | (omNo == 60) | (omNo == 2) | (omNo == 59) | (omNo == 38) | (
            omNo == 39) | (omNo == 40) | (omNo == 41) | (omNo == 42) | (omNo == 58) | (omNo == 57)
        for i in stringB86:
            filter_b1 = np.logical_or((stringNo == i), filter_b1)
        filter_time = copy.copy(filter_b1)
        for i in range(len(filter_time)):
            filter_time[i] = TimeMask(
                self, frame, stringNo[i], omNo[i], pulse_time[i], recoName)
        filter_time_b = np.logical_and(filter_b1, filter_time)
        stringNo_filtered = stringNo[filter_time]
        omNo_filtered = omNo[filter_time]
        pulse_charge_filtered = pulse_charge[filter_time]
        pulse_time_filtered = pulse_time[filter_time]

    def FirstDOMVeto(self, frame):
        if(len(pulse_time[filter_time]) == 0):
            return -1
        filter_starting = (pulse_time == min(pulse_time[filter_time]))
        filter_starting = np.logical_and(filter_starting, filter_b1)
        return len(pulse_charge[filter_starting])

    def TotQ(self, frame):
        return [sum(pulse_charge), sum(pulse_charge[filter_time])]


class ConvetionalFilterPass(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        FilterBool = np.zeros(len(goodFilterList))
        for i in range(len(goodFilterList)):
            FilterBool[i] = frame['QFilterMask'][goodFilterList[i]
                                                 ].condition_passed
        if(any(FilterBool)):
            self.PushFrame(frame)


class HLCcleaner(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if('OfflinePulsesHLC' in frame):
            frame.Delete('OfflinePulsesHLC')
        if('TWOfflinePulsesHLC' in frame):
            frame.Delete('TWOfflinePulsesHLC')
        self.PushFrame(frame)


class TTriggerCut(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if(frame["I3EventHeader"].sub_event_stream == 'TTrigger'):
            self.PushFrame(frame)


class PrePreCuts(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        self.starting_track_def = StartingTrackDef()

    def Physics(self, frame):
        if('CutL3' not in frame.keys()):
            frame['CutL3'] = icetray.I3Bool(True)
            self.starting_track_def.Initializing(frame)
            frame['NCh'] = icetray.I3Int(NchanCounter(self, stringNo, omNo))
            if(frame['NCh'].value > 15):
                frame['SCh'] = icetray.I3Int(SchanCounter(self, stringNo))
                frame.Delete('CutL3')
        self.PushFrame(frame)


class PreCuts(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if('CutL3' not in frame.keys()):
            frame['CutL3'] = icetray.I3Bool(True)
            if(recoName in frame.keys()):
                if(frame[recoName].dir.zenith != None):
                    if(frame[recoName].dir.zenith > 1.4):
                        frame.Delete('CutL3')
        self.PushFrame(frame)


class ExtraValues(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        self.starting_track_def = StartingTrackDef()

    def Physics(self, frame):
        if('CutL3' not in frame.keys()):
            frame['CutL3'] = icetray.I3Bool(True)
            self.starting_track_def.Initializing(frame)
            self.starting_track_def.InitializingFilters(frame)
            frame['SPEFit8Status'] = icetray.I3Bool(
                frame['SPEFit8'].fit_status == 0)
            # direc hits
            direct_hits_map = direct_hits.calculate_direct_hits(dh_defs,
                                                                frame['I3Geometry'],
                                                                frame['TWOfflinePulsesHLC'].apply(
                                                                    frame),
                                                                frame['SPEFit8'])
            frame['LDir'] = dataclasses.I3Double(
                direct_hits_map['Chris'].dir_track_length)
            frame['SDir'] = dataclasses.I3Double(
                direct_hits_map['Chris'].dir_track_hit_distribution_smoothness)
            frame['NDir'] = icetray.I3Int(direct_hits_map['Chris'].n_dir_doms)
            if(frame['LDir'].value > 200):
                frame['firstVeto'] = icetray.I3Bool(
                    self.starting_track_def.FirstDOMVeto(frame))
                tmp = self.starting_track_def.TotQ(frame)
                frame['totQ'] = dataclasses.I3Double(tmp[0])
                f_tmp = frame[recoName + 'FitParams']
                frame['spef_logl'] = dataclasses.I3Double(f_tmp.logl)
                frame['spef_rlogl'] = dataclasses.I3Double(f_tmp.rlogl)
                if(frame['spef_rlogl'].value < 9.0):
                    f_tmp = frame['CascadeLlhVertexFitParams']
                    frame['cascade_logl'] = dataclasses.I3Double(f_tmp.NegLlh)
                    frame['cascade_rlogl'] = dataclasses.I3Double(
                        f_tmp.ReducedLlh)
                    if options.dstype is not "data":
                        if('I3MCTree' in frame.keys()):
                            if options.dstype is "corsika":
                                frame['Multiplicity'] = dataclasses.I3Double(
                                    frame['CorsikaWeightMap']['Multiplicity'])
                            primaries = frame['I3MCTree'].get_primaries()
                            daughters = (
                                frame['I3MCTree'].get_daughters(primaries[0]))
                            for i in range(len(daughters)):
                                if(frame['I3MCTree'].get_daughters(primaries[0])[i].type == dataclasses.I3Particle.MuPlus):
                                    frame['MCMaxInIceTrack'] = frame['I3MCTree'].get_daughters(primaries[0])[
                                        i]
                                    break
                                else:
                                    if(frame['I3MCTree'].get_daughters(primaries[0])[i].type == dataclasses.I3Particle.MuMinus):
                                        frame['MCMaxInIceTrack'] = frame['I3MCTree'].get_daughters(primaries[0])[
                                            i]
                                        break
                            icecube.weighting.get_weighted_primary(
                                frame, MCPrimary='MCPrimary')
                            frame['MCCascade'] = dataclasses.I3Particle(
                                dataclasses.get_most_energetic_cascade(frame['I3MCTree']))
                            if options.dstype is "nugen":
                                frame['MCNu'] = dataclasses.I3Particle(
                                    dataclasses.get_most_energetic_neutrino(frame['I3MCTree']))
                            if options.dstype is not "corsika":
                                frame['MCTrue'] = dataclasses.I3Double(
                                    MCStarting(self, frame))
                    frame.Delete('CutL3')
        self.PushFrame(frame)


class ParaboloidWriter(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        frame['ParaboloidErr1'] = dataclasses.I3Double(
            frame['SPEFitParaboloid2FitParams'].pbfErr1)
        frame['ParaboloidErr2'] = dataclasses.I3Double(
            frame['SPEFitParaboloid2FitParams'].pbfErr2)
        self.PushFrame(frame)


origin = dataclasses.I3Position(0, 0, 0)


class OutOfRangeCut(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Physics(self, frame):
        if('CutL3' not in frame.keys()):
            frame['CutL3'] = icetray.I3Bool(True)
            cap = phys_services.I3Calculator.closest_approach_position(
                frame['SPEFit8'], origin)
            d = (cap - frame['SPEFit8'].pos).magnitude
            dt = d / frame['SPEFit8'].speed
            dir_pos_neg = np.sign(
                (cap.z - frame['SPEFit8'].pos.z) / frame['SPEFit8'].dir.z)
            new_time = frame['SPEFit8'].time + (dir_pos_neg * dt)
            if(new_time < 50000):
                frame.Delete('CutL3')
        self.PushFrame(frame)


@selector
def splitFrames(frame):
    if(frame.Stop != icetray.I3Frame.Physics):
        return(True)
    return (frame["I3EventHeader"].sub_event_stream == "TTrigger")


@selector
def basicRecosAlreadyDone(frame):
    if(frame.Stop != icetray.I3Frame.Physics):
        return(True)
    return(frame.Has(recoName))


@selector
def afterpulses(frame):
    return(frame.Has("IsAfterPulses"))


def add_basic_reconstructions(tray, suffix, pulses, condition):
    tray.AddSegment(improvedLinefit.simple, "LineFit" + suffix,
                    If=condition,
                    inputResponse=pulses,
                    fitName="LineFit" + suffix
                    )
    tray.AddSegment(lilliput.I3SinglePandelFitter, "SPEFitSingle" + suffix,
                    If=condition,
                    domllh="SPE1st",
                    pulses=pulses,
                    seeds=["LineFit" + suffix],
                    #        trayname="tray"
                    )
    tray.AddSegment(lilliput.I3IterativePandelFitter, "SPEFit4" + suffix,
                    If=condition,
                    domllh="SPE1st",
                    n_iterations=4,
                    pulses=pulses,
                    seeds=["SPEFitSingle" + suffix],
                    #        trayname="tray"
                    )
    tray.AddSegment(lilliput.I3SinglePandelFitter, "MPEFit" + suffix,
                    If=condition,
                    domllh="MPE",
                    pulses=pulses,
                    seeds=["SPEFit4" + suffix],
                    #        trayname="tray"
                    )

# fetch masks or pulses transparently


def getRecoPulses(frame, name):
    pulses = frame[name]
    if pulses.__class__ == dataclasses.I3RecoPulseSeriesMapMask:
        pulses = pulses.apply(frame)
    return pulses


class FrameCleaner(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def DAQ(self, frame):
        keyLength = len(frame.keys())
        for i in range(keyLength):
            if(frame.keys()[keyLength - i - 1] not in goodFramesList):
                frame.Delete(frame.keys()[keyLength - i - 1])
        self.PushFrame(frame)

    def Physics(self, frame):
        keyLength = len(frame.keys())
        for i in range(keyLength):
            if(frame.keys()[keyLength - i - 1] not in goodFramesList):
                frame.Delete(frame.keys()[keyLength - i - 1])
        self.PushFrame(frame)


tray = I3Tray()
tray.AddModule('I3Reader', 'reader', FilenameList=args)

tray.AddModule(HLCcleaner, 'HLCcleaner')

tray.AddModule(ConvetionalFilterPass, 'convetionalFilterPass')
tray.AddModule("I3OrphanQDropper", "OrphanQDropper")


# Run TTrigger and do reconstructions
tray.AddModule("I3SeededRTHitMaskingModule", "SRTClean")(
    ("InputResponse", "I3SuperDST"),  # offlinePulses),
    ("OutputResponse", "SRTInIcePulses"),
    ("Stream", icetray.I3Frame.DAQ)
)

tray.AddModule("I3TopologicalSplitter", "TTrigger",
               InputName="SRTInIcePulses",
               OutputName="TTPulses",
               Multiplicity=4,
               TimeWindow=4000 * I3Units.ns,
               MaxDist=300 * I3Units.m,
               TimeCone=800 * I3Units.ns,
               SaveSplitCount=True
               )

tray.AddModule("AfterPulseSpotter", "Afterpulses")(
    ("StreamName", "TTrigger"),
    ("Pulses", "TTPulses"),
    ("TagName", "IsAfterPulse")
)

tray.Add(TTriggerCut, 'tTiggerCut')

tray.AddSegment(CascadeHitCleaning, 'CascadeHitCleaning',
                Pulses='TTPulses',
                #If=lambda f: 'OfflinePulsesHLC' not in f,
                SubEventStreams=['TTrigger']
                )

#add_basic_reconstructions(tray,"_TT","TTPulses",splitFrames & ~afterpulses & ~basicRecosAlreadyDone)

tray.Add(PrePreCuts, 'prePreCuts')

tray.AddSegment(OfflineCascadeReco, 'offlineCascadeReco',
                Pulses='TWOfflinePulsesHLC')

tray.AddSegment(MuonReco, 'muonReco', Pulses='TWOfflinePulsesHLC')

tray.Add(PreCuts, 'preCuts')

tray.AddModule(ExtraValues, 'extras')


tray.AddModule('SplitCutTimeWindowCalculator', 'SplitCutTimeWindowCalculator')(
    ("SubEventStream", "TTrigger"),
    ("BasePulses", "I3SuperDST"),
    ("CutObject", "CutL3"),
    ("SplitPulsesName", "TTPulses"),
    ("OutputPulsesName", "TTPulses_recombined")
)

tray.AddModule(SplitCounter, "SPC")

tray.AddModule("PacketCutter", "Cutter")(
    ("CutStream", "TTrigger"),
    ("CutObject", "CutL3")
)


tray.AddService("I3GulliverMinuitFactory", "Minuit2", Tolerance=0.01)

# The second and third services are simultaneously provided by
# the seed service this tells paraboloid which I3Particle
# to use as the center of the grid as well as the vertex
# location for each point on the grid. More advanced
# usage may use different seeds for these functions.
tray.AddService("I3BasicSeedServiceFactory", "ParaboloidSeed",
                InputReadout="TTPulses",
                FirstGuesses=[recoName],
                TimeShiftType="TFirst")

# The fourth service is provided by  a description of the liklihood
# function
tray.AddService("I3GulliverIPDFPandelFactory", "SPEParaboloidPandel",
                InputReadout="TTPulses",
                Likelihood="SPE1st",
                PEProb='GaussConvolutedFastApproximation',
                JitterTime=4.0 * I3Units.ns,
                NoiseProbability=10 * I3Units.hertz,
                # EventType="DirectionalCascade" ## Do not activate this line
                )

# finally load the Paraboloid module, its self.
tray.AddModule("I3ParaboloidFitter", "SPEFitParaboloid2",
               Minimizer="Minuit2",
               LogLikelihood="SPEParaboloidPandel",
               SeedService="ParaboloidSeed",
               GridpointVertexCorrection="ParaboloidSeed",
               VertexStepSize=5.0 * I3Units.m,
               ZenithReach=2.0 * I3Units.degree,
               AzimuthReach=2.0 * I3Units.degree,
               MaxMissingGridPoints=1,
               NumberOfSamplingPoints=8,
               NumberOfSteps=3,
               )


tray.Add(ParaboloidWriter, 'paraboloidWriter')

tray.AddModule(OutOfRangeCut, 'outOfRangeCut')

tray.AddModule("I3OrphanQDropper", "OrphanQDropper2")


tray.AddModule('I3Writer', 'writer', filename=options.outputFile,
               DropOrphanStreams=map(icetray.I3Frame.Stream, 'IQ'),
               Streams=map(icetray.I3Frame.Stream, 'IQP'))


tray.Execute()
tray.Finish()
