#!/usr/bin/env python

import numpy as np
from icecube import dataclasses
import math

# DOM positions
geoFile = np.loadtxt(
    '/data/user/kevin/sterile/startingTrack/data/Icecube_geometry.20110414.complete.txt')

vetoP0 = np.array([-211.35, -404.48])
vetoP1 = np.array([-447.74, -113.13])
vetoP2 = np.array([-268.90, 354.24])
vetoP3 = np.array([-21.97, 393.24])
vetoP4 = np.array([54.26, 292.97])
vetoP5 = np.array([303.41, 335.64])
vetoP6 = np.array([472.05, 127.90])
vetoP7 = np.array([282.18, -325.74])


def LineFunc0(x, y):
    if(((vetoP1[1] - vetoP0[1]) * (x - vetoP0[0])) - ((y - vetoP0[1]) * (vetoP1[0] - vetoP0[0])) > 0):
        return 1
    else:
        return 0


def LineFunc1(x, y):
    if(((vetoP2[1] - vetoP1[1]) * (x - vetoP1[0])) - ((y - vetoP1[1]) * (vetoP2[0] - vetoP1[0])) > 0):
        return 1
    else:
        return 0


def LineFunc2(x, y):
    if(((vetoP3[1] - vetoP2[1]) * (x - vetoP2[0])) - ((y - vetoP2[1]) * (vetoP3[0] - vetoP2[0])) > 0):
        return 1
    else:
        return 0


def LineFunc3(x, y):
    if(((vetoP4[1] - vetoP3[1]) * (x - vetoP3[0])) - ((y - vetoP3[1]) * (vetoP4[0] - vetoP3[0])) > 0):
        return 1
    else:
        return 0


def LineFunc4(x, y):
    if(((vetoP5[1] - vetoP4[1]) * (x - vetoP4[0])) - ((y - vetoP4[1]) * (vetoP5[0] - vetoP4[0])) > 0):
        return 1
    else:
        return 0


def LineFunc5(x, y):
    if(((vetoP6[1] - vetoP5[1]) * (x - vetoP5[0])) - ((y - vetoP5[1]) * (vetoP6[0] - vetoP5[0])) > 0):
        return 1
    else:
        return 0


def LineFunc6(x, y):
    if(((vetoP7[1] - vetoP6[1]) * (x - vetoP6[0])) - ((y - vetoP6[1]) * (vetoP7[0] - vetoP6[0])) > 0):
        return 1
    else:
        return 0


def LineFunc7(x, y):
    if(((vetoP0[1] - vetoP7[1]) * (x - vetoP7[0])) - ((y - vetoP7[1]) * (vetoP0[0] - vetoP7[0])) > 0):
        return 1
    else:
        return 0


def DOMPosition(self, str_no, om_no):
    """
    Position of DOM from its string and om number
    returns: [x, y, z] array
    """
    for i in range(len(geoFile)):
        if((geoFile[i][0] == str_no) & (geoFile[i][1] == om_no)):
            return geoFile[i][2:5]


def DOMDistance(self, frame, p, recoName):
    """
    distance of point from line
    frame: i3 frame object
    p: point
    recoName: reconstruction name in the frame
    returns: distance in meters
    """
    a = np.array(
        [frame[recoName].pos.x, frame[recoName].pos.y, frame[recoName].pos.z])
    n = np.array(
        [frame[recoName].dir.x, frame[recoName].dir.y, frame[recoName].dir.z])
    return np.linalg.norm(np.subtract(np.subtract(a, p), np.multiply(np.dot(np.subtract(a, p), n), n)))


def IsContained(self, p):
    """ returns True if point 'p' is in IceCube cylinder """
    if((p[2] < 500) & (p[2] > -500)):
        dist = (p[0] * p[0]) + (p[1] * p[1])
        if(dist < 250000):
            return 1
        else:
            return 0
    else:
        return 0


def MCStarting(self, frame):
    """ returns True if starting track (cylindrical approximation) """
    if('MCMaxInIceTrack' not in frame.keys()):
        return -1
    f = frame['MCMaxInIceTrack']
    p = np.array([f.pos.x, f.pos.y, f.pos.z])
    if((p[2] < 500) & (p[2] > -500)):
        dist = (p[0] * p[0]) + (p[1] * p[1])
        if(dist < 250000):
            return 1
    return 0

# the time difference between hit time calculated based on the
# reconstructed track (recoName) and the actual hit on the DOM


def PhysicalTimeDelay(self, frame, p, hitTime, recoName):
    """
    Phsycal time delay of the hit time at point 'p'
    frame: i3 frame object
    hitTime: hit time in 'ns'
    recoName: reconstruction name in the frame
    returns: time delay
    """
    # w: closest point on track to point 'p'
    # u: position on track where Cherenkov photon left the track
    # p: DOM position
    # a: an arbitrary point on the reconstructed track
    # n: direction of the reconstructed track
    
    a = np.array(
        [frame[recoName].pos.x, frame[recoName].pos.y, frame[recoName].pos.z])
    n = np.array(
        [frame[recoName].dir.x, frame[recoName].dir.y, frame[recoName].dir.z])
    ap = np.subtract(p, a)
    aw_norm = np.dot(ap, n)
    aw = np.multiply(aw_norm, n)
    # 1/np.tan(0.711) #cherenkov angle is 41 deg #n=1.32
    uw_norm = np.linalg.norm(np.subtract(ap, aw)) * 1.161
    up_norm = uw_norm * 1.32
    uw = np.multiply(uw_norm, n)
    au = np.subtract(aw, uw)
    au_norm = np.dot(au, n)
    return (hitTime - (frame[recoName].time + ((au_norm + (up_norm * 1.32)) / frame[recoName].speed)))



def TimeMask(self, frame, str_no, om_no, hitTime, recoName):
    """
    Time mask at a paricular DOM
    str_no: DOM string number
    om_no: DOM om number
    frame: i3 frame object
    hitTime: hit time in 'ns'
    recoName: reconstruction name in the frame
    returns: True if within time mask
    """
    timeDelay = PhysicalTimeDelay(self, frame, DOMPosition(
        self, str_no, om_no), hitTime, recoName)
    if(timeDelay > -300 and timeDelay < 1500):
        return 1
    else:
        return 0


def DepositionPosition(self, frame, p, hitTime, recoName):
    # w:closest point on track to point 'p'
    # u:position on track where Cherenkov photon left the track
    # p:DOM position
    # a:an arbitrary point on the reconstructed track
    # n:direction of the reconstructed track
    a = np.array(
        [frame[recoName].pos.x, frame[recoName].pos.y, frame[recoName].pos.z])
    n = np.array(
        [frame[recoName].dir.x, frame[recoName].dir.y, frame[recoName].dir.z])
    ap = np.subtract(p, a)
    aw_norm = np.dot(ap, n)
    aw = np.multiply(aw_norm, n)
    # 1/np.tan(0.711) #cherenkov angle is 41 deg #n=1.32
    uw_norm = np.linalg.norm(np.subtract(ap, aw)) * 1.161
    up_norm = uw_norm * 1.32
    uw = np.multiply(uw_norm, n)
    au = np.subtract(aw, uw)
    au_norm = np.dot(au, n)
    if(up_norm > 250):
        return -1
    return (hitTime - (up_norm * 1.32 / frame[recoName].speed))


# Only up-going muons #startTime is the muon start time excluding cascade veto
def MuonLength(self, frame, startTime, recoName):
    """
    Moun length caclulator
    frame: i3 frame object
    startTime: time of the cascade vertex
    recoName: reconstruction name in the frame
    retruns: muon length in meters
    """
    a = np.array(
        [frame[recoName].pos.x, frame[recoName].pos.y, frame[recoName].pos.z])
    n = np.array(
        [frame[recoName].dir.x, frame[recoName].dir.y, frame[recoName].dir.z])
    startPos = np.add(a, np.multiply(
        n, ((startTime - frame[recoName].time) * frame[recoName].speed)))
    safetyCounter = 0
    while(not IsContained(self, startPos)):
        startPos = np.add(n, startPos)
        safetyCounter += 1
        if(safetyCounter > 500):
            return -1
    endPos = np.add(a, np.multiply(n, (500.0 - a[2]) / n[2]))
    if(not IsContained(self, endPos)):
        prec = 2
        outPos = endPos
        inPos = startPos
        while(prec > 1):
            midPos = np.divide(np.add(outPos, inPos), 2)
            if(IsContained(self, midPos)):
                inPos = midPos
            else:
                outPos = midPos
            prec = np.linalg.norm(np.subtract(outPos, inPos))
        endPos = midPos
    return (np.linalg.norm(np.subtract(startPos, endPos)))


def BadFrameVeto(self, frame, geoframe, omkey, cascadeVetoRange, recoName):
    """
    Bad DOM marker
    frame: i3 frame object
    geoframe: GCD frame
    recoName: reconstruction name in the frame
    omkey: om, string key
    cascadeVetoRange: cascade veto diameter
    returns: '0' for good DOM and '1' for bad DOM
    """
    a = np.array(
        [frame[recoName].pos.x, frame[recoName].pos.y, frame[recoName].pos.z])
    n = np.array(
        [frame[recoName].dir.x, frame[recoName].dir.y, frame[recoName].dir.z])
    # NOTES: if cascade veto time !=-1,
    vetoPos = np.add(a, np.multiply(
        n, frame['cascadeTime'].value + cascadeVetoRange - frame[recoName].time))
    domPos = np.array([geoframe.omgeo[omkey].position.x,
                       geoframe.omgeo[omkey].position.y, geoframe.omgeo[omkey].position.z])
    # NOTE: check if the output is sensable base on the position + some all
    # get cut
    if (np.dot(n, np.subtract(domPos, vetoPos)) < 0):
        return 1
    else:
        return 0


def BadFrameVetoTest(self, frame, geoframe, omkey):
    if (geoframe.omgeo[omkey].position.x < 0):  # NOTE: all gets cut here and why?
        return 1
    else:
        return 0


def I3VectorOMKeyCopy(self, framekey):
    tmpCopy = dataclasses.I3VectorOMKey()
    for omkey in framekey:
        tmpCopy.append(omkey)
    return tmpCopy


def NchanCounter(self, strNo, domNo):
    """
    NChan (number of DOMs hit)
    strNo: array of string
    domNo: array of DOM numbers
    Returns array of Nchan
    """
    stomArray = np.zeros(5160)
    for i in range(len(strNo)):
        stomArray[((strNo[i] * 60) + domNo[i])] = 1
    return int(sum(stomArray))


def SchanCounter(self, strNo):
    """
    SChan (number of strings hit)
    strNo: array of string
    Returns array of Shan
    """
    stArray = np.zeros(86)
    for i in range(len(strNo)):
        stArray[strNo[i]] = 1
    return int(sum(stArray))


def InOrOut(self, x, y):
    """ Returns True if x, y is in 2-D detector Hexagon """
    return LineFunc0(x, y) & LineFunc1(x, y) & LineFunc2(x, y) & LineFunc5(x, y) & LineFunc6(x, y) & LineFunc7(x, y) & ((LineFunc3(x, y) | LineFunc4(x, y)))


def MCStarting_polygon(self, frame):
    """
    Modified 'IsContained' without cylindrical approximation
    using the detector's real shape
    """
    if(dataclasses.get_most_energetic_muon(frame['I3MCTree']) == None):
        return -1
    p = np.array([dataclasses.get_most_energetic_muon(frame['I3MCTree']).pos.x, dataclasses.get_most_energetic_muon(
        frame['I3MCTree']).pos.y, dataclasses.get_most_energetic_muon(frame['I3MCTree']).pos.z])
    if((p[2] < 470) & (p[2] > -440)):
        if(InOrOut(self, p[0], p[1])):
            return 1
    return 0
