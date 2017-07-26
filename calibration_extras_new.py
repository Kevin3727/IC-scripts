
from icecube.icetray import traysegment, module_altconfig

try:
	from icecube.WaveCalibrator import DOMSimulatorCalibrator
except ImportError:
# DOMSimulatorWorkarounds option was removed when DOMsimulator was taken out behind the woodshed,
# as was the DOMSimulatorCalibrator segment. In order to recover detailed errata, however, we
# only need the modified saturation margin.
	DOMSimulatorCalibrator = module_altconfig('I3WaveCalibrator', FADCSaturationMargin=1) 

@traysegment
def CalibrationExtras(tray, name, mc=False, cePulse='OfflinePulses'):
	"""
	Add various calibration goodies and work-arounds concieved of in 2012:
	
	- Readout windows that take the transit time shift into account.
	- I3TimeWindows for digitizer clipping and PMT saturation.
	- Flag instances of an off-by-one bug in DOMsimulator
	"""
	
	from icecube import icetray, dataclasses
	from icecube.icetray import pypick
	from icecube import WaveCalibrator
	from icecube.wavedeform import AddMissingTimeWindow
	
	@pypick
	def raw_data(frame):
		return 'InIceRawData' in frame
	@pypick
	def errata(frame):
		return 'OldCalibrationErrata' in frame
	
	tray.AddModule('Rename', name+'MoveErrata', Keys=['CalibrationErrata', 'OldCalibrationErrata'], If=raw_data)
	tray.AddModule('Delete', name+'DeleteRange', Keys=['CalibratedWaveformRange', 'SaturationWindows'])
	if mc:
		tray.AddSegment(DOMSimulatorCalibrator, name+'WaveCal', If=errata)
	else:
		tray.AddModule('I3WaveCalibrator', name+'WaveCal', If=errata)
	tray.AddModule('I3WaveformTimeRangeCalculator', name+'SuperDSTRange', If=lambda frame: not 'CalibratedWaveformRange' in frame)
	tray.AddModule('I3PMTSaturationFlagger', name+'SaturationWindows', If=errata)
	tray.AddModule('Delete', name+'DeleteWaveforms', Keys=['CalibratedWaveforms'])
	tray.AddModule(AddMissingTimeWindow, name+'PulseRange')
	tray.AddModule(AddMissingTimeWindow, name+'MPulseRange', Pulses='MaskedOfflinePulses')
	
	def flag_borked_slc(frame):
		# Find SLC launches affected by DOMsimulator off-by-one bug,
		# and tell Millipede to ignore them
		#if not raw_data(frame):
		#	return
		bad_keys = set()
		#for om, dls in frame['InIceRawData'].iteritems():
		#	for dl in dls:
		#		if dl.lc_bit == False and dl.charge_stamp_highest_sample == 0:
		#			bad_keys.add(om)
		if raw_data(frame):
			for om, dls in frame['InIceRawData'].iteritems():
				for dl in dls:
					if dl.lc_bit == False and dl.charge_stamp_highest_sample == 0:
						bad_keys.add(om)
		if len(bad_keys) > 0:
			# Remove affected DOMs. This will never trip in real data.
			frame['RemaskedOfflinePulses'] = dataclasses.I3RecoPulseSeriesMapMask(frame, cePulse,lambda om, idx, pulse: om not in bad_keys)
			frame['BorkedSLC'] = dataclasses.I3VectorOMKey(bad_keys)
		else:
			frame['RemaskedOfflinePulses'] = dataclasses.I3RecoPulseSeriesMapMask(frame, cePulse)
			
		frame['RemaskedOfflinePulsesTimeRange'] = frame['MaskedOfflinePulsesTimeRange']
		
	tray.AddModule(flag_borked_slc, name+'FlagBorkedSLC')
	
	
