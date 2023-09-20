#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .Init import *
from .Data_loop import *
from .various_functions import check_param
from .Error import SettingsError

class PreProcessing:

    '''
    ``PreProcessing`` allows you to transform your raw data so that you can
    then perform processing.

    This will allow you to transform your raw data into pixels in degrees or
    degrees per second, to filter them or to remove or extract your data
    between two events.

    - Use ``PreProcessing.Trial`` to test on a trial the different parameters
      of the functions present in PreProcessing in order to adjust them as well
      as possible.
    - Once the right parameters are found, you can use ``PreProcessing.Data``
      to apply the function to a set of data and save it.

    Parameters
    ----------
    dirpath: str
        Data directory path

    sub: str, or None (default None)
        Participant identifier
    task: str, or None (default None)
        Name of the Task
    ses: str, or None (default None)
        Name of the Session
    acq: str, or None (default None)
        Name of the Aquisition
    run: str, or None (default None)
        IndexRun
    RawData: bool (default False)
        If ``True`` open RawData,
        if ``False`` open DataAnemo
    '''

    def __init__(self, dirpath, sub=None, task=None, ses=None, acq=None,
                 run=None, RawData=False):

        self.dirpath = dirpath
        self.sub = sub
        self.task = task
        self.ses = ses
        self.acq = acq
        self.run = run
        self.RawData = RawData

        self.Trial = self.Trial(self)
        self.Data = self.Data(self)

    class Trial:

        '''
        Allows you to perform PreProcessings on a trial.
        '''

        def __init__(self, _):

            self._ = _
            self._.data = Data.open(_.dirpath, _.sub, _.ses, _.acq, _.run,
                                  _.RawData)

        def filter(self, xname, trial, expname, order_filter=2,
                   type_filter='lowpass', cutoff=30, toxname=None,
                   return_=True, **arg):

            '''
            Filters the ``xname`` data from the ``trial`` by applying a digital
            and analogue `Butterworth filter <https://docs.scipy.org/doc/scipy/
            reference/generated/scipy.signal.butter.html>`_ twice, once
            forward and once backward using the `scipy.signal.filtfilt
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal
            .filtfilt.html#scipy.signal.filtfilt>`_ function.

            With the parameter ``return_=True`` this function returns the
            calculated data and allows you to test the different parameters.

            Parameters
            ----------
            xname: str
                Name of the data to be transformed
            trial: int
                Number of the trial to be transformed
            expname: str
                Name of the particular experience to be transformed

            order_filter: int (default 2)
                The order of the filter.
            type_filter: str (default 'lowpass')
                The type of filter. Can be ``'lowpass'``, ``'highpass'``,
                ``'bandpass'``, or ``'bandstop'``
            cutoff: int (default 30)
                The critical frequencies for cutoff of filter

            toxname: str (default None)
                Name of the data to be saved

            return_: bool (default True)
                If ``True`` returns the value,
                else saves it in ``data``

            Returns
            -------
            filtered_data: list
                the calculated data if ``return_=True``
            '''

            check_param(self, PreProcessing.Trial, 'ANEMO.PreProcessing',
                        expname, trial)

            data = self._.data[expname].Data # data
            trial_data = data[data.trial==trial] # trial data

            settings = self._.data[expname].Settings # settings data
            if not 'SamplingFrequency' in settings.keys():
                raise SettingsError('SamplingFrequency', expname)

            sample_rate = settings['SamplingFrequency'].values[0]

            #------------------------------------------------------------------
            # Parameters for applying the digital filter
            #------------------------------------------------------------------
            from scipy import signal
            nyq_rate = sample_rate/2 # The Nyquist rate of the signal.
            Wn = cutoff/nyq_rate
            # Butterworth digital and analog filter design.
            b, a = signal.butter(N=order_filter, Wn=Wn, btype=type_filter)

            #------------------------------------------------------------------
            # Apply a digital filter forward and backward to a signal.
            #------------------------------------------------------------------
            # filtered data
            d = trial_data[xname]
            d = d[~np.isnan(d.values)] # allows you to remove the missing data
            filtered_data = signal.filtfilt(b, a, d.values)
            #------------------------------------------------------------------

            if return_:
                return filtered_data

            else:
                # add trial to the settings data
                settings = self._.data[expname].Settings # settings data
                if trial not in settings.Data[0][toxname]['trial']:
                    settings.Data[0][toxname]['trial'].append(trial)

                # add filtered_data to the data
                data.loc[d.index, toxname] = filtered_data

        def to_deg(self, xname, trial, expname, events_start=None,
                   ref_time='time', events_SSACC='SSACC', events_ESACC='ESACC',
                   ref_time_sacc='time', before_sacc=5, after_sacc=15,
                   toxname=None, return_=True, **arg):

            '''
            Transforms ``xname`` (the position of the eye in pixels of the
            ``trial``) into degrees.

            With the parameter ``return_=True`` this function returns the
            calculated data and allows you to test the different parameters.

            Parameters
            ----------
            xname: str
                Name of the data to be transformed
            trial: int
                Number of the trial to be transformed
            expname: str
                Name of the particular experience to be transformed

            events_start: str, or None (default None)
                Event marking the start of the trial
            ref_time: str (default 'time')
                Name of the reference time for events ``events_start``

            events_SSACC: 'str' (default 'SSACC')
                Event marking the start of the saccades in trial
            events_ESACC: 'str' (default 'ESACC')
                Event marking the end of the saccades in trial
            ref_time_sacc: str (default 'time')
                Name of the reference time for events ``events_SSACC`` and
                ``events_ESACC``

            before_sacc: int (default 5)
                Time to delete before saccades (ms)
            after_sacc: int (default 15)
                Time to delete after saccades (ms)

            toxname: str (default None)
                Name of the data to be saved

            return_: bool (default True)
                If ``True`` returns the value,
                else saves it in ``data``

            Returns
            -------
            data_to_deg: list
                the calculated data if ``return_=True``
            '''

            check_param(self, PreProcessing.Trial, 'ANEMO.PreProcessing',
                        expname, trial)

            data = self._.data[expname].Data # data
            trial_data = data[data.trial==trial] # trial data

            events = self._.data[expname].Events # events data
            trial_events = events[events.trial==trial] # trial events

            settings = self._.data[expname].Settings # settings data


            if not before_sacc: before_sacc = 0
            if not after_sacc:  after_sacc  = 0

            #------------------------------------------------------------------
            # converts the time variables into the sampling frequency
            #------------------------------------------------------------------
            if not 'SamplingFrequency' in settings.keys():
                raise SettingsError('SamplingFrequency', expname)

            SamplingFrequency = settings.SamplingFrequency.values[0]
            before_sacc = (before_sacc/1000) * SamplingFrequency
            after_sacc = (after_sacc/1000) * SamplingFrequency

            #------------------------------------------------------------------
            # determines when the gaze should be in the centre of the screen.
            #------------------------------------------------------------------
            time = trial_data[ref_time].values # trial data time

            if events_start: t0 = trial_events[events_start].values[0]
            else:            t0 = time[0]

            t0_data = int(t0-time[0]) # time when the gaze should be in
                                      # the centre of the screen.

            #------------------------------------------------------------------
            # loop to know if there is a saccade at time t0
            #  if saccade at this time, this time will be at the beginning or
            #  end of the saccade
            #------------------------------------------------------------------
            ssacc = trial_events[events_SSACC].values[0]
            esacc = trial_events[events_ESACC].values[0]
            tsacc = trial_data[ref_time_sacc].values

            start_sacc = list_event(ssacc, - before_sacc - tsacc[0])
            end_sacc   = list_event(esacc, + after_sacc - tsacc[0])

            for s, e in zip(start_sacc, end_sacc):
                if s and not np.isnan(s) and e and not np.isnan(e):
                    if (t0-time[0]) in np.arange(s, e):
                        if abs((t0-time[0]) - s) <= abs((t0-time[0]) - e):
                            t0_data = int(s) -1
                        else :
                            t0_data = int(e) +1

            #------------------------------------------------------------------
            # converting data to degrees
            #------------------------------------------------------------------
            data_x = trial_data[xname].values # trial data x

            p0 = data_x[t0_data] # position of the gaze at time t0
            if np.isnan(p0):
                while np.isnan(p0):
                    t0_data += 1
                    # test if there is no saccade immediately after the blink
                    if abs(data_x[t0_data]-data_x[t0_data+10]) < 10:
                        p0 = data_x[t0_data]

            data_to_deg = (data_x-p0) / settings.px_per_deg.values[0]
            #------------------------------------------------------------------

            if return_:
                return data_to_deg

            else:
                # add trial to the settings data
                if trial not in settings.Data[0][toxname]['trial']:
                    settings.Data[0][toxname]['trial'].append(trial)

                # add data_to_deg to the data
                data.loc[(data.trial==trial), toxname] = data_to_deg

        def to_velocity_deg(self, xname, trial, expname, toxname=None,
                            return_=True, **arg):

            '''
            Transforms ``xname`` (the position of the eye in pixels of the
            ``trial``) into velocity (°/s).

            With the parameter ``return_=True`` this function returns the
            calculated data and allows you to test the different parameters.

            Parameters
            ----------
            xname: str
                Name of the data to be transformed
            trial: int
                Number of the trial to be transformed
            expname: str
                Name of the particular experience to be transformed

            toxname: str (default None)
                Name of the data to be saved

            return_: bool (default True)
                If ``True`` returns the value,
                else saves it in ``data``

            Returns
            -------
            data_to_v_deg: list
                the calculated data if ``return_=True``
            '''


            check_param(self, PreProcessing.Trial, 'ANEMO.PreProcessing',
                        expname, trial)

            data = self._.data[expname].Data # data
            trial_data = data[data.trial==trial] # trial data

            settings = self._.data[expname].Settings # settings data
            if not 'SamplingFrequency' in settings.keys():
                raise SettingsError('SamplingFrequency', expname)
            if not 'px_per_deg' in settings.keys():
                raise SettingsError('px_per_deg', expname)


            #------------------------------------------------------------------
            # converting data to velocity (°/s)
            #------------------------------------------------------------------
            data_x = trial_data[xname].values # trial data x
            data_to_v_deg = np.gradient(data_x)
            data_to_v_deg *= (1/settings.px_per_deg.values[0])
            data_to_v_deg *= settings.SamplingFrequency.values[0]
            #------------------------------------------------------------------

            if return_:
                return data_to_v_deg

            else:
                # add trial to the settings data
                if trial not in settings.Data[0][toxname]['trial']:
                    settings.Data[0][toxname]['trial'].append(trial)

                # add data_to_velocity_deg to the data
                data.loc[(data.trial==trial), toxname] = data_to_v_deg

        def new_time(self, xname, trial, expname, eventname, toxname=None,
                     return_=True, **arg):

            '''
            Recalculates a time data (``xname``) according to a reference event
            (``eventname``).

            With the parameter ``return_=True`` this function returns the
            calculated data and allows you to test the different parameters.

            Parameters
            ----------
            xname: str
                Name of the data to be transformed
            trial: int
                Number of the trial to be transformed
            expname: str
                Name of the particular experience to be transformed

            eventname: str
                Name of the reference event

            toxname: str (default None)
                Name of the data to be saved

            return_: bool (default True)
                If ``True`` returns the value,
                else saves it in ``data``

            Returns
            -------
            new_time: list
                the calculated data if ``return_=True``
            '''

            check_param(self, PreProcessing.Trial, 'ANEMO.PreProcessing',
                        expname, trial)

            data = self._.data[expname].Data # data
            trial_data = data[data.trial==trial] # trial data

            events = self._.data[expname].Events # events data
            trial_events = events[events.trial==trial] # trial events


            new_data = trial_data[xname] - trial_events[eventname].values[0]


            if return_:
                return new_data

            else:
                settings = self._.data[expname].Settings # settings data

                # add trial to the settings data
                if trial not in settings.Data[0][toxname]['trial']:
                    settings.Data[0][toxname]['trial'].append(trial)

                # add new_data to the data
                data.loc[(data.trial==trial), toxname] = new_data

        def idx_events(self, trial, expname, Sevents, Eevents, ref_time='time',
                       add_stime=0, add_etime=0, start_event=None,
                       stop_event=None):

            '''
            Returns event indexes.

            Parameters
            ----------
            trial: int
                Number of the trial to be transformed
            expname: str
                Name of the particular experience to be transformed

            Sevents: str
                Name of the start of the event
            Eevents: str
                Name of the end of the event

            ref_time: str (default 'time')
                Name of the reference time for events ``Sevents``, ``Events``,
                ``start_event`` and ``stop_event``

            add_stime: int (default 0)
                Add time at the start of the event (ms)
            add_etime: int (default 0)
                Add time at the end of the event (ms)
            start_event: str, or None (default None)
                Name of the event marking the start of the search
            stop_event: str, or None (default None)
                Name of the event marking the end of the search

            Returns
            -------
            idx
                event indexes
            '''

            check_param(self, PreProcessing.Trial, 'ANEMO.PreProcessing',
                        expname, trial)

            data = self._.data[expname].Data # data
            trial_data = data[data.trial==trial] # trial data

            events = self._.data[expname].Events # events data
            trial_events = events[events.trial==trial] # trial events

            settings = self._.data[expname].Settings # settings data

            #------------------------------------------------------------------
            # converts the time variables into the sampling frequency
            #------------------------------------------------------------------
            if not 'SamplingFrequency' in settings.keys():
                raise SettingsError('SamplingFrequency', expname)

            SamplingFrequency = settings.SamplingFrequency.values[0]
            add_stime = (add_stime/1000) * SamplingFrequency
            add_etime = (add_etime/1000) * SamplingFrequency
            #------------------------------------------------------------------

            if start_event:
                start_time = trial_events[start_event].values[0]
            else:
                start_time = trial_data[ref_time].values[0]

            if stop_event:
                end_time = trial_events[stop_event].values[0]
            else:
                end_time = trial_data[ref_time].values[-1]

            Start = list_event(trial_events[Sevents].values[0], add_stime)
            End = list_event(trial_events[Eevents].values[0], add_etime)

            idx = np.array([False]*len(trial_data))
            for e in range(len(Start)):
                if Start[e]:
                    if Start[e]>=start_time:
                        if Start[e]<=end_time:
                            idx |= (trial_data[ref_time]>=Start[e]) & \
                                   (trial_data[ref_time]<=End[e])

            return idx

        def remove_events(self, xname, trial, expname, Sevents, Eevents,
                          ref_time='time', add_stime=0, add_etime=0,
                          start_event=None, stop_event=None, toxname=None,
                          return_=True, **arg):

            '''
            Removes the ``xname`` data between ``Sevents`` and ``Eevents``
            from the ``trial``.

            With the parameter ``return_=True`` this function returns the
            calculated data and allows you to test the different parameters.

            Parameters
            ----------
            xname: str
                Name of the data to be transformed
            trial: int
                Number of the trial to be transformed
            expname: str
                Name of the particular experience to be transformed

            Sevents: str
                Name of the start of the event
            Eevents: str
                Name of the end of the event

            ref_time: str (default 'time')
                Name of the reference time for events ``Sevents``, ``Events``,
                ``start_event`` and ``stop_event``

            add_stime: int (default 0)
                Add time at the start of the event (ms)
            add_etime: int (default 0)
                Add time at the end of the event (ms)
            start_event: str, or None (default None)
                Name of the event marking the start of the search
            stop_event: str, or None (default None)
                Name of the event marking the end of the search

            toxname: str (default None)
                Name of the data to be saved

            return_: bool (default True)
                If ``True`` returns the value,
                else saves it in ``data``

            Returns
            -------
            data_remove: list
                the calculated data if ``return_=True``
            '''

            check_param(self, PreProcessing.Trial, 'ANEMO.PreProcessing',
                        expname, trial)

            data = self._.data[expname].Data # data

            trial_data = data[data.trial==trial] # trial data
            trial_data[toxname] = trial_data[xname]

            idx = PreProcessing.Trial.idx_events(self, trial, expname,
                                                 Sevents, Eevents, ref_time,
                                                 add_stime, add_etime,
                                                 start_event, stop_event)

            if idx.any()!=False:
                trial_data.loc[idx, toxname] = None

            new_data = trial_data[toxname].values

            if return_:
                return new_data

            else:
                settings = self._.data[expname].Settings # settings data

                # add trial to the settings data
                if trial not in settings.Data[0][toxname]['trial']:
                    settings.Data[0][toxname]['trial'].append(trial)

                # add new_data to the data
                data.loc[(data.trial==trial), toxname] = new_data

        def extract_events(self, xname, trial, expname, Sevents, Eevents,
                           ref_time='time', add_stime=0, add_etime=0,
                           start_event=None, stop_event=None, toxname=None,
                           return_=True, **arg):

            '''
            Extracts the ``xname`` data between ``Sevents`` and ``Eevents``
            from the ``trial``.

            With the parameter ``return_=True`` this function returns the
            calculated data and allows you to test the different parameters.

            Parameters
            ----------
            xname: str
                Name of the data to be transformed
            trial: int
                Number of the trial to be transformed
            expname: str
                Name of the particular experience to be transformed

            Sevents: str
                Name of the start of the event
            Eevents: str
                Name of the end of the event

            ref_time: str (default 'time')
                Name of the reference time for events ``Sevents``, ``Events``,
                ``start_event`` and ``stop_event``

            add_stime: int (default 0)
                Add time at the start of the event (ms)
            add_etime: int (default 0)
                Add time at the end of the event (ms)
            start_event: str, or None (default None)
                Name of the event marking the start of the search
            stop_event: str, or None (default None)
                Name of the event marking the end of the search

            toxname: str (default None)
                Name of the data to be saved

            return_: bool (default True)
                If ``True`` returns the value,
                else saves it in ``data``

            Returns
            -------
            data_extract: list
                the calculated data if ``return_=True``
            '''

            check_param(self, PreProcessing.Trial, 'ANEMO.PreProcessing',
                        expname, trial)

            data = self._.data[expname].Data # data

            trial_data = data[data.trial==trial] # trial data
            trial_data[toxname] = None

            idx = PreProcessing.Trial.idx_events(self, trial, expname,
                                                 Sevents, Eevents, ref_time,
                                                 add_stime, add_etime,
                                                 start_event, stop_event)

            if idx.any()!=False:
                trial_data.loc[idx, toxname] = trial_data.loc[idx, xname]

            new_data = trial_data[toxname].values

            if return_:
                return new_data

            else:
                settings = self._.data[expname].Settings # settings data

                # add trial to the settings data
                if trial not in settings.Data[0][toxname]['trial']:
                    settings.Data[0][toxname]['trial'].append(trial)

                # add new_data to the data
                data.loc[(data.trial==trial), toxname] = new_data

    class Data:

        '''
        Allows you to perform PreProcessings on a set of data.
        '''

        def __init__(self, _):

            self._ = _

        def filter(self, xname, order_filter=2, type_filter='lowpass',
                   cutoff=30, toxname=None, expnames=None, trials=None,
                   recalculate=False):

            '''
            Filters the ``xname`` data and saves it in ``data`` by applying a
            digital and analogue `Butterworth filter <https://docs.scipy.org/
            doc/scipy/reference/generated/scipy.signal.butter.html>`_ twice,
            once forward and once backward using the `scipy.signal.filtfilt
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal
            .filtfilt.html#scipy.signal.filtfilt>`_ function.

            Allows you to perform this transformation on a data set.

            Parameters
            ----------
            xname: str
                Name of the data to be transformed

            order_filter: int (default 2)
                The order of the filter.
            type_filter: str (default 'lowpass')
                The type of filter. Can be ``'lowpass'``, ``'highpass'``,
                ``'bandpass'``, or ``'bandstop'``
            cutoff: int (default 30)
                The critical frequencies for cutoff of filter
            sample_rate: int (default 1000)
                Sampling rate of the recording

            toxname: str (default None)
                Name of the data to be saved,
                if ``None``, ``toxname`` will take the value ``xname+'_f'``
            expnames: str, or list(str)
                Name or list of names of the particular experience to be
                transformed,
                if ``None`` all experiences will be transformed
            trials: int, or list(int), or None (default None)
                Number or list of the trial to be transformed,
                if ``None``, all the trials will be transformed

            recalculate: bool (default False)
                Allows you to indicate if you want to force the calculation on
                the trials already processed
            '''

            if not toxname:
                toxname = xname+'_f'

            #------------------------------------------------------------------
            # argument for the loop applying the function on all trials
            #------------------------------------------------------------------
            arg = dict()
            arg['trials'] = trials # the trials to be transformed
            arg['expnames'] = expnames # the expnames to be transformed
            arg['toxname'] = toxname # name of the data to be saved

            # function to execute
            arg['fct'] = PreProcessing.Trial.filter

            # function name
            arg['name_fct'] = 'PreProcessing.Data.filter'

            # argument of the function
            arg['arg_fct'] = dict()
            arg['arg_fct']['xname'] = xname
            arg['arg_fct']['order_filter'] = order_filter
            arg['arg_fct']['type_filter'] = type_filter
            arg['arg_fct']['cutoff'] = cutoff

            arg['recalculate'] = recalculate
            #------------------------------------------------------------------

            # the loop applying the function on all trials
            self._.Trial._.data = Data_loop(self._.Trial, loop='Data', **arg)

        def to_deg(self, xname, events_start=None, ref_time='time',
                   events_SSACC='SSACC', events_ESACC='ESACC',
                   ref_time_sacc='time', before_sacc=5, after_sacc=15,
                   toxname=None, expnames=None, trials=None,
                   recalculate=False):

            '''
            Transforms ``xname`` (the position of the eye in pixels) into
            degrees and saves it in ``data``.

            Allows you to perform this transformation on a data set.

            Parameters
            ----------
            xname: str
                Name of the data to be transformed

            events_start: str, or None (default None)
                Event marking the start of the trial
            ref_time: str (default 'time')
                Name of the reference time for events ``events_start``

            events_SSACC: 'str' (default 'SSACC')
                Event marking the start of the saccades in trial
            events_ESACC: 'str' (default 'ESACC')
                Event marking the end of the saccades in trial
            ref_time_sacc: str (default 'time')
                Name of the reference time for events ``events_SSACC`` and
                ``events_ESACC``

            before_sacc: int (default 5)
                Time to delete before saccades (ms)
            after_sacc: int (default 15)
                Time to delete after saccades (ms)

            toxname: str (default None)
                Name of the data to be saved,
                if ``None``, ``toxname`` will take the value ``xname+'_deg'``
            expnames: str, or list(str)
                Name or list of names of the particular experience to be
                transformed,
                if ``None`` all experiences will be transformed
            trials: int, or list(int), or None (default None)
                Number or list of the trial to be transformed,
                if ``None``, all the trials will be transformed

            recalculate: bool (default False)
                Allows you to indicate if you want to force the calculation on
                the trials already processed
            '''

            if not toxname:
                toxname = xname+'_deg'

            #------------------------------------------------------------------
            # argument for the loop applying the function on all trials
            #------------------------------------------------------------------
            arg = dict()
            arg['trials'] = trials # the trials to be transformed
            arg['expnames'] = expnames # the expnames to be transformed
            arg['toxname'] = toxname # name of the data to be saved

            # function to execute
            arg['fct'] = PreProcessing.Trial.to_deg

            # function name
            arg['name_fct'] = 'PreProcessing.Data.to_deg'

            # argument of the function
            arg['arg_fct'] = dict()
            arg['arg_fct']['xname'] = xname
            arg['arg_fct']['events_start'] = events_start
            arg['arg_fct']['ref_time'] = ref_time
            arg['arg_fct']['events_SSACC'] = events_SSACC
            arg['arg_fct']['events_ESACC'] = events_ESACC
            arg['arg_fct']['ref_time_sacc'] = ref_time_sacc
            arg['arg_fct']['before_sacc'] = before_sacc
            arg['arg_fct']['after_sacc'] = after_sacc

            arg['recalculate'] = recalculate
            #------------------------------------------------------------------

            # the loop applying the function on all trials
            self._.Trial._.data = Data_loop(self._.Trial, loop='Data', **arg)

        def to_velocity_deg(self, xname, toxname=None, expnames=None,
                            trials=None, recalculate=False):

            '''
            Transforms ``xname`` (the position of the eye in pixels) into
            velocity (°/s) and saves it in ``data``.

            Allows you to perform this transformation on a data set.

            Parameters
            ----------
            xname: str
                Name of the data to be transformed

            toxname: str (default None)
                Name of the data to be saved,
                if ``None``, ``toxname`` will take the value ``xname+'_vdeg'``
            expnames: str, or list(str)
                Name or list of names of the particular experience to be
                transformed,
                if ``None`` all experiences will be transformed
            trials: int, or list(int), or None (default None)
                Number or list of the trial to be transformed,
                if ``None``, all the trials will be transformed

            recalculate: bool (default False)
                Allows you to indicate if you want to force the calculation on
                the trials already processed
            '''

            if not toxname:
                toxname = xname+'_vdeg'

            #------------------------------------------------------------------
            # argument for the loop applying the function on all trials
            #------------------------------------------------------------------
            arg = dict()
            arg['trials'] = trials # the trials to be transformed
            arg['expnames'] = expnames # the expnames to be transformed
            arg['toxname'] = toxname # name of the data to be saved

            # function to execute
            arg['fct'] = PreProcessing.Trial.to_velocity_deg

            # function name
            arg['name_fct'] = 'PreProcessing.Data.to_velocity_deg'

            # argument of the function
            arg['arg_fct'] = dict()
            arg['arg_fct']['xname'] = xname

            arg['recalculate'] = recalculate
            #------------------------------------------------------------------

            # the loop applying the function on all trials
            self._.Trial._.data = Data_loop(self._.Trial, loop='Data', **arg)

        def new_time(self, xname, eventname, toxname=None, expnames=None,
                     trials=None, recalculate=False):

            '''
            Recalculates a time data (``xname``) according to a reference event
            (``eventname``).

            Allows you to perform this transformation on a data set.

            Parameters
            ----------
            xname: str
                Name of the data to be transformed

            eventname: str
                Name of the reference event

            toxname: str (default None)
                Name of the data to be saved,
                if ``None``, ``toxname`` will take the value
                ``xname+'_'+eventname``
            expnames: str, or list(str)
                Name or list of names of the particular experience to be
                transformed,
                if ``None`` all experiences will be transformed
            trials: int, or list(int), or None (default None)
                Number or list of the trial to be transformed,
                if ``None``, all the trials will be transformed

            recalculate: bool (default False)
                Allows you to indicate if you want to force the calculation on
                the trials already processed
            '''

            if not toxname:
                toxname = xname+'_'+eventname

            #------------------------------------------------------------------
            # argument for the loop applying the function on all trials
            #------------------------------------------------------------------
            arg = dict()
            arg['trials'] = trials # the trials to be transformed
            arg['expnames'] = expnames # the expnames to be transformed
            arg['toxname'] = toxname # name of the data to be saved

            # function to execute
            arg['fct'] = PreProcessing.Trial.new_time

            # function name
            arg['name_fct'] = 'PreProcessing.Data.new_time'

            # argument of the function
            arg['arg_fct'] = dict()
            arg['arg_fct']['xname'] = xname
            arg['arg_fct']['eventname'] = eventname

            arg['recalculate'] = recalculate
            #------------------------------------------------------------------

            # the loop applying the function on all trials
            self._.Trial._.data = Data_loop(self._.Trial, loop='Data', **arg)

        def remove_events(self, xname, Sevents, Eevents, ref_time='time',
                          add_stime=0, add_etime=0, start_event=None,
                          stop_event=None, toxname=None, expnames=None,
                          trials=None, recalculate=False):

            '''
            Removes the ``xname`` data between ``Sevents`` and ``Eevents`` from
            the ``trial`` and saves it in ``data``.

            Allows you to perform this transformation on a data set.

            Parameters
            ----------
            xname: str
                Name of the data to be transformed

            Sevents: str
                Name of the start of the event
            Eevents: str
                Name of the end of the event

            ref_time: str (default 'time')
                Name of the reference time for events ``Sevents``, ``Events``,
                ``start_event`` and ``stop_event``

            add_stime: int (default 0)
                Add time at the start of the event (ms)
            add_etime: int (default 0)
                Add time at the end of the event (ms)
            start_event: str, or None (default None)
                Name of the event marking the start of the search
            stop_event: str, or None (default None)
                Name of the event marking the end of the search

            toxname: str (default None)
                Name of the data to be saved,
                if ``None``, ``toxname`` will take the value
                ``xname+'__supp_'+Sevents+'_'+Eevents``
            expnames: str, or list(str)
                Name or list of names of the particular experience to be
                transformed,
                if ``None`` all experiences will be transformed
            trials: int, or list(int), or None (default None)
                Number or list of the trial to be transformed,
                if ``None``, all the trials will be transformed

            recalculate: bool (default False)
                Allows you to indicate if you want to force the calculation on
                the trials already processed
            '''

            if not toxname:
                toxname = xname+'__supp_'+Sevents+'_'+Eevents

            #------------------------------------------------------------------
            # argument for the loop applying the function on all trials
            #------------------------------------------------------------------
            arg = dict()
            arg['trials'] = trials # the trials to be transformed
            arg['expnames'] = expnames # the expnames to be transformed
            arg['toxname'] = toxname # name of the data to be saved

            # function to execute
            arg['fct'] = PreProcessing.Trial.remove_events

            # function name
            arg['name_fct'] = 'PreProcessing.Data.remove_events'

            # argument of the function
            arg['arg_fct'] = dict()
            arg['arg_fct']['xname'] = xname
            arg['arg_fct']['Sevents'] = Sevents
            arg['arg_fct']['Eevents'] = Eevents
            arg['arg_fct']['ref_time'] = ref_time
            arg['arg_fct']['add_stime'] = add_stime
            arg['arg_fct']['add_etime'] = add_etime
            arg['arg_fct']['start_event'] = start_event
            arg['arg_fct']['stop_event'] = stop_event

            arg['recalculate'] = recalculate
            #------------------------------------------------------------------

            # the loop applying the function on all trials
            self._.Trial._.data = Data_loop(self._.Trial, loop='Data', **arg)

        def extract_events(self, xname, Sevents, Eevents, ref_time='time',
                           add_stime=0, add_etime=0, start_event=None,
                           stop_event=None, toxname=None, expnames=None,
                           trials=None, recalculate=False):

            '''
            Extracts the ``xname`` data between ``Sevents`` and ``Eevents`` and
            saves it in ``data``.

            Allows you to perform this transformation on a data set.

            Parameters
            ----------
            xname: str
                Name of the data to be transformed

            Sevents: str
                Name of the start of the event
            Eevents: str
                Name of the end of the event

            ref_time: str (default 'time')
                Name of the reference time for events ``Sevents``, ``Events``,
                ``start_event`` and ``stop_event``

            add_stime: int (default 0)
                Add time at the start of the event (ms)
            add_etime: int (default 0)
                Add time at the end of the event (ms)
            start_event: str, or None (default None)
                Name of the event marking the start of the search
            stop_event: str, or None (default None)
                Name of the event marking the end of the search

            toxname: str (default None)
                Name of the data to be saved,
                if ``None``, ``toxname`` will take the value
                ``xname+'__'+Sevents+'_'+Eevents``
            expnames: str, or list(str)
                Name or list of names of the particular experience to be
                transformed,
                if ``None`` all experiences will be transformed
            trials: int, or list(int), or None (default None)
                Number or list of the trial to be transformed,
                if ``None``, all the trials will be transformed

            recalculate: bool (default False)
                Allows you to indicate if you want to force the calculation on
                the trials already processed
            '''

            if not toxname:
                toxname = xname+'__'+Sevents+'_'+Eevents

            #------------------------------------------------------------------
            # argument for the loop applying the function on all trials
            #------------------------------------------------------------------
            arg = dict()
            arg['trials'] = trials # the trials to be transformed
            arg['expnames'] = expnames # the expnames to be transformed
            arg['toxname'] = toxname # name of the data to be saved

            # function to execute
            arg['fct'] = PreProcessing.Trial.extract_events

            # function name
            arg['name_fct'] = 'PreProcessing.Data.extract_events'

            # argument of the function
            arg['arg_fct'] = dict()
            arg['arg_fct']['xname'] = xname
            arg['arg_fct']['Sevents'] = Sevents
            arg['arg_fct']['Eevents'] = Eevents
            arg['arg_fct']['ref_time'] = ref_time
            arg['arg_fct']['add_stime'] = add_stime
            arg['arg_fct']['add_etime'] = add_etime
            arg['arg_fct']['start_event'] = start_event
            arg['arg_fct']['stop_event'] = stop_event

            arg['recalculate'] = recalculate
            #------------------------------------------------------------------

            # the loop applying the function on all trials
            self._.Trial._.data = Data_loop(self._.Trial, loop='Data', **arg)

