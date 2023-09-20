#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .Error import SettingsError

class GenerateParams:

    '''
    Generates parameters to fit the models defined in ``Model`` class.
    '''

    def SmoothPursuit(data, xname, trial, expname, modelName, stime, etime,
                      xNanName='px_NaN', eventName_TargetOn='TargetOn',
                      eventName_StimulusOff='StimulusOff',
                      eventName_dir_target='dir_target', ref_time='time',
                      init_steady_state=15, min_steady_state=5,
                      max_steady_state=40, init_ramp_pursuit_line=40,
                      min_ramp_pursuit_line=40, max_ramp_pursuit_line=80,
                      init_tau=15, min_tau=13, max_tau=80,
                      init_ramp_pursuit_sigmo=100, min_ramp_pursuit_sigmo=40,
                      max_ramp_pursuit_sigmo=800, do_whitening=False, **opt):

        """
        This function allows to generate automatically the parameters of the
        predefined models in :ref:`Model.SmoothPursuit` in order to
        :ref:`fit <Smooth pursuit fitting function>` them to the data.

        This function returns a python tuple containing two dictionaries:

            - a first dictionary containing for each **parametric variable** of
              the function to be fitted a dictionary containing the following
              keys:
                - ``value`` - corresponding to the initial value that this
                  variable should take
                - ``min`` - corresponding to the minimum value that this
                  variable can take on the fitted, or ``None`` for not defining
                  a value
                - ``max`` - corresponding to the maximum value this variable
                  can take on fitting, or ``None`` for not setting a value
                - ``vary`` - can take three different values:
                    - ``True`` - if you want the value of this variable to vary
                      during fitting
                    - ``False`` - if you want the value of this variable not to
                      vary during the fitting and to remain at the value set by
                      ``value``.
                    - ``'vary'`` - if you want to perform a two-step fit and
                      you want the value of this variable not to vary during
                      the first fit, but to be allowed to vary during the
                      second fit.
            - a second dictionary containing the values of each
              **non-parametric variable** (independent variable) in the
              function to be fitted.

        Parameters
        ----------
        data: dict, or None (default None)
            Dictionary containing for each file the ``Data``, the
            ``Results``, the ``Events``, and the ``Settings``

        xname: str
            Name of the data to be transformed
        trial: int
            Number of the trial to be transformed
        expname: str
            Name of the particular experience to be transformed

        modelName: function
            Model equation
        stime: int, or None (defaut None)
            Start time of the fitting (ms)
        etime: int, or None (default -280)
            End time of the fitting (ms)

        xNanName: str (default 'px_NaN')
            Data name without saccades,
            this variable is particularly important to fit the position
        eventName_TargetOn: str (default 'TargetOn')
            Name of the event marking the appearance of the target
        eventName_StimulusOff: str (default 'StimulusOff')
            Name of the event marking the disappearance of the fixation point
        eventName_dir_target: str (default 'dir_target')
            Name of the event giving the direction of the target
        ref_time: str (default 'time')
            Name of the reference time for events

        init_steady_state: int (defaut 15)
            the initial value of velocity that the steady state reaches during
            the pursuit
        min_steady_state: int (defaut 5)
            minimum velocity that the steady state reaches during the pursuit
        max_steady_state: int (defaut 40)
            maximum velocity that the steady state reaches during the pursuit

        init_ramp_pursuit_line: int (defaut 40)
            the initial value of acceleration of pursuit in seconds for model
            velocity line
        min_ramp_pursuit_line: int (defaut 40)
            minimum acceleration of pursuit in seconds for model velocity line
        max_ramp_pursuit_line: int (defaut 80)
            maximum acceleration of pursuit in seconds for model velocity line

        init_tau: int (defaut 15)
            the initial value of curve of the pursuit for model velocity and
            position
        min_tau: int (defaut 13)
            minimum curve of the pursuit for model velocity and position
        max_tau: int (defaut 80)
            maximum curve of the pursuit for model velocity and position

        init_ramp_pursuit_sigmo: int (defaut 100)
            the initial value of curve of the pursuit for model velocity and
            position
        min_ramp_pursuit_sigmo: int (defaut 40)
            minimum curve of the pursuit for model velocity sigmo
        max_ramp_pursuit_sigmo: int (defaut 800)
            maximum curve of the pursuit for model velocity sigmo

        do_whitening: bool, optional (default False)
            If ``True`` return the whitened fit

        Returns
        -------
        tuple
            tuple containing two dictionaries
        """


        # settings data
        settings = data[expname].Settings # settings data

        #----------------------------------------------------------------------
        # converts the time variables into the sampling frequency
        #----------------------------------------------------------------------
        if not 'SamplingFrequency' in settings.keys():
            raise SettingsError('SamplingFrequency', expname)

        SamplingFrequency = settings.SamplingFrequency.values[0]
        if stime:
            stime = round((stime/1000) * SamplingFrequency)
        if etime:
            etime = round((etime/1000) * SamplingFrequency)
        #----------------------------------------------------------------------


        # data of the trial
        data_trial = data[expname].Data[data[expname].Data.trial==trial]

        # data xname of the trial
        data_ = data_trial[xname].values[stime:etime]
        # data time of the trial
        time = data_trial[ref_time].values[stime:etime]

        # events
        events = data[expname].Events
        # events of the trial
        events_trial = events[events.trial==trial]
        # time of the event marking the appearance of the target in trial
        TargetOn = events_trial[eventName_TargetOn].values[0] - time[0]
        # time of the event marking the disappearance of the fixation point
        StimulusOff = events_trial[eventName_StimulusOff].values[0] - time[0]


        #######################################################################
        # Dictionary containing the values of each parametric variable
        #######################################################################
        params = dict()

        #======================================================================
        # the variable corresponding to the direction of the target
        #  must not vary during the fitting
        #======================================================================
        # Allows you to set the dir_target variable between [-1, 1]
        #----------------------------------------------------------------------
        min_dir = np.min(events[eventName_dir_target].values)
        max_dir = np.max(events[eventName_dir_target].values)

        dir_target = events_trial[eventName_dir_target].values[0]
        dir_target = ((dir_target-min_dir)*2-(max_dir-min_dir))
        dir_target /= (max_dir-min_dir)
        #----------------------------------------------------------------------

        params['dir_target'] = {'value':dir_target, 'min':None, 'max':None,
                                'vary':False}

        #======================================================================
        # the variable corresponding to the time when anticipation begins
        #  takes an initial value at the time of fitting
        #  does not vary during the first fitting
        #  but can vary during the second fitting from a minimum value
        #  to a maximum value
        #======================================================================
        params['start_anti'] = {'value':TargetOn-100, 'min':StimulusOff-200,
                                'max':TargetOn+75, 'vary':'vary'}

        #======================================================================
        # the variable corresponding to the acceleration of anticipation
        #  takes an initial value at the time of fitting
        #  but can vary from a minimum value to a maximum value
        #======================================================================
        params['a_anti'] = {'value':0, 'min':-40., 'max':40., 'vary':True}

        #======================================================================
        # the variable corresponding to the time when the movement begins
        #  takes an initial value at the time of fitting
        #  but can vary from a minimum value to a maximum value
        #======================================================================
        # blocks the latency variable before the first saccade
        #----------------------------------------------------------------------
        start_latency = TargetOn+100 # initial value
        max_latency = None # maximum value

        # loop to lock max_latency at the beginning of the first saccade after
        # start_latency
        #
        # time where the saccades are located in the data
        saccades = np.argwhere(np.isnan(data_))[:, 0]
        for x in saccades:
            if x>=start_latency:
                # takes the value of the start of the saccade if it comes just
                # after start_latency.
                max_latency=x
                break # stop the loop

        # if there was no saccade just after start_latency then max_latency is
        # the end of the data
        if not max_latency:
            max_latency=len(data_)

        # if start_latency is too close to max_latency
        #  then we modify this variable to allow the fit to have a little more
        #  freedom
        if start_latency>=max_latency-50:
            start_latency = max_latency-150
        if start_latency>250:
            start_latency = TargetOn+100
        #----------------------------------------------------------------------

        params['latency'] = {'value':start_latency, 'min':TargetOn+75,
                             'max':max_latency, 'vary':True}

        #======================================================================
        # the variable corresponding to the steady_state velocity reached
        # during the pursuit
        #  takes an initial value at the time of fitting
        #  but can vary from a minimum value to a maximum value
        #======================================================================
        params['steady_state'] = {'value':init_steady_state,
                                  'min':min_steady_state,
                                  'max':max_steady_state,
                                  'vary':True}

        #======================================================================
        # the variable corresponding to the sampling frequency
        #======================================================================
        params['SamplingFrequency'] = {'value':SamplingFrequency,
                                       'min':None, 'max':None, 'vary':False }

        #======================================================================
        # the variable corresponding to the whitened function
        #  must not vary during the fitting
        #======================================================================
        params['do_whitening'] = {'value':do_whitening, 'min':None, 'max':None,
                                  'vary':False }


        #######################################################################
        # Dictionary containing the values of each non-parametric variable
        #######################################################################
        independent_vars = dict()

        #======================================================================
        # the variable corresponding to the time of the function (x)
        #  takes as value an arange list of the same length as the data
        #======================================================================
        independent_vars['x'] = np.arange(len(time))

        if modelName=='velocity_line':
            #==================================================================
            # the variable corresponding to the acceleration of pursuit
            #  takes an initial value at the time of fitting
            #  does not vary during the first fitting
            #  but can vary during the second fitting from a minimum value
            #  to a maximum value
            #==================================================================
            params['ramp_pursuit'] = {'value':init_ramp_pursuit_line,
                                      'min':min_ramp_pursuit_line,
                                      'max':max_ramp_pursuit_line,
                                      'vary':'vary'}

        elif modelName=='velocity':
            #==================================================================
            # the variable corresponding to the curve of the pursuit
            #  takes an initial value at the time of fitting
            #  does not vary during the first fitting
            #  but can vary during the second fitting from a minimum value
            #  to a maximum value
            #==================================================================
            params['tau'] = {'value':init_tau, 'min':min_tau, 'max':max_tau,
                             'vary':'vary'}

        elif modelName=='velocity_sigmo':
            #==================================================================
            # the variable corresponding to the acceleration of pursuit
            #  takes an initial value at the time of fitting
            #  does not vary during the first fitting
            #  but can vary during the second fitting from a minimum value
            #  to a maximum value
            #==================================================================
            params['ramp_pursuit'] = {'value':init_ramp_pursuit_sigmo,
                                      'min':min_ramp_pursuit_sigmo,
                                      'max':max_ramp_pursuit_sigmo,
                                      'vary':'vary'}

        elif modelName=='position':
            #==================================================================
            # the variable corresponding to the curve of the pursuit
            #  takes an initial value at the time of fitting
            #  does not vary during the first fitting
            #  but can vary during the second fitting from a minimum value
            #  to a maximum value
            #==================================================================
            params['tau'] = {'value':init_tau, 'min':min_tau, 'max':max_tau,
                             'vary':'vary'}

            #==================================================================
            # the variable corresponding to the unsaccade data, containing NaN
            #  instead of saccades (x_nan) takes as value data
            #==================================================================
            independent_vars['x_nan']=data_trial[xNanName].values[stime:etime]

        return params, independent_vars

    def saccade(data, xname, trial, expname, stime, etime, do_whitening=False,
                **opt):

        """
        This function allows to generate automatically the parameters of the
        predefined model in :ref:`Model.saccade` in order to
        :ref:`fit <Saccades fitting function>` them to the data.

        This function returns a python tuple containing two dictionaries:

            - a first dictionary containing for each **parametric variable** of
              the function to be fitted a dictionary containing the following
              keys:
                - ``value`` - corresponding to the initial value that this
                  variable should take
                - ``min`` - corresponding to the minimum value that this
                  variable can take on the fitted, or ``None`` for not defining
                  a value
                - ``max`` - corresponding to the maximum value this variable
                  can take on fitting, or ``None`` for not setting a value
                - ``vary`` - can take three different values:
                    - ``True`` - if you want the value of this variable to vary
                      during fitting
                    - ``False`` - if you want the value of this variable not to
                      vary during the fitting and to remain at the value set by
                      ``value``.
                    - ``'vary'`` - if you want to perform a two-step fit and
                      you want the value of this variable not to vary during
                      the first fit, but to be allowed to vary during the
                      second fit.
            - a second dictionary containing the values of each
              **non-parametric variable** (independent variable) in the
              function to be fitted.

        Parameters
        ----------
        data: dict, or None (default None)
            Dictionary containing for each file the ``Data``, the
            ``Results``, the ``Events``, and the ``Settings``

        xname: str
            Name of the data to be transformed
        trial: int
            Number of the trial to be transformed
        expname: str
            Name of the particular experience to be transformed

        stime: int, or None (defaut None)
            Start time of the fitting (ms)
        etime: int, or None (default -280)
            End time of the fitting (ms)

        do_whitening: bool, optional (default False)
            If ``True`` return the whitened fit

        Returns
        -------
        tuple
            tuple containing two dictionaries
        """

        # settings data
        settings = data[expname].Settings # settings data

        #----------------------------------------------------------------------
        # converts the time variables into the sampling frequency
        #----------------------------------------------------------------------
        if not 'SamplingFrequency' in settings.keys():
            raise SettingsError('SamplingFrequency', expname)

        SamplingFrequency = settings.SamplingFrequency.values[0]
        if stime:
            stime = round((stime/1000) * SamplingFrequency)
        if etime:
            etime = round((etime/1000) * SamplingFrequency)
        #----------------------------------------------------------------------

        # data of the trial
        data_trial = data[expname].Data[data[expname].Data.trial==trial]

        # data of the saccade
        data_ = data_trial[xname].values[stime:etime]

        #######################################################################
        # Dictionary containing the values of each parametric variable
        #######################################################################
        params = dict()

        #======================================================================
        # the variable corresponding to the initial position of the beginning
        # of the saccade in deg
        #  takes an initial value at the time of fitting
        #  does not vary during the first fitting
        #  but can vary during the second fitting from a minimum value
        #  to a maximum value
        #======================================================================
        # initial position of the eye at the beginning of the saccade
        init_position_eye = data_[0]
        params['x_0'] = {'value':init_position_eye,
                         'min':init_position_eye-0.1,
                         'max':init_position_eye+0.1,
                         'vary':'vary'}

        #======================================================================
        # the variable corresponding to the curvature of the saccade
        #  takes an initial value at the time of fitting
        #  but can vary from a minimum value to a maximum value
        #======================================================================
        params['tau'] = {'value':13., 'min':5., 'max':40., 'vary':True}

        #======================================================================
        # the variable corresponding to the time of the beginning of the first
        # curvature after x_0 in ms
        #  takes an initial value at the time of fitting
        #  but can vary from a minimum value to a maximum value
        #======================================================================
        params['T0'] = {'value':0., 'min':-15, 'max':10, 'vary':True}

        #======================================================================
        # the variable corresponding to the maximum time of the first curvature
        # after T0 in ms
        #  takes an initial value at the time of fitting
        #  but can vary from a minimum value to a maximum value
        #======================================================================
        # if the saccade is too small then max_t1 will take as value 15ms
        #  otherwise it will take as value the saccade time - 10ms
        if (len(data_)-10.)<=10.: max_t1 = 15.
        else:                     max_t1 = len(data_)-10.

        params['t1'] = {'value':15., 'min':10., 'max':max_t1, 'vary':True}

        #======================================================================
        # the variable corresponding to the time of the maximum of the second
        # curvature after t1 in ms
        #  takes an initial value at the time of fitting
        #  does not vary during the first fitting
        #  but can vary during the second fitting from a minimum value
        #  to a maximum value
        #======================================================================
        # if the saccade is too small then max_t2 will take as value 12ms
        #  otherwise it will take as value the saccade time - 10ms
        if (len(data_)-10.)<=10.: max_t2 = 12.
        else:                     max_t2 = len(data_)-10.

        params['t2'] = {'value':12., 'min':10., 'max':max_t2, 'vary':'vary'}

        #======================================================================
        # the variable corresponding to the time of the end of the second
        # curvature after t2 in ms
        #  takes an initial value at the time of fitting
        #  does not vary during the first fitting
        #  but can vary during the second fitting from a minimum value
        #  to a maximum value
        #======================================================================
        params['tr'] = {'value':1., 'min':0., 'max':15., 'vary':'vary'}

        #======================================================================
        # the variable corresponding to the maximum of the first curvature
        # in deg
        #  takes an initial value at the time of fitting
        #  but can vary from a minimum value to a maximum value
        #======================================================================
        params['x1'] = {'value':2., 'min':-5., 'max':5., 'vary':True}

        #======================================================================
        # the variable corresponding to the maximum of the second curvature
        # in deg
        #  takes an initial value at the time of fitting
        #  does not vary during the first fitting
        #  but can vary during the second fitting from a minimum value
        #  to a maximum value
        #======================================================================
        params['x2'] = {'value':1., 'min':-5., 'max':5., 'vary':'vary'}

        #======================================================================
        # the variable corresponding to the sampling frequency
        #======================================================================
        params['SamplingFrequency'] = {'value':SamplingFrequency,
                                       'min':None, 'max':None, 'vary':False }

        #======================================================================
        # the variable corresponding to the whitened function
        #  must not vary during the fitting
        #======================================================================
        params['do_whitening'] = {'value':do_whitening, 'min':None, 'max':None,
                                  'vary':False }

        #######################################################################
        # Dictionary containing the values of each non-parametric variable
        #######################################################################
        independent_vars = dict()

        #======================================================================
        # the variable corresponding to the time of the function (x)
        #  takes as value an arange list of the same length as the data
        #======================================================================
        independent_vars['x'] = np.arange(len(data_))


        return params, independent_vars
