#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .various_functions import *
from .Init import *
from .Data_loop import *
from .GenerateParams import *
from .Error import SettingsError, ParamsError

import numpy as np

class ProcessingSmoothPursuit:

    '''
    ``ProcessingSmoothPursuit`` allows you to apply certain calculations
    to the data in order to extract parameters, including the latency of
    pursuite, the anticipation speed or the steady state speed.

        - Use ``ProcessingSmoothPursuit.Trial`` to test on a trial the
          different parameters of the functions present in
          ``ProcessingSmoothPursuit`` in order to adjust them as well as
          possible

        - Once the right parameters are found, you can use
          ``ProcessingSmoothPursuit.Data`` to apply the function to a set of
          data and save it.

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
        Allows you to perform smooth pursuit-related processing of eye data
        from a trial.

        Parameters
        ----------
        data: dict, or None (default None)
            Dictionary containing for each file the ``Data``, the ``Results``,
            the ``Events``, and the ``Settings``
        '''

        def __init__(self, _):


            self._ = _
            self._.data = Data.open(_.dirpath, _.sub, _.ses, _.acq,  _.run,
                                    _.RawData)

            self.classical_method = self.classical_method(_)

        class classical_method:

            '''
            Allows you to extract some smooth pursuit parameters of eye data
            from trial using "classical methods".
            '''

            def __init__(self, _):

                self._ = _

            def anticipation(self, xname, trial, expname, add_stime=-50,
                             add_etime=50, eventName_TargetOn='TargetOn',
                             ref_time='time', toxname=None, return_=True,
                             **arg):

                '''
                Calculates the velocity of the anticipation during the smooth
                pursuit of the trial.
                This velocity is the average of the velocity of the eye in a
                100ms window around the target appearance.

                With the parameter ``return_=True`` this function returns
                the calculated data and allows you to test the different
                parameters.

                Parameters
                ----------
                xname: str
                    Name of the data to be transformed
                trial: int
                    Number of the trial to be transformed
                expname: str
                    Name of the particular experience to be transformed

                add_stime: int (default -50)
                    Add time at the start of the event (ms)
                add_etime: int (default 50)
                    Add time at the end of the event (ms)

                eventName_TargetOn: str (default 'TargetOn')
                    Name of the event marking the appearance of the target
                ref_time: str (default 'time')
                    Name of the reference time for event ``eventName_TargetOn``

                toxname: str (default None)
                    Name of the data to be saved

                return_: bool (default True)
                    If ``True`` returns the value,
                    else saves it in ``results``

                Returns
                -------
                anticipation: float
                    the calculated data if ``return_=True``
                '''

                check_param(self,
                            ProcessingSmoothPursuit.Trial.classical_method,
                            'ANEMO.ProcessingSmoothPursuit', expname, trial)

                data = self._.data[expname].Data # data

                events = self._.data[expname].Events # events data
                trial_events = events[events.trial==trial] # trial events

                settings = self._.data[expname].Settings # settings data
                #--------------------------------------------------------------
                # converts the time variables into the sampling frequency
                #--------------------------------------------------------------
                if not 'SamplingFrequency' in settings.keys():
                    raise SettingsError('SamplingFrequency', expname)

                SamplingFrequency = settings.SamplingFrequency.values[0]
                add_stime = (add_stime/1000) * SamplingFrequency
                add_etime = (add_etime/1000) * SamplingFrequency
                #--------------------------------------------------------------

                # start of the time for the average
                t_start = trial_events[eventName_TargetOn].values[0]+add_stime
                # end of the time for the average
                t_end = trial_events[eventName_TargetOn].values[0]+add_etime

                #--------------------------------------------------------------
                # calculation of the anticipation
                #--------------------------------------------------------------
                # data for time in anticipation
                data_anti = data[(data[ref_time]>=t_start) & \
                                 (data[ref_time]<=t_end)]
                # data x for time in anticipation
                data_anti_x = data_anti[xname]

                anticipation = np.nanmean(data_anti_x)
                #--------------------------------------------------------------

                if return_:
                    return anticipation

                else:
                    # add trial to the settings data
                    # settings data
                    settings = self._.data[expname].Settings
                    if trial not in settings.Results[0][toxname]['trial']:
                        settings.Results[0][toxname]['trial'].append(trial)

                    # add anticipation to the results data
                    results = self._.data[expname].Results # results data
                    results.loc[(results.trial==trial), toxname] = anticipation

            def latency(self, xname, trial, expname, w1=300, w2=50, off=50,
                        crit=0.17, eventName_TargetOn='TargetOn',
                        ref_time_event='time', ref_time_latency='time',
                        toxname=None, return_=True, **arg):

                '''
                Calculates the pursuit latency during the smooth pursuit of the
                trial.
                This latency corresponds to the interception of two regression
                lines.

                This function extracts different parameters:
                    - ``t_0`` - corresponds to the start time of the first
                      regression line
                    - ``slope1`` - corresponds to the slope of the first
                      regression line
                    - ``intercept1`` - corresponds to the intercept of the
                      first regression line
                    - ``slope2`` - corresponds to the slope of the second
                      regression line
                    - ``intercept2`` - corresponds to the intercept of the
                      second regression line
                    - ``latency`` - corresponds to the intercept of the two
                      regression lines

                With the parameter ``return_=True`` this function returns
                the calculated data and allows you to test the different
                parameters.

                Parameters
                ----------
                xname: str
                    Name of the data to be transformed
                trial: int
                    Number of the trial to be transformed
                expname: str
                    Name of the particular experience to be transformed

                w1 : int, optional (default 300)
                    Size of the window 1 in ms
                w2 : int, optional (default 50)
                    Size of the window 2 in ms
                off : int, optional (default 50)
                    Gap between the two windows in ms
                crit : float, optional (default 0.17)
                    Difference criterion between the two linregress detecting
                    if the pursuit begins

                eventName_TargetOn: str (default 'TargetOn')
                    Name of the event marking the appearance of the target
                ref_time_event: str (default 'time')
                    Name of the reference time for event ``eventName_TargetOn``

                ref_time_latency: str (default 'time')
                    Name of the reference time for the latency

                toxname: str (default None)
                    Name of the data to be saved

                return_: bool (default True)
                    If ``True`` returns the value,
                    else saves it in ``results``

                Returns
                -------
                latency: dict
                    the calculated data if ``return_=True``
                '''

                check_param(self,
                            ProcessingSmoothPursuit.Trial.classical_method,
                            'ANEMO.ProcessingSmoothPursuit', expname, trial)

                from scipy import stats

                data = self._.data[expname].Data # data
                trial_data = data[data.trial==trial] # trial data

                events = self._.data[expname].Events # events data
                trial_events = events[events.trial==trial] # trial events

                settings = self._.data[expname].Settings # settings data
                #--------------------------------------------------------------
                # converts the time variables into the sampling frequency
                #--------------------------------------------------------------
                if not 'SamplingFrequency' in settings.keys():
                    raise SettingsError('SamplingFrequency', expname)

                SamplingFrequency = settings.SamplingFrequency.values[0]
                w1 = round((w1/1000) * SamplingFrequency)
                w2 = round((w2/1000) * SamplingFrequency)
                off = round((off/1000) * SamplingFrequency)
                #--------------------------------------------------------------

                time = trial_data[ref_time_event].values # trial data time
                if time != []:

                    # time marking the appearance of the target
                    TargetOn = trial_events[eventName_TargetOn].values[0]
                    TargetOn -= time[0]

                    velocity = trial_data[xname].values # trial data velocity
                    time = time - time[0]

                    #---------------------------------------------------
                    # loop to find the pursuit latency
                    #---------------------------------------------------
                    latency_found = None

                    t = 0
                    while t < (len(time)-(w1+off+w2)-300):

                        # first regression
                        regress1 = stats.linregress(time[t:t+w1],
                                                    velocity[t:t+w1])
                        slope1, intercept1, _, _, _ = regress1

                        # second regression
                        regress2 = stats.linregress(time[t+w1+off:t+w1+off+w2],
                                                velocity[t+w1+off:t+w1+off+w2])
                        slope2, intercept2, _, _, _ = regress2

                        # difference between the first slope and the second
                        diff = abs(slope2) - abs(slope1)

                        if abs(diff) >= crit:

                            # time of the two windows
                            tw = time[t:t+w1+off+w2]
                            timew = np.linspace(np.min(tw), np.max(tw),
                                                len(tw))

                            # first regression line
                            fitLine1 = slope1*timew + intercept1
                            # second regression line
                            fitLine2 = slope2*timew + intercept2

                            # index where the two regression lines cross
                            idxlat = np.argwhere(np.isclose(fitLine1, fitLine2,
                                                         atol=0.1)).reshape(-1)

                            #Latency of the pursuit
                            latency = timew[idxlat]
                            if len(latency)!=0:
                                # checks if the pursuit latency found is above
                                # TargetOn+80ms
                                if latency[0] > (TargetOn+80):
                                    latency_found = True
                                    t_lat = trial_data[ref_time_latency].values
                                    lat = t_lat[round(latency[0])]
                                    s1 = slope1
                                    i1 = intercept1
                                    s2 = slope2
                                    i2= intercept2
                                    t0 = t

                        if latency_found is None:
                            t += 1
                        else:
                            t = (len(time)-(w1+off+w2)-300) +1

                if (latency_found is None) or (len(latency)==0):
                    lat = None
                    s1 = None
                    i1 = None
                    s2 = None
                    i2 = None
                    t0 = None


                if return_:
                    return {'latency':    lat,
                            't0':         t0,
                            'slope1':     s1,
                            'intercept1': i1,
                            'slope2':     s2,
                            'intercept2': i2}

                else:

                    # add trial to the settings data
                    # settings data
                    settings = self._.data[expname].Settings.Results[0]
                    if trial not in settings[toxname]['trial']:
                        settings[toxname]['trial'].append(trial)

                    # add anticipation to the results data
                    results = self._.data[expname].Results # results data
                    results.loc[(results.trial==trial), toxname] = lat
                    for k, v in zip(['__t0', '__slope1', '__intercept1',
                                     '__slope2', '__intercept2'],
                                     [t0, s1, i1, s2, i2]):
                        results.loc[(results.trial==trial), toxname+k] = v

            def steady_state(self, xname, trial, expname, add_stime=400,
                             add_etime=600, eventName_TargetOn='TargetOn',
                             ref_time='time', toxname=None, return_=True,
                             **arg):

                '''
                Calculates the velocity of the steady-state during the smooth
                pursuit of the trial.
                This velocity is the average of the eye velocity during target
                tracking after the target has appeared.

                With the parameter ``return_=True`` this function returns
                the calculated data and allows you to test the different
                parameters.

                Parameters
                ----------
                xname: str
                    Name of the data to be transformed
                trial: int
                    Number of the trial to be transformed
                expname: str
                    Name of the particular experience to be transformed

                add_stime: int (default 400)
                    Add time at the start of the event in ms
                add_etime: int (default 600)
                    Add time at the end of the event in ms

                eventName_TargetOn: str (default 'TargetOn')
                    Name of the event marking the appearance of the target
                ref_time: str (default 'time')
                    Name of the reference time for event ``eventName_TargetOn``

                toxname: str (default None)
                    Name of the data to be saved

                return_: bool (default True)
                    If ``True`` returns the value,
                    else saves it in ``results``

                Returns
                -------
                steady_state: float
                    the calculated data if ``return_=True``
                '''

                check_param(self,
                            ProcessingSmoothPursuit.Trial.classical_method,
                            'ANEMO.ProcessingSmoothPursuit', expname, trial)

                data = self._.data[expname].Data # data

                events = self._.data[expname].Events # events data
                trial_events = events[events.trial==trial] # trial events

                settings = self._.data[expname].Settings # settings data
                #--------------------------------------------------------------
                # converts the time variables into the sampling frequency
                #--------------------------------------------------------------
                if not 'SamplingFrequency' in settings.keys():
                    raise SettingsError('SamplingFrequency', expname)

                SamplingFrequency = settings.SamplingFrequency.values[0]
                add_stime = (add_stime/1000) * SamplingFrequency
                add_etime = (add_etime/1000) * SamplingFrequency
                #--------------------------------------------------------------


                # start of the time for the average
                t_start = trial_events[eventName_TargetOn].values[0]+add_stime
                # end of the time for the average
                t_end = trial_events[eventName_TargetOn].values[0]+add_etime

                #--------------------------------------------------------------
                # calculation of the steady state
                #--------------------------------------------------------------
                # data for time in steady state
                data_steady = data[(data[ref_time]>=t_start ) & \
                                   (data[ref_time]<=t_end)]
                # data x for time in steady state
                data_steady_x = data_steady[xname]

                steady_state = abs(np.nanmean(data_steady_x))
                #--------------------------------------------------------------

                if return_:
                    return steady_state

                else:
                    # add trial to the settings data
                    # settings data
                    settings = self._.data[expname].Settings.Results[0]
                    if trial not in settings[toxname]['trial']:
                        settings[toxname]['trial'].append(trial)

                    # add anticipation to the results data
                    results = self._.data[expname].Results # results data
                    results.loc[(results.trial==trial), toxname] = steady_state


        def Fit(self, xname, trial, expname,  model,
                generate_params=GenerateParams.SmoothPursuit,
                stime=None, etime=-280, step_fit=2,
                arg_generate_params=dict(xNanName='px_NaN',
                                         eventName_TargetOn='TargetOn',
                                         eventName_StimulusOff='StimulusOff',
                                         eventName_dir_target='dir_target',
                                         ref_time='time',
                                         do_whitening=False),
                toxname=None, return_=True, **opt):

            '''
            Allows you to fit the parameters of a model defined by an
            ``model`` to the ``xname`` data of a trial

            With the parameter ``return_=True`` this function returns
            the calculated data and allows you to test the different
            parameters.

            Parameters
            ----------
            xname: str
                Name of the data to be transformed
            trial: int
                Number of the trial to be transformed
            expname: str
                Name of the particular experience to be transformed

            model: function
                Model equation
            generate_params: function (default GenerateParams.SmoothPursuit)
                Function generating parameters to perform the fitting

            stime: int, or None (default None)
                Start time of the fitting in ms
            etime: int, or None (default -280)
                End time of the fitting in ms
            step_fit: int, optional (default 2)
                Number of steps for the fit
            arg_generate_params: dict
                Dictionary containing the parameters for the generate_params
                function,
                its default value is :
                ``dict(xNanName='px_NaN', TargetOn='TargetOn',
                StimulusOff='StimulusOff', dir_target='dir_target',
                ref_time='time', do_whitening=False)``

            toxname: str (default None)
                Name of the data to be saved

            return_: bool (default True)
                If ``True`` returns the value,
                else saves it in ``results`` and ``data``

            Returns
            -------
            fit: dict
                the calculated data if ``return_=True``
            '''

            check_param(self, ProcessingSmoothPursuit.Trial,
                        'ANEMO.ProcessingSmoothPursuit', expname, trial)

            data = self._.data[expname].Data # data
            trial_data = data[data.trial==trial] # trial data

            events = self._.data[expname].Events # events data
            trial_events = events[events.trial==trial] # trial events

            settings = self._.data[expname].Settings # settings data
            results  = self._.data[expname].Results # results data




            def fitting(data, model, params, inde_vars, step_fit):

                #--------------------------------------------------------------
                # checks if the function parameters are correct
                #--------------------------------------------------------------
                import inspect
                arg_fct = inspect.getfullargspec(model).args
                arg = list(params.keys()) + list(inde_vars.keys())
                if not sorted(arg_fct)==sorted(arg):
                    raise ParamsError(arg, arg_fct)
                #--------------------------------------------------------------

                from lmfit import  Model, Parameters

                #--------------------------------------------------------------
                # parameters of the model to be fitted
                #--------------------------------------------------------------
                params_ = Parameters()
                for p in params.keys():

                    if 'expr' in params[p].keys():
                        params_.add(p, expr=params[p]['expr'])

                    else:
                        vary = params[p]['vary']
                        if params[p]['vary']=='vary':
                            if step_fit==1:
                                vary=True
                            elif step_fit==2:
                                vary=False

                        params_.add(p,
                                    value=params[p]['value'],
                                    min=params[p]['min'],
                                    max=params[p]['max'],
                                    vary=vary)

                #--------------------------------------------------------------
                # the model to be fitted
                #--------------------------------------------------------------
                model_ = Model(model, independent_vars=inde_vars.keys())

                #--------------------------------------------------------------
                # fitting the model to the data
                #--------------------------------------------------------------
                if step_fit==1:
                    result = model_.fit(data, params_, nan_policy='omit',
                                        **inde_vars)

                elif step_fit==2:
                    out = model_.fit(data, params_, nan_policy='omit',
                                     **inde_vars)

                    # make the other parameters vary now
                    for p in params.keys():
                        if params[p]['vary']=='vary':
                            out.params[p].set(vary=True)

                    result = model_.fit(data, out.params, method='nelder',
                                        nan_policy='omit', **inde_vars)
                #--------------------------------------------------------------

                return result

            #------------------------------------------------------------------
            # adds parameters to the dictionary containing the parameters for
            # the generate_params function
            #------------------------------------------------------------------
            for arg_name, arg in zip(['data', 'xname', 'trial', 'expname',
                                      'modelName', 'stime', 'etime'],
                                      [self._.data, xname, trial, expname,
                                       model.__name__, stime, etime]):

                if arg_name in generate_params.__code__.co_varnames:
                    arg_generate_params[arg_name] = arg

            #------------------------------------------------------------------
            # converts the time variables into the sampling frequency
            #------------------------------------------------------------------
            if not 'SamplingFrequency' in settings.keys():
                raise SettingsError('SamplingFrequency', expname)

            SamplingFrequency = settings.SamplingFrequency.values[0]
            if stime:
                stime = round((stime/1000) * SamplingFrequency)
            if etime:
                etime = round((etime/1000) * SamplingFrequency)
            #------------------------------------------------------------------

            data_x = trial_data[xname].values # trial data x
            fit = np.zeros_like(data_x)*np.nan
            data_x = data_x[stime:etime] # trial data x for fitting

            if np.all(np.isnan(data_x)):

                print('There is no %s data for the trial %s,'%(xname, trial),
                      'the fitting cannot be done for this trial')
                if return_:
                    return None
            else:

                #--------------------------------------------------------------
                # fitting
                #--------------------------------------------------------------
                params, inde_vars = generate_params(**arg_generate_params)
                r = fitting(data_x, model, params, inde_vars, step_fit)

                #--------------------------------------------------------------
                # results of fitting
                #--------------------------------------------------------------
                # fitted values
                values_fit = r.values
                if etime or stime:
                    fit[stime:etime] = model(**inde_vars, **values_fit)
                else:
                    fit= model(**inde_vars, **values_fit)

                # fit statistics
                FitStatistics = {'nfev': r.nfev, # number of function evals
                                 'chisqr': r.chisqr, # chi-sqr
                                 'redchi': r.redchi, # reduce chi-sqr
                                 'aic': r.aic, # Akaike info crit
                                 'bic': r.bic} # Bayesian info crit
                #--------------------------------------------------------------

                if return_:

                    return_dict = dict()
                    return_dict['values_fit']= values_fit
                    return_dict['FitStatistics'] = FitStatistics
                    return_dict['fit'] = fit

                    return return_dict

                else:

                    # add results of fitting to the results
                    for k, v in zip(['', '_FitStatistics'],
                                    [[values_fit], [FitStatistics]]):
                        results.loc[(results.trial==trial), toxname+k] = v

                    # add trial to the settings data
                    if trial not in settings.Data[0][toxname]['trial']:
                        settings.Data[0][toxname]['trial'].append(trial)

                    # add trial to the settings results
                    if trial not in settings.Results[0][toxname]['trial']:
                        settings.Results[0][toxname]['trial'].append(trial)

                    # add fit to the data
                    data.loc[(data.trial==trial), toxname] = fit

    class Data:

        '''
        Allows you to perform smooth pursuit-related processing on a set of
        data.
        '''

        def __init__(self, _):

            self._ = _
            self.classical_method = self.classical_method(_)

        class classical_method:

            '''
            Allows you to extract some smooth pursuit parameters on a set of
            data using "classical methods".
            '''

            def __init__(self, _):

                self._ = _

            def anticipation(self, xname, add_stime=-50, add_etime=50,
                             eventName_TargetOn='TargetOn', ref_time='time',
                             toxname=None, expnames=None, trials=None,
                             recalculate=False):

                '''
                Calculates the velocity of the anticipation during the smooth
                pursuit and saves it in ``results``.
                This velocity is the average of the velocity of the eye in a
                100ms window around the target appearance.

                Allows you to perform this fonction on a data set.

                Parameters
                ----------
                xname: str
                    Name of the data to be transformed

                add_stime: int (default -50)
                    Add time at the start of the event
                add_etime: int (default 50)
                    Add time at the end of the event

                eventName_TargetOn: str (default 'TargetOn')
                    Name of the event marking the appearance of the target
                ref_time: str (default 'time')
                    Name of the reference time for event ``eventName_TargetOn``

                toxname: str (default None)
                    Name of the data to be saved,
                    if ``None`` ``toxname`` will take the value
                    ``'classical_anticipation'``
                expnames: str, or list(str)
                    Name or list of names of the particular experience to be
                    transformed,
                    if ``None`` all experiences will be transformed
                trials: int, or list(int), or None (default None)
                    Number or list of the trial to be transformed,
                    if ``None`` all the trials will be transformed

                recalculate: bool (default False)
                    Allows you to indicate if you want to force the calculation
                    on the trials already processed
                '''

                if not toxname:
                    toxname = 'classical_anticipation'

                #--------------------------------------------------------------
                # argument for the loop applying the function on all trials
                #--------------------------------------------------------------
                arg = dict()
                arg['trials'] = trials # the trials to be transformed
                arg['expnames'] = expnames # the expnames to be transformed
                arg['toxname'] = toxname # name of the data to be saved

                # function to execute
                arg['fct'] = ProcessingSmoothPursuit.Trial\
                            .classical_method.anticipation

                # function name
                arg['name_fct'] = 'ProcessingSmoothPursuit.Data.' + \
                                  'classical_method.anticipation'

                # argument of the function
                arg['arg_fct'] = dict()
                arg['arg_fct']['xname'] = xname
                arg['arg_fct']['add_stime'] = add_stime
                arg['arg_fct']['add_etime'] = add_etime
                arg['arg_fct']['eventName_TargetOn'] = eventName_TargetOn
                arg['arg_fct']['ref_time'] = ref_time

                arg['recalculate'] = recalculate
                #--------------------------------------------------------------

                # the loop applying the function on all trials
                self._.Trial._.data = Data_loop(self._.Trial.classical_method,
                                                loop='Results', **arg)

            def latency(self, xname, w1=300, w2=50, off=50, crit=0.1,
                        eventName_TargetOn='TargetOn', ref_time_event='time',
                        ref_time_latency='time', toxname=None, expnames=None,
                        trials=None, recalculate=False):

                '''
                Calculates the pursuit latency during the smooth pursuit and
                saves it in ``results``.
                This latency corresponds to the interception of two regression
                lines.

                This function extracts different parameters:
                    - ``t_0`` - corresponds to the start time of the first
                      regression line
                    - ``slope1`` - corresponds to the slope of the first
                      regression line
                    - ``intercept1`` - corresponds to the intercept of the
                      first regression line
                    - ``slope2`` - corresponds to the slope of the second
                      regression line
                    - ``intercept2`` - corresponds to the intercept of the
                      second regression line
                    - ``latency`` - corresponds to the intercept of the two
                      regression lines

                Allows you to perform this fonction on a data set.

                Parameters
                ----------
                xname: str
                    Name of the data to be transformed

                w1 : int, optional (default 300)
                    Size of the window 1 in ms
                w2 : int, optional (default 50)
                    Size of the window 2 in ms
                off : int, optional (default 50)
                    Gap between the two windows in ms
                crit : float, optional (default 0.17)
                    Difference criterion between the two linregress detecting
                    if the pursuit begins

                eventName_TargetOn: str (default 'TargetOn')
                    Name of the event marking the appearance of the target
                ref_time_event: str (default 'time')
                    Name of the reference time for event ``eventName_TargetOn``

                ref_time_latency: str (default 'time')
                    Name of the reference time for the latency

                toxname: str (default None)
                    Name of the data to be saved,
                    if ``None`` ``toxname`` will take the value
                    ``'classical_latency'``
                expnames: str, or list(str)
                    Name or list of names of the particular experience to be
                    transformed,
                    if ``None`` all experiences will be transformed
                trials: int, or list(int), or None (default None)
                    Number or list of the trial to be transformed,
                    if ``None`` all the trials will be transformed

                recalculate: bool (default False)
                    Allows you to indicate if you want to force the calculation
                    on the trials already processed
                '''

                if not toxname:
                    toxname = 'classical_latency'

                #--------------------------------------------------------------
                # argument for the loop applying the function on all trials
                #--------------------------------------------------------------
                arg = dict()
                arg['trials'] = trials # the trials to be transformed
                arg['expnames'] = expnames # the expnames to be transformed
                arg['toxname'] = toxname # name of the data to be saved

                # function to execute
                arg['fct'] = ProcessingSmoothPursuit.Trial\
                             .classical_method.latency

                # function name
                arg['name_fct'] = 'ProcessingSmoothPursuit.Data.' + \
                                  'classical_method.latency'

                # argument of the function
                arg['arg_fct'] = dict()
                arg['arg_fct']['xname'] = xname
                arg['arg_fct']['w1'] = w1
                arg['arg_fct']['w2'] = w2
                arg['arg_fct']['off'] = off
                arg['arg_fct']['crit'] = crit
                arg['arg_fct']['eventName_TargetOn'] = eventName_TargetOn
                arg['arg_fct']['ref_time_event'] = ref_time_event
                arg['arg_fct']['ref_time_latency'] = ref_time_latency

                arg['recalculate'] = recalculate
                #--------------------------------------------------------------

                # the loop applying the function on all trials
                self._.Trial._.data = Data_loop(self._.Trial.classical_method,
                                                loop='Results', **arg)

            def steady_state(self, xname, add_stime=400, add_etime=600,
                             eventName_TargetOn='TargetOn', ref_time='time',
                             toxname=None, expnames=None, trials=None,
                             recalculate=False):

                '''
                Calculates the velocity of the steady-state during the smooth
                pursuit and saves it in ``results``.
                This velocity is the average of the eye velocity during target
                tracking after the target has appeared.

                Allows you to perform this fonction on a data set.

                Parameters
                ----------
                xname: str
                    Name of the data to be transformed

                add_stime: int (default 400)
                    Add time at the start of the event
                add_etime: int (default 600)
                    Add time at the end of the event

                eventName_TargetOn: str (default 'TargetOn')
                    Name of the event marking the appearance of the target
                ref_time: str (default 'time')
                    Name of the reference time for event ``eventName_TargetOn``

                toxname: str (default None)
                    Name of the data to be saved
                    if ``None`` ``toxname`` will take the value
                    ``'classical_steady_state'``
                expnames: str, or list(str)
                    Name or list of names of the particular experience to be
                    transformed,
                    if ``None`` all experiences will be transformed
                trials: int, or list(int), or None (default None)
                    Number or list of the trial to be transformed,
                    if ``None`` all the trials will be transformed

                recalculate: bool (default False)
                    Allows you to indicate if you want to force the calculation
                    on the trials already processed
                '''

                if not toxname:
                    toxname = 'classical_steady_state'

                #--------------------------------------------------------------
                # argument for the loop applying the function on all trials
                #--------------------------------------------------------------
                arg = dict()
                arg['trials'] = trials # the trials to be transformed
                arg['expnames'] = expnames # the expnames to be transformed
                arg['toxname'] = toxname # name of the data to be saved

                # function to execute
                arg['fct'] = ProcessingSmoothPursuit.Trial\
                             .classical_method.steady_state

                # function name
                arg['name_fct'] = 'ProcessingSmoothPursuit.Data.' + \
                                  'classical_method.steady_state'

                # argument of the function
                arg['arg_fct'] = dict()
                arg['arg_fct']['xname'] = xname
                arg['arg_fct']['add_stime'] = add_stime
                arg['arg_fct']['add_etime'] = add_etime
                arg['arg_fct']['eventName_TargetOn'] = eventName_TargetOn
                arg['arg_fct']['ref_time'] = ref_time

                arg['recalculate'] = recalculate
                #--------------------------------------------------------------

                # the loop applying the function on all trials
                self._.Trial._.data = Data_loop(self._.Trial.classical_method,
                                                loop='Results', **arg)

        def Fit(self, xname, model,
                generate_params=GenerateParams.SmoothPursuit,
                stime=None, etime=-280, step_fit=2,
                arg_generate_params=dict(xNanName='px_NaN',
                                         eventName_TargetOn='TargetOn',
                                         eventName_StimulusOff='StimulusOff',
                                         eventName_dir_target='dir_target',
                                         ref_time='time',
                                         do_whitening=False),
                toxname=None, expnames=None, trials=None, recalculate=False):

            '''
            Allows you to fit the parameters of a model defined by an
            ``model`` to the ``xname`` data.

            Parameters
            ----------
            xname: str
                Name of the data to be transformed
            model: function
                Model equation
            generate_params: function (default GenerateParams.SmoothPursuit)
                Function generating parameters to perform the fitting

            stime: int, or None (defaut None)
                Start time of the fitting in ms
            etime: int, or None (default -280)
                End time of the fitting in ms
            step_fit: int, optional (default 2)
                Number of steps for the fit
            arg_generate_params: dict
                Dictionary containing the parameters for the generate_params
                function,
                its default value is :
                ``dict(xNanName='px_NaN', TargetOn='TargetOn',
                StimulusOff='StimulusOff', dir_target='dir_target',
                ref_time='time', do_whitening=False)``

            toxname: str (default None)
                Name of the data to be saved,
                if ``None`` ``toxname`` will take the value
                ``'Fit_'+Name_of_model_equation``
            expnames: str, or list(str)
                Name or list of names of the particular experience to be
                transformed,
                if ``None`` all experiences will be transformed
            trials: int, or list(int), or None (default None)
                Number or list of the trial to be transformed,
                if ``None`` all the trials will be transformed

            recalculate: bool (default False)
                Allows you to indicate if you want to force the calculation on
                the trials already processed
            '''

            if not toxname:
                toxname = 'Fit_%s'%model.__name__

            #------------------------------------------------------------------
            # argument for the loop applying the function on all trials
            #------------------------------------------------------------------
            arg = dict()
            arg['trials'] = trials # the trials to be transformed
            arg['expnames'] = expnames # the expnames to be transformed
            arg['toxname'] = toxname # name of the data to be saved

            # function to execute
            arg['fct'] = ProcessingSmoothPursuit.Trial.Fit

            # function name
            arg['name_fct'] = 'ProcessingSmoothPursuit.Data.Fit'

            # argument of the function
            arg['arg_fct'] = dict()
            arg['arg_fct']['xname'] = xname
            arg['arg_fct']['model'] = model
            arg['arg_fct']['generate_params'] = generate_params
            arg['arg_fct']['stime'] = stime
            arg['arg_fct']['etime'] = etime
            arg['arg_fct']['step_fit'] = step_fit
            arg['arg_fct']['arg_generate_params'] = arg_generate_params

            arg['recalculate'] = recalculate
            #------------------------------------------------------------------

            # the loop applying the function on all trials
            self._.Trial._.data = Data_loop(self._.Trial, loop=None, **arg)
