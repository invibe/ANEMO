#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .various_functions import *
from .Init import *
from .Data_loop import *
from .GenerateParams import *
from .Error import SettingsError, ParamsError


import numpy as np

class ProcessingSaccades:

    '''
    ``ProcessingSaccades`` is used to perform saccade-related processing on eye
    data.

        - Use ``ProcessingSaccades.Trial`` to test on a trial the different
          parameters of the functions present in ``ProcessingSaccades`` in
          order to adjust them as well as possible

        - Once the right parameters are found, you can use
          ``ProcessingSaccades.Data`` to apply the function to a set of data
          and save it.

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
        Allows you to perform saccade-related processing on a eye data from a
        trial.
        '''

        def __init__(self, _):

            self._ = _
            self._.data = Data.open(_.dirpath, _.sub, _.ses, _.acq,  _.run,
                                  _.RawData)

        def detect_misacc(self, vxname, vyname, trial, expname,
                          ref_time='time', threshold=5, mint=5, maxt=100,
                          minstep=30, toxname=None, return_=True, **arg):

            '''
            Detects microsaccades of the eye during the ``trial`` not
            previously detected by your eye-tracker.

            With the parameter ``return_=True`` this function returns the
            calculated data and allows you to test the different parameters.

            Parameters
            ----------
            vxname : str
                Name of the velocity data by degrees in x
            vyname : str
                Name of the velocity data by degrees in y
            trial: int
                Number of the trial to be transformed
            expname: str
                Name of the particular experience to be transformed

            ref_time: str (default 'time')
                Name of the reference time for microsaccades
            threshold : int (default 5)
                Relative velocity threshold
            mint : int (default 5)
                Minimal saccade duration (ms)
            maxt : int (default 100)
                Maximal saccade duration (ms)
            minstep : int (default 30)
                Minimal time interval between two detected saccades (ms)

            toxname: str (default None)
                Name of the data to be saved

            return_: bool (default True)
                If ``True`` returns the value,
                else saves it in ``events``

            Returns
            -------
            MISACC: dict
                the calculated data if ``return_=True``
            '''

            check_param(self, ProcessingSaccades.Trial,
                        'ANEMO.ProcessingSaccades', expname, trial)

            data = self._.data[expname].Data # data
            trial_data = data[data.trial==trial]

            settings = self._.data[expname].Settings # settings data

            #------------------------------------------------------------------
            # converts the time variables into the sampling frequency
            #------------------------------------------------------------------
            if not 'SamplingFrequency' in settings.keys():
                raise SettingsError('SamplingFrequency', expname)
            mint    = (mint/1000)    * settings.SamplingFrequency.values[0]
            maxt    = (maxt/1000)    * settings.SamplingFrequency.values[0]
            minstep = (minstep/1000) * settings.SamplingFrequency.values[0]

            #------------------------------------------------------------------
            # time detection of micro-saccades
            #------------------------------------------------------------------
            vx = np.array(trial_data[vxname].to_list()) # velocity x trial data
            msdx = np.sqrt((np.nanmedian(vx**2))-(np.nanmedian(vx)**2))
            radiusx = threshold*msdx

            vy = np.array(trial_data[vyname].to_list()) # velocity y trial data
            msdy = np.sqrt((np.nanmedian(vy**2))-(np.nanmedian(vy)**2))
            radiusy = threshold*msdy

            test = (vx/radiusx)**2 + (vy/radiusy)**2

            # time of micro-saccades
            time = trial_data[ref_time].values # trial data time
            time_misacc = [time[x] for x in range(len(test)) if test[x]>1]

            #------------------------------------------------------------------
            # test if the misacc are well within the defined time interval
            #------------------------------------------------------------------
            misacc = []
            t = 0
            start = 0
            for i in range(len(time_misacc)-1):

                if time_misacc[i+1]-time_misacc[i] == 1:
                    t = t+1

                else :

                    # end micro-saccade
                    if t>=mint and t<maxt:
                        misacc.append([time_misacc[start], time_misacc[i]])

                    t = 1
                    start = i+1

            # end of the last micro-saccade
            if t>=mint and t<maxt:
                misacc.append([time_misacc[start], time_misacc[-1]])

            #------------------------------------------------------------------
            # test if the different misacc are separated enough otherwise merge
            # them
            #------------------------------------------------------------------
            if len(misacc)>1:

                s = 0
                while s < len(misacc)-1:

                    # temporal separation between onset of saccade s+1 and
                    # offset of saccade s
                    step = misacc[s+1][0]-misacc[s][1]

                    if step < minstep :
                        # the two saccades are fused into one
                        misacc[s][1] = misacc[s+1][1]
                        del(misacc[s+1])
                        s = s-1

                    s = s+1

            #------------------------------------------------------------------
            # retest if the misacc are well within the defined time interval
            #------------------------------------------------------------------
            s = 0
            while s < len(misacc):
                t = misacc[s][1]-misacc[s][0] # duration of sth saccade
                if t >= maxt:
                    del(misacc[s])
                    s = s-1
                s = s+1
            #------------------------------------------------------------------

            if len(misacc)!=0:

                if not toxname: toxname='MISACC'
                Sname = 'S'+toxname
                Ename = 'E'+toxname

                MISACC = {Sname: str(list(np.array(misacc)[:, 0])),
                          Ename: str(list(np.array(misacc)[:, 1]))}

                if return_:
                    return MISACC

                else:
                    # add trial to the settings data
                    if trial not in settings.Events[0][toxname]['trial']:
                        settings.Events[0][toxname]['trial'].append(trial)

                    # add MISSAC to the events data
                    events = self._.data[expname].Events # events data
                    events.loc[(events.trial==trial), Sname] = MISACC[Sname]
                    events.loc[(events.trial==trial), Ename] = MISACC[Ename]

            else:
                if return_:
                    print("No microsacades were detected in the trial", trial)
                    return None


        def Fit(self, xname, trial, expname,  model,
                generate_params=GenerateParams.saccade,
                stime=None, etime=-280, step_fit=2,
                arg_generate_params=dict(do_whitening=False),
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
            generate_params: function (default GenerateParams.saccade)
                Function generating parameters to perform the fitting

            stime: int, or None (default None)
                Start time of the fitting (ms)
            etime: int, or None (default -280)
                End time of the fitting (ms)
            step_fit: int, optional (default 2)
                Number of steps for the fit
            arg_generate_params: dict
                Dictionary containing the parameters for the generate_params
                function,
                its default value is : ``dict(do_whitening=False)``

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

            check_param(self, ProcessingSaccades.Trial,
                        'ANEMO.ProcessingSaccades', expname, trial)

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
            data_x = np.array([x if x else np.nan for x in data_x])
            fit = np.zeros_like(data_x)*np.nan
            if return_:
                return_dict = {'fit':[], 'values_fit':[], 'FitStatistics':[]}

            data_x = data_x[stime:etime] # trial data x for fitting

            ssacc = [t+1 for t in range(0, len(data_x)-1)
                     if np.isnan(data_x[t]) and not np.isnan(data_x[t+1])]
            esacc = [t-1 for t in range(1, len(data_x))
                     if np.isnan(data_x[t]) and not np.isnan(data_x[t-1])]

            for n, (s, e) in enumerate(zip(ssacc, esacc)):

                arg_generate_params['stime']=round((s/SamplingFrequency)*1000)
                arg_generate_params['etime']=round((e/SamplingFrequency)*1000)

                data_s = data_x[s:e] # trial data x of saccade
                #print(len(data_s), len(data_x[s:e]))
                #----------------------------------------------------------
                # fitting
                #----------------------------------------------------------
                params, inde_vars = generate_params(**arg_generate_params)
                r = fitting(data_s, model, params, inde_vars, step_fit)

                #----------------------------------------------------------
                # results of fitting
                #----------------------------------------------------------
                # fitted values
                values_fit = r.values
                fit[range(s, e)] = model(**inde_vars, **values_fit)
                # fit statistics
                FitStatistics = {'nfev': r.nfev, # number of function evals
                                 'chisqr': r.chisqr, # chi-sqr
                                 'redchi': r.redchi, # reduce chi-sqr
                                 'aic': r.aic, # Akaike info crit
                                 'bic': r.bic} # Bayesian info crit
                #----------------------------------------------------------

                if return_:
                    return_dict['values_fit'].append(values_fit)
                    return_dict['FitStatistics'].append(FitStatistics)

                else:
                    n = str(n)
                    # add results of fitting to the results
                    for k, v in zip(['_'+n, '_'+n+'_FitStatistics'],
                                    [[values_fit], [FitStatistics]]):
                        results.loc[(results.trial==trial), toxname+k] = v

            if return_:
                return_dict['fit'] = fit
                return return_dict

            else:
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
        Allows you to perform saccade-related processing on a set of data.
        '''

        def __init__(self, _):

            self._ = _

        def detect_misacc(self, vxname, vyname, ref_time='time',
                          threshold=5, mint=5, maxt=100, minstep=30,
                          toxname=None, expnames=None, trials=None,
                          recalculate=False):

            '''
            Detects microsaccades of the eye not previously detected by your
            eye-tracker and saves it in ``events``.

            Allows you to perform this transformation on a data set.

            Parameters
            ----------
            vxname : str
                Name of the velocity data by degrees in x
            vyname : str
                Name of the velocity data by degrees in y

            ref_time: str (default 'time')
                Name of the reference time for microsaccades
            threshold : int (default 5)
                Relative velocity threshold
            mint : int (default 5)
                Minimal saccade duration (ms)
            maxt : int (default 100)
                Maximal saccade duration (ms)
            minstep : int (default 30)
                Minimal time interval between two detected saccades (ms)

            toxname: str (default None)
                Name of the data to be saved,
                if ``None``, ``toxname`` will take the value ``MISACC``
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
                toxname = 'MISACC'

            #------------------------------------------------------------------
            # argument for the loop applying the function on all trials
            #------------------------------------------------------------------
            arg = dict()
            arg['trials'] = trials # the trials to be transformed
            arg['expnames'] = expnames # the expnames to be transformed
            arg['toxname'] = toxname # name of the data to be saved

            # function to execute
            arg['fct'] = ProcessingSaccades.Trial.detect_misacc

            # function name
            arg['name_fct'] = 'ProcessingSaccades.Data.detect_misacc'

            # argument of the function
            arg['arg_fct'] = dict()
            arg['arg_fct']['vxname'] = vxname
            arg['arg_fct']['vyname'] = vyname
            arg['arg_fct']['ref_time'] = ref_time
            arg['arg_fct']['threshold'] = threshold
            arg['arg_fct']['mint'] = mint
            arg['arg_fct']['maxt'] = maxt
            arg['arg_fct']['minstep'] = minstep

            arg['recalculate'] = recalculate
            #------------------------------------------------------------------

            # the loop applying the function on all trials
            self._.Trial._.data = Data_loop(self._.Trial, loop='Events', **arg)

        def Fit(self, xname, model, generate_params=GenerateParams.saccade,
                stime=None, etime=-280, step_fit=2,
                arg_generate_params=dict(do_whitening=False),
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
            generate_params: function (default GenerateParams.saccade)
                Function generating parameters to perform the fitting

            stime: int, or None (defaut None)
                Start time of the fitting (ms)
            etime: int, or None (default -280)
                End time of the fitting (ms)
            step_fit: int, optional (default 2)
                Number of steps for the fit

            arg_generate_params: dict
                Dictionary containing the parameters for the generate_params
                function,
                its default value is :
                ``dict(do_whitening=False)``

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
            arg['fct'] = ProcessingSaccades.Trial.Fit

            # function name
            arg['name_fct'] = 'ProcessingSaccades.Data.Fit'

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
