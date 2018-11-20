#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

''' Revoir les docstring'''

N_freq = 1301
    

def whitening_filt(N_freq, white_f_0, white_alpha, white_steepness):
    """
    Returns the envelope of the whitening filter.

        then we return a 1/f spectrum based on the assumption that the structure of signals
        is self-similar and thus that the Fourier spectrum scales a priori in 1/f.
        
    """
    freq = np.fft.fftfreq(N_freq, d=1.)
    K = np.abs(freq)**(white_alpha)
    K *= np.exp(-(np.abs(freq)/white_f_0)**white_steepness)
    K /= np.mean(K)
    
    return freq, K


def whitening(position, white_f_0=.4, white_alpha=.5, white_steepness=4):
    """
    Returns the whitened image
    
    /!\ position must not contain Nan

    """
    try :
        N_freq = position.shape[0]
    except AttributeError :
        N_freq = len(position)
    freq, K = whitening_filt(N_freq=N_freq, white_f_0=white_f_0, white_alpha=white_alpha, white_steepness=white_steepness)        
    f_position = np.fft.fft(position)
    return np.real(np.fft.ifft(f_position*K))


class Test(object):
    '''function set used in the code to do tests'''

    def crash_None(name, value, print_crash=None) :
        '''
        Test if value is None, if the program stops and returns print_crash

        Parameters
        ----------
        name : str
            name of the variable
        value : int, float
            value to test
        print_crash : str
            message to return
            by default: "% s is not defined"% name

        Returns
        -------
        value : int
            value of the variable
        or Raise
        '''
        if not print_crash :
            print_crash = "%s is not defined"%name
        if value is None :
            raise ValueError(print_crash)
        else :
            return value

    def test_value(name, dic, value=None, crash=True, print_crash=None) :
        '''
        Test if name is in dic, if not return value data

        Parameters
        ----------
        name : str
            name of the variable
        dic : dict
            dictionary to check
        value : int, float
            new value has given
        crash : bool
            if true if the value of name is none then the program stops and returns print_crash
        print_crash : str
            message to return
            by default: "% s is not defined"% name

        Returns
        -------
        value : int
            value of the variable
        or Raise
        '''
        new_value = name
        try:
            new_value = dic[name]
            return new_value

        except KeyError:
            new_value = value
            return new_value

        finally :
            if crash is True :
                Test.crash_None(name, new_value, print_crash)

    def test_None(var, value) :
        '''
        Test if var is None or nan, if it is the case returns value

        Parameters
        ----------
        var :
            variable to test
        value :
            new value has given

        Returns
        -------
        value
        '''
        if type(var) is float :
            if np.isnan(var):
                return value
        elif var is None :
            return value
        else :
            return var


class ANEMO(object):
    """ docstring for the ANEMO class. """

    def __init__(self, param_exp={}) :

        '''
        Parameters
        ----------
        param_exp : dict
            dictionary containing the parameters of the experiment
        '''
        self.param_exp = param_exp


    def arg(self, data_trial, param_exp=None, dir_target=None, list_events=None, trial=None, block=None):

        '''
        generates a dictionary of the parameters of the trial

        Parameters
        ----------
        data_trial : list
            edf data for a trial recorded by the eyetracker transformed by the read_edf function of the edfreader module

        param_exp : dict
            dictionary containing the parameters of the experiment
        
        dir_target : int
            the direction of the target
            if None goes looking for it in param_exp,
                be : param_exp['dir_target'][block][trial] = 1 ou -1
                be : param_exp['p'][trial, block, 0] = 0 ou 1

        list_events : list
            list of the names of the events of the trial
            by default : ['StimulusOn\n', 'StimulusOff\n', 'TargetOn\n', 'TargetOff\n']
        
        must be defined if dir_target is None :
            trial : int
                number of the trial in the block
            block : int
                block number

        Returns
        -------
        arg : dict
            dictionary of the parameters of the trial
        '''

        list_events = Test.test_None(list_events, value=['StimulusOn\n', 'StimulusOff\n', 'TargetOn\n', 'TargetOff\n'])
        param_exp = Test.test_None(param_exp, value=self.param_exp)

        kwargs = {}

        for events in range(len(data_trial['events']['msg'])) :
            if data_trial['events']['msg'][events][1] == list_events[0] :
                kwargs["StimulusOn"] = data_trial['events']['msg'][events][0]
            if data_trial['events']['msg'][events][1] == list_events[1] :
                kwargs["StimulusOf"] = data_trial['events']['msg'][events][0]
            if data_trial['events']['msg'][events][1] == list_events[2] :
                kwargs["TargetOn"] = data_trial['events']['msg'][events][0]
            if data_trial['events']['msg'][events][1] == list_events[3] :
                kwargs["TargetOff"] = data_trial['events']['msg'][events][0]

        kwargs.update({
                    "data_x": data_trial['x'],
                    "data_y": data_trial['y'],
                    "trackertime":data_trial['trackertime'],
                    "saccades":data_trial['events']['Esac'],
                    "t_0": data_trial['trackertime'][0],
                    })

        if 'px_per_deg' in param_exp.keys():
            kwargs["px_per_deg"] = param_exp['px_per_deg']

        # juste pour figure position
        if 'screen_width_px' in param_exp.keys():
            kwargs["screen_width_px"] = param_exp['screen_width_px']
        if 'screen_height_px' in param_exp.keys():
            kwargs["screen_height_px"] = param_exp['screen_height_px']

        # juste pour figure position
        if 'V_X' in param_exp.keys():
            kwargs["V_X"] = param_exp['V_X']
        if 'RashBass' in param_exp.keys():
            kwargs["RashBass"] = param_exp['RashBass']
        if 'stim_tau' in param_exp.keys():
            kwargs["stim_tau"] = param_exp['stim_tau']

        kwargs["dir_target"] = dir_target
        if dir_target is None :
            try :
                kwargs["dir_target"] = param_exp['dir_target'][block][trial]
            except :
                try :
                    kwargs["dir_target"] = (param_exp['p'][trial, block, 0]*2)-1
                except :
                    pass

        import easydict
        return easydict.EasyDict(kwargs)


    def velocity_deg(self, data_x, px_per_deg=None) :
        '''
        Return the speed of the eye in deg/sec

        Parameters
        ----------
        data_x : ndarray
            x position for the trial recorded by the eyetracker transformed by the read_edf function of the edfreader module
        px_per_deg : float
            number of px per degree for the experiment
                screen_width_deg = 2. * np.arctan((screen_width_cm/2) / viewing_Distance_cm) * 180/np.pi
                px_per_deg = screen_width_px / screen_width_deg
            if None : goes looking for it in param_exp

        Returns
        -------
        gradient_deg : ndarray
            speed of the eye in deg/sec
        '''

        if px_per_deg is None :
            px_per_deg = Test.test_value('px_per_deg', self.param_exp) 

        gradient_x = np.gradient(data_x)
        gradient_deg = gradient_x * 1/px_per_deg * 1000 # gradient in deg/sec

        return gradient_deg

    def detec_misac (self, velocity_x, velocity_y, t_0=0, VFAC=5, mindur=5, maxdur=100, minsep=30):
        '''
        Detection of micro-saccades not detected by eyelink in the data

        Parameters
        ----------
        velocity_x : ndarray
            speed x of the eye in deg/sec
        velocity_y : ndarray
            speed y of the eye in deg/sec

        t_0 : int
            time 0 of the trial
        VFAC : int
            relative velocity threshold
        mindur : int
            minimal saccade duration (ms)
        maxdur : int
            maximal saccade duration (ms)
        minsep : int
            minimal time interval between two detected saccades (ms)

        Returns
        -------
        misaccades : list
            list of lists, each containing [start micro-saccade, end micro-saccade]

        '''

        msdx = np.sqrt((np.nanmedian(velocity_x**2))-((np.nanmedian(velocity_x))**2))
        msdy = np.sqrt((np.nanmedian(velocity_y**2))-((np.nanmedian(velocity_y))**2))

        radiusx = VFAC*msdx
        radiusy = VFAC*msdy

        test = (velocity_x/radiusx)**2 + (velocity_y/radiusy)**2
        index = [x for x in range(len(test)) if test[x] > 1]

        dur = 0
        start_misaccades = 0
        k = 0
        misaccades = []

        for i in range(len(index)-1) :
            if index[i+1]-index[i]==1 :
                dur = dur + 1;
            else :
                if dur >= mindur and dur < maxdur :
                    end_misaccades = i
                    misaccades.append([index[start_misaccades]+t_0, index[end_misaccades]+t_0])
                start_misaccades = i+1
                dur = 1
            i = i + 1

        if len(misaccades) > 1 :
            s=0
            while s < len(misaccades)-1 :
                sep = misaccades[s+1][0]-misaccades[s][1] # temporal separation between onset of saccade s+1 and offset of saccade s
                if sep < minsep :
                    misaccades[s][1] = misaccades[s+1][1] #the two saccades are fused into one
                    del(misaccades[s+1])
                    s=s-1
                s=s+1

        s=0
        while s < len(misaccades) :
            dur = misaccades[s][1]-misaccades[s][0] # duration of sth saccade
            if dur >= maxdur :
                del(misaccades[s])
                s=s-1
            s=s+1

        return misaccades

    def supp_sacc(self, velocity, saccades, trackertime, before_sacc, after_sacc) :
        '''
        Eliminates saccades detected by eyelink data

        Parameters
        ----------
        velocity : ndarray
            velocity of the eye in deg/sec
        saccades : list
            list of edf saccades for the trial recorded by the eyetracker transformed by the function read_edf of the module edfreader
        trackertime : ndarray
            the time of the tracker
        before_sacc: int
            time to delete before saccades
        after_sacc: int
            time to delete after saccades

        Returns
        -------
        new_velocity : ndarray
            velocity of the eye in deg/sec without saccades
        '''

        t_0 = trackertime[0]

        for s in range(len(saccades)) :
            if saccades[s][1]-t_0+after_sacc <= (len(trackertime)) :
                for x_data in np.arange((saccades[s][0]-t_0-before_sacc), (saccades[s][1]-t_0+after_sacc)) :
                    velocity[x_data] = np.nan
            else :
                for x_data in np.arange((saccades[s][0]-t_0-before_sacc), (len(trackertime))) :
                    velocity[x_data] = np.nan

        return velocity

    def velocity_NAN(self, data_x, data_y, saccades, trackertime, TargetOn, before_sacc=5, after_sacc=15, stop_search_misac=None, px_per_deg=None, **opt) :
        '''
        returns speed of the eye in deg / sec without the saccades

        Parameters
        ----------
        data_x : ndarray
            x position for the trial recorded by the eyetracker transformed by the read_edf function of the edfreader module
        data_y : ndarray
            y position for the trial recorded by the eyetracker transformed by the read_edf function of the edfreader module
        saccades : list
            list of edf saccades for the trial recorded by the eyetracker transformed by the function read_edf of the module edfreader
        trackertime : ndarray
            the time of the tracker
        TargetOn : int
            time when the target to follow appears
        before_sacc: int
            time to delete before saccades
        after_sacc: int
            time to delete after saccades
        stop_search_misac : int
            stop search of micro_saccade
            if None: stops searching at the end of fixation + 100ms
        px_per_deg : float
            number of px per degree for the experiment
                screen_width_deg = 2. * np.arctan((screen_width_cm/2) / viewing_Distance_cm) * 180/np.pi
                px_per_deg = screen_width_px / screen_width_deg
            if None : goes looking for it in param_exp

        Returns
        -------
        velocity : ndarray
            speed of the eye in deg / sec without the saccades
        '''

        if px_per_deg is None :
            px_per_deg = Test.test_value('px_per_deg', self.param_exp) 
        stop_search_misac = Test.test_None(stop_search_misac, value=TargetOn-trackertime[0]+100)


        velocity = ANEMO.velocity_deg(self, data_x=data_x, px_per_deg=px_per_deg)
        velocity_y = ANEMO.velocity_deg(self, data_x=data_y, px_per_deg=px_per_deg)

        new_saccades = saccades.copy()
        misac = ANEMO.detec_misac(self, velocity_x=velocity[:stop_search_misac], velocity_y=velocity_y[:stop_search_misac], t_0=trackertime[0])
        new_saccades.extend(misac)

        velocity_NAN = ANEMO.supp_sacc(self, velocity=velocity, saccades=new_saccades, trackertime=trackertime, before_sacc=before_sacc, after_sacc=after_sacc)

        return velocity_NAN, new_saccades


    ######################################################################################

    class classical_method(object):

        def latence(data, w1=300, w2=50, off=50, crit=0.17) :

            from scipy import stats

            time = np.arange(len(data))
            tps = time
            a = None
            for t in range(len(time)-(w1+off+w2)-300) :
                slope1, intercept1, r_, p_value, std_err = stats.linregress(tps[t:t+w1], data[t:t+w1])
                slope2, intercept2, r_, p_value, std_err = stats.linregress(tps[t+w1+off:t+w1+off+w2], data[t+w1+off:t+w1+off+w2])
                diff = abs(slope2) - abs(slope1)
                if abs(diff) >= crit :
                    a = True
                    tw = time[t:t+w1+off+w2]
                    timew = np.linspace(np.min(tw), np.max(tw), len(tw))

                    fitLine1 = slope1 * timew + intercept1
                    fitLine2 = slope2 * timew + intercept2

                    idx = np.argwhere(np.isclose(fitLine1, fitLine2, atol=0.1)).reshape(-1)
                    old_latence = timew[idx]
                    break

            if a is None :
                old_latence = [np.nan]
            if len(old_latence)==0 :
                old_latence = [np.nan]

            return old_latence[0]

        def maximum(data, TargetOn_0):
            return abs(np.nanmean(data[TargetOn_0+400:TargetOn_0+600]))

        def anticipation(data, TargetOn_0) :
            return np.nanmean(data[TargetOn_0-50:TargetOn_0+50])

        def Full(velocity_NAN, TargetOn_0, w1=300, w2=50, off=50, crit=0.17):

            latence = ANEMO.classical_method.latence(velocity_NAN, w1, w2, off, crit)
            maximum = ANEMO.classical_method.maximum(velocity_NAN, TargetOn_0)
            anticipation = ANEMO.classical_method.anticipation(velocity_NAN, TargetOn_0)

            return latence, maximum, anticipation/0.1

    ######################################################################################

    class Equation(object):

        def fct_velocity (x, dir_target, start_anti, v_anti, latence, tau, maxi, do_whitening) :

            '''
            Function reproducing the speed of the eye during the smooth pursuit of a moving target

            Parameters
            ----------
            x : ndarray
            dir_target : int
                direction of the target -1 ou 1
            start_anti : int
                time when anticipation begins
            v_anti : float
                speed of anticipation in seconds
            latence : int
                time when the movement begins
            tau : float
                curve of the pursuit
            maxi : float
                maximum speed reached during the pursuit
            do_whitening : bool
                if True return the whitened velocity

            Returns
            -------
            velocity : list
                velocity of the eye in deg/sec
            '''

            v_anti = v_anti/1000 # pour passer de sec à ms
            time = x # np.arange(len(x))
            vitesse = []
            y = 0

            for t in range(len(time)):

                if start_anti >= latence :
                    if time[t] < latence :
                        vitesse.append(0)
                    else :
                        vitesse.append(dir_target*maxi*(1-np.exp(-1/tau*(time[t]-latence))))
                else :
                    if time[t] < start_anti :
                        vitesse.append(0)
                    else :
                        if time[t] < latence :
                            vitesse.append((time[t]-start_anti)*v_anti)
                            y = (time[t]-start_anti)*v_anti
                        else :
                            vitesse.append(dir_target*maxi*(1-np.exp(-1/tau*(time[t]-latence)))+y)

            if do_whitening is True:
                vitesse = whitening(vitesse)

            return vitesse

        def fct_position(x, data_x, saccades, nb_sacc, dir_target, start_anti, v_anti, latence, tau, maxi, t_0, px_per_deg, before_sacc, after_sacc, do_whitening):

            '''
            Function reproducing the position of the eye during the smooth pursuit of a moving target

            Parameters
            ----------
            x : ndarray
            data_x : ndarray
                position x of the eye during the trial
            saccades : ndarray the same size as data_x
                List of saccades perform during the trial 
                for i in range(len(saccade)) :
                    saccades[i] -> onset, saccades[i+1] -> end, saccades[i+2] -> time sacc

            nb_sacc : int
                number of saccades during the trial
            dir_target : int
                direction of the target -1 ou 1
            start_anti : int
                time when anticipation begins
            v_anti : float
                speed of anticipation in seconds
            latence : int
                time when the movement begins
            tau : float
                curve of the pursuit
            maxi : float
                maximum speed reached during the pursuit
            t_0 : int
                time 0 of the trial
            px_per_deg : float
                number of px per degree for the experiment
            before_sacc: int
                time to delete before saccades
            after_sacc: int
                time to delete after saccades
            do_whitening : bool
                if True return the whitened position

            Returns
            -------
            position : list
                position of the eye in deg
            '''

            ms = 1000
            v_anti = (v_anti/ms)
            maxi = maxi /ms
            
            speed = ANEMO.Equation.fct_velocity(x=x, dir_target=dir_target, start_anti=start_anti, v_anti=v_anti, latence=latence, tau=tau, maxi=maxi, do_whitening=False)
            pos = np.cumsum(speed)

            i=0
            for s in range(nb_sacc) :
                sacc = saccades[i:i+3] # obligation d'avoir les variable indé a la même taille :/
                                        # saccades[i] -> onset, saccades[i+1] -> end, saccades[i+2] -> time sacc
                #if sacc[0]-t_0 < len(pos) :

                if int(sacc[1]-t_0)+int(after_sacc)+1 <= len(pos) :

                    if do_whitening is True:
                        a = pos[int(sacc[0]-t_0)-int(before_sacc)-1]
                    else :
                        a = np.nan

                    pos[int(sacc[0]-t_0)-int(before_sacc):int(sacc[1]-t_0)+int(after_sacc)] = a
                    if sacc[0]-t_0 >= int(latence-1) :
                        pos[int(sacc[1]-t_0)+int(after_sacc):] += ((data_x[int(sacc[1]-t_0)+int(after_sacc)]-data_x[int(sacc[0]-t_0)-int(before_sacc)-1])/px_per_deg) - np.mean(speed[int(sacc[0]-t_0):int(sacc[1]-t_0)]) * sacc[2]

                else :
                    pos[int(sacc[0]-t_0)-int(before_sacc):] = a

                i = i+3
            if do_whitening is True:
                pos = whitening(pos)
            return pos

        def fct_saccade(x, x_0, tau, x1, x2, T0, t1, t2, tr, do_whitening):
            '''
            Function reproducing the position of the eye during the sacades

            Parameters
            ----------
            x : ndarray
            x_0 :
            tau :
            x1 :
            x2 :
            T0 :
            t1 :
            t2 :
            tr :
            do_whitening : bool
                if True return the whitened position

            Returns
            -------
            position : list
                position of the eye during the sacades in deg
            '''
            time = x-T0
            T1 = t1
            T2 = t1+t2
            TR = T2+tr
            
            rho = (tau/T1) * np.log((1+np.exp(T1/tau))/2)
            rhoT = int(np.round(T1*rho))
            
            r = (tau/T2) * np.log((np.exp(T1/tau) + np.exp(T2/tau)) /2)
            rT = int(np.round(T2*r))
            Umax1 = (1/tau) * x1 / ((2*rho-1)*T1 - tau*(2-np.exp(-(rho*T1)/tau) - np.exp((1-rho)*T1/tau)))
            Umax2 = (1/tau) * (x2-x1) / ((2*r-1)*T2-T1)

            xx = []
            
            for t in time :
                if t < 0 :
                    xx.append(x_0)
                elif t < rhoT :
                    xx.append((x_0 +      Umax1*tau * ((t)    - tau*(1-np.exp(-t/tau)))))
                elif t < T1 :
                    xx.append(x_0 + (x1 + Umax1*tau * ((T1-t) + tau*(1-np.exp((T1-t)/tau)))))
                elif t < rT :
                    xx.append(x_0 + (x1 + Umax2*tau * ((t-T1) - tau*(1-np.exp(-(t-T1)/tau)))))
                elif t < TR :
                    xx.append(x_0 + (x2 + Umax2*tau * ((T2-t) + tau*(1-np.exp((T2-t)/tau)))))
                else :
                    xx.append(xx[-1])
            if do_whitening:
                xx = whitening(xx)

            return xx

    ######################################################################################

    class Fit(object) :

        def __init__(self, param_exp={}) :
            ANEMO.__init__(self, param_exp)


        def generation_param_fit(self, equation='fct_velocity',
                trackertime=None,TargetOn=None, StimulusOf=None, saccades=None, dir_target=None, 
                value_latence=None, value_maxi=15., value_anti=0.,
                before_sacc=5, after_sacc=15,
                px_per_deg=None, data_x=None, **opt) :

            '''
            Generates the parameters and independent variables of the fit

            Parameters
            ----------
            equation : str
                si 'fct_velocity' generates the parameters for a velocity fit
                si 'fct_position' generates the parameters for a fit position
                si 'fct_saccades' generates the parameters for a fit saccades

            if equation in ['fct_velocity', 'fct_position'] :

                option obligatory :
                    dir_target : int
                        the direction of the target
                        si None vas le cherché dans param_exp
                    trackertime : ndarray
                        the time of the tracker
                    TargetOn : int
                        time when the target to follow appears
                    StimulusOf : int
                        time when the stimulus disappears
                    saccades : list
                        list of edf saccades for the trial recorded by the eyetracker transformed by the function read_edf of the module edfreader

                optional not obligatory :
                    value_latence : int
                        value that takes the parameter latence to begin the fit
                        by default = TargetOn-t_0+100
                    value_maxi: float
                        value that takes the parameter maxi to begin the fit
                    value_anti: float
                        value that takes the parameter v_anti to begin the fit
                    before_sacc: int
                        time to remove before saccades
                    after_sacc: int
                        time to delete after saccades

            if equation in ['fct_position', 'fct_saccades'] :

                option obligatory :
                     data_x : ndarray
                        x position for the trial recorded by the eyetracker transformed by the read_edf function of the edfreader module

            if equation == 'fct_position' :

                optional not obligatory :
                    px_per_deg : float
                        number of px per degree for the experiment
                            screen_width_deg = 2. * np.arctan((screen_width_cm/2) / viewing_Distance_cm) * 180/np.pi
                            px_per_deg = screen_width_px / screen_width_deg
                        if None : goes looking for it in param_exp

            Returns
            -------
            param_fit : dict
                dictionary containing the parameters of the fit
            inde_vars : dict
                dictionary containing the independent variables of the fit
            '''

            if equation in ['fct_velocity', 'fct_position'] :

                TargetOn = Test.crash_None('TargetOn', TargetOn)
                StimulusOf = Test.crash_None('StimulusOf', StimulusOf)
                saccades = Test.crash_None('saccades', saccades)

                trackertime = Test.crash_None('trackertime', trackertime)
                t_0 = trackertime[0]

                if dir_target is None :
                    dir_target = Test.test_value('dir_target', self.param_exp, value=None)

                value_latence = Test.test_None(value_latence, value=TargetOn-t_0+100)


                max_latence = []

                for s in range(len(saccades)) :
                    if (saccades[s][0]-t_0) >= (TargetOn-t_0+100) :
                        max_latence.append((saccades[s][0]-t_0))
                if max_latence==[] :
                    max_latence.append(len(trackertime))
                max_latence = max_latence[0]

                param_fit=[{'name':'tau', 'value':15., 'min':13., 'max':80., 'vary':'vary'},
                           {'name':'maxi', 'value':value_maxi, 'min':1., 'max':40., 'vary':True},
                           {'name':'dir_target', 'value':dir_target, 'min':None, 'max':None, 'vary':False},
                           {'name':'v_anti', 'value':value_anti, 'min':-40., 'max':40., 'vary':'vary'},
                           {'name':'latence', 'value':value_latence, 'min':TargetOn-t_0+75, 'max':max_latence, 'vary':True},
                           {'name':'start_anti', 'value':TargetOn-t_0-100, 'min':StimulusOf-t_0-200, 'max':TargetOn-t_0+75, 'vary':'vary'}]


                inde_vars={'x':np.arange(len(trackertime))}

            if equation == 'fct_position' :

                data_x = Test.crash_None('data_x', data_x)
                if px_per_deg is None :
                    px_per_deg = Test.test_value('px_per_deg', self.param_exp, value=None)

                param_fit.extend(({'name':'px_per_deg', 'value':px_per_deg, 'min':None, 'max':None, 'vary':False},
                                  {'name':'t_0', 'value':t_0, 'min':None, 'max':None, 'vary':False},
                                  {'name':'before_sacc', 'value':before_sacc, 'min':None, 'max':None, 'vary':False},
                                  {'name':'after_sacc', 'value':after_sacc, 'min':None, 'max':None, 'vary':False},
                                  {'name':'nb_sacc', 'value':len(saccades), 'min':None, 'max':None, 'vary':False}))

                sacc = np.zeros(len(trackertime))
                i=0
                for s in range(len(saccades)):
                    sacc[i] = saccades[s][0] # onset sacc
                    sacc[i+1] = saccades[s][1] # end sacc
                    sacc[i+2] = saccades[s][2] # time sacc
                    i = i+3

                inde_vars.update({'data_x':data_x, 'saccades':sacc})


            if equation == 'fct_saccade' :
                data_x = Test.crash_None('data_x', data_x)

                if (len(data_x)-10.) <= 10. :
                    max_t1 = 15.
                    max_t2 = 12.
                else :
                    max_t1 = len(data_x)-10.
                    max_t2 = len(data_x)-10.

                param_fit=[{'name':'x_0', 'value':data_x[0], 'min':data_x[0]-0.1, 'max':data_x[0]+0.1, 'vary':'vary'},
                           {'name':'tau', 'value':13., 'min':5., 'max':40., 'vary':True},
                           {'name':'T0', 'value':0., 'min':-15, 'max':10, 'vary':True},
                           {'name':'t1', 'value':15., 'min':10., 'max':max_t1, 'vary':True},
                           {'name':'t2', 'value':12., 'min':10., 'max':max_t2, 'vary':'vary'},
                           {'name':'tr', 'value':1., 'min':0., 'max':15., 'vary':'vary'},
                           {'name':'x1', 'value':2., 'min':-5., 'max':5., 'vary':True},
                           {'name':'x2', 'value':1., 'min':-5., 'max':5., 'vary':'vary'}]

                inde_vars={'x':np.arange(len(data_x))}

            return param_fit, inde_vars


        def Fit_trial(self, data_trial, equation='fct_velocity',
                     trackertime=None, data_x=None, time_sup=280,
                     param_fit=None, inde_vars=None,
                     step_fit=2, do_whitening=False,
                     TargetOn=None, StimulusOf=None, saccades=None,
                     dir_target=None, px_per_deg=None,
                     value_latence=None, value_maxi=15.,
                     value_anti=0., before_sacc=5, after_sacc=15, **opt) :
            '''
            Returns the result of the fit of a trial

            Parameters
            ----------
            equation : str or function
                if 'fct_velocity' : does a data fit with the function 'fct_velocity'
                if 'fct_position' : does a data fit with the function 'fct_position'
                if 'fct_saccades' : does a data fit with the function 'fct_saccades'
                if function : does a data fit with the function
                

            data_trial : ndarray
                if 'fct_velocity' = velocity data for a trial in deg/sec
                if 'fct_position' = position data for a trial in deg
                if 'fct_saccades' = position data for a trial in deg
                if function : velocity or position for a trial


            OBLIGATORY :
                if équation = 'fct_position' :
                     data_x : ndarray
                        x position for the trial recorded by the eyetracker transformed by the read_edf function of the edfreader module

                if no param_fit or no inde_vars:

                    if equation in ['fct_velocity', 'fct_position'] :

                        option obligatory :
                            dir_target : int
                                the direction of the target
                                si None vas le cherché dans param_exp
                            trackertime : ndarray
                                the time of the tracker
                            TargetOn : int
                                time when the target to follow appears
                            StimulusOf : int
                                time when the stimulus disappears
                            saccades : list
                                list of edf saccades for the trial recorded by the eyetracker transformed by the function read_edf of the module edfreader

                        optional not obligatory :
                            value_latence : int
                                value that takes the parameter latence to begin the fit
                                by default = TargetOn-t_0+100
                            value_maxi: float
                                value that takes the parameter maxi to begin the fit
                            value_anti: float
                                value that takes the parameter v_anti to begin the fit
                            before_sacc: int
                                time to remove before saccades
                                    it is advisable to put 5
                            after_sacc: int
                                time to delete after saccades
                                    it is advisable to put 15

                    if equation in ['fct_position', 'fct_saccades'] :

                        option obligatory :
                             data_x : ndarray
                                x position for the trial recorded by the eyetracker transformed by the read_edf function of the edfreader module

                    if equation == 'fct_position' :

                        optional not obligatory :
                            px_per_deg : float
                                number of px per degree for the experiment
                                    screen_width_deg = 2. * np.arctan((screen_width_cm/2) / viewing_Distance_cm) * 180/np.pi
                                    px_per_deg = screen_width_px / screen_width_deg
                                if None : goes looking for it in param_exp

            optional not obligatory :

                trackertime : ndarray
                    tracker time
                    if None : np.arrange((len(data_trial))
                step_fit : int
                    number of steps for the fit
                do_whitening : bool
                    if True return the whitened fit
                time_sup: int
                    time that will be deleted to perform the fit (for data that is less good at the end of the test)
                param_fit : dic
                    fit parameter dictionary, each parameter is a dict containing :
                        'name': name of the variable,
                        'value': initial value,
                        'min': minimum value,
                        'max': maximum value,
                        'vary': True if varies during fit, 'vary' if only varies for step 2, False if not varies during fit
                inde_vars : dic
                    independent variable dictionary of fit

            Returns
            -------
            result_deg : lmfit.model.ModelResult
            '''

            from lmfit import  Model, Parameters

            #-----------------------------------------------------------------------------
            if equation in ['fct_position'] :
                data_x = Test.crash_None('data_x', data_x)

            trackertime = Test.test_None(trackertime, value=np.arange(len(data_trial)))
            #-----------------------------------------------------------------------------

            if step_fit == 1 : vary = True
            elif step_fit == 2 : vary = False

            if equation == 'fct_saccade' :
                time_sup = None
                data_x = data_trial

            if time_sup is not None :
                data_trial = data_trial[:-time_sup]
                trackertime = trackertime[:-time_sup]
                if equation == 'fct_position' :
                    data_x = data_x[:-time_sup]


            if do_whitening:
                for x in range(len(data_trial)) :
                    if np.isnan(data_trial[x]) :
                        data_trial[x] = data_trial[x-1]

                data_trial = whitening(data_trial)
                if equation in ['fct_position'] :
                    data_x = whitening(data_x)

            if param_fit is None or inde_vars is None :
                opt = {'TargetOn' : TargetOn, 'StimulusOf' : StimulusOf,
                        'value_latence' : value_latence, 'value_maxi' : value_maxi, 'value_anti' : value_anti,
                        'saccades' : saccades, 'before_sacc' : before_sacc, 'after_sacc' : after_sacc,
                        'dir_target' : dir_target, 'px_per_deg' : px_per_deg}

            if param_fit is None :
                param_fit = ANEMO.Fit.generation_param_fit(self, equation=equation, trackertime=trackertime, data_x=data_x, **opt)[0]

            if inde_vars is None :
                inde_vars = ANEMO.Fit.generation_param_fit(self, equation=equation, trackertime=trackertime, data_x=data_x, **opt)[1]

            if equation == 'fct_velocity' :
                equation = ANEMO.Equation.fct_velocity
            elif equation == 'fct_position':
                equation = ANEMO.Equation.fct_position
            elif equation == 'fct_saccade':
                equation = ANEMO.Equation.fct_saccade

            params = Parameters()
            model = Model(equation, independent_vars=inde_vars.keys())

            for num_par in range(len(param_fit)) :
                if param_fit[num_par]['vary'] == 'vary' :
                    var = vary
                else :
                    var = param_fit[num_par]['vary']
                params.add(param_fit[num_par]['name'], value=param_fit[num_par]['value'],
                           min=param_fit[num_par]['min'], max=param_fit[num_par]['max'],
                           vary=var)

            params.add('do_whitening', value=do_whitening, vary=False)

            if step_fit == 1 :
                result_deg = model.fit(data_trial, params, nan_policy='omit', **inde_vars)
            elif step_fit == 2 :
                out = model.fit(data_trial, params, nan_policy='omit', **inde_vars)

                # make the other parameters vary now
                for num_par in range(len(param_fit)) :
                    if param_fit[num_par]['vary'] == 'vary' :
                        out.params[param_fit[num_par]['name']].set(vary=True)

                result_deg = model.fit(data_trial, out.params, method='nelder', nan_policy='omit', **inde_vars)

            return result_deg


        def Fit_full(self, data, equation='fct_velocity', fitted_data='velocity',
                    N_blocks=None, N_trials=None, list_param_enre=None,
                    plot=None, file_fig=None,
                    param_fit=None, inde_vars=None, step_fit=2,
                    do_whitening=False, time_sup=280, before_sacc=5, after_sacc=15, 
                    list_events=None, stop_search_misac=None,
                    fig_width=12, t_label=20, t_text=14, ) :

            '''
            Return the parameters of the fit present in list_param_enre

            Parameters
            ----------
            data : list
                edf data for the trials recorded by the eyetracker transformed by the read_edf function of the edfreader module

            equation : str or function
                if 'fct_velocity' : does a data fit with the function 'fct_velocity'
                if 'fct_position' : does a data fit with the function 'fct_position'
                if 'fct_saccades' : does a data fit with the function 'fct_saccades'
                if function : does a data fit with the function
                


            fitted_data : bool
                if 'velocity' = fit the velocity data for a trial in deg/sec
                if 'position' = fit the position data for a trial in deg
                if 'saccade' = fit the position data for sacades in trial in deg

            N_blocks : int
                number of blocks
                if None went searched in param_exp
            N_trials : int
                number of trials per block
                if None went searched in param_exp

            list_param_enre : list
                list of fit parameters to record
                if None :
                    if equation in ['fct_velocity', 'fct_position'] : ['fit', 'start_anti', 'v_anti', 'latence', 'tau', 'maxi', 'saccades', 'old_anti', 'old_max', 'old_latence']
                    if equation is 'fct_saccades' : ['fit', 'T0', 't1', 't2', 'tr', 'x_0', 'x1', 'x2', 'tau']

            plot : bool
                if true : save the figure in file_fig
            file_fig : str
                name of file figure reccorded
                if None file_fig is 'Fit'

            param_fit : dic
                fit parameter dictionary, each parameter is a dict containing :
                    'name': name of the variable,
                    'value': initial value,
                    'min': minimum value,
                    'max': maximum value,
                    'vary': True if varies during fit, 'vary' if only varies for step 2, False if not varies during fit
                if None : Generate by generation_param_fit
            inde_vars : dic
                independent variable dictionary of fit
                if None : Generate by generation_param_fit

            step_fit : int
                number of steps for the fit
            do_whitening : bool
                if True return the whitened fit
            time_sup: int
                time that will be deleted to perform the fit (for data that is less good at the end of the test)

            before_sacc: int
                time to remove before saccades
                    it is advisable to put :
                        5 for 'fct_velocity' and 'fct_position'
                        0 for 'fct_saccade'

            after_sacc: int
                time to delete after saccades
                    it is advisable to put : 15

            list_events : list
                list of the names of the events of the trial
                by default : ['StimulusOn\n', 'StimulusOff\n', 'TargetOn\n', 'TargetOff\n']

            stop_search_misac : int
                stop search of micro_saccade
                if None: stops searching at the end of fixation + 100ms


            fig_width : int
                figure size
            t_label : int
                size x and y label
            t_text : int
                size of the text of the figure

            Returns
            -------
            param : dict
                each parameter are ordered : [block][trial]
            '''

            #------------------------------------------------------------------------------
            if N_blocks is None :
                N_blocks = Test.test_value('N_blocks', self.param_exp, crash=None)
                N_blocks = Test.test_None(N_blocks, 1)
            if N_trials is None :
                N_trials = Test.test_value('N_trials', self.param_exp, crash=None)
                N_trials = Test.test_None(N_trials, int(len(data)/N_blocks))
            #------------------------------------------------------------------------------

            if plot is not None :
                import matplotlib.pyplot as plt
                if equation=='fct_saccade' :
                    import matplotlib.gridspec as gridspec

            if equation in ['fct_velocity', 'fct_position'] :
                list_param_enre = Test.test_None(list_param_enre, value=['fit', 'start_anti', 'v_anti', 'latence', 'tau', 'maxi',
                                                                'old_anti', 'old_max', 'old_latence'])

            if equation == 'fct_saccade' :
                list_param_enre = Test.test_None(list_param_enre, value=['fit', 'T0', 't1', 't2', 'tr', 'x_0', 'x1', 'x2', 'tau'])

            if list_param_enre is None :
                print('Warning list_param_enre is None, no parameter will be returned !!!')
                list_param_enre = []

            opt_base = {'stop_search_misac':stop_search_misac, 'time_sup':time_sup,
                        'param_fit':param_fit, 'inde_vars':inde_vars,
                        'step_fit':step_fit, 'do_whitening':do_whitening,
                        'before_sacc':before_sacc, 'after_sacc':after_sacc,
                        't_label':t_label}

            param = {}
            if 'observer' in self.param_exp.keys() :
                param['observer'] = self.param_exp['observer']
            for name in list_param_enre :
                param[name] = []

            for block in range(N_blocks) :
                if plot is not None :
                    if equation=='fct_saccade' or fitted_data=='saccade' :
                        fig = plt.figure(figsize=(fig_width, (fig_width*(N_trials/2)/1.6180)))
                        axs = gridspec.GridSpec(N_trials, 1, hspace=0.4)
                    else :
                        fig, axs = plt.subplots(N_trials, 1, figsize=(fig_width, (fig_width*(N_trials/2))/1.6180))

                result_fit = {}
                for name in list_param_enre :
                    result_fit[name] = []

                for trial in range(N_trials) :

                    print('block, trial = ', block, trial)

                    trial_data = trial + N_trials*block
                    arg = ANEMO.arg(self, data[trial_data], trial=trial, block=block, list_events=list_events)
                    opt = opt_base.copy()
                    opt.update(arg)

                    if plot is not None :
                        start = arg.TargetOn
                        trackertime_s = arg.trackertime - start

                    velocity_NAN = ANEMO.velocity_NAN(self, **opt)[0]

                    if equation=='fct_velocity' :
                        fitted_data = 'velocity'
                    if equation=='fct_position' :
                        fitted_data = 'position'
                    if equation=='fct_saccade' :
                        fitted_data = 'saccade'

                    if fitted_data=='velocity' :
                        data_x = arg.data_x
                        data_1 = velocity_NAN
                        data_trial = np.copy(data_1)
                    else :
                        data_x = (arg.data_x - (arg.data_x[arg.StimulusOf-arg.t_0]))/arg.px_per_deg
                        data_1 = data_x
                        data_trial = np.copy(data_1)


                    if fitted_data != 'saccade' :

                        old_latence, old_max, old_anti = ANEMO.classical_method.Full(velocity_NAN, arg.TargetOn-arg.t_0)
                        #-------------------------------------------------
                        # FIT
                        #-------------------------------------------------
                        f = ANEMO.Fit.Fit_trial(self, data_trial, equation=equation, value_latence=old_latence, value_max=old_max, value_anti=old_anti, **opt)
                        #-------------------------------------------------

                        onset  = arg.TargetOn - arg.t_0 # TargetOn - time_0
                        for name in list_param_enre :
                            if name in f.values.keys() :
                                if name in ['start_anti', 'latence'] :
                                    val = f.values[name] - onset
                                else :
                                    val = f.values[name]
                                result_fit[name].append(val)

                        if 'fit' in list_param_enre :
                            result_fit['fit'].append(f.best_fit)
                        if 'old_anti' in list_param_enre :
                            result_fit['old_anti'].append(old_anti)
                        if 'old_max' in list_param_enre :
                            result_fit['old_max'].append(old_max)
                        if 'old_latence' in list_param_enre :
                            result_fit['old_latence'].append(old_latence-onset)

                        if plot is not None :
                            if N_trials==1:
                                ax = axs
                            else :
                                ax = axs[trial]
                            ax.cla() # to put ax figure to zero

                            inde_v = Test.test_None(inde_vars, ANEMO.Fit.generation_param_fit(self, equation=equation, **opt)[1])
                            rv = f.values
                            if 'do_whitening' in f.values.keys() :
                                rv['do_whitening'] = False
                            rv.update(inde_v)

                            if fitted_data=='velocity' :
                                scale = 1
                                ax.set_ylabel('%s\nVelocity (°/s)'%(trial+1), fontsize=t_label)
                            if fitted_data=='position' :
                                scale = 1/2
                                ax.set_ylabel('%s\nDistance (°)'%(trial+1), fontsize=t_label)

                            TargetOn_s = arg.TargetOn - start
                            TargetOff_s = arg.TargetOff - start

                            ax.axis([TargetOn_s-700, TargetOff_s+10, -39.5*scale, 39.5*scale])
                            ax = ANEMO.Plot.deco(self, ax, **opt)

                            if trial==0 :
                                StimulusOf_s = arg.StimulusOf - start
                                ax.text(StimulusOf_s+(TargetOn_s-StimulusOf_s)/2, 31*scale, "GAP", color='k', fontsize=t_label*1.5, ha='center', va='center', alpha=0.5)
                                ax.text((TargetOn_s-700)+(StimulusOf_s-(TargetOn_s-700))/2, 31*scale, "FIXATION", color='k', fontsize=t_label*1.5, ha='center', va='center', alpha=0.5)
                                ax.text(TargetOn_s+(TargetOff_s-TargetOn_s)/2, 31*scale, "PURSUIT", color='k', fontsize=t_label*1.5, ha='center', va='center', alpha=0.5)
                                #ax.text(result_fit['latence'][trial]+25, -35*scale, "Latence"%(result_fit['latence'][trial]), color='r', fontsize=t_text, alpha=0.5)#,  weight='bold')

                            if equation=='fct_velocity' :
                                eqt = ANEMO.Equation.fct_velocity
                            elif equation=='fct_position' :
                                eqt = ANEMO.Equation.fct_position
                            else :
                                eqt = equation

                            fit = eqt(**rv)
                            ax.plot(trackertime_s[:-time_sup], fit[:-time_sup], color='r', linewidth=2)
                            ax.plot(trackertime_s, data_1, color='k', alpha=0.4)
                            x = 0
                            for name in list_param_enre :
                                if name in f.values.keys() :
                                    ax.text((TargetOff_s-10), 35*scale+x, "%s: %0.3f"%(name, result_fit[name][trial]) , color='k', fontsize=t_text, va='center', ha='right')
                                    x = x - 5*scale

                            ax.set_xlabel('Time (ms)', fontsize=t_label)


                    if fitted_data == 'saccade' :
                        if plot is not None :
                            if N_trials==1:
                                ax = axs
                            else :
                                ax = axs[trial]
                            ax0 = gridspec.GridSpecFromSubplotSpec(1, len(arg.saccades), subplot_spec=ax)
                            y = 0

                        for name in list_param_enre :
                            result_fit[name].append([])

                        for s in range(len(arg.saccades)):
                            data_sacc = data_1[arg.saccades[s][0]-arg.t_0-before_sacc:arg.saccades[s][1]-arg.t_0+after_sacc]
                            #-------------------------------------------------
                            # FIT
                            #-------------------------------------------------
                            f = ANEMO.Fit.Fit_trial(self, data_sacc, equation=equation, **opt)
                            #-------------------------------------------------

                            for name in list_param_enre :
                                if name in f.values.keys() :
                                    result_fit[name][trial].append(f.values[name])
                            if 'fit' in list_param_enre :
                                result_fit['fit'][trial].append(f.best_fit)

                            if plot is not None :
                                ax1 = plt.Subplot(fig, ax0[s])
                                time = trackertime_s[arg.saccades[s][0]-arg.t_0-before_sacc:arg.saccades[s][1]-arg.t_0+after_sacc]

                                opt['data_x'] = data_sacc
                                inde_v = Test.test_None(inde_vars, ANEMO.Fit.generation_param_fit(self, equation=equation, **opt)[1])
                                rv = f.values
                                if 'do_whitening' in f.values.keys() :
                                    rv['do_whitening'] = False
                                rv.update(inde_v)

                                if equation=='fct_saccade' :
                                    eqt = ANEMO.Equation.fct_saccade
                                else :
                                    eqt = equation

                                #-----------------------------------------------------------------------------
                                fit = eqt(**rv)

                                ax1.plot(time, data_sacc, color='k', alpha=0.4)
                                ax1.plot(time, fit, color='r', alpha=0.6)

                                minx, maxx = time[0], time[-1]
                                miny, maxy = min(data_sacc), max(data_sacc)
                                #-----------------------------------------------------------------------------
                                px = 0
                                for name in list_param_enre :
                                    if name in f.values.keys() :
                                        ax1.text(minx+(maxx-minx)/50, (maxy+(maxy-miny)/5)-px, "%s: %0.3f"%(name, f.values[name]) , color='k', 
                                                        ha='left', va='center', fontsize=t_label/1.8) #, alpha=0.8)
                                        px = px + ((maxy+(maxy-miny)/5)-(miny-(maxy-miny)/5))/(len(list_param_enre)-1)
                                #-----------------------------------------------------------------------------
                                ax1.set_title('Saccade %s'%(s+1), fontsize=t_label/1.5, x=0.5, y=1.01)
                                ax1.axis([minx-(maxx-minx)/30, maxx+(maxx-minx)/30, miny-(maxy-miny)/3, maxy+(maxy-miny)/3])
                                #-----------------------------------------------------------------------------
                                ax1.set_xlabel('Time (ms)', fontsize=t_label/2)
                                if y==0 :
                                    ax1.set_ylabel('%s\nDistance (°)'%(trial+1), fontsize=t_label)
                                ax1.tick_params(labelsize=t_label/2.5 , bottom=True, left=True)
                                #-----------------------------------------------------------------------------
                                fig.add_subplot(ax1)

                                y=y+1

                for name in list_param_enre :
                    param[name].append(result_fit[name])

                if plot is not None :
                    if equation=='fct_saccade' :
                        axs.tight_layout(fig) # to remove too much margin
                    else :
                        plt.tight_layout() # to remove too much margin
                    plt.subplots_adjust(hspace=0) # to remove space between figures
                    file_fig = Test.test_None(file_fig, 'Fit')
                    plt.savefig(file_fig+'_%s.pdf'%(block+1))
                    plt.close()

            return param

    ######################################################################################


    class Plot(object) :

        def __init__(self, param_exp=None) :
            ANEMO.__init__(self, param_exp)

        def deco (self, ax, StimulusOn=None, StimulusOf=None, TargetOn=None, TargetOff=None, saccades=None, t_label=20, **opt) :

            try :

                StimulusOn = Test.crash_None('StimulusOn', StimulusOn)
                StimulusOf = Test.crash_None('StimulusOf', StimulusOf)
                TargetOn = Test.crash_None('TargetOn', TargetOn)
                TargetOff = Test.crash_None('TargetOff', TargetOff)
                saccades = Test.crash_None('saccades', saccades)

                start = TargetOn
                StimulusOn_s = StimulusOn - start
                StimulusOf_s = StimulusOf - start
                TargetOn_s = TargetOn - start
                TargetOff_s = TargetOff - start

                ax.axvspan(StimulusOn_s, StimulusOf_s, color='k', alpha=0.2)
                ax.axvspan(StimulusOf_s, TargetOn_s, color='r', alpha=0.2)
                ax.axvspan(TargetOn_s, TargetOff_s, color='k', alpha=0.15)

                # Saccade
                for s in range(len(saccades)) :
                    ax.axvspan(saccades[s][0]-start, saccades[s][1]-start, color='k', alpha=0.15)
                #-----------------------------------------------------------------------------
            finally :
                #-----------------------------------------------------------------------------
                ax.set_xlabel('Time (ms)', fontsize=t_label)
                ax.tick_params(labelsize=t_label/2 , bottom=True, left=True)
                #-----------------------------------------------------------------------------

            return ax


            import easydict
            return easydict.EasyDict(kwargs)

        ######################################################################################
        def plot_equation(self, equation='fct_velocity', fig_width=15, t_titre=35, t_label=20):

            '''
            Returns figure of the equation used for the fit with the parameters of the fit

            Parameters
            ----------
            equation : str or function
                if 'fct_velocity' displays the fct_velocity equation
                if 'fct_position' displays the fct_position equation
                if 'fct_saccades' displays the fct_saccades equation
                if function displays the function equation

            fig_width : int
                figure size

            t_titre : int
                size of the title of the figure

            t_label : int
                size x and y label

            Returns
            -------
            fig : matplotlib.figure.Figure
                figure
            ax : AxesSubplot
                figure
            '''


            import matplotlib.pyplot as plt


            fig, ax = plt.subplots(1, 1, figsize=(fig_width, (fig_width*(1/2)/1.6180)))


            if equation in ['fct_velocity','fct_position'] :

                time = np.arange(-750, 750, 1)
                StimulusOn, StimulusOf = -750, -300
                TargetOn, TargetOff = 0, 750
                start_anti, latence = 650, 850
                #-----------------------------------------------------------------------------

                if equation=='fct_velocity' :
                    ax.set_title('Function Velocity', fontsize=t_titre, x=0.5, y=1.05)
                    ax.set_ylabel('Velocity (°/s)', fontsize=t_label)

                    scale = 1
                    result_fit = ANEMO.Equation.fct_velocity (x=np.arange(len(time)), start_anti=start_anti, latence=latence,
                                                              v_anti=-20, tau=15., maxi=15., dir_target=-1, do_whitening=False)

                if equation=='fct_position' :
                    ax.set_title('Function Position', fontsize=t_titre, x=0.5, y=1.05)
                    ax.set_ylabel('Distance (°)', fontsize=t_label)

                    scale = 1/2
                    result_fit = ANEMO.Equation.fct_position(x=np.arange(len(time)), data_x=np.zeros(len(time)),
                                                            saccades=np.zeros(len(time)), nb_sacc=0, before_sacc=5, after_sacc=15,
                                                            start_anti=start_anti, v_anti=-20, latence=latence, tau=15., maxi=15.,
                                                            t_0=0, dir_target=-1, px_per_deg=36.51807384230632,  do_whitening=False)

                #-----------------------------------------------------------------------------
                ax.axis([-750, 750, -39.5*scale, 39.5*scale])
                ax.set_xlabel('Time (ms)', fontsize=t_label)
                #-----------------------------------------------------------------------------

                ax.plot(time[latence+250:],        result_fit[latence+250:],        c='k', linewidth=2)
                ax.plot(time[:start_anti],         result_fit[:start_anti],         c='k', linewidth=2)
                ax.plot(time[start_anti:latence],  result_fit[start_anti:latence],  c='r', linewidth=2)
                ax.plot(time[latence:latence+250], result_fit[latence:latence+250], c='darkred', linewidth=2)

                # V_a ------------------------------------------------------------------------
                ax.text(TargetOn, 15*scale, "Anticipation", color='r', fontsize=t_label, ha='center')
                ax.text(TargetOn-50, -5*scale, r"A$_a$", color='r', fontsize=t_label/1.5, ha='center', va='top')

                if equation=='fct_velocity' :
                    ax.annotate('', xy=(time[latence], result_fit[latence]-3), xycoords='data', fontsize=t_label/1.5,
                                xytext=(time[start_anti], result_fit[start_anti]-3), textcoords='data', arrowprops=dict(arrowstyle="->", color='r'))

                # Start_a --------------------------------------------------------------------
                ax.text(TargetOn-125, -35*scale, "Start anticipation", color='k', fontsize=t_label, alpha=0.7, ha='right')
                ax.bar(time[start_anti], 80*scale, bottom=-40*scale, color='k', width=4, linewidth=0, alpha=0.7)

                # latence --------------------------------------------------------------------
                ax.text(TargetOn+125, -35*scale, "Latency", color='firebrick', fontsize=t_label)
                ax.bar(time[latence], 80*scale, bottom=-40*scale, color='firebrick', width=4, linewidth=0, alpha=1)

                # tau ------------------------------------------------------------------------
                ax.annotate(r'$\tau$', xy=(time[latence]+15, result_fit[latence+15]), xycoords='data', fontsize=t_label/1., color='darkred', va='bottom',
                        xytext=(time[latence]+70, result_fit[latence+7]), textcoords='data', arrowprops=dict(arrowstyle="->", color='darkred'))

                # Max ------------------------------------------------------------------------
                ax.text(TargetOn+400+25, ((result_fit[latence+400])/2), 'Steady State', color='k', fontsize=t_label, va='center')

                if equation=='fct_velocity' :
                    ax.annotate('', xy=(TargetOn+400, 0), xycoords='data', fontsize=t_label/1.5, xytext=(TargetOn+400, result_fit[latence+400]), textcoords='data', arrowprops=dict(arrowstyle="<->"))
                    ax.plot(time, np.zeros(len(time)), '--k', linewidth=1, alpha=0.5)
                    ax.plot(time[latence:], np.ones(len(time[latence:]))*result_fit[latence+400], '--k', linewidth=1, alpha=0.5)

                # COSMETIQUE -----------------------------------------------------------------
                ax.axvspan(StimulusOn, StimulusOf, color='k', alpha=0.2)
                ax.axvspan(StimulusOf, TargetOn, color='r', alpha=0.2)
                ax.axvspan(TargetOn, TargetOff, color='k', alpha=0.15)
                
                ax.text(StimulusOf+(TargetOn-StimulusOf)/2, 31*scale, "GAP", color='k', fontsize=t_label*1.5, ha='center', va='center', alpha=0.5)
                ax.text((StimulusOf-750)/2, 31*scale, "FIXATION", color='k', fontsize=t_label*1.5, ha='center', va='center', alpha=0.5)
                ax.text((750-TargetOn)/2, 31*scale, "PURSUIT", color='k', fontsize=t_label*1.5, ha='center', va='center', alpha=0.5)
                #-----------------------------------------------------------------------------

            elif equation=='fct_saccade' :

                time = np.arange(30)
                #-----------------------------------------------------------------------------

                T0,  t1,  t2,  tr = 0, 15, 12, 1
                x_0, x1, x2, tau = 0, 2, 1, 13
                
                fit = ANEMO.Equation.fct_saccade(time, x_0, tau, x1, x2, T0, t1, t2, tr,do_whitening=False)

                ax.plot(time, fit, color='R')
                
                minx, maxx = min(time[0], T0 + time[0]), max(time[-1], T0+t1+t2+tr + time[0])# time[0], time[-1]
                miny, maxy = min(fit), max(fit)
                #-----------------------------------------------------------------------------
                kwarg = {'fontsize':t_label/1.5, 'va':'center'}
                
                # T0 -------------------------------------------------------------------------
                ax.axvspan(T0 + time[0], T0+t1 + time[0], color='r', alpha=0.2)
                ax.text(T0+time[0]+(maxx-minx)/100, maxy+(maxy-miny)/6, 'T0', color='r', alpha=0.5, **kwarg)

                # T1 -------------------------------------------------------------------------
                ax.axvspan(T0+t1 + time[0], T0+t1+t2 + time[0], color='k', alpha=0.2)
                ax.text(T0+t1+time[0]+(maxx-minx)/100, maxy+(maxy-miny)/6, 't1', color='k', alpha=0.5, **kwarg)

                # T2 -------------------------------------------------------------------------
                ax.axvspan(T0+t1+t2 + time[0], T0+t1+t2+tr + time[0], color='r', alpha=0.2)
                ax.text(T0+t1+t2+time[0]+(maxx-minx)/100, maxy+(maxy-miny)/6, 't2', color='r', alpha=0.5, **kwarg)

                # tr -------------------------------------------------------------------------
                ax.text(T0+t1+t2+tr+time[0]+(maxx-minx)/100, miny-(maxy-miny)/6, 'tr', color='k', alpha=0.5, **kwarg)

                # x_0 -------------------------------------------------------------------------
                ax.hlines(x_0, minx, maxx, color='k', lw=1, linestyles='--', alpha=0.3)
                ax.text(maxx+(maxx-minx)/100, x_0, 'x_0', color='k', **kwarg)

                # x1 -------------------------------------------------------------------------
                ax.hlines(x1+x_0, minx, maxx, color='k', lw=1, linestyles='--', alpha=0.5)
                ax.text(maxx+(maxx-minx)/100, x1+x_0, 'x1', color='k', **kwarg)

                # x2 -------------------------------------------------------------------------
                ax.hlines(x2+x_0, minx, maxx, color='r', lw=1, linestyles='--', alpha=0.5)
                ax.text(maxx+(maxx-minx)/100, x2+x_0, 'x2', color='r', **kwarg)

                # tau ------------------------------------------------------------------------
                ax.text(T0+t1 + time[0], x1+x_0-(maxy-miny)/10, 'tau', color='k', ha='center', **kwarg)

                #-----------------------------------------------------------------------------
                ax.set_title('Saccade Function', fontsize=t_titre, x=0.5, y=1.05)
                ax.axis([minx-(maxx-minx)/100, maxx+(maxx-minx)/20, miny-(maxy-miny)/3, maxy+(maxy-miny)/3])
                ax.set_xlabel('Time (ms)', fontsize=t_label/1.3)
                ax.set_ylabel('Distance (°)', fontsize=t_label)
                #-----------------------------------------------------------------------------

            else :
                ax.plot(equation, c='k', linewidth=2)

            ax.tick_params(labelsize=t_label/2 , bottom=True, left=True)
            plt.tight_layout() # to remove the margin too large
            return fig, ax

        def plot_data(self, data, show='velocity', trials=0, block=0,
                        N_trials=None, list_events=None,
                        fig_width=15, t_titre=35, t_label=20,
                        stop_search_misac=None, name_trial_show=False, before_sacc=5, after_sacc=15):
            '''
            Returns the data figure

            Parameters
            ----------

            data : list
                edf data for the trials recorded by the eyetracker transformed by the read_edf function of the edfreader module
            show : str
                if 'velocity' show the velocity of the eye
                if 'position' show the position of the eye
                if 'saccades' shows the saccades of the eye

            trials : int or list
                number or list of trials to display
            block : int
                number of the block in which it finds the trials to display
            N_trials : int
                number of trials per block
                if None went searched in param_exp

            list_events : list
                list of the names of the events of the trial
                by default : ['StimulusOn\n', 'StimulusOff\n', 'TargetOn\n', 'TargetOff\n']

            before_sacc: int
                time to remove before saccades
                    it is advisable to put :
                        5 for 'fct_velocity' and 'fct_position'
                        0 for 'fct_saccade'

            after_sacc: int
                time to delete after saccades
                    it is advisable to put : 15

            stop_search_misac : int
                stop search of micro_saccade
                if None: stops searching at the end of fixation + 100ms
            name_trial_show : bool
                if True the num is written of the trial in y_label

            fig_width : int
                figure size
            t_titre : int
                size of the title of the figure
            t_label : int
                size x and y label

            Returns
            -------
            fig : matplotlib.figure.Figure
                figure
            ax : AxesSubplot
            figure
            '''

            import matplotlib.pyplot as plt

            if type(trials) is not list : trials = [trials]

            if show in ['velocity', 'position'] :
                fig, axs = plt.subplots(len(trials), 1, figsize=(fig_width, (fig_width*(len(trials)/2)/1.6180)))

            if show == 'saccade' :
                import matplotlib.gridspec as gridspec
                fig = plt.figure(figsize=(fig_width, (fig_width*(len(trials))/1.6180)))
                axs = gridspec.GridSpec(len(trials), 1, hspace=0.4)

            if N_trials is None :
                N_trials = Test.test_value('N_trials', self.param_exp)

            opt_base = {'t_label':t_label, 'stop_search_misac':stop_search_misac, 'before_sacc':before_sacc, 'after_sacc':after_sacc}

            x = 0
            for t in trials :

                if name_trial_show is True :
                    print('block, trial = ', block, t)
                trial_data = t + N_trials*block
                arg = ANEMO.arg(self, data[trial_data], list_events=list_events, trial=t, block=block)

                opt = opt_base.copy()
                opt.update(arg)

                start = arg.TargetOn
                StimulusOn_s = arg.StimulusOn - start
                StimulusOf_s = arg.StimulusOf - start
                TargetOn_s = arg.TargetOn - start
                TargetOff_s = arg.TargetOff - start
                trackertime_s = arg.trackertime - start


                if show in ['velocity', 'position'] :
                    if len(trials)==1:
                        ax = axs
                    else :
                        ax = axs[x]
                    if x!= (len(trials)-1) :
                        ax.set_xticklabels([])


                if show=='velocity' :
                    ax.axis([-750, 750, -39.5, 39.5])
                    #-----------------------------------------------------------------------------
                    if x==0 :
                        ax.set_title('Eye Movement', fontsize=t_titre, x=0.5, y=1.05)
                    if name_trial_show is True :
                        ax.set_ylabel('%s\nVelocity (°/s)'%(t+1), fontsize=t_label)
                    else :
                        ax.set_ylabel('Velocity (°/s)', fontsize=t_label)
                    #-----------------------------------------------------------------------------
                    velocity_NAN = ANEMO.velocity_NAN(self, **opt)[0]
                    ax.plot(trackertime_s, velocity_NAN, color='k', alpha=0.4)
                    #-----------------------------------------------------------------------------


                if show in ['position', 'saccade'] :
                    if show=='saccade' :
                        axs0 = gridspec.GridSpecFromSubplotSpec(2, len(arg.saccades), subplot_spec=axs[x], hspace=0.45, wspace=0.2)
                        ax = plt.Subplot(fig, axs0[0,:])
                        if name_trial_show is True :
                            ax.set_ylabel('%s\nDistance (°)'%(t+1), fontsize=t_label)
                        else :
                            ax.set_ylabel('Distance (°)', fontsize=t_label)
                        fig.add_subplot(ax)

                    ax.axis([-750, 750, -39.5/2, 39.5/2])
                    if x==0 :
                        ax.set_title('Eye Position', fontsize=t_titre, x=0.5, y=1.05)

                    data_x = (arg.data_x - (arg.data_x[arg.StimulusOf-arg.t_0])) / arg.px_per_deg
                    data_y = (arg.data_y - (arg.data_y[arg.StimulusOf-arg.t_0])) / arg.px_per_deg
                    ax.plot(trackertime_s, data_x, color='k', linewidth=1.5)
                    ax.plot(trackertime_s, data_y, color='c', linewidth=1.5)


                if show=='position' :
                    #-----------------------------------------------------------------------------
                    if name_trial_show is True :
                        ax.set_ylabel('%s\nDistance (°)'%(t+1), fontsize=t_label)
                    else :
                        ax.set_ylabel('Distance (°)', fontsize=t_label)
                    #------------------------------------------------
                    # TARGET
                    #------------------------------------------------
                    Target_trial = []
                    for tps in trackertime_s :
                        if tps < TargetOn_s :
                            pos_target = 0
                        elif tps == TargetOn_s :
                            # the target at t = 0 retreats from its velocity * latency = RashBass (here set in ms)
                            pos_target = pos_target -(arg.dir_target * ((arg.V_X/1000)*arg.RashBass)) / arg.px_per_deg
                        elif (tps > TargetOn_s and tps <= (TargetOn_s+arg.stim_tau*1000)) :
                            pos_target = pos_target + (arg.dir_target*(arg.V_X/1000)) / arg.px_per_deg
                        else :
                            pos_target = pos_target
                        Target_trial.append(pos_target)
                    ax.plot(trackertime_s, Target_trial, color='r', linewidth=1.5)
                    #------------------------------------------------

                if show=='saccade' :
                    y=0

                    for s in range(len(arg.saccades)):
                        ax1 = plt.Subplot(fig, axs0[1,s])
                        ax1.set_title('Saccade %s'%(s+1), fontsize=t_label, x=0.5, y=1.05)
                        if y==0:
                            if name_trial_show is True :
                                ax1.set_ylabel('%s\nDistance (°)'%(t+1), fontsize=t_label)
                            else :
                                ax1.set_ylabel('Distance (°)', fontsize=t_label)
                        data_sacc  = data_x[arg.saccades[s][0]-arg.t_0-before_sacc:arg.saccades[s][1]-arg.t_0+after_sacc]
                        time = trackertime_s[arg.saccades[s][0]-arg.t_0-before_sacc:arg.saccades[s][1]-arg.t_0+after_sacc]
                        ax1.plot(time, data_sacc, color='k', alpha=0.6)
                        x1, x2 = time[0], time[-1]
                        y1, y2 = min(data_sacc), max(data_sacc)
                        ax1 = ANEMO.Plot.deco(self, ax1, **opt)

                        ax1.axis([x1-(x2-x1)/10, x2+(x2-x1)/10, y1-(y2-y1)/10, y2+(y2-y1)/10])
                        fig.add_subplot(ax1)

                        y=y+1

                ax = ANEMO.Plot.deco(self, ax, **opt)
                x=x+1

            if show in ['velocity', 'position'] :
                plt.tight_layout() # to remove the margin too large
                plt.subplots_adjust(hspace=0) # to remove space between figures
            if show =='saccade' :
                axs.tight_layout(fig) # to remove the margin too large

            return fig, axs

        def plot_fit(self, data, equation='fct_velocity', trials=0, block=0, N_trials=None,
                        fig_width=15, t_titre=35, t_label=20,
                        list_events=None, report=None, before_sacc=5, after_sacc=15,
                        step_fit=2, do_whitening=False, time_sup=280, param_fit=None, inde_vars=None,
                        stop_search_misac=None):

            '''
            Returns figure of data fits

            Parameters
            ----------
            data : list
                edf data for the trials recorded by the eyetracker transformed by the read_edf function of the edfreader module

            equation : str or function
                if 'fct_velocity' : does a data fit with the function 'fct_velocity'
                if 'fct_position' : does a data fit with the function 'fct_position'
                if 'fct_saccades' : does a data fit with the function 'fct_saccades'
                if function : does a data fit with the function

            trials : int or list
                number or list of trials to display
            block : int
                number of the block in which it finds the trials to display
            N_trials : int
                number of trials per block
                if None went searched in param_exp


            list_events : list
                list of the names of the events of the trial
                by default : ['StimulusOn\n', 'StimulusOff\n', 'TargetOn\n', 'TargetOff\n']

            stop_search_misac : int
                stop search of micro_saccade
                if None: stops searching at the end of fixation + 100ms


            report : bool
                if true return the report of the fit for each trial
            step_fit : int
                number of steps for the fit
            do_whitening : bool
                if true the fit perform on filtered data with a whitening filter

            time_sup: int
                time that will be deleted to perform the fit (for data that is less good at the end of the test)
            param_fit : dict
                dictionary containing the parameters of the fit
            inde_vars : dict
                dictionary containing the independent variables of the fit

            before_sacc: int
                time to remove before saccades
                    it is advisable to put :
                        5 for 'fct_velocity' and 'fct_position'
                        0 for 'fct_saccade'

            after_sacc: int
                time to delete after saccades
                    it is advisable to put : 15


            fig_width : int
                figure size
            t_titre : int
                size of the title of the figure
            t_label : int
                size x and y label

 
            Returns
            -------
            fig : matplotlib.figure.Figure
                figure
            ax : AxesSubplot
                figure
            report : list
                list of the reports of the fit for each trial
            '''

            if N_trials is None :
                N_trials = Test.test_value('N_trials', self.param_exp)

            opt_base = {'stop_search_misac':stop_search_misac,'equation':equation,'time_sup':time_sup,
                        'param_fit':param_fit, 'inde_vars':inde_vars, 'step_fit':step_fit, 'do_whitening':do_whitening,
                        'before_sacc':before_sacc, 'after_sacc':after_sacc,'t_label':t_label,}

            import matplotlib.pyplot as plt

            if type(trials) is not list : trials = [trials]

            if equation=='fct_saccade' :
                import matplotlib.gridspec as gridspec
                fig = plt.figure(figsize=(fig_width, (fig_width*(len(trials))/1.6180)))
                axs = gridspec.GridSpec(len(trials), 1, hspace=0.4)
            else :
                fig, axs = plt.subplots(len(trials), 1, figsize=(fig_width, (fig_width*(len(trials)/2)/1.6180)))

            results = []
            x = 0
            for t in trials :

                trial_data = t + N_trials*block
                arg = ANEMO.arg(self, data[trial_data], list_events=list_events, trial=t, block=block)

                opt = opt_base.copy()
                opt.update(arg)

                start = arg.TargetOn
                StimulusOn_s = arg.StimulusOn - start
                StimulusOf_s = arg.StimulusOf - start
                TargetOn_s = arg.TargetOn - start
                TargetOff_s = arg.TargetOff - start
                trackertime_s = arg.trackertime - start

                data_x = (arg.data_x - (arg.data_x[arg.StimulusOf-arg.t_0])) / arg.px_per_deg

                if equation in ['fct_position', 'fct_saccade'] :
                    Title, ylabel, scale = 'Position Fit', 'Distance (°)', 1/2
                    data_1 = data_x 

                if equation in ['fct_velocity', 'fct_position'] :

                    if len(trials)==1:
                        ax = axs
                    else :
                        ax = axs[x]

                    velocity_NAN = ANEMO.velocity_NAN(self, **opt)[0]
                    old_latence, old_max, old_anti = ANEMO.classical_method.Full(velocity_NAN, arg.TargetOn-arg.t_0)

                    if equation=='fct_velocity' :

                        Title, ylabel, scale = 'Velocity Fit', 'Velocity (°/s)', 1
                        data_1 = velocity_NAN

                    #-------------------------------------------------
                    # FIT
                    #-------------------------------------------------
                    f = ANEMO.Fit.Fit_trial(self, data_trial=data_1, value_latence=old_latence, value_max=old_max, value_anti=old_anti, **opt)
                    #-------------------------------------------------

                    onset  = arg.TargetOn - arg.t_0 # TargetOn - temps_0
                    start_anti = f.values['start_anti']
                    v_anti = f.values['v_anti']
                    latence = f.values['latence']
                    tau = f.values['tau']
                    maxi = f.values['maxi']

                    if equation=='fct_velocity' :
                        result_fit = ANEMO.Equation.fct_velocity (x=np.arange(len(trackertime_s)), dir_target=arg.dir_target,
                                                                    start_anti=start_anti, v_anti=v_anti, latence=latence,
                                                                    tau=tau, maxi=maxi, do_whitening=False)

                    if equation=='fct_position' :

                        sacc, i = np.zeros(len(arg.trackertime), dtype=int), 0
                        for s in range(len(arg.saccades)):
                            sacc[i] = arg.saccades[s][0] # onset sacc
                            sacc[i+1] = arg.saccades[s][1] # end sacc
                            sacc[i+2] = arg.saccades[s][2] # time sacc
                            i = i+3

                        result_fit = ANEMO.Equation.fct_position(x=np.arange(len(trackertime_s)), data_x=arg.data_x, saccades=sacc,
                                                        nb_sacc=len(arg.saccades), dir_target=arg.dir_target, start_anti=start_anti,
                                                        v_anti=v_anti, latence=latence, tau=tau, maxi=maxi, t_0=arg.t_0,
                                                        px_per_deg=arg.px_per_deg, before_sacc=before_sacc, after_sacc=after_sacc, do_whitening=False)

                    if report is not None :
                        results.append(result_deg.fit_report())

                    ax.plot(trackertime_s[:int(start_anti)],              result_fit[:int(start_anti)],              c='k', linewidth=2)
                    ax.plot(trackertime_s[int(start_anti):int(latence)],  result_fit[int(start_anti):int(latence)],  c='r', linewidth=2)
                    ax.plot(trackertime_s[int(latence):int(latence)+250], result_fit[int(latence):int(latence)+250], c='darkred', linewidth=2)

                    y = {}
                    for y_pos in [int(start_anti), int(latence), int(latence)+50, int(latence)+250, int(latence)+400] :
                        if np.isnan(result_fit[y_pos]) :
                            y[y_pos] = data_1[y_pos]
                        else :
                            y[y_pos] = result_fit[y_pos]
                    #-----------------------------------------------------------------------------
                    # V_a ------------------------------------------------------------------------
                    ax.text((trackertime_s[int(start_anti)]+trackertime_s[int(latence)])/2, y[int(start_anti)]-15*scale,
                            r"A$_a$ = %0.2f °/s$^2$"%(v_anti), color='r', fontsize=t_label/1.5, ha='center')

                    # Start_a --------------------------------------------------------------------
                    ax.text(trackertime_s[int(start_anti)]-25, -35*scale, "Start anticipation = %0.2f ms"%(start_anti-onset),
                            color='k', alpha=0.7, fontsize=t_label/1.5, ha='right')
                    ax.bar(trackertime_s[int(start_anti)], 80*scale, bottom=-40*scale, color='k', width=4, linewidth=0, alpha=0.7)

                    # latence --------------------------------------------------------------------
                    ax.text(trackertime_s[int(latence)]+25, -35*scale, "Latency = %0.2f ms"%(latence-onset),
                            color='firebrick', fontsize=t_label/1.5, va='center')
                    ax.bar(trackertime_s[int(latence)], 80*scale, bottom=-40*scale, color='firebrick', width=4, linewidth=0, alpha=1)

                    # tau ------------------------------------------------------------------------
                    ax.text(trackertime_s[int(latence)]+70+t_label, y[int(latence)],
                            r"= %0.2f"%(tau), color='darkred',va='bottom', fontsize=t_label/1.5)
                    ax.annotate(r'$\tau$', xy=(trackertime_s[int(latence)]+50, y[int(latence)+50]), xycoords='data', fontsize=t_label/1., color='darkred', va='bottom',
                                xytext=(trackertime_s[int(latence)]+70, y[int(latence)]), textcoords='data', arrowprops=dict(arrowstyle="->", color='darkred'))

                    # Max ------------------------------------------------------------------------
                    ax.text(TargetOn_s+475, (y[int(latence)]+y[int(latence)+250])/2,
                            "Steady State = %0.2f °/s"%(maxi), color='k', va='center', fontsize=t_label/1.5)
                    #-----------------------------------------------------------------------------

                    if equation=='fct_velocity' :
                        # V_a ------------------------------------------------------------------------
                        ax.annotate('', xy=(trackertime_s[int(latence)], y[int(latence)]-3), xycoords='data', fontsize=t_label/1.5,
                                xytext=(trackertime_s[int(start_anti)], y[int(start_anti)]-3), textcoords='data', arrowprops=dict(arrowstyle="->", color='r'))
                        # Max ------------------------------------------------------------------------
                        ax.annotate('', xy=(TargetOn_s+450, y[int(latence)]), xycoords='data', fontsize=t_label/1.5,
                                    xytext=(TargetOn_s+450, y[int(latence)+250]), textcoords='data', arrowprops=dict(arrowstyle="<->"))
                        ax.plot(trackertime_s, np.zeros(len(trackertime_s)), '--k', linewidth=1, alpha=0.5)
                        ax.plot(trackertime_s[int(latence):], np.ones(len(trackertime_s[int(latence):]))*y[int(latence)+400], '--k', linewidth=1, alpha=0.5)

                    #-----------------------------------------------------------------------------
                    if x == int((len(trials)-1)/2) :
                        ax.set_ylabel(ylabel, fontsize=t_label)

                    if x!= (len(trials)-1) :
                        ax.set_xticklabels([])

                    #-----------------------------------------------------------------------------

                if equation=='fct_saccade' :
                    axs0 = gridspec.GridSpecFromSubplotSpec(2, len(arg.saccades), subplot_spec=axs[x], hspace=0.45, wspace=0.2)
                    ax = plt.Subplot(fig, axs0[0,:])
                    ax.set_ylabel(ylabel, fontsize=t_label)
                    fig.add_subplot(ax)

                    y = 0
                    for s in range(len(arg.saccades)):

                        if len(arg.saccades)==1:
                            ax1 = axs0[1]
                        else :
                            ax1 = plt.Subplot(fig, axs0[1,s])

                        data_sacc  = data_1[arg.saccades[s][0]-arg.t_0-before_sacc:arg.saccades[s][1]-arg.t_0+after_sacc]
                        time = trackertime_s[arg.saccades[s][0]-arg.t_0-before_sacc:arg.saccades[s][1]-arg.t_0+after_sacc]

                        #-------------------------------------------------
                        # FIT
                        #-------------------------------------------------
                        f = ANEMO.Fit.Fit_trial(self, data_trial=data_sacc, **opt)
                        #-------------------------------------------------

                        if report is not None :
                            results.append(f.fit_report())
                        T0,  t1,  t2,  tr = f.values['T0'], f.values['t1'], f.values['t2'], f.values['tr']
                        x_0, x1, x2, tau = f.values['x_0'], f.values['x1'], f.values['x2'], f.values['tau']


                        fit = ANEMO.Equation.fct_saccade(range(len(data_sacc)), x_0, tau, x1, x2, T0, t1, t2, tr,do_whitening=False)
                        ax.plot(time, fit, color='r')

                        ax1.plot(time, data_sacc, color='k', alpha=0.4)
                        ax1.plot(time, fit, color='r')
                        
                        minx, maxx = min(time[0], T0 + time[0]), max(time[-1], T0+t1+t2+tr + time[0])# time[0], time[-1]
                        miny, maxy = min(data_sacc), max(data_sacc)
                        #-----------------------------------------------------------------------------
                        name = ['T0', 't1', 't2', 'tr', 'x_0', 'x1', 'x2', 'tau']
                        px = 0
                        for n in name :
                            ax1.text(maxx+(maxx-minx)/10, (maxy+(maxy-miny)/5)-px, "%s: %0.3f"%(n, f.values[n]) , color='k', 
                                            ha='right', va='center', fontsize=t_label/1.3, alpha=0.8)
                            px = px + ((maxy+(maxy-miny)/5)-(miny-(maxy-miny)/5))/(len(name)-1)
                        
                        # T0 -------------------------------------------------------------------------
                        ax1.axvspan(T0 + time[0], T0+t1 + time[0], color='r', alpha=0.2)
                        # T1 -------------------------------------------------------------------------
                        ax1.axvspan(T0+t1 + time[0], T0+t1+t2 + time[0], color='k', alpha=0.2)
                        # T2 -------------------------------------------------------------------------
                        ax1.axvspan(T0+t1+t2 + time[0], T0+t1+t2+tr + time[0], color='r', alpha=0.2)
                        # x_0 -------------------------------------------------------------------------
                        ax1.hlines(x_0, minx, maxx, color='k', lw=1, linestyles='--', alpha=0.3)
                        # x1 -------------------------------------------------------------------------
                        ax1.hlines(x1+x_0, minx, maxx, color='k', lw=1, linestyles='--', alpha=0.5)
                        # x2 -------------------------------------------------------------------------
                        ax1.hlines(x2+x_0, minx, maxx, color='r', lw=1, linestyles='--', alpha=0.5)

                        #-----------------------------------------------------------------------------
                        ax1.set_title('Saccade %s'%(s+1), fontsize=t_label, x=0.5, y=1.05)
                        ax1.axis([minx-(maxx-minx)/100, maxx+(maxx-minx)/8, miny-(maxy-miny)/3, maxy+(maxy-miny)/3])
                        #-----------------------------------------------------------------------------
                        ax1.set_xlabel('Time (ms)', fontsize=t_label)
                        if y==0 :
                            ax1.set_ylabel('Distance (°)', fontsize=t_label)
                        ax1.tick_params(labelsize=t_label/2 , bottom=True, left=True)
                        #-----------------------------------------------------------------------------
                        fig.add_subplot(ax1)

                        y=y+1

                ax.plot(trackertime_s, data_1, color='k', alpha=0.4)
                ax = ANEMO.Plot.deco(self, ax, **opt)
                ax.axis([-750, 750, -39.5*scale, 39.5*scale])
                if x==0 :
                    ax.set_title(Title, fontsize=t_titre, x=0.5, y=1.05)

                x=x+1

            if equation in ['fct_velocity', 'fct_position'] :
                plt.tight_layout() # pour supprimer les marge trop grande
                plt.subplots_adjust(hspace=0) # pour enlever espace entre les figures

            if report is None :
                return fig, axs
            else :
                return fig, axs, results

        def plot_Full_data(self, data, show='velocity', N_blocks=None,
                        N_trials=None, list_events=None,
                        fig_width=12, t_titre=20, t_label=14,
                        stop_search_misac=None, file_fig=None) :

            '''
            Save the full data figure

            Parameters
            ----------

            data : list
                edf data for the trials recorded by the eyetracker transformed by the read_edf function of the edfreader module
            show : str
                if 'velocity' show velocity of the eye
                if 'position' show the position of the eye
                if 'saccades' shows the saccades of the eye

            N_blocks : int
                number of blocks
                if None went searched in param_exp
            N_trials : int
                number of trials per block
                if None went searched in param_exp

            list_events : list
                list of the names of the events of the trial
                by default : ['StimulusOn\n', 'StimulusOff\n', 'TargetOn\n', 'TargetOff\n']

            stop_search_misac : int
                stop search of micro_saccade
                if None: stops searching at the end of fixation + 100ms

            fig_width : int
                figure size
            t_titre : int
                size of the title of the figure
            t_label : int
                size x and y label

            file_fig : str
                name of file figure reccorded
                if None file_fig is show

            Returns
            -------
            save the figure
            '''

            import matplotlib.pyplot as plt

            if N_blocks is None :
                N_blocks = Test.test_value('N_blocks', self.param_exp)
            if N_trials is None :
                N_trials = Test.test_value('N_trials', self.param_exp)

            for block in range(N_blocks) :
                fig, axs = ANEMO.Plot.plot_data(self, data, show=show, trials=list(np.arange(N_trials)), block=block,
                                    N_trials=N_trials, list_events=list_events,
                                    fig_width=fig_width, t_titre=t_titre, t_label=t_label,
                                    stop_search_misac=stop_search_misac, name_trial_show=True)

                file_fig = Test.test_None(file_fig, show)
                plt.savefig(file_fig+'_%s.pdf'%(block+1))
                plt.close()

        ######################################################################################

        '''
        Parameters
        ----------
        ax : AxesSubplot
            ax sur lequel le figure doit être afficher


        data : list
            données edf enregistré par l'eyetracker transformé par la fonction read_edf du module edfreader
        data_x : ndarray
            position x pour un essaie enregistré par l'eyetracker transformé par la fonction read_edf du module edfreader
        data_y : ndarray
            position y pour un essaie enregistré par l'eyetracker transformé par la fonction read_edf du module edfreader
        velocity : ndarray
            vitesse de l'œil en deg/sec

        trials : int ou  list
            numéro des essais que l'on veux afficher
        block : int
            numéro du block
        N_trials : int
            nombre de trial par block
        list_events : list
            liste des noms des évenements dans le fichier asc ['onset fixation', 'end fixation', 'début poursuite', 'end poursuite']


        time_sup : int
            temps qui vas être supprimer pour effectuer le fit (pour les données qui sont moins bonne a la end de l'essai)
        stop_search_misac : int
            stop recherche de micro_saccade, si None alors arrête la recherche à la fin de la fixation +100ms


        saccades : list
            liste des saccades edf pour un essaie enregistré par l'eyetracker transformé par la fonction read_edf du module edfreader
        StimulusOn : int
            temps ou le point de fixation apparait
        StimulusOf : int
            temps ou le point de fixation disparait
        TargetOn : int
            temps ou la cible à suivre apparait
        TargetOff : int
            temps ou la cible à suivre disparait
        trackertime : ndarray
            temps du tracker
        start : int
            temps 0 sur la figure



        param_fit : dic
            dictionnaire des parametre du fit, chaque parametre est une liste [value fit, min, max]
        step_fit : nombre de step pour le fit

        V_X : float
            vitesse de la cible en pixel/s
        RashBass : int
            temps que met la cible a arriver au centre de l'écran en ms (pour reculer la cible à t=0 de sa vitesse * latence=RashBass)
        stim_tau : float
        screen_width_px : int
            widht ecran en pixel
        screen_height_px : int
            height écran en pixel


        report : NoneType or bool
            si != None renvoie le rapport fit

        plot : str
            si 'fonction' n'affiche que la fonction exponentiel
            si 'Fitvelocity' affiche la vitesse œil + fit
        plot : str
            si 'fonction' n'affiche que la fonction exponentiel
            si 'velocity' n'affiche que la vitesse de l'œil
            si 'Fitvelocity' affiche la vitesse œil + fit
        fig_width : int
            taille figure
        t_titre : int
            taille titre de la figur
        t_label : int
            taille x et y label


        Returns
        -------
        fig : matplotlib.figure.Figure
            figure
        ax : AxesSubplot
            figure
        '''



