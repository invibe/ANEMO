#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


N_freq = 1301
def whitening_filt(N_freq, white_f_0, white_alpha, white_steepness) :

    """
    Returns the envelope of the whitening filter.

        then we return a 1/f spectrum based on the assumption that the structure of signals
        is self-similar and thus that the Fourier spectrum scales a priori in 1/f.

    """

    freq = np.fft.fftfreq(N_freq, d=1.)

    K  = np.abs(freq)**(white_alpha)
    K *= np.exp(-(np.abs(freq)/white_f_0)**white_steepness)
    K /= np.mean(K)

    return freq, K


def whitening(position, white_f_0=.4, white_alpha=.5, white_steepness=4) :

    """
    Returns the whitened image

    /!\ position must not contain Nan

    """

    try :                   N_freq = position.shape[0]
    except AttributeError : N_freq = len(position)

    freq, K = whitening_filt(N_freq=N_freq, white_f_0=white_f_0, white_alpha=white_alpha, white_steepness=white_steepness)
    f_position = np.fft.fft(position)

    return np.real(np.fft.ifft(f_position*K))



class Test(object) :
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
        print_crash : str (default None)
            message to return
            if ``None``, the message is "%s is not defined"%name

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
        value : int, float (default None)
            new value has given
        crash : bool (default True)
            if ``True`` if the value of name is ``None`` then the program stops and returns print_crash
        print_crash : str (default None)
            message to return
            if ``None``, the message is "%s is not defined"%name

        Returns
        -------
        value : int
            value of the variable
        or Raise
        '''

        new_value = name

        try :
            new_value = dic[name]
            return new_value

        except KeyError :
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

        None_or_nan = False

        if type(var) in [float, np.float64, np.float32, np.float16] :
            if np.isnan(var) :
                None_or_nan = True
                return value

        elif var is None :
            None_or_nan = True
            return value

        if None_or_nan is False :
            return var



class ANEMO(object) :
    """
    ANEMO allows you to perform Fits on data of Smooth Pursuite Eyes Movements.
    You could use the functions 'velocity', 'position' and 'saccades' already present, but also your own functions.
    It must be initialized with the parameters of the experiment :

    **param_exp** (dict) :
        dictionary containing the parameters of the experiment :

        **'px_per_deg'** (float) - number of px per degree for the experiment ::

                tan = np.arctan((screen_width_cm/2)/viewing_Distance_cm)
                screen_width_deg = 2. * tan * 180/np.pi
                px_per_deg = screen_width_px / screen_width_deg

        **'dir_target'** (list(list(int))) - list of lists for each block containing the direction of the target for each trial ::

                #the direction of the target is to -1 for left 1 for right
                dir_target = param_exp['dir_target'][block][trial]

        or **'p'** (ndarray) - ndarray containing for each trial of each block the direction of the target, its probability of direction and the switches of this probability ::

                # the direction of the target is to 0 for left 1 for right
                dir_target = param_exp['p'][trial, block, 0]
                proba = param_exp['p'][trial, block, 1]
                switch = param_exp['p'][trial, block, 2]

        **'N_trials'** (int) - number of trials per block

        **'N_blocks'** (int) - number of blocks

        **'observer'** (str, optional) - subject name

        **'list_events'** (list(str), optional) - list of the names of the events of the trial ::

            list_events = ['onset fixation', 'end fixation',
                           'start pursuit', 'end pursuit']

            by default :
                list_events = ['StimulusOn\\n', 'StimulusOff\\n',
                               'TargetOn\\n', 'TargetOff\\n']

        optional not obligatory, just to display the target in ANEMO.Plot :

            **'V_X_deg'** (float, optional) - target velocity in deg/s

            **'stim_tau'** (float, optional) - presentation time of the target

            **'RashBass'** (int, optional) - the time the target has to arrive at the center of the screen in ms, to move the target back to t=0 of its ``RashBass = velocity*latency``

    """

    def __init__(self, param_exp=None) :
        self.param_exp = Test.crash_None('param_exp', param_exp)


    def arg(self, data_trial, dir_target=None, trial=None, block=None) :

        '''
        Generates a dictionary of the parameters of the trial

        Parameters
        ----------
        data_trial : list
            edf data for a trial recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`

        dir_target : int, or None (default None)
            the direction of the target \n
            if ``None`` went searched in ``param_exp`` (see :mod:`~ANEMO.ANEMO.ANEMO`), can be : ::

                - param_exp['dir_target'][block][trial] = 1 or -1
                - param_exp['p'][trial, block, 0] = 0 or 1

        trial : int, optional (default None)
            number of the trial in the block
        block : int, optional (default None)
            block number

        Warning
        -------
        **some parameters must be defined :**

        - if ``dir_target`` is ``None`` :
            - **trial**
            - **block**

        Returns
        -------
        arg : dict
            dictionary of the parameters of the trial
        '''

        import easydict

        list_events = Test.test_value('list_events', self.param_exp, value=['StimulusOn\n', 'StimulusOff\n', 'TargetOn\n', 'TargetOff\n'])
        kwargs = {}

        for events in range(len(data_trial['events']['msg'])) :
            if data_trial['events']['msg'][events][1] == list_events[0] : kwargs["StimulusOn"] = data_trial['events']['msg'][events][0]
            if data_trial['events']['msg'][events][1] == list_events[1] : kwargs["StimulusOf"] = data_trial['events']['msg'][events][0]
            if data_trial['events']['msg'][events][1] == list_events[2] : kwargs["TargetOn"]   = data_trial['events']['msg'][events][0]
            if data_trial['events']['msg'][events][1] == list_events[3] : kwargs["TargetOff"]  = data_trial['events']['msg'][events][0]

        kwargs.update({
                       "data_x"      : data_trial['x'],
                       "data_y"      : data_trial['y'],
                       "trackertime" : data_trial['trackertime'],
                       "saccades"    : data_trial['events']['Esac'],
                       "t_0"         : data_trial['trackertime'][0],
                       })

        kwargs["px_per_deg"] = Test.test_value('px_per_deg', self.param_exp, print_crash="px_per_deg is not defined in param_exp")

        kwargs["dir_target"] = dir_target
        if dir_target is None :
            try :
                kwargs["dir_target"] = self.param_exp['dir_target'][block][trial]
            except :
                try : kwargs["dir_target"] = (self.param_exp['p'][trial, block, 0]*2)-1
                except : pass

        return easydict.EasyDict(kwargs)


    def filter_data(self, data, cutoff=30, sample_rate=1000) :

        '''
        Return the filtering of data

        Parameters
        ----------
        data : ndarray
            position or velocity for the trial recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`

        cutoff : int (default 30)
            the critical frequencies for cutoff of filter
        sample_rate : int (default 1000)
            sampling rate of the recording

        Returns
        -------
        filt_data : ndarray
            Filtered position or filtered velocity of the eye
        '''

        from scipy import signal

        nyq_rate = sample_rate/2                # The Nyquist rate of the signal.
        Wn = cutoff/nyq_rate
        N = 2                                   # The order of the filter.

        b, a = signal.butter(N, Wn, 'lowpass')  # Butterworth digital and analog filter design.
        filt_data = signal.filtfilt(b, a, data) # Apply a digital filter forward and backward to a signal.

        return filt_data


    def data_deg(self, data, StimulusOf, t_0, saccades, before_sacc, after_sacc, filt=None, cutoff=30, sample_rate=1000, **opt) :

        '''
        Return the position of the eye in deg

        Parameters
        ----------
        data : ndarray
            position for the trial recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`

        StimulusOf : int
            time when the stimulus disappears
        t_0 : int
            time 0 of the trial

        saccades : list
            list of edf saccades for the trial recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`
        before_sacc : int
            time to delete before saccades
        after_sacc : int
            time to delete after saccades

        filt : str {'position', 'velocity-position'} or None (default None)
            to filter the data can be :
                - ``'position'`` : filter the position,
                - ``'velocity-position'`` : filter the position then the speed
                - ``None`` : the data will not be filtered
        cutoff : int, optional (default 30)
            the critical frequencies for cutoff of filter
        sample_rate : int, optional (default 1000)
            sampling rate of the recording for the filtre

        Returns
        -------
        data_deg : ndarray
            position of the eye in deg
        '''

        px_per_deg = Test.test_value('px_per_deg', self.param_exp, print_crash="px_per_deg is not defined in param_exp")

        if filt in ['position', 'velocity-position'] :
            data = ANEMO.filter_data(self, data, cutoff, sample_rate)

        t_data_0 = StimulusOf-t_0
        for s in range(len(saccades)) :
            for x_data in np.arange((saccades[s][0]-t_0-before_sacc), (saccades[s][1]-t_0+after_sacc)) :
                if x_data == StimulusOf-t_0 :
                    if (saccades[s][0]-t_0-before_sacc)-t_data_0 <= (saccades[s][1]-t_0+after_sacc)-t_data_0 :
                        t_data_0 = saccades[s][0]-t_0-before_sacc-1
                    else :
                        t_data_0 = saccades[s][1]-t_0+after_sacc+1

        data_deg = (data - (data[t_data_0]))/px_per_deg

        return data_deg


    def velocity_deg(self, data_x, filt=None, cutoff=30, sample_rate=1000) :

        '''
        Return the velocity of the eye in deg/sec

        Parameters
        ----------
        data_x : ndarray
            x position for the trial recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`

        filt : str {'position', 'velocity', 'velocity-position'} or None (default None)
            to filter the data can be :
                - ``'position'`` : filter the position,
                - ``'velocity'`` : filter the speed,
                - ``'velocity-position'`` : filter the position then the speed
                - ``None`` : the data will not be filtered
        cutoff : int, optional (default 30)
            the critical frequencies for cutoff of filter
        sample_rate : int, optional (default 1000)
            sampling rate of the recording for the filtre


        Returns
        -------
        gradient_deg : ndarray
            velocity of the eye in deg/sec
        '''


        px_per_deg = Test.test_value('px_per_deg', self.param_exp, print_crash="px_per_deg is not defined in param_exp")

        if filt in ['position', 'velocity-position'] :
            data_x = ANEMO.filter_data(self, data_x, cutoff=cutoff, sample_rate=sample_rate)

        gradient_x = np.gradient(data_x)
        gradient_deg = gradient_x * 1/px_per_deg * 1000 # gradient in deg/sec

        if filt in ['velocity', 'velocity-position'] :
            gradient_deg = ANEMO.filter_data(self, gradient_deg, cutoff=cutoff, sample_rate=sample_rate)

        return gradient_deg


    def detec_misac(self, velocity_x, velocity_y, t_0=0, VFAC=5, mindur=5, maxdur=100, minsep=30) :

        '''
        Detection of micro-saccades not detected by eyelink in the data

        Parameters
        ----------
        velocity_x : ndarray
            velocity x of the eye in deg/sec
        velocity_y : ndarray
            velocity y of the eye in deg/sec

        t_0 : int, optional (default 0)
            time 0 of the trial

        VFAC : int, optional (default 5)
            relative velocity threshold
        mindur : int, optional (default 5)
            minimal saccade duration (ms)
        maxdur : int, optional (default 100)
            maximal saccade duration (ms)
        minsep : int, optional (default 30)
            minimal time interval between two detected saccades (ms)

        Returns
        -------
        misaccades : list(list(int))
            list of lists, each containing ``[start micro-saccade, end micro-saccade]``
        '''

        msdx = np.sqrt((np.nanmedian(velocity_x**2))-((np.nanmedian(velocity_x))**2))
        msdy = np.sqrt((np.nanmedian(velocity_y**2))-((np.nanmedian(velocity_y))**2))

        radiusx, radiusy = VFAC*msdx, VFAC*msdy

        test = (velocity_x/radiusx)**2 + (velocity_y/radiusy)**2
        index = [x for x in range(len(test)) if test[x] > 1]

        dur, start_misaccades, k = 0, 0, 0
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
        Eliminates saccades detected

        Parameters
        ----------
        velocity : ndarray
            velocity of the eye in deg/sec
        saccades : list
            list of edf saccades for the trial recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`

        trackertime : ndarray
            the time of the tracker
        before_sacc : int
            time to delete before saccades
        after_sacc : int
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

    def velocity_NAN(self, data_x, data_y, saccades, trackertime, TargetOn,
                     before_sacc=5, after_sacc=15, stop_search_misac=None,
                     filt=None, cutoff=30, sample_rate=1000, **opt) :

        '''
        Returns velocity of the eye in deg / sec without the saccades

        Parameters
        ----------
        data_x : ndarray
            x position for the trial recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`
        data_y : ndarray
            y position for the trial recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`

        saccades : list
            list of edf saccades for the trial recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`
        trackertime : ndarray
            the time of the tracker
        TargetOn : int
            time when the target to follow appears

        before_sacc : int, optional (default 5)
            time to delete before saccades
        after_sacc : int, optional (default 15)
            time to delete after saccades
        stop_search_misac : int, optional (default None)
            stop search of micro_saccade -- if ``None`` stops searching at the ``end of fixation + 100ms``

        filt : str {'position', 'velocity', 'velocity-position'} or None (default None)
            to filter the data can be :
                - ``'position'`` : filter the position,
                - ``'velocity'`` : filter the speed,
                - ``'velocity-position'`` : filter the position then the speed
                - ``None`` : the data will not be filtered
        cutoff : int, optional (default 30)
            the critical frequencies for cutoff of filter
        sample_rate : int, optional (default 1000)
            sampling rate of the recording for the filtre


        Returns
        -------
        velocity : ndarray
            velocity of the eye in deg / sec without the saccades
        '''

        stop_search_misac = Test.test_None(stop_search_misac, value=TargetOn-trackertime[0]+100)

        velocity   = ANEMO.velocity_deg(self, data_x=data_x, filt=None)
        velocity_y = ANEMO.velocity_deg(self, data_x=data_y, filt=None)

        new_saccades = saccades.copy()
        misac = ANEMO.detec_misac(self, velocity_x=velocity[:stop_search_misac], velocity_y=velocity_y[:stop_search_misac], t_0=trackertime[0])
        new_saccades.extend(misac)

        if filt != False :
            velocity = ANEMO.velocity_deg(self, data_x=data_x, filt=filt, cutoff=cutoff, sample_rate=sample_rate)

        velocity_NAN = ANEMO.supp_sacc(self, velocity=velocity, saccades=new_saccades, trackertime=trackertime, before_sacc=before_sacc, after_sacc=after_sacc)

        return velocity_NAN



    class classical_method(object) :
        """
        Function used to calculate in a 'classical' way :

                - the latency of the pursuit,
                - the steady_state velocity of the pursuit, and
                - the anticipation of the pursuit, during smooth pursuit
        """

        def latency(velocity_NAN, w1=300, w2=50, off=50, crit=0.17) :

            '''
            Return the latency of the pursuit during a smooth pursuit calculated in a 'classic' way

            Parameters
            ----------
            velocity_NAN : ndarray
                velocity of the eye in deg/sec without the saccades

            w1 : int, optional (default 300)
                size of the window 1 in ms
            w2 : int, optional (default 50)
                size of the window 2 in ms
            off : int, optional (default 50)
                gap between the two windows
            crit : float, optional (default 0.17)
                difference criterion between the two linregress detecting if the pursuit begins

            Returns
            -------
            latency : int
                the latency in ms
            '''

            from scipy import stats

            time = np.arange(len(velocity_NAN))
            tps = time

            a = None
            for t in range(len(time)-(w1+off+w2)-300) :
                slope1, intercept1, r_, p_value, std_err = stats.linregress(tps[t:t+w1], velocity_NAN[t:t+w1])
                slope2, intercept2, r_, p_value, std_err = stats.linregress(tps[t+w1+off:t+w1+off+w2], velocity_NAN[t+w1+off:t+w1+off+w2])
                diff = abs(slope2) - abs(slope1)
                if abs(diff) >= crit :
                    a = True
                    tw = time[t:t+w1+off+w2]
                    timew = np.linspace(np.min(tw), np.max(tw), len(tw))

                    fitLine1 = slope1 * timew + intercept1
                    fitLine2 = slope2 * timew + intercept2

                    idx = np.argwhere(np.isclose(fitLine1, fitLine2, atol=0.1)).reshape(-1)
                    old_latency = timew[idx]
                    break

            if a is None or len(old_latency)==0 : old_latency = [np.nan]

            return old_latency[0]

        def steady_state(velocity_NAN, TargetOn_0) :

            '''
            Return the steady_state velocity of the pursuit during a smooth pursuit calculated in a 'classic' way

            Parameters
            ----------
            velocity_NAN : ndarray
                velocity of the eye in deg/sec without the saccades

            TargetOn_0 : int
                time since the beginning of the trial when the target to follow appears

            Returns
            -------
            steady_state : int
                the steady_state velocity in deg/s
            '''

            return abs(np.nanmean(velocity_NAN[TargetOn_0+400:TargetOn_0+600]))

        def anticipation(velocity_NAN, TargetOn_0) :

            '''
            Return the anticipation of the pursuit during a smooth pursuit calculated in a 'classic' way

            Parameters
            ----------
            velocity_NAN : ndarray
                velocity of the eye in deg/sec without the saccades

            TargetOn_0 : int
                time since the beginning of the trial when the target to follow appears

            Returns
            -------
            anticipation : int
                the anticipation in deg/s
            '''

            return np.nanmean(velocity_NAN[TargetOn_0-50:TargetOn_0+50])

        def Full(velocity_NAN, TargetOn_0, w1=300, w2=50, off=50, crit=0.17) :

            '''
            Return :

                - the latency of the pursuit,
                - the steady_state velocity of the pursuit,
                - the anticipation of the pursuit,

            during smooth pursuit calculated in a 'classical' way

            Parameters
            ----------
            velocity_NAN : ndarray
                velocity of the eye in deg/sec without the saccades

            TargetOn_0 : int
                time since the beginning of the trial when the target to follow appears

            w1 : int, optional (default 300)
                size of the window 1 to detect latency in ms
            w2 : int, optional (default 50)
                size of the window 2 to detect latency in ms
            off : int, optional (default 50)
                gap between the two windows to detect latency
            crit : float, optional (default 0.17)
                difference criterion between the two linregress detecting if the pursuit begins to detect latency

            Returns
            -------
            latency : int
                the latency in ms
            steady_state : int
                the steady_state velocity in deg/s
            anticipation : int
                the anticipation in deg/s
            '''


            latency      = ANEMO.classical_method.latency(velocity_NAN, w1, w2, off, crit)
            steady_state = ANEMO.classical_method.steady_state(velocity_NAN, TargetOn_0)
            anticipation = ANEMO.classical_method.anticipation(velocity_NAN, TargetOn_0)

            return latency, steady_state, anticipation/0.1



    class Equation(object) :
        """ Function used to perform the Fits """

        def fct_velocity(x, dir_target, start_anti, a_anti, latency, tau, steady_state, do_whitening) :

            '''
            Function reproducing the velocity of the eye during the smooth pursuit of a moving target

            Parameters
            ----------
            x : ndarray
                time of the function

            dir_target : int
                direction of the target -1 or 1
            start_anti : int
                time when anticipation begins
            a_anti : float
                velocity of anticipation in seconds
            latency : int
                time when the movement begins
            tau : float
                curve of the pursuit
            steady_state : float
                steady_state velocity reached during the pursuit
            do_whitening : bool
                if ``True`` return the whitened velocity

            Returns
            -------
            velocity : list
                velocity of the eye in deg/sec
            '''

            if start_anti >= latency :
                velocity = None

            else :
                a_anti = a_anti/1000 # to switch from sec to ms
                time = x
                velocity = []
                y = ((latency-1)-start_anti)*a_anti
                maxi = (dir_target*steady_state) - y

                for t in range(len(time)) :

                    if time[t] < start_anti :
                        velocity.append(0)
                    else :
                        if time[t] < latency :
                            velocity.append((time[t]-start_anti)*a_anti)
                        else :
                            velocity.append(maxi*(1-np.exp(-1/tau*(time[t]-latency)))+y)

                if do_whitening is True : velocity = whitening(velocity)

            return velocity

        def fct_velocity_sigmo(x, dir_target, start_anti, a_anti, latency, ramp_pursuit, steady_state, do_whitening) :

            '''
            Function reproducing the velocity of the eye during the smooth pursuit of a moving target

            Parameters
            ----------
            x : ndarray
                time of the function

            dir_target : int
                direction of the target -1 or 1
            start_anti : int
                time when anticipation begins
            a_anti : float
                velocity of anticipation in seconds
            latency : int
                time when the movement begins
            ramp_pursuit : float
                curve of the pursuit
            steady_state : float
                steady_state velocity reached during the pursuit
            do_whitening : bool
                if ``True`` return the whitened velocity

            Returns
            -------
            velocity : list
                velocity of the eye in deg/sec
            '''

            if start_anti >= latency :
                velocity = None

            else :
                a_anti = a_anti/1000 # to switch from sec to ms
                ramp_pursuit = -ramp_pursuit/1000
                time = x
                velocity = []

                e = np.exp(1)
                time_r = np.arange(-e, len(time), 1)

                y = ((latency-1)-start_anti)*a_anti
                maxi = (dir_target*steady_state) - y
                start_rampe = (maxi/(1+np.exp(((ramp_pursuit*time_r[0])+e))))

                for t in range(len(time)):
                    if time[t] < start_anti :
                        velocity.append(0)
                    else :
                        if time[t] < latency :
                            velocity.append((time[t]-start_anti)*a_anti)
                        else :
                            velocity.append((maxi/(1+np.exp(((ramp_pursuit*time_r[int(time[t]-latency)])+e))))+(y-start_rampe))

                if do_whitening is True : velocity = whitening(velocity)

            return velocity

        def fct_velocity_line(x, dir_target, start_anti, a_anti, latency, ramp_pursuit, steady_state, do_whitening) :

            '''
            Function reproducing the velocity of the eye during the smooth pursuit of a moving target

            Parameters
            ----------
            x : ndarray
                time of the function

            dir_target : int
                direction of the target -1 or 1
            start_anti : int
                time when anticipation begins
            a_anti : float
                velocity of anticipation in seconds
            latency : int
                time when the movement begins
            ramp_pursuit : float
                velocity of pursuit in seconds
            steady_state : float
                steady_state velocity reached during the pursuit
            do_whitening : bool
                if ``True`` return the whitened velocity

            Returns
            -------
            velocity : list
                velocity of the eye in deg/sec
            '''

            if start_anti >= latency :
                velocity = None
            else :
                a_anti = a_anti/1000 # to switch from sec to ms
                ramp_pursuit = dir_target*(ramp_pursuit)/1000
                time = x
                vitesse = []

                y = ((latency-1)-start_anti)*a_anti
                maxi = (dir_target*steady_state) - y
                end_ramp_pursuit = (maxi/ramp_pursuit) + latency

                for t in range(len(time)):
                    if time[t] < start_anti :
                        vitesse.append(0)
                    else :
                        if time[t] < latency :
                            vitesse.append((time[t]-start_anti)*a_anti)

                        else :
                            if latency >= end_ramp_pursuit :
                                vitesse.append(maxi)
                            else :
                                if time[t] < int(end_ramp_pursuit) :
                                    vitesse.append((time[t]-latency)*ramp_pursuit+y)
                                else :
                                    vitesse.append(maxi+y)

                if do_whitening is True : velocity = whitening(velocity)

            return vitesse

        def fct_position(x, data_x, saccades, nb_sacc, dir_target, start_anti, a_anti, latency, tau, steady_state, t_0, px_per_deg, before_sacc, after_sacc, do_whitening) :

            '''
            Function reproducing the position of the eye during the smooth pursuit of a moving target

            Parameters
            ----------
            x : ndarray
                time of the function
            data_x : ndarray
                position x of the eye during the trial
            saccades : ndarray the same size as data_x
                List of saccades perform during the trial ::

                    saccades = np.zeros(len(data_x))
                    i=0
                    # sacc is list of edf saccades for the trial recorded
                    # by the eyetracker transformed by the read_edf function
                    for s in range(len(sacc)) :
                        saccades[i]   = sacc[s][0] # onset sacc
                        saccades[i+1] = sacc[s][1] # end sacc
                        saccades[i+2] = sacc[s][2] # time sacc
                        i = i+3

            nb_sacc : int
                number of saccades during the trial
            dir_target : int
                direction of the target -1 or 1
            start_anti : int
                time when anticipation begins
            a_anti : float
                velocity of anticipation in seconds
            latency : int
                time when the movement begins
            tau : float
                curve of the pursuit
            steady_state : float
                steady_state velocity reached during the pursuit
            t_0 : int
                time 0 of the trial
            px_per_deg : float
                number of px per degree for the experiment
            before_sacc : int
                time to delete before saccades
            after_sacc : int
                time to delete after saccades
            do_whitening : bool
                if ``True`` return the whitened position

            Returns
            -------
            position : list
                position of the eye in deg
            '''

            if start_anti >= latency :
                pos = None

            else :
                ms = 1000
                a_anti = a_anti/ms
                steady_state   = steady_state/ms

                speed = ANEMO.Equation.fct_velocity(x=x, dir_target=dir_target, start_anti=start_anti, a_anti=a_anti, latency=latency, tau=tau, steady_state=steady_state, do_whitening=False)
                pos = np.cumsum(speed)


                i=0
                for s in range(nb_sacc) :
                    sacc = saccades[i:i+3] # obligation to have the independent variable at the same size :/
                                            # saccades[i] -> onset, saccades[i+1] -> end, saccades[i+2] -> time sacc

                    if do_whitening is True :
                        if int(sacc[0]-t_0)-int(before_sacc)-1 < len(pos) : a = pos[int(sacc[0]-t_0)-int(before_sacc)-1]
                        else :                                              a = pos[-1]
                    else :                                                  a = np.nan

                    if int(sacc[1]-t_0)+int(after_sacc)+1 <= len(pos) :
                        pos[int(sacc[0]-t_0)-int(before_sacc):int(sacc[1]-t_0)+int(after_sacc)] = a
                        if sacc[0]-t_0 >= int(latency-1) :
                            pos[int(sacc[1]-t_0)+int(after_sacc):] += ((data_x[int(sacc[1]-t_0)+int(after_sacc)]-data_x[int(sacc[0]-t_0)-int(before_sacc)-1])/px_per_deg) - np.mean(speed[int(sacc[0]-t_0):int(sacc[1]-t_0)]) * sacc[2]
                    else :
                        pos[int(sacc[0]-t_0)-int(before_sacc):] = a

                    i = i+3

                if do_whitening is True : pos = whitening(pos)

            return pos

        def fct_saccade(x, x_0, tau, x1, x2, T0, t1, t2, tr, do_whitening) :

            '''
            Function reproducing the position of the eye during the sacades

            Parameters
            ----------
            x : ndarray
                time of the function

            x_0 : float
                initial position of the beginning of the saccade in deg
            tau : float
                curvature of the saccade
            x1 : float
                maximum of the first curvature in deg
            x2 : float
                maximum of the second curvature in deg
            T0 : float
                time of the beginning of the first curvature after x_0 in ms
            t1 : float
                maximum time of the first curvature after T0 in ms
            t2 : float
                time of the maximum of the second curvature after t1 in ms
            tr : float
                time of the end of the second curvature after t2 in ms
            do_whitening : bool
                if ``True`` return the whitened position

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

            Umax1 = (1/tau) *    x1   / ((2*rho-1)*T1 - tau*(2-np.exp(-(rho*T1)/tau) - np.exp((1-rho)*T1/tau)))
            Umax2 = (1/tau) * (x2-x1) / ((2*r-1) * T2-T1)

            xx = []

            for t in time :
                if t < 0 :      xx.append( x_0)
                elif t < rhoT : xx.append(x_0 +       Umax1*tau * ((t)    - tau*(1-np.exp(-t/tau))))
                elif t < T1 :   xx.append(x_0 + (x1 + Umax1*tau * ((T1-t) + tau*(1-np.exp((T1-t)/tau)))))
                elif t < rT :   xx.append(x_0 + (x1 + Umax2*tau * ((t-T1) - tau*(1-np.exp(-(t-T1)/tau)))))
                elif t < TR :   xx.append(x_0 + (x2 + Umax2*tau * ((T2-t) + tau*(1-np.exp((T2-t)/tau)))))
                else :          xx.append(xx[-1])

            if do_whitening : xx = whitening(xx)

            return xx



    class Fit(object) :
        """
        Fit allows you to perform Fits on Smooth Pursuite Eyes Movements data.
        You could use the functions 'velocity', 'position' and 'saccades' already present, but also your own functions.
        It must be initialized with the parameters of the experiment
        (see :mod:`~ANEMO.ANEMO.ANEMO` for more precisions on the parameters)
        """

        def __init__(self, param_exp=None) :

            '''
            Parameters
            ----------
            param_exp : dict
                dictionary containing the parameters of the experiment :

                'px_per_deg': float
                        number of px per degree for the experiment
                            screen_width_deg = 2. * np.arctan((screen_width_cm/2) / viewing_Distance_cm) * 180/np.pi
                            px_per_deg = screen_width_px / screen_width_deg


                'dir_target' : list
                        list of lists for each block containing the direction of the target for each trial,
                        dir_target = param_exp['dir_target'][block][trial]
                            the direction of the target must be equal to -1 for left or 1 for right
                or 'p' : ndarray
                        ndarray containing for each trial of each block the direction of the target, its probability of direction and the switches of this probability
                        dir_target = param_exp['p'][trial, block, 0]
                            the direction of the target must be equal to 0 for left or 1 for right
                        proba = param_exp['p'][trial, block, 1]
                        swich = param_exp['p'][trial, block, 2]

                'N_trials' : int
                    number of trials per block

                'N_blocks' : int
                    number of blocks

                'observer' : str
                    subject name

                'list_events' : list
                    list of the names of the events of the trial : ['onset fixation', 'end fixation', 'start pursuit', 'end pursuit']
                    by default : ['StimulusOn\n', 'StimulusOff\n', 'TargetOn\n', 'TargetOff\n']
            '''

            ANEMO.__init__(self, param_exp)


        def generation_param_fit(self, equation='fct_velocity', data_x=None, dir_target=None,
                                 trackertime=None, TargetOn=None, StimulusOf=None, saccades=None,
                                 value_latency=None, value_steady_state=None, value_anti=None,
                                 before_sacc=5, after_sacc=15, **opt) :

            '''
            Generates the parameters and independent variables of the fit

            Parameters
            ----------
            equation : str {'fct_velocity', 'fct_position', 'fct_saccades'} (default 'fct_velocity')
                name of the equation for the fit :
                    - ``'fct_velocity'`` : generates the parameters for a fit velocity
                    -``'fct_velocity_sigmo'`` : generates the parameters for a fit velocity sigmoid
                    - ``'fct_velocity_line'`` : generates the parameters for a fit velocity linear
                    - ``'fct_position'`` : generates the parameters for a fit position
                    - ``'fct_saccades'`` : generates the parameters for a fit saccades

            data_x : ndarray, optional (default None)
                x position for the trial recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`
            dir_target : int, optional (default None)
                the direction of the target -- if ``None`` went searched in ``param_exp`` (see :mod:`~ANEMO.ANEMO.ANEMO`)

            trackertime : ndarray, optional (default None)
                the time of the tracker
            TargetOn : int, optional (default None)
                time when the target to follow appears
            StimulusOf : int, optional (default None)
                time when the stimulus disappears
            saccades : list, optional (default None)
                list of edf saccades for the trial recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`

            value_latency : int, optional (default None)
                value that takes the parameter latency to begin the fit -- if ``None`` or ``nan`` by default ``TargetOn-t_0+100``
            value_steady_state : float, optional (default 15)
                value that takes the parameter steady_state to begin the fit -- if ``None`` or ``nan`` by default ``15.``
            value_anti : float, optional (default 0)
                value that takes the parameter a_anti to begin the fit -- if ``None`` or ``nan`` by default ``0.``

            before_sacc : int, optional (default 5)
                time to remove before saccades -- it is advisable to put : ``5`` for ``'fct_velocity'`` and ``'fct_position'``, ``0`` for ``'fct_saccade'``
            after_sacc : int, optional (default 15)
                time to delete after saccades -- it is advisable to put ``15``

            Warning
            -------
            **some parameters must be defined :**

            - if ``equation`` is ``'fct_velocity'``, ``'fct_velocity_sigmo'``, ``'fct_velocity_line'`` or ``'fct_position'`` :
                - dir_target
                - trackertime
                - TargetOn
                - StimulusOf
                - saccades

            - if ``equation`` is ``'fct_position'`` or ``'fct_saccades'`` :
                - data_x

            Returns
            -------
            param_fit : dict
                dictionary containing the parameters of the fit
            inde_vars : dict
                dictionary containing the independent variables of the fit
            '''

            if equation in ['fct_velocity', 'fct_velocity_sigmo', 'fct_velocity_line', 'fct_position'] :

                TargetOn    = Test.crash_None('TargetOn', TargetOn)
                StimulusOf  = Test.crash_None('StimulusOf', StimulusOf)
                saccades    = Test.crash_None('saccades', saccades)

                trackertime = Test.crash_None('trackertime', trackertime)
                t_0 = trackertime[0]

                if dir_target is None : dir_target = Test.test_value('dir_target', self.param_exp, value=None)

                value_latency = Test.test_None(value_latency, value=TargetOn-t_0+100)
                value_anti    = Test.test_None(value_anti, value=0.)
                value_steady_state    = Test.test_None(value_steady_state, value=15.)

                #----------------------------------------------
                max_latency = []
                for s in range(len(saccades)) :
                    if (saccades[s][0]-t_0) >= (TargetOn-t_0+100) : max_latency.append((saccades[s][0]-t_0))
                if max_latency == [] :                              max_latency.append(len(trackertime))
                max_latency = max_latency[0]

                if value_latency >= max_latency-50 : value_latency = max_latency-150
                if value_latency > 250 :             value_latency = TargetOn-t_0+100
                #----------------------------------------------

                param_fit=[{'name':'steady_state', 'value':value_steady_state, 'min':5.,                 'max':40.,             'vary':True  },
                           {'name':'dir_target',   'value':dir_target,         'min':None,               'max':None,            'vary':False },
                           {'name':'a_anti',       'value':value_anti,         'min':-40.,               'max':40.,             'vary':True  },
                           {'name':'latency',      'value':value_latency,      'min':TargetOn-t_0+75,    'max':max_latency,     'vary':True  },
                           {'name':'start_anti',   'value':TargetOn-t_0-100,   'min':StimulusOf-t_0-200, 'max':TargetOn-t_0+75, 'vary':'vary'}]

                inde_vars={'x':np.arange(len(trackertime))}

            if equation in ['fct_velocity', 'fct_position'] :
                param_fit.extend([{'name':'tau',  'value':15., 'min':13., 'max':80., 'vary':'vary'}])

            if equation == 'fct_velocity_sigmo' :
                param_fit.extend([{'name':'ramp_pursuit', 'value':100, 'min':40., 'max':800., 'vary':'vary'}])

            if equation == 'fct_velocity_line' :
                param_fit.extend([{'name':'ramp_pursuit', 'value':40, 'min':40., 'max':80., 'vary':'vary'}])


            if equation == 'fct_position' :

                data_x = Test.crash_None('data_x', data_x)
                px_per_deg = Test.test_value('px_per_deg', self.param_exp, print_crash="px_per_deg is not defined in param_exp")

                param_fit.extend(({'name':'px_per_deg',  'value':px_per_deg,    'min':None, 'max':None, 'vary':False},
                                  {'name':'t_0',         'value':t_0,           'min':None, 'max':None, 'vary':False},
                                  {'name':'before_sacc', 'value':before_sacc,   'min':None, 'max':None, 'vary':False},
                                  {'name':'after_sacc',  'value':after_sacc,    'min':None, 'max':None, 'vary':False},
                                  {'name':'nb_sacc',     'value':len(saccades), 'min':None, 'max':None, 'vary':False}))

                sacc = np.zeros(len(trackertime))
                i=0
                for s in range(len(saccades)) :
                    sacc[i]   = saccades[s][0] # onset sacc
                    sacc[i+1] = saccades[s][1] # end sacc
                    sacc[i+2] = saccades[s][2] # time sacc
                    i = i+3

                inde_vars.update({'data_x':data_x, 'saccades':sacc})


            if equation == 'fct_saccade' :
                data_x = Test.crash_None('data_x', data_x)

                if (len(data_x)-10.) <= 10. : max_t1, max_t2 = 15., 12.
                else : max_t1, max_t2 = len(data_x)-10., len(data_x)-10.

                param_fit=[{'name':'x_0', 'value':data_x[0], 'min':data_x[0]-0.1, 'max':data_x[0]+0.1, 'vary':'vary'},
                           {'name':'tau', 'value':13.,       'min':5.,            'max':40.,           'vary':True  },
                           {'name':'T0',  'value':0.,        'min':-15,           'max':10,            'vary':True  },
                           {'name':'t1',  'value':15.,       'min':10.,           'max':max_t1,        'vary':True  },
                           {'name':'t2',  'value':12.,       'min':10.,           'max':max_t2,        'vary':'vary'},
                           {'name':'tr',  'value':1.,        'min':0.,            'max':15.,           'vary':'vary'},
                           {'name':'x1',  'value':2.,        'min':-5.,           'max':5.,            'vary':True  },
                           {'name':'x2',  'value':1.,        'min':-5.,           'max':5.,            'vary':'vary'}]

                inde_vars={'x':np.arange(len(data_x))}


            if equation not in ['fct_velocity', 'fct_velocity_sigmo', 'fct_velocity_line', 'fct_position', 'fct_saccade'] :
                param_fit, inde_vars = None, None

            return param_fit, inde_vars


        def Fit_trial(self, data_trial, equation='fct_velocity', data_x=None, dir_target=None,
                      trackertime=None, TargetOn=None, StimulusOf=None, saccades=None,
                      time_sup=280, step_fit=2, do_whitening=False,
                      param_fit=None, inde_vars=None,
                      value_latency=None, value_steady_state=15., value_anti=0.,
                      before_sacc=5, after_sacc=15, **opt) :

            '''
            Returns the result of the fit of a trial

            Parameters
            ----------
            data_trial : ndarray
                data for a trial :
                    - if ``equation`` is ``'fct_velocity'`` : velocity data in deg/sec
                    - if ``equation`` is ``'fct_velocity_sigmo'`` : velocity data in deg/sec
                    - if ``equation`` is ``'fct_velocity_line'`` : velocity data in deg/sec
                    - if ``equation`` is ``'fct_position'`` : position data in deg
                    - if ``equation`` is ``'fct_saccades'`` : position data in deg
                    - if ``equation`` is ``function`` : velocity or position

            equation : str {'fct_velocity', 'fct_position', 'fct_saccades'} or function (default 'fct_velocity')
                function or name of the equation for the fit :
                    - ``'fct_velocity'`` : does a data fit with function ``'fct_velocity'``
                    - ``'fct_velocity_sigmo'`` : does a data fit with function ``'fct_velocity_sigmo'``
                    - ``'fct_velocity_line'`` : does a data fit with function ``'fct_velocity_line'``
                    - ``'fct_position'`` : does a data fit with function ``'fct_position'``
                    - ``'fct_saccades'`` : does a data fit with function ``'fct_saccades'``
                    - ``function`` : does a data fit with function

            data_x : ndarray, optional (default None)
                x position for the trial recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`
            dir_target : int, optional (default None)
                the direction of the target -- if ``None`` went searched in ``param_exp`` (see :mod:`~ANEMO.ANEMO.ANEMO`)

            trackertime : ndarray, optional (default None)
                the time of the tracker, if ``None`` = ``np.arrange((len(data_trial))``
            TargetOn : int, optional (default None)
                time when the target to follow appears
            StimulusOf : int, optional (default None)
                time when the stimulus disappears
            saccades : list, optional (default None)
                list of edf saccades for the trial recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`

            time_sup : int, optional (default 280)
                time that will be deleted to perform the fit (for data that is less good at the end of the test)
            step_fit : int, optional (default 2)
                number of steps for the fit
            do_whitening : bool, optional (default False)
                if ``True`` return the whitened fit

            param_fit : dic, optional (default None)
                fit parameter dictionary, each parameter is a dict containing :
                    - ``'name'`` : name of the variable,
                    - ``'value'`` : initial value,
                    - ``'min'`` : minimum value,
                    - ``'max'`` : maximum value,
                    - ``'vary'`` :
                        - ``True`` if varies during fit,
                        - ``'vary'`` if only varies for step 2,
                        - ``False`` if not varies during fit

                if ``None`` generate by :func:`~ANEMO.ANEMO.ANEMO.Fit.generation_param_fit`
            inde_vars : dic, optional (default None)
                independent variable dictionary of fit -- if ``None`` generate by :func:`~ANEMO.ANEMO.ANEMO.Fit.generation_param_fit`

            value_latency : int, optional (default None)
                value that takes the parameter latency to begin the fit -- if ``None`` or ``nan`` by default ``TargetOn-t_0+100``
            value_steady_state : float, optional (default 15)
                value that takes the parameter steady_state to begin the fit -- if ``None`` or ``nan`` by default ``15.``
            value_anti : float, optional (default 0)
                value that takes the parameter a_anti to begin the fit -- if ``None`` or ``nan`` by default ``0.``

            before_sacc : int, optional (default 5)
                time to remove before saccades -- it is advisable to put : ``5`` for ``'fct_velocity'`` and ``'fct_position'``, ``0`` for ``'fct_saccade'``
            after_sacc : int, optional (default 15)
                time to delete after saccades -- it is advisable to put ``15``

            Warning
            -------
            **some parameters must be defined :**

            - if ``equation`` is ``'fct_position'`` :
                - data_x

            - if ``param_fit`` is ``None`` or ``inde_vars`` is ``None`` :
                - if ``equation`` is ``'fct_position'`` or ``'fct_saccades'`` :
                    - data_x
                - if ``equation`` is ``'fct_velocity'``, ``'fct_velocity_sigmo'``, ``'fct_velocity_line'`` or ``'fct_position'`` :
                    - dir_target
                    - trackertime
                    - TargetOn
                    - StimulusOf
                    - saccades

            Returns
            -------
            result : lmfit.model.ModelResult
            '''

            from lmfit import  Model, Parameters

            #-----------------------------------------------------------------------------
            if equation in ['fct_position'] : data_x = Test.crash_None('data_x', data_x)

            trackertime = Test.test_None(trackertime, value=np.arange(len(data_trial)))
            #-----------------------------------------------------------------------------

            if   step_fit == 1 : vary = True
            elif step_fit == 2 : vary = False

            if equation == 'fct_saccade' : time_sup = None ; data_x = data_trial

            if time_sup is not None :
                data_trial = data_trial[:-time_sup]
                trackertime = trackertime[:-time_sup]
                if equation == 'fct_position' : data_x = data_x[:-time_sup]

            if do_whitening :
                for x in range(len(data_trial)) :
                    if np.isnan(data_trial[x]) :
                        if x == 0 : data_trial[x] = 0
                        else :      data_trial[x] = data_trial[x-1]

                data_trial = whitening(data_trial)
                if equation in ['fct_position'] : data_x = whitening(data_x)

            if param_fit is None or inde_vars is None :
                opt = {'dir_target':dir_target,
                       'TargetOn':TargetOn,           'StimulusOf':StimulusOf, 'saccades':saccades,
                       'value_latency':value_latency, 'value_steady_state':value_steady_state, 'value_anti':value_anti,
                       'before_sacc':before_sacc,     'after_sacc':after_sacc}

            if param_fit is None : param_fit = ANEMO.Fit.generation_param_fit(self, equation=equation, trackertime=trackertime, data_x=data_x, **opt)[0]
            if inde_vars is None : inde_vars = ANEMO.Fit.generation_param_fit(self, equation=equation, trackertime=trackertime, data_x=data_x, **opt)[1]

            if equation == 'fct_velocity' :          equation = ANEMO.Equation.fct_velocity
            elif equation == 'fct_velocity_sigmo' :  equation = ANEMO.Equation.fct_velocity_sigmo
            elif equation == 'fct_velocity_line' :  equation = ANEMO.Equation.fct_velocity_line
            elif equation == 'fct_position' :        equation = ANEMO.Equation.fct_position
            elif equation == 'fct_saccade' :         equation = ANEMO.Equation.fct_saccade

            params = Parameters()
            model = Model(equation, independent_vars=inde_vars.keys())

            for num_par in range(len(param_fit)) :

                if 'expr' in param_fit[num_par].keys() :
                    params.add(param_fit[num_par]['name'], expr=param_fit[num_par]['expr'])
                else :
                    if param_fit[num_par]['vary'] == 'vary' : var = vary
                    else :                                    var = param_fit[num_par]['vary']
                    params.add(param_fit[num_par]['name'],
                               value=param_fit[num_par]['value'],
                               min=param_fit[num_par]['min'],
                               max=param_fit[num_par]['max'],
                               vary=var)

            params.add('do_whitening', value=do_whitening, vary=False)

            if step_fit == 1 :

                result_deg = model.fit(data_trial, params, nan_policy='omit', **inde_vars)

            elif step_fit == 2 :

                out = model.fit(data_trial, params, nan_policy='omit', **inde_vars)

                # make the other parameters vary now
                for num_par in range(len(param_fit)) :
                    if 'vary' in param_fit[num_par].keys() :
                        if param_fit[num_par]['vary'] == 'vary' :
                            out.params[param_fit[num_par]['name']].set(vary=True)

                result_deg = model.fit(data_trial, out.params, method='nelder', nan_policy='omit', **inde_vars)

            return result_deg


        def Fit_full(self, data, equation='fct_velocity', fitted_data='velocity',
                     N_blocks=None, N_trials=None,
                     time_sup=280, step_fit=2, do_whitening=False,
                     list_param_enre=None, param_fit=None, inde_vars=None,
                     before_sacc=5, after_sacc=15, stop_search_misac=None,
                     filt=None, cutoff=30, sample_rate=1000,
                     plot=None, file_fig=None, show_target=False,
                     fig_width=12, t_label=20, t_text=14) :

            '''
            Return the parameters of the fit present in list_param_enre

            Parameters
            ----------
            data : list
                edf data for the trials recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`

            equation : str {'fct_velocity', 'fct_position', 'fct_saccades'} or function (default 'fct_velocity')
                function or name of the equation for the fit :
                    - ``'fct_velocity'`` : does a data fit with function ``'fct_velocity'``
                    - ``'fct_velocity_sigmo'`` : does a data fit with function ``'fct_velocity_sigmo'``
                    - ``'fct_velocity_line'`` : does a data fit with function ``'fct_velocity_line'``
                    - ``'fct_position'`` : does a data fit with function ``'fct_position'``
                    - ``'fct_saccades'`` : does a data fit with function ``'fct_saccades'``
                    - ``function`` : does a data fit with function

            fitted_data : str {'velocity', 'position', 'saccade'}, optional (default 'velocity')
                nature of fitted data :
                    - ``'velocity'`` : fit velocity data for trial in deg/sec
                    - ``'position'`` : fit position data for trial in deg
                    - ``'saccade'`` : fit position data for sacades in trial in deg

            N_blocks : int, optional (default None)
                number of blocks -- if ``None`` went searched in ``param_exp`` (see :mod:`~ANEMO.ANEMO.ANEMO`)
            N_trials : int, optional (default None)
                number of trials per block  --  if ``None`` went searched in ``param_exp`` (see :mod:`~ANEMO.ANEMO.ANEMO`)

            time_sup : int, optional (default 280)
                time that will be deleted to perform the fit (for data that is less good at the end of the test)
            step_fit : int, optional (default 2)
                number of steps for the fit
            do_whitening : bool, optional (default False)
                if ``True`` return the whitened fit

            list_param_enre : list, optional (default None)
                list of fit parameters to record \n
                if ``None`` :
                    - if ``equation`` is ``'fct_velocity'`` or ``'fct_position'`` : ::

                        list_param_enre = ['fit', 'start_anti', 'a_anti',
                                           'latency', 'tau', 'steady_state',
                                           'saccades', 'old_anti',
                                           'old_steady_state', 'old_latency']

                    - if ``equation`` is ``'fct_saccades'`` : ::

                        list_param_enre = ['fit', 'T0', 't1', 't2', 'tr',
                                           'x_0', 'x1', 'x2', 'tau']

            param_fit : dic, optional (default None)
                fit parameter dictionary, each parameter is a dict containing :
                    - ``'name'`` : name of the variable,
                    - ``'value'`` : initial value,
                    - ``'min'`` : minimum value,
                    - ``'max'`` : maximum value,
                    - ``'vary'`` :
                        - ``True`` if varies during fit,
                        - ``'vary'`` if only varies for step 2,
                        - ``False`` if not varies during fit

                if ``None`` generate by :func:`~ANEMO.ANEMO.ANEMO.Fit.generation_param_fit`
            inde_vars : dic, optional (default None)
                independent variable dictionary of fit -- if ``None`` generate by :func:`~ANEMO.ANEMO.ANEMO.Fit.generation_param_fit`

            before_sacc : int, optional (default 5)
                time to remove before saccades -- it is advisable to put : ``5`` for ``'fct_velocity'`` and ``'fct_position'``, ``0`` for ``'fct_saccade'``
            after_sacc : int, optional (default 15)
                time to delete after saccades -- it is advisable to put ``15``
            stop_search_misac : int, optional (default None)
                stop search of micro_saccade -- if ``None`` stops searching at the ``end of fixation + 100ms``

            filt : str {'position', 'velocity', 'velocity-position'} or None (default None)
                to filter the data can be :
                    - ``'position'`` : filter the position,
                    - ``'velocity'`` : filter the speed,
                    - ``'velocity-position'`` : filter the position then the speed
                    - ``None`` : the data will not be filtered
            cutoff : int, optional (default 30)
                the critical frequencies for cutoff of filter
            sample_rate : int, optional (default 1000)
                sampling rate of the recording for the filtre

            plot : bool, optional (default None)
                if ``True`` : save the figure in ``file_fig``
            file_fig : str, optional (default None)
                name of file figure reccorded -- if ``None`` file_fig is ``'Fit'``
            show_target : bool, optional (default False)
                if ``True`` show the target on the plot

            fig_width : int, optional (default 12)
                figure size
            t_label : int, optional (default 20)
                size x and y label
            t_text : int, optional (default 14)
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

            if equation == 'fct_velocity' :       fitted_data = 'velocity'
            if equation == 'fct_velocity_sigmo' : fitted_data = 'velocity'
            if equation == 'fct_velocity_line' :  fitted_data = 'velocity'
            if equation == 'fct_position' :       fitted_data = 'position'
            if equation == 'fct_saccade' :        fitted_data = 'saccade'


            if plot is not None :
                import matplotlib.pyplot as plt
                if fitted_data == 'saccade' : import matplotlib.gridspec as gridspec

            if equation in ['fct_velocity', 'fct_position'] :
                list_param_enre = Test.test_None(list_param_enre, value=['start_anti', 'a_anti', 'latency', 'tau', 'steady_state', 'old_anti', 'old_steady_state', 'old_latency', 'goodness_of_fit'])

            if equation in ['fct_velocity_sigmo', 'fct_velocity_line'] :
                list_param_enre = Test.test_None(list_param_enre, value=['start_anti', 'a_anti', 'latency', 'ramp_pursuit', 'steady_state', 'old_anti', 'old_steady_state', 'old_latency', 'goodness_of_fit'])

            if equation == 'fct_saccade' :
                list_param_enre = Test.test_None(list_param_enre, value=['T0', 't1', 't2', 'tr', 'x_0', 'x1', 'x2', 'tau', 'goodness_of_fit'])

            if list_param_enre is None :
                print('Warning list_param_enre is None, no parameter will be returned !!!')
                list_param_enre = []

            opt_base = {'N_blocks':N_blocks,               'N_trials':N_trials,
                        'time_sup':time_sup,               'step_fit':step_fit,       'do_whitening':do_whitening,
                        'list_param_enre':list_param_enre, 'param_fit':param_fit,     'inde_vars':inde_vars,
                        'before_sacc':before_sacc,         'after_sacc':after_sacc,   'stop_search_misac':stop_search_misac,
                        'filt':filt, 'cutoff':cutoff,      'sample_rate':sample_rate,
                        'show_target':show_target,
                        'fig_width':fig_width,             't_label':t_label,         't_text':t_text}

            param = {'N_blocks':N_blocks,       'N_trials':N_trials,
                     'time_sup':time_sup,       'step_fit':step_fit,     'do_whitening':do_whitening,
                     'before_sacc':before_sacc, 'after_sacc':after_sacc, 'stop_search_misac':stop_search_misac,
                     'filt':filt,               'cutoff':cutoff,         'sample_rate':sample_rate}

            if 'observer' in self.param_exp.keys() : param['observer'] = self.param_exp['observer']
            for name in list_param_enre :
                if name == 'goodness_of_fit' :
                    list_goodness_of_fit = ['nfev', 'residual', 'chisqr', 'redchi','aic', 'bic']
                    param[name] = {}
                    for g in list_goodness_of_fit : param[name][g] = []
                else : param[name] = []


            for block in range(N_blocks) :

                if plot is not None :
                    if fitted_data=='saccade' :
                        fig = plt.figure(figsize=(fig_width, (fig_width*(N_trials/2)/1.6180)))
                        axs = gridspec.GridSpec(N_trials, 1)
                    else :
                        fig, axs = plt.subplots(N_trials, 1, figsize=(fig_width, (fig_width*(N_trials/2))/1.6180))

                for name in list_param_enre :
                    if name == 'goodness_of_fit' :
                        for g in list_goodness_of_fit : param[name][g].append([])
                    else : param[name].append([])

                for trial in range(N_trials) :

                    print('block, trial = ', block, trial)

                    trial_data = trial + N_trials*block
                    arg = ANEMO.arg(self, data[trial_data], trial=trial, block=block)

                    if arg.trackertime[-1] < arg.TargetOn + 500 :
                        print('Warning : Not Data! The values saved for the fit parameters will be NaN!')
                        for name in list_param_enre :
                            if name == 'goodness_of_fit' :
                                for g in list_goodness_of_fit : param[name][g][block].append(np.nan)
                            else : param[name][block].append(np.nan)

                    else :
                        opt = opt_base.copy()
                        opt.update(arg)

                        velocity_NAN = ANEMO.velocity_NAN(self, **opt)

                        if fitted_data=='velocity' :
                            data_x = arg.data_x
                            data_1 = velocity_NAN
                            data_trial = np.copy(data_1)
                        else :
                            data_x = ANEMO.data_deg(self, data=arg.data_x, **opt)
                            data_1 = data_x
                            data_trial = np.copy(data_1)


                        if fitted_data != 'saccade' :

                            old_latency, old_steady_state, old_anti = ANEMO.classical_method.Full(velocity_NAN, arg.TargetOn-arg.t_0)
                            onset  = arg.TargetOn - arg.t_0

                            try :
                                #-------------------------------------------------
                                # FIT
                                #-------------------------------------------------
                                f = ANEMO.Fit.Fit_trial(self, data_trial, equation=equation, value_latency=old_latency, value_steady_state=old_steady_state, value_anti=old_anti, **opt)
                                #-------------------------------------------------

                                for name in list_param_enre :
                                    if name in f.values.keys() :
                                        if name in ['start_anti', 'latency'] : val = f.values[name] - onset
                                        else :                                 val = f.values[name]
                                        param[name][block].append(val)
                                if 'fit' in list_param_enre : param['fit'][block].append(f.best_fit)

                                if 'goodness_of_fit' in list_param_enre :
                                    param['goodness_of_fit']['nfev'][block].append(f.nfev)
                                    param['goodness_of_fit']['residual'][block].append(f.residual)
                                    param['goodness_of_fit']['chisqr'][block].append(f.chisqr)
                                    param['goodness_of_fit']['redchi'][block].append(f.redchi)
                                    param['goodness_of_fit']['aic'][block].append(f.aic)
                                    param['goodness_of_fit']['bic'][block].append(f.bic)

                                if 'near_sacc' in list_param_enre :
                                    near_sacc = False
                                    for s in range(len(arg.saccades)) :
                                        if arg.saccades[s][0] >= arg.TargetOn+100 :
                                            if f.values['latency'] >= arg.saccades[s][0]-arg.t_0 - 50 : near_sacc = True
                                    param['near_sacc'][block].append(near_sacc)


                            except:
                                print('Warning : The fit did not work! The values saved for the fit parameters will be NaN!')
                                for name in list_param_enre :
                                    if name == 'goodness_of_fit' :
                                        for g in list_goodness_of_fit : param[name][g][block].append(np.nan)
                                    else : param[name][block].append(np.nan)

                            if 'old_anti' in list_param_enre :         param['old_anti'][block].append(old_anti)
                            if 'old_steady_state' in list_param_enre : param['old_steady_state'][block].append(old_steady_state)
                            if 'old_latency' in list_param_enre :      param['old_latency'][block].append(old_latency-onset)

                            if 'nb_sacc' in list_param_enre :
                                NB_sacc = 0
                                for s in range(len(arg.saccades)) :
                                    if arg.saccades[s][0] >= arg.TargetOn : NB_sacc += 1
                                param['nb_sacc'][block].append(NB_sacc)


                        if fitted_data == 'saccade' :

                            for name in list_param_enre :
                                if name == 'goodness_of_fit' :
                                    for g in list_goodness_of_fit : param[name][g][block].append([])
                                else : param[name][block].append([])

                            for s in range(len(arg.saccades)) :
                                data_sacc = data_1[arg.saccades[s][0]-arg.t_0-before_sacc:arg.saccades[s][1]-arg.t_0+after_sacc]
                                if len(data_sacc) > 0 :
                                    try :
                                        #-------------------------------------------------
                                        # FIT
                                        #-------------------------------------------------
                                        f = ANEMO.Fit.Fit_trial(self, data_sacc, equation=equation, **opt)
                                        #-------------------------------------------------

                                        for name in list_param_enre :
                                            if name in f.values.keys() : param[name][block][trial].append(f.values[name])
                                        if 'fit' in list_param_enre :    param['fit'][block][trial].append(f.best_fit)

                                        if 'goodness_of_fit' in list_param_enre :
                                            param['goodness_of_fit']['nfev'][block][trial].append(f.nfev)
                                            param['goodness_of_fit']['residual'][block][trial].append(f.residual)
                                            param['goodness_of_fit']['chisqr'][block][trial].append(f.chisqr)
                                            param['goodness_of_fit']['redchi'][block][trial].append(f.redchi)
                                            param['goodness_of_fit']['aic'][block][trial].append(f.aic)
                                            param['goodness_of_fit']['bic'][block][trial].append(f.bic)

                                    except:
                                        print('Warning : The fit did not work for the saccade %s! The values saved for the fit parameters will be NaN!'%s)
                                        for name in list_param_enre :
                                            if name == 'goodness_of_fit' :
                                                for g in list_goodness_of_fit : param[name][g][block][trial].append(np.nan)
                                            else : param[name][block][trial].append(np.nan)

                    if plot is not None :

                        if N_trials==1 : ax1 = axs
                        else :           ax1 = axs[trial]

                        if fitted_data == 'saccade' : ax = gridspec.GridSpecFromSubplotSpec(1, len(arg.saccades), subplot_spec=ax1, hspace=0.25, wspace=0.15)
                        else :                        ax = ax1 ; ax.cla() # to put ax figure to zero

                        if trial==0 : write_step_trial = True
                        else :        write_step_trial = False

                        param_fit_trial = {}
                        for name in list_param_enre :
                            if name != 'goodness_of_fit' :
                                param_fit_trial[name] = param[name][block][trial]

                        opt['param_fit'] = param_fit_trial

                        ax = ANEMO.Plot.generate_fig(self, ax=ax, data=data, trial=trial, block=block, fig=fig,
                                                     show_data=fitted_data, equation=equation,
                                                     write_step_trial=write_step_trial,
                                                     show='fit', show_num_trial=True, show_pos_sacc=False,
                                                     plot_detail=None,report=None,
                                                     title='', c='k', out=None, **opt)

                if plot is not None :
                    if equation=='fct_saccade' : axs.tight_layout(fig) # to remove too much margin
                    else :                       plt.tight_layout() # to remove too much margin
                    plt.subplots_adjust(hspace=0) # to remove space between figures
                    file_fig = Test.test_None(file_fig, 'Fit')
                    plt.savefig(file_fig+'_%s.pdf'%(block+1))
                    plt.close()

            return param


    class Plot(object) :
        """
        Plot allows to display the data as well as their Fits.
        You could use the functions 'velocity', 'position' and 'saccades' already present, but also your own functions.
        It must be initialized with the parameters of the experiment
        (see :mod:`~ANEMO.ANEMO.ANEMO` for more precisions on the parameters)
        """

        def __init__(self, param_exp=None) :

            '''
            Parameters
            ----------
            param_exp : dict
                dictionary containing the parameters of the experiment :

                'px_per_deg' : float
                        number of px per degree for the experiment
                            screen_width_deg = 2. * np.arctan((screen_width_cm/2) / viewing_Distance_cm) * 180/np.pi
                            px_per_deg = screen_width_px / screen_width_deg


                'dir_target' : list
                        list of lists for each block containing the direction of the target for each trial,
                        dir_target = param_exp['dir_target'][block][trial]
                            the direction of the target must be equal to -1 for left or 1 for right
                or 'p' : ndarray
                        ndarray containing for each trial of each block the direction of the target, its probability of direction and the switches of this probability
                        dir_target = param_exp['p'][trial, block, 0]
                            the direction of the target must be equal to 0 for left or 1 for right
                        proba = param_exp['p'][trial, block, 1]
                        swich = param_exp['p'][trial, block, 2]

                'N_trials' : int
                    number of trials per block

                'N_blocks' : int
                    number of blocks

                'observer' : str
                    subject name

                'list_events' : list
                    list of the names of the events of the trial : ['onset fixation', 'end fixation', 'start pursuit', 'end pursuit']
                    by default : ['StimulusOn\n', 'StimulusOff\n', 'TargetOn\n', 'TargetOff\n']

                optional not obligatory, just to display the target in ANEMO.Plot :

                    'V_X_deg' : float
                        target velocity in deg/s
                    'stim_tau' : float
                        presentation time of the target
                    'RashBass' : int
                        the time the target has to arrive at the center of the screen in ms (to move the target back to t=0 of its velocity * latency = RashBass)
            '''

            ANEMO.__init__(self, param_exp)

        def deco (self, ax, StimulusOn=None, StimulusOf=None, TargetOn=None, TargetOff=None, saccades=None, t_label=20, **opt) :

            '''
            Allows to display the fixation, the gap and the pursuit on a figure

            Parameters
            ----------
            ax : AxesSubplot
                ax on which deco should be displayed

            StimulusOn : int, optional (default None)
                time when the stimulus appears
            StimulusOf : int, optional (default None)
                time when the stimulus disappears
            TargetOn : int, optional (default None)
                time when the target to follow appears
            TargetOff : int, optional (default None)
                time when the target to follow disappears
            saccades : list, optional (default None)
                list of edf saccades for the trial recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`

            t_label : int, optional (default 20)
                size x and y label

            Returns
            -------
            fig : matplotlib.figure.Figure
                figure
            ax : AxesSubplot
                figure
            '''

            try :

                StimulusOn = Test.crash_None('StimulusOn', StimulusOn)
                StimulusOf = Test.crash_None('StimulusOf', StimulusOf)
                TargetOn   = Test.crash_None('TargetOn', TargetOn)
                TargetOff  = Test.crash_None('TargetOff', TargetOff)
                saccades   = Test.crash_None('saccades', saccades)

                start = TargetOn
                StimOn_s = StimulusOn - start
                StimOf_s = StimulusOf - start
                TarOn_s  = TargetOn   - start
                TarOff_s = TargetOff  - start

                ax.axvspan(StimOn_s, StimOf_s, color='k', alpha=0.2)
                ax.axvspan(StimOf_s, TarOn_s,  color='r', alpha=0.2)
                ax.axvspan(TarOn_s,  TarOff_s, color='k', alpha=0.15)

                for s in range(len(saccades)) : ax.axvspan(saccades[s][0]-start, saccades[s][1]-start, color='k', alpha=0.15)

            finally :
                ax.set_xlabel('Time (ms)', fontsize=t_label)
                ax.tick_params(labelsize=t_label/2 , bottom=True, left=True)

            return ax


        def generate_fig(self, ax, data, trial, block,
                         show='data', show_data='velocity', equation='fct_velocity',
                         N_blocks=None, N_trials=None,
                         time_sup=280, step_fit=2, do_whitening=False,
                         list_param_enre=None, param_fit=None, inde_vars=None,
                         before_sacc=5, after_sacc=15, stop_search_misac=None,
                         filt=None, cutoff=30, sample_rate=1000,
                         show_pos_sacc=True, plot_detail=None,
                         show_target=False, show_num_trial=None, write_step_trial=True,
                         title='', c='k', fig=None, out=None, report=None,
                         fig_width=15, t_label=20, t_text=14, **opt) :

            '''
            Return the parameters of the fit present in list_param_enre

            Parameters
            ----------
            ax : AxesSubplot
                axis on which the figure is to be displayed

            data : list
                edf data for the trials recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`

            trial : int
                number of the trial in the block
            block : int
                block number

            show : str {'data', 'fit'}, optional (default 'data')
                - ``'data'`` : show a data
                - ``'fit'`` : show a data fit with the function defined with equation parameter
            show_data : str {'velocity', 'position', 'saccade'}, optional (default 'velocity')
                - ``'velocity'`` : show the velocity data for a trial in deg/sec
                - ``'position'`` : show the position data for a trial in deg
                - ``'saccade'`` : show the position data for sacades in trial in deg

            equation : str {'fct_velocity', 'fct_position', 'fct_saccades'} or function (default 'fct_velocity')
                function or name of the equation for the fit :
                    - ``'fct_velocity'`` : does a data fit with function ``'fct_velocity'``
                    - ``'fct_velocity_sigmo'`` : does a data fit with function ``'fct_velocity_sigmo'``
                    - ``'fct_velocity_line'`` : does a data fit with function ``'fct_velocity_line'``
                    - ``'fct_position'`` : does a data fit with function ``'fct_position'``
                    - ``'fct_saccades'`` : does a data fit with function ``'fct_saccades'``
                    - ``function`` : does a data fit with function

            N_blocks : int, optional (default None)
                number of blocks -- if ``None`` went searched in ``param_exp`` (see :mod:`~ANEMO.ANEMO.ANEMO`)
            N_trials : int, optional (default None)
                number of trials per block  --  if ``None`` went searched in ``param_exp`` (see :mod:`~ANEMO.ANEMO.ANEMO`)

            time_sup : int, optional (default 280)
                time that will be deleted to perform the fit (for data that is less good at the end of the test)
            step_fit : int, optional (default 2)
                number of steps for the fit
            do_whitening : bool, optional (default False)
                if ``True`` return the whitened fit

            list_param_enre : list, optional (default None)
                list of fit parameters to record \n
                if ``None`` :
                    - if ``equation`` is ``'fct_velocity'`` or ``'fct_position'`` : ::

                        list_param_enre = ['fit', 'start_anti', 'a_anti',
                                           'latency', 'tau', 'steady_state',
                                           'saccades', 'old_anti',
                                           'old_steady_state', 'old_latency']

                    - if ``equation`` is ``'fct_saccades'`` : ::

                        list_param_enre = ['fit', 'T0', 't1', 't2', 'tr',
                                           'x_0', 'x1', 'x2', 'tau']

            param_fit : dic, optional (default None)
                fit parameter dictionary, each parameter is a dict containing : ::
                    - ``'name'`` : name of the variable,
                    - ``'value'`` : initial value,
                    - ``'min'`` : minimum value,
                    - ``'max'`` : maximum value,
                    - ``'vary'`` :
                        - ``True`` if varies during fit,
                        - ``'vary'`` if only varies for step 2,
                        - ``False`` if not varies during fit

                if ``None`` generate by :func:`~ANEMO.ANEMO.ANEMO.Fit.generation_param_fit`
            inde_vars : dic, optional (default None)
                independent variable dictionary of fit -- if ``None`` generate by :func:`~ANEMO.ANEMO.ANEMO.Fit.generation_param_fit`

            before_sacc : int, optional (default 5)
                time to remove before saccades -- it is advisable to put : ``5`` for ``'fct_velocity'`` and ``'fct_position'``, ``0`` for ``'fct_saccade'``
            after_sacc : int, optional (default 15)
                time to delete after saccades -- it is advisable to put ``15``
            stop_search_misac : int, optional (default None)
                stop search of micro_saccade -- if ``None`` stops searching at the ``end of fixation + 100ms``

            filt : str {'position', 'velocity', 'velocity-position'} or None (default None)
                to filter the data can be :
                    - ``'position'`` : filter the position,
                    - ``'velocity'`` : filter the speed,
                    - ``'velocity-position'`` : filter the position then the speed
                    - ``None`` : the data will not be filtered
            cutoff : int, optional (default 30)
                the critical frequencies for cutoff of filter
            sample_rate : int, optional (default 1000)
                sampling rate of the recording for the filtre

            show_pos_sacc : bool, optional (default True)
                if ``True`` shows in a first figure the location of saccades during the pousuite
            plot_detail : bool, optional (default None)
                if ``True`` show the fit parameters on the data

            show_target : bool, optional (default False)
                if ``True`` show the target on the plot
            show_num_trial : bool, optional (default None)
                if ``True`` the num is written of the trial in y_label
            write_step_trial : bool, optional (default True)
                if ``True`` : write the steps of the trial on the figure

            title : str, optional (default '')
                title of the figure
            c : str, optional (default 'k')
                text color and fit
            fig :  matplotlib.figure.Figure, optional (default None)
                figure on which the function should be displayed -- if ``None`` a figure is created
            out : for the function show_fig, optional (default None)
            report : bool, optional (default None)
                if ``True`` return the report of the fit for each trial

            fig_width : int, optional (default 15)
                figure size
            t_label : int, optional (default 20)
                size x and y label
            t_text : int, optional (default 14)
                size of the text of the figure

            Returns
            -------
            ax : AxesSubplot
                figure
            result : dict
                if report is True
            '''

            if fig_width < 15 : lw = 1
            else : lw = 2

            #------------------------------------------------------------------------------
            if N_blocks is None :
                N_blocks = Test.test_value('N_blocks', self.param_exp, crash=None)
                N_blocks = Test.test_None(N_blocks, 1)
            if N_trials is None :
                N_trials = Test.test_value('N_trials', self.param_exp, crash=None)
                N_trials = Test.test_None(N_trials, int(len(data)/N_blocks))
            #------------------------------------------------------------------------------

            if equation in ['fct_velocity', 'fct_velocity_sigmo', 'fct_velocity_line'] : show_data = 'velocity'
            if equation == 'fct_position' :                                              show_data = 'position'
            if equation == 'fct_saccade' :                                               show_data = 'saccade'

            import matplotlib.pyplot as plt
            if show_data=='saccade' : import matplotlib.gridspec as gridspec

            if out is not None : plt.close('all')

            if show=='fit' :

                if   equation == 'fct_velocity' :       eqt = ANEMO.Equation.fct_velocity
                elif equation == 'fct_velocity_sigmo' : eqt = ANEMO.Equation.fct_velocity_sigmo
                elif equation == 'fct_velocity_line' :  eqt = ANEMO.Equation.fct_velocity_line
                elif equation == 'fct_position' :       eqt = ANEMO.Equation.fct_position
                elif equation == 'fct_saccade' :        eqt = ANEMO.Equation.fct_saccade
                else :                                  eqt = equation

                if equation in ['fct_velocity', 'fct_position'] :
                    list_param = ['start_anti', 'a_anti', 'latency', 'tau', 'steady_state']
                    list_param_enre = Test.test_None(list_param_enre, value=list_param+['fit', 'old_anti', 'old_steady_state', 'old_latency'])

                if equation in ['fct_velocity_sigmo', 'fct_velocity_line'] :
                    list_param = ['start_anti', 'a_anti', 'latency', 'ramp_pursuit', 'steady_state']
                    list_param_enre = Test.test_None(list_param_enre, value=list_param+['fit', 'old_anti', 'old_steady_state', 'old_latency'])


                if equation == 'fct_saccade' :
                    list_param = ['T0', 't1', 't2', 'tr', 'x_0', 'x1', 'x2', 'tau']
                    list_param_enre = Test.test_None(list_param_enre, value=list_param+['fit'])

                if list_param_enre is None :
                    print('Warning list_param_enre is None, no parameter will be returned !!!')
                    list_param_enre = []

                param = {}
                if 'observer' in self.param_exp.keys() : param['observer'] = self.param_exp['observer']
                for name in list_param_enre :            param[name] = []

            opt_base = {'time_sup':time_sup,          'step_fit':step_fit,     'do_whitening':do_whitening,
                        'param_fit':param_fit,        'inde_vars':inde_vars,
                        'before_sacc':before_sacc,    'after_sacc':after_sacc, 'stop_search_misac':stop_search_misac,
                        'filt':filt,                  'cutoff':cutoff,         'sample_rate':sample_rate,
                        't_label':t_label}

            if show_data != 'saccade' : show_pos_sacc=True
            if show_pos_sacc is not True : show_target=False

            trial_data = trial + N_trials*block
            arg = ANEMO.arg(self, data[trial_data], trial=trial, block=block)
            opt = opt_base.copy()
            opt.update(arg)

            if fig is None :
                if show_data=='saccade' :
                    fig, axs = plt.subplots(1, 1, figsize=(fig_width, (fig_width)/1.6180))
                    axs.set_xticks([]) ; axs.set_yticks([])
                    for loc, spine in axs.spines.items() : spine.set_visible(False)
                    axs0 = gridspec.GridSpecFromSubplotSpec(2, len(arg.saccades), subplot_spec=axs, hspace=0.25, wspace=0.15)
                    ax = plt.Subplot(fig, axs0[0,:]) ; fig.add_subplot(ax)
                else :
                    fig, ax = plt.subplots(1, 1, figsize=(fig_width, (fig_width*(1/2))/1.6180))

            else :
                if show_data=='saccade' :
                    axs0 = ax
                    if show_pos_sacc is True : ax = plt.Subplot(fig, axs0[0,:]) ; fig.add_subplot(ax)

            start = arg.TargetOn
            time_s   = arg.trackertime - start
            TarOn_s  = arg.TargetOn - start
            TarOff_s = arg.TargetOff - start
            StimOf_s = arg.StimulusOf - start

            velocity_NAN = ANEMO.velocity_NAN(self, **opt)

            if show_data=='velocity' :
                scale = 1
                data_x = arg.data_x
                data_1 = velocity_NAN
                data_trial = np.copy(data_1)
                if show_num_trial is True : ax.set_ylabel('%s\nVelocity (°/s)'%(trial+1), fontsize=t_label, color=c)
                else : ax.set_ylabel('Velocity (°/s)', fontsize=t_label, color=c)

            else :
                scale = 1/2
                data_x = ANEMO.data_deg(self, data=arg.data_x, **opt)
                data_1 = data_x
                data_trial = np.copy(data_1)
                if show_pos_sacc is True :
                    if show_num_trial is True : ax.set_ylabel('%s\nDistance (°)'%(trial+1), fontsize=t_label, color=c)
                    else : ax.set_ylabel('Distance (°)', fontsize=t_label, color=c)

            if show_target is True :

                #------------------------------------------------
                # TARGET
                #------------------------------------------------
                # the target at t = 0 retreats from its velocity * latency = RashBass (here set in ms)

                try :

                    V_X = Test.test_value('V_X_deg', self.param_exp, print_crash="V_X_deg is not defined in param_exp")
                    stim_tau = Test.test_value('stim_tau', self.param_exp, print_crash="stim_tau is not defined in param_exp")
                    Target_trial = []

                    if show_data=='velocity' :
                        for tps in time_s :
                            if tps < TarOn_s :                                           V_target = 0
                            elif (tps >= TarOn_s and tps <= (TarOn_s + stim_tau*1000)) : V_target = arg.dir_target * V_X
                            else :                                                       V_target = 0
                            Target_trial.append(V_target)

                    else :
                        RashBass = Test.test_value('RashBass', self.param_exp, print_crash="RashBass is not defined in param_exp")
                        for tps in time_s :
                            if tps < TarOn_s :                                          pos_target = 0
                            elif tps == TarOn_s :                                       pos_target = pos_target -(arg.dir_target * ((V_X/1000)*RashBass))
                            elif (tps > TarOn_s and tps <= (TarOn_s + stim_tau*1000)) : pos_target = pos_target + (arg.dir_target*(V_X/1000))
                            else :                                                      pos_target = pos_target
                            Target_trial.append(pos_target)

                    ax.plot(time_s, Target_trial, color='r', linewidth=lw, alpha=0.4)

                except : print('the target can not be displayed, some parameters are missing !') ; pass
                #------------------------------------------------

            if show_pos_sacc is True :
                ax.plot(time_s, data_1, color='k', alpha=0.4)
                ax.axis([TarOn_s-700, TarOff_s+10, -39.5*scale, 39.5*scale])
                ax = ANEMO.Plot.deco(self, ax, **opt)

                ax.set_xlabel('Time (ms)', fontsize=t_label, color=c)
                ax.set_title(title, fontsize=t_label, color=c)

                if write_step_trial is True :
                    opt_text = dict(color='k', size=t_label*.75, ha='center', va='center', alpha=0.5)
                    ax.text(StimOf_s+(TarOn_s-StimOf_s)/2,            31*scale, "GAP",      **opt_text)
                    ax.text((TarOn_s-700)+(StimOf_s-(TarOn_s-700))/2, 31*scale, "FIXATION", **opt_text)
                    ax.text(TarOn_s+(TarOff_s-TarOn_s)/2,             31*scale, "PURSUIT",  **opt_text)

            if report is not None : result = []

            if show_data != 'saccade' and show=='fit' :

                no_fit = False

                from inspect import getargspec
                #-----------------------------------------------------------------------------
                onset  = arg.TargetOn - arg.t_0
                result_fit = {}
                param_f = {}

                if param_fit is None :

                    try :
                        #-------------------------------------------------
                        # FIT
                        #-------------------------------------------------
                        old_latency, old_steady_state, old_anti = ANEMO.classical_method.Full(velocity_NAN, arg.TargetOn-arg.t_0)

                        f = ANEMO.Fit.Fit_trial(self, data_trial, equation=equation, value_latency=old_latency, value_steady_state=old_steady_state, value_anti=old_anti, **opt)
                        for name in list_param_enre :
                            if name in f.values.keys() :
                                if name in ['start_anti', 'latency'] : val = f.values[name] - onset
                                else :                                 val = f.values[name]
                                result_fit[name], param_f[name] = val, val
                        if report is not None : result = f.fit_report()

                        rv = {}
                        for name in getargspec(eqt).args :
                            if name in f.values.keys() : rv[name] = f.values[name]

                        if 'fit' in list_param_enre :         result_fit['fit'] = f.best_fit

                    except:
                        print('Warning : The fit did not work!  The fit will not be displayed on the figure!')
                        no_fit = True
                        for name in list_param_enre : result_fit[name] = np.nan
                        if report is not None : result = np.nan

                    if 'old_anti' in list_param_enre :         result_fit['old_anti'] = old_anti
                    if 'old_steady_state' in list_param_enre : result_fit['old_steady_state'] = old_steady_state
                    if 'old_latency' in list_param_enre :      result_fit['old_latency'] = old_latency-onset
                    #-------------------------------------------------

                else :
                    for name in list_param :
                        param_f[name] = param_fit[name]

                    rv = {}
                    for name in getargspec(eqt).args :
                        if name in param_fit.keys() :
                            if np.isnan(param_fit[name]) :
                                print('The parameter %s is missing ! The fit will not be displayed on the figure'%name)
                                no_fit = True
                            else :
                                if name in ['start_anti', 'latency'] : rv[name] = param_fit[name] + onset
                                else :                                 rv[name] = param_fit[name]
                    rv['do_whitening'] = False
                    rv['dir_target'] = arg.dir_target

                    if equation=='fct_position' :
                        rv['t_0']         = arg.t_0
                        rv['px_per_deg']  = arg.px_per_deg
                        rv['nb_sacc']     = len(arg.saccades)
                        rv['before_sacc'] = before_sacc
                        rv['after_sacc']  = after_sacc

                    for name in list_param_enre :
                        if name in param_fit.keys() : result_fit[name] = param_fit[name]
                        else :                        result_fit[name] = None # TODO mettre un warning

                if no_fit is False :
                    #-----------------------------------------------------------------------------
                    inde_v = Test.test_None(inde_vars, ANEMO.Fit.generation_param_fit(self, equation=equation, **opt)[1])

                    if 'do_whitening' in getargspec(eqt).args : rv['do_whitening'] = False
                    rv.update(inde_v)

                    #-----------------------------------------------------------------------------

                    fit = eqt(**rv)

                    if plot_detail is None or equation not in ['fct_velocity', 'fct_velocity_sigmo', 'fct_velocity_line', 'fct_position'] :

                        if time_sup is None : ax.plot(time_s,             fit,             color=c, linewidth=lw)
                        else :                ax.plot(time_s[:-time_sup], fit[:-time_sup], color=c, linewidth=lw)
                        #-----------------------------------------------------------------------------
                        if arg.dir_target < 0 : list_param.reverse()
                        x = 0
                        for name in list_param :
                            if name in param_f.keys() :
                                ax.text((TarOff_s-10), -arg.dir_target*35*scale+(-arg.dir_target*x),
                                        "%s : %0.3f"%(name, param_f[name]) , color=c, size=t_text, va='center', ha='right')
                                x = x - 5*scale
                        if 'latency' in param_f.keys() :    ax.bar(param_f['latency'],    80, bottom=-40, color=c, width=3, lw=0)
                        if 'start_anti' in param_f.keys() : ax.bar(param_f['start_anti'], 80, bottom=-40, color=c, width=3, lw=0)

                    else :

                        ax.plot(time_s[:int(rv['start_anti'])],                    fit[:int(rv['start_anti'])],                    c='k',       lw=lw)
                        ax.plot(time_s[int(rv['start_anti']):int(rv['latency'])],  fit[int(rv['start_anti']):int(rv['latency'])],  c='r',       lw=lw)
                        ax.plot(time_s[int(rv['latency']):int(rv['latency'])+250], fit[int(rv['latency']):int(rv['latency'])+250], c='darkred', lw=lw)

                        #-----------------------------------------------------------------------------
                        y = {}
                        for y_pos in [int(rv['start_anti']), int(rv['latency']), int(rv['latency'])+50, int(rv['latency'])+250, int(rv['latency'])+400] :
                            if np.isnan(fit[y_pos]) : y[y_pos] = data_1[y_pos]
                            else :                    y[y_pos] = fit[y_pos]

                        # V_a ------------------------------------------------------------------------
                        ax.text((time_s[int(rv['start_anti'])]+time_s[int(rv['latency'])])/2, y[int(rv['start_anti'])]-15*scale, r"A$_a$ = %0.2f °/s$^2$"%(rv['a_anti']), color='r', size=t_label/1.5, ha='center')

                        # Start_a --------------------------------------------------------------------
                        ax.text(time_s[int(rv['start_anti'])]-25, -35*scale, "Start anticipation = %0.2f ms"%(rv['start_anti']-onset), color='k', alpha=0.7, size=t_label/1.5, ha='right')
                        ax.bar( time_s[int(rv['start_anti'])], 80*scale, bottom=-40*scale, width=4, lw=0, color='k', alpha=0.7)

                        # latency --------------------------------------------------------------------
                        ax.text(time_s[int(rv['latency'])]+25, -35*scale, "Latency = %0.2f ms"%(rv['latency']-onset), color='firebrick', size=t_label/1.5, va='center')
                        ax.bar( time_s[int(rv['latency'])], 80*scale, bottom=-40*scale, width=4, lw=0, color='firebrick', alpha=1)

                        if equation in ['fct_velocity', 'fct_position'] :
                            # tau --------------------------------------------------------------------
                            ax.text(time_s[int(rv['latency'])]+70+t_label, y[int(rv['latency'])], r"= %0.2f"%(rv['tau']), color='darkred', size=t_label/1.5, va='bottom')
                            ax.annotate(r'$\tau$', xy=(time_s[int(rv['latency'])]+50, y[int(rv['latency'])+50]), xycoords='data', size=t_label/1., color='darkred', va='bottom',
                                        xytext=(time_s[int(rv['latency'])]+70, y[int(rv['latency'])]), textcoords='data', arrowprops=dict(arrowstyle="->", color='darkred'))

                        if equation in ['fct_velocity_sigmo', 'fct_velocity_line'] :
                            # ramp_pursuit -----------------------------------------------------------
                            ax.text(time_s[int(rv['latency'])]+70+((t_label/1.5)*11), y[int(rv['latency'])]+(-10*arg.dir_target), r"= %0.2f"%(rv['ramp_pursuit']), color='darkred', size=t_label/1.5, va='bottom')
                            ax.annotate('Ramp Pursuit', xy=(time_s[int(rv['latency'])]+50, y[int(rv['latency'])+50]), xycoords='data', size=t_label/1.5, color='darkred', va='bottom',
                                        xytext=(time_s[int(rv['latency'])]+70, y[int(rv['latency'])]+(-10*arg.dir_target)), textcoords='data', arrowprops=dict(arrowstyle="->", color='darkred'))

                        # Steady State ------------------------------------------------------------------------
                        ax.text(TarOn_s+475, (y[int(rv['latency'])]+y[int(rv['latency'])+250])/2, "Steady State = %0.2f °/s"%(rv['steady_state']), color='k', size=t_label/1.5, va='center')
                        #-----------------------------------------------------------------------------

                        if equation in ['fct_velocity', 'fct_velocity_sigmo', 'fct_velocity_line'] :
                            # A_a ------------------------------------------------------------------------
                            ax.annotate('', xy=(time_s[int(rv['latency'])], y[int(rv['latency'])]-3), xycoords='data', size=t_label/1.5,
                                        xytext=(time_s[int(rv['start_anti'])], y[int(rv['start_anti'])]-3), textcoords='data', arrowprops=dict(arrowstyle="->", color='r'))
                            # Max ------------------------------------------------------------------------
                            ax.annotate('', xy=(TarOn_s+450, y[int(rv['latency'])]), xycoords='data', size=t_label/1.5,
                                        xytext=(TarOn_s+450, y[int(rv['latency'])+250]), textcoords='data', arrowprops=dict(arrowstyle="<->"))
                            ax.plot(time_s, np.zeros(len(time_s)), '--k', lw=1, alpha=0.5)
                            ax.plot(time_s[int(rv['latency']):], np.ones(len(time_s[int(rv['latency']):]))*y[int(rv['latency'])+400], '--k', lw=1, alpha=0.5)
                        #-----------------------------------------------------------------------------

            if show_data == 'saccade' :

                if show=='fit' :
                    result_fit = {}
                    for name in list_param_enre : result_fit[name] = []

                for s in range(len(arg.saccades)):

                    if len(arg.saccades)==1 :
                        if show_pos_sacc is True : ax1 = axs0[1]
                        else :                     ax1 = axs0
                    else :
                        if show_pos_sacc is True : ax1 = plt.Subplot(fig, axs0[1,s])
                        else :                     ax1 = plt.Subplot(fig, axs0[s])
                    fig.add_subplot(ax1)

                    #-----------------------------------------------------------------------------
                    data_sacc = data_1[arg.saccades[s][0]-arg.t_0-before_sacc:arg.saccades[s][1]-arg.t_0+after_sacc]
                    time = time_s[arg.saccades[s][0]-arg.t_0-before_sacc:arg.saccades[s][1]-arg.t_0+after_sacc]
                    ax1.plot(time, data_sacc, color='k', alpha=0.4)
                    ax1 = ANEMO.Plot.deco(self, ax1, **opt)

                    #-----------------------------------------------------------------------------
                    ax1.set_title('Saccade %s'%(s+1), fontsize=t_label/1.5, x=0.5, y=1.01, color=c)

                    ax1.set_xlabel('Time (ms)', fontsize=t_label/2, color=c)
                    if s==0 :
                        if show_num_trial is True : ax1.set_ylabel('%s\nDistance (°)'%(trial+1), fontsize=t_label/2, color=c)
                        else :                      ax1.set_ylabel('Distance (°)', fontsize=t_label/2, color=c)
                    ax1.tick_params(labelsize=t_label/2.5 , bottom=True, left=True)

                    minx, maxx, miny, maxy = time[0], time[-1], min(data_sacc), max(data_sacc)
                    ax1.axis([minx-(maxx-minx)/10, maxx+(maxx-minx)/10, miny-(maxy-miny)/10, maxy+(maxy-miny)/10])

                    #-----------------------------------------------------------------------------
                    if show_pos_sacc is True :
                        start_sacc, end_sacc = (arg.saccades[s][0])-start, (arg.saccades[s][1])-start
                        if start_sacc > TarOn_s-700 :
                            if end_sacc < TarOff_s+10 :
                                ax.text(start_sacc+(end_sacc-start_sacc)/2, -17, s+1, color=c, ha='center', size=t_text)

                    if show=='fit' :
                        no_fit = False

                        if param_fit is None :

                            try :
                                #-------------------------------------------------
                                # FIT
                                #-------------------------------------------------
                                f = ANEMO.Fit.Fit_trial(self, data_sacc, equation=equation, **opt)

                                for name in list_param_enre :
                                    if name in f.values.keys() : result_fit[name].append(f.values[name])
                                if 'fit' in list_param_enre :    result_fit['fit'].append(f.best_fit)
                                if report is not None :          result.append(f.fit_report())
                                #-------------------------------------------------
                                param_f = f.values

                            except:
                                print('Warning : The fit did not work  for the saccade %s!  The fit will not be displayed on the figure!'%(s))
                                no_fit = True
                                for name in list_param_enre : result_fit[name].append(np.nan)
                                if report is not None :       result.append(np.nan)

                        else :
                            param_f = {}
                            for name in list_param :
                                if np.isnan(param_fit[name][s]) :
                                    print('The parameter %s is missing for the saccade %s! The fit will not be displayed on the figure'%(name, s))
                                    no_fit = True
                                else :
                                    param_f[name] = param_fit[name][s]
                            for name in list_param_enre :
                                    if name in param_fit.keys() : result_fit[name].append(param_fit[name][s])
                                    else :                        result_fit[name].append(None)
                            param_f['do_whitening'] = False


                        if no_fit is False :
                            rv = param_f
                            if 'do_whitening' in param_f.keys() : rv['do_whitening'] = False
                            opt['data_x'] = data_sacc
                            inde_v = Test.test_None(inde_vars, ANEMO.Fit.generation_param_fit(self, equation=equation, **opt)[1])
                            rv.update(inde_v)

                            #-----------------------------------------------------------------------------
                            fit = eqt(**rv)

                            if show_pos_sacc is True : ax.plot(time, fit, color=c, linewidth=2)
                            ax1.plot(time, fit, color=c, linewidth=2)

                            miny, maxy = min(min(data_sacc), min(fit)), max(max(data_sacc), max(fit))
                            ax1.axis([minx-(maxx-minx)/3-((maxx-minx)/(t_text*3))*len(arg.saccades), maxx+(maxx-minx)/2+(3*(maxx-minx)/t_text)*len(arg.saccades), miny-(maxy-miny)/10, maxy+(maxy-miny)/10])

                            #-----------------------------------------------------------------------------
                            px = 0
                            for name in list_param :
                                if name in param_f.keys() :
                                    ax1.text(maxx+(maxx-minx)/3+(3*(maxx-minx)/t_text)*len(arg.saccades), (maxy+(maxy-miny)/20)-px, "%s : %0.2f"%(name, param_f[name]), color=c, ha='right', va='top', size=t_text)
                                    px = px + (maxy-miny)/(t_text/1.7)

                            opt_lines = dict(ymin=miny, ymax=maxy, color='k', lw=1, linestyles='--', alpha=0.5)
                            opt_text = dict(color='k', ha='center', va='top', size=t_text/1.5)

                            if 'T0' in param_f.keys() :
                                ax1.vlines(param_f['T0']+time[0], **opt_lines)
                                ax1.text(param_f['T0']+time[0], miny-(maxy-miny)/30, "T0", **opt_text)

                            if 't1' and 'T0' in param_f.keys() :
                                ax1.vlines(param_f['t1']+param_f['T0']+time[0], **opt_lines)
                                ax1.text(param_f['t1']+param_f['T0']+time[0], miny-(maxy-miny)/30, "t1", **opt_text)

                            if 't2' and 't1' and 'T0' in param_f.keys() :
                                ax1.vlines(param_f['t2']+param_f['t1']+param_f['T0']+time[0], **opt_lines)
                                ax1.text(param_f['t2']+param_f['t1']+param_f['T0']+time[0], miny-(maxy-miny)/30, "t2", **opt_text)

                            if 'tr' and 't2' and 't1' and 'T0' in param_f.keys() :
                                ax1.vlines(param_f['tr']+param_f['t2']+param_f['t1']+param_f['T0']+time[0], **opt_lines)
                                ax1.text(param_f['tr']+param_f['t2']+param_f['t1']+param_f['T0']+time[0], miny-(maxy-miny)/30, "tr", **opt_text)

                            opt_lines = dict(xmin=minx, xmax=maxx, color='k', lw=1, linestyles='--', alpha=0.5)
                            opt_text = dict(color='k', ha='right', va='center', size=t_text/1.5)

                            if 'x_0' in param_f.keys() :
                                ax1.hlines(param_f['x_0'], **opt_lines)
                                ax1.text(minx-(maxx-minx)/20, param_f['x_0'], "x_0", **opt_text)

                            if 'x1' and 'x_0' in param_f.keys() :
                                ax1.hlines(param_f['x1']+param_f['x_0'], **opt_lines)
                                ax1.text(minx-(maxx-minx)/20, param_f['x1']+param_f['x_0'], "x1",**opt_text)

                            if 'x2' and 'x_0' in param_f.keys() :
                                ax1.hlines(param_f['x2']+param_f['x_0'], **opt_lines)
                                ax1.text(minx-(maxx-minx)/20, param_f['x2']+param_f['x_0'], "x2", **opt_text)

            if out is not None :
                from IPython.display import display,clear_output
                with out : clear_output(wait=True) ; display(ax.figure)
                if show=='fit' : return result_fit

            else :
                if report is not None : return ax, result
                else : return ax


        def plot_equation(self, equation='fct_velocity', fig_width=15, t_titre=35, t_label=20) :

            '''
            Returns figure of the equation used for the fit with the parameters of the fit

            Parameters
            ----------
            equation : str {'fct_velocity', 'fct_position', 'fct_saccades'} or function (default 'fct_velocity')
                function or name of the equation for the fit :
                    - ``'fct_velocity'`` : displays the ``fct_velocity`` equation
                    - ``'fct_velocity_sigmo'`` : displays the ``fct_velocity_sigmo`` equation
                    - ``'fct_velocity_line'`` : displays the ``fct_velocity_line`` equation
                    - ``'fct_position'`` : displays the ``fct_position`` equation
                    - ``'fct_saccades'`` : displays the ``fct_saccades`` equation
                    - ``function`` : displays the function equation

            fig_width : int, optional (default 15)
                figure size

            t_titre : int, optional (default 35)
                size of the title of the figure

            t_label : int, optional (default 20)
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

            if fig_width < 15 : lw = 1
            else : lw = 2

            if equation in ['fct_velocity', 'fct_velocity_sigmo', 'fct_velocity_line', 'fct_position'] :

                time = np.arange(-750, 750, 1)
                StimulusOn, StimulusOf = -750, -300
                TargetOn, TargetOff    = 0, 750
                start_anti, latency    = 650, 850
                #-----------------------------------------------------------------------------

                if equation in ['fct_velocity', 'fct_velocity_sigmo', 'fct_velocity_line'] :
                    ax.set_ylabel('Velocity (°/s)', fontsize=t_label)
                    scale = 1

                if equation=='fct_velocity' :
                    ax.set_title('Function Velocity', fontsize=t_titre, x=0.5, y=1.05)
                    result_fit = ANEMO.Equation.fct_velocity(x=np.arange(len(time)), start_anti=start_anti, latency=latency,
                                                              a_anti=-20, tau=15., steady_state=15., dir_target=-1, do_whitening=False)

                if equation=='fct_velocity_sigmo' :
                    ax.set_title('Function Velocity Sigmoid', fontsize=t_titre, x=0.5, y=1.05)
                    result_fit = ANEMO.Equation.fct_velocity_sigmo(x=np.arange(len(time)), start_anti=start_anti, latency=latency,
                                                                   a_anti=-20, ramp_pursuit=40., steady_state=15., dir_target=-1, do_whitening=False)

                if equation=='fct_velocity_line' :
                    ax.set_title('Function Velocity Linear', fontsize=t_titre, x=0.5, y=1.05)
                    result_fit = ANEMO.Equation.fct_velocity_line(x=np.arange(len(time)), start_anti=start_anti, latency=latency,
                                                                   a_anti=-20, ramp_pursuit=100., steady_state=15., dir_target=-1, do_whitening=False)

                if equation=='fct_position' :
                    ax.set_title('Function Position', fontsize=t_titre, x=0.5, y=1.05)
                    ax.set_ylabel('Distance (°)', fontsize=t_label)

                    scale = 1/2
                    result_fit = ANEMO.Equation.fct_position(x=np.arange(len(time)), data_x=np.zeros(len(time)),
                                                            saccades=np.zeros(len(time)), nb_sacc=0, before_sacc=5, after_sacc=15,
                                                            start_anti=start_anti, a_anti=-20, latency=latency, tau=15., steady_state=15.,
                                                            t_0=0, dir_target=-1, px_per_deg=36.51807384230632,  do_whitening=False)

                #-----------------------------------------------------------------------------
                ax.axis([-750, 750, -39.5*scale, 39.5*scale])
                ax.set_xlabel('Time (ms)', fontsize=t_label)
                #-----------------------------------------------------------------------------

                ax.plot(time[latency+250:],        result_fit[latency+250:],        c='k',       lw=lw)
                ax.plot(time[:start_anti],         result_fit[:start_anti],         c='k',       lw=lw)
                ax.plot(time[start_anti:latency],  result_fit[start_anti:latency],  c='r',       lw=lw)
                ax.plot(time[latency:latency+250], result_fit[latency:latency+250], c='darkred', lw=lw)

                # V_a ------------------------------------------------------------------------
                ax.text(TargetOn, 15*scale, "Anticipation", color='r', size=t_label, ha='center')
                ax.text(TargetOn-50, -5*scale, r"A$_a$", color='r', size=t_label/1.5, ha='center', va='top')

                if equation=='fct_velocity' :
                    ax.annotate('', xy=(time[latency], result_fit[latency]-3), xycoords='data', size=t_label/1.5, xytext=(time[start_anti], result_fit[start_anti]-3), textcoords='data', arrowprops=dict(arrowstyle="->", color='r'))

                # Start_a --------------------------------------------------------------------
                ax.text(TargetOn-125, -35*scale, "Start anticipation", color='k', size=t_label, alpha=0.7, ha='right')
                ax.bar(time[start_anti], 80*scale, bottom=-40*scale, color='k', width=4, lw=0, alpha=0.7)

                # latency --------------------------------------------------------------------
                ax.text(TargetOn+125, -35*scale, "Latency", color='firebrick', size=t_label)
                ax.bar(time[latency], 80*scale, bottom=-40*scale, color='firebrick', width=4, lw=0, alpha=1)

                if equation in ['fct_velocity', 'fct_position'] :
                    # tau --------------------------------------------------------------------
                    ax.annotate(r'$\tau$', xy=(time[latency]+15, result_fit[latency+15]), xycoords='data', size=t_label, color='darkred', va='bottom',
                                xytext=(time[latency]+70, result_fit[latency+7]), textcoords='data', arrowprops=dict(arrowstyle="->", color='darkred'))
                if equation in ['fct_velocity_sigmo', 'fct_velocity_line'] :
                    # ramp_pursuit -----------------------------------------------------------
                    ax.annotate('Ramp Pursuit', xy=(time[latency]+50, result_fit[latency+50]), xycoords='data', size=t_label, color='darkred', va='bottom',
                                xytext=(time[latency]+70, result_fit[latency+10]), textcoords='data', arrowprops=dict(arrowstyle="->", color='darkred'))


                # Max ------------------------------------------------------------------------
                ax.text(TargetOn+400+25, ((result_fit[latency+400])/2), 'Steady State', color='k', size=t_label, va='center')

                if equation=='fct_velocity' :
                    ax.annotate('', xy=(TargetOn+400, 0), xycoords='data', size=t_label/1.5, xytext=(TargetOn+400, result_fit[latency+400]), textcoords='data', arrowprops=dict(arrowstyle="<->"))
                    ax.plot(time, np.zeros(len(time)), '--k', lw=1, alpha=0.5)
                    ax.plot(time[latency:], np.ones(len(time[latency:]))*result_fit[latency+400], '--k', lw=1, alpha=0.5)

                # COSMETIQUE -----------------------------------------------------------------
                ax.axvspan(StimulusOn, StimulusOf, color='k', alpha=0.2)
                ax.axvspan(StimulusOf, TargetOn,   color='r', alpha=0.2)
                ax.axvspan(TargetOn,   TargetOff,  color='k', alpha=0.15)

                ax.text(StimulusOf+(TargetOn-StimulusOf)/2, 31*scale, "GAP",      color='k', size=t_label*1.5, ha='center', va='center', alpha=0.5)
                ax.text((StimulusOf-750)/2,                 31*scale, "FIXATION", color='k', size=t_label*1.5, ha='center', va='center', alpha=0.5)
                ax.text((750-TargetOn)/2,                   31*scale, "PURSUIT",  color='k', size=t_label*1.5, ha='center', va='center', alpha=0.5)
                #-----------------------------------------------------------------------------

            elif equation=='fct_saccade' :

                time = np.arange(30)
                #-----------------------------------------------------------------------------

                T0,  t1,  t2,  tr = 0, 15, 12, 1
                x_0, x1, x2, tau = 0, 2, 1, 13

                fit = ANEMO.Equation.fct_saccade(time, x_0, tau, x1, x2, T0, t1, t2, tr,do_whitening=False)

                ax.plot(time, fit, c='R')

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
                ax.set_title('Saccade Function', size=t_titre, x=0.5, y=1.05)
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
                      N_blocks=None, N_trials=None,
                      before_sacc=5, after_sacc=15, stop_search_misac=None,
                      filt=None, cutoff=30, sample_rate=1000,
                      show_pos_sacc=True, show_target=False, show_num_trial=False,
                      title=None, fig_width=15, t_titre=35, t_label=20, t_text=14) :

            '''
            Returns the data figure

            Parameters
            ----------
            data : list
                edf data for the trials recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`

            show : str {'velocity', 'position', 'saccade'}, optional (default 'velocity')
                - ``'velocity'`` : show the velocity data for a trial in deg/sec
                - ``'position'`` : show the position data for a trial in deg
                - ``'saccade'`` : show the position data for sacades in trial in deg

            trials : int or list, optional (default 0)
                number or list of trials to display
            block : int, optional (default 0)
                number of the block in which it finds the trials to display

            N_blocks : int, optional (default None)
                number of blocks -- if ``None`` went searched in ``param_exp`` (see :mod:`~ANEMO.ANEMO.ANEMO`)
            N_trials : int, optional (default None)
                number of trials per block  --  if ``None`` went searched in ``param_exp`` (see :mod:`~ANEMO.ANEMO.ANEMO`)

            before_sacc : int, optional (default 5)
                time to remove before saccades -- it is advisable to put : ``5`` for ``'fct_velocity'`` and ``'fct_position'``, ``0`` for ``'fct_saccade'``
            after_sacc : int, optional (default 15)
                time to delete after saccades -- it is advisable to put ``15``
            stop_search_misac : int, optional (default None)
                stop search of micro_saccade -- if ``None`` stops searching at the ``end of fixation + 100ms``

            filt : str {'position', 'velocity', 'velocity-position'} or None (default None)
                to filter the data can be :
                    - ``'position'`` : filter the position,
                    - ``'velocity'`` : filter the speed,
                    - ``'velocity-position'`` : filter the position then the speed
                    - ``None`` : the data will not be filtered
            cutoff : int, optional (default 30)
                the critical frequencies for cutoff of filter
            sample_rate : int, optional (default 1000)
                sampling rate of the recording for the filtre

            show_pos_sacc : bool, optional (default True)
                if ``True`` shows in a first figure the location of saccades during the pousuite
            show_target : bool, optional (default False)
                if ``True`` show the target on the plot
            show_num_trial : bool, optional (default None)
                if ``True`` the num is written of the trial in y_label

            title : str, optional (default None)
                title of the figure

            fig_width : int, optional (default 15)
                figure size
            t_titre : int, optional (default 35)
                size of the title of the figure
            t_label : int, optional (default 20)
                size x and y label
            t_text : int, optional (default 14)
                size of the text of the figure

            Returns
            -------
            fig : matplotlib.figure.Figure
                figure
            ax : AxesSubplot
                figure
            '''

            if fig_width < 15 : lw = 1
            else : lw = 1.5

            import matplotlib.pyplot as plt

            if type(trials) is not list : trials = [trials]


            if show == 'saccade' :
                import matplotlib.gridspec as gridspec
                fig = plt.figure(figsize=(fig_width, (fig_width*(len(trials))/1.6180)))
                axs = gridspec.GridSpec(len(trials), 1)
            else :
                fig, axs = plt.subplots(len(trials), 1, figsize=(fig_width, (fig_width*(len(trials)/2)/1.6180)))


            if N_trials is None : N_trials = Test.test_value('N_trials', self.param_exp)

            opt_base = {'before_sacc':before_sacc, 'after_sacc':after_sacc, 'stop_search_misac':stop_search_misac, 't_label':t_label}

            x = 0
            for t in trials :

                if show_num_trial is True : print('block, trial = ', block, t)

                trial_data = t + N_trials*block
                arg = ANEMO.arg(self, data[trial_data], trial=t, block=block)

                if show=='saccade' : ax = gridspec.GridSpecFromSubplotSpec(2, len(arg.saccades), subplot_spec=axs[x], hspace=0.25, wspace=0.15)

                else :
                    if len(trials)==1 : ax = axs
                    else : ax = axs[x]
                    if x!= (len(trials)-1) : ax.set_xticklabels([])


                if x==0 :
                    write_step_trial = True
                    if title is None :
                        if show=='velocity' : title = 'Eye Movement'
                        else :                title = 'Eye Position'
                else :
                    title, write_step_trial = '', False

                ax = ANEMO.Plot.generate_fig(self, ax=ax, data=data, trial=t, block=block,
                                             show='data', show_data=show, equation=None,
                                             N_blocks=N_blocks, N_trials=N_trials,
                                             before_sacc=before_sacc, after_sacc=after_sacc, stop_search_misac=stop_search_misac,
                                             filt=filt, cutoff=cutoff, sample_rate=sample_rate,
                                             show_pos_sacc=show_pos_sacc,
                                             show_target=show_target, show_num_trial=show_num_trial, write_step_trial=write_step_trial,
                                             title=title, fig=fig,
                                             fig_width=fig_width, t_label=t_label, t_text=t_text)

                x=x+1

            if show in ['velocity', 'position'] :
                plt.tight_layout() # to remove the margin too large
                plt.subplots_adjust(hspace=0) # to remove space between figures
            if show =='saccade' :
                axs.tight_layout(fig, h_pad=5) # to remove the margin too large

            return fig, axs


        def plot_fit(self, data, trials=0, block=0,
                     fitted_data='velocity', equation='fct_velocity',
                     N_blocks=None, N_trials=None,
                     time_sup=280, step_fit=2, do_whitening=False,
                     list_param_enre=None, param_fit=None, inde_vars=None,
                     before_sacc=5, after_sacc=15, stop_search_misac=None,
                     filt=None, cutoff=30, sample_rate=1000,
                     show_target=False, show_num_trial=False, report=None,
                     title=None, fig_width=15, t_titre=35, t_label=20,  t_text=14) :

            '''
            Returns figure of data fits

            Parameters
            ----------
            data : list
                edf data for the trials recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`

            trials : int or list, optional (default 0)
                number or list of trials to display
            block : int, optional (default 0)
                number of the block in which it finds the trials to display

            fitted_data : str {'velocity', 'position', 'saccade'}, optional (default 'velocity')
                nature of fitted data :
                    - ``'velocity'`` : fit velocity data for trial in deg/sec
                    - ``'position'`` : fit position data for trial in deg
                    - ``'saccade'`` : fit position data for sacades in trial in deg

            equation : str {'fct_velocity', 'fct_position', 'fct_saccades'} or function (default 'fct_velocity')
                function or name of the equation for the fit :
                    - ``'fct_velocity'`` : does a data fit with function ``'fct_velocity'``
                    - ``'fct_velocity_sigmo'`` : does a data fit with function ``'fct_velocity_sigmo'``
                    - ``'fct_velocity_line'`` : does a data fit with function ``'fct_velocity_line'``
                    - ``'fct_position'`` : does a data fit with function ``'fct_position'``
                    - ``'fct_saccades'`` : does a data fit with function ``'fct_saccades'``
                    - ``function`` : does a data fit with function

            N_blocks : int, optional (default None)
                number of blocks -- if ``None`` went searched in ``param_exp`` (see :mod:`~ANEMO.ANEMO.ANEMO`)
            N_trials : int, optional (default None)
                number of trials per block  --  if ``None`` went searched in ``param_exp`` (see :mod:`~ANEMO.ANEMO.ANEMO`)

            time_sup : int, optional (default 280)
                time that will be deleted to perform the fit (for data that is less good at the end of the test)
            step_fit : int, optional (default 2)
                number of steps for the fit
            do_whitening : bool, optional (default False)
                if ``True`` return the whitened fit

            list_param_enre : list, optional (default None)
                list of fit parameters to record \n
                if ``None`` :
                    - if ``equation`` is ``'fct_velocity'`` or ``'fct_position'`` : ::

                        list_param_enre = ['fit', 'start_anti', 'a_anti',
                                           'latency', 'tau', 'steady_state',
                                           'saccades', 'old_anti',
                                           'old_steady_state', 'old_latency']

                    - if ``equation`` is ``'fct_saccades'`` : ::

                        list_param_enre = ['fit', 'T0', 't1', 't2', 'tr',
                                           'x_0', 'x1', 'x2', 'tau']

            param_fit : dic, optional (default None)
                fit parameter dictionary, each parameter is a dict containing : ::
                    - ``'name'`` : name of the variable,
                    - ``'value'`` : initial value,
                    - ``'min'`` : minimum value,
                    - ``'max'`` : maximum value,
                    - ``'vary'`` :
                        - ``True`` if varies during fit,
                        - ``'vary'`` if only varies for step 2,
                        - ``False`` if not varies during fit

                if ``None`` generate by :func:`~ANEMO.ANEMO.ANEMO.Fit.generation_param_fit`
            inde_vars : dic, optional (default None)
                independent variable dictionary of fit -- if ``None`` generate by :func:`~ANEMO.ANEMO.ANEMO.Fit.generation_param_fit`

            before_sacc : int, optional (default 5)
                time to remove before saccades -- it is advisable to put : ``5`` for ``'fct_velocity'`` and ``'fct_position'``, ``0`` for ``'fct_saccade'``
            after_sacc : int, optional (default 15)
                time to delete after saccades -- it is advisable to put ``15``
            stop_search_misac : int, optional (default None)
                stop search of micro_saccade -- if ``None`` stops searching at the ``end of fixation + 100ms``

            filt : str {'position', 'velocity', 'velocity-position'} or None (default None)
                to filter the data can be :
                    - ``'position'`` : filter the position,
                    - ``'velocity'`` : filter the speed,
                    - ``'velocity-position'`` : filter the position then the speed
                    - ``None`` : the data will not be filtered
            cutoff : int, optional (default 30)
                the critical frequencies for cutoff of filter
            sample_rate : int, optional (default 1000)
                sampling rate of the recording for the filtre

            show_target : bool, optional (default False)
                if ``True`` show the target on the plot
            show_num_trial : bool, optional (default None)
                if ``True`` the num is written of the trial in y_label
            report : bool, optional (default None)
                if ``True`` return the report of the fit for each trial

            title : str, optional (default None)
                title of the figure

            fig_width : int, optional (default 15)
                figure size
            t_titre : int, optional (default 35)
                size of the title of the figure
            t_label : int, optional (default 20)
                size x and y label
            t_text : int, optional (default 14)
                size of the text of the figure

            Returns
            -------
            fig : matplotlib.figure.Figure
                figure
            ax : AxesSubplot
                figure
            report : list
                list of the reports of the fit for each trial
            '''

            if N_trials is None : N_trials = Test.test_value('N_trials', self.param_exp)

            if type(trials) is not list : trials = [trials]

            opt_base = {'equation':equation,
                        'time_sup':time_sup, 'step_fit':step_fit, 'do_whitening':do_whitening,
                        'param_fit':param_fit, 'inde_vars':inde_vars,
                        'before_sacc':before_sacc, 'after_sacc':after_sacc, 'stop_search_misac':stop_search_misac,
                        't_label':t_label}

            if   equation=='fct_velocity' :       fitted_data, eqt = 'velocity', ANEMO.Equation.fct_velocity
            elif equation=='fct_velocity_sigmo' : fitted_data, eqt = 'velocity', ANEMO.Equation.fct_velocity_sigmo
            elif equation=='fct_velocity_line' :  fitted_data, eqt = 'velocity', ANEMO.Equation.fct_velocity_line
            elif equation=='fct_position' :       fitted_data, eqt = 'position', ANEMO.Equation.fct_position
            elif equation=='fct_saccade' :        fitted_data, eqt = 'saccade',  ANEMO.Equation.fct_saccade
            else : eqt = equation

            import matplotlib.pyplot as plt

            if fitted_data=='saccade' :
                import matplotlib.gridspec as gridspec
                fig = plt.figure(figsize=(fig_width, (fig_width*(len(trials))/1.6180)))
                axs = gridspec.GridSpec(len(trials), 1)
            else :
                fig, axs = plt.subplots(len(trials), 1, figsize=(fig_width, (fig_width*(len(trials)/2)/1.6180)))

            results = []
            x = 0
            for t in trials :

                if show_num_trial is True : print('block, trial = ', block, t)

                trial_data = t + N_trials*block
                arg = ANEMO.arg(self, data[trial_data], trial=t, block=block)


                if fitted_data=='saccade' : ax = gridspec.GridSpecFromSubplotSpec(2, len(arg.saccades), subplot_spec=axs[x], hspace=0.25, wspace=0.15)

                else :
                    if len(trials)==1 : ax = axs
                    else :              ax = axs[x]
                    if x!= (len(trials)-1) : ax.set_xticklabels([])

                if x==0 :
                    write_step_trial = True
                    if title is None :
                        if fitted_data=='velocity' : title = 'Velocity Fit'
                        else :                       title = 'Position Fit'
                else :
                    title, write_step_trial = '', False

                if param_fit is not None :
                    param_fit_trial = {}
                    for name in param_fit.keys() :
                        param_fit_trial[name] = param_fit[name][block][t]
                else :  param_fit_trial = None

                param_fct = dict(ax=ax, data=data, trial=t, block=block, fig=fig,
                                 title=title, N_blocks=N_blocks, N_trials=N_trials,
                                 show='fit', show_data=fitted_data, equation=equation,
                                 show_target=show_target, show_num_trial=show_num_trial, show_pos_sacc=True,
                                 write_step_trial=write_step_trial, plot_detail=True,
                                 list_param_enre=list_param_enre, param_fit=param_fit_trial, inde_vars=inde_vars,
                                 step_fit=step_fit, do_whitening=do_whitening, time_sup=time_sup, before_sacc=before_sacc, after_sacc=after_sacc,
                                 stop_search_misac=stop_search_misac,  report=report,
                                 fig_width=fig_width, t_label=t_label, t_text=t_text,
                                 filt=filt, cutoff=cutoff, sample_rate=sample_rate)

                if report is not None :
                    ax, result = ANEMO.Plot.generate_fig(self, **param_fct)
                    results.append(result)
                else :
                    ax = ANEMO.Plot.generate_fig(self, **param_fct)
                x=x+1

            if fitted_data in ['velocity', 'position'] :
                plt.tight_layout() # to remove the margin too large
                plt.subplots_adjust(hspace=0) # to remove space between figures
            if fitted_data =='saccade' :
                axs.tight_layout(fig) # to remove the margin too large


            if report is None : return fig, axs
            else : return fig, axs, results


        def plot_Full_data(self, data, show='velocity',
                           N_blocks=None, N_trials=None,
                           before_sacc=5, after_sacc=15, stop_search_misac=None,
                           filt=None, cutoff=30, sample_rate=1000,
                           file_fig=None, show_pos_sacc=True, show_target=False,
                           fig_width=12, t_titre=20, t_label=14, t_text=10) :

            '''
            Save the full data figure

            Parameters
            ----------
            data : list
                edf data for the trials recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`

            show : str {'velocity', 'position', 'saccade'}, optional (default 'velocity')
                - ``'velocity'`` : show the velocity data for a trial in deg/sec
                - ``'position'`` : show the position data for a trial in deg
                - ``'saccade'`` : show the position data for sacades in trial in deg

            N_blocks : int, optional (default None)
                number of blocks -- if ``None`` went searched in ``param_exp`` (see :mod:`~ANEMO.ANEMO.ANEMO`)
            N_trials : int, optional (default None)
                number of trials per block  --  if ``None`` went searched in ``param_exp`` (see :mod:`~ANEMO.ANEMO.ANEMO`)

            before_sacc : int, optional (default 5)
                time to remove before saccades -- it is advisable to put : ``5`` for ``'fct_velocity'`` and ``'fct_position'``, ``0`` for ``'fct_saccade'``
            after_sacc : int, optional (default 15)
                time to delete after saccades -- it is advisable to put ``15``
            stop_search_misac : int, optional (default None)
                stop search of micro_saccade -- if ``None`` stops searching at the ``end of fixation + 100ms``

            filt : str {'position', 'velocity', 'velocity-position'} or None (default None)
                to filter the data can be :
                    - ``'position'`` : filter the position,
                    - ``'velocity'`` : filter the speed,
                    - ``'velocity-position'`` : filter the position then the speed
                    - ``None`` : the data will not be filtered
            cutoff : int, optional (default 30)
                the critical frequencies for cutoff of filter
            sample_rate : int, optional (default 1000)
                sampling rate of the recording for the filtre

            file_fig : str, optional (default None)
                name of file figure reccorded -- if ``None`` file_fig is ``show``
            show_pos_sacc : bool, optional (default True)
                if ``True`` shows in a first figure the location of saccades during the pousuite
            show_target : bool, optional (default False)
                if ``True`` show the target on the plot

            fig_width : int, optional (default 12)
                figure size
            t_titre : int, optional (default 20)
                size of the title of the figure
            t_label : int, optional (default 14)
                size x and y label
            t_text : int, optional (default 10)
                size of the text of the figure

            Returns
            -------
            save the figure
            '''

            import matplotlib.pyplot as plt

            if N_blocks is None : N_blocks = Test.test_value('N_blocks', self.param_exp)
            if N_trials is None : N_trials = Test.test_value('N_trials', self.param_exp)

            for block in range(N_blocks) :
                fig, axs = ANEMO.Plot.plot_data(self, data, trials=list(np.arange(N_trials)), block=block,
                                                show=show,
                                                N_blocks=N_blocks, N_trials=N_trials,
                                                before_sacc=before_sacc, after_sacc=after_sacc, stop_search_misac=stop_search_misac,
                                                filt=filt, cutoff=cutoff, sample_rate=sample_rate,
                                                show_pos_sacc=show_pos_sacc,
                                                show_target=show_target, show_num_trial=True,
                                                fig_width=fig_width, t_titre=t_titre, t_label=t_label, t_text=t_text)

                file_fig = Test.test_None(file_fig, show)
                plt.savefig(file_fig+'_%s.pdf'%(block+1))
                plt.close()


        def show_fig(self, data, list_data_fitfct, Full_param_fit, list_delete=None,
                     show_data='velocity',
                     N_blocks=None, N_trials=None,
                     list_param_enre=None, inde_vars=None,
                     time_sup=280, step_fit=2, do_whitening=False,
                     before_sacc=5, after_sacc=15, stop_search_misac=None,
                     sample_rate=1000,
                     show_target=False,
                     fig_width=15, t_label=20, t_text=14) :

            '''
            Return the parameters of the fit present in list_param_enre

            Parameters
            ----------
            data : list
                edf data for the trials recorded by the eyetracker transformed by :func:`~ANEMO.read_edf`

            list_data_fitfct : dict
                dictionary of correspondence between the data and the fit --
                if ``None`` by default ``{'velocity':'fct_velocity', 'position':'fct_position', 'saccade':'fct_saccade'}``
            Full_param_fit : dict
                dictionary containing all the parameters of fit -- if ``None`` fit is for each figure

            show_data : str {'velocity', 'position', 'saccade'}, optional (default 'velocity')
                - ``'velocity'`` : show the velocity data for a trial in deg/sec
                - ``'position'`` : show the position data for a trial in deg
                - ``'saccade'`` : show the position data for sacades in trial in deg

            N_blocks : int, optional (default None)
                number of blocks -- if ``None`` went searched in ``param_exp`` (see :mod:`~ANEMO.ANEMO.ANEMO`)
            N_trials : int, optional (default None)
                number of trials per block  --  if ``None`` went searched in ``param_exp`` (see :mod:`~ANEMO.ANEMO.ANEMO`)

            list_param_enre : list, optional (default None)
                list of fit parameters to record \n
                if ``None`` :
                    - if ``equation`` is ``'fct_velocity'`` or ``'fct_position'`` : ::

                        list_param_enre = ['fit', 'start_anti', 'a_anti',
                                           'latency', 'tau', 'steady_state',
                                           'saccades', 'old_anti',
                                           'old_steady_state', 'old_latency']

                    - if ``equation`` is ``'fct_saccades'`` : ::

                        list_param_enre = ['fit', 'T0', 't1', 't2', 'tr',
                                           'x_0', 'x1', 'x2', 'tau']

            inde_vars : dic, optional (default None)
                independent variable dictionary of fit -- if ``None`` generate by :func:`~ANEMO.ANEMO.ANEMO.Fit.generation_param_fit`

            time_sup : int, optional (default 280)
                time that will be deleted to perform the fit (for data that is less good at the end of the test)
            step_fit : int, optional (default 2)
                number of steps for the fit
            do_whitening : bool, optional (default False)
                if ``True`` return the whitened fit

            before_sacc : int, optional (default 5)
                time to delete before saccades
            after_sacc : int, optional (default 15)
                time to delete after saccades
            stop_search_misac : int, optional (default None)
                stop search of micro_saccade -- if ``None`` stops searching at the ``end of fixation + 100ms``

            sample_rate : int, optional (default 1000)
                sampling rate of the recording for the filtre

            show_target : bool, optional (default False)
                if ``True`` show the target on the plot

            fig_width : int, optional (default 15)
                figure size
            t_label : int, optional (default 20)
                size x and y label
            t_text : int, optional (default 14)
                size of the text of the figure

            Returns
            -------
            Full_list : dict
                dictionary containing all good and bad trials
            '''


            import matplotlib.pyplot as plt
            import ipywidgets as widgets
            from IPython.display import display,clear_output


            if N_blocks is None :
                N_blocks = Test.test_value('N_blocks', self.param_exp, crash=None)
                N_blocks = Test.test_None(N_blocks, 1)
            if N_trials is None :
                N_trials = Test.test_value('N_trials', self.param_exp, crash=None)
                N_trials = Test.test_None(N_trials, int(len(data)/N_blocks))

            if list_data_fitfct is None :
                list_data_fitfct = {'velocity':'fct_velocity', 'position':'fct_position', 'saccade':'fct_saccade'}

            list_param_enre = {}
            list_param_enre['fct_velocity'] = ['start_anti', 'a_anti', 'latency', 'tau', 'steady_state']
            list_param_enre['fct_velocity_sigmo'] = ['start_anti', 'a_anti', 'latency', 'ramp_pursuit', 'steady_state']
            list_param_enre['fct_velocity_line'] = ['start_anti', 'a_anti', 'latency', 'ramp_pursuit', 'steady_state']
            list_param_enre['fct_position'] = ['start_anti', 'a_anti', 'latency', 'tau', 'steady_state']
            list_param_enre['fct_saccade'] =  ['T0', 't1', 't2', 'tr', 'x_0', 'x1', 'x2', 'tau']

            if Full_param_fit is None :
                Full_param_fit = {}
                for fct in list_data_fitfct.values() :
                    Full_param_fit[fct] = None

            for fct in list_data_fitfct.values() :
                if Full_param_fit[fct] is None :
                    Full_param_fit[fct] = {}
                    for name in list_param_enre[fct] :
                        Full_param_fit[fct][name] = (np.zeros((N_blocks, N_trials))*np.nan).tolist()

            Full_list = {}
            for s in ['data', 'fit'] :
                Full_list[s] = {}
                for d in list_data_fitfct.keys() :
                    Full_list[s][d] = []
                    option_block = []
                    for block in range(N_blocks) :
                        Full_list[s][d].append({})
                        option_block.append(str(block))
                        for trial in range(N_trials) :
                            if list_delete is not None :
                                if trial in list_delete[block] :
                                    Full_list[s][d][block][trial] = 'Bad'
                                else :
                                    Full_list[s][d][block][trial] = None
                            else :
                                Full_list[s][d][block][trial] = None

            trial, block = 0, 0

            if show_data in list_data_fitfct.keys() : show_data=show_data
            else :                                    show_data= list(list_data_fitfct.keys())[0]

            show = 'data'
            fct = list_data_fitfct[show_data]
            filt = None
            cutoff = 30

            plt.ioff()
            fig, ax = plt.subplots(1, 1, figsize=(12, (12*(1/2)/1.6180)))

            h = '%spx'%((fig.get_figheight()+1)*fig.get_dpi())
            w = '%spx'%((fig.get_figwidth()+1)*fig.get_dpi())

            out = widgets.Output(layout=widgets.Layout(grid_area='out', width='100%'))

            button = {}
            for b, style in zip(['Previews', 'Next', 'OK', 'Bad'], ['', '', 'success', 'danger']) :
                button[b] = widgets.Button(description=b, button_style=style, layout=widgets.Layout(grid_area=b, width='90%'))

            _filt   = widgets.Dropdown( value='None',          options=['None', 'position', 'velocity', 'velocity-position'], description='filter low-pass', disabled=False, layout=widgets.Layout(grid_area='filter',  width='90%' ))
            _show   = widgets.Dropdown( value='data',          options=['data', 'fit'],                                       description='show',            disabled=False, layout=widgets.Layout(grid_area='show',    width='90%' ))
            _data   = widgets.Dropdown( value=show_data,       options=list_data_fitfct.keys(),                               description='Data',            disabled=False, layout=widgets.Layout(grid_area='data',    width='90%' ))
            _block  = widgets.Dropdown( value=option_block[0], options=option_block,                                          description='Block',           disabled=False, layout=widgets.Layout(grid_area='block',   width='90%' ))
            _trial  = widgets.IntSlider(value=trial, min=0, max=N_trials-1, step=1,                                           description='Trial',           readout=True,   layout=widgets.Layout(grid_area='trial',   width='110%'))
            _cutoff = widgets.IntText(value=30, step=5,                                                                       description='cutoff',          readout=True,   layout=widgets.Layout(grid_area='cutoff',  width='90%' ))

            grid = widgets.GridBox(children=[_show, _data,  _trial, _block, _filt, _cutoff, out, button['Previews'],  button['Next'], button['OK'], button['Bad']],
                                   layout=widgets.Layout(grid_template_columns='20% 20% 20% 5% 5% 20% 10%',
                                                         grid_template_areas=''' "show   data     trial trial trial block .  "
                                                                                 "filter cutoff   .     .     .     .     .  "
                                                                                 "out    out      out   out   out   out   out"
                                                                                 ".      Previews Next  OK    Bad   .     .  " '''))

            display(grid)

            def fig(ss_title, c, out) :
                nonlocal trial, block, ax, fct, show_data, Full_param_fit, show, filt, cutoff
                title = 'block %s trial %s%s'%(block, trial, ss_title)

                param_fit = {}
                for name in list_param_enre[fct] :
                    if type(Full_param_fit[fct][name][block][trial])!= list :
                        if np.isnan(Full_param_fit[fct][name][block][trial]) : param_fit = None ; break
                        else : param_fit[name] = Full_param_fit[fct][name][block][trial]
                    else : param_fit[name] = Full_param_fit[fct][name][block][trial]

                param_f = ANEMO.Plot.generate_fig(self, ax=ax, data=data, trial=trial, block=block,
                                                  show=show, show_data=show_data, equation=fct,
                                                  N_blocks=N_blocks, N_trials=N_trials,
                                                  time_sup=time_sup, step_fit=step_fit, do_whitening=do_whitening,
                                                  list_param_enre=list_param_enre[fct], param_fit=param_fit, inde_vars=inde_vars,
                                                  before_sacc=before_sacc, after_sacc=after_sacc, stop_search_misac=stop_search_misac,
                                                  filt=filt, cutoff=cutoff, sample_rate=sample_rate,
                                                  show_target=show_target,
                                                  title=title, c=c, out=out,
                                                  fig_width=fig_width, t_label=t_label, t_text=t_text)

                if param_f is not None :
                    if type(Full_param_fit[fct][name][block][trial])!= list :
                        if np.isnan(Full_param_fit[fct][name][block][trial]) :
                            for name in list_param_enre[fct] :
                                Full_param_fit[fct][name][block][trial] = param_f[name]


            #-----------------------------------
            def check_list() :
                nonlocal trial, block, show_data, show, Full_list
                if Full_list[show][show_data][block][trial]=='Bad' :   c='darkred' ;   ss_title=' -- Bad'
                elif Full_list[show][show_data][block][trial]=='OK' :  c='darkgreen' ; ss_title=' -- OK'
                else :                                                 c='k' ;         ss_title=''
                return c, ss_title

            #-----------------------------------
            def check_filt(b) :
                nonlocal filt
                filt = _filt.value
                c, ss_title = check_list()
                fig(ss_title, c, out)

            def check_cutoff(b) :
                nonlocal cutoff
                if _cutoff.value < 1 : _cutoff.value = 1
                if _cutoff.value > ((sample_rate)/2-1) : _cutoff.value = ((sample_rate)/2-1)
                cutoff = _cutoff.value
                c, ss_title = check_list()
                fig(ss_title, c, out)


            def check_show(b) :
                nonlocal show
                show = _show.value
                c, ss_title = check_list()
                fig(ss_title, c, out)

            def check_data(b) :
                nonlocal fct, show_data
                show_data = _data.value
                fct = list_data_fitfct[show_data]
                c, ss_title = check_list()
                fig(ss_title, c, out)

            def check_block(b) :
                nonlocal block
                block = int(_block.value)
                c, ss_title = check_list()
                fig(ss_title, c, out)

            def check_trial(b) :
                nonlocal trial
                trial =  _trial.value
                c, ss_title = check_list()
                fig(ss_title, c, out)

            #-----------------------------------
            def Next(b) :
                nonlocal trial, block
                trial = trial+1
                if trial >= N_trials : trial = 0 ;          block = block+1
                if block >= N_blocks : trial = N_trials-1 ; block = N_blocks-1
                _trial.value, _block.value = trial, '%s'%block

            def Previews(b) :
                nonlocal trial, block
                trial = trial-1
                if trial < 0 : trial = N_trials-1 ; block = block-1
                if block < 0 : trial = 0 ;          block = 0
                _trial.value, _block.value = trial, '%s'%block

            #-----------------------------------
            def Bad(b) :
                nonlocal show_data, trial, block, Full_list, show
                Full_list[show][show_data][block][trial] = 'Bad'
                fig(' -- Bad', 'r', out)

            def no_Bad(b) :
                nonlocal show_data, trial, block, Full_list, show
                Full_list[show][show_data][block][trial] = 'OK'
                fig(' -- OK', 'g', out)

            #-----------------------------------
            button['Previews'].on_click(Previews)
            button['Next'].on_click(Next)
            button['Bad'].on_click(Bad)
            button['OK'].on_click(no_Bad)

            _show.observe(  check_show,  names='value')
            _data.observe(  check_data,  names='value')
            _filt.observe(  check_filt,  names='value')
            _cutoff.observe(check_cutoff,  names='value')
            _block.observe( check_block, names='value')
            _trial.observe( check_trial, names='value')

            check_trial(None)

            return Full_list


