#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from .various_functions import whitening

class Model:

    """
    Predefined models to fit the data.
    """

    class SmoothPursuit:

        """
        Predefined models to fit the eye data during the smooth pursuit of a
        moving target.
        """

        def velocity_line(x, dir_target, start_anti, a_anti, latency,
                          ramp_pursuit, steady_state, SamplingFrequency,
                          do_whitening):

            """
            Model reproducing the velocity of the eye during the smooth pursuit
            of a moving target.

            This model includes different parts:

                - ``FIXATION`` - corresponding to a fixation of the eye on a
                  fixation point
                - ``ANTICIPATION`` - corresponding to an anticipatory movement
                  of the eye before the moving target appears
                - ``PURSUIT`` - corresponding to the movement of the eye when
                  tracking a moving target.
                  This part corresponds to a linear ``acceleration`` at the
                  beginning of the pursuit followed by a ``steady state`` of
                  the pursuit.

            Parameters
            ----------
            x: ndarray
                Time of the function

            dir_target: int
                Direction of the target -1 or 1
            start_anti: int
                Time when anticipation begins (ms)
            a_anti: float
                Acceleration of anticipation in deg/(s**2)
            latency: int
                Time when the movement begins (ms)
            ramp_pursuit: float
                Acceleration of pursuit in seconds
            steady_state: float
                Steady_state velocity reached during the pursuit

            SamplingFrequency: int
                Sampling frequency

            do_whitening: bool
                If ``True`` return the whitened velocity

            Returns
            -------
            velocity: list
                Velocity of the eye in deg/sec
            """

            # check that the start time of the anticipation is not above the
            # start time of the pursuit.
            if start_anti>=latency:
                return None

            else:

                start_anti = int((start_anti/1000) * SamplingFrequency)
                latency = int((latency/1000) * SamplingFrequency)

                # acceleration of anticipation to switch from sec to ms
                a_anti = a_anti/1000
                # acceleration of pursuit to switch from sec to ms
                r_pursuit = (dir_target*ramp_pursuit)/1000
                # velocity at pursuit latency
                v_latency = ((latency-1)-start_anti)*a_anti
                # steady state of the pursuit
                s_pursuit = (dir_target*steady_state) - v_latency

                # time of the end of the acceleration of the pursuit
                end_a = (s_pursuit/r_pursuit) + latency

                # check that the start time of the pursuit is not above the
                # time of the end of the acceleration of the pursuit
                if latency>=end_a:
                    return None

                else:

                    velocity = []
                    for t in x:

                        #------------------------------------------------------
                        # FIXATION
                        #------------------------------------------------------
                        if t<start_anti:
                            # velocity during the fixation
                            v_fixation = 0
                            velocity.append(v_fixation)

                        else:

                            #--------------------------------------------------
                            # ANTICIPATION
                            #--------------------------------------------------
                            if t<latency:
                                # velocity during the anticipation
                                v_anti = (t-start_anti) * a_anti
                                velocity.append(v_anti)

                            else:

                                #----------------------------------------------
                                # PURSUIT
                                #----------------------------------------------
                                # acceleration of the pursuit
                                if t<int(end_a):
                                    # velocity during the acceleration
                                    v_a = (t-latency)*r_pursuit + v_latency
                                    velocity.append(v_a)
                                # steady state of the pursuit
                                else:
                                    # velocity during the steady state
                                    v_steady = s_pursuit
                                    v_steady += v_latency
                                    velocity.append(v_steady)


                    # the whitened function
                    if do_whitening is True:
                        velocity = whitening(velocity)


                    return np.array(velocity)

        def velocity(x, dir_target, start_anti, a_anti, latency, tau,
                     steady_state, SamplingFrequency, do_whitening):

            r"""
            Model reproducing the velocity of the eye during the smooth pursuit
            of a moving target.

            This model includes different parts:

                - ``FIXATION`` - corresponding to a fixation of the eye on a
                  fixation point
                - ``ANTICIPATION`` - corresponding to an anticipatory movement
                  of the eye before the moving target appears
                - ``PURSUIT`` - corresponding to the movement of the eye when
                  tracking a moving target
                  This part corresponds to an exponential whose function is:
                  :math:`\mathtt{velocity = \
                  {\color{darkred}\text{steady_state}}\
                  \cdot (1 - e^{-\frac{t}{{\color{darkred}tau}}})}`


            Parameters
            ----------
            x: ndarray
                Time of the function

            dir_target: int
                Direction of the target -1 or 1
            start_anti: int
                Time when anticipation begins (ms)
            a_anti: float
                Acceleration of anticipation in °/s²
            latency: int
                Time when the movement begins (ms)
            tau: float
                Curve of the pursuit
            steady_state: float
                Steady_state velocity reached during the pursuit

            SamplingFrequency: int
                Sampling frequency

            do_whitening: bool
                If ``True`` return the whitened velocity

            Returns
            -------
            velocity: list
                Velocity of the eye in deg/sec
            """

            # check that the start time of the anticipation is not above the
            # start time of the pursuit.
            if start_anti>=latency:
                return None

            else:

                start_anti = int((start_anti/1000) * SamplingFrequency)
                latency = int((latency/1000) * SamplingFrequency)

                # acceleration of anticipation to switch from sec to ms
                a_anti = a_anti/1000
                # velocity at pursuit latency
                v_latency = ((latency-1)-start_anti)*a_anti
                # steady state of the pursuit
                s_pursuit = (dir_target*steady_state) - v_latency

                velocity = []
                for t in x:

                    #----------------------------------------------------------
                    # FIXATION
                    #----------------------------------------------------------
                    if t<start_anti:
                        # velocity during the fixation
                        v_fixation = 0
                        velocity.append(v_fixation)

                    else:

                        #------------------------------------------------------
                        # ANTICIPATION
                        #------------------------------------------------------
                        if t<latency:
                            # velocity during the anticipation
                            v_anti = (t-start_anti) * a_anti
                            velocity.append(v_anti)

                        #------------------------------------------------------
                        # PURSUIT
                        #------------------------------------------------------
                        else:
                            # velocity during the pursuit
                            v_pursuit = s_pursuit
                            v_pursuit *= (1-np.exp(-1/tau*(t-latency)))
                            v_pursuit += v_latency

                            velocity.append(v_pursuit)


                # the whitened function
                if do_whitening is True:
                    velocity=whitening(velocity)


                return np.array(velocity)

        def velocity_sigmo(x, dir_target, start_anti, a_anti, latency,
                           ramp_pursuit, steady_state, SamplingFrequency,
                           do_whitening):

            r"""
            Model reproducing the velocity of the eye during the smooth pursuit
            of a moving target

            This model includes different parts:

                - ``FIXATION`` - corresponding to a fixation of the eye on a
                  fixation point
                - ``ANTICIPATION`` - corresponding to an anticipatory movement
                  of the eye before the moving target appears
                - ``PURSUIT`` - corresponding to the movement of the eye when
                  tracking a moving target.
                  This part corresponds to an sigmoid whose function is:
                  :math:`\mathtt{velocity = \
                  \frac{{\color{darkred}\text{steady_state}}} \
                  {1 + e^{t \cdot {\color{darkred}\text{ramp_pursuit}}}}}`


            Parameters
            ----------
            x: ndarray
                Time of the function

            dir_target: int
                Direction of the target -1 or 1
            start_anti: int
                Time when anticipation begins (ms)
            a_anti: float
                Acceleration of anticipation in deg/(s**2)
            latency: int
                Time when the movement begins (ms)
            ramp_pursuit: float
                Curve of the pursuit
            steady_state: float
                Steady_state velocity reached during the pursuit

            SamplingFrequency: int
                Sampling frequency

            do_whitening: bool
                If ``True`` return the whitened velocity

            Returns
            -------
            velocity: list
                Velocity of the eye in deg/sec
            """

            # check that the start time of the anticipation is not above the
            # start time of the pursuit.
            if start_anti>=latency:
                return None

            else:

                start_anti = int((start_anti/1000) * SamplingFrequency)
                latency = int((latency/1000) * SamplingFrequency)

                # acceleration of anticipation to switch from sec to ms
                a_anti = a_anti/1000
                # curve of the pursuit to switch from sec to ms
                r_pursuit = -ramp_pursuit/1000
                # velocity at pursuit latency
                v_latency = ((latency-1)-start_anti)*a_anti
                # steady state of the pursuit
                s_pursuit = (dir_target*steady_state) - v_latency

                e = np.exp(1)
                # time of the sigmoid representing pursuit
                time_s = np.arange(-e, np.max(x), 1)
                # start of the sigmoid representing pursuit
                start_s = s_pursuit
                start_s /= (1+np.exp(((r_pursuit*time_s[0])+e)))

                velocity = []
                for t in x:

                    #----------------------------------------------------------
                    # FIXATION
                    #----------------------------------------------------------
                    if t<start_anti:
                        # velocity during the fixation
                        v_fixation = 0
                        velocity.append(v_fixation)

                    else:

                        #------------------------------------------------------
                        # ANTICIPATION
                        #------------------------------------------------------
                        if t<latency:
                            # velocity during the anticipation
                            v_anti = (t-start_anti) * a_anti
                            velocity.append(v_anti)

                        #------------------------------------------------------
                        # PURSUIT
                        #------------------------------------------------------
                        else:
                            # velocity during the pursuit
                            exp = np.exp((r_pursuit*time_s[int(t-latency)])+e)

                            v_pursuit = s_pursuit
                            v_pursuit /= (1+exp)
                            v_pursuit += (v_latency-start_s)

                            velocity.append(v_pursuit)

                # the whitened function
                if do_whitening is True:
                    velocity = whitening(velocity)

                return np.array(velocity)

        def position(x, x_nan, dir_target, start_anti, a_anti, latency, tau,
                     steady_state, SamplingFrequency, do_whitening):

            r"""
            Model reproducing the position of the eye during the smooth pursuit
            of a moving target.
            It corresponds to the derivative of the ``velocity`` model.

            This model of ``velocity`` includes different parts:

                - ``FIXATION`` - corresponding to a fixation of the eye on a
                  fixation point
                - ``ANTICIPATION`` - corresponding to an anticipatory movement
                  of the eye before the moving target appears
                - ``PURSUIT`` - corresponding to the movement of the eye when
                  tracking a moving target.
                  This part corresponds to an exponential whose function is:
                  :math:`\mathtt{velocity = \
                  {\color{darkred}\text{steady_state}} \cdot \
                  (1 - e^{-\frac{t}{{\color{darkred}tau}}})}`


            Parameters
            ----------
            x: ndarray
                Time of the function
            x_nan: ndarray
                Unsaccade data, containing NaN instead of saccades

            dir_target: int
                Direction of the target -1 or 1
            start_anti: int
                Time when anticipation begins (ms)
            a_anti: float
                Acceleration of anticipation in deg/(s**2)
            latency: int
                Time when the movement begins (ms)
            tau: float
                Curve of the pursuit
            steady_state: float
                Steady_state velocity reached during the pursuit

            SamplingFrequency: int
                Sampling frequency

            do_whitening: bool
                If ``True`` return the whitened position

            Returns
            -------
            position: list
                Position of the eye in deg
            """

            # check that the start time of the anticipation is not above the
            # start time of the pursuit.
            if start_anti>=latency:
                return None

            else:

                start_anti = int((start_anti/1000) * SamplingFrequency)
                latency = int((latency/1000) * SamplingFrequency)

                # acceleration of anticipation to switch from sec to ms
                a_anti = a_anti/1000
                # steady state of the pursuit to switch from sec to ms
                steady_state = steady_state/1000

                #--------------------------------------------------------------
                # velocity model
                #--------------------------------------------------------------
                velocity = Model.SmoothPursuit.velocity(x, dir_target,
                                                        start_anti, a_anti,
                                                        latency, tau,
                                                        steady_state,
                                                        SamplingFrequency,
                                                        do_whitening)

                #--------------------------------------------------------------
                # position model
                #--------------------------------------------------------------
                position = np.cumsum(velocity)

                #--------------------------------------------------------------
                # loop allowing the model to be readjusted according to the
                # saccades present in the data
                #--------------------------------------------------------------
                # list of saccade start times in the data
                ssacc = [t for t in range(1, len(x_nan)-1) \
                         if not np.isnan(x_nan[t]) and np.isnan(x_nan[t+1])]
                # list of saccade end times in the data
                esacc = [t for t in range(1, len(x_nan)-1) \
                         if not np.isnan(x_nan[t]) and np.isnan(x_nan[t-1])]

                for s in range(len(ssacc)):

                    if do_whitening is True:
                        if ssacc[s]-1 < len(pos):
                            p = position[ssacc[s]-1]
                        else:
                            p = position[-1]
                    else:
                        p = np.nan

                    if len(esacc)>s:

                        if esacc[s]+1 <= len(x):
                            position[ssacc[s]:esacc[s]] = p
                            if ssacc[s] >= int(latency-1):

                                # difference in position between the end of the
                                # saccade and the beginning of the previous
                                # saccade
                                diff_pos = (x_nan[esacc[s]]-x_nan[ssacc[s]-1])
                                # average velocity during the saccade
                                m_vsacc = np.mean(velocity[ssacc[s]:esacc[s]])
                                # duration of the saccade
                                t_sacc = (esacc[s]-ssacc[s])

                                position[esacc[s]:] += diff_pos
                                position[esacc[s]:] -= m_vsacc * t_sacc
                        else:
                            position[ssacc[s]:] = p

                    else:
                        position[ssacc[s]:] = p
                #--------------------------------------------------------------

                # the whitened function
                if do_whitening is True:
                    position = whitening(position)

                return position


    def saccade(x, x_0, tau, x1, x2, T0, t1, t2, tr, SamplingFrequency,
                do_whitening):

        """
        Model reproducing the position of the eye during the sacades.

        Parameters
        ----------
        x: ndarray
            Time of the function

        x_0: float
            Initial position of the beginning of the saccade in deg
        tau: float
            Curvature of the saccade
        x1: float
            Maximum of the first curvature in deg
        x2: float
            Maximum of the second curvature in deg
        T0: float
            Time of the beginning of the first curvature after x_0 in ms
        t1: float
            Maximum time of the first curvature after T0 in ms
        t2: float
            Time of the maximum of the second curvature after t1 in ms
        tr: float
            Time of the end of the second curvature after t2 in ms
        SamplingFrequency: int
            Sampling frequency

        do_whitening: bool
            If ``True`` return the whitened position

        Returns
        -------
        position: list
            Position of the eye during the sacades in deg
        """

        T0 = int((T0/1000) * SamplingFrequency)
        t1 = int((t1/1000) * SamplingFrequency)
        t2 = int((t2/1000) * SamplingFrequency)
        tr = int((tr/1000) * SamplingFrequency)

        time = x-T0
        T1 = t1
        T2 = t1+t2
        TR = T2+tr

        rho = (tau/T1) * np.log((1+np.exp(T1/tau))/2)
        rhoT = int(np.round(T1*rho))

        r = (tau/T2) * np.log((np.exp(T1/tau) + np.exp(T2/tau)) /2)
        rT = int(np.round(T2*r))

        exp1 = np.exp(-(rho*T1)/tau)
        exp2 = np.exp((1-rho)*T1/tau)

        Umax1 = (1/tau)
        Umax1 *= x1
        Umax1 /= ((2*rho-1)*T1 - tau*(2-exp1-exp2))

        Umax2 = (1/tau)
        Umax2 *= (x2-x1)
        Umax2 /= ((2*r-1)*T2 - T1)

        def curvature(Umax, t, orientation):
            c = Umax * tau
            c *= (t + orientation*tau*(1-np.exp(orientation*t/tau)))
            return  c

        saccade = []
        for t in time:
            if t < 0:
                saccade.append(x_0)
            elif t < rhoT:
                saccade.append(x_0 + curvature(Umax1, t, -1))
            elif t < T1:
                saccade.append(x_0 + x1 + curvature(Umax1, (T1-t), 1))
            elif t < rT:
                saccade.append(x_0 + x1 + curvature(Umax2, (t-T1), -1))
            elif t < TR:
                saccade.append(x_0 + x2 + curvature(Umax2, (T2-t), 1))
            else:
                saccade.append(saccade[-1])

        # the whitened function
        if do_whitening:
            saccade = whitening(saccade)

        return saccade
