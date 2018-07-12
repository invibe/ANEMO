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
    """
    try :
        N_freq = position.shape[0]
    except AttributeError :
        N_freq = len(position)
    freq, K = whitening_filt(N_freq=N_freq, white_f_0=white_f_0, white_alpha=white_alpha, white_steepness=white_steepness)        
    f_position = np.fft.fft(position)
    return np.real(np.fft.ifft(f_position*K))


def fct_old_latence(data, w1=300, w2=50, off=50, crit=0.17) :
    
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

    return(old_latence[0])

def classical_methode(velocity_NAN, StimulusOn, StimulusOf, TargetOn, t_0):

    old_latence = fct_old_latence(velocity_NAN)

    a1, b1 = TargetOn-t_0+400, TargetOn-t_0+600
    old_max = abs(np.nanmean(velocity_NAN[a1:b1]))

    old_anti = np.nanmean(velocity_NAN[TargetOn-t_0-50:TargetOn-t_0+50])

    return old_latence, old_max, old_anti/0.1


def fct_velocity (x, dir_target, start_anti, v_anti, latence, tau, maxi, do_whitening) :

    '''
    Fonction reproduisant la vitesse de l'œil lors de la pousuite lisse d'une cible en mouvement

    Parameters
    ----------
    x : ndarray
    dir_target : float
        -1 ou 1 donne la direction de la cible
    start_anti : int
        debut de l'anticipation
    v_anti : float
        vitesse de l'anticipation en seconde
    latence : int
        temps où commence le mvt
    tau : float
        courbe de la partie poursuite
    maxi : float
        maximum de la vitesse atteinte lors de la poursuite

    Returns
    -------
    vitesse : list
        vitesse de l'œil
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

def fct_position(x, data_x, saccades, nb_sacc, dir_target, start_anti, v_anti, latence, tau, maxi, t_0, px_per_deg, avant, apres, do_whitening):
    ms = 1000
    v_anti = (v_anti/ms)
    maxi = maxi /ms
    
    speed = fct_velocity(x=x, dir_target=dir_target, start_anti=start_anti, v_anti=v_anti, latence=latence, tau=tau, maxi=maxi, do_whitening=False)
    pos = np.cumsum(speed)

    i=0
    for s in range(nb_sacc) :
        sacc = saccades[i:i+3] # obligation d'avoir les variable indé a la même taille :/
                                # saccades[i] -> debut, saccades[i+1] -> fin, saccades[i+2] -> tps sacc
        if sacc[0]-t_0 < len(pos) :
            if sacc[0]-t_0 > int(latence) :
                if int(sacc[1]-t_0)+apres+1 <= len(pos) :
                    pos[int(sacc[0]-t_0)-avant:int(sacc[1]-t_0)+apres] = pos[int(sacc[0]-t_0)-avant-1] #np.nan 
                    pos[int(sacc[1]-t_0)+apres:] += ((data_x[int(sacc[1]-t_0)+apres]-data_x[int(sacc[0]-t_0)-avant-1])/px_per_deg) - np.mean(speed[int(sacc[0]-t_0):int(sacc[1]-t_0)]) * sacc[2]

                else :
                    pos[int(sacc[0]-t_0)-avant:] =  pos[int(sacc[0]-t_0)-avant-1] #np.nan
        i = i+3
    if do_whitening is True:
        pos = whitening(pos)
    return pos

def fct_saccade(x, x_0, tau, x1, x2, T0, t1, t2, tr, do_whitening):
    
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




class ANEMO(object):
    """ docstring for the ANEMO class. """
    ######################################################################################

    def __init__(self, param_exp) :

        #--------------------------------------------------------------------
        # Vérifie que param_exp est bon!
        def warning(var):
            #import sys
            print("/!\ %s n'est pas définit dans parm_exp"%var)
            #sys.exit()

        var = ['N_trials', 'screen_width_px', 'px_per_deg', 'V_X', 'RashBass', 'stim_tau', 'p', 'observer', 'N_blocks']
        try :
            for v in var :
                if param_exp[v] is None :
                    warning(v)
                    #return
        except KeyError :
            warning(v)
            #return
        #--------------------------------------------------------------------
        self.param_exp = param_exp

    ######################################################################################

    def arg(self, data, trial, block, list_events=None):

        trial_data = trial + self.param_exp['N_trials']*block

        if list_events is None :
            list_events = ['StimulusOn\n', 'StimulusOff\n', 'TargetOn\n', 'TargetOff\n']


        for events in range(len(data[trial_data]['events']['msg'])) :
            if data[trial_data]['events']['msg'][events][1] == list_events[0] :
                StimulusOn = data[trial_data]['events']['msg'][events][0]
            if data[trial_data]['events']['msg'][events][1] == list_events[1] :
                StimulusOf = data[trial_data]['events']['msg'][events][0]
            if data[trial_data]['events']['msg'][events][1] == list_events[2] :
                TargetOn = data[trial_data]['events']['msg'][events][0]
            if data[trial_data]['events']['msg'][events][1] == list_events[3] :
                TargetOff = data[trial_data]['events']['msg'][events][0]

        kwargs = {
                    "N_trials" : self.param_exp['N_trials'],
                    "screen_width_px" : self.param_exp['screen_width_px'],
                    "px_per_deg" : self.param_exp['px_per_deg'],
                    "V_X" : self.param_exp['V_X'],
                    "RashBass" : self.param_exp['RashBass'],
                    "stim_tau" : self.param_exp['stim_tau'],
                    "p" : self.param_exp['p'],
                    "bino" : self.param_exp['p'][trial, block, 0],
                    "data_x": data[trial_data]['x'],
                    "data_y": data[trial_data]['y'],
                    "trackertime":data[trial_data]['trackertime'],
                    "saccades":data[trial_data]['events']['Esac'],
                    "t_0": data[trial_data]['trackertime'][0],
                    "StimulusOn":StimulusOn,
                    "StimulusOf": StimulusOf,
                    "TargetOn": TargetOn,
                    "TargetOff": TargetOff,
            }

        import easydict
        return easydict.EasyDict(kwargs)


    def velocity_deg(self, data_x) :
        '''
        Return la vitesse de l'œil en deg/sec

        Parameters
        ----------
        data_x : ndarray
            position x pour un essai enregistré par l'eyetracker transformé par la fonction read_edf du module edfreader

        Returns
        -------
        gradient_deg : ndarray
            vitesse de l'œil en deg/sec
        '''
        gradient_x = np.gradient(data_x)
        gradient_deg = gradient_x * 1/self.param_exp['px_per_deg'] * 1000 # gradient en deg/sec

        return gradient_deg

    # y ajouter microsaccade ?
    def Microsaccade (self, velocity_x, velocity_y, VFAC=5, mindur=5, maxdur=100, minsep=30, t_0=0):
        '''
        Détection des micro_saccades non-detectés par eyelink dans les données

        Parameters
        ----------
        velocity_x : ndarray
            vitesse x de l'œil en deg/sec
        velocity_y : ndarray
            vitesse y de l'œil en deg/sec

        VFAC : int
            relative velocity threshold
        mindur : int
            minimal saccade duration (ms)
        maxdur : int
            maximal saccade duration (ms)
        minsep : int
            minimal time interval between two detected saccades (ms)
        t_0 : int
            temps 0 de l'essais

        Returns
        -------
        misaccades : list
            list of lists, each containing [debut microsaccades, fin microsaccade]

        '''
        msdx = np.sqrt((np.nanmedian(velocity_x**2))-((np.nanmedian(velocity_x))**2))
        msdy = np.sqrt((np.nanmedian(velocity_y**2))-((np.nanmedian(velocity_y))**2))

        radiusx = VFAC*msdx
        radiusy = VFAC*msdy

        test = (velocity_x/radiusx)**2 + (velocity_y/radiusy)**2
        index = [x for x in range(len(test)) if test[x] > 1]

        dur = 0
        debut_misaccades = 0
        k = 0
        misaccades = []

        for i in range(len(index)-1) :
            if index[i+1]-index[i]==1 :
                dur = dur + 1;
            else :
                if dur >= mindur and dur < maxdur :
                    fin_misaccades = i
                    misaccades.append([index[debut_misaccades]+t_0, index[fin_misaccades]+t_0])
                debut_misaccades = i+1
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

    def suppression_saccades(self, velocity, saccades, trackertime, avant=5, apres=15) :
        '''
        Supprime les saccades detectés par eyelink des données

        Parameters
        ----------
        velocity : ndarray
            vitesse de l'œil en deg/sec
        saccades : list
            liste des saccades edf pour un essaie enregistré par l'eyetracker transformé par la fonction read_edf du module edfreader
        trackertime : ndarray
            temps du tracker
        avant : int
            ms supprimer avant debut saccades
        apres : int
            ms supprimer après début saccades

        Returns
        -------
        new_velocity : ndarray
            vitesse de l'œil en deg/sec sans les saccades

        '''

        t_0 = trackertime[0]

        for s in range(len(saccades)) :
            if saccades[s][1]-t_0+apres <= (len(trackertime)) :
                for x_data in np.arange((saccades[s][0]-t_0-avant), (saccades[s][1]-t_0+apres)) :
                    velocity[x_data] = np.nan
            else :
                for x_data in np.arange((saccades[s][0]-t_0-avant), (len(trackertime))) :
                    velocity[x_data] = np.nan

        return velocity

    ######################################################################################

    def Fct_velocity (self, x, dir_target, start_anti, v_anti, latence, tau, maxi, do_whitening) :
        return fct_velocity (x, dir_target, start_anti, v_anti, latence, tau, maxi, do_whitening)

    def Fct_position(self, x, data_x, saccades, nb_sacc, dir_target, start_anti, v_anti, latence, tau, maxi, t_0, px_per_deg, avant, apres, do_whitening):
        return fct_position(x, data_x, saccades, nb_sacc, dir_target, start_anti, v_anti, latence, tau, maxi, t_0, px_per_deg, avant, apres, do_whitening)


    def Fct_saccade(self, x, x_0, tau, x1, x2, T0, t1, t2, tr, do_whitening):
        return fct_saccade(x, x_0, tau, x1, x2, T0, t1, t2, tr, do_whitening)



    def Fit_saccade(self, position_x, do_whitening=True, step=2) :

        if do_whitening:
            position_x = whitening(position_x)
            
        from lmfit import  Model, Parameters

        if step == 1 :
            vary = True
        elif step == 2 :
            vary = False


        model = Model(fct_saccade)

        params = Parameters()
        params.add('do_whitening', value=do_whitening, vary=False)
        params.add('x_0', value=position_x[0], min=position_x[0]-0.1, max=position_x[0]+0.1, vary=vary)
        params.add('tau', value=13, min=5, max=40)
        params.add('T0', value=0, min=-15, max=10)
        params.add('t1', value=15, min=10, max=len(position_x))
        params.add('t2', value=12, min=10, max=len(position_x), vary=vary)
        params.add('tr', value=1, min=0, max=15, vary=vary)
        params.add('x1', value=2, min=-5, max=5)
        params.add('x2', value=1, min=-5, max=5, vary=vary)

        if step == 1 :
            result_deg = model.fit(position_x, params, x=np.arange(len(position_x)), nan_policy='omit')

        elif step == 2 :

            out = model.fit(position_x, params, x=np.arange(len(position_x)), nan_policy='omit')
            out.params['x_0'].set(vary=True)
            out.params['t2'].set(vary=True)
            out.params['tr'].set(vary=True)
            out.params['x2'].set(vary=True)

            result_deg = model.fit(position_x, out.params, x=np.arange(len(position_x)), method='nelder', nan_policy='omit')


        return result_deg



    def Fit_trial(self, data_trial, trackertime, dir_target, fct_fit='fct_velocity', data_x=None,
                  param_fit=None, old_latence=None, old_max=None, old_anti=None, TargetOn=None, StimulusOf=None, saccades=None, sup=True, time_sup=-280,
                  avant=5, apres=10, do_whitening=True, step=2) :
                        #maxiter=1000):
        '''
        Returns le resultat du fits de la vitesse de l'œil a un essais avec la fonction reproduisant la vitesse de l'œil lors de la pousuite lisse d'une cible en mouvement

        Parameters
        ----------
        data_trial : ndarray
            si fct_fit = velocity data_trial est la vitesse x pour un essaie enregistré par l'eyetracker
            si fct_fit = position data_trial est la position x pour un essaie enregistré par l'eyetracker
            
        fct_fit :
            fonction utiliser pour fiter les datas

        trackertime : ndarray
            temps du tracker
        TargetOn : int
            temps ou la cible à suivre apparait
        StimulusOf : int
            temps ou le point de fixation disparait
        saccades : list
            liste des saccades edf pour un essaie enregistré par l'eyetracker transformé par la fonction read_edf du module edfreader

        dir_target : float
            -1 ou 1 donne la direction de la cible
        sup : bool
            si True ne fait pas le fit jusqu'au bout, jusqua time_sup avant la fin
        time_sup : int
            temps supprimer si sup is True

        param_fit : dic
            dictionnaire des parametre du fit, chaque parametre est une liste [value fit, min, max]

        step :  nombre de step pour le fit

        Returns
        -------
        result_deg : lmfit.model.ModelResult

        '''

        from lmfit import  Model, Parameters

        t_0 = trackertime[0]

        if param_fit is None :
            if np.isnan(old_latence)==True :
                old_latence = None
            if old_latence is None :
                old_latence = TargetOn-t_0+100
            if old_max is None :
                old_max = 15.
            if old_anti is None :
                old_anti = 0.

            param_fit={'tau':[15.,13.,80.], 'maxi':[old_max,1.,40], 'v_anti':[old_anti,-40.,40.],
                       'latence':[old_latence,TargetOn-t_0+75,'STOP'],
                       'start_anti':[TargetOn-t_0-100, StimulusOf-t_0-200, TargetOn-t_0+75]}

        if param_fit['latence'][2]=='STOP' :
            stop_latence = []
            for s in range(len(saccades)) :
                if (saccades[s][0]-t_0) >= (TargetOn-t_0+100) :
                    stop_latence.append((saccades[s][0]-t_0))
            if stop_latence==[] :
                stop_latence.append(len(trackertime))
            stop = stop_latence[0]
        else :
            stop = param_fit['latence'][2]


        if sup==True :
            data_trial = data_trial[:time_sup]
            trackertime = trackertime[:time_sup]
            if fct_fit == 'fct_position' :
                data_x = data_x[:time_sup]

        if do_whitening:
            data_trial = whitening(data_trial)
            data_x = whitening(data_x)

        params = Parameters()

        if fct_fit == 'fct_velocity' :
            model = Model(fct_velocity, independent_vars=['x'])
        elif fct_fit == 'fct_position' :
            model = Model(fct_position, independent_vars=['x', 'data_x', 'saccades'])
            params.add('px_per_deg', value=self.param_exp['px_per_deg'], vary=False)
            params.add('t_0', value=t_0, vary=False)
            params.add('avant', value=avant, vary=False)
            params.add('apres', value=apres, vary=False)
            params.add('nb_sacc', value=len(saccades), vary=False)

            sacc = np.zeros(len(trackertime))
            i=0
            for s in range(len(saccades)):
                sacc[i] = saccades[s][0] # debut sacc
                sacc[i+1] = saccades[s][1] # fin sacc
                sacc[i+2] = saccades[s][2] # tps sacc
                i = i+3

        if step == 1 :
            vary = True
        elif step == 2 :
            vary = False

        params.add('maxi', value=param_fit['maxi'][0], min=param_fit['maxi'][1], max=param_fit['maxi'][2])
        params.add('latence', value=param_fit['latence'][0], min=param_fit['latence'][1], max=stop)
        params.add('dir_target', value=dir_target, vary=False)
        params.add('tau', value=param_fit['tau'][0], min=param_fit['tau'][1], max=param_fit['tau'][2], vary=vary)
        params.add('start_anti', value=param_fit['start_anti'][0], min=param_fit['start_anti'][1], max=param_fit['start_anti'][2], vary=vary)
        params.add('v_anti', value=param_fit['v_anti'][0], min=param_fit['v_anti'][1], max=param_fit['v_anti'][2], vary=vary)
        params.add('do_whitening', value=do_whitening, vary=False)

        if step == 1 :
            if fct_fit=='fct_velocity' :
                result_deg = model.fit(data_trial, params, x=np.arange(len(trackertime)), nan_policy='omit')
            elif fct_fit=='fct_position' :
                result_deg = model.fit(data_trial, params, x=np.arange(len(trackertime)), data_x=data_x, saccades=sacc, nan_policy='omit')

        elif step == 2 :
            if fct_fit=='fct_velocity' :
                out = model.fit(data_trial, params, x=np.arange(len(trackertime)), nan_policy='omit')
            elif fct_fit=='fct_position' :
                out = model.fit(data_trial, params, x=np.arange(len(trackertime)), data_x=data_x, saccades=sacc, nan_policy='omit')

            # make the other parameters vary now
            out.params['tau'].set(vary=True)
            out.params['start_anti'].set(vary=True)
            out.params['v_anti'].set(vary=True)

            if fct_fit=='fct_velocity' :
                result_deg = model.fit(data_trial, out.params,
                                    x=np.arange(len(trackertime)),
                                    method='nelder', nan_policy='omit')
                                    #fit_kws=dict(maxiter=maxiter))
                # par défaut dans scipy.optimize.minimize(method=’Nelder-Mead’) maxiter=N*200 (N nb de variable)
            elif fct_fit=='fct_position' :
                result_deg = model.fit(data_trial, out.params, x=np.arange(len(trackertime)), data_x=data_x, saccades=sacc, method='nelder', nan_policy='omit')

        return result_deg


    def Fit(self, data, list_events=None, sup=True, time_sup=-280,
            plot=None, fig_width=12, t_label=20, t_text=14, file_fig=None, param_fit=None, stop_recherche_misac=None,
            fct_fit='fct_velocity', step_fit=2, avant=0, apres=15, do_whitening=True) :

        '''
        Renvoie un dictionnaire des paramètres Fit sur l'ensemble des data

        Parameters
        ----------
        data : list
            données edf enregistré par l'eyetracker transformé par la fonction read_edf du module edfreader

        N_trials : int
            nombre de trial par block
        N_blocks : int
            nombre de block
        binomial : ndarray
            direction de la cible pour chaque essais 0 gauche 1 droite [trial, block]
        list_events : list
            liste des noms des évenements dans le fichier asc ['début fixation', 'fin fixation', 'début poursuite', 'fin poursuite']
        sup : bool
            si True ne fait pas le fit jusqu'au bout, jusqua time_sup avant la fin
        time_sup : int
            temps supprimer avant la fin
        observer : str
            nom du sujet
        plot : NoneType or bool
            pour enregistre une figure des fits de tous les essais pour chaque block


        fig_width : int
            taille figure
        t_label : int
            taille x et y label
        t_text : int
            taille text
        file_fig : str
            nom enregistrement figures

        param_fit : dic
            dictionnaire des parametre du fit, chaque parametre est une liste [value fit, min, max]
        stop_recherche_misac : int
            stop recherche de micro_saccade, si None alors arrête la recherche à la fin de la fixation +100ms
        step_fit : nombre de step pour le fit


        Returns
        -------
        param : dict
            chaque parametre sont ordonnée : [block][trial]
                'fit' : best_fit -- resultat du fit avec les parametre changer afin de coller au mieu au data
                'observer' : observer si definit
                'start_anti' : tps ou commence l'anticipation
                'v_anti' : vitesse de l'anticipation
                'latence' : latence visuel
                'tau' : courbe
                'maxi' : maximum
                'moyenne' : moyenne des vitesses de -50 a +50 - ancienne manière de récolté les anticipations

        '''

        if plot is not None :
            import matplotlib.pyplot as plt

        if list_events is None :
            list_events = ['StimulusOn\n', 'StimulusOff\n', 'TargetOn\n', 'TargetOff\n']

        liste_fit = []
        liste_start_anti = []
        liste_liste_v_anti = []
        liste_latence = []
        liste_tau = []
        liste_maxi = []
        liste_saccades = []

        liste_old_anti = []
        liste_old_max = []
        liste_old_latence = []


        for block in range(self.param_exp['N_blocks']) :
            if plot is not None :
                fig, axs = plt.subplots(self.param_exp['N_trials'], 1, figsize=(fig_width, (fig_width*(self.param_exp['N_trials']/2))/1.6180))

            block_fit = []
            block_start_anti = []
            block_liste_v_anti = []
            block_latence = []
            block_tau = []
            block_maxi = []
            block_saccades = []

            block_old_anti = []
            block_old_max = []
            block_old_latence = []

            for trial in range(self.param_exp['N_trials']) :

                print('block, trial = ', block, trial)

                arg = ANEMO.arg(self, data, trial, block, list_events=None)
                saccades = arg.saccades
                dir_target = (arg.bino*2-1)

                start = arg.TargetOn

                StimulusOn_s = arg.StimulusOn - start
                StimulusOf_s = arg.StimulusOf - start
                TargetOn_s = arg.TargetOn - start
                TargetOff_s = arg.TargetOff - start
                trackertime_s = arg.trackertime - start


                if plot is not None :
                    axs[trial].cla() # pour remettre ax figure a zero
                    axs[trial].axvspan(StimulusOn_s, StimulusOf_s, color='k', alpha=0.2)
                    axs[trial].axvspan(StimulusOf_s, TargetOn_s, color='r', alpha=0.2)
                    axs[trial].axvspan(TargetOn_s, TargetOff_s, color='k', alpha=0.15)
                    for s in range(len(saccades)) :
                        axs[trial].axvspan(saccades[s][0]-start, saccades[s][1]-start, color='k', alpha=0.2)


                velocity = ANEMO.velocity_deg(self, data_x=arg.data_x)
                if fct_fit=='fct_velocity' :
                    velocity_y = ANEMO.velocity_deg(self, data_x=arg.data_y)
                    if stop_recherche_misac is None :
                        stop_recherche_misac = arg.TargetOn-arg.t_0+100
                    misac = ANEMO.Microsaccade(self, velocity_x=velocity[:stop_recherche_misac], velocity_y=velocity_y[:stop_recherche_misac], t_0=arg.t_0)
                    saccades.extend(misac)
                velocity_NAN = ANEMO.suppression_saccades(self, velocity=velocity, saccades=saccades, trackertime=arg.trackertime)

                if fct_fit=='fct_velocity' :
                    data_x = arg.data_x
                    data_1 = velocity_NAN
                    data_trial = np.copy(data_1)

                param_sac = []
                if fct_fit == 'fct_position' :
                    data_x = arg.data_x - (arg.data_x[arg.StimulusOf-arg.t_0])
                    data_1 = np.copy(data_x)/arg.px_per_deg
                    data_2 = np.copy(data_1)
                    for s in range(len(arg.saccades)):
                        param_sac.append({})
                        data_sacc = data_1[arg.saccades[s][0]-arg.t_0-avant:arg.saccades[s][1]-arg.t_0+apres]
                        f = ANEMO.Fit_saccade(self, data_sacc, do_whitening=do_whitening)
                        param_sac[s]['x_0'] = f.values['x_0']
                        param_sac[s]['tau'] = f.values['tau']
                        param_sac[s]['x1'] = f.values['x1']
                        param_sac[s]['x2'] = f.values['x2']
                        param_sac[s]['T0'] = f.values['T0']
                        param_sac[s]['t1'] = f.values['t1']
                        param_sac[s]['t2'] = f.values['t2']
                        param_sac[s]['tr'] = f.values['tr']

                        data_2[arg.saccades[s][0]-arg.t_0-avant:arg.saccades[s][1]-arg.t_0+apres] = f.residual+f.values['x_0']

                        axs[trial].plot(trackertime_s[arg.saccades[s][0]-arg.t_0+avant:arg.saccades[s][1]-arg.t_0+apres]-start, f.best_fit , color='darkred', linewidth=2)
                    data_trial = data_2

                debut  = arg.TargetOn - arg.t_0 # TargetOn - temps_0

                ##################################################
                # FIT
                ##################################################
                old_latence, old_max, old_anti = classical_methode(velocity_NAN, arg.StimulusOn, arg.StimulusOf, arg.TargetOn, arg.t_0)
                result_deg = ANEMO.Fit_trial(self, data_trial=data_trial, data_x=data_x, trackertime=arg.trackertime, dir_target=dir_target, param_fit=param_fit,
                                             old_latence=old_latence, old_max=old_max, old_anti=old_anti,
                                             TargetOn=arg.TargetOn, StimulusOf=arg.StimulusOf, saccades=saccades, sup=sup, time_sup=time_sup, step=step_fit,
                                             fct_fit=fct_fit, avant=avant, apres=apres, do_whitening=do_whitening)
                ##################################################

                start_anti = result_deg.values['start_anti']-debut
                v_anti = result_deg.values['v_anti']
                latence = result_deg.values['latence']-debut
                tau = result_deg.values['tau']
                maxi = result_deg.values['maxi']

                '''if np.isnan(velocity_NAN[int(result_deg.values['latence'])]) and np.isnan(velocity_NAN[int(result_deg.values['latence'])-30]) and np.isnan(velocity_NAN[int(result_deg.values['latence'])-70]) ==True :
                    start_anti = np.nan
                    v_anti = np.nan
                    latence = np.nan
                    tau = np.nan
                    maxi = np.nan
                else :
                    axs[trial].bar(latence, 80, bottom=-40, color='r', width=6, linewidth=0)
                    if trial==0 :
                        axs[trial].text(latence+25, -35, "Latence"%(latence), color='r', fontsize=14)'''
                
                block_fit.append(result_deg.best_fit) # result_deg
                block_start_anti.append(start_anti)
                block_liste_v_anti.append(v_anti)
                block_latence.append(latence)
                block_tau.append(tau)
                block_maxi.append(maxi)
                block_saccades.append(param_sac)

                block_old_anti.append(old_anti)
                block_old_max.append(old_max)
                block_old_latence.append(old_latence-debut)

                if plot is not None :
                    #axs[trial].cla() # pour remettre ax figure a zero
                    #axs[trial].axis([StimulusOn_s-10, TargetOff_s+10, -40, 40])
                    axs[trial].xaxis.set_ticks(range(StimulusOf_s-199, TargetOff_s+10, 500))


                    #axs[trial].plot(trackertime_s[:time_sup], result_deg.init_fit, 'r--', linewidth=2)

                    if fct_fit=='fct_velocity' :
                        axs[trial].axis([TargetOn_s-700, TargetOff_s+10, -40, 40])
                        axs[trial].plot(trackertime_s, np.ones(np.shape(trackertime_s)[0])*dir_target*(15), color='k', linewidth=0.2, alpha=0.2)
                        axs[trial].plot(trackertime_s, np.ones(np.shape(trackertime_s)[0])*dir_target*(10), color='k', linewidth=0.2, alpha=0.2)
                        rere = fct_velocity(x=range(len(arg.trackertime)), dir_target=dir_target, start_anti=start_anti+debut,
                                            v_anti=v_anti, latence=latence+debut, tau=tau, maxi=maxi, do_whitening=False)

                    if fct_fit=='fct_position' :
                        axs[trial].axis([TargetOn_s-700, TargetOff_s+10, -(arg.screen_width_px/arg.px_per_deg)/2, (arg.screen_width_px/arg.px_per_deg)/2])
                        sacc = np.zeros(len(arg.trackertime))
                        i=0
                        for s in range(len(saccades)):
                            sacc[i] = saccades[s][0] # debut sacc
                            sacc[i+1] = saccades[s][1] # fin sacc
                            sacc[i+2] = saccades[s][2] # tps sacc
                            i = i+3
                        rere = fct_position(x=range(len(arg.trackertime)), data_x=arg.data_x, saccades=sacc, nb_sacc=len(saccades),
                                            dir_target=dir_target, start_anti=start_anti+debut,v_anti=v_anti, latence=latence+debut,
                                            tau=tau, maxi=maxi, t_0=arg.t_0, px_per_deg=self.param_exp['px_per_deg'], avant=avant,
                                            apres=apres, do_whitening=False)

                    axs[trial].plot(trackertime_s[:time_sup], rere[:time_sup], color='r', linewidth=2)
                    axs[trial].bar(latence, 80, bottom=-40, color='r', width=6, linewidth=0)
                    axs[trial].plot(trackertime_s, data_1, color='k', alpha=0.6)

                    if trial==0 :
                        axs[trial].text(StimulusOn_s+(StimulusOf_s-StimulusOn_s)/2, 31, "FIXATION", color='k', fontsize=t_text+2, ha='center', va='bottom')
                        axs[trial].text(StimulusOf_s+(TargetOn_s-StimulusOf_s)/2, 31, "GAP", color='r', fontsize=t_text+2, ha='center', va='bottom')
                        axs[trial].text(TargetOn_s+(TargetOff_s-TargetOn_s)/2, 31, "POURSUITE", color='k', fontsize=t_text+2, ha='center', va='bottom')
                        axs[trial].text(latence+25, -35, "Latence"%(latence), color='r', fontsize=t_text)#,  weight='bold')
                    #axs[trial].text(StimulusOn+15, -2, "%s"%(result.fit_report()), color='k', fontsize=15)
                    axs[trial].text(StimulusOn_s+15, 18, "start_anti: %s \nv_anti: %s"%(start_anti, v_anti), color='k', fontsize=t_text, va='bottom')
                    axs[trial].text(StimulusOn_s+15, -18, "latence: %s \ntau: %s \nmaxi: %s"%(latence, tau, maxi), color='k', fontsize=t_text, va='top')

                    axs[trial].set_xlabel('Time (ms)', fontsize=t_label)
                    axs[trial].set_ylabel(trial+1, fontsize=t_label)

                    axs[trial].xaxis.set_ticks_position('bottom')
                    axs[trial].yaxis.set_ticks_position('left')

            liste_fit.append(block_fit)
            liste_start_anti.append(block_start_anti)
            liste_liste_v_anti.append(block_liste_v_anti)
            liste_latence.append(block_latence)
            liste_tau.append(block_tau)
            liste_maxi.append(block_maxi)
            liste_saccades.append(block_saccades)

            liste_old_anti.append(block_old_anti)
            liste_old_max.append(block_old_max)
            liste_old_latence.append(block_old_latence)

            if plot is not None :
                plt.tight_layout() # pour supprimer les marge trop grande
                plt.subplots_adjust(hspace=0) # pour enlever espace entre les figures

                if file_fig is None :
                    f = 'Fit_%s.pdf'%(block+1)
                else :
                    f = file_fig + '_%s.pdf'%(block+1)
                plt.savefig(f)
                plt.close()

        param = {}
        if self.param_exp['observer'] is not None :
            param['observer'] = self.param_exp['observer']
        param['fit'] = liste_fit
        param['start_anti'] = liste_start_anti
        param['v_anti'] = liste_liste_v_anti
        param['latence'] = liste_latence
        param['tau'] = liste_tau
        param['maxi'] = liste_maxi
        param['saccades'] = liste_saccades

        param['old_anti'] = liste_old_anti
        param['old_max'] = liste_old_max
        param['old_latence'] = liste_old_latence

        return param


    ######################################################################################

    def figure(self, ax, velocity, saccades, StimulusOn, StimulusOf, TargetOn, TargetOff, trackertime, start, bino, plot, t_label,
               sup=True, time_sup=-280, report=None, param_fit=None, step_fit=2, fct_fit='fct_velocity') :
        '''
        Returns figure

        Parameters
        ----------
        ax : AxesSubplot
            ax sur lequel le figure doit être afficher

        velocity : ndarray
            vitesse de l'œil en deg/sec

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

        bino : float
            0 ou 1 donne la direction de la cible

        sup : bool
            si True ne fait pas le fit jusqu'au bout, jusqua time_sup avant la fin
        time_sup : int
            temps supprimer avant la fin

        plot : str
            si 'fonction' n'affiche que la fonction exponentiel
            si 'velocity' n'affiche que la vitesse de l'œil
            si 'Fitvelocity' affiche la vitesse œil + fit

        t_label : int
            taille x et y label

        report : NoneType or bool
            si != None renvoie le rapport fit

        param_fit : dic
            dictionnaire des parametre du fit, chaque parametre est une liste [value fit, min, max]

        step_fit : nombre de step pour le fit

        Returns
        -------
        ax : AxesSubplot
            figure
        report : str
            fit report
        '''

        t_0 = trackertime[0]
        StimulusOn_s = StimulusOn - start
        StimulusOf_s = StimulusOf - start
        TargetOn_s = TargetOn - start
        TargetOff_s = TargetOff - start
        trackertime_s = trackertime - start
        #bino = self.param_exp['p'][trial, block, 0]
        dir_target = (bino*2-1)


        if plot != 'fonction' :
            ax.plot(trackertime_s, velocity, color='k', alpha=0.4)

            # Saccade
            for s in range(len(saccades)) :
                ax.axvspan(saccades[s][0]-start, saccades[s][1]-start, color='k', alpha=0.15)

        if plot != 'velocity' :
            # FIT
            result_deg = ANEMO.Fit_trial(self, data_trial=velocity, trackertime=trackertime, dir_target=dir_target,param_fit=param_fit, TargetOn=TargetOn, StimulusOf=StimulusOf,
                                         saccades=saccades, sup=sup, time_sup=time_sup, step=step_fit, fct_fit=fct_fit)

        if plot == 'Fitvelocity' :

            debut  = TargetOn - t_0 # TargetOn - temps_0
            start_anti = result_deg.values['start_anti']
            v_anti = result_deg.values['v_anti']
            latence = result_deg.values['latence']
            tau = result_deg.values['tau']
            maxi = result_deg.values['maxi']
            #result_fit = result_deg.best_fit
            result_fit = fct_velocity (x=np.arange(len(trackertime_s)), dir_target=dir_target, start_anti=start_anti, v_anti=v_anti, latence=latence, tau=tau, maxi=maxi)

        if plot == 'fonction' :

            debut  = TargetOn - t_0 # TargetOn - temps_0
            start_anti = TargetOn-t_0-100
            v_anti = -20
            latence = TargetOn-t_0+100
            tau = 15.
            maxi = 15.
            result_fit = fct_velocity (x=np.arange(len(trackertime_s)), dir_target=dir_target, start_anti=start_anti, v_anti=v_anti, latence=latence, tau=tau, maxi=maxi)
            maxi = bino*maxi + bino*result_fit[latence]
            ax.plot(trackertime_s[int(latence)+250:], result_fit[int(latence)+250:], 'k', linewidth=2)

        ax.axvspan(StimulusOn_s, StimulusOf_s, color='k', alpha=0.2)
        ax.axvspan(StimulusOf_s, TargetOn_s, color='r', alpha=0.2)
        ax.axvspan(TargetOn_s, TargetOff_s, color='k', alpha=0.15)

        if plot != 'velocity' :
            # COSMETIQUE
            ax.plot(trackertime_s[:int(start_anti)], result_fit[:int(start_anti)], 'k', linewidth=2)
            #ax.plot(trackertime_s[int(latence)+250:-280], result_fit[int(latence)+250:], 'k', linewidth=2)
            # V_a ------------------------------------------------------------------------
            ax.plot(trackertime_s[int(start_anti):int(latence)], result_fit[int(start_anti):int(latence)], c='r', linewidth=2)
            ax.annotate('', xy=(trackertime_s[int(latence)], result_fit[int(latence)]-3), xycoords='data', fontsize=t_label/1.5,
                        xytext=(trackertime_s[int(start_anti)], result_fit[int(start_anti)]-3), textcoords='data', arrowprops=dict(arrowstyle="->", color='r'))
            # Start_a --------------------------------------------------------------------
            ax.bar(trackertime_s[int(start_anti)], 80, bottom=-40, color='k', width=4, linewidth=0, alpha=0.7)
            # latence --------------------------------------------------------------------
            ax.bar(trackertime_s[int(latence)], 80, bottom=-40, color='firebrick', width=4, linewidth=0, alpha=1)
            # tau ------------------------------------------------------------------------
            ax.plot(trackertime_s[int(latence):int(latence)+250], result_fit[int(latence):int(latence)+250], c='darkred', linewidth=2)
            # Max ------------------------------------------------------------------------
            ax.plot(trackertime_s, np.zeros(len(trackertime_s)), '--k', linewidth=1, alpha=0.5)
            ax.plot(trackertime_s[int(latence):], np.ones(len(trackertime_s[int(latence):]))*result_fit[int(latence)+400], '--k', linewidth=1, alpha=0.5)

        if plot == 'Fitvelocity' :

            # V_a ------------------------------------------------------------------------
            ax.text((trackertime_s[int(start_anti)]+trackertime_s[int(latence)])/2, result_fit[int(start_anti)]-15,
                    r"A$_a$ = %0.2f °/s$^2$"%(v_anti), color='r', fontsize=t_label/1.5, ha='center')
            # Start_a --------------------------------------------------------------------
            ax.text(trackertime_s[int(start_anti)]-25, -35, "Start anticipation = %0.2f ms"%(start_anti-debut),
                    color='k', alpha=0.7, fontsize=t_label/1.5, ha='right')
            # latence --------------------------------------------------------------------
            ax.text(trackertime_s[int(latence)]+25, -35, "Latency = %0.2f ms"%(latence-debut),
                    color='firebrick', fontsize=t_label/1.5, va='center')
            # tau ------------------------------------------------------------------------
            ax.text(trackertime_s[int(latence)]+70+t_label, (result_fit[int(latence)]),
                    r"= %0.2f"%(tau), color='darkred',va='bottom', fontsize=t_label/1.5)
            ax.annotate(r'$\tau$', xy=(trackertime_s[int(latence)]+50, result_fit[int(latence)+50]), xycoords='data', fontsize=t_label/1., color='darkred', va='bottom',
                        xytext=(trackertime_s[int(latence)]+70, result_fit[int(latence)]), textcoords='data', arrowprops=dict(arrowstyle="->", color='darkred'))
            # Max ------------------------------------------------------------------------
            ax.text(TargetOn_s+450+25, (result_fit[int(latence)]+result_fit[int(latence)+250])/2,
                    "Steady State = %0.2f °/s"%(maxi), color='k', va='center', fontsize=t_label/1.5)
            ax.annotate('', xy=(TargetOn_s+450, result_fit[int(latence)]), xycoords='data', fontsize=t_label/1.5,
                        xytext=(TargetOn_s+450, result_fit[int(latence)+250]), textcoords='data', arrowprops=dict(arrowstyle="<->"))

        if plot == 'fonction' :

            # COSMETIQUE
            ax.text(StimulusOf_s+(TargetOn_s-StimulusOf_s)/2, 31, "GAP", color='k', fontsize=t_label*1.5, ha='center', va='center', alpha=0.5)
            ax.text((StimulusOf_s-750)/2, 31, "FIXATION", color='k', fontsize=t_label*1.5, ha='center', va='center', alpha=0.5)
            ax.text((750-TargetOn_s)/2, 31, "PURSUIT", color='k', fontsize=t_label*1.5, ha='center', va='center', alpha=0.5)
            ax.text(TargetOn_s, 15, "Anticipation", color='r', fontsize=t_label, ha='center')

            # V_a ------------------------------------------------------------------------
            ax.text(TargetOn_s-50, -5, r"A$_a$", color='r', fontsize=t_label/1.5, ha='center', va='top')
            # Start_a --------------------------------------------------------------------
            ax.text(TargetOn_s-100-25, -35, "Start anticipation", color='k', fontsize=t_label, alpha=0.7, ha='right')
            # latence --------------------------------------------------------------------
            ax.text(TargetOn_s+99+25, -35, "Latency", color='firebrick', fontsize=t_label)
            # tau ------------------------------------------------------------------------
            ax.annotate(r'$\tau$', xy=(trackertime_s[int(latence)]+15, result_fit[int(latence)+15]), xycoords='data', fontsize=t_label/1., color='darkred', va='bottom',
                    xytext=(trackertime_s[int(latence)]+70, result_fit[int(latence)+7]), textcoords='data', arrowprops=dict(arrowstyle="->", color='darkred'))
            # Max ------------------------------------------------------------------------
            ax.text(TargetOn_s+400+25, ((result_fit[int(latence)+400])/2),
                   'Steady State', color='k', fontsize=t_label, va='center')
            ax.annotate('', xy=(TargetOn_s+400, 0), xycoords='data', fontsize=t_label/1.5,
                    xytext=(TargetOn_s+400, result_fit[int(latence)+400]), textcoords='data', arrowprops=dict(arrowstyle="<->"))

        #axs[x].axis([StimulusOn_s-10, TargetOff_s+10, -40, 40])
        ax.axis([-750, 750, -39.5, 39.5])
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_tick_params(labelsize=t_label/2)
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_tick_params(labelsize=t_label/2)
        ax.set_xlabel('Time (ms)', fontsize=t_label)

        if report is None :
            return ax
        else :
            return ax, result_deg.fit_report()


    def figure_position(self, ax, data_x, data_y, dir_target, saccades, StimulusOn, StimulusOf, TargetOn, TargetOff, trackertime, start, t_label) :

        '''
        Returns figure de la position de l'œil pendant l'enregistrement pout un essai

        Parameters
        ----------
        ax : AxesSubplot
            ax sur lequel le figure doit être afficher

        data_x : ndarray
            position x pour un essaie enregistré par l'eyetracker transformé par la fonction read_edf du module edfreader
        data_y : ndarray
            position y pour un essaie enregistré par l'eyetracker transformé par la fonction read_edf du module edfreader

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

        bino : float
            0 ou 1 donne la direction de la cible

        V_X : float
            vitesse de la cible en pixel/s
        RashBass : int
            temps que met la cible a arriver au centre de l'écran en ms (pour reculer la cible à t=0 de sa vitesse * latence=RashBass)
        stim_tau : float

        screen_width_px : int
            widht ecran en pixel
        screen_height_px : int
            height écran en pixel

        t_label : int
            taille x et y label


        Returns
        -------
        ax : AxesSubplot
            figure
        '''

        StimulusOn_s = StimulusOn - start
        StimulusOf_s = StimulusOf - start
        TargetOn_s = TargetOn - start
        TargetOff_s = TargetOff - start
        trackertime_s = trackertime - start
        t_0 = trackertime_s[0]

        #------------------------------------------------
        # TARGET
        #------------------------------------------------

        Target_trial = []
        x = self.param_exp['screen_width_px']/2

        for t in range(len(trackertime_s)):
            if t < (TargetOn_s-t_0) :
                x = self.param_exp['screen_width_px']/2
            elif t == (TargetOn_s-t_0) :
                # la cible à t=0 recule de sa vitesse * latence=RashBass (ici mis en ms)
                x = x -(dir_target * ((self.param_exp['V_X']/1000)*self.param_exp['RashBass']))
            elif (t > (TargetOn_s-t_0) and t <= ((TargetOn_s-t_0)+self.param_exp['stim_tau']*1000)) :
                x = x + (dir_target*(self.param_exp['V_X']/1000))
            else :
                x = x
            Target_trial.append(x)
        #------------------------------------------------

        ax.plot(trackertime_s, np.ones(len(trackertime_s))*(self.param_exp['screen_height_px']/2), color='grey', linewidth=1.5)
        ax.plot(trackertime_s, data_y, color='c', linewidth=1.5)

        ax.plot(trackertime_s, Target_trial, color='k', linewidth=1.5)
        ax.plot(trackertime_s, data_x, color='r', linewidth=1.5)

        ax.axvspan(StimulusOn_s, StimulusOf_s, color='k', alpha=0.2)
        ax.axvspan(StimulusOf_s, TargetOn_s, color='r', alpha=0.2)
        ax.axvspan(TargetOn_s, TargetOff_s, color='k', alpha=0.15)

        for s in range(len(saccades)) :
            ax. axvspan(saccades[s][0]-start, saccades[s][1]-start, color='k', alpha=0.15)

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.yaxis.set_ticks(range(0, self.param_exp['screen_width_px']+1, int(self.param_exp['screen_width_px']/2)))
        ax.yaxis.set_ticklabels(range(0, self.param_exp['screen_width_px']+1, int(self.param_exp['screen_width_px']/2)), fontsize=t_label/2)

        ax.xaxis.set_ticks(range(TargetOn_s-1100, TargetOff_s, 100))
        ax.xaxis.set_ticklabels(range(TargetOn_s-1100, TargetOff_s, 100), fontsize=t_label/2)

        return ax


    ######################################################################################
    def plot_position(self, data, list_events=None, fig=None, axs=None, fig_width=10, t_label=20, file_fig=None):

        '''
        Returns figure de la position de l'œil pendant l'enregistrement pour tous les essais

        Parameters
        ----------
        data : list
            données edf enregistré par l'eyetracker transformé par la fonction read_edf du module edfreader

        N_trials : int
            nombre de trial par block
        N_blocks : int
            nombre de blocks
        bino : ndarray
            direction de la cible pour chaque essais 0 gauche 1 droite
        V_X : float
            vitesse de la cible en pixel/s
        RashBass : int
            temps que met la cible a arriver au centre de l'écran en ms (pour reculer la cible à t=0 de sa vitesse * latence=RashBass)
        stim_tau : float
        list_events : list
            liste des noms des évenements dans le fichier asc ['début fixation', 'fin fixation', 'début poursuite', 'fin poursuite']

        screen_width_px : int
            widht ecran en pixel
        screen_height_px : int
            height écran en pixel

        fig : NoneType ou matplotlib.figure.Figure
            figure ou sera afficher les position de l'œil
        axs : NoneType ou ndarray
            axs ou sera afficher
        fig_width : int
            taille figure
        t_label : int
            taille x et y label
        file_fig : str
            nom enregistrement figures
        '''
        import matplotlib.pyplot as plt

        if list_events is None :
            list_events = ['StimulusOn\n', 'StimulusOff\n', 'TargetOn\n', 'TargetOff\n']

        for block in range(self.param_exp['N_blocks']) :

            if fig is None:
                fig, axs = plt.subplots(self.param_exp['N_trials'], 1, figsize=(fig_width, (fig_width*(self.param_exp['N_trials']/2))/1.6180))

            for trial in range(self.param_exp['N_trials']) :

                trial_data = trial + self.param_exp['N_trials']*block

                data_x = data[trial_data]['x']
                data_y = data[trial_data]['y']
                trackertime = data[trial_data]['trackertime']

                for events in range(len(data[trial_data]['events']['msg'])) :
                    if data[trial_data]['events']['msg'][events][1] == list_events[0] :
                        StimulusOn = data[trial_data]['events']['msg'][events][0]
                    if data[trial_data]['events']['msg'][events][1] == list_events[1] :
                        StimulusOf = data[trial_data]['events']['msg'][events][0]
                    if data[trial_data]['events']['msg'][events][1] == list_events[2] :
                        TargetOn = data[trial_data]['events']['msg'][events][0]
                    if data[trial_data]['events']['msg'][events][1] == list_events[3] :
                        TargetOff = data[trial_data]['events']['msg'][events][0]

                saccades = data[trial_data]['events']['Esac']
                bino_trial = self.param_exp['p'][trial, block, 0]
                dir_target = bino_trial*2 - 1
                start = TargetOn

                axs[trial] = ANEMO.figure_position(self, ax=axs[trial], data_x=data_x, data_y=data_y, dir_target=dir_target, saccades=saccades, StimulusOn=StimulusOn,
                                                   StimulusOf=StimulusOf, TargetOn=TargetOn, TargetOff=TargetOff, trackertime=trackertime,
                                                   start=start, t_label=t_label)

                axs[trial].axis([StimulusOf-start-30, TargetOff-start+30, -30, 1280+30])
                axs[trial].set_xlabel('Time (ms)', fontsize=t_label)
                axs[trial].set_ylabel(trial+1, fontsize=t_label)

            plt.tight_layout() # pour supprimer les marge trop grande
            plt.subplots_adjust(hspace=0) # pour enlever espace entre les figures

            if file_fig is None :
                file_fig = 'enregistrement'
            plt.savefig('%s_%s.pdf'%(file_fig, block+1))
        plt.close()


    def plot_velocity(self, data, trials=0, block=0, list_events=None, stop_recherche_misac=None,
                      fig_width=15, t_titre=35, t_label=20, fct_fit='fct_velocity'):
        '''
        Renvoie les figures de la vitesse de l'œil

        Parameters
        ----------
        data : list
            données edf enregistré par l'eyetracker transformé par la fonction read_edf du module edfreader

        trials : int ou  list
            numéro des essais que l'on veux afficher
        block : int
            numéro du block
        N_trials : int
            nombre de trial par block
        list_events : list
            liste des noms des évenements dans le fichier asc ['début fixation', 'fin fixation', 'début poursuite', 'fin poursuite']
        stop_recherche_misac : int
            stop recherche de micro_saccade, si None alors arrête la recherche à la fin de la fixation +100ms

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

        import matplotlib.pyplot as plt

        if list_events is None :
            list_events = ['StimulusOn\n', 'StimulusOff\n', 'TargetOn\n', 'TargetOff\n']

        if type(trials) is not list :
            trials = [trials]

        fig, axs = plt.subplots(len(trials), 1, figsize=(fig_width, (fig_width*(len(trials)/2)/1.6180)))

        x = 0
        for t in trials :

            trial_data = t + self.param_exp['N_trials']*block

            if len(trials)==1:
                ax = axs
            else :
                ax = axs[x]

            data_x = data[trial_data]['x']
            data_y = data[trial_data]['y']
            trackertime = data[trial_data]['trackertime']

            for events in range(len(data[trial_data]['events']['msg'])) :
                if data[trial_data]['events']['msg'][events][1] == list_events[0] :
                    StimulusOn = data[trial_data]['events']['msg'][events][0]
                if data[trial_data]['events']['msg'][events][1] == list_events[1] :
                    StimulusOf = data[trial_data]['events']['msg'][events][0]
                if data[trial_data]['events']['msg'][events][1] == list_events[2] :
                    TargetOn = data[trial_data]['events']['msg'][events][0]
                if data[trial_data]['events']['msg'][events][1] == list_events[3] :
                    TargetOff = data[trial_data]['events']['msg'][events][0]

            saccades = data[trial_data]['events']['Esac']

            t_0 = data[trial_data]['trackertime'][0]

            velocity = ANEMO.velocity_deg(self, data_x=data_x)
            velocity_y = ANEMO.velocity_deg(self, data_x=data_y)

            if stop_recherche_misac is None :
                stop_recherche_misac = TargetOn-t_0+100

            misac = ANEMO.Microsaccade(self, velocity_x=velocity[:stop_recherche_misac], velocity_y=velocity_y[:stop_recherche_misac], t_0=t_0)
            saccades.extend(misac)

            velocity_NAN = ANEMO.suppression_saccades(self, velocity=velocity, saccades=saccades, trackertime=trackertime)

            start = TargetOn

            ax = ANEMO.figure(self, ax=ax, velocity=velocity_NAN, saccades=saccades, StimulusOn=StimulusOn, StimulusOf=StimulusOf, TargetOn=TargetOn,
                              TargetOff=TargetOff, trackertime=trackertime, start=start, bino=0, plot='velocity', t_label=t_label, fct_fit=fct_fit)

            if x == int((len(trials)-1)/2) :
                ax.set_ylabel('Velocity (°/s)', fontsize=t_label)
            if x!= (len(trials)-1) :
                ax.set_xticklabels([])
            if x==0 :
                ax.set_title('Eye Movement', fontsize=t_titre, x=0.5, y=1.05)

            x=x+1

        plt.tight_layout() # pour supprimer les marge trop grande
        plt.subplots_adjust(hspace=0) # pour enlever espace entre les figures
        return fig, axs


    def plot_Fit(self, data, trials=0, block=0, list_events=None, stop_recherche_misac=None, param_fit=None,
                 plot='fonction', fig_width=15, t_titre=35, t_label=20, report=None, sup=True, time_sup=-280, step_fit=2, fct_fit='fct_velocity'):

        '''
        Renvoie les figures du Fit

        Parameters
        ----------
        data : list
            données edf enregistré par l'eyetracker transformé par la fonction read_edf du module edfreader

        bino : ndarray
            direction de la cible pour chaque essais 0 gauche 1 droite

        trials : int ou  list
            numéro des essais que l'on veux afficher
        block : int
            numéro du block
        N_trials : int
            nombre de trial par block
        list_events : list
            liste des noms des évenements dans le fichier asc ['début fixation', 'fin fixation', 'début poursuite', 'fin poursuite']
        stop_recherche_misac : int
            stop recherche de micro_saccade, si None alors arrête la recherche à la fin de la fixation +100ms

        plot : str
            si 'fonction' n'affiche que la fonction exponentiel
            si 'Fitvelocity' affiche la vitesse œil + fit

        fig_width : int
            taille figure
        t_titre : int
            taille titre de la figur
        t_label : int
            taille x et y label
        report : NoneType or bool
            si != None renvoie le rapport fit

        Returns
        -------
        fig : matplotlib.figure.Figure
            figure
        ax : AxesSubplot
            figure
        '''

        import matplotlib.pyplot as plt

        if type(trials) is not list :
            trials = [trials]

        if list_events is None :
            list_events = ['StimulusOn\n', 'StimulusOff\n', 'TargetOn\n', 'TargetOff\n']

        fig, axs = plt.subplots(len(trials), 1, figsize=(fig_width, (fig_width*(len(trials)/2)/1.6180)))

        results = []
        x = 0
        for t in trials :

            trial_data = t + self.param_exp['N_trials']*block
            bino_trial = self.param_exp['p'][t, block, 0]

            if len(trials)==1:
                ax = axs
            else :
                ax = axs[x]

            data_x = data[trial_data]['x']
            data_y = data[trial_data]['y']
            trackertime = data[trial_data]['trackertime']

            for events in range(len(data[trial_data]['events']['msg'])) :
                if data[trial_data]['events']['msg'][events][1] == list_events[0] :
                    StimulusOn = data[trial_data]['events']['msg'][events][0]
                if data[trial_data]['events']['msg'][events][1] == list_events[1] :
                    StimulusOf = data[trial_data]['events']['msg'][events][0]
                if data[trial_data]['events']['msg'][events][1] == list_events[2] :
                    TargetOn = data[trial_data]['events']['msg'][events][0]
                if data[trial_data]['events']['msg'][events][1] == list_events[3] :
                    TargetOff = data[trial_data]['events']['msg'][events][0]

            saccades = data[trial_data]['events']['Esac']
            t_0 = data[trial_data]['trackertime'][0]

            velocity = ANEMO.velocity_deg(self, data_x=data_x)
            velocity_y = ANEMO.velocity_deg(self, data_x=data_y)

            if stop_recherche_misac is None :
                stop_recherche_misac = TargetOn-t_0+100

            misac = ANEMO.Microsaccade(self, velocity_x=velocity[:stop_recherche_misac], velocity_y=velocity_y[:stop_recherche_misac], t_0=t_0)
            saccades.extend(misac)

            velocity_NAN = ANEMO.suppression_saccades(self, velocity=velocity, saccades=saccades, trackertime=trackertime)

            start = TargetOn

            if report is None :
                ax = ANEMO.figure(self, ax=ax, velocity=velocity_NAN, saccades=saccades, StimulusOn=StimulusOn, StimulusOf=StimulusOf,
                                  TargetOn=TargetOn, TargetOff=TargetOff, trackertime=trackertime, start=start, bino=bino_trial,
                                  plot=plot, t_label=t_label, sup=sup, time_sup=time_sup, report=report, param_fit=param_fit,
                                  step_fit=step_fit, fct_fit=fct_fit)
            else :
                ax, result = ANEMO.figure(self, ax=ax, velocity=velocity_NAN, saccades=saccades, StimulusOn=StimulusOn, StimulusOf=StimulusOf,
                                          TargetOn=TargetOn, TargetOff=TargetOff, trackertime=trackertime, start=start, bino=bino_trial,
                                          plot=plot, t_label=t_label, sup=sup, time_sup=time_sup, report=report, param_fit=param_fit,
                                          step_fit=step_fit, fct_fit=fct_fit)
                results.append(result)

            if x == int((len(trials)-1)/2) :
                ax.set_ylabel('Velocity (°/s)', fontsize=t_label)
            if x!= (len(trials)-1) :
                ax.set_xticklabels([])
            if x==0 :
                if plot=='fonction':
                    ax.set_title('Fit Function', fontsize=t_titre, x=0.5, y=1.05)
                if plot=='velocity':
                    ax.set_title('Eye Movement', fontsize=t_titre, x=0.5, y=1.05)
                else :
                    ax.set_title('Velocity Fit', fontsize=t_titre, x=0.5, y=1.05)

            x=x+1

        plt.tight_layout() # pour supprimer les marge trop grande
        plt.subplots_adjust(hspace=0) # pour enlever espace entre les figures

        if report is None :
            return fig, axs
        else :
            return fig, axs, results
