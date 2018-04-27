#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

''' Revoir les docstring'''


class ANEMO(object):
    """ docstring for the ANEMO class. """

    ######################################################################################

    def velocity_deg(data_x, px_per_deg) :
        '''
        Return la vitesse de l'œuil en deg/sec

        Parameters
        ----------
        data_x : ndarray
            position x pour un essai enregistré par l'eyetracker transformé par la fonction read_edf du module edfreader
        px_per_deg : float
            nb de pixel par degres


        Returns
        -------
        gradient_deg : ndarray
            vitesse de l'œuil en deg/sec
        '''
        gradient_x = np.gradient(data_x)
        gradient_deg = gradient_x * 1/px_per_deg * 1000 # gradient en deg/sec

        return gradient_deg

    # y ajouter microsaccade ?
    def suppression_saccades(velocity, saccades, trackertime) :
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


        Returns
        -------
        new_velocity : ndarray
            vitesse de l'œil en deg/sec sans les saccades

        '''

        trackertime_0 = trackertime[0]

        for s in range(len(saccades)) :
            if saccades[s][1]-trackertime_0+15 <= (len(trackertime)) :
                for x_data in np.arange((saccades[s][0]-trackertime_0-5), (saccades[s][1]-trackertime_0+15)) :
                    velocity[x_data] = np.nan
            else :
                for x_data in np.arange((saccades[s][0]-trackertime_0-5), (len(trackertime))) :
                    velocity[x_data] = np.nan

        return velocity


    ######################################################################################

    def fct_exponentiel (x, bino, start_anti, v_anti, latence, tau, maxi) : # RENOMMER !!!

        '''
        Fonction reproduisant la vitesse de l'œuil lors de la pousuite lisse d'une cible en mouvement

        Parameters
        ----------
        x : ndarray


        bino : float
            0 ou 1 donne la direction de la cible
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
            vitesse de l'œuil
        '''

        v_anti = v_anti/1000 # pour passer de sec à ms
        time = np.arange(len(x))
        vitesse = []

        for t in range(len(time)):

            if start_anti >= latence :
                if time[t] < latence :
                    vitesse.append(0)
                else :
                    vitesse.append((bino*2-1)*maxi*(1-np.exp(-1/tau*(time[t]-latence))))
            else :
                if time[t] < start_anti :
                    vitesse.append(0)
                else :
                    if time[t] < latence :
                        #vitesse.append((bino*2-1)*(time[t]-start_anti)*v_anti)
                        vitesse.append((time[t]-start_anti)*v_anti)
                        x = (time[t]-start_anti)*v_anti
                    else :
                        vitesse.append((bino*2-1)*maxi*(1-np.exp(-1/tau*(time[t]-latence)))+x)

        return vitesse

    def Fit_exponentiel(data_x, trackertime, TargetOn, StimulusOf, saccades, bino, sup=True):

        '''
        Returns le resultat du fits de la vitesse de l'œuil a un essais avec la fonction reproduisant la vitesse de l'œuil lors de la pousuite lisse d'une cible en mouvement

        Parameters
        ----------
        data_x : ndarray
            position x pour un essaie enregistré par l'eyetracker transformé par la fonction read_edf du module edfreader

        trackertime : ndarray
            temps du tracker
        TargetOn : int
            temps ou la cible à suivre apparait
        StimulusOf : int
            temps ou le point de fixation disparait
        saccades : list
            liste des saccades edf pour un essaie enregistré par l'eyetracker transformé par la fonction read_edf du module edfreader

        bino : float
            0 ou 1 donne la direction de la cible
        sup : bool
            si True ne fait pas le fit jusqu'au bout, jusqua -280 avant la fin


        Returns
        -------
        result_deg : lmfit.model.ModelResult

        '''

        from lmfit import  Model, Parameters
        #import lmfit
        #print(lmfit.__version__)

        trackertime_0 = trackertime[0]

        stop_latence = []
        for s in range(len(saccades)) :
            if (saccades[s][0]-trackertime_0) >= (TargetOn-trackertime_0+100) :
                stop_latence.append((saccades[s][0]-trackertime_0))
        if stop_latence==[] :
            stop_latence.append(len(trackertime))

        model = Model(ANEMO.fct_exponentiel)#, nan_policy='propagate')# a tester pour lmfit 0.9.9
        params = Parameters()

        params.add('tau', value=15., min=13., max=80.)#, vary=False)
        params.add('maxi', value=15., min=1., max=40.)#, vary=False)
        params.add('latence', value=TargetOn-trackertime_0+100, min=TargetOn-trackertime_0+75, max=stop_latence[0])
        params.add('start_anti', value=TargetOn-trackertime_0-100, min=StimulusOf-trackertime_0-200, max=TargetOn-trackertime_0+75)
        params.add('v_anti', value=(bino*2-1)*0, min=-40., max=40.)
        params.add('bino', value=bino, min=0, max=1, vary=False)

        #result_deg = model.fit(new_gradient_deg, params, x=new_time)
        if sup==True :
            result_deg = model.fit(data_x[:-280], params, x=trackertime[:-280], fit_kws={'nan_policy': 'omit'}) #, fit_kws={'nan_policy': 'propagate'}) #lmfit 0.9.9
        else :
            result_deg = model.fit(data_x, params, x=trackertime, fit_kws={'nan_policy': 'omit'}) #, fit_kws={'nan_policy': 'propagate'}) #lmfit 0.9.9

        return result_deg

    def Fit (data, N_trials, N_blocks, binomial, px_per_deg, observer=None, plot=None, fig_width=12, t_label=20, t_text=14, file_fig=None) : # a netoyer

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
        px_per_deg : float
            nb de pixel par degres
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

        liste_fit = []
        liste_start_anti = []
        liste_liste_v_anti = []
        liste_latence = []
        liste_tau = []
        liste_maxi = []
        liste_mean = []

        for block in range(N_blocks) :
            if plot is not None :
                fig, axs = plt.subplots(N_trials, 1, figsize=(fig_width, (fig_width*(N_trials/2))/1.6180))

            block_fit = []
            block_start_anti = []
            block_liste_v_anti = []
            block_latence = []
            block_tau = []
            block_maxi = []
            block_mean = []

            for trial in range(N_trials) :

                print('block, trial = ', block, trial)

                trial_data = trial + N_trials*block
                data_x = data[trial_data]['x']
                trackertime = data[trial_data]['trackertime']

                StimulusOn = data[trial_data]['events']['msg'][10][0]
                StimulusOf = data[trial_data]['events']['msg'][14][0]
                TargetOn = data[trial_data]['events']['msg'][15][0]
                TargetOff = data[trial_data]['events']['msg'][16][0]
                saccades = data[trial_data]['events']['Esac']
                bino=binomial[trial, block]

                trackertime_0 = data[trial_data]['trackertime'][0]

                velocity = ANEMO.velocity_deg(data_x, px_per_deg)
                velocity_NAN = ANEMO.suppression_saccades(velocity, saccades, trackertime)

                start = TargetOn

                StimulusOn_s = StimulusOn - start
                StimulusOf_s = StimulusOf - start
                TargetOn_s = TargetOn - start
                TargetOff_s = TargetOff - start
                trackertime_s = trackertime - start

                ##################################################
                # FIT
                ##################################################
                result_deg = ANEMO.Fit_exponentiel(velocity_NAN, trackertime, TargetOn, StimulusOf, saccades, bino)
                ##################################################


                debut  = TargetOn - trackertime_0 # TargetOn - temps_0

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
                block_mean.append(np.nanmean(velocity_NAN[debut-50:debut+50]))


                if plot is not None :
                    axs[trial].cla() # pour remettre ax figure a zero
                    axs[trial].axis([StimulusOn_s-10, TargetOff_s+10, -40, 40])
                    axs[trial].xaxis.set_ticks(range(StimulusOf_s-199, TargetOff_s+10, 500))

                    axs[trial].plot(trackertime_s, velocity_NAN, color='k', alpha=0.6)
                    axs[trial].plot(trackertime_s[:-280], result_deg.init_fit, 'r--', linewidth=2)
                    axs[trial].plot(trackertime_s[:-280], result_deg.best_fit, color='r', linewidth=2)
                    axs[trial].plot(trackertime_s, np.ones(np.shape(trackertime_s)[0])*(bino*2-1)*(15), color='k', linewidth=0.2, alpha=0.2)
                    axs[trial].plot(trackertime_s, np.ones(np.shape(trackertime_s)[0])*(bino*2-1)*(10), color='k', linewidth=0.2, alpha=0.2)
                    axs[trial].axvspan(StimulusOn_s, StimulusOf_s, color='k', alpha=0.2)
                    axs[trial].axvspan(StimulusOf_s, TargetOn_s, color='r', alpha=0.2)
                    axs[trial].axvspan(TargetOn_s, TargetOff_s, color='k', alpha=0.15)
                    for s in range(len(saccades)) :
                        axs[trial].axvspan(saccades[s][0]-start, saccades[s][1]-start, color='k', alpha=0.2)

                    axs[trial].bar(latence, 80, bottom=-40, color='r', width=6, linewidth=0)

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
            liste_mean.append(block_mean)

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
        if observer is not None :
            param['observer'] = observer
        param['fit'] = liste_fit
        param['start_anti'] = liste_start_anti
        param['v_anti'] = liste_liste_v_anti
        param['latence'] = liste_latence
        param['tau'] = liste_tau
        param['maxi'] = liste_maxi
        param['moyenne'] = liste_mean

        return param



    ######################################################################################

    def figure(ax, velocity, saccades, StimulusOn, StimulusOf, TargetOn, TargetOff, trackertime, start, bino, plot, t_label, report=None) :
        '''
        Returns figure

        Parameters
        ----------
        ax : AxesSubplot
            ax sur lequel le figure doit être afficher

        velocity : ndarray
            vitesse de l'œuil en deg/sec

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

        plot : str
            si 'fonction' n'affiche que la fonction exponentiel
            si 'velocity' n'affiche que la vitesse de l'œuil
            si 'Fitvelocity' affiche la vitesse œuil + fit

        t_label : int
            taille x et y label

        report : NoneType or bool
            si != None renvoie le rapport fit


        Returns
        -------
        ax : AxesSubplot
            figure
        report : str
            fit report
        '''

        trackertime_0 = trackertime[0]
        StimulusOn_s = StimulusOn - start
        StimulusOf_s = StimulusOf - start
        TargetOn_s = TargetOn - start
        TargetOff_s = TargetOff - start
        trackertime_s = trackertime - start

        if plot != 'fonction' :
            ax.plot(trackertime_s, velocity, color='k', alpha=0.4)

            # Saccade
            for s in range(len(saccades)) :
                ax.axvspan(saccades[s][0]-start, saccades[s][1]-start, color='k', alpha=0.15)

        if plot != 'velocity' :
            # FIT
            result_deg = ANEMO.Fit_exponentiel(velocity, trackertime, TargetOn, StimulusOf, saccades, bino, sup=False)

        if plot == 'Fitvelocity' :

            debut  = TargetOn - trackertime_0 # TargetOn - temps_0
            start_anti = result_deg.values['start_anti']
            v_anti = result_deg.values['v_anti']
            latence = result_deg.values['latence']
            tau = result_deg.values['tau']
            maxi = result_deg.values['maxi']
            result_fit = result_deg.best_fit

        if plot == 'fonction' :

            start_anti = TargetOn-trackertime_0-100
            latence = TargetOn-trackertime_0+100
            result_fit = result_deg.init_fit

        ax.axvspan(StimulusOn_s, StimulusOf_s, color='k', alpha=0.2)
        ax.axvspan(StimulusOf_s, TargetOn_s, color='r', alpha=0.2)
        ax.axvspan(TargetOn_s, TargetOff_s, color='k', alpha=0.15)

        if plot != 'velocity' :
            # COSMETIQUE
            ax.plot(trackertime_s[:int(start_anti)], result_fit[:int(start_anti)], 'k', linewidth=2)
            ax.plot(trackertime_s[int(latence)+250:], result_fit[int(latence)+250:], 'k', linewidth=2)
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
            ax.plot(trackertime_s[int(latence):], np.ones(len(trackertime_s[int(latence):]))*result_fit[int(latence)], '--k', linewidth=1, alpha=0.5)
            ax.plot(trackertime_s[int(latence):], np.ones(len(trackertime_s[int(latence):]))*result_fit[int(latence)+250], '--k', linewidth=1, alpha=0.5)


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
                    "Max = %0.2f °/s"%(-maxi), color='k', va='center', fontsize=t_label/1.5)
            ax.annotate('', xy=(TargetOn_s+450, result_fit[int(latence)]), xycoords='data', fontsize=t_label/1.5,
                        xytext=(TargetOn_s+450, result_fit[int(latence)+250]), textcoords='data', arrowprops=dict(arrowstyle="<->"))


        if plot == 'fonction' :

            # COSMETIQUE
            ax.text(StimulusOf_s+(TargetOn_s-StimulusOf_s)/2, 31, "GAP", color='k', fontsize=t_label, ha='center', va='bottom')
            ax.text((StimulusOf_s-750)/2, 31, "FIXATION", color='k', fontsize=t_label, ha='center', va='bottom')
            ax.text((750-TargetOn_s)/2, 31, "PURSUIT", color='k', fontsize=t_label, ha='center', va='bottom')
            ax.text(TargetOn_s, 15, "Anticipation", color='r', fontsize=t_label/1.5, ha='center')

            # V_a ------------------------------------------------------------------------
            ax.text(TargetOn_s-50, -5, r"A$_a$", color='r', fontsize=t_label/1.5, ha='center', va='top')
            # Start_a --------------------------------------------------------------------
            ax.text(TargetOn_s-100-25, -35, "Start anticipation", color='k', fontsize=t_label/1.5, alpha=0.7, ha='right')
            # latence --------------------------------------------------------------------
            ax.text(TargetOn_s+99+25, -35, "Latency", color='firebrick', fontsize=t_label/1.5)
            # tau ------------------------------------------------------------------------
            ax.annotate(r'$\tau$', xy=(TargetOn_s+140, result_fit[TargetOn-trackertime_0+140]), xycoords='data', fontsize=t_label/1., color='darkred', va='bottom',
                    xytext=(TargetOn_s+170, result_fit[TargetOn-trackertime_0]), textcoords='data', arrowprops=dict(arrowstyle="->", color='darkred'))
            # Max ------------------------------------------------------------------------
            ax.text(TargetOn_s+400+25, ((result_fit[TargetOn-trackertime_0+100]+result_fit[TargetOn-trackertime_0+250])/2),
                   'Max', color='k', fontsize=t_label/1.5, va='center')
            ax.annotate('', xy=(TargetOn_s+400, result_fit[TargetOn-trackertime_0+100]), xycoords='data', fontsize=t_label/1.5,
                    xytext=(TargetOn_s+400, result_fit[TargetOn-trackertime_0+250]), textcoords='data', arrowprops=dict(arrowstyle="<->"))


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

    def figure_position(ax, data_x, data_y,  saccades, StimulusOn, StimulusOf, TargetOn, TargetOff, trackertime, start, bino, V_X, RashBass, stim_tau, screen_width_px, screen_height_px, t_label) :
        '''
        Returns figure de la position de l'œuil pendant l'enregistrement pout un essai

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
        trackertime_0 = trackertime_s[0]

        #------------------------------------------------
        # TARGET
        #------------------------------------------------
        dir_bool = bino*2 - 1
        Target_trial = []
        x = screen_width_px/2

        for t in range(len(trackertime_s)):
            if t < (TargetOn_s-trackertime_0) :
                x = screen_width_px/2
            elif t == (TargetOn_s-trackertime_0) :
                # la cible à t=0 recule de sa vitesse * latence=RashBass (ici mis en ms)
                x = x -(dir_bool * ((V_X/1000)*RashBass))
            elif (t > (TargetOn_s-trackertime_0) and t <= ((TargetOn_s-trackertime_0)+stim_tau*1000)) :
                x = x + (dir_bool*(V_X/1000))
            else :
                x = x
            Target_trial.append(x)
        #------------------------------------------------

        ax.plot(trackertime_s, np.ones(len(trackertime_s))*(screen_height_px/2), color='grey', linewidth=1.5)
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

        ax.yaxis.set_ticks(range(0, screen_width_px+1, int(screen_width_px/2)))
        ax.yaxis.set_ticklabels(range(0, screen_width_px+1, int(screen_width_px/2)), fontsize=t_label/2)

        ax.xaxis.set_ticks(range(TargetOn_s-1100, TargetOff_s, 100))
        ax.xaxis.set_ticklabels(range(TargetOn_s-1100, TargetOff_s, 100), fontsize=t_label/2)

        return ax


    ######################################################################################
    def plot_position(data, N_trials, N_blocks, bino, V_X, RashBass=100, stim_tau=1.5, screen_width_px=1280, screen_height_px=1024, fig=None, axs=None, fig_width=10, t_label=20, file_fig=None) :

        '''
        Returns figure de la position de l'œuil pendant l'enregistrement pour tous les essais

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

        screen_width_px : int
            widht ecran en pixel
        screen_height_px : int
            height écran en pixel

        fig : NoneType ou matplotlib.figure.Figure
            figure ou sera afficher les position de l'œuil
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

        for block in range(N_blocks) :

            if fig is None:
                fig, axs = plt.subplots(N_trials, 1, figsize=(fig_width, (fig_width*(N_trials/2))/1.6180))

            for trial in range(N_trials) :

                trial_data = trial + N_trials*block

                data_x = data[trial_data]['x']
                data_y = data[trial_data]['y']
                trackertime = data[trial_data]['trackertime']

                StimulusOn = data[trial_data]['events']['msg'][10][0]
                StimulusOf = data[trial_data]['events']['msg'][14][0]
                TargetOn = data[trial_data]['events']['msg'][15][0]
                TargetOff = data[trial_data]['events']['msg'][16][0]
                saccades = data[trial_data]['events']['Esac']
                bino_trial = bino[trial, block]
                start = TargetOn

                axs[trial] = ANEMO.figure_position(axs[trial], data_x, data_y,  saccades, StimulusOn, StimulusOf, TargetOn, TargetOff, trackertime, start, bino_trial, V_X, RashBass, stim_tau, screen_width_px, screen_height_px, t_label)

                axs[trial].axis([StimulusOf-start-30, TargetOff-start+30, -30, 1280+30])
                axs[trial].set_xlabel('Time (ms)', fontsize=t_label)
                axs[trial].set_ylabel(trial+1, fontsize=t_label)


            plt.tight_layout() # pour supprimer les marge trop grande
            plt.subplots_adjust(hspace=0) # pour enlever espace entre les figures

            if file_fig is None :
                file_fig = 'enregistrement'
            plt.savefig('%s_%s.pdf'%(file_fig, block+1))
        plt.close()


    def plot_velocity(data, trials=0, block=0,  N_trials=200, px_per_deg=36.51, fig_width=15, t_titre=35, t_label=20):
        '''
        Renvoie les figures de la vitesse de l'œuil

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
        px_per_deg : float
            nb de pixel par degres

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

        if type(trials) is not list :
            trials = [trials]

        fig, axs = plt.subplots(len(trials), 1, figsize=(fig_width, (fig_width*(len(trials)/2)/1.6180)))

        x = 0
        for t in trials :

            trial_data = t + N_trials*block

            if len(trials)==1:
                ax = axs
            else :
                ax = axs[x]


            data_x = data[trial_data]['x']
            trackertime = data[trial_data]['trackertime']

            StimulusOn = data[trial_data]['events']['msg'][10][0]
            StimulusOf = data[trial_data]['events']['msg'][14][0]
            TargetOn = data[trial_data]['events']['msg'][15][0]
            TargetOff = data[trial_data]['events']['msg'][16][0]
            saccades = data[trial_data]['events']['Esac']

            velocity = ANEMO.velocity_deg(data_x, px_per_deg)
            velocity_NAN = ANEMO.suppression_saccades(velocity, saccades, trackertime)

            start = TargetOn

            ax = ANEMO.figure(ax, velocity_NAN, saccades, StimulusOn, StimulusOf, TargetOn, TargetOff, trackertime, start, 0, 'velocity', t_label)

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


    def plot_Fit(data, bino, trials=0, block=0, N_trials=200, px_per_deg=36.51, plot='fonction', fig_width=15, t_titre=35, t_label=20, report=None):

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
        px_per_deg : float
            nb de pixel par degres

        plot : str
            si 'fonction' n'affiche que la fonction exponentiel
            si 'Fitvelocity' affiche la vitesse œuil + fit

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

        fig, axs = plt.subplots(len(trials), 1, figsize=(fig_width, (fig_width*(len(trials)/2)/1.6180)))

        results = []
        x = 0
        for t in trials :

            trial_data = t + N_trials*block
            bino_trial = bino[t, block]

            if len(trials)==1:
                ax = axs
            else :
                ax = axs[x]

            data_x = data[trial_data]['x']
            trackertime = data[trial_data]['trackertime']

            StimulusOn = data[trial_data]['events']['msg'][10][0]
            StimulusOf = data[trial_data]['events']['msg'][14][0]
            TargetOn = data[trial_data]['events']['msg'][15][0]
            TargetOff = data[trial_data]['events']['msg'][16][0]
            saccades = data[trial_data]['events']['Esac']

            velocity = ANEMO.velocity_deg(data_x, px_per_deg)
            velocity_NAN = ANEMO.suppression_saccades(velocity, saccades, trackertime)

            start = TargetOn

            if report is None :
                ax = ANEMO.figure(ax, velocity_NAN, saccades, StimulusOn, StimulusOf, TargetOn, TargetOff, trackertime, start, bino_trial, plot, t_label, report)
            else :
                ax, result = ANEMO.figure(ax, velocity_NAN, saccades, StimulusOn, StimulusOf, TargetOn, TargetOff, trackertime, start, bino_trial, plot, t_label, report)
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