#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ANEMO import ANEMO
from ANEMO.edfreader import read_edf
import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit import  Model, Parameters

import pickle
with open('param_synthetic.pkl', 'rb') as fichier :
    param = pickle.load(fichier, encoding='latin1')
with open('name_data_file.pkl', 'rb') as fichier :
    exp = pickle.load(fichier, encoding='latin1')


datafile = 'name_data_file.asc'
data = read_edf(datafile, 'TRIALID')


A = ANEMO(exp)
Fit = ANEMO.Fit(exp)



def Comparison_trial(data=data, trial=10, block=0, plot=True, nb_plot=10, fig_width=15, t_titre=35, t_label=20, simulate=False) :


    fig1, axs1 = plt.subplots(1, 1, figsize=(fig_width, 1*(fig_width*1/2)/1.6180))
    fig, axs = plt.subplots(1, 1, figsize=(fig_width, 1*(fig_width*1/2)/1.6180))

    if simulate is False :

        trial_data = trial + exp['N_trials']*block
        arg = A.arg(data[trial_data], trial=trial, block=block)

        trackertime, saccades, dir_target = arg.trackertime, arg.saccades, arg.dir_target
        StimulusOn, StimulusOf = arg.StimulusOn, arg.StimulusOf
        TargetOn,   TargetOff  = arg.TargetOn,   arg.TargetOff

        onset  = arg.TargetOn - arg.t_0
        velocity = A.velocity_NAN(**arg)


    else :
        trackertime = np.arange(0, 1500, 1)

        StimulusOn, StimulusOf = 0, 750-300
        TargetOn,   TargetOff = 750, 1499

        onset  = TargetOn
        saccades = []

        dir_target = int(exp['p'][trial][0][0]*2 - 1)
        start_anti_true = param['start_anti'][0][trial]+onset
        a_anti_true     = param['a_anti'][0][trial]
        latency_true    = param['latency'][0][trial]+onset
        tau_true        = param['tau'][0][trial]
        maxi_true       = param['maxi'][0][trial]

        true_test = A.Equation.fct_velocity(trackertime, dir_target, start_anti_true, a_anti_true,
                                            latency_true, tau_true, maxi_true, False)

        velocity = np.copy(true_test)
        np.random.seed(7)
        bruit=10
        velocity += np.random.rand(len(trackertime))*bruit
        velocity -= np.random.rand(len(trackertime))*bruit

    start = TargetOn
    time_s = trackertime - start
    StimulusOf_s, TargetOn_s = StimulusOf-start, TargetOn-start
    #-------------------------------------------------


    #-------------------------------------------------
    # FIT
    #-------------------------------------------------

    list_old_lat, old_max, old_a_anti = ANEMO.classical_method.Full(velocity, onset, full_return=True, time=time_s)
    old_latency, slope1, intercept1, slope2, intercept2, t = list_old_lat

    f = Fit.Fit_trial(velocity, equation='fct_velocity', value_latency=old_latency+onset,
                      value_maxi=old_max, value_anti=old_a_anti, dir_target=dir_target,
                      TargetOn=TargetOn, StimulusOf=StimulusOf, saccades=saccades,
                      trackertime=trackertime)#, **arg)


    start_anti = f.values['start_anti']-onset
    a_anti = f.values['a_anti']
    latency = f.values['latency']-onset
    tau = f.values['tau']
    steady_state = f.values['steady_state']*dir_target

    best_fit = A.Equation.fct_velocity(time_s[:-250], dir_target, start_anti, a_anti, latency, tau, steady_state, False)
    init_fit = A.Equation.fct_velocity(time_s[:-250], dir_target, -100, old_a_anti, old_latency, 15, old_max, False)
    #-------------------------------------------------

    #-----------------------------------------------------------------------------------------------------------
    # vieux paramêtres
    #-----------------------------------------------------------------------------------------------------------
    old_a_anti_fit = np.nanmean(best_fit[int(start_anti)+onset:int(latency)+onset])
    #-------------------------------------------------

    for a in [axs, axs1] :

        a = A.Plot.deco(exp, a, t_label=t_label, StimulusOn=StimulusOn, StimulusOf=StimulusOf,
                        TargetOn=TargetOn, TargetOff=TargetOff, saccades=saccades)
        a.set_ylabel('velocity (°/s)', fontsize=t_label)
        a.plot(time_s, velocity, color='k', alpha=0.3, lw=1)

        opt = dict(color='k', fontsize=t_label*1.5, ha='center', va='center', alpha=0.5)
        a.text(StimulusOf_s+(TargetOn_s-StimulusOf_s)/2, 31, "GAP", **opt)
        a.text((StimulusOf_s-750)/2, 31, "FIXATION", **opt)
        a.text((750-TargetOn_s)/2, 31, "PURSUIT", **opt)

        a.plot(time_s, np.zeros(len(time_s)), '--k', linewidth=1, alpha=0.5)

        if simulate is False :
            a.set_title('True trial', fontsize=t_titre, x=0.5, y=1.05)
            a.axis([arg.StimulusOn-start - 10, arg.TargetOff-start + 10, -40, 40])
        else :
            a.set_title('Simulated trial', fontsize=t_titre, x=0.5, y=1.05)
            a.axis([-750, 750, -39.5, 39.5])


    #------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------
    axs.plot(time_s[:-250], best_fit, color='k', linewidth=2)
    axs.plot(time_s[:-250], init_fit, '--k', linewidth=1)


    # V_a ------------------------------------------------------------------------
    for aa, a, c, x_, y_ in zip([a_anti, old_a_anti], [axs, axs1], ['r', 'k'],
                                [(start_anti+latency)/2, TargetOn_s], [15, 20]) :
        a.text(x_, y_, r"Anticipation", color=c, fontsize=t_label, ha='center', va='center')

        a.text(x_, -25 if simulate else -20,
               "A$_a$ = %0.2f °/s$^2$ \n(initially %0.2f °/s$^2$)"%(aa, a_anti_true) if simulate
               else "A$_a$ = %0.2f °/s$^2$"%(aa),
               color=c, fontsize=t_label/1.5, ha='center',
               va='top' if simulate else 'center')

    t_start_anti = int(start_anti+onset)
    t_latency = int(latency+onset)


    axs.plot(time_s[t_start_anti:t_latency], best_fit[t_start_anti:t_latency],c='r', linewidth=2)
    axs.annotate('', xy=(latency, best_fit[t_latency]-3), xycoords='data', fontsize=t_label/1.5,
                 xytext=(start_anti, best_fit[t_start_anti]-3), textcoords='data',
                 arrowprops=dict(arrowstyle="->", color='r'))

    axs1.text(TargetOn_s, 15, "mean(t$_{-50}$,t$_{50}$)*100/1000", color='k', fontsize=t_label/1.5,
              ha='center', va='top')
    axs1.axvspan(TargetOn_s-50, TargetOn_s+50, color='c', alpha=0.3)


    # Start_a --------------------------------------------------------------------
    axs.text(start_anti-25, -20 if simulate else -30 ,
             "Anticipation Onset", color='k', alpha=0.7, fontsize=t_label,
             ha='right', va='center' if simulate else 'bottom')

    axs.bar(start_anti, 80, bottom=-40, color='k', width=4, linewidth=0, alpha=0.7)

    axs.text(start_anti-25, -35,
             "%0.2f ms \n(initially %0.2f ms)"%(start_anti, start_anti_true-onset) if simulate
             else "%0.2f ms"%(start_anti),
             color='k', alpha=0.7, fontsize=t_label/1.5, ha='right',
             va='bottom' if simulate else 'center')

    # latency --------------------------------------------------------------------

    for l, a, c in zip([latency, old_latency], [axs, axs1], ['firebrick', 'k']) :
        a.text(l+25, -20 if simulate else -30, "Latency", color=c, fontsize=t_label,
               va='center' if simulate else 'bottom')
        a.bar(l, 80, bottom=-40, color=c if l==latency else 'c', width=4, linewidth=0, alpha=1)
        a.text(l+25, -35,
               "%0.2f ms \n(initially %0.2f ms)"%(l, latency_true-onset) if simulate
               else "%0.2f ms"%(l),
               color=c, fontsize=t_label/1.5,
               va='bottom' if simulate else 'center')


    w1, w2, off, crit = 300, 50, 50, 0.2
    tw = time_s[t:t+w1+off+w2] #[x1:x1+w1]
    timew = np.linspace(np.min(tw), np.max(tw), len(tw))

    fitLine1 = slope1 * timew + intercept1
    axs1.plot(tw[:w1+off+50], fitLine1[:w1+off+50], '--k', linewidth=1.5)
    axs1.plot(tw[:w1], fitLine1[:w1], c='k', linewidth=1.5)

    fitLine2 = slope2 * timew + intercept2
    axs1.plot(tw[w1:], fitLine2[w1:], '--k', linewidth=1.5)
    axs1.plot(tw[w1+off:], fitLine2[w1+off:], c='k', linewidth=1.5)

    # tau ------------------------------------------------------------------------
    axs.plot(time_s[t_latency:t_latency+250], best_fit[t_latency:t_latency+250], c='darkred', linewidth=2)
    axs.annotate(r'$\tau$', xy=(latency+29, best_fit[t_latency+29]), xycoords='data', fontsize=t_label,
                 color='darkred', va='bottom', xytext=(latency+70, best_fit[t_latency-15]), textcoords='data',
                 arrowprops=dict(arrowstyle="->", color='darkred'))
    axs.text(latency+70+t_label, (best_fit[t_latency]),
             "  %0.2f ms \n(initially %0.2f ms)"%(tau, tau_true) if simulate
             else r"  %0.2f ms"%(tau),
             color='darkred', fontsize=t_label/1.5,
             va='center' if simulate else 'bottom')


    # Max ------------------------------------------------------------------------

    for m, a, t_lat, y1, y2 in zip([steady_state, old_max], [axs, axs1], [t_latency, int(old_latency+onset)],
                                   [steady_state+5, -20], [steady_state, old_max]) :

        a.text(TargetOn_s+500, y1, "Steady State", color='k', va='bottom' if m==steady_state else 'center',
               ha='center' if m==old_max else 'left', fontsize=t_label)
        a.plot(time_s[t_lat:], np.ones(len(time_s[t_lat:]))*y2, '--k', linewidth=1, alpha=0.5)

        a.annotate('', xy=(TargetOn_s+435, 0), xycoords='data', fontsize=t_label/1.5,
                   xytext=(TargetOn_s+435, y2), textcoords='data', arrowprops=dict(arrowstyle="<->"))

        a.text(TargetOn_s+455 if simulate else TargetOn_s+500, y2/2,
               "%0.2f °/s \n(initially %0.2f °/s)"%(m, maxi_true) if simulate
               else "%0.2f °/s"%(m),
               color='k', fontsize=t_label/1.5,
               va='center', ha='left' if simulate else 'center')

    axs1.text(TargetOn_s+500, -25, "mean(t$_{400}$,t$_{600}$)", color='k', ha='center', va='top', fontsize=t_label/1.5)
    axs1.axvspan(TargetOn_s+400, TargetOn_s+600, color='c', alpha=0.3)


    #------------------------------------------------------------------------------------------
    fig.tight_layout()
    fig1.tight_layout()

    plt.show()


def fit_simulate_trial(N_trials=200):

    np.random.seed(7)

    x = np.arange(0, 1500, 1)
    TargetOn, StimulusOf = 750, 750-300

    result = {'true':{}, 'fit':{}, 'old':{}}
    for v in ['start_anti', 'a_anti', 'latency', 'tau', 'steady_state'] :
        result['true'][v], result['fit'][v] = [], []
        if v in ['latency', 'steady_state', 'a_anti'] : result['old'][v] = []

    for trial in range(N_trials):

        block=1
        #print(trial, end=' ')

        dir_target = exp['p'][trial][block][0]*2 - 1


        for v in ['start_anti', 'a_anti', 'latency', 'tau', 'steady_state'] :
            if v=='steady_state' : v1='maxi'
            else :                 v1 = v
            result['true'][v].append(param[v1][block][trial])


        #------------------------------------------------------------------------------------
        true_test = A.Equation.fct_velocity(x, dir_target,
                                            result['true']['start_anti'][-1]+TargetOn,
                                            result['true']['a_anti'][-1],
                                            result['true']['latency'][-1]+TargetOn,
                                            result['true']['tau'][-1],
                                            result['true']['steady_state'][-1], False)


        np.random.seed(7)
        bruit=10

        test  = np.copy(true_test)
        test += np.random.rand(len(x))*bruit
        test -= np.random.rand(len(x))*bruit

        #if trial==10 : plt.plot(test) ; plt.show()
        #------------------------------------------------------------------------------------
        old_lat, old_max, old_a_anti = ANEMO.classical_method.Full(test, TargetOn)

        if np.isnan(old_lat)==False : result['old']['latency'].append(old_lat-TargetOn)
        else :                        result['old']['latency'].append(old_lat)

        result['old']['steady_state'].append(old_max)
        result['old']['a_anti'].append(old_a_anti)

        #------------------------------------------------------------------------------------
        f = Fit.Fit_trial(test, equation='fct_velocity',
                          value_latency=old_lat, value_maxi=old_max, value_anti=old_a_anti,
                          dir_target=dir_target, TargetOn=TargetOn, StimulusOf=StimulusOf,
                          saccades=[], trackertime=x)

        for v in ['start_anti', 'a_anti', 'latency', 'tau', 'steady_state'] :
            if v in ['start_anti', 'latency'] : var = f.values[v]-TargetOn
            else :                              var = f.values[v]
            result['fit'][v].append(var)


    return result



def Comparison_scatter(result,fig_width=15/3, t_titre=35/2, t_label=20/2) :


    fig, axs = plt.subplots(3, 3, figsize=(4*3, (4*3)/1.6180))

    def regress(ax, p, data, minx, miny, maxx, maxy, t_label=12) :
        from scipy import stats
        slope, intercept, r_, p_value, std_err = stats.linregress(p, data)
        x_test = np.linspace(np.min(p), np.max(p), 100)
        fitLine = slope * x_test + intercept
        ax.plot(x_test, fitLine, c='k', linewidth=2)
        ax.text(maxx-((maxx-minx)/16.180),miny+((maxy-miny)/10), 'r = %0.3f'%(r_), fontsize=t_label/1.2, ha='right')

        return ax


    def scatter(ax, title, label, y_label, c, x, y,  min_x, min_y, max_x, max_y, fig_width=15/3, t_titre=35/2, t_label=20/2) :

        #fig, ax = plt.subplots(1, 1, figsize=(fig_width*1, (fig_width*1)/1.6180))
        ax.set_title(title, fontsize=t_titre, x=0.5, y=1.05)

        ax.scatter(x, y, c=c, alpha=0.5)
        ax = regress(ax, x, y, min_x, min_y, max_x, max_y, t_label=t_titre/1.5)

        ax.plot([-2000, 2000], [-2000, 2000], '--k', alpha=0.5)

        ax.set_xlabel('True %s'%label, fontsize=t_label)
        ax.set_ylabel('%s %s'%(y_label, label), fontsize=t_label)
        ax.axis([min_x-((max_x-min_x)/10), max_x+((max_x-min_x)/10),
                 min_y-((max_y-min_y)/10), max_y+((max_y-min_y)/10)])



    #-----------------------------------------------------------------------------------

    for var, title, label, num_col in zip(['latency', 'a_anti', 'steady_state', 'start_anti', 'start_anti', 'tau'],
                                         ['Latency of the pursuit', 'Anticipatory acceleration', 'Steady-state velocity','Anticipation onset', 'Anticipatory onset for A$_a$ > 3°/s$^2$', r'$\tau$'],
                                         ['Latency (ms)', 'A$_a$ (°/s$^2$)', 'Steady-state (°/s)', 'Anticipation onset (ms)', 'Anticipatory onset (ms)', r'$\tau$ (ms)'],
                                         [0, 1, 2, 0, 1, 2]):


        if var in ['latency', 'a_anti', 'steady_state'] :
            x1, y1, y2 = result['true'][var], result['fit'][var], result['old'][var]

            if var == 'latency' :
                x2 = np.copy(x1)
                y2 = np.copy(y2)
                x2 = x2[~np.isnan(result['old'][var])]
                y2 = y2[~np.isnan(result['old'][var])]

            elif var == 'a_anti' :
                x2 = np.copy(x1)

            elif var == 'steady_state' :
                del y1[np.argmin(x1)]
                del y2[np.argmin(x1)]
                del x1[np.argmin(x1)]
                x2 = np.copy(x1)

            min_x, min_y = min(min(x1), min(x2)), min(min(y1), min(y2))
            max_x, max_y = max(max(x1), max(x2)), max(max(y1), max(y2))

            for x, y, c, y_label, num_line in zip([x1, x2], [y1, y2], ['r', 'c'], ['Fit', 'Classical'], [1, 0]) :
                scatter(axs[num_line][num_col], title, label, y_label, c, x, y,  min_x, min_y, max_x, max_y)

        else :
            x, y = result['true']['%s'%var], result['fit']['%s'%var]

            if title=='Anticipatory onset for A$_a$ > 3°/s$^2$' :
                new_x, new_y = [], []
                for s in range(len(y)) :
                    if abs(result['true']['a_anti'][s]) > 3 :
                        if abs(result['fit']['a_anti'][s]) > 3 :
                            new_x.append(x[s]) ; new_y.append(y[s])
                x, y = new_x, new_y

            scatter(axs[2][num_col], title, label, 'Fit', 'r', x, y,  min(x), min(y), max(x), max(y))

    fig.tight_layout()
    plt.show()
