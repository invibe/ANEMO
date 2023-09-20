#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import ANEMO

from ANEMO.Model import *
from ANEMO.Init import *


t_titre, t_label  = 25, 16
cf, ca, cp = 'darkred', 'darkorange', 'darkgreen'
time = np.arange(1200)
SOff = 200 # StimulusOff
TOn = SOff+300 # TargetOn
DATA = Data.open('../dataANEMO')

def show_fct_pursuit(ax, fct_pursuit, time, sa, sp, dymin, dymax, cf, ca, cp,
                     t_label):

    ax.plot(time[:sa],   fct_pursuit[:sa],   c=cf, lw=2)
    ax.plot(time[sa:sp], fct_pursuit[sa:sp], c=ca, lw=2)
    ax.plot(time[sp:],   fct_pursuit[sp:],   c=cp, lw=2)

    arg = dict(y=dymax-(dymax-dymin)/5, alpha=.6, size=t_label*1.1,
               ha='center', va='bottom', weight='bold')
    ax.text((sa-time[0])/2,     s="FIXATION",     color=cf, **arg)
    ax.text(sa+(sp-sa)/2,       s="ANTICIPATION", color=ca, **arg)
    ax.text(sp+(time[-1]-sp)/2, s="PURSUIT",      color=cp, **arg)

    arg = dict(height=abs(dymin)+abs(dymax), bottom=dymin, lw=0, align='edge')
    ax.bar(time[0], width=sa,          color=cf, alpha=0.2, **arg)
    ax.bar(sa,      width=sp-sa,       color=ca, alpha=0.2, **arg)
    ax.bar(sp,      width=time[-1]-sp, color=cp, alpha=0.2, **arg)

    ax.vlines(sa, ymin=dymin, ymax=dymax, color=ca)
    ax.vlines(sp, ymin=dymin, ymax=dymax, color=cp)

    ax.plot(time, np.zeros(len(time)), '--k', lw=1, alpha=0.5)

    return ax

def show_param(ax, s1, s2, xt, yt, dyt, axy, axyt, c, t_label, t_labelp=None,
               ha='left', arrow='->', xtp=None, ytp=None, hap=None):
    if not hap: hap=ha
    if not xtp: xtp=xt
    if not ytp: ytp=yt-(2*dyt/3)
    if not t_labelp: t_labelp=t_label/1.5


    ax.text(xt,  yt,  s1, color=c, ha=ha,  va='center', size=t_label,
            weight='bold')
    ax.text(xtp, ytp, s2, color=c, ha=hap, va='center', size=t_labelp,
            family='monospace')
    ax.annotate('', xy=axy, xytext=axyt, color=c,
                arrowprops=dict(arrowstyle=arrow, color=c), xycoords='data',
                textcoords='data', weight='bold', size=t_label, va='center')
    return ax


def show_exp(ax, SOff, TOn, time, aymin, aymax, dymin, dymax, t_label,
             plot='Model') :
    arg_text_exp = dict(y=aymax-(aymax-aymin)/20, alpha=.4, size=t_label,
                        ha='center', va='top', weight='bold')
    ax.text(SOff/2, s="Fixation point", color='k', **arg_text_exp)
    ax.text(SOff+(TOn-SOff)/2, s="GAP", color='k', **arg_text_exp)
    ax.text(TOn+(time[-1]-TOn)/2, s="Target in motion", color='k',
            **arg_text_exp)

    ax.plot(time[:SOff], np.ones(len(time[:SOff]))*(dymax+3), 'k', lw=1,
            alpha=.8)
    ax.plot(time[TOn:], np.ones(len(time[TOn:]))*(dymax+3), 'k', lw=1,
            alpha=.8)

    if plot=='Model':
        arg_bar_fct = dict(height=abs(dymin)+abs(dymax), bottom=dymin, lw=0,
                           align='edge')
        arg_bar_exp = dict(height=abs(aymax)-abs(dymax), bottom=dymax, lw=0,
                           align='edge')
        ax.bar(time[0], width=SOff, color='k', alpha=0.1, **arg_bar_exp)
        ax.bar(SOff, width=TOn-SOff, color='k', alpha=0.2, **arg_bar_exp)
        ax.bar(SOff, width=TOn-SOff, color='k', alpha=0.1, **arg_bar_fct)
        ax.bar(TOn, width=time[-1]-TOn, color='k', alpha=0.1, **arg_bar_exp)
        ax.vlines(TOn, ymin=dymin, ymax=aymax, ls='--', lw=1, color='k',
                  alpha=.5)

    else:
        arg_bar_fct = dict(height=abs(dymin)+abs(aymax), bottom=dymin, lw=0,
                           align='edge')
        ax.bar(time[0], width=SOff, color='k', alpha=0.1, **arg_bar_fct)
        ax.bar(SOff, width=TOn-SOff, color='k', alpha=0.2, **arg_bar_fct)
        ax.bar(TOn,  width=time[-1]-TOn, color='k', alpha=0.1, **arg_bar_fct)
        ax.vlines(TOn, ymin=dymin, ymax=aymax, ls='-', lw=1, color='k',
                  alpha=1)

        xt, yt = TOn-35, dymin/1.7
        dyt = (dymax-dymin)/7
        axy, axyt = (TOn, yt), (xt, yt)
        ax = show_param(ax, 'The event marking the appearance of the target',
                        'eventName_TargetOn', xt, yt, dyt, axy, axyt, 'k',
                        t_label/1.3, t_label/1.5,  ha='right')

    ax.vlines(SOff, ymin=dymin, ymax=aymax, ls='--', lw=1,color='k', alpha=.5)

    return ax

def cosmetique(ax, dxmin, dxmax, dymin, dymax, axmin, axmax, aymin, aymax,
               t_label, step_yticks=10):

    ax.set_xlabel('Time (ms)', fontsize=t_label)
    ax.axis([axmin, axmax, aymin, aymax])
    ax.set_yticks(np.arange(dymin, dymax+(step_yticks/10), step_yticks))
    ax.spines.bottom.set_bounds(dxmin, dxmax)
    ax.spines.left.set_bounds(dymin, dymax)
    for a in ['right', 'top'] : ax.spines[a].set_visible(False)
    ax.tick_params(labelsize=t_label/2)

    return ax


def plot_pursuit(ax, fct, time, SOff, TOn, sa, l, ss, cf, ca, cp, t_label):

    dxmin, dxmax, dymin, dymax = np.min(time),    np.max(time),    -10, 25
    axmin, axmax, aymin, aymax = dxmin-dxmax/100, dxmax+dxmax/100, -12, 35

    ax = cosmetique(ax, dxmin, dxmax, dymin, dymax, axmin, axmax, aymin, aymax,
                    t_label)
    ###########################################################################
    # Model
    ###########################################################################
    ax = show_fct_pursuit(ax, fct, time, sa, l, dymin, dymax, cf, ca, cp,
                          t_label)

    #--------------------------------------------------------------------------
    # Parameters
    #--------------------------------------------------------------------------
    exf, eyf = 10, -2 # arrow gap
    dyt = (dymax-dymin)/7

    # Start_a -----------------------------------------------------------------
    xt, yt = sa-50, dymin+dyt
    axy, axyt = (sa-exf, yt), (xt+exf, yt)
    ax = show_param(ax, "Start anticipation", 'start_anti', xt, yt, dyt, axy,
                    axyt, ca, t_label, ha='right')
    # V_a ---------------------------------------------------------------------
    xt, yt = sa+(l-sa)/2, fct[l-sa]+eyf-(dyt/10)
    axy, axyt = (l-exf, fct[l]+eyf), (sa+exf, fct[sa]+eyf)
    ax = show_param(ax, r"A$_a$", 'a_anti', xt, yt, dyt, axy, axyt, ca,
                    t_label, ha='center')
    # latency -----------------------------------------------------------------
    xt, yt = l+50, dymin+dyt
    axy, axyt = (l+exf, yt), (xt-exf, yt)
    ax = show_param(ax, "Latency", 'latency', xt, yt, dyt, axy, axyt, cp,
                    t_label)
    # Steady State ------------------------------------------------------------
    xt, yt = l+400, (ss*2)/3
    axy, axyt = (xt-25, 0), (xt-25, ss)
    ax = show_param(ax, 'Steady State', 'steady_state', xt, yt, dyt, axy, axyt,
                    cp, t_label, arrow='<->')

    ###########################################################################
    # exp
    ###########################################################################
    ax = show_exp(ax, SOff, TOn, time, aymin, aymax, dymin, dymax, t_label)

    return ax, dyt

class Plot_classical_method:

    def latency(expname='sub-002_task-aSPEM', trial=8):

        time = np.arange(1200)
        dxmin, dxmax, dymin, dymax = np.min(time),    np.max(time),    -20, 38
        axmin, axmax, aymin, aymax = dxmin-dxmax/100, dxmax+dxmax/100, -22, 50


        results = DATA[expname].Results
        results_trial = results[results.trial==trial]

        xname = 'classical_latency'
        latency = results_trial[xname].values[0]

        if latency and not np.isnan(latency):

            fig, ax = plt.subplots(1, 1, figsize=(15, (15*(1/2)/1.6180)))
            ax.set_title('Latency', fontsize=t_titre, x=0.5, y=1.05)
            ax.set_ylabel('Velocity (°/s)', fontsize=t_label)
            ax = cosmetique(ax, dxmin, dxmax, dymin, dymax, axmin, axmax,
                            aymin, aymax, t_label)


            ###################################################################
            # DATA
            ###################################################################

            data = DATA[expname].Data
            trial_data = data[data.trial==trial]
            events = DATA[expname].Events
            trial_events = events[events.trial==trial]


            targeton = int(trial_events.TargetOn.values[0] \
                           - trial_data.time.values[0])
            d = trial_data[targeton-500:].vx_NaN
            ax.plot(d.values[:len(time)], 'k', alpha=.2)
            #------------------------------------------------------------------

            slope1 = results_trial[xname+'__slope1'].values[0]
            intercept1 = results_trial[xname+'__intercept1'].values[0]
            slope2 = results_trial[xname+'__slope2'].values[0]
            intercept2 = results_trial[xname+'__intercept2'].values[0]
            t0 = int(results_trial[xname+'__t0'].values[0])
            latency = int(latency - (targeton-500))

            w1, w2, off, crit = 300, 50, 50, 0.17

            new_t = np.arange(10000)
            tw = new_t[t0:t0+w1+off+w2]
            timew = np.linspace(np.min(tw), np.max(tw), len(tw))

            arg_b = dict(height=abs(dymin)+abs(aymax), bottom=dymin, lw=0,
                         align='edge')
            arg_param = dict(t_label=t_label/1.3, t_labelp=t_label/1.5)

            dyt = (dymax-dymin)/7

            # w1 --------------------------------------------------------------
            xt, yt = tw[0] - int(targeton-500) + (w1/2), dymax-(dymax-dymin)/15
            axy = (tw[0] - int(targeton-500), dymax)
            axyt = (tw[0] - int(targeton-500) + w1, dymax)
            ax = show_param(ax, r"size of the window 1", 'w1=300', xt, yt, dyt,
                            axy, axyt, 'k', ha='center', arrow='<->',
                            **arg_param)

            ax.bar(tw[0] - int(targeton-500), width=w1, color=ca, alpha=0.1,
                   **arg_b)
            ax.vlines(tw[0] - int(targeton-500),  ymin=dymin, ymax=aymax,
                      ls='--', lw=1, color=ca, alpha=.7)
            ax.vlines(tw[0] - int(targeton-500) + w1,  ymin=dymin, ymax=aymax,
                      ls='--', lw=1, color=ca, alpha=.7)

            # off -------------------------------------------------------------
            xt = tw[0] - int(targeton-500) + w1 + (off/2)
            yt = dymax-(dymax-dymin)/15 - 15
            axy = (tw[0] - int(targeton-500)+w1, dymax-15)
            axyt = (tw[0] - int(targeton-500) + w1 + off, dymax-15)
            ax = show_param(ax, 'gap', 'off=50', xt, yt, dyt, axy, axyt, 'k',
                            ha='center', arrow='<->', **arg_param)


            # w2 --------------------------------------------------------------
            xt = tw[w1+off] - int(targeton-500) + (w2/2)
            yt = dymax-(dymax-dymin)/15
            axy = (tw[w1+off] - int(targeton-500), dymax)
            axyt = (tw[w1+off] - int(targeton-500) + w2, dymax)
            ax = show_param(ax, 'size of the window 2', 'w2=50', xt, yt, dyt,
                            axy, axyt, 'k', ha='center', arrow='<->',
                            **arg_param)

            ax.bar(tw[w1+off] - int(targeton-500), width=w2, color=cp,
                   alpha=0.1, **arg_b)
            ax.vlines(tw[w1+off] - int(targeton-500),  ymin=dymin, ymax=aymax,
                      ls='--', lw=1,color=cp, alpha=.7)
            ax.vlines(tw[w1+off] - int(targeton-500) + w2,  ymin=dymin,
                      ymax=aymax, ls='--', lw=1,color=cp, alpha=.7)


            # PARAM
            arg_param = dict(t_label=t_label/1.5, t_labelp=t_label/1.5)

            # t_0 -------------------------------------------------------------
            ax.vlines(tw[0] - int(targeton-500), ymin=dymin, ymax=aymax,
                      ls='-', lw=1, color=ca, alpha=1)
            xt, yt = tw[0] - int(targeton-500) -20, dymax-(dymax-dymin)/4
            axy, axyt = (tw[0] - int(targeton-500), yt), (xt, yt)
            ax = show_param(ax, 'time 0', 't_0', xt, yt, dyt, axy, axyt, ca,
                            ha='right', **arg_param)

            # fitLine1 --------------------------------------------------------
            fitLine1 = slope1 * timew + intercept1
            ax.plot(tw[:w1+off+50]-int(targeton-500), fitLine1[:w1+off+50],
                    ls='--', c=ca, linewidth=1.5)
            ax.plot(tw[:w1]- int(targeton-500), fitLine1[:w1], c=ca,
                    linewidth=1.5)

            xt, yt = tw[int(w1/2.3)]-int(targeton-500), dymax-(dymax-dymin)/2.7
            axy = (tw[int(w1/2.3)]-int(targeton-500)+50,
                   fitLine1[int(w1/2.3)+50])
            axyt =  (xt, yt)
            ax = show_param(ax, 'regression line 1', 'slope1\nintercept1', xt,
                            yt, dyt, axy, axyt, ca, ha='right', **arg_param)

            # fitLine2 --------------------------------------------------------
            fitLine2 = slope2 * timew + intercept2
            ax.plot(tw[w1:]- int(targeton-500), fitLine2[w1:], ls='--', c=cp,
                    linewidth=1.5)
            ax.plot(tw[w1+off:]- int(targeton-500), fitLine2[w1+off:], c=cp,
                    linewidth=1.5)

            xt = tw[int(w1+off+w2)-1] + 25-int(targeton-500)
            yt = dymax-(dymax-dymin)/2.1
            axy = (tw[int(w1+off - w2/2.3)]-int(targeton-500)+50,
                   fitLine2[int(w1+off-w2/2.3)+50])
            axyt = (xt, yt)
            ax = show_param(ax, 'regression line 2', 'slope2\nintercept2', xt,
                            yt, dyt, axy, axyt, cp, ha='left', **arg_param)

            # latency ---------------------------------------------------------
            ax.vlines(latency,  ymin=dymin, ymax=aymax, ls='-', lw=1, color=cp,
                      alpha=1)
            xt, yt = latency+20, dymin+(dymax-dymin)/5
            axy, axyt = (latency, yt), (xt, yt)
            ax = show_param(ax, 'Latency of pursuit', 'latency', xt, yt, dyt,
                            axy, axyt, cp, ha='left', **arg_param)

            ###################################################################
            # exp
            ###################################################################
            ax = show_exp(ax, SOff, TOn, time, aymin, aymax, dymin, dymax,
                          t_label, plot='')

            plt.tight_layout()
            plt.savefig('classical_method_latency.svg')
            plt.show()


        else:
            print('pas latency')


    def steady_state(expname='sub-002_task-aSPEM', trial=8):

        time = np.arange(1200)
        dxmin, dxmax, dymin, dymax = np.min(time),    np.max(time),    -20, 35
        axmin, axmax, aymin, aymax = dxmin-dxmax/100, dxmax+dxmax/100, -22, 50

        fig, ax = plt.subplots(1, 1, figsize=(15, (15*(1/2)/1.6180)))
        ax.set_title('Steady State', fontsize=t_titre, x=0.5, y=1.05)
        ax.set_ylabel('Velocity (°/s)', fontsize=t_label)
        ax = cosmetique(ax, dxmin, dxmax, dymin, dymax, axmin, axmax, aymin,
                        aymax, t_label)


        #######################################################################
        # DATA
        #######################################################################
        data = DATA[expname].Data
        events = DATA[expname].Events

        data_trial = data[data.trial==trial]
        trial_events = events[events.trial==trial]
        targeton = int(trial_events.TargetOn.values[0] \
                       - data_trial.time.values[0])
        d = data_trial[targeton-500:].vx_NaN
        ax.plot(d.values[:len(time)], 'k', alpha=.2)
        #----------------------------------------------------------------------

        arg = dict(height=abs(dymin)+abs(aymax), bottom=dymin, lw=0,
                   align='edge')
        ax.bar(TOn+400, width=200, color=cp, alpha=0.2, **arg)
        ax.vlines(TOn+400, ymin=dymin, ymax=aymax, ls='--', lw=1,color=cp,
                  alpha=.7)
        ax.vlines(TOn+600, ymin=dymin, ymax=aymax, ls='--', lw=1,color=cp,
                  alpha=.7)


        arg_param = dict(t_label=t_label/1.3, t_labelp=t_label/1.5)

        dyt = (dymax-dymin)/7
        xt, yt = TOn+500, dymax-(dymax-dymin)/15
        axy, axyt = (TOn+400, dymax), (TOn+600, dymax)
        ax = show_param(ax, 'average velocity of the eye in this window',
                        'steady_state', xt, yt, dyt, axy, axyt, cp,
                        ha='center', arrow='<->', **arg_param)



        xt, yt = TOn+380, (dymax-dymin)/3.5
        axy, axyt = (TOn+400, yt), (xt, yt)
        ax = show_param(ax, 'add time at the start of the event',
                        'add_stime=400', xt, yt, dyt, axy, axyt, 'k',
                        ha='right', **arg_param)

        xt, yt = TOn+580, (dymax-dymin)/15
        axy, axyt = (TOn+600, yt), (xt, yt)
        ax = show_param(ax, 'add time at the end of the event',
                        'add_etime=600', xt, yt, dyt, axy, axyt, 'k',
                        ha='right', **arg_param)

        #######################################################################
        # exp
        #######################################################################
        ax = show_exp(ax, SOff, TOn, time, aymin, aymax, dymin, dymax, t_label,
                      plot='')

        plt.tight_layout()
        plt.savefig('classical_method_steady_state.svg')
        plt.show()



    def anticipation(expname='sub-002_task-aSPEM', trial=8):

        time = np.arange(1200)
        dxmin, dxmax, dymin, dymax = np.min(time),    np.max(time),    -20, 35
        axmin, axmax, aymin, aymax = dxmin-dxmax/100, dxmax+dxmax/100, -22, 50

        fig, ax = plt.subplots(1, 1, figsize=(15, (15*(1/2)/1.6180)))
        ax.set_title('Anticipation', fontsize=t_titre, x=0.5, y=1.05)
        ax.set_ylabel('Velocity (°/s)', fontsize=t_label)
        ax = cosmetique(ax, dxmin, dxmax, dymin, dymax, axmin, axmax, aymin,
                        aymax, t_label)


        #######################################################################
        # DATA
        #######################################################################
        data = DATA[expname].Data
        events = DATA[expname].Events

        data_trial = data[data.trial==trial]
        trial_events = events[events.trial==trial]
        targeton = int(trial_events.TargetOn.values[0] \
                       - data_trial.time.values[0])
        d = data_trial[targeton-500:].vx_NaN
        ax.plot(d.values[:len(time)], 'k', alpha=.2)
        #----------------------------------------------------------------------

        dyt = (dymax-dymin)/7
        arg = dict(height=abs(dymin)+abs(aymax), bottom=dymin, lw=0,
                   align='edge')
        ax.bar(TOn-50, width=100, color=ca, alpha=0.2, **arg)
        ax.vlines(TOn-50, ymin=dymin, ymax=aymax, ls='--', lw=1,color=ca,
                  alpha=.7)
        ax.vlines(TOn+50, ymin=dymin, ymax=aymax, ls='--', lw=1,color=ca,
                  alpha=.7)

        arg_param = dict(t_label=t_label/1.3, t_labelp=t_label/1.5)

        xt, yt = TOn, dymax-(dymax-dymin)/15
        axy, axyt = (TOn-50, dymax), (TOn+50, dymax)
        ax = show_param(ax, 'average velocity of the eye in this window',
                        'anticipation', xt, yt, dyt, axy, axyt, ca,
                        ha='center', arrow='<->', **arg_param)


        xt, yt = TOn-50-20, (dymax-dymin)/3.5
        axy, axyt = (TOn-50, yt), (xt, yt)
        ax = show_param(ax, 'add time at the start of the event',
                        'add_stime=-50', xt, yt, dyt, axy, axyt, 'k',
                        ha='right', **arg_param)

        xt, yt = TOn+50-20, (dymax-dymin)/15
        axy, axyt = (TOn+50, yt), (xt, yt)
        ax = show_param(ax, 'add time at the end of the event', 'add_etime=50',
                        xt, yt, dyt, axy, axyt, 'k', ha='right', **arg_param)


        #######################################################################
        # exp
        #######################################################################
        ax = show_exp(ax, SOff, TOn, time, aymin, aymax, dymin, dymax, t_label,
                      plot='')

        plt.tight_layout()
        plt.savefig('classical_method_anticipation.svg')
        plt.show()


class Plot_Model:

    class SmoothPursuit:

        def velocity():

            #------------------------------------------------------------------
            dir_target = 1
            sa = TOn-100 # start_anti
            l  = TOn+100 # latency
            aa = 15      # a_anti
            t  = 13      # tau
            ss = 15      # steady_state
            do_whitening = False

            velocity = Model.SmoothPursuit.velocity(time, dir_target, sa,
                                                       aa, l, t, ss,
                                                       do_whitening)
            #------------------------------------------------------------------

            fig, ax = plt.subplots(1, 1, figsize=(15, (15*(1/2)/1.6180)))
            ax.set_title('Model velocity', fontsize=t_titre, x=0.5, y=1.05)
            ax.set_ylabel('Velocity (°/s)', fontsize=t_label)

            #------------------------------------------------------------------
            ax, dyt = plot_pursuit(ax, velocity, time, SOff, TOn, sa, l, ss,
                                   cf, ca, cp, t_label)

            # tau -------------------------------------------------------------
            xt, yt = l+50, velocity[-1]/1.5
            axy, axyt = (l+20, velocity[l+20]), (xt, yt)
            ax = show_param(ax, r'$\tau$', 'tau', xt, yt, dyt, axy, axyt, cp,
                            t_label)

            plt.tight_layout()
            plt.savefig('model_velocity.svg')
            plt.show()


        def velocity_sigmo():

            #------------------------------------------------------------------
            dir_target = 1
            sa = TOn-100 # start_anti
            l  = TOn+100 # latency
            aa = 15      # a_anti
            rp = 100     # ramp_pursuit
            ss = 15      # steady_state
            do_whitening = False

            velocity_sigmo = Model.SmoothPursuit.velocity_sigmo(time,
                                                                dir_target, sa,
                                                                aa, l, rp, ss,
                                                                do_whitening)
            #------------------------------------------------------------------

            fig, ax = plt.subplots(1, 1, figsize=(15, (15*(1/2)/1.6180)))
            ax.set_title('Model Sigmoid Velocity', fontsize=t_titre, x=0.5,
                         y=1.05)
            ax.set_ylabel('Velocity (°/s)', fontsize=t_label)

            #------------------------------------------------------------------
            ax, dyt = plot_pursuit(ax, velocity_sigmo, time, SOff, TOn, sa, l,
                                   ss, cf, ca, cp, t_label)

            # ramp_pursuit ----------------------------------------------------
            xt, yt = l+100, velocity_sigmo[-1]/2
            axy, axyt = (l+60, velocity_sigmo[l+60]), (xt, yt)
            ax = show_param(ax, 'Ramp Pursuit', 'ramp_pursuit', xt, yt, dyt,
                            axy, axyt, cp, t_label)

            plt.tight_layout()
            plt.savefig('model_velocity_sigmo.svg')
            plt.show()


        def velocity_line():

            #------------------------------------------------------------------
            dir_target = 1
            sa = TOn-100 # start_anti
            l  = TOn+100 # latency
            aa = 15      # a_anti
            rp = 50      # ramp_pursuit
            ss = 15      # steady_state
            do_whitening = False

            velocity_line = Model.SmoothPursuit.velocity_line(time,
                                                                 dir_target,
                                                                 sa, aa, l, rp,
                                                                 ss,
                                                                 do_whitening)
            #------------------------------------------------------------------

            fig, ax = plt.subplots(1, 1, figsize=(15, (15*(1/2)/1.6180)))
            ax.set_title('Model Line Velocity', fontsize=t_titre, x=0.5,
                         y=1.05)
            ax.set_ylabel('Velocity (°/s)', fontsize=t_label)

            #------------------------------------------------------------------
            ax, dyt = plot_pursuit(ax, velocity_line, time, SOff, TOn, sa, l,
                                   ss, cf, ca, cp, t_label)

            # ramp_pursuit ----------------------------------------------------
            xt, yt = l+150, velocity_line[-1]/3
            axy, axyt = (l+100, velocity_line[l+100]), (xt, yt)
            ax = show_param(ax, 'Ramp Pursuit', 'ramp_pursuit', xt, yt, dyt,
                            axy, axyt, cp, t_label)

            plt.tight_layout()
            plt.savefig('model_velocity_line.svg')
            plt.show()



        def position():

            #------------------------------------------------------------------
            dir_target = 1
            sa = TOn-100 # start_anti
            l  = TOn+100 # latency
            aa = 15      # a_anti
            t  = 13      # tau
            ss = 15      # steady_state
            do_whitening = False

            position = Model.SmoothPursuit.position(time, time, dir_target,
                                                       sa, aa, l, t, ss,
                                                       do_whitening)
            #------------------------------------------------------------------

            fig, ax = plt.subplots(1, 1, figsize=(15, (15*(1/2)/1.6180)))
            ax.set_title('Model position', fontsize=t_titre, x=0.5, y=1.05)
            ax.set_ylabel('position (°)', fontsize=t_label)

            #------------------------------------------------------------------
            dxmin, dxmax, dymin, dymax = np.min(time), np.max(time), -15, 25
            axmin, axmax = dxmin-dxmax/100, dxmax+dxmax/100
            aymin, aymax = -17, 35
            ax = cosmetique(ax, dxmin, dxmax, dymin, dymax, axmin, axmax,
                            aymin, aymax, t_label)
            ###################################################################
            # Model
            ###################################################################
            ax = show_fct_pursuit(ax, position, time, sa, l, dymin, dymax, cf,
                                  ca, cp, t_label)

            #------------------------------------------------------------------
            # Parameters
            #------------------------------------------------------------------
            exf, eyf = 10, -2 # arrow gap
            dyt = (dymax-dymin)/7

            # Start_a ---------------------------------------------------------
            xt, yt = sa-50, dymin+dyt
            axy, axyt = (sa-exf, yt), (xt+exf, yt)
            ax = show_param(ax, "Start anticipation", 'start_anti', xt, yt,
                            dyt, axy, axyt, ca, t_label, ha='right')
            # V_a -------------------------------------------------------------
            xt, yt = sa+(l-sa)/2, position[l-sa]+eyf-(dyt/2)
            xtp = sa+7
            axy, axyt = (l-exf, position[l]+eyf), (sa+exf, position[sa]+eyf)
            ax = show_param(ax, r"V$_a$",
                            '\nd_t=(latency-start_anti)\nv_anti=a_anti*d_t',
                            xt, yt, dyt, axy, axyt, ca, t_label, ha='center',
                            xtp=xtp, hap='left')
            # latency ---------------------------------------------------------
            xt, yt = l+50, dymin+dyt
            axy, axyt = (l+exf, yt), (xt-exf, yt)
            ax = show_param(ax, "Latency", 'latency', xt, yt, dyt, axy, axyt,
                            cp, t_label)
             # tau ------------------------------------------------------------
            xt, yt = l+50, position[-1]/1.5
            axy, axyt = (l+20, position[l+20]), (xt, yt)
            ax = show_param(ax, r'$\tau$', 'tau', xt, yt, dyt, axy, axyt, cp,
                            t_label)
            # Steady State ----------------------------------------------------
            xt, yt = l+300, position[l+300]+eyf-(dyt/2)
            axy = (l+400, position[l+400]+eyf)
            axyt = (l+200, position[l+200]+eyf)
            ax = show_param(ax, 'Steady State', 'steady_state', xt, yt, dyt,
                            axy, axyt, cp, t_label, arrow='->')

            ###################################################################
            # exp
            ###################################################################
            ax = show_exp(ax, SOff, TOn, time, aymin, aymax, dymin, dymax,
                          t_label)

            plt.tight_layout()
            plt.savefig('model_position.svg')
            plt.show()


    def saccade():

        #t_titre, t_label  = 20, 10
        #----------------------------------------------------------------------
        time = np.arange(30)

        T0,  t1, t2, tr  = 1, 15, 12, 1
        x_0, x1, x2, tau = 0.01, 2, 1, 13

        sac = Model.saccade(time, x_0, tau, x1, x2, T0, t1, t2, tr,
                               do_whitening = False)
        #----------------------------------------------------------------------

        #----------------------------------------------------------------------
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(15, (15*(1/2)/1.6180)))
        gs1 = gridspec.GridSpec(1, 2)
        ax = fig.add_subplot(gs1[0])
        #fig, ax = plt.subplots(1, 1, figsize=(15*(1/4), (15*(1/2)/1.6180)))
        ax.set_title('Model saccade', fontsize=t_titre, x=0.5, y=1.05)
        ax.set_ylabel('position (°)', fontsize=t_label)

        #----------------------------------------------------------------------
        dxmin, dxmax, dymin, dymax = 0, time[-1], 0, 2.01
        axmin, axmax = dxmin-(dxmax-dxmin)/30, dxmax+(dxmax-dxmin)/30
        aymin, aymax = dymin-(dymax-dymin)/30, dymax+(dymax-dymin)/20

        ax = cosmetique(ax, dxmin, dxmax, dymin, dymax, axmin, axmax, aymin,
                        aymax, t_label, step_yticks=1)
        #######################################################################
        # Model
        #######################################################################
        ax.plot(time, sac, c='k')
        #----------------------------------------------------------------------

        #----------------------------------------------------------------------
        # Parameters
        #----------------------------------------------------------------------
        exf, eyf = 1, -2 # arrow gap
        dyt = (dymax-dymin)/30

        arg_bar = dict(height=abs(dymin)+abs(dymax+dyt), bottom=dymin, lw=0,
                       align='edge')


        # T0 ------------------------------------------------------------------
        xt, yt = time[0]+T0 + 2*exf, dymax- 2*dyt
        axy, axyt = (time[0]+T0, yt), (xt, yt)
        ax = show_param(ax, "", 'T0', xt, yt, dyt, axy, axyt, 'k', t_label*1.1,
                        ha='left', ytp=yt)
        ax.vlines(time[0]+T0, ymin=dymin, ymax=dymax+dyt, color='k', lw=1,
                  alpha=.7)

        # t1 ------------------------------------------------------------------
        xt, yt = time[0]+T0+(t1/2), dymax+2*dyt
        axy, axyt = (time[0]+T0, yt-dyt), (time[0]+T0+t1, yt-dyt)
        ax = show_param(ax, "", 't1', xt, yt, dyt, axy, axyt, 'k', t_label*1.1,
                        ha='center', ytp=yt, arrow='<->')
        ax.bar(time[0]+T0, width=t1, color='k', alpha=0.1, **arg_bar)
        ax.vlines(time[0]+T0+t1, ymin=dymin, ymax=dymax+dyt, color='k', lw=1,
                  alpha=.7)


        # t2 ------------------------------------------------------------------
        xt, yt = time[0]+T0+t1+(t2/2), dymax+2*dyt
        axy, axyt = (time[0]+T0+t1, yt-dyt), (time[0]+T0+t1+t2, yt-dyt)
        ax = show_param(ax, "", 't2', xt, yt, dyt, axy, axyt, 'k', t_label*1.1,
                        ha='center', ytp=yt, arrow='<->')
        ax.bar(time[0]+T0+t1, width=t2, color='k', alpha=0.2, **arg_bar)
        ax.vlines(time[0]+T0+t1+t2, ymin=dymin, ymax=dymax+dyt, color='k',
                  lw=1, alpha=.7)

        # tr ------------------------------------------------------------------
        xt, yt = time[0]+T0+t1+t2+(tr/2), dymax+2*dyt
        axy, axyt = (time[0]+T0+t1+t2, yt-dyt), (time[0]+T0+t1+t2+tr, yt-dyt)
        ax = show_param(ax, "", 'tr', xt, yt, dyt, axy, axyt, 'k', t_label*1.1,
                        ha='center', ytp=yt, arrow='-')
        ax.bar(time[0]+T0+t1+t2, width=tr, color='k', alpha=0.3, **arg_bar)
        ax.vlines(time[0]+T0+t1+t2+tr, ymin=dymin, ymax=dymax+dyt, color='k',
                  lw=1, alpha=.7)

        # x_0 -----------------------------------------------------------------
        xt, yt = dxmax + 3*exf, x_0
        axy, axyt = (dxmax+exf, yt), (xt, yt)
        ax = show_param(ax, "", 'x_0', xt, yt, dyt, axy, axyt, 'k',
                        t_label*1.1, ha='left', ytp=yt)
        ax.hlines(x_0, dxmin, axmax, color='k', lw=1, linestyles='--',
                  alpha=0.5)

        # x1 ------------------------------------------------------------------
        xt, yt = dxmax + 3*exf, x_0+x1
        axy, axyt = (dxmax+exf, yt), (xt, yt)
        ax = show_param(ax, "", 'x1', xt, yt, dyt, axy, axyt, 'k', t_label*1.1,
                        ha='left', ytp=yt)
        ax.hlines(x_0+x1, dxmin, axmax, color='k', lw=1, linestyles='--',
                  alpha=0.5)

        # x2 ------------------------------------------------------------------
        xt, yt = dxmax + 3*exf, x_0+x2
        axy, axyt = (dxmax+exf, yt), (xt, yt)
        ax = show_param(ax, "", 'x2', xt, yt, dyt, axy, axyt, 'k', t_label*1.1,
                        ha='left', ytp=yt)
        ax.hlines(x_0+x2, dxmin, axmax, color='k', lw=1, linestyles='--',
                  alpha=0.5)

        # tau -----------------------------------------------------------------
        xt, yt = time[0]+T0+t1+1, x1-(x1-x2)/4
        axy, axyt = (time[0]+T0+t1, x1), (xt, yt+dyt/2)
        ax = show_param(ax, '', 'tau', xt, yt, dyt, axy, axyt, 'k',
                        t_label*1.1, ha='center', ytp=yt)

        #----------------------------------------------------------------------

        plt.tight_layout()
        plt.savefig('model_saccade.svg')
        plt.show()



