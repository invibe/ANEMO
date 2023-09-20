#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .Error import CallError, fileError
import os
import numpy as np
import pandas as pd


def test_toxname(toxname, settings, fonction, arg):

    '''
    Test if the toxname is already used, if yes it returns a new unused toxname
    '''

    new_arg = arg.copy()
    for k in arg.keys():

        if callable(arg[k]):
            new_arg[k] = arg[k].__name__

        if k in ['data', 'expname', 'trial']:
            new_arg.pop(k)

        if type(arg[k])==dict:

            new_arg[k] = arg[k].copy()
            for k_ in arg[k].keys():

                if callable(arg[k][k_]):
                    new_arg[k][k_] = arg[k][k_].__name__

                if k_ in ['data', 'expname', 'trial']:
                    new_arg[k].pop(k_)

    if toxname in settings.keys():

        new_toxname = None
        n = [1]
        for k in settings.keys():

            if type(k)==str and len(k)>=len(toxname):

                if k[:len(toxname)]==toxname:

                    if (fonction!=settings[k]['fonction']
                        or new_arg!=settings[k]['arg']):

                        try:
                            n.append(int(k.split('_')[-1])+1)
                        except:
                            pass

                    else:
                        new_toxname = k

        if not new_toxname:
            new_toxname = toxname+'_'+str(max(n))

    else:
        new_toxname = toxname

    if new_toxname not in settings.keys():
        settings.loc['Description', new_toxname] = 'new ANEMO data'
        settings.loc['fonction', new_toxname] = fonction
        settings.loc['arg', new_toxname] = None
        settings.loc['trial', new_toxname] = None

        settings[new_toxname]['arg'] = new_arg
        settings[new_toxname]['trial'] = []

    return new_toxname


def check_param(self, ClassTrial, NameClass, expname, trial):

    if not isinstance(self, ClassTrial):
        raise CallError(NameClass)

    if not expname in self._.data.keys():
        msg = "the parameter expname is not filled in correctly, \n"
        msg += "the names of the particular experiments available are:"
        msg += "\n%s"%(list(self._.data.keys()))
        raise ValueError(msg)

    if not trial in np.unique(list(self._.data[expname].Events.trial)):
        msg = "the parameter trial is not filled in correctly, \n"
        msg += "the numbers of the trial available are:"
        msg += "\n%s"%(np.unique(list(self._.data[expname].Events.trial)))
        raise ValueError(msg)


def list_event(values, add_time):

    '''
    Return a list time event
    '''

    v = eval(values) if type(values)==str else values
    v = [v] if type(v)!=list else v
    v = np.array(v)
    if np.all(v): v = v +add_time
    return v

def list_filesname(filesnameprop, dirpath):

    '''
    Return list filenames
    '''

    filesnames = []
    for root, dirs, files in os.walk(dirpath):
        for f in files:
            f_prop = {x.split('-')[0]:x.split('-')[1] for x in f.split('_')
                      if len(x.split('-'))>1}
            if f not in filesnames and 'sub' in f_prop.keys():
                check = [(f_prop[k]==v if k in f_prop else False)
                         for k, v in zip(filesnameprop.keys(),
                                         filesnameprop.values()) if v]
                if all(check):
                    filesnames.append(os.path.join(root, f))
    return filesnames

def create_filepath(path, sub, ses):

    '''
    Create a file path
    '''

    filepath = os.path.join(path,     'sub-'+sub)
    filepath = os.path.join(filepath, 'ses-'+ses) if ses else filepath
    if not os.path.isdir(filepath):
        os.makedirs(filepath)

    return filepath

def add_trial_Data(data):

    '''
    Add trial data
    '''

    data.Data.insert(0, 'trial', None)
    data.Data.eye_timestamp = data.Data.eye_timestamp.astype(float)
    data.Data = data.Data.rename(columns={'eye_timestamp': 'time'})
    data.Events['sample'] = data.Events['sample'].astype(float)

    trials = []
    for trial in data.Events.trial:
        start = data.Events[data.Events.trial==trial]['sample'].values[0]
        try:
            end = data.Events[data.Events.trial==trial+1]['sample'].values[0]
            idx = (data.Data.time>=start) & (data.Data.time<end)
        except:
            idx = (data.Data.time>=start)

        data.Data.loc[idx, 'trial'] = int(trial)

    return data

def add_px_per_deg_Settings(data):

    '''
    Add px-per_deg in settings data
    '''

    screen_width_cm = float(data.Settings.ScreenSize[0][0])
    ScreenDistance  = float(data.Settings.ScreenDistance[0])
    screen_width_px = float(data.Settings.ScreenResolution[0][0])

    screen_width_deg = 2. * np.arctan((screen_width_cm/2)/ScreenDistance)
    screen_width_deg *= 180/np.pi
    data.Settings['px_per_deg'] = screen_width_px / screen_width_deg
    return data




def save_file(data, filename, filepath):

    '''
    Save the files json, tsv or pickle
    '''

    fileform = filename.split('.')[-1]
    if fileform in ['json', 'gz', 'pkl']:
        if type(data)==dict:
            data = pd.DataFrame.from_dict(data)

        if filepath:
            filename = os.path.join(filepath, filename)
        if fileform=='json':
            data.to_json(filename, orient='index', indent=4)
        elif fileform=='gz':
            if filename.split('.')[-2]=='tsv':
                #data = data.astype(str)
                data.to_csv(filename, sep=' ', index=False,
                            compression={'method': 'gzip',
                                         'compresslevel': 2,
                                         'mtime': 0})
            elif filename.split('.')[-2]=='pkl':
                data.to_pickle(filename,
                               compression={'method': 'gzip',
                                            'compresslevel': 2,
                                            'mtime': 0})
        del data

def read_file(filename, filepath):

    '''
    Open the files json, tsv or pickle
    '''

    fileform = filename.split('.')[-1]
    if fileform in ['json', 'tsv', 'gz', 'pkl']:

        if filepath:
            filename = os.path.join(filepath, filename)

        if fileform=='json':
            file = pd.read_json(filename, orient='index')

        else:
            if fileform=='tsv':
                file = pd.read_csv(filename, sep=' ',
                                   dtype={'participant_id':str})
            elif fileform=='gz':
                if filename.split('.')[-2]=='tsv':
                    file = pd.read_csv(filename, sep=' ', compression='gzip')
                elif filename.split('.')[-2]=='pkl':
                    file = pd.read_pickle(filename, compression='gzip')

            if 'trial' in file.columns:
                if file.trial.dtype!='int':
                    file.trial = [int(t) for t in file.trial]

        return file

    else:
        return None

def open_file(data, filename, dirpath, RawData, filepath=None):

    #--------------------------------------------------------------------------
    # creation of a sub-dictionary in the data for each expname
    #--------------------------------------------------------------------------
    expname = filename.split('/')[-1][:-len(filename.split('_')[-1])-1] \
               if '-' not in filename.split('_')[-1] \
               else filename[:-len(filename.split('.')[-1])-1]

    if expname not in data.keys():
        data[expname] = {}

    if not 'Settings' in data[expname].keys():
        settings = {'Data':[pd.DataFrame()],
                    'Results':[pd.DataFrame()],
                    'Events':[pd.DataFrame()]}
        data[expname]['Settings'] = pd.DataFrame.from_dict(settings)
    #--------------------------------------------------------------------------
    if RawData:
        ext = ['eyetrack.tsv.gz', 'eyetrack.json', 'events.tsv', 'events.json']
    else:
        ext = ['data.pkl.gz', 'events.pkl.gz', 'results.pkl.gz',
               'data.json', 'events.json', 'results.json', 'settings.json']

    good_ext = False
    for e in ext:
        if filename[-len(e):]==e:
            good_ext = True

    if not good_ext:
        stop_open = True
    else:
        stop_open = False

    while not stop_open:

        #--------------------------------------------------------------------------
        # extract Data
        #--------------------------------------------------------------------------
        if RawData:
            ext = 'eyetrack.tsv.gz'
        else:
            ext = 'data.pkl.gz'

        if filename[-len(ext):]==ext:
            x = read_file(filename, filepath)
            if type(x)!=type(None):
                data[expname]['Data'] = x
                stop_open = True
                break

        #----------------------------------------------------------------------
        # extract Events
        #----------------------------------------------------------------------
        if RawData:
            ext = 'events.tsv'
        else:
            ext = 'events.pkl.gz'

        if filename[-len(ext):]==ext:
            x = read_file(filename, filepath)
            if type(x)!=type(None):
                data[expname]['Events'] = x

                if RawData:
                    if not 'Results' in data[expname].keys():
                        results = pd.DataFrame({'trial': x['trial'].values})
                        data[expname]['Results'] = results

                stop_open = True
                break

        #----------------------------------------------------------------------
        # extrat Results
        #----------------------------------------------------------------------
        if not RawData:
            ext = 'results.pkl.gz'
            if filename[-len(ext):]==ext:
                x = read_file(filename, filepath)
                if type(x)!=type(None):
                    data[expname]['Results'] = x
                    stop_open = True
                    break

        #----------------------------------------------------------------------
        # extract Settings
        #----------------------------------------------------------------------
        if RawData:
            ext = 'eyetrack.json'
        else:
            ext = 'settings.json'

        if filename[-len(ext):]==ext:
            settings = read_file(filename, filepath).T

            if RawData:
                sub = filename.split('/')[-1].split('_')[0].split('-')[1]
                for f in os.listdir(dirpath):
                    if 'participants' in f:
                        infos = read_file(f, dirpath)
                try:
                    infoParticipant = infos[infos.participant_id==sub]
                except:
                    raise fileError(self.dirpath, None, 'participants.tsv')
                for c in infoParticipant.columns:
                    settings[c] = infoParticipant[c].values

            x = settings
            if type(x)!=type(None):
                data[expname]['Settings'] = data[expname]['Settings'].join(x)
                stop_open = True
                break

        if RawData:
            ext = 'events.json'
            if filename[-len(ext):]==ext:
                x = read_file(filename, filepath)
                if type(x)!=type(None):
                    data[expname]['Settings']['Events'] = [x.T]
                    stop_open = True
                    break

        else:
            for ext, key in zip(['data.json', 'results.json', 'events.json'],
                                ['Data', 'Results', 'Events']):
                if filename[-len(ext):]==ext:
                    x = read_file(filename, filepath)
                    if type(x)!=type(None):
                        data[expname]['Settings'][key] = [x.T]
                        stop_open = True
                        break
        #----------------------------------------------------------------------


    return data



# whitening fonction
def whitening_filt(N_freq, white_f_0, white_alpha, white_steepness):

    """
    Returns the envelope of the whitening filter.

        then we return a 1/f spectrum based on the assumption that the
        structure of signals
        is self-similar and thus that the Fourier spectrum scales a priori in
        1/f.
    """

    freq = np.fft.fftfreq(N_freq, d=1.)

    K  = np.abs(freq)**(white_alpha)
    K *= np.exp(-(np.abs(freq)/white_f_0)**white_steepness)
    K /= np.mean(K)

    return freq, K

def whitening(position, white_f_0=.4, white_alpha=.5, white_steepness=4):

    """
    Returns the whitened image
    /!\ position must not contain Nan
    """

    try:
        N_freq = position.shape[0]
    except AttributeError:
        N_freq = len(position)

    freq, K = whitening_filt(N_freq=N_freq, white_f_0=white_f_0,
                             white_alpha=white_alpha,
                             white_steepness=white_steepness)
    f_position = np.fft.fft(position)

    return np.real(np.fft.ifft(f_position*K))

