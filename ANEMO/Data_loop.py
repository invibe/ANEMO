#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .Init import *
from .various_functions import *

def Data_loop(self, trials, expnames, toxname, fct, name_fct, arg_fct, loop,
              recalculate):

    """
    Execute the requested function on the data.

    Parameters
    ----------
    trials: int, or list(int)
        Number or list of the trial to be transformed,
        if ``None`` all the trials will be transformed
    expnames: str, or list(str)
        Name or list of names of the particular experience to be transformed,
        if ``None`` all experiences will be transformed
    toxname: str
        Name of the data to be saved

    fct: function
        Function execute in the loop
    name_fct: str
        Name of the function to be executed in the loop
    arg_fct: dict
        Argument to the execute function in the loop
    loop: str
        ``PreProcessing`` if loop for PreProcessing function
        ``Processing`` if loop for Processing function

    recalculate: bool (default False)
        Allows you to indicate if you want to force the calculation on
        the trials already processed

    Returns
    -------
    data: dict
        Dictionary containing for each experiment the ``Data``,
        the ``Results``, the ``Events``, and the ``Settings``
    """

    #--------------------------------------------------------------------------
    # dictionnary of files
    #--------------------------------------------------------------------------
    if expnames:
        # check if expnames is a list
        # otherwise transforms expnames into a list
        expnames = expnames if type(expnames)==list else [expnames]

        #----------------------------------------------------------------------
        # creation of a dictionary corresponding to the file list for each
        # particular experiment
        #----------------------------------------------------------------------
        dict_files = {}
        for exp in expnames:

            filesnameprop = {e.split('-')[0]:e.split('-')[1]
                             for e in exp.split('_')}
            files = list_filesname(filesnameprop, self._.dirpath)

            dict_files = {exp:files, **dict_files}

    else:
        # if expnames are not defined,
        # then all files are taken.
        filesnameprop = {'sub':self._.sub, 'ses':self._.ses, 'acq':self._.acq,
                         'run':self._.run}
        files = list_filesname(filesnameprop, self._.dirpath)

        #----------------------------------------------------------------------
        # creation of a dictionary corresponding to the file list for each
        # particular experiment
        #----------------------------------------------------------------------
        dict_files = {}
        for f in files:

            exp = f.split('/')[-1][:-len(f.split('_')[-1])-1] \
                   if '-' not in f.split('_')[-1] \
                   else f[:-len(f.split('.')[-1])-1]

            if exp not in dict_files.keys():
                dict_files[exp] = []

            dict_files[exp].append(f)


    #--------------------------------------------------------------------------
    # Loop to perform the function on all the expnames
    #--------------------------------------------------------------------------
    arg_fct['return_'] = False

    save = len(dict_files.keys())

    import easydict
    for exp in dict_files.keys():


        #----------------------------------------------------------------------
        # list trial
        #----------------------------------------------------------------------
        if trials:
            # check if trials is a list
            # otherwise transforms trials into a list
            trials_ = trials if type(trials)==list else [trials]
        else:
            # if trials are not defined
            # then all trials in the data are taken.
            trials_ = list(np.unique(self._.data[exp].Data.trial))

        #----------------------------------------------------------------------
        # Loop to test if toxname already exists.
        #----------------------------------------------------------------------
        # If it turns out is that it corresponds to a function already in use
        # or the same function but with different parameters
        # then a number will be added to the end of toxname
        # so as not to overwrite the existing ones.
        if loop:
            settings = self._.data[exp].Settings[loop][0]
            toxname = test_toxname(toxname, settings, name_fct, arg_fct)

        else:
            settings = self._.data[exp].Settings.Results[0]
            for s in [self._.data[exp].Settings.Data[0],
                      self._.data[exp].Settings.Results[0]]:
                toxname = test_toxname(toxname, s, name_fct, arg_fct)

        #----------------------------------------------------------------------
        # Loop to perform the function on all the trials
        #----------------------------------------------------------------------
        print("calculates %s from the file %s..."%(toxname, exp), end=' ')

        trials_already_calculated = []
        for t_ in np.unique(trials_):
            # check if the function is already performed on the trial
            if (t_ in settings[toxname]['trial']) and (not recalculate):
                trials_already_calculated.append(t_)
            else:
                # perform the function on trial
                fct(self, trial=t_, expname=exp, toxname=toxname, **arg_fct)

        #----------------------------------------------------------------------
        # print a warning if the function has already been performed on trials
        #----------------------------------------------------------------------

        if len(trials_already_calculated)!=0:
            msg = "\n\n%s was already calculated "%toxname

            if trials_already_calculated==trials_:
                save -= 1
                msg += "for this file, so it was not recalculated"
            else:
                msg += "for some trials, so it was not recalculated "
                msg += "for these trials\n"
                msg += "list trial already calculated:"
                msg += "%s"%trials_already_calculated

            msg += "\nto force the calculation modify the variable: "
            msg += "recalculate=True\n"

            print(msg)
        else:
            print('finished')

    #----------------------------------------------------------------------
    # Save Data
    #----------------------------------------------------------------------
    if save!=0:
        print('save data...', end=' ')
        # save the data
        Data.save(self._.dirpath, self._.data)
        print('finished')
    #----------------------------------------------------------------------

    return self._.data

