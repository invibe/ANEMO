#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .various_functions import *

def init(RawDatadirpath, Datadirpath, sub=None, task=None, ses=None, acq=None,
         run=None, resave=False):

    """
    Allows to create from a folder containing **BIDSified data**
    ``RawDatadirpath`` a new folder ``Datadirpath`` which will contain the
    results of ANEMO, this new folder will contain four files per subject:

        - ``*_data.tsv`` - contains the raw data, those calculated from
          :ref:`PreProcessing`, and the **Fit** calculated from
          :ref:`ProcessingSaccades` and :ref:`ProcessingSmoothPursuit`
        - ``*_results.tsv`` - contains the results calculated from
          :ref:`ProcessingSaccades` and :ref:`ProcessingSmoothPursuit`
        - ``*_events.tsv`` - contains the raw events, and those calculated from
          :ref:`ProcessingSaccades`
        - ``*_settings.tsv`` - contains the settings of the experiment, and
          function names and their parameters used for ``data``, ``events`` and
          ``results`` calculated from **ANEMO**

    You can choose to create this file only for a subject, a session, a task,
    an acquisition or a particular run by changing the parameters.
    ``None`` by default means that all files will be taken into account.

    .. Warning::
       Attention the RawDatadirpath directory must contain the **raw BIDSified
       data**.

       To BIDSify your data we invite you to use the
       `BIDSification_eyetrackingData
       <https://chloepasturel.github.io/BIDSification_eyetrackingData>`_
       package.


    Parameters
    ----------
    RawDatadirpath: str
        Raw data directory path
    Datadirpath: str
        Processed data directory path

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

    resave: bool (default False)
        If ``True`` resave files if they already exist,
        if ``False`` do not resave files if they already exist
    """

    if resave:

        msg = 'Warning: the parameter resave=True, '
        msg += 'if you use this function to add files '
        msg += 'to your already existing %s folder '%(Datadirpath)
        msg += 'and it contains files with modified or added data, '
        msg += 'these will be lost'

        print(msg)

    # open the raw data
    data = Data.open(RawDatadirpath, sub, task, ses, acq, run, RawData=True)

    #--------------------------------------------------------------------------
    # loop allowing to create a data ANEMO
    #--------------------------------------------------------------------------
    for expname in data.keys():

        if 'trial' not in data[expname].Data.columns:
            # add trial data
            data[expname] = add_trial_Data(data[expname])

        if 'px_per_deg' not in data[expname].Settings.columns:
            # add px-per_deg in settings data
            data[expname] = add_px_per_deg_Settings(data[expname])
    #--------------------------------------------------------------------------

    # save the new data
    Data.save(Datadirpath, data, resave)


class Data:

    def dirtree(dirpath, return_=False):

        """
        Allows to display the tree structure of a ``dirpath`` data folder.

        Parameters
        ----------
        dirpath: str
            Data directory path
        return_: bool (default False)
            If ``False`` prints the path tree in the console,
            if ``True`` does not print the dirpath tree but returns a variable
            containing it

        Returns
        -------
        tree: str
            Return a variable containing tree of dirpath if ``return_=True``
        """

        # creation of the tree structure
        tree = ''

        # allows you to add indentation to the text to make it easier to read
        indent = ' '*4

        #----------------------------------------------------------------------
        # loop allowing to add in the tree variable the tree of the folder
        #----------------------------------------------------------------------
        for root, dirs, files in os.walk(dirpath):
            level = root.replace(dirpath, '').count(os.sep)
            tree += indent*level + os.path.basename(root) + '/\n'
            for f in files:
                tree += indent*(level+1) + f + '\n'
        #----------------------------------------------------------------------

        if return_:
            return tree

        else:
            print(tree)

    def open(dirpath, sub=None, task=None, ses=None, acq=None, run=None,
             RawData=False):

        """
        Allows you to open a data folder.

        This function returns a python dictionary containing all the data in
        the directory.

        The keys in this dictionary correspond to each particular experiment in
        the directory, and for each experiment, this dictionary contains a
        sub-dictionary containing ``DataFrame pandas``:

            - ``Data`` - contains the raw data, those calculated from
              :ref:`PreProcessing`, and the **Fit** calculated from
              :ref:`ProcessingSaccades` and :ref:`ProcessingSmoothPursuit`
            - ``Results`` - contains the results calculated from
              :ref:`ProcessingSaccades` and :ref:`ProcessingSmoothPursuit`
            - ``Events`` - contains the raw events and those calculated from
              :ref:`ProcessingSaccades`
            - ``Settings`` - contains the settings of the experiment, and
              function names and their parameters used for ``Data``, ``Events``
              and ``Results`` calculated from **ANEMO**.

              The *Data_column* contains the characteristics of the
              functions used to calculate the data contained in ``Data``
              and ``Events``,
              the *Results column* those used to calculate the data
              contained in ``Results``.


        Parameters
        ----------
        dirpath: str
            Data directory path

        sub: str, or None (default None)
            Participant identifier
        ses: str, or None (default None)
            Name of the Session
        task: str, or None (default None)
            Name of the Task
        acq: str, or None (default None)
            Name of the Aquisition
        run: str, or None (default None)
            IndexRun
        RawData: bool (default False)
            If ``True`` open RawData,
            if ``False`` open DataAnemo

        Returns
        -------
        data: dict
            Dictionary containing for each experiment the ``Data``,
            the ``Results``, the ``Events``, and the ``Settings``
        """

        arg = locals() # retrieves the parameters of the function

        #----------------------------------------------------------------------
        # creation of a list of files present in the data folder corresponding
        # to the requested parameters
        #----------------------------------------------------------------------
        filesnameprop = {k:str(arg[k]) if arg[k] else '' for k in ['sub',
                                                                   'ses',
                                                                   'acq',
                                                                   'run']}
        files = list_filesname(filesnameprop, dirpath)
        #----------------------------------------------------------------------

        # dictionnary of the data
        import easydict
        data = easydict.EasyDict()

        #----------------------------------------------------------------------
        # loop to open all files in the files list
        #----------------------------------------------------------------------
        for f in files:
            data = open_file(data, f, dirpath, RawData)

        return data

    def save(dirpath, data, resave=True):

        """
        Allows you to save a data folder.

        Parameters
        ----------
        dirpath: str
            Data directory path

        data: dict
            Dictionary containing for each experiment the ``Data``,
            the ``Results``, the ``Events``, and the ``Settings``

        resave: bool (default True)
            If ``True`` resave files if they already exist,
            if ``False`` do not resave files if they already exist
        """


        def check_save(filepath,filename):

            savefile = True
            if not resave:
                if os.path.isfile(os.path.join(filepath,filename)):
                    savefile = False

            if not savefile:
                msg = 'The file %s already exists '%(filename)
                msg += 'in the folder %s '%(filepath)
                msg += 'and will not be modified.'
                msg += '\nTo force the save use the parameter: resave=True\n'
                print(msg)

            return savefile

        #----------------------------------------------------------------------
        # loop to save all files in the data
        #----------------------------------------------------------------------
        for expname in data.keys():

            # creation of filepath or will save data
            #------------------------------------------------------------------
            sub, ses = None, None
            for f in expname.split('_'):
                if f.split('-')[0]=='sub':
                    sub = f.split('-')[1]
                elif f.split('-')[0]=='ses':
                    ses = f.split('-')[1]
            filepath = create_filepath(dirpath, sub, ses)
            #------------------------------------------------------------------

            for d in data[expname].keys():

                # save Data, Events, Results
                if d in ['Data', 'Events', 'Results']:

                    if d=='Data': filename = expname+'_data.pkl.gz'
                    elif d=='Events': filename = expname+'_events.pkl.gz'
                    elif d=='Results': filename = expname+'_results.pkl.gz'

                    savefile = check_save(filepath, filename)
                    if savefile:
                        save_file(data[expname][d], filename, filepath)


                # save Settings
                elif d=='Settings':

                    settings = data[expname][d].copy()

                    for s, ext in zip(['Data', 'Events', 'Results'],
                                      ['data', 'events', 'results']):

                        filename = expname+'_'+ext+'.json'
                        savefile = check_save(filepath,filename)
                        if savefile:
                            save_file(settings[s][0].T, filename, filepath)
                        settings.pop(s)

                    filename = expname+'_settings.json'

                    savefile = check_save(filepath, filename)
                    if savefile:
                        save_file(settings.T[0], filename, filepath)

                    del settings
        #----------------------------------------------------------------------
