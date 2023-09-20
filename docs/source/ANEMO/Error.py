#!/usr/bin/env python
# -*- coding: utf-8 -*-

class fileError(Exception):

    '''
    Returns an error if the file not in dirpath
    '''

    def __init__(self, path, filenameprop, ext):

        self.message  = "There is no %s file in your directory."%ext
        if filenameprop:
            self.message += " for the properties requested:\n"
            for k in filenameprop.keys():
                if k!='ext' and filenameprop[k]!='':
                    self.message += k+': '+filenameprop[k]+', '
            self.message = self.message[:-2]
        self.message += '\n\n'+Data.dirtree(path, True)

    def __str__(self):
        return self.message

class subjectError(Exception):

    '''
    Returns an error if the subject not in dirpath
    '''

    def __init__(self, dirpath, sub):

        self.message  = "There is no subject %s "%sub
        self.message += "in the directory %s\n"%dirpath
        self.message += "\n"+Data.dirtree(dirpath, True)

    def __str__(self):
        return self.message

class ParamsError(Exception):

    '''
    Returns an error if the parameters of the function are not correct
    '''

    def __init__(self, arg, arg_fct):

        self.message = "The parameters given for the function to be fitted:\n"
        self.message += arg+"\n"
        self.message += "do not correspond to those expected:\n"
        self.message += arg_fct

    def __str__(self):
        return self.message

class CallError(Exception):

    '''
    Returns an error if the parameters of the function are not correct
    '''

    def __init__(self, NameClass):

        self.message = "%s was not correctly called \n"%NameClass
        self.message += "the correct way to call it is: "
        self.message += "\n\t%s(dirpath)"%NameClass

    def __str__(self):
        return self.message

class SettingsError(Exception):

    '''
    Returns an error if the Settings are not correct
    '''

    def __init__(self, variable, expname):
        self.message = "The settings for the %s experiment "%expname
        self.message += "do not include the variable %s\n"%variable
        self.message += "you must complete your settings file "
        self.message += "with this variable before continuing"

    def __str__(self):
        return self.message
