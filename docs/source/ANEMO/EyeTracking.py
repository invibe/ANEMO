#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    import pylink
except:
    pass
import gc
import time
import os, sys
import numpy as np


#  TODO: check why the above lines were used...
# spath = os.path.dirname(sys.argv[0])
# if len(spath) !=0: os.chdir(spath)

#eyelinktracker = pylink.EyeLink()
#eyelink = pylink.getEYELINK()

Oeil_droit = 1
#Oeil_gauche = 0
#Binoculaire = 2

class EyeTracking(object):

    def __init__(self, screen_width_px, screen_height_px, dot_size, N_trials,
                 observer, datadir, timeStr):

        self.eyelinktracker = pylink.EyeLink()
        self.eyelink = pylink.getEYELINK()

        self.screen_width_px = screen_width_px
        self.screen_height_px = screen_height_px
        self.dot_size = dot_size
        self.N_trials = N_trials
        self.mode = 'enregistrement'

        # Must contains no more than 8 characters
        self.edfFileName = "%s.EDF"%(observer)
        filename = self.mode + '_' + observer + '_' + timeStr + '.edf'
        self.edfFileName_2 = os.path.join(datadir, filename)





    ###########################################################################
    # Start + End TRIAL !!!
    ###########################################################################

    def End_trial(self):
        pylink.endRealTimeMode() # End of real time mode
        pylink.pumpDelay(100) # add 100ms of data for the final events
        self.eyelink.stopRecording()
        while self.eyelink.getkey() :
            pass

    def Start_trial(self, trial):
        # displays the title of the current test at the bottom of the
        # eyetracker screen
        print(trial)

        msg = "record_status_message 'Trial %d/%d'" %(trial+1, self.N_trials)
        self.eyelink.sendCommand(msg)

        # EyeLink Data Viewer defines the start of a trial with the message
        # TRIALID.
        msg = "TRIALID %d" % trial
        self.eyelink.sendMessage(msg)
        msg = "!V TRIAL_VAR_DATA %d" % trial
        self.eyelink.sendMessage(msg)

        # Switch the tracker to ide and give it time to perform a full mode
        # switch
        self.eyelink.setOfflineMode()
        pylink.msecDelay(50)

        # Start recording samples and events on the edf file and on the link.
        error = self.eyelink.startRecording(1, 1, 1, 1) # 0 if ok
        if error :
            self.End_trial()
            print('error =', error)
            #return error

        #gc.disable() # Disable python collection to avoid delays ->TEST REMOVE
        pylink.beginRealTimeMode(100)  # Start the real time mode

        # Reads and deletes events in the data queue until they are in a record
        # block.
        try:
            self.eyelink.waitForBlockStart(100,1,0)
        except RuntimeError:
            # Waiting time expired without link data
            if pylink.getLastError()[0] == 0:
                self.End_trial()
                self.eyelink.sendMessage("TRIAL ERROR")
                print ("ERROR: No link samples received!")
                return pylink.TRIAL_ERROR
            else:
                raise

    def Fixation(self, point, tps_start_fix, win, escape_possible) :

        # Variable fixation time

        # fixation point duration (400-800 ms)
        duree_fixation = np.random.uniform(0.4, 0.8)
        tps_fixation = 0
        escape_possible(self.mode)
        # ---------------------------------------------------------------------
        # Initialize sample data and button input variables
        nSData = None
        sData = None
        button = 0

        # Determine which eye(s) are available
        eye_used = self.eyelink.eyeAvailable()
        if eye_used == Oeil_droit:
            self.eyelink.sendMessage("EYE_USED 1 RIGHT")
        else:
            print ("Error in getting the eye information!")
            self.End_trial()
            return pylink.TRIAL_ERROR


        while (tps_fixation < duree_fixation) :
            escape_possible(self.mode)
            tps_actuel = time.time()
            tps_fixation = tps_actuel - tps_start_fix
            # Check the new sample update
            nSData = self.eyelink.getNewestSample()
            if(nSData != None
               and (sData == None or nSData.getTime() != sData.getTime())):
                sData = nSData

                escape_possible(self.mode)

                # Detect if the new sample has data for the eye being monitored
                if eye_used == Oeil_droit and sData.isRightSample():
                    # Get the sample in the form of an event structure
                    gaze = sData.getRightEye().getGaze()
                    valid_gaze_pos = isinstance(gaze, (tuple, list))

                    # If the data is valid, compare the gaze position with the
                    # limits of the tolerance window
                    if valid_gaze_pos :
                        escape_possible(self.mode)
                        x_eye = gaze[0]
                        y_eye = gaze[1]
                        point.draw()
                        win.flip()

                        # Size of the FIXATION window
                        W_FW = self.dot_size + 120 # Width in pixels
                        H_FW = self.dot_size + 120 # Height in pixels

                        diffx = abs(x_eye-self.screen_width_px/2) - W_FW/2
                        diffy = abs(y_eye-self.screen_height_px/2) - H_FW/2

                        if diffx>0 or diffy>0 :
                            escape_possible(self.mode)
                            win.flip()
                            tps_start_fix = time.time()

                    else : # If the data is invalid (e.g. blinking)
                        escape_possible(self.mode)
                        win.flip()
                        point.draw()
                        win.flip()
                        core.wait(0.1)
                        tps_start_fix = time.time()

            else :
                escape_possible(self.mode)
                error = self.eyelink.isRecording()
                if(error != 0) :
                    core.wait(0.1)
                tps_start_fix = time.time()



    ###########################################################################
    # Start + End EXP !!!
    ###########################################################################

    def Start_exp(self) :
        # ---------------------------------------------------------------------
        # starting point of the experiment
        # ---------------------------------------------------------------------
        # Initialize graphics
        pylink.openGraphics((self.screen_width_px, self.screen_height_px),32)
        self.eyelink.openDataFile(self.edfFileName) # Open the EDF file.

        # Reset the keys and set the tracking mode to offline.
        pylink.flushGetkeyQueue()
        self.eyelink.setOfflineMode()

        # Sets the display coordinate system and sends a message to that effect
        # to the EDF file
        command = "screen_pixel_coords =  0 0 %d %d"%(self.screen_width_px-1,
                                                      self.screen_height_px-1)
        self.eyelink.sendCommand(command)

        msg = "DISPLAY_COORDS  0 0 %d %d" %(self.screen_width_px-1,
                                            self.screen_height_px-1)
        self.eyelink.sendMessage(msg)

        # ---------------------------------------------------------------------
        # CLEANING ??? version = 3
        # ---------------------------------------------------------------------
        software_v = 0 # tracker software version
        eyelink_v = self.eyelink.getTrackerVersion()

        if eyelink_v==3:
            tvstr = self.eyelink.getTrackerVersionString()
            vindex = tvstr.find("EYELINK CL")
            software_v = int(float(tvstr[(vindex+len("EYELINK CL")):].strip()))

        if eyelink_v>=2:
            self.eyelink.sendCommand("select_parser_configuration 0")
            if eyelink_v==2: # Turn off the scenelink cameras
                self.eyelink.sendCommand("scene_camera_gazemap = NO")
        else:
            self.eyelink.sendCommand("saccade_velocity_threshold = 35")
            self.eyelink.sendCommand("saccade_acceleration_threshold = 9500")

        # Define the content of the EDF file
        self.eyelink.sendCommand("file_event_filter = LEFT,RIGHT,FIXATION,"
                                 "SACCADE,BLINK,MESSAGE,BUTTON,INPUT")
        if software_v>=4:
            self.eyelink.sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,"
                                     "AREA,GAZERES,STATUS,HTARGET,INPUT")
        else:
            self.eyelink.sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,"
                                     "AREA,GAZERES,STATUS,INPUT")

        # Define link data (used for the eye cursor)
        self.eyelink.sendCommand("link_event_filter = LEFT,RIGHT,FIXATION,"
                                 "FIXUPDATE,SACCADE,BLINK,BUTTON,INPUT")
        if software_v>=4:
            self.eyelink.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,"
                                     "GAZERES,AREA,STATUS,HTARGET,INPUT")
        else:
            self.eyelink.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,"
                                     "GAZERES,AREA,STATUS,INPUT")

        #######################################################################
        # Calibration
        #######################################################################
        # Defines color of calibration target (white) and background (gray)
        pylink.setCalibrationColors((255, 255, 255),(128, 128, 128))
        # Defines the size of the calibration target
        pylink.setTargetSize(self.screen_width_px//70,
                             self.screen_width_px//300)
        pylink.setCalibrationSounds("", "", "")
        pylink.setDriftCorrectSounds("", "off", "off")

    def End_exp(self):
        # File transfer and cleaning!
        self.eyelink.setOfflineMode()
        pylink.msecDelay(500)

        # Close the file and transfer it to Display PC
        self.eyelink.closeDataFile()

        self.eyelink.receiveDataFile(self.edfFileName, self.edfFileName_2)
        self.eyelink.close()

        # Close the graphs of the experiment
        pylink.closeGraphics()

    def fin_enregistrement(self):
        #eyelink.sendMessage("TRIAL_RESULT %d" % button)
        # status of the output record
        ret_value = self.eyelink.getRecordingStatus()
        self.End_trial()

        # Réactivez la collecte python pour nettoyer la mémoire à la fin de
        # l'essai
        #gc.enable()

        pylink.endRealTimeMode()
        return ret_value

    ###########################################################################
    # Check Calibration + Correction !!!
    ###########################################################################

    def check(self):
        if(not self.eyelink.isConnected() or self.eyelink.breakPressed()):
            self.End_trial()
            self.End_exp()
           #break

        # Reset keys and buttons on tracker
        self.eyelink.flushKeybuttons(0)
        # First check if the recording is interrupted
        error = self.eyelink.isRecording()
        if error != 0:
            self.End_trial()
            print ('error =', error)

        # Checks the end of the program or the ALT-F4 or CTRL-C keys
        if(self.eyelink.breakPressed()):
            self.eyelink.sendMessage("EXPERIMENT ABORTED")
            print("EXPERIMENT ABORTED")
            self.End_trial()
            self.End_exp()

        elif(self.eyelink.escapePressed()): # Check if escape pressed
            self.eyelink.sendMessage("TRIAL ABORTED")
            print("TRIAL ABORTED")
            self.End_trial()

    def check_trial(self, ret_value) :
        if (ret_value == pylink.TRIAL_OK):
            self.eyelink.sendMessage("TRIAL OK")
            print("TRIAL OK")
            #break
        elif (ret_value == pylink.SKIP_TRIAL):
            self.eyelink.sendMessage("TRIAL ABORTED")
            print("TRIAL ABORTED")
            #break
        elif (ret_value == pylink.ABORT_EXPT):
            self.eyelink.sendMessage("EXPERIMENT ABORTED")
            print("EXPERIMENT ABORTED")
        elif (ret_value == pylink.REPEAT_TRIAL):
            self.eyelink.sendMessage("TRIAL REPEATED")
            print("TRIAL REPEATED")
        else:
            self.eyelink.sendMessage("TRIAL ERROR")
            print("TRIAL ERROR")
            #break

    def calibration(self) :
        self.eyelink.doTrackerSetup() # tracking configuration

    def drift_correction(self) :
        try:
            self.eyelink.doDriftCorrect((self.screen_width_px/2),
                                        (self.screen_height_px/2), 1, 1)
        except:
            self.eyelink.doTrackerSetup()
        pylink.msecDelay(50)


    ###########################################################################
    # Stimulus + Target !!!
    ###########################################################################

    def StimulusON(self, tps_start_fix):
        self.eyelink.sendMessage('StimulusOn')
        self.eyelink.sendMessage("%d DISPLAY ON" %tps_start_fix)
        self.eyelink.sendMessage("SYNCTIME %d" %tps_start_fix)
        self.eyelink.sendCommand("clear_screen 0")

        # ---------------------------------------------------------------------
        # FIXING BOX
        # ---------------------------------------------------------------------
        colour = 7 # white
        x = self.screen_width_px/2
        y = self.screen_height_px/2
        # Size of the FIXATION window
        W_FW = self.dot_size + 120 # Width in pixels
        H_FW = self.dot_size + 120 # Height in pixels
        boite_fixation = [x-W_FW/2, y-H_FW/2, x+W_FW/2, y+H_FW/2]

        self.eyelink.sendCommand('draw_box %d %d %d %d %d'%(boite_fixation[0],
                                                            boite_fixation[1],
                                                            boite_fixation[2],
                                                            boite_fixation[3],
                                                            colour))

    def StimulusOFF(self) :
        # Clears the Eyelink screen box
        self.eyelink.sendCommand("clear_screen 0")

        self.eyelink.sendMessage('StimulusOff')


    def TargetON(self) :
        self.eyelink.sendMessage('TargetOn')

        # ---------------------------------------------------------------------
        # MOTION BOX
        # ---------------------------------------------------------------------
        colour = 7 # white
        x = self.screen_width_px/2
        y = self.screen_height_px/2
        # Size of the MOVEMENT window

        #  Width in pixels (equal to 2 * target displacement length)
        W_MW = 2*(0.9*(self.screen_width_px/2))
        # Height in pixels
        H_MW = 200
        boite_mouvement = [x-W_MW/2, y-H_MW/2, x+W_MW/2, y+H_MW/2]

        self.eyelink.sendCommand('draw_box %d %d %d %d %d'%(boite_mouvement[0],
                                                            boite_mouvement[1],
                                                            boite_mouvement[2],
                                                            boite_mouvement[3],
                                                            colour))

    def TargetOFF(self) :
        self.eyelink.sendMessage('TargetOff')
        # Clears the Eyelink screen box
        self.eyelink.sendCommand("clear_screen 0")

    ###########################################################################
