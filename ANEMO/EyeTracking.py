#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pylink
import gc
import time
import os, sys
import numpy as np


######################################################################################
##################### FAIRE LA TRADUCTION !!!! #######################################
######################################################################################


spath = os.path.dirname(sys.argv[0])
if len(spath) !=0: os.chdir(spath)

#eyelinktracker = pylink.EyeLink()
#eyelink = pylink.getEYELINK()

Oeil_droit = 1
#Oeil_gauche = 0
#Binoculaire = 2

class EyeTracking(object):

    def __init__(self, screen_width_px, screen_height_px, dot_size, N_trials, observer, datadir, timeStr):

        self.eyelinktracker = pylink.EyeLink()
        self.eyelink = pylink.getEYELINK()

        self.screen_width_px = screen_width_px
        self.screen_height_px = screen_height_px
        self.dot_size = dot_size
        self.N_trials = N_trials
        self.mode = 'enregistrement'
        self.edfFileName = "%s.EDF"%(observer) # ne doit pas contenir plus de 8 caractères -- Must contains no more than 8 characters
        self.edfFileName_2 = os.path.join(datadir, self.mode + '_' + observer + '_' + timeStr + '.edf')





    #######################################################
    # Start + End TRIAL !!!
    #######################################################

    def End_trial(self):
        pylink.endRealTimeMode() # Fin mode temps réel
        pylink.pumpDelay(100) # ajout 100ms de donnée pour les évenement finaux
        self.eyelink.stopRecording()
        while self.eyelink.getkey() :
            pass

    def Start_trial(self, trial):
        # Titre essai en cours au bas de l'écran d'eyetracker
        print(trial)
        message = "record_status_message 'Trial %d/%d'" %(trial + 1, self.N_trials)
        self.eyelink.sendCommand(message)

        # EyeLink Data Viewer définit le début d'un essai par le message TRIALID.
        msg = "TRIALID %d" % trial
        self.eyelink.sendMessage(msg)
        msg = "!V TRIAL_VAR_DATA %d" % trial
        self.eyelink.sendMessage(msg)

        # Commutez le tracker sur ide et donnez-lui le temps d'effectuer un interrupteur de mode complet
        self.eyelink.setOfflineMode()
        pylink.msecDelay(50) 

        # Commencez à enregistrer des échantillons et des événements sur le fichier edf et sur le lien.
        error = self.eyelink.startRecording(1, 1, 1, 1) # 0 si tout se passe bien !
        if error :
            self.End_trial()
            print('error =', error)
            #return error

        #gc.disable()  # Désactiver la collecte python pour éviter les retards   TESTER ENLEVER !!!
        pylink.beginRealTimeMode(100)  # Commencer le mode temps réel

        # Lit et supprime les événements dans la file d'attente de données jusqu'à ce qu'ils soient dans un bloc d'enregistrement.
        try: 
            self.eyelink.waitForBlockStart(100,1,0) 
        except RuntimeError: 
            if pylink.getLastError()[0] == 0: # Temps d'attente expiré sans données de lien
                self.End_trial()
                self.eyelink.sendMessage("TRIAL ERROR")
                print ("ERROR: No link samples received!") 
                return pylink.TRIAL_ERROR 
            else:
                raise

    def Fixation(self, point, tps_start_fix, win, escape_possible) :

        # Tps de fixation à durée variable
        duree_fixation = np.random.uniform(0.4, 0.8) # durée du point de fixation (400-800 ms)
        tps_fixation = 0
        escape_possible(self.mode)
        # ---------------------------------------------------
        # Initialiser les données d'échantillons et les variables d'entrée de bouton
        nSData = None
        sData = None
        button = 0

        eye_used = self.eyelink.eyeAvailable() # Déterminez quel œil(s) est disponible
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
            nSData = self.eyelink.getNewestSample() # Vérifiez la nouvelle mise à jour de l'échantillon
            if(nSData != None and (sData == None or nSData.getTime() != sData.getTime())):
                sData = nSData

                escape_possible(self.mode)

                # Détectez si le nouvel échantillon a des données pour l'oeil en cours de suivi
                if eye_used == Oeil_droit and sData.isRightSample():
                    # Obtenir l'échantillon sous la forme d'une structure événementielle
                    gaze = sData.getRightEye().getGaze()
                    valid_gaze_pos = isinstance(gaze, (tuple, list))

                    if valid_gaze_pos : #Si les données sont valides, comparez la position du regard avec les limites de la fenêtre de tolérance
                        escape_possible(self.mode)
                        x_eye = gaze[0]
                        y_eye = gaze[1]
                        point.draw()
                        win.flip()

                        # Taille de la fenêtre de FIXATION
                        W_FW = self.dot_size + 120 # Largeur en pixels
                        H_FW = self.dot_size + 120 # Hauteur

                        diffx = abs(x_eye-self.screen_width_px/2) - W_FW/2
                        diffy = abs(y_eye-self.screen_height_px/2) - H_FW/2

                        if diffx>0 or diffy>0 :
                            escape_possible(self.mode)
                            win.flip()
                            tps_start_fix = time.time()

                    else : #Si les données sont invalides (par exemple, en cas de clignotement)
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

    #######################################################
    # Start + End EXP !!!
    #######################################################

    def Start_exp(self) :
        # ---------------------------------------------------
        # point de départ de l'expérience
        # ---------------------------------------------------
        pylink.openGraphics((self.screen_width_px, self.screen_height_px),32) # Initialise les graphiques
        self.eyelink.openDataFile(self.edfFileName) # Ouvre le fichier EDF.

        # réinitialise les touches et réglez le mode de suivi en mode hors connexion.
        pylink.flushGetkeyQueue()
        self.eyelink.setOfflineMode()

        # Définit le système de coordonnées d'affichage et envoie un message à cet effet au fichier EDF;
        self.eyelink.sendCommand("screen_pixel_coords =  0 0 %d %d" %(self.screen_width_px - 1, self.screen_height_px - 1))
        self.eyelink.sendMessage("DISPLAY_COORDS  0 0 %d %d" %(self.screen_width_px - 1, self.screen_height_px - 1))

        # ---------------------------------------------------
        # NETOYER ??? version = 3
        # ---------------------------------------------------
        tracker_software_ver = 0
        eyelink_ver = self.eyelink.getTrackerVersion()

        if eyelink_ver == 3:
            tvstr = self.eyelink.getTrackerVersionString()
            vindex = tvstr.find("EYELINK CL")
            tracker_software_ver = int(float(tvstr[(vindex + len("EYELINK CL")):].strip()))

        if eyelink_ver>=2:
            self.eyelink.sendCommand("select_parser_configuration 0")
            if eyelink_ver == 2: # Éteignez les caméras scenelink
                self.eyelink.sendCommand("scene_camera_gazemap = NO")
        else:
            self.eyelink.sendCommand("saccade_velocity_threshold = 35")
            self.eyelink.sendCommand("saccade_acceleration_threshold = 9500")

        # Définir le contenu du fichier EDF
        self.eyelink.sendCommand("file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT")
        if tracker_software_ver>=4:
            self.eyelink.sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS,HTARGET,INPUT")
        else:
            self.eyelink.sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS,INPUT")

        # Définir les données du lien (utilisé pour le curseur du regard)
        self.eyelink.sendCommand("link_event_filter = LEFT,RIGHT,FIXATION,FIXUPDATE,SACCADE,BLINK,BUTTON,INPUT")
        if tracker_software_ver>=4:
            self.eyelink.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,HTARGET,INPUT")
        else:
            self.eyelink.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT")

        #############################
        # Calibration
        #############################
        pylink.setCalibrationColors((255, 255, 255),(128, 128, 128)) # Définit couleur de la cible d'étalonnage (blanc) et de l'arrière-plan (gris)
        pylink.setTargetSize(self.screen_width_px//70, self.screen_width_px//300) # Définit taille de la cible d'étalonnage
        pylink.setCalibrationSounds("", "", "")
        pylink.setDriftCorrectSounds("", "off", "off")

    def End_exp(self):
        # Transfert et nettoyage de fichiers!
        self.eyelink.setOfflineMode()
        pylink.msecDelay(500) 

        # Fermez le fichier et transférez-le sur Display PC
        self.eyelink.closeDataFile()

        self.eyelink.receiveDataFile(self.edfFileName, self.edfFileName_2)
        self.eyelink.close()

        # Fermer les graphiques de l'expérience
        pylink.closeGraphics()

    def fin_enregistrement(self):
        #eyelink.sendMessage("TRIAL_RESULT %d" % button)
        ret_value = self.eyelink.getRecordingStatus() # état de l'enregistrement de sortie
        self.End_trial()

        #gc.enable() # Réactivez la collecte python pour nettoyer la mémoire à la fin de l'essai
        pylink.endRealTimeMode()
        return ret_value

    #######################################################
    # Check Calibration + Correction !!!
    #######################################################

    def check(self):
        if(not self.eyelink.isConnected() or self.eyelink.breakPressed()):
            self.End_trial()
            self.End_exp()
           #break

        self.eyelink.flushKeybuttons(0) # Réinitialise touches et boutons sur tracker
        error = self.eyelink.isRecording()  # Vérifiez d'abord si l'enregistrement est interrompu
        if error != 0:
            self.End_trial()
            print ('error =', error)

        if(self.eyelink.breakPressed()): # Vérifie fin du programme ou les touches ALT-F4 ou CTRL-C
            self.eyelink.sendMessage("EXPERIMENT ABORTED")
            print("EXPERIMENT ABORTED")
            self.End_trial()
            self.End_exp()

        elif(self.eyelink.escapePressed()): # Vérifier si escape pressé
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
        self.eyelink.doTrackerSetup() # configuration du suivi

    def drift_correction(self) :
        try:
            self.eyelink.doDriftCorrect((self.screen_width_px/2), (self.screen_height_px/2), 1, 1)
        except:
            self.eyelink.doTrackerSetup()
        pylink.msecDelay(50)


    #######################################################
    # Stimulus + Target !!!
    #######################################################

    def StimulusON(self, tps_start_fix):
        self.eyelink.sendMessage('StimulusOn')
        self.eyelink.sendMessage("%d DISPLAY ON" %tps_start_fix)
        self.eyelink.sendMessage("SYNCTIME %d" %tps_start_fix)
        self.eyelink.sendCommand("clear_screen 0")

        # ---------------------------------------------------
        # BOITE FIXATION
        # ---------------------------------------------------
        colour = 7 # blanc
        x = self.screen_width_px/2
        y = self.screen_height_px/2
        # Taille de la fenêtre de FIXATION
        W_FW = self.dot_size + 120 # Largeur en pixels
        H_FW = self.dot_size + 120 # Hauteur
        boite_fixation = [x-W_FW/2, y-H_FW/2, x+W_FW/2, y+H_FW/2]

        self.eyelink.sendCommand('draw_box %d %d %d %d %d'%(boite_fixation[0], boite_fixation[1], boite_fixation[2], boite_fixation[3], colour))

    def StimulusOFF(self) :
        self.eyelink.sendCommand("clear_screen 0") # Efface la boîte de l'écran Eyelink
        self.eyelink.sendMessage('StimulusOff')


    def TargetON(self) :
        self.eyelink.sendMessage('TargetOn')

        # ---------------------------------------------------
        # BOITE MOUVEMENT
        # ---------------------------------------------------
        colour = 7 # blanc
        x = self.screen_width_px/2
        y = self.screen_height_px/2
        # Taille de la fenêtre de MOUVEMENT
        W_MW = 2*(0.9*(self.screen_width_px/2)) # Largeur en pixels (égale à 2 * longueur déplacement cible)
        H_MW = 200 # Hauteur
        boite_mouvement = [x-W_MW/2, y-H_MW/2, x+W_MW/2, y+H_MW/2]

        self.eyelink.sendCommand('draw_box %d %d %d %d %d'%(boite_mouvement[0], boite_mouvement[1], boite_mouvement[2], boite_mouvement[3], colour))

    def TargetOFF(self) :
        self.eyelink.sendMessage('TargetOff')
        self.eyelink.sendCommand("clear_screen 0") # Efface la boîte de l'écran Eyelink

    #######################################################



