
# coding: utf-8

# ## Demo
- The parameters of the experiment see : <a href="ANEMO_doc.html">doc</a>
# In[1]:

import numpy as np

screen_width_px = 1280 # px
screen_width_cm = 36   # cm
viewingDistance = 57.  # cm

screen_width_deg = 2. * np.arctan((screen_width_cm/2) / viewingDistance) * 180/np.pi


param_exp = {'N_trials' : 2,
             'N_blocks' : 2,
             'dir_target' : [[-1, 1], [1, -1]],
             'px_per_deg' : screen_width_px / screen_width_deg,
              }


# - Retrieving the data

# In[2]:

from ANEMO import read_edf


# In[3]:

# path to the file that has to be read
datafile = 'data/enregistrement_AM_2017-10-23_100057.asc'

# trial start string
start = 'TRIALID'

data = read_edf(datafile, start)

print(data[0])

