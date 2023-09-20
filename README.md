[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/invibe/ANEMO/master)

# ANEMO: quantitative tools for the ANalysis of Eye MOvements

Eye movements are crucial bio-markers for a wide range of cognitive behaviours. While the recordings of such movements may be provided by low- to high-cost measurement devices, there is no unique, commonly agreed method to quantify the different phases of their dynamics. Here we focus on eye movements performed during motion tracking. Based on some prior knowledge on the dynamics of the different types of eye movements, we propose here a set of robust fitting methods for the extraction of characteristic parameters of eye movements.In particular, we show how we can robustly extract the latency, initial acceleration and steady state of visually-guided smooth pursuit eye movements, as well as the velocity ramp of anticipatory pursuit. Compared with classical methods based on local linear regression, for pursuit latency, and velocity thresholding for saccade detection, this method provides a more efficient tool for validating and categorizing tracking performance globally. We validated it on a large set of experimental data. Moreover, this code is made available as an open-source package at http://github.com/invibe/ANEMO, allowing for the community to use and modify these methods.

<center><img src="https://raw.githubusercontent.com/invibe/ANEMO/master/2018-05-04_Poster_Grenoble/figures/Fit_simulation.png" width="100%"></center>

 * see a poster presentation @ https://laurentperrinet.github.io/publication/pasturel-18-anemo/
 * report [Issues on GtiHub](https://github.com/invibe/ANEMO/issues) 
 * maintainer: Laurent Perrinet <laurent.perrinet@univ-amu.fr>


````

@inproceedings{Pasturel18anemo,
 author = {Pasturel, Chloé and Montagnini, Anna and Perrinet, Laurent U},
 bdsk-url-1 = {https://laurentperrinet.github.io/publication/pasturel-18-anemo},
 booktitle = {Grenoble Workshop on Models and Analysis of Eye Movements, Grenoble, France},
 date-added = {2019-02-25 23:41:40 +0100},
 date-modified = {2020-03-31 11:07:30 +0200},
 grants = {pace-itn},
 keywords = {motion anticipation},
 projects = {open-science},
 title = {ANEMO: Quantitative tools for the ANalysis of Eye MOvements},
 url = {https://laurentperrinet.github.io/publication/pasturel-18-anemo},
 url_hal = {https://hal.science/hal-04157003},
 year = {2018}
}
      
````

## installation

First install dependencies:

```
pip install -U -r requirements.txt
```

Then, the library:
```
pip install -e  git+https://github.com/invibe/ANEMO#egg=ANEMO
```


---
This work was supported by the Innovative Training Network "Perception and Action in Complex Environments" (PACE ITN), a Marie Skodowska-Curie program of the H2020 European Union program (grant agreement No 642961).
