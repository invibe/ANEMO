.. ANEMO documentation master file, created by
   sphinx-quickstart on Fri Feb 15 11:23:18 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**ANEMO**: a tools for the ANalysis of Eye Movements
====================================================

ANEMO is a python package for preprocessing and processing your eye data.

Before starting, it is important to initialize your data with the :doc:`1__init` function.

As a first step, it is recommended to pre-process your data using :doc:`3.0__PreProcessing`. This will allow you to transform your raw data initially into pixels in :doc:`degrees <3.1__PreProcessing_to_deg>` or :doc:`degrees per second <3.2__PreProcessing_to_velocity_deg>`, to :doc:`filter <3.3__PreProcessing_filter>` them or to :doc:`remove <3.4__PreProcessing_remove_events>` or :doc:`extract <3.5__PreProcessing_extract_events>` your data between two events.

You will be able to process your saccade data in a second time with :doc:`4.0__ProcessingSaccades`. In particular by using the :doc:`4.1__ProcessingSaccades_detect_misacc` function which will allow you to detect saccades and micro-saccades in your data, which were not previously detected by your eye-tracker. You can then :doc:`remove <4.2__ProcessingSaccades_remove_saccades>` them from your data or :doc:`extract <4.3__ProcessingSaccades_extract_saccades>` them to fit the parameters of a :doc:`saccade model <4.4.2__ProcessingSaccades__Model>` to your data.

If your data correspond to smooth pursuit eye data, you can use :doc:`5.0__ProcessingSmoothPursuit` to extract different parameters. In particular the anticipation velocity, the pursuit latency or the steady state velocity of the pursuit by using :doc:`classical methods <5.1.0__ProcessingSmoothPursuit__classical_method>` or by :doc:`fitting <5.2.0__ProcessingSmoothPursuit__Fit>` parameters of different smooth pursuit models. During this step, it is **STRONGLY** recommended to use data where saccades and microsaccades have been removed.

You can find all the demo codes in the :doc:`DEMO` section.


Summary
--------
- :doc:`1__init` - allows to initialize the file which will contain the results of ANEMO
- :doc:`2__Data` - sample code to access the results
- :doc:`3.0__PreProcessing`- allows certain calculations to be applied to the data in order to transform it
    - :doc:`3.1__PreProcessing_to_deg` - transforms position data into degrees
    - :doc:`3.2__PreProcessing_to_velocity_deg` - transforms position data into degrees per second
    - :doc:`3.3__PreProcessing_filter` - allows to filter the data
    - :doc:`3.4__PreProcessing_remove_events` - allows you to remove events from the data
    - :doc:`3.5__PreProcessing_extract_events` - allows to extract events from data
- :doc:`4.0__ProcessingSaccades` -  allows you to apply certain calculations to the data in order to extract parameters
    - :doc:`4.1__ProcessingSaccades_detect_misacc` - detect micro-saccades
    - :doc:`4.2__ProcessingSaccades_remove_saccades` - sample code to remove saccade and micro-saccades
    - :doc:`4.3__ProcessingSaccades_extract_saccades` - sample code to extract saccade
    - :doc:`4.4.0__ProcessingSaccades__Fit` - fits the parameters of a model to the eye data.
       - :doc:`4.4.1__ProcessingSaccades__Fit_function` - function to fit the parameters of a model to the eye data.
       - :doc:`4.4.2__ProcessingSaccades__Model` - predefined model reproducing the position of the eye during the saccades
       - :doc:`4.4.3__ProcessingSaccades__GenerateParams` - generate automatically the parameters of the predefined models in :doc:`4.4.2__ProcessingSaccades__Model` in order to fit them to the data.
       - :doc:`4.4.4__ProcessingSaccades__Fit_example_of_a_user-defined_model` - exemple of fit the parameters of a user-defined model to the eye data.
- :doc:`5.0__ProcessingSmoothPursuit` -  allows you to apply certain calculations to the data in order to extract parameters
    - :doc:`5.1.0__ProcessingSmoothPursuit__classical_method` - the "classical method" allowing the extraction of parameters
        - :doc:`5.1.1__ProcessingSmoothPursuit__classical_method_anticipation` - the "classical method" allowing the extraction of the velocity of anticipation of pursuit
        - :doc:`5.1.2__ProcessingSmoothPursuit__classical_method_latency` - the "classical method" allowing the extraction of the pursuit latency
        - :doc:`5.1.3__ProcessingSmoothPursuit__classical_method_steady_state` - the "classical method" allowing the extraction of the steady state velocity
    - :doc:`5.2.0__ProcessingSmoothPursuit__Fit` - fits the parameters of a model to the eye data.
       - :doc:`5.2.1__ProcessingSmoothPursuit__Fit_function` - function to fit the parameters of a model to the eye data.
       - :doc:`5.2.2.0__ProcessingSmoothPursuit__Model` - predefined model to fit the data
           - :doc:`5.2.2.1__ProcessingSmoothPursuit__Model_velocity_line` - model reproducing the velocity of the eye during the smooth pursuit of a moving target
           - :doc:`5.2.2.2__ProcessingSmoothPursuit__Model_velocity` - model reproducing the velocity of the eye during the smooth pursuit of a moving target
           - :doc:`5.2.2.3__ProcessingSmoothPursuit__Model_velocity_sigmo` - model reproducing the velocity of the eye during the smooth pursuit of a moving target
           - :doc:`5.2.2.4__ProcessingSmoothPursuit__Model_position` - model reproducing the position of the eye during the smooth pursuit of a moving target
       - :doc:`5.2.3__ProcessingSmoothPursuit__GenerateParams` - generate automatically the parameters of the predefined models in :doc:`5.2.2.0__ProcessingSmoothPursuit__Model` in order to fit them to the data.
       - :doc:`5.2.4__ProcessingSmoothPursuit__Fit_example_of_a_user-defined_model` - exemple of fit the parameters of a user-defined model to the eye data.

:doc:`DEMO` - code set allowing the extraction of smooth pursuit parameters with the "classical method" and the fit of the velocity_sigmo function to your velocity data starting from your raw data

.. toctree::
   :hidden:
   
   1__init
   2__Data
   3.0__PreProcessing
   4.0__ProcessingSaccades
   5.0__ProcessingSmoothPursuit
   DEMO
