# drift_analysis
An interative Python/Matplotlib tool for analysing subpulse drifting in pulsars

## Install
This needs the following Python packages to be installed on your system:
 - numpy
 - scipy
 - matplotlib
 - tk

## Run
If starting a new analysis, this can be run on a "pdv" file (i.e. a text file containing the output of PSRCHIVE's pdv utility) as follows:

    python drift_analysis.py <pdv_file> <stokes>
    
where "stokes" is one of (I,Q,U,V)

The program uses the JSON format to save out the analyses to file. If such a file has been saved, it can be loaded directly from the command line using

    python drift_analysis.py <json_file>
