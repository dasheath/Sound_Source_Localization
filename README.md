# Sound_Source_Localization
- This was a class project for EN.525.744 Passive Emitter Geo-Location at Johns Hopkins University
- Please see PDF presentation in this repo.
- This repo also contains a streamlit app that is a re-implementation of the software part of that project.

## TDOA
The actual class project had multiple components. 
- The first is an arduino board connected to 3 microphones which were sampling loudness values and transmitting buffers of data for each sensor to a laptop. 
- I then used cross-correlation to determine the time difference of arrival between the sensors. Essentially, knowing how often samples were being taken I could detect that a shift of `N` samples from one microphone to another is then `N * sampling rate` seconds of TDOA.
- I then used the speed of sound to convert those into distances and then derived my "measurements" `z_i` which are 'distance from sensor i to sensor 0' for i in [1, number of sensors]. This results in one measurement for each sensor pair (i,0).


## Iterated Least Squares
This project uses Iterated Least Squares approach to solve for speaker position from an initial guess and measurements.

Please see the presentation PDF and the 'run_ils' method in the src directory for the details of this. 

## The Streamlit app

The streamlit app defined in `main.py` lets the user input sensor positions (where the microphones would be) and a source location where sound is being emitted.

I then calculate what TDOA measurements `z` would be generated, and use those as inputs with the sensor positions into the iterated least squares algorithm to estimate the source position.

The plot at the bottom shows the TDOA hyperbolas for each sensor pair (i,0), the sensor locations, and finally the estimate resultant from the ILS algorithm.

## Note on accuracy
Users will notice that there are certain favorable and less-favorable tdoa geometries. Not (yet) shown on this plot is the error ellipse probable. That is to say the ellipse that we can be "very sure" the actual source location is inside. This ellipse can be close to a cirle for certain sensor laydowns, or very stretched out for others.

Feel free to experiment with the inputs and then zoom the plot way in to see how moving the source position and sensor laydown affects the accuracy of the estimated source position.
