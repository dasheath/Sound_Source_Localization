# Sound_Source_Localization
Class Project

## TDOA
This project uses Time Difference of Arrival (TDOA) between pairs of sound detectors to determine the location of a sound source. This involves sampling audio levels at the sensors, and using signal cross correlation methods to find the delay in number of samples between the sound at each sensor. Then use knowledge of sampling rate and wave speed to determine the time difference of arrival between the two sensors. 

I am using a form 
where x is the sound source position vector (using a 2-D position) and a_i is the location of sensor i.


![TDOA \alpha h(x) = r_1 - r_2 = ||x - a_1|| - ||x - a_2||](<img src="https://latex.codecogs.com/svg.image?\bg_white&space;TDOA&space;\alpha&space;h(x)&space;=&space;r_1&space;-&space;r_2&space;=&space;||x-a_1||&space;-&space;||x-a_2||" title="\bg_white TDOA \alpha h(x) = r_1 - r_2 = ||x-a_1|| - ||x-a_2||" />)
