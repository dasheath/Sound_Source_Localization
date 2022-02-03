# Sound_Source_Localization
Class Project

## TDOA
This project uses Time Difference of Arrival (TDOA) between pairs of sound detectors to determine the location of a sound source. This involves sampling audio levels at the sensors, and using signal cross correlation methods to find the delay in number of samples between the sound at each sensor. Then use knowledge of sampling rate and wave speed to determine the time difference of arrival between the two sensors. 

I am using a form 
$$TDOA \alpha h(x) = r_1 - r_2 = ||x-a_1|| - ||x-a_2||$$
where x is the sound source position vector (using a 2-D position) and $a_i$ is the location of sensor $i$.

![\sum_{\forall i}{x_i^{2}}](https://latex.codecogs.com/svg.latex?%5Csum_%7B%5Cforall+i%7D%7Bx_i%5E%7B2%7D%7D)
![TDOA \alpha h(x) = r_1 - r_2 = ||x - a_1|| - ||x - a_2||](https://latex.codecogs.com/svg.image?TDOA%20%5Calpha%20h(x)%20=%20r_1%20-%20r_2%20=%20%7C%7Cx-a_1%7C%7C%20-%20%7C%7Cx-a_2%7C%7C)
