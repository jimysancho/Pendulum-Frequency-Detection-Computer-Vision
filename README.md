# Pendulum-Frequency-Detection-Computer-Vision

## 1. Techniques applied

We are trying to obtain the frequency of a pendulum using computer vision. In order to do that 
we are going to use the following techniques:

1. Computer vision. With the **`OpenCV`** library we will apply some common computer vision techniques 
to obtain the position of the pendulum's center of mass. 

2. **`Bayesian Analysis`**. Once we have observed the data, we will apply a Bayesian approach 
to obtain the frequency and amplitudes probabilities distributions of the pendulum using the Bayes Theorem: 

$$
P(\theta | \mathbf{X}) \propto p(\theta | \mathbf{X}) p(\theta)
$$

where $\theta = \{w, A, B\}$. Then using marginalization we will obtain the probability 
distribution of the frequency: 

$$
P(w | \mathbf{X}) \propto \int P(\theta | \mathbf{X}) dA dB
$$

## 2. Center of mass detection

This process can be found in the `code/pendulum_cm.py` file. 

### 2.1 Tracking

The very first thing we need to do is follow the movement of the pendulum. To do that the tracker we will use is the `KCF tracker`. For more information of the subject check: 
https://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf

The result can be seen in the following video:

https://user-images.githubusercontent.com/105709376/208386560-a7118d81-064b-458c-9247-013e46bea1ef.mov


### 2.2 Segmentation

One option to begin the Bayesian analysis could be to use the position of the window itself. Instead of that, 
we are going to get a nice figure of the pendulum to obtain the position of the center of mass. This way the data will be much precise so we will diminish as much as possible the reductible error we may encouter along the process. 


We begin using the Otsu threshold method to get the first glance of the pendulum itself. 

https://user-images.githubusercontent.com/105709376/208387793-a9092f2f-37ba-4ae4-844f-f3ea13d9a953.mov

### 2.3 Morphological operations

To really get a nice round figure of the pendulum, we will apply dilation and erosion operations. 

https://user-images.githubusercontent.com/105709376/208388446-4699dcdb-e62f-4b9e-84f6-aa54248bd33d.mov


### 2.4 Center of mass

Applying the classical discrete formulas to get the center of mass of a physical system: 

$$
X_{cm} = \frac{1}{M} \sum_{i=1}^{N} x_i m_i
$$

$$
Y_{cm} \frac{1}{M} \sum_{i=1}^{M} y_i m_i
$$

(where M is the total mass of the system)
to the nice round binary pendulum, we get the CM of the pendulum. We will store this data in a `csv` file to begin the bayesian analysis.

https://user-images.githubusercontent.com/105709376/208389072-8fd761c6-9df2-4ff7-9b0c-e3a9aef2edb7.mov

The data can be obtained in the file `data/pendulum-CM.csv`. 

## 3. Bayesian Analyisis

Finally, we can obtain the frequency of the pendulum. The code to apply bayesian analysis is the `code/utils.py` file. The analysis is made to simulate a `live analysis`. In order words, it is made to simulate that the measure of the center of mass position and the computation of the frequency probability distribution are made at the same time. 

With this file we obtain the probability distribution of the amplitudes of the pendulum, and the quantity of interest: the frequency. Marginalizing the get the frequency probability distribution. 

The final result: 

https://user-images.githubusercontent.com/105709376/208393140-1a1d61b3-9dec-4015-9d49-802155ef36ec.mov

And after some time: 

https://user-images.githubusercontent.com/105709376/208393175-8a421ce8-e6c2-4211-80dc-c504e011c228.mov

As we can see the future points and the actual points are predicted / calculated as perfect as a measure can be. With some uncertainty (the basic principle of the Bayesian approach) but very very little. 

Under the hood what is happening is that we are computing the frequency probability distribution and using the frequency that maximizes the probability to compute future and actual points (the same applies to the amplitudes A and B).  

https://user-images.githubusercontent.com/105709376/208393946-c99bd7ac-2d7f-4477-a50f-5d1a341aa70e.mov

