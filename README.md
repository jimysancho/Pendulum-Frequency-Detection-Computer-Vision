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
X_{cm} = \frac{1}{M} \sum_{i}^{N} x_i m_i
$$

$$
Y_{cm} \frac{1}{M} \sum_{i}^{M} y_i m_i
$$

to the nice round binary pendulum, we get the CM of the pendulum. We will store this data in a `csv` file to begin the bayesian analysis. 

https://user-images.githubusercontent.com/105709376/208389072-8fd761c6-9df2-4ff7-9b0c-e3a9aef2edb7.mov
