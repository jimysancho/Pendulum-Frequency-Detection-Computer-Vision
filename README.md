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

https://user-images.githubusercontent.com/105709376/208385354-4c7ce6a2-2bcb-4025-b595-3ee5ef340c95.mov

