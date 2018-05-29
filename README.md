### k-Beam Minimax Optimization
#### For faster and more stable training of GANs and other adversarial optimization problems such domain adaptation, privacy preservation, and robust learning. 
---

![concept figure](minimaxfilter2.jpg "Example minimax filter")

#### Abstract

Minimax optimization plays a key role in adversarial training of machine learning algorithms, such as learning generative models, domain adaptation, privacy preservation, and robust learning. 
In this paper, we demonstrate the failure of alternating gradient descent in minimax optimization problems due to the discontinuity of solutions of the inner maximization. 
To address this, we propose a new epsilon-subgradient descent algorithm that addresses this problem by simultaneously tracking K candidate solutions. 
Practically, the algorithm can find solutions that previous saddle-point algorithms cannot find, with only a sublinear increase of complexity in K.
We analyze the conditions under which the algorithm converges to the true solution in detail. 
A significant improvement in stability and convergence speed of the algorithm is observed in simple representative problems, GAN training, and domain-adaptation problems.

### Requirements
---
Python 2, Scipy, Tensorflow, Keras,
Download from ...

### Examples
---
#### 1. Run [test_minimax_simple.py](test_minimax_simple.py)
This script demonstrates the use of the k-beam minimax (and a few other optimization algorithms) in pure python.
The simple 2D surfaces are defined in the minimax_examples.py file.
If you run this script, you will get ....


#### 2. Run [test_minimax_mog_long.py](test_minimax_mog_long.py) 

The task is to learn a filter of face image from the Genki dataset which allows accurate classification of 'smile' vs 'non-smile' but prevents accurate classification of 'male' vs 'female'. 

The script finds a minimax filter by alternating optimization. The filer is a two-layer sigmoid neural net and the classifiers are softmax classifiers. 

The script will run for a few minutes on a desktop. 
After 50 iterations, the filter will achieve ~88% accuracy in facial expression classification and ~66% accuracy in gender classification.
```
minimax-NN: rho=10.000000, d=10, trial=0, rate1=0.88, rate2=0.66
```
Results will be save to a file named 'test_NN_genki.npz'

#### 3. Run [test/test_all_genki.py](test/test_all_genki.py)
The task is the same as before (accurate facial expression and inaccurate gender classification.)

The script trains and compares several private and non-private algorithms for the same task, including a linear minimax filter.

The script will also run for a few minutes on a desktop. 
You will see similar results as follows.
```
rand: d=10, trial=0, rate1=0.705000, rate2=0.705000

pca: d=10, trial=0, rate1=0.840000, rate2=0.665000

pls: d=10, trial=0, rate1=0.850000, rate2=0.685000

alt: rho=10.000000, d=10, trial=0, rate1=0.825000, rate2=0.520000
```
Here 'alt' is the linear minimax filter, 'rate1' is the accuracy of expression classification and 'rate 2' is the accuracy of gender classification.


### Reference
---
* [J. Hamm and Yung-Kyun Noh, "K-Beam Subgradient Descent for Minimax Optimization," 
 ICML, 2018]()
* [J. Hamm, "Mimimax Filter: Learning to Preserve Privacy from Inference Attacks," arXiv:1610.03577, 2016](http://arxiv.org/abs/1610.03577)






