# kbeam
### Minimax Filter
#### can preserve privacy of images, audios, or biometric data by making it difficult for an adversary to infer sensitive or identifying information from those data after filtering.
---

![concept figure](minimaxfilter2.jpg "Example minimax filter")

#### Abstract

Preserving privacy of continuous and/or high-dimensional data such as images, videos
and audios, can be challenging with syntactic anonymization methods such as k-anonymity
which are designed for discrete attributes. Differential privacy, which provides a different
and more formal type of privacy, has shown more success in sanitizing continuous data.
However, both syntactic and differential privacy are susceptible to inference attacks, i.e., an
adversary can accurately guess sensitive attributes from insensitive attributes. This paper
proposes a learning approach to finding a minimax filter of raw features which retains infor-
mation for target tasks but removes information from which an adversary can infer sensitive
attributes. Privacy and utility of filtered data are measured by expected risks, and an opti-
mal tradeoff of the two goals is found by a variant of minimax optimization. Generalization
performance of the empirical solution is analyzed and and a new and simple optimization
algorithm is presented. In addition to introducing minimax filter, the paper proposes noisy
minimax filter that combines minimax filter and differentially private noisy mechanism,
and compare resilience to inference attack and differentially privacy both quantitatively
and qualitatively. Experiments with several real-world tasks including facial expression
recognition, speech emotion recognition, and activity recognition from motion, show that
the minimax filter can simultaneously achieve similar or better target task accuracy and
lower inference accuracy, often significantly lower, than previous methods.


### Getting Started
---
#### 1. Download all files in [src/](src) and [test/](test)
Make sure you can access scripts in /src, for example by downloading files from both /src and /test into the same folder.
Description of the scripts are in [src/readme.md](src/readme.md).
The Genki dataset [test/genki.mat](test/genki.mat) is originally downloaded from http://mplab.ucsd.edu. 

#### 2. Run [test/test_NN_genki.py](test/test_NN_genki.py) 
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
* [J. Hamm, "Preserving privacy of continuous high-dimensional data with minimax filters," 
 AISTATS, 2015](http://web.cse.ohio-state.edu/~hammj/papers/aistats15_2_jh_final.pdf)
* [J. Hamm, "Mimimax Filter: Learning to Preserve Privacy from Inference Attacks," arXiv:1610.03577, 2016](http://arxiv.org/abs/1610.03577)


### License
---
Released under the Apache License 2.0.  See the [LICENSE.txt](LICENSE.txt) file for further details.





