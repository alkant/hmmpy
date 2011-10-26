# Summary

This is a simple implementation of discrete hidden markov models for Python > 2.5. Both pure Python and Cython implementations are maintained. The Cython module is 10x (learning) to 75x (inference) faster than the Python implementation, but you'll need to compile it. Also, the Cython module trades off memory usage for speed by caching the dynamic program array between runs.

# Features

* Supervised learning - no regularization
* Viterbi inference

The following features are out-of-scope:

* Unsupervised learning (Baum-Welch)
* Regularization

# Building the Cython module

Assuming a working installation of Cython, running:
	
	python setup.py build_ext --inplace

will build the chmmpy module.

# Basic usage

	>>> from hmmpy import hmm
	>>> from chmmpy import hmm as chmm
	
	>>> m=hmm(2, 2)
	>>> cm=chmm(2,2)
	
	>>> observations=[[0,0,0,0,0,1,1,1,1,1,0,1,0,0,0,1,0,1,1,1,1],[0,0,0,0,1,0,1,1,0,1,1,0,0,1,0,0,1,1,1,1,0,0,1,0,0]]
	>>> ground_truths=[[0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1],[0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0]]
	
	>>> m.learn(observations, ground_truths)
	>>> cm.learn(observations, ground_truths)
	
	>>> m.viterbi(observations[1])
	([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0], -21.944735122525493)
	
	>>> cm.viterbi(observations[1])
	([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0], -21.944734573364258)


